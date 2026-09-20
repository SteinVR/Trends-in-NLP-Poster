"""Deterministic offline indexing/retrieval, reusing upstream chunking and models."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import fitz
import tiktoken
from scipy.sparse import csr_matrix

from src.common.schemas import CanonicalPageRecord, ContentBlock, QuestionRecord
from src.indexing.chunking_core import build_index_chunks
from src.indexing.embedders import BM25SparseEncoder, Qwen3DenseEmbedder
from src.retrieval.reranker import TransformersQwenRerankerBackend

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiment_data"
OUT = ROOT / "experiment_runs"
PROTOCOL = json.loads((ROOT / "experiments/protocol.json").read_text())
ENC = tiktoken.get_encoding("cl100k_base")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def read_pages(name: str) -> list[dict]:
    # PDF text can contain literal Unicode line separators; JSONL uses ASCII LF only.
    pages = [json.loads(x) for x in (DATA / "pages" / f"{name}.jsonl").read_text().split("\n") if x]
    source = ROOT / "data/warmup/documents/pdfs" if name.startswith("original") else DATA / "mixed/pdfs"
    documents = {}
    try:
        for page in pages:
            if page["source_mode"] == "ocr":
                # OCR produces one detected text block per line. Preserve these boundaries.
                page["text"] = "\n\n".join(x.strip() for x in page["text"].split("\n") if x.strip())
            elif page["native_chars"]:
                doc = documents.setdefault(page["doc_id"], None)
                if doc is None:
                    doc = documents[page["doc_id"]] = fitz.open(source / f"{page['doc_id']}.pdf")
                blocks = doc[page["page_number"]-1].get_text("blocks")
                page["text"] = "\n\n".join(b[4].strip() for b in blocks if b[6] == 0 and b[4].strip())
    finally:
        for doc in documents.values():
            if doc is not None:
                doc.close()
    return pages


def condition_specs():
    for variant in PROTOCOL["configurations"]:
        yield "mixed", variant
    yield "original", "R0"
    yield "original", "R1"


def index_name(corpus: str, variant: str) -> str:
    config = PROTOCOL["configurations"][variant]
    return f"{corpus}-ocr{int(config['ocr'])}-multi{int(config['multiscale'])}"


def make_chunks(pages: list[dict], multiscale: bool) -> list[dict]:
    chunks = []
    if not multiscale:
        for page in pages:
            tokens = ENC.encode(page["text"])
            step = PROTOCOL["chunk_token_size"] - PROTOCOL["chunk_token_overlap"]
            for start in range(0, len(tokens), step):
                text = ENC.decode(tokens[start:start+PROTOCOL["chunk_token_size"]])
                chunks.append({"chunk_id": f"{page['doc_id']}-p{page['page_number']}-fixed-{start}",
                               "doc_id": page["doc_id"], "chunk_type": "fixed",
                               "segments": [{"page_number": page["page_number"], "text": text}]})
                if start + PROTOCOL["chunk_token_size"] >= len(tokens):
                    break
    else:
        canonical = [CanonicalPageRecord(
            doc_id=p["doc_id"], page_number=p["page_number"], text=p["text"],
            triage_label="scan" if p["native_chars"] < 50 else "native_clean",
            source_mode=p["source_mode"], quality_score=1.0,
            blocks=[ContentBlock(block_id=f"{p['doc_id']}-p{p['page_number']}-table-{i}",
                                 type="table", text=t, page_number=p["page_number"])
                    for i,t in enumerate(p["tables"])]) for p in pages if p["text"].strip()]
        by_page = {(p["doc_id"],p["page_number"]):p for p in pages}
        for c in build_index_chunks(canonical, token_chunk_size=300, token_chunk_overlap=50):
            # Explicit page segments ensure truncation never emits unseen page references.
            if len(c.page_span) == 1:
                segments = [{"page_number": c.page_span[0], "text": c.text}]
            else:
                segments = [{"page_number": n, "text": by_page[c.doc_id,n]["text"]}
                            for n in c.page_span]
            chunks.append({"chunk_id": c.chunk_id,"doc_id":c.doc_id,
                           "chunk_type":c.chunk_type,"segments":segments})
    # Bound indexed fragment size equally; section spans retain exact page provenance.
    unique = {}
    for c in chunks:
        remaining = PROTOCOL["max_indexed_chunk_tokens"]
        segments = []
        for s in c["segments"]:
            tokens = ENC.encode(s["text"])
            if not tokens or remaining <= 0:
                continue
            kept = tokens[:remaining]
            segments.append({"page_number": s["page_number"], "text": ENC.decode(kept)})
            remaining -= len(kept)
        if not segments:
            continue
        c = {**c, "segments": segments,
             "page_span": [s["page_number"] for s in segments],
             "text": "\n".join(s["text"] for s in segments)}
        key = (c["doc_id"],tuple(c["page_span"]),c["text"])
        unique.setdefault(key,c)
    return sorted(unique.values(), key=lambda x:x["chunk_id"])


def build_indices(only: str | None = None) -> None:
    import torch
    torch.set_num_threads(4)
    embedder = Qwen3DenseEmbedder(batch_size=4, device="cuda", local_files_only=True)
    # A single embedding function is used for all representations and all conditions.
    seen = set()
    for corpus, variant in condition_specs():
        name = index_name(corpus, variant)
        if only is not None and name != only:
            continue
        if name in seen:
            continue
        seen.add(name)
        config = PROTOCOL["configurations"][variant]
        root = DATA / "indices" / name
        root.mkdir(parents=True, exist_ok=True)
        pages = read_pages(f"{corpus}-ocr{int(config['ocr'])}")
        chunks = make_chunks(pages, config["multiscale"])
        digest = hashlib.sha256(json.dumps(chunks, sort_keys=True).encode()).hexdigest()
        if (root/"manifest.json").exists():
            assert json.loads((root/"manifest.json").read_text())["chunks_sha256"] == digest
            print(f"index exists {name}",flush=True)
            continue
        write_json(root/"chunks.json",chunks)
        # Persistent batches allow a failed GPU/download session to resume safely.
        arrays = []
        for start in range(0,len(chunks),64):
            shard = root/f"embedding-{start:06d}.npz"
            if shard.exists():
                saved=np.load(shard)
                assert str(saved["digest"]) == digest
                batch=saved["vectors"]
            else:
                batch = np.asarray(embedder.encode([c["text"] for c in chunks[start:start+64]]),dtype=np.float32)
                np.savez(shard,vectors=batch,digest=digest)
            arrays.append(batch)
            print(f"index {name} {min(start+64,len(chunks))}/{len(chunks)}",flush=True)
        vectors=np.concatenate(arrays)
        assert len(vectors)==len(chunks) and np.isfinite(vectors).all()
        np.save(root/"vectors.npy",vectors)
        write_json(root/"manifest.json",{"chunks_sha256":digest,"chunk_count":len(chunks),
                   "embedding_model":PROTOCOL["embedding_model"],"index":name})
    qs=json.loads((ROOT/"data/warmup/questions/questions.json").read_text())
    np.save(DATA/"query_vectors.npy",np.asarray(embedder.encode([q["question"] for q in qs]),dtype=np.float32))
    del embedder
    gc.collect()
    torch.cuda.empty_cache()


def pack_context(candidates: list[dict]) -> list[dict]:
    remaining=PROTOCOL["context_token_budget"]
    evidence=[]
    seen=set()
    for candidate in candidates:
        for segment in candidate["segments"]:
            key=(candidate["doc_id"],segment["page_number"],segment["text"])
            if key in seen:
                continue
            tokens=ENC.encode(segment["text"])
            take=min(len(tokens),remaining,PROTOCOL["max_evidence_item_tokens"])
            if not take:
                continue
            evidence.append({"doc_id":candidate["doc_id"],"page_number":segment["page_number"],
                             "text":ENC.decode(tokens[:take]),"chunk_id":candidate["chunk_id"],
                             "chunk_type":candidate["chunk_type"],"tokens":take})
            seen.add(key)
            remaining-=take
            if remaining<=0 or len(evidence)>=PROTOCOL["max_evidence_chunks"]:
                return evidence
    return evidence


def rank_indices(scores: np.ndarray) -> list[int]:
    return np.argsort(-scores,kind="stable").tolist()


def cosine_scores(queries: np.ndarray, documents: np.ndarray) -> np.ndarray:
    # Model-side BF16 normalization is approximate. Qdrant COSINE normalizes again;
    # reproduce that operation in FP32 rather than treating raw dot products as cosine.
    queries = queries.astype(np.float32)
    documents = documents.astype(np.float32)
    qnorm = np.linalg.norm(queries, axis=1, keepdims=True)
    dnorm = np.linalg.norm(documents, axis=1, keepdims=True)
    assert np.all(qnorm > 0) and np.all(dnorm > 0)
    return (queries / qnorm) @ (documents / dnorm).T


def retrieve_all() -> None:
    import torch
    torch.set_num_threads(4)
    questions=json.loads((ROOT/"data/warmup/questions/questions.json").read_text())
    qvectors=np.load(DATA/"query_vectors.npy")
    cache={}
    reranker=None
    for corpus,variant in condition_specs():
        config=PROTOCOL["configurations"][variant]
        name=index_name(corpus,variant)
        target=OUT/f"{corpus}-{variant}"/"retrieval.json"
        key=(name,config["hybrid"],config["rerank"])
        if key in cache:
            write_json(target,cache[key])
            continue
        if target.exists():
            cache[key]=json.loads(target.read_text())
            continue
        folder=DATA/"indices"/name
        chunks=json.loads((folder/"chunks.json").read_text())
        dense_scores=cosine_scores(qvectors,np.load(folder/"vectors.npy"))
        sparse_scores=None
        if config["hybrid"]:
            # Same sparse encoder and dot-product weighting as upstream Qdrant indexing.
            encoder=BM25SparseEncoder([c["text"] for c in chunks])
            rows,cols,vals=[],[],[]
            for i,c in enumerate(chunks):
                v=encoder.encode(c["text"])
                rows.extend([i]*len(v.indices));cols.extend(v.indices);vals.extend(v.values)
            matrix=csr_matrix((vals,(rows,cols)),shape=(len(chunks),len(encoder.term_index)))
            sparse_scores=[]
            for q in questions:
                v=encoder.encode(q["question"])
                vector=csr_matrix((v.values,([0]*len(v.indices),v.indices)),shape=(1,len(encoder.term_index)))
                sparse_scores.append((matrix@vector.T).toarray().ravel())
        result=[]
        checkpoint=target.with_name("retrieval.checkpoint.json")
        if checkpoint.exists():
            result=json.loads(checkpoint.read_text())
        for qi,q in enumerate(questions):
            if qi<len(result):
                assert result[qi]["question_id"]==q["id"]
                continue
            dense=dense_scores[qi]
            scores=dense
            if sparse_scores is not None:
                scores=np.zeros(len(chunks))
                # Upstream overfetches each channel by 3x, then fuses a bounded union.
                for channel in (dense,sparse_scores[qi]):
                    for rank,idx in enumerate(rank_indices(channel)[:PROTOCOL["candidate_budget"]*3],1):
                        scores[idx]+=1/(PROTOCOL["rrf_k"]+rank)
            selected=rank_indices(scores)[:PROTOCOL["candidate_budget"]]
            candidates=[{**chunks[i],"retrieval_score":float(scores[i])} for i in selected]
            if config["rerank"]:
                if reranker is None:
                    reranker=TransformersQwenRerankerBackend(batch_size=2,max_length=2048,device="cuda")
                rr=reranker.score(q["question"],[c["text"] for c in candidates])
                candidates=[{**c,"rerank_score":s} for c,s in zip(candidates,rr,strict=True)]
                candidates.sort(key=lambda c:(-c["rerank_score"],-c["retrieval_score"],c["chunk_id"]))
            evidence=pack_context(candidates)
            assert sum(e["tokens"] for e in evidence)<=PROTOCOL["context_token_budget"]
            result.append({"question_id":q["id"],"evidence":evidence,
                           "candidates":[{k:c[k] for k in ("chunk_id","doc_id","page_span","chunk_type","retrieval_score")} for c in candidates]})
            write_json(checkpoint,result)
            print(f"retrieve {corpus}-{variant} {qi+1}/100",flush=True)
        write_json(target,result)
        cache[key]=result
    for corpus,variant in condition_specs():
        write_json(OUT/f"{corpus}-{variant}"/"config.json",{**PROTOCOL,"corpus":corpus,"variant":variant})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["index","retrieve"])
    parser.add_argument("--index-name")
    args=parser.parse_args()
    build_indices(args.index_name) if args.stage=="index" else retrieve_all()
