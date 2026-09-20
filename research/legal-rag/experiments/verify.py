"""Verify completed offline runs and record the exact experiment implementation."""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.retrieval import DATA, ENC, OUT, PROTOCOL, ROOT, condition_specs, read_pages, write_json


def main():
    questions = json.loads((ROOT / "data/warmup/questions/questions.json").read_text())
    ids = {q["id"] for q in questions}
    gold = json.loads((DATA / "gold_pages.json").read_text())
    references = json.loads((ROOT / "experiments/reference/warmup100.benchmark.json").read_text())["references"]
    assert len(ids) == 100 and ids == set(gold) == {r["question_id"] for r in references}
    assert sum(bool(pages) for pages in gold.values()) == 95
    manifest = json.loads((DATA / "corpus_manifest.json").read_text())
    selected = set(manifest["selected_doc_ids"])
    missing = [p for p in read_pages("mixed-ocr0") if p["doc_id"] in selected]
    recovered = [p for p in read_pages("mixed-ocr1") if p["doc_id"] in selected]
    assert len(missing) == len(recovered) == 107
    assert all(not p["text"] and not p["tables"] for p in missing)
    assert all(p["source_mode"] == "ocr" and p["text"].strip() for p in recovered)
    original_off, original_on = read_pages("original-ocr0"), read_pages("original-ocr1")
    assert [(p["doc_id"], p["page_number"], p["text"]) for p in original_off] == [
        (p["doc_id"], p["page_number"], p["text"]) for p in original_on]
    runs = {}
    for corpus, variant in condition_specs():
        name = f"{corpus}-{variant}"
        rows = json.loads((OUT / name / "retrieval.json").read_text())
        assert len(rows) == 100 and {r["question_id"] for r in rows} == ids
        for row in rows:
            assert len(row["evidence"]) <= PROTOCOL["max_evidence_chunks"]
            assert sum(e["tokens"] for e in row["evidence"]) <= PROTOCOL["context_token_budget"]
            # Decoding a token prefix may change re-tokenization by a boundary token.
            assert sum(len(ENC.encode(e["text"])) for e in row["evidence"]) <= PROTOCOL["context_token_budget"] + 8
            if name == "mixed-R0":
                assert all(e["doc_id"] not in selected for e in row["evidence"])
        runs[name] = rows
    assert runs["mixed-R4"] == runs["mixed-R5"] == runs["mixed-R6"]
    assert runs["original-R0"] == runs["original-R1"]
    indices = {}
    for folder in sorted((DATA / "indices").iterdir()):
        chunks = json.loads((folder / "chunks.json").read_text())
        vectors = np.load(folder / "vectors.npy")
        assert vectors.shape == (len(chunks), 1024) and np.isfinite(vectors).all()
        # Stored BF16 model outputs are approximately normalized; retrieval applies
        # exact FP32 cosine normalization, matching upstream Qdrant COSINE.
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=0.005)
        indices[folder.name] = dict(Counter(c["chunk_type"] for c in chunks))
    assert len(indices) == 5
    tracked = subprocess.check_output(["git", "ls-files", "src"], cwd=ROOT, text=True).splitlines()
    paths = [ROOT / p for p in tracked]
    paths += sorted((ROOT / "experiments").glob("*.py"))
    paths += [ROOT / "experiments/protocol.json", ROOT / "experiments/reference/warmup100.benchmark.json",
              ROOT / "experiments/reference/free_text_review_template.md", DATA / "corpus_manifest.json"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write_json(OUT / "offline_verification.json", {
        "status": "passed", "questions": len(ids), "citation_questions": 95,
        "scan_documents": len(selected), "scan_pages": len(missing),
        "scan_affected_questions": sum(any(d in selected for d, p in pages) for pages in gold.values()),
        "conditions": list(runs), "indices": indices,
        "original_ocr_control_identical": True, "R4_R5_R6_contexts_identical": True,
        "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sha256": hashes,
        "completion_scope": "Extraction, indexing and retrieval only; answer generation and scoring are separate."
    })
    print("Offline verification passed: 9 conditions, 900 question contexts, 5 indices.")


if __name__ == "__main__":
    main()
