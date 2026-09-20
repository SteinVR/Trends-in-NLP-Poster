"""Replay the same model outputs through controlled answering/attribution variants."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pydantic import TypeAdapter

from experiments.environment import load_environment, record_response

from src.answering.page_attribution import PageAttributionConfig
from src.answering.service import AnsweringService
from src.answering.service_helpers import _coerce_retrieval_payload, _group_page_references
from src.answering.service_types import ANSWER_SCHEMA_REGISTRY
from src.common.schemas import QuestionRecord
from src.evaluation.contracts import AnswerType
from src.providers.codex_provider import (CodexAnsweringProvider, _STRUCTURED_SYSTEM_PROMPT,
                                          _build_structured_user_prompt)

from experiments.retrieval import OUT, PROTOCOL, ROOT, condition_specs, write_json


class CachedProvider:
    """Content-addressed generation cache excludes configuration names and all gold data."""
    model = PROTOCOL["answer_model"]

    def generate_structured(self, **kwargs):
        request = {**kwargs, "answer_schema": kwargs["answer_schema"].model_json_schema(),
                   "system_prompt": _STRUCTURED_SYSTEM_PROMPT, "model": self.model,
                   "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                   "max_output_tokens": PROTOCOL["answer_max_output_tokens"]}
        canonical = {k:request[k] for k in ("answer_schema","system_prompt","model","base_url",
                                            "reasoning_effort","max_output_tokens")}
        canonical["user_prompt"] = _build_structured_user_prompt(
            question=kwargs["question"], answer_type=kwargs["answer_type"], evidence_pages=kwargs["evidence_pages"])
        key = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
        path = OUT / "model_cache" / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["response"]
        api_responses = []
        with CodexAnsweringProvider(model=self.model, base_url=request["base_url"],
                                    timeout_seconds=180, max_output_tokens=request["max_output_tokens"]) as provider:
            provider._client.event_hooks["response"].append(lambda r: record_response(r, api_responses))
            response = provider.generate_structured(**kwargs)
        write_json(path, {"request": request, "response": response, "api_responses": api_responses})
        return response


class FixedRetriever:
    def __init__(self, evidence):
        self.evidence = evidence

    def retrieve(self, question, **kwargs):
        return [{k:e[k] for k in ("doc_id","page_number","text")} for e in self.evidence]


def emit_answer(question: dict, retrieval: dict, variant: str, provider=None) -> dict:
    config = PROTOCOL["configurations"][variant]
    q = QuestionRecord.model_validate(question)
    evidence = [{k:e[k] for k in ("doc_id","page_number","text")} for e in retrieval["evidence"]]
    provider = provider or CachedProvider()
    # R0–R4 use the same generation contract as R5/R6, without task-specific postprocessing.
    # This measures the additional existing confidence/normalization/validation block in H5.
    if not config["typed"]:
        response = provider.generate_structured(question=q.question, answer_type=q.answer_type,
            evidence_pages=evidence, answer_schema=ANSWER_SCHEMA_REGISTRY[q.answer_type],
            reasoning_effort=PROTOCOL["reasoning_effort"])
        result = {"answer": response["final_answer"], "confidence": response.get("confidence"),
                  "pages": sorted({(e["doc_id"],e["page_number"]) for e in evidence}),
                  "postprocessing": "none"}
    else:
        service = AnsweringService(retriever=FixedRetriever(evidence), provider=provider,
            reasoning_effort_by_type={t:PROTOCOL["reasoning_effort"] for t in ANSWER_SCHEMA_REGISTRY},
            page_attribution_config=PageAttributionConfig(pass_a_enabled=False,
                suppress_title_pages=False, suppress_repeated_boilerplate=False,
                validation_mode="degrade_first", allow_solver_page_narrowing=config["attribution"]))
        # Pass A is outside this experiment. Even disabled, the upstream wrapper
        # applies NFKC text normalization. Bypass that wrapper so R4/R5/R6 send
        # byte-identical prompts and reuse the same generation, while retaining
        # the original typed postprocessing and attribution implementation.
        pages, _ = _coerce_retrieval_payload(evidence)
        refs = _group_page_references(pages)
        answer = service._answer_with_structured_solver(
            question=q, answer_type=AnswerType(q.answer_type), pages=pages,
            raw_retrieval_refs=refs, pass_a_refs=refs)
        result = {"answer": answer.answer, "confidence": answer.confidence,
                  "pages": sorted({(r.doc_id,p) for r in answer.evidence_pages for p in r.page_numbers}),
                  "postprocessing": TypeAdapter(type(answer.page_trace)).dump_python(answer.page_trace,mode="json")}
    return {"question_id":q.id,"answer_type":q.answer_type,**result}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--env-file",type=Path,default=ROOT/".env")
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--limit",type=int,default=0)
    args=parser.parse_args()
    load_environment(args.env_file)
    if not any(os.environ.get(k) for k in ("OPENAI_API_KEY","CODEX_OAUTH_TOKEN","CODEX_API_KEY")):
        raise RuntimeError("Missing provider credential. Set OPENAI_API_KEY in the experiment .env file.")
    questions=json.loads((ROOT/"data/warmup/questions/questions.json").read_text())
    for corpus,variant in condition_specs():
        folder=OUT/f"{corpus}-{variant}"
        retrieval={r["question_id"]:r for r in json.loads((folder/"retrieval.json").read_text())}
        pending=[q for q in questions if not (folder/"answers"/f"{q['id']}.json").exists()]
        if args.limit:
            pending=pending[:args.limit]

        def run(q):
            result=emit_answer(q,retrieval[q["id"]],variant)
            write_json(folder/"answers"/f"{q['id']}.json",result)
            return q["id"]

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i,qid in enumerate(pool.map(run,pending),1):
                print(f"answer {corpus}-{variant} {i}/{len(pending)} {qid[:10]}",flush=True)
        if all((folder/"answers"/f"{q['id']}.json").exists() for q in questions):
            answers=[json.loads((folder/"answers"/f"{q['id']}.json").read_text()) for q in questions]
            write_json(folder/"answers.json",answers)


if __name__=="__main__":
    main()
