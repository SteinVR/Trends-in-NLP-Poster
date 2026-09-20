"""Scientific-control tests: isolation, citations, and actual component interventions."""
import json
import numpy as np

from experiments.retrieval import DATA, PROTOCOL, cosine_scores, make_chunks, pack_context, read_pages
from experiments.answer import emit_answer
from src.providers.codex_provider import _build_structured_user_prompt


def test_each_step_changes_one_component():
    variants=list(PROTOCOL["configurations"].values())
    for old,new in zip(variants,variants[1:]):
        changed=[key for key in old if old[key]!=new[key]]
        assert len(changed)==1 and new[changed[0]] is True


def test_cosine_ranking_does_not_depend_on_vector_magnitude():
    queries=np.array([[1.0,0.0]])
    documents=np.array([[0.9,0.0],[100.0,100.0]])
    scores=cosine_scores(queries,documents)[0]
    assert scores[0] > scores[1]
    assert np.isclose(scores[0],1.0)


def test_scans_have_no_native_text_or_original_tables():
    chosen=set(json.loads((DATA/"corpus_manifest.json").read_text())["selected_doc_ids"])
    pages=read_pages("mixed-ocr0")
    missing=[p for p in pages if p["doc_id"] in chosen]
    assert len(missing)==107
    assert all(not p["text"] and not p["tables"] and p["native_chars"]==0 for p in missing)
    assert all(c["doc_id"] not in chosen for c in make_chunks(pages,False))


def test_context_never_cites_truncated_away_page():
    candidates=[{"doc_id":"d","chunk_id":"c","chunk_type":"section","segments":[
        {"page_number":i,"text":"one two three "*2000} for i in range(1,10)]}]
    context=pack_context(candidates)
    assert sum(e["tokens"] for e in context)<=PROTOCOL["context_token_budget"]
    assert [e["page_number"] for e in context]==[1,2,3,4]


class FakeProvider:
    model="fixture"
    def __init__(self,confidence=0.9):
        self.confidence=confidence
    def generate_structured(self,**kwargs):
        return {"final_answer":"Acme", "is_answerable":True,"confidence":self.confidence,
                "relevant_evidence_indices":[1],"relevant_pages":[1],"reasoning_summary":"Named in evidence."}


def fixture():
    return ({"id":"q","question":"Who is the claimant?","answer_type":"name"},
            {"evidence":[{"doc_id":"d","page_number":1,"text":"The claimant is Acme."},
                         {"doc_id":"d","page_number":2,"text":"Other procedural information."}]})


def test_typed_threshold_really_changes_answer():
    q,r=fixture()
    baseline=emit_answer(q,r,"R4",FakeProvider(0.1))
    typed=emit_answer(q,r,"R5",FakeProvider(0.1))
    assert baseline["answer"]=="Acme" and typed["answer"] is None
    assert baseline["pages"]==typed["pages"]


def test_attribution_changes_citations_not_answer():
    q,r=fixture()
    before=emit_answer(q,r,"R5",FakeProvider())
    after=emit_answer(q,r,"R6",FakeProvider())
    assert before["answer"]==after["answer"]=="Acme"
    assert before["pages"]==[("d",1),("d",2)]
    assert after["pages"]==[("d",1)]
    assert json.loads(json.dumps(after))["postprocessing"]["final_emitted_pages"]


def test_typed_path_preserves_unicode_in_generation_prompt():
    class Recorder(FakeProvider):
        def generate_structured(self,**kwargs):
            self.prompt=_build_structured_user_prompt(question=kwargs["question"],
                answer_type=kwargs["answer_type"],evidence_pages=kwargs["evidence_pages"])
            return super().generate_structured(**kwargs)
    q,r=fixture()
    r["evidence"][1]["text"]="Contents………2. The claimant is Acme."
    providers=[Recorder(),Recorder(),Recorder()]
    for variant,provider in zip(("R4","R5","R6"),providers,strict=True):
        emit_answer(q,r,variant,provider)
    assert providers[0].prompt==providers[1].prompt==providers[2].prompt
