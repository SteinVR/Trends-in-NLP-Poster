"""Validate completed measurements, export evidence, and render publication figures."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from shutil import copy2
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.retrieval import DATA, OUT, PROTOCOL, ROOT, condition_specs, write_json

DEST = ROOT.parents[1] / "output/poster-narrative-3/results"


def validate_and_export():
    metrics = json.loads((OUT / "metrics.json").read_text())
    assert {(r["corpus"], r["variant"]) for r in metrics} == set(condition_specs())
    assert len(metrics) == 9
    gold = json.loads((DATA / "gold_pages.json").read_text())
    questions = {q["id"]: q for q in json.loads((ROOT / "data/warmup/questions/questions.json").read_text())}
    references = {r["question_id"]: r for r in json.loads((ROOT / "experiments/reference/warmup100.benchmark.json").read_text())["references"]}
    per_question, all_answers, all_scores = [], {}, {}
    for m in metrics:
        name = f"{m['corpus']}-{m['variant']}"
        folder = OUT / name
        answers = {r["question_id"]: r for r in json.loads((folder / "answers.json").read_text())}
        scores = {r["question_id"]: r for r in json.loads((folder / "scores.json").read_text())}
        retrieval = {r["question_id"]: r for r in json.loads((folder / "retrieval.json").read_text())}
        assert set(answers) == set(scores) == set(retrieval) == set(questions)
        precisions, recalls = [], []
        for qid, q in questions.items():
            a, s = answers[qid], scores[qid]
            pred, expected = set(map(tuple, a["pages"])), set(map(tuple, gold[qid]))
            available = {(e["doc_id"], e["page_number"]) for e in retrieval[qid]["evidence"]}
            assert pred <= available, (name, qid, "citation outside supplied context")
            assert 0 <= s["answer_score"] <= 1
            if expected:
                p = len(pred & expected) / len(pred) if pred else 0
                r = len(pred & expected) / len(expected)
                assert abs(p-s["precision"]) < 1e-12 and abs(r-s["recall"]) < 1e-12
                precisions.append(p); recalls.append(r)
            else:
                assert s["precision"] is None and s["recall"] is None
            if q["answer_type"] == "free_text":
                review=s["review"]
                criteria=("correctness","completeness","grounding","confidence_calibration","clarity_and_relevance")
                assert s["answer_score"] == sum(review[k] for k in criteria)/5
            per_question.append({"condition":name,"question_id":qid,"question":q["question"],
                "answer_type":q["answer_type"],"answer":json.dumps(a["answer"],ensure_ascii=False),
                "reference_answer":json.dumps(references[qid]["answer"],ensure_ascii=False),
                "answer_score":s["answer_score"],"precision":s["precision"],"recall":s["recall"],
                "pages":json.dumps(sorted(pred)),"gold_pages":json.dumps(sorted(expected)),
                "scan_affected":s["scan_affected"],"review_notes":(s["review"] or {}).get("notes","")})
        # With 70 structured and 30 free-text questions, the specified 0.7/0.3
        # aggregate must also equal the direct mean of all 100 answer scores.
        assert Counter(s["answer_type"]=="free_text" for s in scores.values()) == {False:70,True:30}
        assert abs(mean(s["answer_score"] for s in scores.values()) - m["Q"]) < 1e-12
        assert len(precisions) == 95
        assert abs(mean(precisions)-m["precision"]) < 1e-12
        assert abs(mean(recalls)-m["recall"]) < 1e-12
        all_answers[name], all_scores[name] = answers, scores
    assert all_answers["original-R0"] == all_answers["original-R1"]
    for qid in questions:
        a,b=all_answers["mixed-R5"][qid],all_answers["mixed-R6"][qid]
        assert a["answer"] == b["answer"] and a["confidence"] == b["confidence"]
        assert all_scores["mixed-R5"][qid]["answer_score"] == all_scores["mixed-R6"][qid]["answer_score"]
    DEST.mkdir(parents=True,exist_ok=True)
    for name in ("metrics.json","metrics.csv","retrieval_diagnostics.json","retrieval_diagnostics.csv","offline_verification.json"):
        copy2(OUT/name, DEST/name)
    copy2(ROOT/"experiments/protocol.json", DEST/"protocol.json")
    with (DEST/"per-question-scores.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(per_question[0]))
        writer.writeheader();writer.writerows(per_question)
    mixed={m["variant"]:m for m in metrics if m["corpus"]=="mixed"}
    comparisons=[]
    for i in range(1,7):
        a,b=mixed[f"R{i-1}"],mixed[f"R{i}"]
        comparisons.append({"comparison":f"R{i} - R{i-1}",**{
            key+"_delta_pp":100*(b[key]-a[key]) for key in ("Q","precision","recall","structured_score","free_text_score")}})
    comparisons.append({"comparison":"R6 - R0",**{
        key+"_delta_pp":100*(mixed["R6"][key]-mixed["R0"][key]) for key in ("Q","precision","recall","structured_score","free_text_score")}})
    write_json(DEST/"comparisons.json",comparisons)
    with (DEST/"comparisons.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    metadata={}
    for kind in ("model_cache","review_cache"):
        caches=[json.loads(p.read_text()) for p in (OUT/kind).glob("*.json")]
        calls=[call for cache in caches for call in cache["api_responses"]]
        metadata[kind]={"unique_cached_requests":len(caches),"recorded_responses":len(calls),
            "model_snapshots":dict(Counter(c["model"] for c in calls)),
            "statuses":dict(Counter(c["status"] for c in calls)),
            "input_tokens":sum((c["usage"] or {}).get("input_tokens",0) for c in calls),
            "output_tokens":sum((c["usage"] or {}).get("output_tokens",0) for c in calls)}
        assert all(cache["api_responses"][-1]["status"] == "completed" for cache in caches)
    write_json(DEST/"run-verification.json",{"status":"passed","conditions":9,"answers":900,
        "structured_scores":630,"free_text_scores":270,"citation_scores":855,
        "checks":["complete question coverage","Q equals direct mean for 70/30 question mix",
                  "citation metrics independently recomputed from sets","citations contained in supplied context",
                  "original R0/R1 answers identical","R5/R6 answers and quality scores identical"],
        "api":metadata,
        "artifact_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.glob("*.csv")}})
    return metrics


def figures(metrics):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.titlesize":13,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.labelcolor":"#30343B","xtick.color":"#30343B","ytick.color":"#30343B",
                         "svg.fonttype":"none","pdf.fonttype":42})
    mixed=sorted((m for m in metrics if m["corpus"]=="mixed"),key=lambda x:x["variant"])
    series=[("Q","Качество ответа Q · 100 вопросов","#28658A","o"),
            ("precision","Точность ссылок · 95 вопросов","#AE6D16","s"),
            ("recall","Полнота ссылок · 95 вопросов","#62743B","^")]
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharey=True)
    for ax,(key,title,color,marker) in zip(axes,series,strict=True):
        values=[100*m[key] for m in mixed]
        ax.plot(range(7),values,color=color,marker=marker,linewidth=2,markersize=6)
        for x,y in enumerate(values):
            ax.annotate(f"{y:.1f}",(x,y),xytext=(0,9),textcoords="offset points",ha="center",fontsize=10)
        ax.set(xticks=range(7),xticklabels=[m["variant"] for m in mixed],ylim=(0,105),yticks=range(0,101,20),title=title)
        ax.grid(axis="y",color="#E2E5E8",linewidth=.7);ax.set_axisbelow(True)
        ax.set_xlabel("Последовательное включение компонентов")
    axes[0].set_ylabel("Балл / доля, %")
    fig.suptitle("Legal RAG: итоговые метрики на смешанном корпусе",fontsize=16,y=.99)
    fig.text(.5,.035,"R0 baseline  ·  R1 + OCR  ·  R2 + масштабы  ·  R3 + лексический поиск  ·  R4 + reranker  ·  R5 + обработка ответа  ·  R6 + выбор цитат",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.08,1,.94))
    save(fig,"pipeline-metrics")
    lookup={(m["corpus"],m["variant"]):m for m in metrics}
    fig,axes=plt.subplots(1,3,figsize=(13,5.2),sharey=True)
    for ax,(key,title,color,marker) in zip(axes,series,strict=True):
        for variant,offset,barcolor in (("R0",-.19,"#8699A8"),("R1",.19,"#28658A")):
            values=[100*lookup[corpus,variant][key] for corpus in ("original","mixed")]
            bars=ax.bar(np.arange(2)+offset,values,width=.34,color=barcolor,edgecolor="#30343B",linewidth=.6,
                        label="R0 · без OCR" if variant=="R0" else "R1 · с OCR")
            for bar,y in zip(bars,values,strict=True):
                ax.text(bar.get_x()+bar.get_width()/2,y+2,f"{y:.1f}",ha="center",fontsize=10)
        ax.set(xticks=range(2),xticklabels=["Исходный корпус","Смешанный корпус"],ylim=(0,105),yticks=range(0,101,20),title=title)
        ax.grid(axis="y",color="#E2E5E8",linewidth=.7);ax.set_axisbelow(True)
    axes[0].set_ylabel("Балл / доля, %")
    fig.suptitle("OCR: потеря и восстановление качества после замены PDF сканами",fontsize=15,y=.99)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.065),ncol=2,frameon=False)
    fig.text(.5,.025,"В смешанном корпусе 10 из 30 документов заменены сканами (107 страниц)",ha="center",fontsize=11)
    fig.tight_layout(rect=(0,.15,1,.93))
    save(fig,"ocr-control")


def save(fig,stem):
    for extension in ("png","svg","pdf"):
        fig.savefig(DEST/f"{stem}.{extension}",dpi=200,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    measurements=validate_and_export()
    figures(measurements)
    print("Validated all nine conditions and exported metrics, per-question evidence and figures.")
