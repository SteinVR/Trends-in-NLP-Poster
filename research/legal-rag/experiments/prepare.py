"""Freeze corpus selection and create image-only copies without using answer scores."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "experiment_data"
SOURCE = ROOT / "data/warmup/documents/pdfs"
SEED = 20260920
DPI = 300
SCAN_DOCUMENTS = 10


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    WORK.mkdir(exist_ok=True)
    manifest_path = WORK / "corpus_manifest.json"
    paths = sorted(SOURCE.glob("*.pdf"))
    assert len(paths) == 30
    chosen = set(random.Random(SEED).sample([p.stem for p in paths], SCAN_DOCUMENTS))
    settings = {"seed": SEED, "dpi": DPI, "selected_doc_ids": sorted(chosen)}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        assert all(old[k] == v for k, v in settings.items()), "Frozen corpus settings differ"
    mixed = WORK / "mixed/pdfs"
    mixed.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in paths:
        target = mixed / source.name
        with fitz.open(source) as src:
            scanned = source.stem in chosen
            if scanned:
                if not target.exists():
                    output = fitz.open()
                    for page in src:
                        pix = page.get_pixmap(dpi=DPI, colorspace=fitz.csGRAY, alpha=False)
                        out_page = output.new_page(width=page.rect.width, height=page.rect.height)
                        out_page.insert_image(out_page.rect, stream=pix.tobytes("png"))
                    temporary = target.with_suffix(".partial.pdf")
                    output.save(temporary, garbage=4, deflate=True)
                    output.close()
                    temporary.replace(target)
                with fitz.open(target) as copy:
                    assert len(src) == len(copy)
                    assert all(not p.get_text().strip() for p in copy), "Text leaked into scan"
                    assert all(src[i].rect == copy[i].rect for i in range(len(src)))
            elif not target.exists():
                target.symlink_to(os.path.relpath(source, target.parent))
            rows.append({"doc_id": source.stem, "pages": len(src), "scanned": scanned,
                         "original_sha256": digest(source), "mixed_sha256": digest(target)})
            print(f"prepared {len(rows)}/30 scan={scanned} pages={len(src)}", flush=True)
    manifest = {**settings, "upstream_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "documents": rows, "total_pages": sum(x["pages"] for x in rows),
        "scanned_pages": sum(x["pages"] for x in rows if x["scanned"]),
        "questions_sha256": digest(ROOT / "data/warmup/questions/questions.json")}
    assert manifest["total_pages"] == 590
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # Page-level gold is recoverable from the historical ledger, independently of predictions.
    ledger = ROOT / "artifacts/warmup_runs/runs/submission_e2e_20260424_solver_narrowing_no_guard/grounding/ledger.jsonl"
    gold = {}
    for line in ledger.read_text().splitlines():
        item = json.loads(line)
        gold[item["question_id"]] = [[d, int(p)] for d, p in
            (v.rsplit(":", 1) for v in item["gold_pages"].split("|") if v)]
    assert len(gold) == 100 and sum(bool(v) for v in gold.values()) == 95
    (WORK / "gold_pages.json").write_text(json.dumps(gold, indent=2) + "\n")
    print(json.dumps({"documents": len(rows), "pages": manifest["total_pages"],
                      "scanned_pages": manifest["scanned_pages"],
                      "affected_questions": sum(any(d in chosen for d, _ in v) for v in gold.values())}), flush=True)


if __name__ == "__main__":
    main()
