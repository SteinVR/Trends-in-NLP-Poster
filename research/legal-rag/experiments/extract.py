"""Isolated native/PaddleOCR extraction with resumable page checkpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import fitz
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiment_data"


def make_ocr():
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PaddleOCR
    return PaddleOCR(device="cpu", enable_mkldnn=True, cpu_threads=4,
                     use_doc_orientation_classify=False, use_doc_unwarping=False,
                     use_textline_orientation=False, text_det_limit_side_len=1536,
                     text_det_limit_type="max",
                     text_detection_model_name="PP-OCRv5_server_det",
                     text_recognition_model_name="en_PP-OCRv5_mobile_rec",
                     text_recognition_batch_size=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["original", "mixed"], required=True)
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--max-new-pages", type=int, default=0)
    args = parser.parse_args()
    source = ROOT / "data/warmup/documents/pdfs" if args.corpus == "original" else DATA / "mixed/pdfs"
    out = DATA / "pages" / f"{args.corpus}-ocr{int(args.ocr)}"
    out.mkdir(parents=True, exist_ok=True)
    engine = None
    done = 0
    created = 0
    for path in sorted(source.glob("*.pdf")):
        with fitz.open(path) as doc:
            for page in doc:
                dst = out / f"{path.stem}-{page.number+1:04d}.json"
                done += 1
                if dst.exists():
                    continue
                native = page.get_text("text").strip()
                text, mode, tables = native, "native", []
                # Same native extraction in all conditions. OCR is triggered only by scant text.
                if args.ocr and len(native) < 50:
                    if engine is None:
                        engine = make_ocr()
                    pix = page.get_pixmap(dpi=300, colorspace=fitz.csRGB, alpha=False)
                    im = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
                    results = list(engine.predict(im))
                    lines = [str(t) for res in results for t in res.get("rec_texts", [])]
                    text = "\n".join(lines).strip()
                    mode = "ocr"
                    if not text:
                        raise RuntimeError(f"OCR returned no text for {path.stem}:{page.number+1}")
                elif native:
                    try:
                        tables = [t.to_markdown() for t in page.find_tables().tables]
                    except Exception as exc:
                        # Extraction problems are recorded, never replaced with original text.
                        tables = []
                        print(f"table extraction warning {path.stem}:{page.number+1}: {type(exc).__name__}", flush=True)
                row = {"doc_id": path.stem, "page_number": page.number+1, "text": text,
                       "source_mode": mode, "native_chars": len(native), "tables": tables,
                       "corpus": args.corpus, "ocr_enabled": args.ocr}
                temp = dst.with_suffix(".tmp")
                temp.write_text(json.dumps(row, ensure_ascii=False) + "\n")
                temp.replace(dst)
                print(f"{args.corpus} ocr={args.ocr} {done}/590 {mode} chars={len(text)}", flush=True)
                created += 1
                if args.max_new_pages and created >= args.max_new_pages:
                    return
    files = sorted(out.glob("*.json"))
    assert len(files) == 590, len(files)
    merged = out.with_suffix(".jsonl")
    merged.write_text("".join(p.read_text() for p in files))
    print(f"completed {merged}", flush=True)


if __name__ == "__main__":
    main()
