"""Read/write helpers for raw parser sidecar artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from src.common.schemas import RawParseSidecar


def write_raw_parse_sidecar(directory: str | Path, sidecar: RawParseSidecar) -> Path:
    """Persist one raw parse sidecar as JSON and return the written path."""

    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{sidecar.doc_id}.json"
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(sidecar.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return target_path


def load_raw_parse_sidecar(path: str | Path) -> RawParseSidecar:
    """Load one raw parse sidecar from disk."""

    payload = Path(path).read_text(encoding="utf-8")
    return RawParseSidecar.model_validate_json(payload)

