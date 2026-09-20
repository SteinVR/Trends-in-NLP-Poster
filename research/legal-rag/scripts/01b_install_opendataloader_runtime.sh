#!/usr/bin/env bash
# 01b_install_opendataloader_runtime.sh - Install/verify OpenDataLoader + Docling hybrid runtime.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GIT_COMMON_DIR="$(git -C "$PROJECT_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
COMMON_ROOT="$PROJECT_ROOT"
if [ -n "$GIT_COMMON_DIR" ]; then
    COMMON_ROOT="$(dirname "$GIT_COMMON_DIR")"
fi

if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif [ -x "$COMMON_ROOT/.venv/bin/python" ]; then
    PYTHON="$COMMON_ROOT/.venv/bin/python"
else
    echo "FAIL: python runtime missing; run ./scripts/01_install_deps.sh first." >&2
    exit 1
fi

VENV_DIR="$(cd "$(dirname "$PYTHON")/.." && pwd)"
OPENDATALOADER_BIN="$VENV_DIR/bin/opendataloader-pdf"
OPENDATALOADER_HYBRID_BIN="$VENV_DIR/bin/opendataloader-pdf-hybrid"
INSTALL=1
SMOKE_PARSE=1

usage() {
    cat <<'USAGE'
Usage: ./scripts/01b_install_opendataloader_runtime.sh [options]

Options:
  --verify-only      Do not install; only verify current runtime.
  --no-smoke         Skip the local opendataloader-pdf smoke parse.
  -h, --help         Show this help.

This project uses the PyPI opendataloader-pdf package; no vendored build is used.
The hybrid backend is provided by opendataloader-pdf[hybrid] and exposed as:
  .venv/bin/opendataloader-pdf-hybrid --host 127.0.0.1 --port 5002 --device cpu --force-ocr
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --verify-only)
            INSTALL=0
            shift
            ;;
        --no-smoke)
            SMOKE_PARSE=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "FAIL: unknown option '$1'" >&2
            usage >&2
            exit 1
            ;;
    esac
done

require_cmd() {
    local name="$1"
    if ! command -v "$name" >/dev/null 2>&1; then
        echo "FAIL: required command is missing: $name" >&2
        exit 1
    fi
}

cd "$PROJECT_ROOT"

if [ "$INSTALL" -eq 1 ]; then
    require_cmd uv
    echo "=== Installing OpenDataLoader Hybrid Runtime ==="
    uv pip install --python "$PYTHON" "opendataloader-pdf[hybrid]>=2.3.0"
fi

echo "=== Verifying OpenDataLoader Runtime ==="
require_cmd java
echo "java: $(java -version 2>&1 | head -n 1)"
"$PYTHON" - <<'PY'
import importlib
from importlib.resources import files

for module_name in ("opendataloader_pdf", "docling", "fastapi", "uvicorn"):
    module = importlib.import_module(module_name)
    print(f"{module_name}: {getattr(module, '__version__', 'OK')}")

jar_path = files("opendataloader_pdf").joinpath("jar/opendataloader-pdf-cli.jar")
if not jar_path.is_file():
    raise SystemExit(f"FAIL: bundled OpenDataLoader JAR is missing: {jar_path}")
print(f"bundled jar: {jar_path}")
PY

for cli_path in "$OPENDATALOADER_BIN" "$OPENDATALOADER_HYBRID_BIN"; do
    if [ -x "$cli_path" ]; then
        echo "CLI OK: $cli_path"
    else
        echo "FAIL: expected CLI is missing: $cli_path" >&2
        exit 1
    fi
done

"$OPENDATALOADER_BIN" --help >/dev/null
"$OPENDATALOADER_HYBRID_BIN" --help >/dev/null

if [ "$SMOKE_PARSE" -eq 1 ]; then
    echo "=== Running OpenDataLoader Smoke Parse ==="
    smoke_dir="$(mktemp -d)"
    trap 'rm -rf "$smoke_dir"' EXIT
    "$PYTHON" - <<'PY' "$smoke_dir"
import sys
from pathlib import Path

import fitz

base = Path(sys.argv[1])
pdf_path = base / "smoke.pdf"
doc = fitz.open()
page = doc.new_page()
page.insert_text((72, 72), "DIFC LAW NO. 1\nArticle 1\nSmoke parse for opendataloader runtime.")
doc.save(pdf_path)
doc.close()
PY
    JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -Djava.awt.headless=true" \
        "$OPENDATALOADER_BIN" \
        "$smoke_dir/smoke.pdf" \
        --output-dir "$smoke_dir/out" \
        --format json \
        --table-method cluster \
        --reading-order xycut \
        --quiet
    if [ ! -f "$smoke_dir/out/smoke.json" ]; then
        echo "FAIL: opendataloader smoke parse did not produce JSON output" >&2
        exit 1
    fi
    echo "opendataloader smoke parse: OK"
fi

echo "=== OpenDataLoader Runtime OK ==="
echo "Hybrid backend command: $OPENDATALOADER_HYBRID_BIN --host 127.0.0.1 --port 5002 --device cpu --force-ocr --log-level info"
