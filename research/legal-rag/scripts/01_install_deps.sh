#!/usr/bin/env bash
# 01_install_deps.sh - Create/repair .venv and install the canonical runtime stack.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GIT_COMMON_DIR="$(git -C "$PROJECT_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
COMMON_ROOT="$PROJECT_ROOT"
if [ -n "$GIT_COMMON_DIR" ]; then
    COMMON_ROOT="$(dirname "$GIT_COMMON_DIR")"
fi

if [ -f "$PROJECT_ROOT/.python-version" ]; then
    DEFAULT_PYTHON_VERSION="$(tr -d '[:space:]' < "$PROJECT_ROOT/.python-version")"
else
    DEFAULT_PYTHON_VERSION="3.12"
fi
PYTHON_REQUESTED="${PYTHON_BIN:-${PYTHON_VERSION:-$DEFAULT_PYTHON_VERSION}}"
SKIP_PADDLE_GPU=0
VERIFY=1
PADDLE_GPU_VERSION="3.3.0"

usage() {
    cat <<'USAGE'
Usage: ./scripts/01_install_deps.sh [options]

Options:
  --python BIN           Python interpreter/version for a new .venv (default: $PYTHON_BIN, $PYTHON_VERSION, .python-version, then 3.12).
  --skip-paddle-gpu      Do not install the pinned Paddle CUDA wheel.
  --cpu                  Alias for --skip-paddle-gpu.
  --no-verify            Install only; skip import/CLI verification.
  -h, --help             Show this help.

Canonical install includes:
  - locked project sync from uv.lock with paddle+dev extras
  - opendataloader-pdf[hybrid] / Docling hybrid backend runtime
  - pinned Paddle GPU wheel unless --skip-paddle-gpu is used
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --python)
            PYTHON_REQUESTED="$2"
            shift 2
            ;;
        --skip-paddle-gpu|--cpu)
            SKIP_PADDLE_GPU=1
            shift
            ;;
        --no-verify)
            VERIFY=0
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

resolve_python() {
    if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
        printf '%s\n' "$PROJECT_ROOT/.venv/bin/python"
        return
    fi
    if [ -x "$COMMON_ROOT/.venv/bin/python" ]; then
        printf '%s\n' "$COMMON_ROOT/.venv/bin/python"
        return
    fi

    echo "=== Creating .venv ===" >&2
    echo "Target: $COMMON_ROOT/.venv" >&2
    uv venv "$COMMON_ROOT/.venv" --python "$PYTHON_REQUESTED"
    printf '%s\n' "$COMMON_ROOT/.venv/bin/python"
}

select_paddle_gpu_wheel_url() {
    local python_minor
    python_minor="$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

    case "$python_minor" in
        3.11)
            printf '%s\n' "https://paddle-whl.bj.bcebos.com/stable/cu126/paddlepaddle-gpu/paddlepaddle_gpu-${PADDLE_GPU_VERSION}-cp311-cp311-linux_x86_64.whl"
            ;;
        3.12)
            printf '%s\n' "https://paddle-whl.bj.bcebos.com/stable/cu126/paddlepaddle-gpu/paddlepaddle_gpu-${PADDLE_GPU_VERSION}-cp312-cp312-linux_x86_64.whl"
            ;;
        *)
            echo "FAIL: pinned paddlepaddle-gpu wheel is only defined for Python 3.11 and 3.12" >&2
            exit 1
            ;;
    esac
}

require_cmd uv
if [ ! -f "$PROJECT_ROOT/uv.lock" ]; then
    echo "FAIL: uv.lock is required for deterministic setup." >&2
    exit 1
fi
PYTHON="$(resolve_python)"
VENV_DIR="$(cd "$(dirname "$PYTHON")/.." && pwd)"
HF_BIN="$VENV_DIR/bin/hf"
AGENTIC_RAG_BIN="$VENV_DIR/bin/agentic-rag"
OPENDATALOADER_BIN="$VENV_DIR/bin/opendataloader-pdf"
OPENDATALOADER_HYBRID_BIN="$VENV_DIR/bin/opendataloader-pdf-hybrid"

cd "$PROJECT_ROOT"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "=== Installing Dependencies ==="
echo "Project root: $PROJECT_ROOT"
echo "Common root: $COMMON_ROOT"
echo "Python: $PYTHON ($($PYTHON --version 2>&1))"

UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync --locked --python "$PYTHON" --extra paddle --extra dev

if [ "$SKIP_PADDLE_GPU" -eq 0 ]; then
    PADDLE_GPU_WHEEL_URL="$(select_paddle_gpu_wheel_url)"
    # Install Paddle GPU as a direct wheel without dependency resolution so the
    # existing CUDA/NVIDIA stack and local PyTorch source selection stay stable.
    uv pip install --python "$PYTHON" --no-deps "$PADDLE_GPU_WHEEL_URL"
else
    echo "=== Skipping pinned Paddle GPU wheel by request ==="
fi

if [ "$VERIFY" -eq 0 ]; then
    echo "=== Dependencies Installed (verification skipped) ==="
    exit 0
fi

echo "--- Validating Python imports ---"
"$PYTHON" - <<'PY'
import importlib

checks = [
    ("sentence_transformers", "sentence-transformers"),
    ("transformers", "transformers"),
    ("huggingface_hub", "huggingface_hub"),
    ("qdrant_client", "qdrant-client"),
    ("torch", "torch"),
    ("fitz", "pymupdf"),
    ("httpx", "httpx"),
    ("pydantic", "pydantic"),
    ("dotenv", "python-dotenv"),
    ("paddle", "paddle"),
    ("paddleocr", "paddleocr"),
    ("opendataloader_pdf", "opendataloader-pdf"),
    ("docling", "docling"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
]
for module_name, label in checks:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", "OK")
    print(f"{label}: {version}")

import torch
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")

import paddle
print(f"paddle.device.is_compiled_with_cuda: {paddle.device.is_compiled_with_cuda()}")
PY

for cli_path in "$HF_BIN" "$AGENTIC_RAG_BIN" "$OPENDATALOADER_BIN" "$OPENDATALOADER_HYBRID_BIN"; do
    if [ -x "$cli_path" ]; then
        echo "CLI OK: $cli_path"
    else
        echo "FAIL: expected CLI is missing: $cli_path" >&2
        exit 1
    fi
done

require_cmd java
echo "java: $(java -version 2>&1 | head -n 1)"
"$OPENDATALOADER_BIN" --help >/dev/null
"$OPENDATALOADER_HYBRID_BIN" --help >/dev/null

echo "--- OCR model cache status ---"
OCR_MODEL_ROOT="${HOME}/.paddlex/official_models"
MISSING_OCR_MODELS=0
for MODEL_DIR in \
    "$OCR_MODEL_ROOT/PP-OCRv5_server_det" \
    "$OCR_MODEL_ROOT/en_PP-OCRv5_mobile_rec"
do
    if [ -d "$MODEL_DIR" ]; then
        echo "OCR model cache: present ($MODEL_DIR)"
    else
        echo "WARN: OCR model cache missing ($MODEL_DIR)"
        MISSING_OCR_MODELS=1
    fi
done
if [ "$MISSING_OCR_MODELS" -eq 1 ]; then
    echo "WARN: run ./scripts/02_download_models.sh to provision missing OCR caches."
fi

echo "=== Dependencies OK ==="
