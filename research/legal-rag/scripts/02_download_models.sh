#!/usr/bin/env bash
# 02_download_models.sh — Download retrieval models with visible progress and verify local GPU readiness.
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
    echo "FAIL: python runtime missing; expected .venv/bin/python in $PROJECT_ROOT or $COMMON_ROOT"
    exit 1
fi
VENV_DIR="$(cd "$(dirname "$PYTHON")/.." && pwd)"
HF_BIN="$VENV_DIR/bin/hf"
EMBED_REPO="Qwen/Qwen3-Embedding-0.6B"
RERANK_REPO="Qwen/Qwen3-Reranker-0.6B"
MAX_WORKERS="${HF_DOWNLOAD_MAX_WORKERS:-8}"
CONFIG="configs/baseline/v001/config.yaml"
ENV_FILE=".env"
MODE="download"

usage() {
    cat <<'EOF'
Usage:
  ./scripts/02_download_models.sh [--config PATH] [--env-file PATH]
  ./scripts/02_download_models.sh --verify [--config PATH] [--env-file PATH]

Modes:
  default   Download the embedder and reranker into the active Hugging Face cache and provision the OCR model cache if missing.
  --verify  Validate retrieval cache offline, validate OCR cache, then require a working CUDA runtime for GPU smoke tests.
EOF
}

ensure_tooling() {
    if [ ! -x "$PYTHON" ]; then
        echo "FAIL: Python interpreter not found at $PYTHON"
        exit 1
    fi
    if [ ! -x "$HF_BIN" ]; then
        echo "FAIL: Hugging Face CLI not found at $HF_BIN"
        exit 1
    fi
}

resolve_default_env_file() {
    if [ "$ENV_FILE" = ".env" ] && [ ! -f "$PROJECT_ROOT/.env" ] && [ -f "$COMMON_ROOT/.env" ]; then
        ENV_FILE="$COMMON_ROOT/.env"
    fi
}

print_cache_context() {
    "$PYTHON" - <<'PY'
from huggingface_hub.constants import HF_HUB_CACHE
print(f"HF cache: {HF_HUB_CACHE}")
PY
}

resolve_ocr_cache_context() {
    "$PYTHON" - <<PY
from pathlib import Path
import shlex

from src.common.config import load_app_config
from src.ingestion.ocr_engine import DEFAULT_TEXT_DETECTION_MODEL_NAME, DEFAULT_TEXT_RECOGNITION_MODEL_NAME

config = load_app_config(Path(${CONFIG@Q}), env_file=Path(${ENV_FILE@Q}))
root = config.ingestion.ocr_model_root_dir.expanduser()
det = (config.ingestion.ocr_text_detection_model_dir or (root / DEFAULT_TEXT_DETECTION_MODEL_NAME)).expanduser()
rec = (config.ingestion.ocr_text_recognition_model_dir or (root / DEFAULT_TEXT_RECOGNITION_MODEL_NAME)).expanduser()
print(f"OCR_MODEL_ROOT={shlex.quote(str(root))}")
print(f"OCR_DET_DIR={shlex.quote(str(det))}")
print(f"OCR_REC_DIR={shlex.quote(str(rec))}")
print(f"OCR_DET_NAME={shlex.quote(DEFAULT_TEXT_DETECTION_MODEL_NAME)}")
print(f"OCR_REC_NAME={shlex.quote(DEFAULT_TEXT_RECOGNITION_MODEL_NAME)}")
PY
}

ensure_ocr_cache() {
    eval "$(resolve_ocr_cache_context)"

    if [ -d "$OCR_DET_DIR" ] && [ -d "$OCR_REC_DIR" ]; then
        echo "=== OCR Cache Already Present ==="
        echo "OCR det cache: $OCR_DET_DIR"
        echo "OCR rec cache: $OCR_REC_DIR"
        return
    fi

    echo "=== Provisioning OCR Cache ==="
    echo "OCR model root: $OCR_MODEL_ROOT"
    "$PYTHON" - <<PY
from paddleocr import PaddleOCR

PaddleOCR(
    device="cpu",
    enable_hpi=False,
    enable_mkldnn=False,
    cpu_threads=1,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_detection_model_name=${OCR_DET_NAME@Q},
    text_recognition_model_name=${OCR_REC_NAME@Q},
)
PY

    if [ ! -d "$OCR_DET_DIR" ] || [ ! -d "$OCR_REC_DIR" ]; then
        echo "FAIL: OCR cache provision step completed without materializing both model directories."
        exit 1
    fi
    echo "OCR det cache: $OCR_DET_DIR"
    echo "OCR rec cache: $OCR_REC_DIR"
}

verify_ocr_cache() {
    eval "$(resolve_ocr_cache_context)"

    echo "=== Verifying OCR Cache ==="
    for model_dir in "$OCR_DET_DIR" "$OCR_REC_DIR"; do
        if [ -d "$model_dir" ]; then
            echo "OCR model cache: present ($model_dir)"
        else
            echo "FAIL: required OCR model cache is missing: $model_dir"
            exit 1
        fi
    done

    "$PYTHON" - <<PY
import paddle
from paddleocr import PaddleOCR

print(f"paddle: {paddle.__version__}")
print(f"paddle.device.is_compiled_with_cuda: {paddle.device.is_compiled_with_cuda()}")
print(f"paddle.device.cuda.device_count: {paddle.device.cuda.device_count()}")

PaddleOCR(
    device="gpu:0",
    enable_hpi=False,
    enable_mkldnn=False,
    cpu_threads=1,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_detection_model_name=${OCR_DET_NAME@Q},
    text_detection_model_dir=${OCR_DET_DIR@Q},
    text_recognition_model_name=${OCR_REC_NAME@Q},
    text_recognition_model_dir=${OCR_REC_DIR@Q},
)
print("OCR GPU init OK")
PY
}

download_models() {
    export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

    echo "=== Downloading Retrieval Models ==="
    print_cache_context
    echo "HF max workers: $MAX_WORKERS"

    echo "--- $EMBED_REPO ---"
    "$HF_BIN" download --max-workers "$MAX_WORKERS" "$EMBED_REPO"

    echo "--- $RERANK_REPO ---"
    "$HF_BIN" download --max-workers "$MAX_WORKERS" "$RERANK_REPO"

    echo "--- Cache summary ---"
    "$PYTHON" - <<'PY'
from huggingface_hub import scan_cache_dir

repos_of_interest = {
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Reranker-0.6B",
}
cache = scan_cache_dir()
for repo in cache.repos:
    if repo.repo_id in repos_of_interest:
        print(f"{repo.repo_id}: {repo.size_on_disk / 1e6:.1f} MB")
PY

    ensure_ocr_cache

    echo "=== Download Complete ==="
}

verify_models() {
    export HF_HUB_OFFLINE=1
    export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

    echo "=== Verifying Retrieval Models ==="
    print_cache_context

    "$PYTHON" - <<'PY'
import sys

import torch
from sentence_transformers import SentenceTransformer

from src.retrieval.reranker import TransformersQwenRerankerBackend

EMBED_REPO = "Qwen/Qwen3-Embedding-0.6B"
RERANK_QUERY = "what is the filing deadline?"
RERANK_DOCUMENT = "the filing deadline is 30 days after service."


def reranker_forward(device: str) -> float:
    backend = TransformersQwenRerankerBackend(device=device)
    score = backend.score(RERANK_QUERY, [RERANK_DOCUMENT])[0]
    return float(score)


print(f"torch: {torch.__version__}")
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"torch.cuda.device_count: {torch.cuda.device_count()}")

embedder_cpu = SentenceTransformer(
    EMBED_REPO,
    trust_remote_code=True,
    local_files_only=True,
    device="cpu",
)
cpu_vectors = embedder_cpu.encode(
    ["smoke test"],
    convert_to_numpy=True,
    show_progress_bar=False,
    normalize_embeddings=True,
)
print(f"Embedder CPU smoke OK: shape={tuple(cpu_vectors.shape)}")

cpu_score = reranker_forward("cpu")
print(f"Reranker CPU smoke OK: score={cpu_score:.4f}")

if not torch.cuda.is_available():
    print("FAIL: model cache is valid, but CUDA is unavailable on this host.")
    sys.exit(2)

embedder_gpu = SentenceTransformer(
    EMBED_REPO,
    trust_remote_code=True,
    local_files_only=True,
    device="cuda",
)
gpu_vectors = embedder_gpu.encode(
    ["gpu smoke test"],
    convert_to_numpy=True,
    show_progress_bar=False,
    normalize_embeddings=True,
)
print(f"Embedder GPU smoke OK: shape={tuple(gpu_vectors.shape)}")

gpu_score = reranker_forward("cuda")
print(f"Reranker GPU smoke OK: score={gpu_score:.4f}")
print("=== Verification Complete ===")
PY

    verify_ocr_cache
}

main() {
    ensure_tooling

    while [ $# -gt 0 ]; do
        case "$1" in
            --config)
                CONFIG="$2"
                shift 2
                ;;
            --env-file)
                ENV_FILE="$2"
                shift 2
                ;;
            --verify)
                MODE="verify"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "FAIL: unknown argument '$1'"
                usage
                exit 1
                ;;
        esac
    done

    resolve_default_env_file

    case "$MODE" in
        download)
            download_models
            ;;
        verify)
            verify_models
            ;;
        *)
            echo "FAIL: unknown mode '$MODE'"
            usage
            exit 1
            ;;
    esac
}

main "$@"
