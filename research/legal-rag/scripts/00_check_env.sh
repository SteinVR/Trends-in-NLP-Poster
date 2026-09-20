#!/usr/bin/env bash
# 00_check_env.sh — Validate environment prerequisites for the RAG pipeline.
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
    echo "FAIL: .venv not found at $PROJECT_ROOT/.venv or $COMMON_ROOT/.venv"
    exit 1
fi
if [ -f "$PROJECT_ROOT/.env" ]; then
    ENV_FILE="$PROJECT_ROOT/.env"
elif [ -f "$COMMON_ROOT/.env" ]; then
    ENV_FILE="$COMMON_ROOT/.env"
else
    ENV_FILE=""
fi
WARN_COUNT=0

warn() {
    echo "WARN: $1"
    WARN_COUNT=$((WARN_COUNT + 1))
}

echo "=== Environment Check ==="
echo "Project root: $PROJECT_ROOT"
echo "Common root: $COMMON_ROOT"

# Python version
if [ ! -x "$PYTHON" ]; then
    echo "FAIL: Python runtime missing"
    exit 1
fi
PY_VERSION=$("$PYTHON" --version 2>&1)
echo "Python: $PY_VERSION"

# .env file
if [ -z "$ENV_FILE" ]; then
    echo "FAIL: .env file not found. Create .env with API keys or export them in the process environment."
    exit 1
fi
echo ".env: present ($ENV_FILE)"

provider_key_count=0
while IFS='|' read -r KEY VALUE_LEN; do
    if [ -z "$KEY" ]; then
        continue
    fi
    provider_key_count=$((provider_key_count + 1))
    echo "$KEY: set (${VALUE_LEN} chars)"
done < <("$PYTHON" - "$ENV_FILE" <<'PY'
import os
import sys
from pathlib import Path

try:
    from dotenv import dotenv_values
except Exception:
    dotenv_values = None

env_file = Path(sys.argv[1])
env_values = dotenv_values(env_file) if dotenv_values is not None and env_file.exists() else {}
for key in ("CODEX_OAUTH_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY"):
    value = os.environ.get(key, "").strip()
    if not value:
        value = str(env_values.get(key) or "").strip()
    if not value or value == "set_in_real_env":
        continue
    print(f"{key}|{len(value)}")
PY
)
if [ "$provider_key_count" -eq 0 ]; then
    warn "No provider credential found (checked CODEX_OAUTH_TOKEN, OPENAI_API_KEY, CODEX_API_KEY)."
fi

# GPU
if command -v nvidia-smi &>/dev/null; then
    echo "--- GPU runtime ---"

    if GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader >/dev/null 2>&1); then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null)
        echo "GPU: $GPU_INFO"
    else
        echo "GPU: unavailable"
        warn "NVIDIA tooling is present but the driver or device is not accessible."
        warn "Python GPU packages may install, but CUDA runtime checks will fail until host GPU access is fixed."
    fi

    if [ -d /proc/driver/nvidia/gpus ] && find /proc/driver/nvidia/gpus -mindepth 1 -maxdepth 1 -type d | grep -q .; then
        GPU_COUNT=$(find /proc/driver/nvidia/gpus -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "Kernel NVIDIA GPUs: $GPU_COUNT"
    else
        echo "Kernel NVIDIA GPUs: not detected"
    fi

    MISSING_NODES=0
    for NODE in /dev/nvidiactl /dev/nvidia0 /dev/nvidia-uvm; do
        if [ -e "$NODE" ]; then
            echo "Device node: present ($NODE)"
        else
            echo "Device node: missing ($NODE)"
            MISSING_NODES=1
        fi
    done

    if [ "$MISSING_NODES" -eq 1 ]; then
        warn "Required NVIDIA device nodes are missing."
        if [ -d /proc/driver/nvidia/gpus ] && find /proc/driver/nvidia/gpus -mindepth 1 -maxdepth 1 -type d | grep -q .; then
            warn "Kernel sees an NVIDIA GPU, but /dev/nvidia* was not created. This usually points to a broken nvidia-modprobe/udev/device-node path."
            warn "Run ./scripts/00_repair_gpu_runtime.sh to recreate the device nodes without reinstalling NVIDIA/CUDA packages."
        fi
    fi
else
    echo "GPU: not detected (CPU mode)"
fi

# Disk space
DISK_FREE=$(df -h "$PROJECT_ROOT" 2>/dev/null | tail -1 | awk '{print $4}')
echo "Disk free: $DISK_FREE"

if command -v java >/dev/null 2>&1; then
    echo "Java: $(java -version 2>&1 | head -n 1)"
else
    warn "Java runtime is missing; opendataloader-pdf structural conversion requires java."
fi

# Python runtime readiness
echo "--- Python runtime ---"
"$PYTHON" - <<'PY'
from pathlib import Path

try:
    import torch

    print(f"torch: {torch.__version__}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
except Exception as exc:
    print(f"WARN: torch check failed: {exc!r}")

try:
    import paddle

    print(f"paddle: {paddle.__version__}")
    print(f"paddle.device.is_compiled_with_cuda: {paddle.device.is_compiled_with_cuda()}")
except Exception as exc:
    print(f"WARN: paddle check failed: {exc!r}")

try:
    import opendataloader_pdf

    print(f"opendataloader_pdf: {getattr(opendataloader_pdf, '__version__', 'present')}")
except Exception as exc:
    print(f"WARN: opendataloader_pdf check failed: {exc!r}")

try:
    import docling

    print(f"docling: {getattr(docling, '__version__', 'present')}")
except Exception as exc:
    print(f"WARN: docling check failed: {exc!r}")

hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
embedder_dir = hf_cache / "models--Qwen--Qwen3-Embedding-0.6B"
reranker_dir = hf_cache / "models--Qwen--Qwen3-Reranker-0.6B"
ocr_root = Path.home() / ".paddlex" / "official_models"
ocr_det_dir = ocr_root / "PP-OCRv5_server_det"
ocr_rec_dir = ocr_root / "en_PP-OCRv5_mobile_rec"
print(f"HF cache: {hf_cache}")
print(f"Embedder cache: {'present' if embedder_dir.exists() else 'missing'}")
print(f"Reranker cache: {'present' if reranker_dir.exists() else 'missing'}")
print(f"OCR det cache: {'present' if ocr_det_dir.exists() else 'missing'}")
print(f"OCR rec cache: {'present' if ocr_rec_dir.exists() else 'missing'}")
PY

if [ -x "$(dirname "$PYTHON")/opendataloader-pdf" ]; then
    echo "OpenDataLoader CLI: present ($(dirname "$PYTHON")/opendataloader-pdf)"
else
    warn "OpenDataLoader CLI missing. Run ./scripts/01_install_deps.sh or ./scripts/01b_install_opendataloader_runtime.sh."
fi

if [ -x "$(dirname "$PYTHON")/opendataloader-pdf-hybrid" ]; then
    echo "OpenDataLoader hybrid CLI: present ($(dirname "$PYTHON")/opendataloader-pdf-hybrid)"
else
    warn "OpenDataLoader hybrid CLI missing. Docling hybrid backend will not start."
fi

# Data directories
if [ -d "$PROJECT_ROOT/data/warmup/documents/pdfs" ]; then
    PDF_COUNT=$(find "$PROJECT_ROOT/data/warmup/documents/pdfs" -name "*.pdf" | wc -l)
    echo "Warmup PDFs: $PDF_COUNT"
else
    echo "Warmup PDFs: not present"
    echo "Hint: run ./scripts/02a_sync_datasets.sh --phase warmup"
fi

if [ -f "$PROJECT_ROOT/data/warmup/questions/questions.json" ]; then
    echo "Warmup questions: present"
else
    echo "Warmup questions: not present"
    echo "Hint: run ./scripts/02a_sync_datasets.sh --phase warmup"
fi

if [ -d "$PROJECT_ROOT/data/final/documents/pdfs" ]; then
    FINAL_PDF_COUNT=$(find "$PROJECT_ROOT/data/final/documents/pdfs" -name "*.pdf" | wc -l)
    echo "Final PDFs: $FINAL_PDF_COUNT"
else
    echo "Final PDFs: not present"
    echo "Hint: run ./scripts/02a_sync_datasets.sh --phase final"
fi

if [ -f "$PROJECT_ROOT/data/final/questions/questions.json" ]; then
    echo "Final questions: present"
else
    echo "Final questions: not present"
    echo "Hint: run ./scripts/02a_sync_datasets.sh --phase final"
fi

if [ "$WARN_COUNT" -gt 0 ]; then
    echo "=== Environment Check Completed With $WARN_COUNT Warning(s) ==="
else
    echo "=== Environment OK ==="
fi
