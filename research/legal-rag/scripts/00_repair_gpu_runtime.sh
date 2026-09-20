#!/usr/bin/env bash
# 00_repair_gpu_runtime.sh — Recreate NVIDIA device nodes and verify CUDA visibility without reinstalling packages.
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
NVIDIA_MODPROBE="/usr/bin/nvidia-modprobe"

echo "=== Repairing NVIDIA GPU Runtime ==="

if [ ! -x "$NVIDIA_MODPROBE" ]; then
    echo "FAIL: nvidia-modprobe not found at $NVIDIA_MODPROBE"
    exit 1
fi

if [ ! -x "$PYTHON" ]; then
    echo "FAIL: Python interpreter not found at $PYTHON"
    exit 1
fi

echo "Running nvidia-modprobe..."
"$NVIDIA_MODPROBE"
"$NVIDIA_MODPROBE" -u -c 0

for NODE in /dev/nvidiactl /dev/nvidia0 /dev/nvidia-uvm; do
    if [ ! -e "$NODE" ]; then
        echo "FAIL: expected device node is still missing: $NODE"
        exit 1
    fi
done

echo "--- Device nodes ---"
ls -l /dev/nvidia*

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- nvidia-smi ---"
    nvidia-smi
fi

echo "--- Python runtime ---"
"$PYTHON" - <<'PY'
import sys

import torch

print(f"torch: {torch.__version__}")
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
print(f"torch.version.cuda: {torch.version.cuda}")

try:
    import paddle

    print(f"paddle: {paddle.__version__}")
    print(f"paddle.device.is_compiled_with_cuda: {paddle.device.is_compiled_with_cuda()}")
except Exception as exc:
    print(f"WARN: paddle check failed: {exc!r}")

if not torch.cuda.is_available():
    print("FAIL: CUDA is still unavailable after recreating NVIDIA device nodes.")
    sys.exit(1)
PY

echo "=== GPU Runtime Ready ==="
echo "Next check: ./scripts/02_download_models.sh --verify"
