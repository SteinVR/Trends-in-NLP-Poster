#!/usr/bin/env bash
# 03_lint_and_test.sh — Run shell syntax checks, lint, compile check, and tests.
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
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    echo "FAIL: python runtime missing; expected .venv/bin/python, common .venv, or python3 on PATH"
    exit 1
fi
cd "$PROJECT_ROOT"

export PYTHONPATH=.
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "=== Quality Gates ==="

echo "--- bash -n scripts/*.sh ---"
for script_path in scripts/*.sh; do
    bash -n "$script_path"
done
echo "shell syntax: PASS"

CHECK_DIRS=()
for path in src tools scripts; do
    if [ -e "$path" ]; then
        CHECK_DIRS+=("$path")
    fi
done
if [ -d tests ]; then
    CHECK_DIRS+=(tests)
fi

echo "--- ruff check ---"
if "$PYTHON" -m ruff --version >/dev/null 2>&1; then
    "$PYTHON" -m ruff check "${CHECK_DIRS[@]}"
elif command -v ruff >/dev/null 2>&1; then
    ruff check "${CHECK_DIRS[@]}"
elif command -v uv >/dev/null 2>&1; then
    uv tool run ruff check "${CHECK_DIRS[@]}"
else
    echo "FAIL: ruff is unavailable; install dev dependencies or install uv."
    exit 1
fi
echo "ruff: PASS"

echo "--- compileall ---"
"$PYTHON" -m compileall -q "${CHECK_DIRS[@]}"
echo "compileall: PASS"

echo "--- pytest ---"
if [ -d tests ]; then
    "$PYTHON" -m pytest tests/ -q --tb=short
    echo "pytest: PASS"
else
    echo "pytest: SKIP (tests/ is not part of the public snapshot)"
fi

echo "=== All Gates Passed ==="
