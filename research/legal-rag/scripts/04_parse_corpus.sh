#!/usr/bin/env bash
# 04_parse_corpus.sh — Parse extracted PDFs into canonical corpus artifacts.
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
cd "$PROJECT_ROOT"

export PYTHONPATH=.
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

CONFIG="configs/baseline/v001/config.yaml"
ENV_FILE=".env"
PHASE_OVERRIDE=""
FORCE=0
DRY_RUN=0
DOC_IDS=()
POSITIONAL=()

resolve_default_env_file() {
    if [ "$ENV_FILE" = ".env" ] && [ ! -f "$PROJECT_ROOT/.env" ] && [ -f "$COMMON_ROOT/.env" ]; then
        ENV_FILE="$COMMON_ROOT/.env"
    fi
}

usage() {
    cat <<'EOF'
Usage:
  ./scripts/04_parse_corpus.sh [CONFIG] [PHASE]
  ./scripts/04_parse_corpus.sh --config PATH --env-file PATH [--phase PHASE] [--force] [--dry-run] [--doc-id DOC_ID]...
EOF
}

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
        --phase)
            PHASE_OVERRIDE="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --doc-id)
            DOC_IDS+=("$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "FAIL: unknown option '$1'"
            usage
            exit 1
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [ "${#POSITIONAL[@]}" -ge 1 ]; then
    CONFIG="${POSITIONAL[0]}"
fi
if [ "${#POSITIONAL[@]}" -ge 2 ]; then
    PHASE_OVERRIDE="${POSITIONAL[1]}"
fi
resolve_default_env_file

CONTEXT_ARGS=(--config "$CONFIG" --env-file "$ENV_FILE")
if [ -n "$PHASE_OVERRIDE" ]; then
    CONTEXT_ARGS+=(--phase "$PHASE_OVERRIDE")
fi
eval "$("$PYTHON" tools/operator_context.py "${CONTEXT_ARGS[@]}" --format shell)"

echo "=== Parse Corpus ==="
echo "  Config: $CONFIG"
echo "  Env file: $ENV_FILE"
echo "  Phase: $PHASE"
echo "  Documents: $DOCUMENTS_DIR"
echo "  Corpus path: $CORPUS_PATH"
echo "  Page map path: $PAGE_MAP_PATH"
echo "  Manifest path: $CORPUS_MANIFEST_PATH"
echo "  Debug dir: $DEBUG_DIR"

COMMAND=("$PYTHON" -m src.main --config "$CONFIG" --env-file "$ENV_FILE" parse-corpus --phase "$PHASE")
if [ "$DRY_RUN" -eq 1 ]; then
    COMMAND+=(--dry-run)
fi
if [ "$FORCE" -eq 1 ]; then
    COMMAND+=(--force)
fi
for doc_id in "${DOC_IDS[@]}"; do
    COMMAND+=(--doc-id "$doc_id")
done

"${COMMAND[@]}"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "=== Dry Run Complete ==="
    exit 0
fi

echo "--- Validating artifacts ---"
for artifact_path in "$CORPUS_PATH" "$PAGE_MAP_PATH" "$CORPUS_MANIFEST_PATH"; do
    if [ -f "$artifact_path" ]; then
        SIZE=$(stat --printf="%s" "$artifact_path")
        echo "  $(basename "$artifact_path"): ${SIZE} bytes"
    else
        echo "  FAIL: missing artifact $artifact_path"
        exit 1
    fi
done

if [ -d "$DEBUG_DIR" ]; then
    DEBUG_COUNT=$(find "$DEBUG_DIR" -maxdepth 1 -type f -name '*.md' | wc -l)
    echo "  debug/: $DEBUG_COUNT markdown files"
else
    echo "  FAIL: debug directory missing at $DEBUG_DIR"
    exit 1
fi

PAGE_COUNT=$("$PYTHON" - <<PY
from pathlib import Path

corpus_path = Path("$CORPUS_PATH")
lines = [line for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
print(len(lines))
PY
)
echo "  Total pages: $PAGE_COUNT"

echo "=== Corpus OK ==="
