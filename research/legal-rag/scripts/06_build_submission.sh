#!/usr/bin/env bash
# 06_build_submission.sh — Run QA pipeline and build submission artifacts.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GIT_COMMON_DIR="$(git -C "$PROJECT_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
COMMON_ROOT="$PROJECT_ROOT"
if [ -n "$GIT_COMMON_DIR" ]; then
    COMMON_ROOT="$(dirname "$GIT_COMMON_DIR")"
fi

if [ -x "$COMMON_ROOT/.venv/bin/python" ]; then
    PYTHON="$COMMON_ROOT/.venv/bin/python"
elif [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
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
CUSTOM_QUESTIONS_PATH=""
CUSTOM_DOCUMENTS_DIR=""
CUSTOM_OUTPUT_DIR=""
RESUME_FROM_CHECKPOINT=0
CHECKPOINT_PATH=""
CHECKPOINT_EVERY=1
PREFLIGHT=0
POSITIONAL=()

resolve_default_env_file() {
    if [ "$ENV_FILE" = ".env" ] && [ ! -f "$PROJECT_ROOT/.env" ] && [ -f "$COMMON_ROOT/.env" ]; then
        ENV_FILE="$COMMON_ROOT/.env"
    fi
}

usage() {
    cat <<'EOF'
Usage:
  ./scripts/06_build_submission.sh [CONFIG] [PHASE] [QUESTIONS_PATH] [OUTPUT_DIR]
  ./scripts/06_build_submission.sh --config PATH --env-file PATH [--phase PHASE] [--questions-path PATH] [--documents-dir PATH] [--output-dir PATH] [--resume-from-checkpoint] [--checkpoint-path PATH] [--checkpoint-every N] [--preflight]
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
        --questions-path)
            CUSTOM_QUESTIONS_PATH="$2"
            shift 2
            ;;
        --documents-dir)
            CUSTOM_DOCUMENTS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            CUSTOM_OUTPUT_DIR="$2"
            shift 2
            ;;
        --resume-from-checkpoint)
            RESUME_FROM_CHECKPOINT=1
            shift
            ;;
        --checkpoint-path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --checkpoint-every)
            CHECKPOINT_EVERY="$2"
            shift 2
            ;;
        --preflight)
            PREFLIGHT=1
            shift
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
if [ "${#POSITIONAL[@]}" -ge 3 ] && [ -z "$CUSTOM_QUESTIONS_PATH" ]; then
    CUSTOM_QUESTIONS_PATH="${POSITIONAL[2]}"
fi
if [ "${#POSITIONAL[@]}" -ge 4 ] && [ -z "$CUSTOM_OUTPUT_DIR" ]; then
    CUSTOM_OUTPUT_DIR="${POSITIONAL[3]}"
fi
resolve_default_env_file

CONTEXT_ARGS=(--config "$CONFIG" --env-file "$ENV_FILE")
if [ -n "$PHASE_OVERRIDE" ]; then
    CONTEXT_ARGS+=(--phase "$PHASE_OVERRIDE")
fi
if ! CONTEXT_SHELL="$("$PYTHON" tools/operator_context.py "${CONTEXT_ARGS[@]}" --format shell)"; then
    echo "FAIL: unable to resolve operator context with $PYTHON"
    exit 1
fi
eval "$CONTEXT_SHELL"

PHASE_QUESTIONS_PATH="$QUESTIONS_PATH"
PHASE_DOCUMENTS_DIR="$DOCUMENTS_DIR"
PHASE_OUTPUT_DIR="$SUBMISSION_DIR"

QUESTIONS_PATH="${CUSTOM_QUESTIONS_PATH:-$PHASE_QUESTIONS_PATH}"
DOCUMENTS_DIR="${CUSTOM_DOCUMENTS_DIR:-$PHASE_DOCUMENTS_DIR}"
OUTPUT_DIR="${CUSTOM_OUTPUT_DIR:-$PHASE_OUTPUT_DIR}"

echo "=== Build Submission ==="
echo "  Config: $CONFIG"
echo "  Env file: $ENV_FILE"
echo "  Phase: $PHASE"
echo "  Questions: $QUESTIONS_PATH"
echo "  Documents: $DOCUMENTS_DIR"
echo "  Output: $OUTPUT_DIR"
if [ "$RESUME_FROM_CHECKPOINT" -eq 1 ]; then
    echo "  Resume from checkpoint: true"
    if [ -n "$CHECKPOINT_PATH" ]; then
        echo "  Checkpoint path: $CHECKPOINT_PATH"
    fi
    echo "  Checkpoint every: $CHECKPOINT_EVERY"
fi

if [ ! -f "$QUESTIONS_PATH" ]; then
    echo "FAIL: questions file missing: $QUESTIONS_PATH"
    exit 1
fi
if [ ! -d "$DOCUMENTS_DIR" ]; then
    echo "FAIL: documents directory missing: $DOCUMENTS_DIR"
    exit 1
fi
if [ "$DOCUMENTS_DIR" != "$PHASE_DOCUMENTS_DIR" ]; then
    echo "FAIL: build-submission expects the phase-local documents directory: $PHASE_DOCUMENTS_DIR"
    exit 1
fi
echo "  Provider key source: ${PROVIDER_API_KEY_SOURCE:-none}"
echo "  Provider required: $QUESTION_SET_REQUIRES_PROVIDER"
if [ "$QUESTION_SET_REQUIRES_PROVIDER" = "1" ] && [ "$PROVIDER_API_KEY_PRESENT" != "1" ]; then
    echo "FAIL: no provider credential found in process env or $ENV_FILE (checked CODEX_OAUTH_TOKEN, OPENAI_API_KEY, CODEX_API_KEY)"
    exit 1
fi

CORPUS_READY=0
INDEX_READY=0
if [ -f "$CORPUS_PATH" ] && [ -f "$PAGE_MAP_PATH" ] && [ -f "$CORPUS_MANIFEST_PATH" ] && [ -d "$DEBUG_DIR" ]; then
    CORPUS_READY=1
fi
if [ -f "$INDEX_MANIFEST_PATH" ] && [ -f "$PAGE_PARENT_MAP_PATH" ] && [ -f "$SPARSE_ENCODER_STATE_PATH" ] && [ -d "$QDRANT_DIR" ]; then
    INDEX_READY=1
fi
echo "  Corpus artifacts ready: $CORPUS_READY"
echo "  Index artifacts ready: $INDEX_READY"

if [ "$PREFLIGHT" -eq 1 ]; then
    echo "=== Preflight OK ==="
    exit 0
fi

BUILD_CMD=(
    "$PYTHON" -m src.main
    --config "$CONFIG"
    --env-file "$ENV_FILE"
    build-submission
    --questions-path "$QUESTIONS_PATH"
    --documents-dir "$DOCUMENTS_DIR"
    --output-dir "$OUTPUT_DIR"
    --checkpoint-every "$CHECKPOINT_EVERY"
)
if [ -n "$PHASE_OVERRIDE" ]; then
    BUILD_CMD+=(--phase "$PHASE_OVERRIDE")
fi
if [ "$RESUME_FROM_CHECKPOINT" -eq 1 ]; then
    BUILD_CMD+=(--resume-from-checkpoint)
fi
if [ -n "$CHECKPOINT_PATH" ]; then
    BUILD_CMD+=(--checkpoint-path "$CHECKPOINT_PATH")
fi
"${BUILD_CMD[@]}"

echo "--- Validating artifacts ---"
if [ -f "$OUTPUT_DIR/submission.json" ]; then
    ANSWER_COUNT=$("$PYTHON" - <<PY
import json
from pathlib import Path

payload = json.loads(Path("$OUTPUT_DIR/submission.json").read_text(encoding="utf-8"))
print(len(payload.get("answers", [])))
PY
)
    echo "  submission.json: $ANSWER_COUNT answers"
else
    echo "  FAIL: submission.json missing"
    exit 1
fi

for artifact_path in "$OUTPUT_DIR/code_archive.zip" "$OUTPUT_DIR/trace_manifest.json"; do
    if [ -f "$artifact_path" ]; then
        SIZE=$(stat --printf="%s" "$artifact_path")
        echo "  $(basename "$artifact_path"): ${SIZE} bytes"
    else
        echo "  FAIL: missing artifact $artifact_path"
        exit 1
    fi
done

if [ -d "$OUTPUT_DIR/traces" ]; then
    TRACE_COUNT=$(find "$OUTPUT_DIR/traces" -maxdepth 1 -type f -name '*.json' | wc -l)
    echo "  traces/: $TRACE_COUNT files"
else
    echo "  FAIL: traces directory missing"
    exit 1
fi

echo "=== Submission OK ==="
