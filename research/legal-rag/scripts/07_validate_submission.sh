#!/usr/bin/env bash
# 07_validate_submission.sh — Validate a submission.json against local rules.
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
CUSTOM_SUBMISSION_PATH=""
CUSTOM_QUESTIONS_PATH=""
CUSTOM_DOCUMENTS_DIR=""
OUTPUT_PATH=""
POSITIONAL=()

resolve_default_env_file() {
    if [ "$ENV_FILE" = ".env" ] && [ ! -f "$PROJECT_ROOT/.env" ] && [ -f "$COMMON_ROOT/.env" ]; then
        ENV_FILE="$COMMON_ROOT/.env"
    fi
}

usage() {
    cat <<'EOF'
Usage:
  ./scripts/07_validate_submission.sh [CONFIG] [PHASE] [SUBMISSION_PATH]
  ./scripts/07_validate_submission.sh --config PATH --env-file PATH [--phase PHASE] [--submission-path PATH] [--questions-path PATH] [--documents-dir PATH] [--output-path PATH]
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
        --submission-path)
            CUSTOM_SUBMISSION_PATH="$2"
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
        --output-path)
            OUTPUT_PATH="$2"
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
if [ "${#POSITIONAL[@]}" -ge 3 ] && [ -z "$CUSTOM_SUBMISSION_PATH" ]; then
    CUSTOM_SUBMISSION_PATH="${POSITIONAL[2]}"
fi
resolve_default_env_file

CONTEXT_ARGS=(--config "$CONFIG" --env-file "$ENV_FILE")
if [ -n "$PHASE_OVERRIDE" ]; then
    CONTEXT_ARGS+=(--phase "$PHASE_OVERRIDE")
fi
eval "$("$PYTHON" tools/operator_context.py "${CONTEXT_ARGS[@]}" --format shell)"

PHASE_SUBMISSION_PATH="$SUBMISSION_PATH"
PHASE_QUESTIONS_PATH="$QUESTIONS_PATH"
PHASE_DOCUMENTS_DIR="$DOCUMENTS_DIR"

SUBMISSION_PATH="${CUSTOM_SUBMISSION_PATH:-$PHASE_SUBMISSION_PATH}"
QUESTIONS_PATH="${CUSTOM_QUESTIONS_PATH:-$PHASE_QUESTIONS_PATH}"
DOCUMENTS_DIR="${CUSTOM_DOCUMENTS_DIR:-$PHASE_DOCUMENTS_DIR}"
AUTO_DISCOVERED_SUBMISSION_PATH=""

if [ -z "$CUSTOM_SUBMISSION_PATH" ] && [ ! -f "$SUBMISSION_PATH" ]; then
    AUTO_DISCOVERED_SUBMISSION_PATH="$(
        find "data/$PHASE" -maxdepth 2 -type f -path "*/submission*/submission.json" -printf '%T@ %p\n' \
            | sort -n \
            | tail -n 1 \
            | cut -d' ' -f2-
    )"
    if [ -n "$AUTO_DISCOVERED_SUBMISSION_PATH" ]; then
        SUBMISSION_PATH="$AUTO_DISCOVERED_SUBMISSION_PATH"
    fi
fi

echo "=== Validate Submission ==="
echo "  Config: $CONFIG"
echo "  Env file: $ENV_FILE"
echo "  Phase: $PHASE"
echo "  Submission: $SUBMISSION_PATH"
if [ -n "$AUTO_DISCOVERED_SUBMISSION_PATH" ]; then
    echo "  Submission fallback: auto-discovered latest submission artifact"
fi
echo "  Questions: $QUESTIONS_PATH"
echo "  Documents: $DOCUMENTS_DIR"
if [ -n "$OUTPUT_PATH" ]; then
    echo "  Report output: $OUTPUT_PATH"
fi

for required_file in "$SUBMISSION_PATH" "$QUESTIONS_PATH"; do
    if [ ! -f "$required_file" ]; then
        echo "FAIL: missing file $required_file"
        exit 1
    fi
done
if [ ! -d "$DOCUMENTS_DIR" ]; then
    echo "FAIL: documents directory missing: $DOCUMENTS_DIR"
    exit 1
fi

COMMAND=(
    "$PYTHON" -m src.main
    --config "$CONFIG"
    --env-file "$ENV_FILE"
    validate-submission
    --submission-path "$SUBMISSION_PATH"
    --questions-path "$QUESTIONS_PATH"
    --documents-dir "$DOCUMENTS_DIR"
)
if [ -n "$OUTPUT_PATH" ]; then
    COMMAND+=(--output-path "$OUTPUT_PATH")
fi

"${COMMAND[@]}"

echo "=== Validation OK ==="
