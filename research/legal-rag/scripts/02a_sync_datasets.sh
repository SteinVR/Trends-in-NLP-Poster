#!/usr/bin/env bash
# 02a_sync_datasets.sh - Download competition questions and documents for a local phase.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GIT_COMMON_DIR="$(git -C "$PROJECT_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
COMMON_ROOT="$PROJECT_ROOT"
if [ -n "$GIT_COMMON_DIR" ]; then
    COMMON_ROOT="$(dirname "$GIT_COMMON_DIR")"
fi

CONFIG="configs/baseline/v001/config.yaml"
ENV_FILE=".env"
PHASE="warmup"
FORCE=0
DRY_RUN=0

usage() {
    cat <<'USAGE'
Usage: ./scripts/02a_sync_datasets.sh [options]

Options:
  --phase PHASE      Dataset phase to sync: warmup or final (default: warmup).
  --config PATH      Config path (default: configs/baseline/v001/config.yaml).
  --env-file PATH    Env file with COMPETITION_API_KEY (default: .env).
  --force            Overwrite existing local download artifacts.
  --dry-run          Resolve paths without API calls or credential checks.
  -h, --help         Show this help.

The script calls the maintained CLI flow:
  agentic-rag sync-phase --phase <phase> --extract
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --env-file)
            ENV_FILE="$2"
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

case "$PHASE" in
    warmup|final) ;;
    *)
        echo "FAIL: --phase must be warmup or final." >&2
        exit 1
        ;;
esac

if [ "$DRY_RUN" -eq 1 ]; then
    PHASE_DIR="data/$PHASE"
    cat <<EOF
command: sync-phase
phase: $PHASE
dry_run: true
config: $CONFIG
env_file: $ENV_FILE
planned_paths:
  - $PHASE_DIR/questions/questions.json
  - $PHASE_DIR/documents/documents.zip
  - $PHASE_DIR/documents/pdfs
  - $PHASE_DIR/manifests/sync_manifest.json
EOF
    exit 0
fi

if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif [ -x "$COMMON_ROOT/.venv/bin/python" ]; then
    PYTHON="$COMMON_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    echo "FAIL: python runtime missing; run ./scripts/01_install_deps.sh first." >&2
    exit 1
fi

has_competition_api_key() {
    if [ -n "${COMPETITION_API_KEY:-}" ] || [ -n "${EVAL_API_KEY:-}" ]; then
        return 0
    fi
    if [ -f "$ENV_FILE" ] && awk -F= '
        /^[[:space:]]*(COMPETITION_API_KEY|EVAL_API_KEY)[[:space:]]*=/ {
            value=$0
            sub(/^[^=]*=/, "", value)
            gsub(/^[[:space:]'\''"]+|[[:space:]'\''"]+$/, "", value)
            if (value != "" && value != "set_in_real_env") found=1
        }
        END { exit(found ? 0 : 1) }
    ' "$ENV_FILE"; then
        return 0
    fi
    return 1
}

if ! has_competition_api_key; then
    echo "FAIL: COMPETITION_API_KEY is required in the environment or $ENV_FILE." >&2
    exit 1
fi

cd "$PROJECT_ROOT"
CMD=(
    "$PYTHON" -m src.main
    --config "$CONFIG"
    --env-file "$ENV_FILE"
    sync-phase
    --phase "$PHASE"
    --extract
)
if [ "$FORCE" -eq 1 ]; then
    CMD+=(--force)
fi
PYTHONPATH=. "${CMD[@]}"
