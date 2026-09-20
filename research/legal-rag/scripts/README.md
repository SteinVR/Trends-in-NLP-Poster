# `scripts/`

Operator scripts for deterministic setup, dataset/model provisioning, public quality checks, and pipeline execution.

## Setup Order

```bash
./scripts/01_install_deps.sh
./scripts/02a_sync_datasets.sh --phase warmup
./scripts/02_download_models.sh
./scripts/00_check_env.sh
./scripts/01b_install_opendataloader_runtime.sh --verify-only
```

For final-phase inputs, run:

```bash
./scripts/02a_sync_datasets.sh --phase final
```

`02a_sync_datasets.sh` delegates to the maintained CLI path:
`agentic-rag sync-phase --phase <warmup|final> --extract`.

For normal structural ingestion with Docling hybrid enrichment, start the backend before `parse-corpus`:

```bash
.venv/bin/opendataloader-pdf-hybrid \
  --host 127.0.0.1 \
  --port 5002 \
  --device cpu \
  --force-ocr \
  --log-level info
```

The baseline config expects `opendataloader_hybrid_url: http://localhost:5002`.

## File Map

- `00_check_env.sh`: Preflight for `.venv`, `.env`, provider keys, GPU/CUDA visibility, model caches, OpenDataLoader/Docling runtime, and dataset presence.
- `00_repair_gpu_runtime.sh`: Recreate missing NVIDIA device nodes with `nvidia-modprobe`, then verify CUDA from Python.
- `01_install_deps.sh`: Deterministic dependency setup from `.python-version` and `uv.lock`, followed by runtime import/CLI verification.
- `01b_install_opendataloader_runtime.sh`: OpenDataLoader/Docling runtime verifier and repair helper.
- `02_download_models.sh`: Download or verify Qwen embedder, Qwen reranker, and PaddleOCR model caches.
- `02a_sync_datasets.sh`: Download/extract phase-local competition questions and documents through `agentic-rag sync-phase`.
- `03_lint_and_test.sh`: Public gate: shell syntax, ruff, compileall, and pytest when `tests/` exists.
- `04_parse_corpus.sh`: Wrapper for `parse-corpus` with config-aware phase paths and artifact checks.
- `05_build_indices.sh`: Wrapper for `build-indices` with corpus prerequisite and index artifact checks.
- `06_build_submission.sh`: Wrapper for `build-submission`, provider preflight, checkpointing, and submission artifact checks.
- `07_validate_submission.sh`: Wrapper for `validate-submission` against phase-local questions/documents or explicit paths.

## Pipeline Wrappers

```bash
./scripts/04_parse_corpus.sh --phase warmup --force
./scripts/05_build_indices.sh --phase warmup --force
./scripts/06_build_submission.sh --phase warmup --checkpoint-every 1
./scripts/07_validate_submission.sh --phase warmup
```

The wrapper paths are resolved through `tools/operator_context.py`, so they follow the active config and phase override.

## Demo Run Bundle

The public snapshot keeps one warmup demonstration run:

```text
artifacts/warmup_runs/
├── configs/solver_narrowing_true.yaml
├── summary_runs.tsv
├── top_failures.tsv
└── runs/submission_e2e_20260424_solver_narrowing_no_guard/
    ├── submission/
    ├── eval/
    ├── grounding/
    └── manifest.json
```

Generated runtime logs, corpora, indices, downloaded final data, and new submissions are intentionally ignored by Git.
