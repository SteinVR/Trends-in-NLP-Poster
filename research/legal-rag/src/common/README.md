# `src/common/`

Shared cross-stage infrastructure.

## File graph

```text
common/
├── config.py
├── config_overrides.py
├── file_hash.py
├── runtime_logging.py
└── schemas.py
```

## File map

- `config.py`: Main settings models and config materialization entrypoints.
- `config_overrides.py`: Environment override parsing and override application helpers.
- `file_hash.py`: Shared SHA-256 helper used by ingestion/indexing/submission manifests.
- `runtime_logging.py`: Structured runtime logging setup and event helpers.
- `schemas.py`: Shared typed records used across ingestion/indexing/retrieval/evaluation/submission.
