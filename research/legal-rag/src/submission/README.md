# `src/submission/`

Submission sync, build pipeline, packaging, and telemetry helpers.

## File graph

```text
submission/
├── api_client.py
├── packager.py
├── pipeline.py
├── pipeline_utils.py
├── submit_gate.py
├── sync.py
└── telemetry_packer.py
```

## File map

- `sync.py`: Phase-aware download/sync orchestration and local manifest helpers.
- `api_client.py`: Competition API wrapper for question/document sync and submission status.
- `pipeline.py`: End-to-end build-submission orchestration.
- `pipeline_utils.py`: Shared pipeline helper utilities (artifact checks, token/time estimation, tracing, runtime guards).
- `submit_gate.py`: Gate checks for archive size and pre-submit validation.
- `packager.py`: Submission payload and code archive building.
- `telemetry_packer.py`: Answer-level telemetry serialization into submission payload shape.
