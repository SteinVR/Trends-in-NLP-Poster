# `src/`

Composition root for the CLI and all pipeline stages.

## File graph

```text
src/
├── main.py
├── main_handlers.py
├── main_parser_helpers.py
├── answering/
├── common/
├── evaluation/
├── indexing/
├── ingestion/
├── providers/
├── resources/
├── retrieval/
└── submission/
```

## File map

- `main.py`: CLI entrypoint that builds parser, loads config/env, configures runtime logging, and dispatches commands.
- `main_handlers.py`: Command execution handlers for sync, parsing, indexing, submission, validation, scoring, and benchmark utilities.
- `main_parser_helpers.py`: Modular argparse subcommand registration helpers.
- `answering/`: Typed answering, grounding, page attribution, and decomposition logic.
- `common/`: Shared config, schemas, hashing, and runtime logging utilities.
- `evaluation/`: Validation, scoring, goldset preparation, and dev benchmark tooling.
- `indexing/`: Hybrid index construction, chunking, dense/sparse encoders, and index state validation.
- `ingestion/`: PDF parsing, OCR, routing, structural normalization, and corpus assembly.
- `providers/`: Provider interfaces and Codex provider implementation.
- `resources/`: Non-code runtime assets (configuration bundles).
- `retrieval/`: Hybrid search, reranking, lifting, and retrieval orchestration.
- `submission/`: Sync, package/build pipeline, telemetry packing, and submission gates.
