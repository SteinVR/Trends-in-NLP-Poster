# `src/indexing/`

Hybrid index build pipeline split into modular components.

## File graph

```text
indexing/
├── chunking_core.py
├── chunking_text.py
├── embedders.py
├── service.py
├── state.py
└── text_features.py
```

## File map

- `service.py`: High-level `HybridIndexService` orchestration (artifact validation, index build flow, qdrant persistence).
- `embedders.py`: Dense embedder wrappers and BM25 sparse encoder contracts.
- `chunking_core.py`: Structure-aware chunk dataclasses and chunk-family builders.
- `chunking_text.py`: Heading/paragraph/clause text utilities used by chunk builders.
- `text_features.py`: Shared lexical feature helpers (`tokenize`, entity/date extraction, ordered de-duplication).
- `state.py`: Validation helpers for persisted page-parent and sparse-encoder sidecars.
