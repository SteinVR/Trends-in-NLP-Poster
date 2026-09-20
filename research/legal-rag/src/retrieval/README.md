# `src/retrieval/`

Hybrid retrieval, reranking, and evidence/page grounding.

## File graph

```text
retrieval/
├── evidence_compressor.py
├── hybrid_search.py
├── introspection.py
├── page_lifter.py
├── reranker.py
└── service.py
```

## File map

- `service.py`: Retrieval orchestration across backend search, reranking, compression, and grounded payload shaping.
- `hybrid_search.py`: Qdrant dense+sparse backend and RRF candidate fusion with metadata boosts.
- `reranker.py`: Qwen reranker and lexical baseline implementation (strict path now raises on unavailable preferred reranker).
- `evidence_compressor.py`: Evidence compression and support-page filtering.
- `page_lifter.py`: Chunk-to-page lifting helpers used in grounding.
- `introspection.py`: Shared callable/metadata introspection helpers reused across retrieval modules.
