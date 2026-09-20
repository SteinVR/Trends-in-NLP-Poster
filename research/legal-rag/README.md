# Legal RAG research implementation

This directory contains the executable legal RAG pipeline and the sequential component experiment described in the [repository README](../../README.md).

The pipeline under `src/` implements page-aware ingestion, OCR, multi-granularity indexing, hybrid retrieval, reranking, typed answering, page attribution, and trace generation. The controlled experiment under `experiments/` reuses those components while fixing the corpus, question set, model configuration, candidate budgets, and evaluation protocol across R0-R6.

## Main components

- `src/ingestion/`: native PDF extraction, OCR routing, structure recovery, tables, and page provenance.
- `src/indexing/`: page, section, clause, microchunk, and table representations; dense and sparse features.
- `src/retrieval/`: candidate fusion, Qwen reranking, evidence compression, and page lifting.
- `src/answering/`: typed answer generation, normalization, validation, and page attribution.
- `src/evaluation/`: deterministic scoring, gold-page evaluation, and validation contracts.
- `experiments/`: controlled R0-R6 comparisons, frozen protocol, verification, scoring, and report generation.

See [`experiments/README.md`](experiments/README.md) for the exact run order and artifact definitions.
