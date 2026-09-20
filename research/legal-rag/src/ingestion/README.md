# `src/ingestion/`

PDF parsing, OCR, structural adaptation, and corpus build orchestration.

## File graph

```text
ingestion/
├── corpus_builder.py
├── ocr_engine.py
├── opendataloader_adapter.py
├── page_triage.py
├── parser_routing.py
├── pdf_parser.py
├── pdf_structural.py
├── service.py
├── sidecars.py
├── structural_normalizer.py
└── table_serializer.py
```

## File map

- `service.py`: Phase-aware parse orchestration and canonical artifact writing.
- `pdf_parser.py`: Native page extraction, table candidates, parser-route execution integration.
- `pdf_structural.py`: Structural sidecar route resolution and strict sidecar build flow.
- `opendataloader_adapter.py`: Opendataloader invocation and sidecar adaptation to internal schema.
- `ocr_engine.py`: OCR provider construction and OCR-derived text/table extraction.
- `table_serializer.py`: Table normalization, merge logic, and markdown/html serialization.
- `structural_normalizer.py`: Conversion from raw structural sidecars to normalized blocks.
- `corpus_builder.py`: Canonical corpus/page-map assembly from parsed documents and normalized blocks.
- `page_triage.py`: Page quality/triage signals used by parser routing.
- `parser_routing.py`: Route decision and batching helpers.
- `sidecars.py`: Typed sidecar persistence helpers.
