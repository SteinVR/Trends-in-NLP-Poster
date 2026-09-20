# `src/answering/`

Answer routing, deterministic extraction, grounding validation, and decomposition.

## File graph

```text
answering/
├── decomposition.py
├── page_attribution.py
├── service.py
├── service_helpers.py
├── service_helpers_extractors.py
├── service_helpers_text.py
└── service_types.py
```

## File map

- `decomposition.py`: Multi-case decomposition/recomposition helpers for comparative legal questions.
- `page_attribution.py`: Two-pass page attribution pipeline and trace model.
- `service.py`: `AnsweringService` orchestration (retrieval payload normalization, typed/free-text routing, final page emission).
- `service_types.py`: Shared answering contracts, schemas, constants, metadata extraction, and protocol/dataclass types.
- `service_helpers.py`: Core orchestration helpers (payload coercion, pass-A filtering, confidence/guardrail helpers).
- `service_helpers_extractors.py`: Deterministic extractors (`number`, `boolean`, `name`, `names`, `date`) and grounding checks.
- `service_helpers_text.py`: Text normalization/scoring/entity matching helpers reused by extractors.
