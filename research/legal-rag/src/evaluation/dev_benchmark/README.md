# `src/evaluation/dev_benchmark/`

Dev benchmark reporting models and runtime analysis helpers.

## File graph

```text
dev_benchmark/
├── __init__.py
├── models.py
└── runtime.py
```

## File map

- `models.py`: Benchmark pydantic models (metadata, slice reports, run reports, consistency/comparison reports).
- `runtime.py`: Benchmark build/load helpers, run scoring flow, report comparison, and consistency analytics.
- `__init__.py`: Public export surface for benchmark models and runtime helpers.
