# `src/providers/`

Provider protocols and Codex-backed implementations live here. Package initializers and cache directories are intentionally omitted from the map.

## File graph

```text
providers/
├── base.py
└── codex_provider.py
```

## File map

- `base.py`: Provider protocol contracts for answer generation. It defines the free-text and structured-output interfaces that downstream code depends on.
- `codex_provider.py`: Concrete Codex-backed provider implementation. It covers HTTP client setup, prompt building, strict schema shaping, response parsing, bounded retries, and confidence normalization for both free-text and structured generation.
