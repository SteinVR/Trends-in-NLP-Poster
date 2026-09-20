# `src/evaluation/`

Local validation, scoring, goldset tooling, and benchmark analysis.

## File graph

```text
evaluation/
├── contracts.py
├── grounding_ledger.py
├── goldset.py
├── goldset_core.py
├── goldset_validation.py
├── scorer.py
├── validator.py
├── validator_checks.py
└── dev_benchmark/
```

## File map

- `contracts.py`: Evaluation contracts and typed loaders for submissions/questions/references/assistant scores.
- `grounding_ledger.py`: Recall-first per-question grounding ledger builders and run-level bucket summaries.
- `goldset.py`: Public export wrapper for goldset helpers.
- `goldset_core.py`: Goldset sharding/build pipeline.
- `goldset_validation.py`: Goldset input and review validation helpers.
- `scorer.py`: Local scoring engine and metric aggregation.
- `validator.py`: Submission validator orchestration and report construction.
- `validator_checks.py`: Focused answer/retrieval validation checks extracted from validator core.
- `dev_benchmark/`: Benchmark dataset models, runtime report builders, and consistency/comparison tooling.
