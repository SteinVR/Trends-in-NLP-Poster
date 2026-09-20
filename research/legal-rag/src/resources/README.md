# `src/resources/`

Non-executable runtime assets.

## File graph

```text
resources/
├── configs/
│   └── baseline/
│       └── v001/
│           └── config.yaml
└── prompts/
    └── answering/
        └── system.md
```

## Notes

- `configs/` contains versioned configuration bundles used by `load_app_config`.
- `prompts/` contains provider prompt templates loaded as packaged resources.
- `configs/baseline/v001/config.yaml` is the packaged default baseline profile. In a repo checkout, `configs/baseline/v001/config.yaml` is the primary config and this resource copy is used only when the repo-root config is unavailable.
