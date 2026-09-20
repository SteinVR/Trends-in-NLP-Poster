# Legal RAG research poster

English, white/blue, A1 portrait. Author intentionally remains **[AUTHOR NAME]**.

- `index.html`: editable master; opens locally, fits the browser width, with a 100% viewing toggle.
- `poster.pdf`: single-page print export, vector logo, diagrams and charts.
- `poster.png`: full poster preview.
- `assets/`: local logo, fonts and figures; no CDN or network connection required to display the poster.
- `build-figures.py`: reads `../results/metrics.csv` and renders the two SVG figures.
- `export.mjs`: Chromium export and basic layout checks; `layout-check.json` records the latest result.

## Evidence and design

The narrative is based on `../research-paper.md`. R0–R6 are sequential additions on the mixed corpus, not independent leave-one-out ablations. Q is an aggregate answer score, not the percentage of fully correct answers. Citation metrics are macro-averaged over 95 questions. H5 concerns the structured-answer score. The OCR comparison uses original R0, mixed R0 and mixed R1. Rounded labels use one decimal place; deltas are calculated from unrounded values.

R4–R6 reuse the same generated response. The references and scope note are retained; the poster does not claim statistical significance. The main text and references are 24 pt or larger. The charts use vector paths, so their lettering does not depend on fonts installed on the viewer's machine.

Official logo downloaded unchanged on 2026-09-20 from:
https://www.uni-trier.de/typo3conf/ext/zimktheme_unitrier/Resources/Public/Logos/Logo_Universitaet.svg

Logo source page: https://www.uni-trier.de/

The palette uses blue and white, with the logo's blue `#007ac3` as an accent. This is a custom academic layout, not an official university template. Noto Sans regular/bold are bundled from the local font installation.

## Rebuild from repository root

```bash
research/legal-rag/.venv/bin/python output/poster-narrative-3/poster/build-figures.py
NODE_PATH=/path/to/node_modules node output/poster-narrative-3/poster/export.mjs
```

Requires Python with matplotlib, Node with Playwright, and Chromium (default `/usr/bin/chromium`; override with `CHROMIUM_PATH`). Figures regenerate from measured results; this does not rerun the RAG experiments.

## Before academic submission

Replace the author placeholder. The poster links to this repository, the companion paper, measured results, and the fixed experimental protocol. No appendix or submission archive is included.
