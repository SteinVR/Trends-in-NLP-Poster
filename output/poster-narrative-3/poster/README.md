# Legal RAG poster

- `poster.pdf` — current poster, one landscape DIN A1 page (841 × 594 mm), for printing.
- `index.html` — editable HTML source.
- `poster.png` — preview for viewing.
- `assets/` — fonts, licenses and university logo.
- `../powerpoint/` — editable PowerPoint and its own preview.

The charts are embedded as vectors in the HTML. Experiment data remain in `../results/metrics.csv`. The standalone figures in `../results/` can be regenerated with `node build-figures.mjs`; this does not update the embedded HTML charts.

To export the HTML with Node.js, Playwright and Chromium, run from this directory:

```sh
NODE_PATH=<directory-containing-playwright> node export.mjs
```

Set `CHROMIUM_PATH` to override `/usr/bin/chromium`. The export checks layout and writes diagnostics to `tmp/poster-narrative-3/` at repository root. PowerPoint is maintained separately.
