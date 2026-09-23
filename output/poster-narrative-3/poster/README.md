# Актуальный постер

Утверждённый макет: [index.html](index.html). Готовые экспорты: [poster.pdf](poster.pdf) и [poster.png](poster.png). Один лист DIN A1 (594 × 841 мм), без имён авторов. PDF сохраняет векторную графику; PNG предназначен для просмотра.

Графики читаются из `../results/`, шрифты и логотип — из `assets/`. Редактируемая PowerPoint-версия и её собственный PNG находятся в `../powerpoint/`.

Для пересборки из корня репозитория нужны Node.js, Playwright и Chromium:

```sh
node output/poster-narrative-3/poster/build-figures.mjs
NODE_PATH=<directory-containing-playwright> node output/poster-narrative-3/poster/export.mjs
```

`CHROMIUM_PATH` позволяет указать браузер вместо `/usr/bin/chromium`. Экспорт обновляет PDF/PNG постера и обоих графиков; диагностика сохраняется в `tmp/poster-narrative-3/`. PowerPoint автоматически не пересобирается.

Для просмотра через HTTP запускайте сервер из `output/poster-narrative-3/` и открывайте `/poster/index.html`, чтобы были доступны соседние графики.
