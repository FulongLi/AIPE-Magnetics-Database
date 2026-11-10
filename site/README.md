# AIPE Magnetics Data Explorer (GitHub Pages)

Static, read-only frontend for <https://fulongli.github.io/AIPE-Magnetics-Database/>. Plain HTML, CSS and JavaScript:
no framework, no build step, no third-party scripts. `assets/plot.js` is a small SVG plotter; `assets/app.js` holds
the five sections (overview, materials, measurements, comparison, core catalogue) with hash routing, so the site works under the
`/AIPE-Magnetics-Database/` sub-path.

The look follows [aipel.co.uk](https://aipel.co.uk/): its colour tokens, the AIPE logo and favicon
(`assets/aipe-logo.svg`, `assets/favicon.svg`, copied from aipel.co.uk), and Bricolage Grotesque / Inter loaded from
Google Fonts as on aipel.co.uk (system fonts are the fallback). The chart colours (`--series-1` to `--series-8` in
`assets/app.css`) are an AIPE-toned categorical palette checked for colour-vision-deficiency separation; keep their
order when editing. Table headers, filter labels and general badges are never upper-cased, because they carry symbols,
units and grades (μi, mT, kHz, 3C90).

The data are generated, never edited by hand. `tools/build_site.py` reads the committed store through
`aipe_magnetics_database.store.MagneticDatabase` (the same store as the API, CLI and MCP server) and writes `_site/`:

| File | Content |
| --- | --- |
| `data/summary.json`, `data/sources.json` | counts, build commit, source inventory with licences |
| `data/materials.json` | compact index for client-side search; missing values are absent, never zero |
| `data/materials/<id>.json` | canonical record, every member record, measurement list, fitted loss models (loaded on demand) |
| `data/measurements.json` | index of all measurement tables with condition ranges, data class and status |
| `data/measurements/<id>.json` | record metadata, columns, rows (loaded on demand) |
| `data/cores/shapes.json`, `data/cores/parts.json` | MAS shapes (dimension ranges) and manufacturer parts |

Tables up to 5,000 rows are published in full. Larger tables are published as a deterministic sample (every k-th row
within each waveform/temperature group, the row budget shared equally between the groups so that small groups such as
sinusoidal excitation stay plottable; values unchanged), labelled as such on the page and in downloaded file names,
with a link to the full JSONL at the build commit. MAS B-H loops whose sidecars are not committed (`tables-local/`)
are listed without rows. Rows are only published for sources whose dataset record permits redistribution.

Charts label axes with the quantity and a display unit (°C, mT, kHz); stored values stay SI and the chart converts only
for display. A column with more than eight values colours rows by value band (five bands, one hue from light to dark,
`--seq-1` to `--seq-5`); rows are never averaged. Material pages draw the best-ranked published table of each quantity
(measured before datasheet) at one waveform and temperature. The comparison (`#/compare`) overlays up to six materials
by selecting rows at a chosen frequency (±10 %) and temperature (nearest within ±10 °C); it never interpolates and
lists the table and data class behind every curve.

The dark theme follows the system setting (`prefers-color-scheme`, or `data-theme="dark"` / `"light"` on `<html>`). It
redefines the same colour roles; its chart colours are the same hues stepped for the dark card surface and were
validated against it. The material map (overview and Materials → Map) plots Bsat against μi by class; clicking a point
opens the material.

Local preview:

```sh
python tools/build_site.py --out _site/AIPE-Magnetics-Database
python -m http.server 8000 --directory _site   # open http://localhost:8000/AIPE-Magnetics-Database/
```

Deployment: `.github/workflows/pages.yml` builds and deploys on pushes to `main` that touch `site/`, the builder, the
data or the package, and on manual dispatch. It runs no importer and downloads no source data. One-time setting:
repository Settings → Pages → Build and deployment → Source: **GitHub Actions**.
