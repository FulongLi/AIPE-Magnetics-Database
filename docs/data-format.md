# Data format

## Layout

```
data/
  sources.json                 inventory of every investigated source (generated from ingest/registry.py)
  raw/<source>/SOURCES.json    pinned inputs: URL, version, SHA-256, licence, raw policy
  raw/<source>/files/          original files where redistribution is permitted and they are small (Zenodo, Paderborn)
  normalized/<source>/         schema v0.2 records of one source
      index.json               shards, datasets, tables, record counts, table-row counts
      materials/ shapes/ cores/ loss_models/ measurements/   JSONL shards (one record per line)
      datasets/*.json          dataset records (one per upstream file collection)
      tables/*.jsonl           committed table sidecars (one JSON array per row)
      tables-local/*.jsonl     large regenerated sidecars (git-ignored; MAS B-H loops)
  canonical/                   consolidated layer (materials.jsonl, measurements.jsonl, duplicates.json, index.json)
  derived/                     explicitly derived or fitted data (e.g. Steinmetz fits to MagNet, bundled in the package)
  reports/                     import-<source>.json and coverage.json (generated)
```

Large numerical waveform arrays may still use the v0.2 HDF5 layout (`aipe.magnetics.waveforms/0.2`,
`waveform_dataset` records) through the source adapters; every such file has a JSON record with its SHA-256.

## Records

All records follow the AIPE Magnetic Data Schema v0.2 ([data-standard-v0.2.md](data-standard-v0.2.md); JSON Schema
in `schemas/v0.2/`): envelope (`id`, `kind`, `name`, `license`, `synthetic`), typed `properties`, `evidence` with
`review_status`, `provenance`, `quality.flags`, `artifacts` with SHA-256, namespaced `extensions`.

| kind | used for |
| --- | --- |
| `magnetic_material` | material identity and scalar properties (initial permeability, Bsat(T, H), Br, Hc, Curie, density, resistivity, permeability vs T) |
| `core_geometry` | MAS standard shapes (`extensions.aipe.mas.dimensions_range_m`: min/max in m, nominal values separately) and manufacturer parts |
| `measurement` | one table of measured or datasheet data (`extensions.aipe.table`) |
| `model_metadata` | fitted loss-model coefficients (`data_class: fitted_parameters`), exactly as published |
| `dataset` | upstream collection: URL, version, licence, redistribution flag, checksums, citation |

## `aipe.magnetics.table/0.1`

```json
"aipe.table": {"layout": "aipe.magnetics.table/0.1", "quantity": "core_loss",
  "columns": [{"name": "frequency", "unit": "Hz", "definition": "fundamental frequency: Welch PSD estimate, 10 Hz resolution"},
              {"name": "flux_density_peak", "unit": "T", "definition": "AC amplitude (max B - min B)/2"},
              {"name": "temperature", "unit": "K", "original_unit": "degC", "definition": "oil-bath set point"}, ...],
  "row_count": 142871, "rows_file": "tables/princeton-magnet-n87-core-loss.jsonl", "rows_sha256": "..."}
```

Column names with a fixed SI unit (enforced): `frequency` Hz, `temperature` K, `flux_density_peak` / `flux_density` T, `field` / `dc_bias_field` A/m, `loss_density` W/m^3, `mass_loss_density` W/kg,
`energy_density_per_cycle` J/m^3, `mu_real` / `mu_imag` / `eps_real` / `eps_imag` / `relative_permeability` /
`amplitude_permeability` / `loss_factor` / `duty*` dimensionless. Text columns (`waveform`, `specimen`, `branch`) have
an empty unit. `null` = not reported.

Quantity conventions: `flux_density_peak` is always the **amplitude** (peak, (max-min)/2), never peak-to-peak or RMS;
the sign convention of B and H is "positive along the same winding reference direction"; complex permeability is
`mu = mu' - j mu''` with `mu'' > 0` for loss (negative upstream values are kept and flagged); loss published as energy
per cycle stays `energy_density_per_cycle` (P_v = E * f is not precomputed).

## `aipe.data` (every record)

`data_class` (`experimental_measurement`, `manufacturer_datasheet`, `manufacturer_tool_export`, `derived_from_measurement`,
`fitted_parameters`, `aggregated_literature`, `identity`), `source_id`, and for measurements `underlying_source`
(the original experiment, e.g. `princeton-magnet:n87`) used for deduplication.

## Canonical layer

`canonical/materials.jsonl`: one `magnetic_material` per normalised `(manufacturer, grade)`, id
`material.canonical-<manufacturer>-<grade>`, properties copied unchanged from one member (precedence: curated > MAS >
Paderborn > others), `extensions.aipe.canonical.members` lists every source record. `canonical/measurements.jsonl`:
one index entry per measurement table with condition ranges and `status` (`primary`, `duplicate` + `duplicate_of` +
row-match statistic, `derived_view`). Nothing is deleted from `normalized/`.
