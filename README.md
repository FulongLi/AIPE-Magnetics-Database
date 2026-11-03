# AIPE Magnetics Database

A standardised, provenance-tracked database of soft-magnetic data for power electronics: material properties,
core geometries, and measured core loss, complex permeability, permittivity and B-H data. Records are collected
from open sources, normalised into AIPE Magnetic Data Schema v0.2, and deduplicated across sources. Each value
keeps its source, version, licence and data class. The repository serves data only. Loss models and component
design are in [AIPE-Magnetics-Design](https://github.com/FulongLi/AIPE-Magnetics-Design).

## Available Data

Counts are verified by `aipe-magnetics-db refresh` (`data/reports/coverage.json`).

| Category | Records | Notes |
| --- | --- | --- |
| Magnetic materials | 1,142 canonical (from 1,178 source records) | 17 grades combine several sources (e.g. TDK N87: curated, MAS, Paderborn, MagNet) |
| Core geometry | 2,294 standard shapes + 18,943 manufacturer parts | Shapes are tolerance ranges only; effective parameters are only present where a source publishes them |
| Measurement tables | 1,715 source tables, 1,701 independent | 3 duplicates (MAS copies of MagNet, 100 % row match) and 11 derived views are not counted as independent |
| Table rows | 2,301,754 (786,563 committed) | 604,680 rows are independent experimental measurements |
| Fitted loss models | 664 | Steinmetz, Magnetics, Micrometals, TDG, POCO and Roshen coefficients, kept separate from measurements |

The 1,515,191 MAS B-H loop points are regenerated locally by `refresh`; they are not committed. Measured data:
Princeton MagNet (575,009 core-loss rows, 10 ferrites), Paderborn LEA test-bench permeability and permittivity,
HEFMAG amorphous and nanocrystalline rings, and SMC/ferrite loss under triangular excitation. Datasheet data:
OpenMagnetics MAS and the Paderborn digitised curves.

## Data Format

- **JSON/JSONL, schema v0.2** ([docs/data-standard-v0.2.md](docs/data-standard-v0.2.md)). One record per line,
  sharded by source under `data/normalized/<source>/`. Consolidated records are in `data/canonical/`.
- **SI units** (`Hz`, `T`, `A/m`, `K`, `W/m^3`). Converted values keep `original: {value, unit}`. Missing values
  are omitted or `null`, never zero.
- **Tables** use the `aipe.magnetics.table/0.1` layout inside `measurement` records. Each column has a unit, an
  original unit and a definition, e.g. *peak AC flux density (max-min)/2*. Large tables are JSONL sidecars
  with SHA-256 ([docs/data-format.md](docs/data-format.md)).
- **Provenance** on every record: evidence (URL, commit or DOI, locator, SHA-256), licence, importer version,
  normalisation, quality flags, and `data_class`. The data classes are `experimental_measurement`,
  `manufacturer_datasheet`, `manufacturer_tool_export`, `derived_from_measurement` and `fitted_parameters`.

```json
{"schema_version":"0.2.0","id":"material.mas-tdk-n87-0015","kind":"magnetic_material","name":"TDK N87","synthetic":false,
 "license":"Apache-2.0","properties":{"grade":"N87","material_class":"ferrite_mnzn",
 "saturation_flux_density":[{"value":{"value":0.38980000000000004,"unit":"T"},"source_ref":"evidence.mas-source",
 "temperature":{"value":373.15,"unit":"K","original":{"value":100,"unit":"degC"}},"field":{"value":1210.0,"unit":"A/m"}}]},
 "evidence_refs":["evidence.mas-source"],"evidence":[{"id":"evidence.mas-source","kind":"source",
 "reference":"Power-Supply-Manufacturers-Association/MAS","description":"Third-party material catalogue, not independently tested by AIPE.",
 "created_at":"2026-11-02T20:52:36Z","review_status":"unreviewed","license":"Apache-2.0","locator":"data/core_materials.ndjson:L15",
 "version":"c33ccfec8e00e156d632c9a52b02bc1cfdf714dc"}],"quality":{"status":"normalized","flags":["third_party_unreviewed"]}}
```

(This excerpt of a committed record is shortened but still schema-valid. `tests/store` checks the full record.)

## Quick Start

```sh
pip install "aipe-magnetics-database[mcp] @ git+https://github.com/FulongLi/AIPE-Magnetics-Database.git"
aipe-magnetics-db materials get "TDK N87"
aipe-magnetics-db measurements find --material "TDK N87" --quantity core_loss -f "100 kHz" -t "90 degC"
aipe-magnetics-db measurements table measurement.princeton-magnet-n87-core-loss -f "99 kHz..101 kHz" -t "90 degC" --waveform sinusoidal
```

```python
import aipe_magnetics_database.api as db

hits = db.find_measurements(material="TDK N87", quantity="core_loss", data_class="experimental_measurement",
                            frequency="100 kHz", temperature="90 degC")
table = db.get_table(hits[0]["id"], frequency="95 kHz..105 kHz", temperature="90 degC", waveform="sinusoidal")
print([c["name"] for c in table["columns"]], table["rows"][0], db.provenance(hits[0]["id"])["license"])
```

Conditions always take units, and bare numbers are rejected. Further entry points: `aipe-magnetics-db-mcp` (a
read-only MCP server), the Claude Code and Codex plugins in `plugins/`, and `aipe-magnetics-db refresh` (re-runs
every pinned importer idempotently). See [docs/api.md](docs/api.md) and [docs/ingestion.md](docs/ingestion.md).

## Data Sources

[docs/source-catalogue.md](docs/source-catalogue.md) lists all 25 investigated sources. Integrated:
OpenMagnetics MAS (Apache-2.0), Princeton MagNet (MIT), Paderborn LEA materialdatabase (GPL-3.0-only, separate
subtree), HEFMAG (CC-BY-4.0) and Zenodo 21626116 (CC-BY-4.0). Reference-only, with user-file adapters where
useful: the PSMA core-loss studies (copyright), the ASU MHz aggregate (no licence), MagNet Challenge 1/2 (no
licence) and 12 manufacturers (no redistribution grant). Read
[docs/provenance-and-limitations.md](docs/provenance-and-limitations.md) before using any value.

## Project Information

- Version 0.3.0 (`aipe-magnetics-database`), schema 0.2.0. Status: prototype. Third-party values are unreviewed,
  and schema validation is not an experimental verification.
- Developer: Fulong Li, Spirit Connect / AIPE Labs.
- Licence: code `NOASSERTION` (not yet declared). Data keep their source licences ([LICENSING.md](LICENSING.md)).
- Changes: [CHANGELOG.md](CHANGELOG.md). Migrating from 0.2: [docs/migration-v0.3.md](docs/migration-v0.3.md).
