# AIPE Magnetics Database

A standardised, provenance-tracked database of soft-magnetic data for power electronics: material properties,
core geometries, and measured core loss, complex permeability, permittivity and B-H data. Records are collected from
open sources (Princeton MagNet, Paderborn LEA, OpenMagnetics MAS, HEFMAG, the Nabih MHz core-loss database, Dartmouth
N87 dielectric data, INRIM, Zenodo and a curated datasheet catalogue), converted to SI units, and deduplicated across
sources. Every value keeps its source, version, licence and data class. The current release holds 1,164 materials,
21,237 core shapes and parts, and 1,764 measurement tables with 2.35 million rows, of which 621,066 are independent
experimental measurements.

Original files, the normalised records and large tables are published on Hugging Face as
[**FulongLi/AIPE-Magnetics-Dataset**](https://huggingface.co/datasets/FulongLi/AIPE-Magnetics-Dataset); this repository
builds, validates and publishes that dataset and reads it back at a pinned commit ([docs/huggingface.md](docs/huggingface.md)).

This repository serves data only. Loss models, fitting and component design live in
[AIPE-Magnetics-Design](https://github.com/FulongLi/AIPE-Magnetics-Design).

## Explore Online

**[fulongli.github.io/AIPE-Magnetics-Database](https://fulongli.github.io/AIPE-Magnetics-Database/)**

The AIPE Magnetics Data Explorer is a read-only website for people. Search materials by name, grade, manufacturer or
class; inspect their properties with conditions and evidence; plot core-loss, permeability and B-H tables; browse core
shapes and manufacturer parts; and see the source, licence and data class of every value. It is rebuilt from this
repository's data on every change to `main` ([`tools/build_site.py`](tools/build_site.py), [`site/`](site/README.md)).

## Use with AI Agents

Claude Code and Codex use the same local, read-only MCP server. Install the package first, into the Python that
`python` runs:

```sh
pip install "aipe-magnetics-database[mcp] @ git+https://github.com/FulongLi/AIPE-Magnetics-Database.git"
```

Claude Code:

```sh
claude plugin marketplace add FulongLi/AIPE-Magnetics-Database
claude plugin install aipe-magnetics-database@aipe-magnetics-database
```

Codex:

```sh
codex plugin marketplace add FulongLi/AIPE-Magnetics-Database
codex plugin add aipe-magnetics-database@aipe-magnetics-database
```

The plugins add a data-lookup skill and start `python -m aipe_magnetics_database.mcp.server`. Other MCP clients can
run `aipe-magnetics-db-mcp` (stdio). The tools (`find_measurements`, `get_table`, `material_properties`, `provenance`
and others) return records with units and provenance and never write files. Try: *"Which measured core-loss tables
cover TDK N87 at 100 kHz and 90 °C?"*

## Python / CLI Quick Start

The wheel bundles the committed data, so queries work offline after installation (command above).

```sh
aipe-magnetics-db materials list --query N87          # search
aipe-magnetics-db materials properties "TDK N87"      # values from every source, plus fitted loss models
aipe-magnetics-db measurements find --material "TDK N87" --quantity core_loss -f "100 kHz" -t "90 degC"
aipe-magnetics-db measurements table measurement.princeton-magnet-n87-core-loss -f "99 kHz..101 kHz" -t "90 degC" --waveform sinusoidal
```

```python
import aipe_magnetics_database.api as db

n87 = db.get_material("TDK N87")
hits = db.find_measurements(material="TDK N87", quantity="core_loss", data_class="experimental_measurement",
                            frequency="100 kHz", temperature="90 degC")
table = db.get_table(hits[0]["id"], frequency="95 kHz..105 kHz", temperature="90 degC", waveform="sinusoidal")
print(n87["properties"]["initial_permeability"], table["rows"][0], db.provenance(hits[0]["id"])["license"])
```

Physical conditions always carry units (`"100 kHz"`, `"90 degC"`, `"50 kHz..200 kHz"`); bare numbers are rejected.

## Data and Documentation

Records are JSON/JSONL in AIPE Magnetic Data Schema v0.2: one record per line, sharded by source under
`data/normalized/<source>/`, consolidated in `data/canonical/`. Values are SI with the published unit kept as
`original`; a missing value is omitted or `null`, never zero. Every record carries its evidence (URL, commit or DOI,
locator), licence, quality flags and `data_class`: experimental measurements, manufacturer datasheet curves, derived
tables and fitted parameters are never mixed. A shortened, schema-valid excerpt of a committed record:

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

Each source keeps its own licence (Apache-2.0, MIT, GPL-3.0-only, CC-BY-4.0); nothing is relicensed. Read the
provenance notes before relying on any value.

| Topic | Document |
| --- | --- |
| Python API, CLI and MCP tools | [docs/api.md](docs/api.md) |
| Data format and table layout | [docs/data-format.md](docs/data-format.md), [docs/data-standard-v0.2.md](docs/data-standard-v0.2.md) |
| Sources investigated and integrated | [docs/source-catalogue.md](docs/source-catalogue.md), [docs/discovery-2026-11.md](docs/discovery-2026-11.md) |
| Hugging Face dataset, publishing and pinned downloads | [docs/huggingface.md](docs/huggingface.md) |
| Automated quality checks | [docs/quality-summary.md](docs/quality-summary.md) |
| Provenance and limitations | [docs/provenance-and-limitations.md](docs/provenance-and-limitations.md) |
| Licensing and data rights | [LICENSING.md](LICENSING.md) |
| Re-running the importers | [docs/ingestion.md](docs/ingestion.md) |

## Project Information

- Version 0.3.0 (`aipe-magnetics-database`), schema 0.2.0. Status: prototype; third-party values are unreviewed, and
  schema validation is not an experimental verification.
- Developer: Fulong Li, Spirit Connect / AIPE Labs.
- Licence: no licence has been declared for the code yet (`NOASSERTION`). Data keep their source licences
  ([LICENSING.md](LICENSING.md)).
- Changes: [CHANGELOG.md](CHANGELOG.md). Migrating from 0.2: [docs/migration-v0.3.md](docs/migration-v0.3.md).
