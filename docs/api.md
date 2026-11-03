# Python API, CLI and MCP server

## Python (`aipe_magnetics_database.api`)

| Function | Returns |
| --- | --- |
| `summary()` | record counts per source and of the canonical layer |
| `list_sources(status=None)` | source inventory with status, version, licence, reason |
| `list_materials(query, manufacturer, material_class, source, canonical=True)` | compact material list |
| `get_material(query)` | canonical record by id, name (`"TDK N87"`), member id or unique grade |
| `material_properties(query)` | properties of every member record + fitted loss-model coefficients |
| `list_cores(query, family, manufacturer, material, kind="all"|"shapes"|"parts", limit)` / `get_core(query)` | core shapes and parts |
| `find_measurements(material, quantity, data_class, source, frequency, temperature, flux_density, dc_bias, waveform, include_duplicates=False)` | measurement entries whose condition ranges overlap |
| `get_table(record_id, frequency, temperature, flux_density, dc_bias, waveform, limit)` | `{columns, rows, total_rows}` |
| `get_record(id)`, `provenance(id)`, `export(ids, path, include_rows=True)` | v0.2 record, provenance view, JSONL export |

Conditions carry units: `"100 kHz"`, `(100, "kHz")`, `{"value": 0.1, "unit": "T"}`, ranges `("50 kHz", "200 kHz")` or
`"50 kHz..200 kHz"`. `aipe_magnetics_database.store.MagneticDatabase(root)` is the underlying object (set
`AIPE_MAGNETICS_DATA` to use another data directory). The curated engineering catalogue used by AIPE-Magnetics-Design
is served by `aipe_magnetics_database.database` (`material_data`, `core_data`, `search_materials`, ...).

## CLI (`aipe-magnetics-db`)

`info`, `sources`, `materials list|get|properties`, `cores list|get`, `measurements find|table [--csv FILE]`,
`provenance`, `record`, `export`, `validate`, `refresh`, `user-import asu-mhz-csv|psma-coreloss-db`,
`schema validate|migrate|export`, `import json`, `mcp serve|self-test`. `--json` prints machine-readable output.

## MCP server (`aipe-magnetics-db-mcp`)

Local stdio, read-only tools: `database_summary`, `list_sources`, `list_materials`, `get_material`,
`material_properties`, `list_cores`, `get_core`, `find_measurements`, `get_table` (max 5,000 rows), `provenance`.
Unknown arguments are rejected. Plugins for Claude Code (`plugins/claude/aipe-magnetics-database`, marketplace
`.claude-plugin/marketplace.json`) and Codex (`plugins/codex/aipe-magnetics-database`, `.agents/plugins/marketplace.json`)
start this server; install the package first:

```sh
pip install "aipe-magnetics-database[mcp] @ git+https://github.com/FulongLi/AIPE-Magnetics-Database.git"
# Claude Code
claude plugin marketplace add FulongLi/AIPE-Magnetics-Database
claude plugin install aipe-magnetics-database@aipe-magnetics-database
# Codex
codex plugin marketplace add FulongLi/AIPE-Magnetics-Database
codex plugin add aipe-magnetics-database@aipe-magnetics-database
```

## Web explorer

<https://fulongli.github.io/AIPE-Magnetics-Database/> is a read-only static site generated from the same data store by
`tools/build_site.py` and deployed by `.github/workflows/pages.yml`; see [site/README.md](../site/README.md).
