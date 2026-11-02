# Migrating from AIPE Magnetics 0.2 to the 0.3 two-repository layout

AIPE Magnetics 0.2.0 was one package (`aipe-magnetics`, import `aipe_magnetics`) in the
AIPE-Magnetics-Database repository. In 0.3.0 it is split by responsibility:

| | AIPE-Magnetics-Database | AIPE-Magnetics-Design |
| --- | --- | --- |
| Distribution | `aipe-magnetics-database` | `aipe-magnetics-design` (depends on the Database) |
| Import package | `aipe_magnetics_database` | `aipe_magnetics_design` |
| CLI | `aipe-magnetics-db` | `aipe-magnetics-design` (alias `aipe-magnetics`) |
| MCP server | `aipe-magnetics-db-mcp` (read-only data) | `aipe-magnetics-design-mcp` |
| Agent plugin | `aipe-magnetics-database` | `aipe-magnetics-design` |
| Contents | schema v0.2, records, catalogues, importers, source adapters, data store, queries | models, engineering, design, geometry, simulation, optimisation, verification, workflows, reports |

Dependency direction: **Design -> Database**. The Database never imports Design.

## Install order

```sh
pip install "aipe-magnetics-database @ git+https://github.com/FulongLi/AIPE-Magnetics-Database.git"
pip install "aipe-magnetics-design @ git+https://github.com/FulongLi/AIPE-Magnetics-Design.git"
```

Uninstall the old `aipe-magnetics` 0.2 distribution first (`pip uninstall aipe-magnetics`); it owned the
`aipe_magnetics` import name that the Design package now provides as a shim.

## Module map

| v0.2 module | 0.3 module |
| --- | --- |
| `aipe_magnetics.common.*` | `aipe_magnetics_database.common.*` |
| `aipe_magnetics.data.*` | `aipe_magnetics_database.data.*` |
| `aipe_magnetics.database.accessors / properties / query` | `aipe_magnetics_database.database.*` |
| `aipe_magnetics.engineering.materials` (lookup / summary / list / get) | `aipe_magnetics_database.database.materials` (re-exported by `aipe_magnetics_design.engineering.materials`) |
| `aipe_magnetics.database.loss` (`LossEvaluator`, `FluxShape`) | `aipe_magnetics_design.engineering.loss_evaluator` |
| `aipe_magnetics.database.estimate_material_loss` | `aipe_magnetics_design.engineering.material_loss.estimate_material_loss` |
| `aipe_magnetics.database` (package facade) | `aipe_magnetics_design.data_access` |
| `MagnetHubAdapter.run_model` | `aipe_magnetics_design.models.datadriven.magnethub_backend.run_magnethub_model` |
| every other `aipe_magnetics.<x>` | `aipe_magnetics_design.<x>` |

## Compatibility shim

With `aipe-magnetics-design` installed, `import aipe_magnetics.<module>` still works: a meta-path finder
maps each v0.2 module path to the 0.3 module object (identical classes and functions) and emits a
`DeprecationWarning` naming the new path. The shim will be removed in 0.5.

## Behaviour changes

* Result envelopes of Database queries name the tool `aipe.magnetics.database` (`AIPE Magnetics Database`);
  Design results keep `aipe.magnetics`.
* `search_materials(frequency=..., temperature=...)` in the Database filters by *coverage* of the recorded
  Steinmetz parameter sets (`steinmetz_coverage`) without evaluating any loss model; the selection rule is
  unchanged.
* The v0.2 JSON Schema files are byte-identical; schema version stays `0.2.0`.
