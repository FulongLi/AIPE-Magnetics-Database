# AIPE Magnetics Database

The canonical magnetics database scaffold for **AI for Power Engineering**.
This v0.1 change introduces an evidence-first record boundary, offline validation
and a conservative JSON import path. It contains **no real material, geometry,
winding, component, loss or measurement data yet**.

The original repository contained only the `Rogowski_Coil-` heading; its Git
history remains intact. No historical engineering files or datasets were removed.

## Open-source path

Python 3.10+ and the open-source JSON Schema library are sufficient:

```sh
python -m pip install -r requirements-dev.txt
python scripts/validate.py examples/synthetic-transformer.json
python -m unittest discover -s tests -v
python -m importers.json_record examples/synthetic-transformer.json data/derived/example-import.json
```

The importer creates a review candidate with the original file SHA-256 and
explicitly performs no unit inference. It refuses existing destinations and
existing import provenance. Synthetic examples stay in `examples/` or disposable
derived previews and must not be presented as real canonical data. Validation
does not access the network or execute artifact references.

## Record boundary

[magnetic-record.schema.json](schemas/magnetic-record.schema.json) defines:
magnetic/core material, core family, core geometry, winding, inductor, transformer,
magnetic component, loss model, measurement and operating condition. Every record
has a version, stable ID, source/assumption evidence, review status and explicit
synthetic flag. Physical quantities use SI `{value, unit}` objects. Unknown fields,
wrong dimensions, duplicate IDs and missing record/evidence references fail.

| Directory | Ownership |
| --- | --- |
| `data/raw/` | Original source bytes with source rights and checksums |
| `data/canonical/` | Reviewed, normalized records with retained provenance |
| `data/derived/` | Reproducible import previews/results, distinct from sources |
| `schemas/` | Independent domain schema v0.1.0 |
| `importers/` | Validating, non-overwriting source adapters |
| `references/` | Source acquisition/rights checklist and references |
| `examples/` | Clearly labeled synthetic structure examples |

See [Core mapping and provenance](docs/core-integration.md),
[source policy](references/README.md), [contribution process](CONTRIBUTING.md) and
[capability manifest](aipe.yaml). Materials, measurements and field solvers stay
separate responsibilities; no solver or external MCP is installed or claimed.

## Current limits

This is a scaffold, not a characterized magnetics catalogue or a design engine.
Manufacturer importers, measured loss surfaces, uncertainty, hysteresis and
validated design/solver adapters are future work. An empty property is omitted
instead of guessed. The checker validates declared metadata and references,
not source authenticity or engineering fitness. Core integration is a documented
mapping; no executable Core export is claimed. Repository licensing is unresolved
and remains `NOASSERTION`; [LICENSING.md](LICENSING.md) records this explicitly.
