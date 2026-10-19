# Packaged JSON Schemas (canonical copies)

| Directory | Content | Status |
| --- | --- | --- |
| `v0.1/magnetic-record.schema.json` | byte-exact copy of the legacy repository file `schemas/magnetic-record.schema.json` (schema 0.1.0) | frozen; used by `load_schema("0.1.0")` and the v0.1 validation path when the repository script is not available |
| `v0.2/magnetic-record.schema.json` | record envelope of the AIPE Magnetic Data Standard v0.2 with `allOf`/`if`/`then` blocks binding `properties` per `kind` | **generated** from `aipe_magnetics.data.records` by `aipe_magnetics.data.schema` |
| `v0.2/properties/<kind>.schema.json` | standalone property schema of every record kind | generated |
| `v0.2/quantity.schema.json` | the SI `{value, unit, original?, uncertainty?}` object | generated |

The pydantic models are normative; **do not edit the generated files by hand**. Refresh both the
canonical copies here and the repository mirror `schemas/v0.2/` with:

```sh
python -m aipe_magnetics.data.schema --export schemas/v0.2 --export-package
python -m aipe_magnetics.data.schema --check schemas/v0.2 --check src/aipe_magnetics/schemas/v0.2   # exit 1 when stale
```

Properties of the generated schema: draft 2020-12, no remote `$ref` (everything under `$defs`,
named after the pydantic classes), `schema_version` required, `properties` required for kinds with
mandatory fields, `Evidence.created_at` with `format: date-time`, `uri-reference` on
`measurement.artifact` / `loss_model.model_artifact` (v0.1 parity) and on `Artifact.uri`,
`Evidence.url`, `dataset.url`, `manufacturer.website`, `model_metadata.weights_artifact`; sorted
keys / LF so an export is byte-reproducible (the pydantic / jsonschema versions used are recorded in
the verification evidence). The JSON Schema is the structural layer; per-field SI-unit requirements
(e.g. `leakage_inductance` in `H`), the evidence rules, RFC 3339 timestamps and the v0.1-parity sign
rules are enforced by the semantic layer (the normative `records.py`), and the `[physical]` layer adds
finiteness of every number plus the sign rules restated with exact field paths and extended to the
quantities the model leaves unconstrained - `aipe_magnetics.data.schema.validate_record` runs all
three. See `schemas/README.md` at the repository root for the full description.
