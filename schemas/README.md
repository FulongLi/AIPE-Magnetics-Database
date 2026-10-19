# Schemas

| Path | Content | Policy |
| --- | --- | --- |
| `magnetic-record.schema.json` | AIPE magnetic record **v0.1** (schema 0.1.0, JSON Schema 2020-12) | frozen, kept byte-for-byte; still validated by `scripts/validate.py` and by `aipe_magnetics.data.schema.validate_record` (version dispatch) |
| `v0.2/` | AIPE Magnetic Data Standard **v0.2** (schema 0.2.0): `magnetic-record.schema.json`, `quantity.schema.json`, `properties/<kind>.schema.json` | **mirror** of the canonical packaged copies in `src/aipe_magnetics/schemas/v0.2/`; must stay byte-identical (tests and the verification check `data.schema-mirror-in-sync` assert it) |

## v0.2 in one paragraph

The normative definition of v0.2 is the set of pydantic models in
`src/aipe_magnetics/data/records.py`. The JSON Schema files are **generated** from them: the record
envelope (`schema_version`, `id`, `kind`, `name`, `synthetic`, `license`, `properties`,
`evidence`/`evidence_refs`, optional `provenance`, `quality`, `artifacts`, `dataset_ref`, `tags`,
namespaced `extensions`) plus one `allOf`/`if`/`then` block per record kind that binds `properties`
to `#/$defs/<KindProperties>`. All physical quantities are SI `{value, unit}` objects (`original`
`{value, unit}` retained when converted; optional, never fabricated `uncertainty`); missing values are
omitted, never zero; large arrays live in HDF5 files described by `waveform_dataset` records
(layout `aipe.magnetics.waveforms/0.2`, see `aipe_magnetics/data/hdf5.py`). The `$id` of the
record schema is
`https://raw.githubusercontent.com/FulongLi/AIPE-Magnetics-Database/v0.2.0/schemas/v0.2/magnetic-record.schema.json`;
there are no remote `$ref`s.

## What the generated JSON Schema enforces (and what it does not)

* Structure of the envelope and of every kind's `properties` object (`additionalProperties: false`
  everywhere); kinds whose property model has mandatory fields (`core_loss_measurement`, `bh_loop`,
  `waveform_dataset`, `dataset`, `excitation_waveform`, `loss_model`, `model_metadata`) also
  **require** the `properties` key in their `then` block.
* Format annotations: `Evidence.created_at` is `date-time` (RFC 3339); `measurement.artifact`,
  `loss_model.model_artifact` (both as in v0.1), `Artifact.uri`, `Evidence.url`, `dataset.url`,
  `manufacturer.website` and `model_metadata.weights_artifact` are `uri-reference`.
* Optional fields are emitted as `anyOf: [<type>, {type: null}]`; the standard says *omit* missing
  values, and the JSON importer rejects explicit `null`s.
* **Not** expressed in the JSON Schema: per-field SI units (`leakage_inductance` in `H` ...), the
  evidence rules and the sign rules. `aipe_magnetics.data.schema.validate_record` therefore runs
  three layers on a v0.2 record and prefixes its messages accordingly: `[schema]` (JSON Schema),
  `[semantic]` (pydantic normative model: SI units, evidence rules, RFC 3339 timestamps and the
  v0.1-parity sign rules of `records.py`) and `[physical]` (every number anywhere in the record
  finite - the model does not check non-Quantity floats such as range bounds, Steinmetz exponents or
  fit metrics - plus the sign rules restated on the JSON form with the exact field path: frequencies,
  absolute temperatures, geometric sizes, inductances, permeabilities, Steinmetz `k` ... `> 0`; peaks,
  rms values, loss densities, air gaps, coercivity ... `>= 0`; signed DC quantities unconstrained;
  never stricter than the model where the model has a rule, and also covering conditions inside
  `ConditionedValue`/`TemperaturePoint`, B-H loop summaries, excitation amplitude/frequency and the
  waveform sampling rate, which the model leaves unconstrained). Every file-level path (catalogue
  loading, importers, migration output, CLI) uses `validate_record`.
* The files are produced by the pydantic JSON Schema generator; the generator versions used for the
  committed export are recorded in the verification evidence (`data.schema-v02-is-valid-draft2020`,
  metric `generator_versions`). A different pydantic release may emit different shapes, in which case
  the export must be refreshed and reviewed, not patched by hand.

## Refreshing the mirror

```sh
python -m aipe_magnetics.data.schema --export schemas/v0.2 --export-package   # regenerate both copies
python -m aipe_magnetics.data.schema --check schemas/v0.2                      # exit 1 when the mirror is stale
```

Never edit the generated files by hand; change the pydantic models and re-export. Files are written
with sorted keys, two-space indent, UTF-8 and LF line endings, so the export is byte-reproducible.

## Validation and migration

```sh
python -m aipe_magnetics.data.schema --validate examples/synthetic-transformer.json   # v0.1 or v0.2, by schema_version
python -m aipe_magnetics.data.schema --validate a.json b.json [--no-bundled-catalogue]  # the files form ONE set
python -m aipe_magnetics.data.migrate examples/synthetic-transformer.json out/synthetic-transformer-v02.json
python scripts/validate.py examples/synthetic-transformer.json                        # legacy v0.1 command, unchanged
```

`--validate` treats the files on the command line as one catalogue, as `scripts/validate.py` did
for v0.1: every file gets its own structural / semantic / physical messages, and `*_ref` fields must
resolve inside the set or inside the bundled catalogue (`--no-bundled-catalogue` drops the latter).
A record that references another file therefore needs that file on the command line; each message
is printed under the file it concerns.

Migration 0.1.0 -> 0.2.0 is mechanical (no physical value is guessed): the input must satisfy the
per-record v0.1 rules (schema, evidence rules including "measurement requires measurement evidence",
finiteness), so a v0.1-invalid record is never "healed"; the tool then bumps the version, derives the
record `license` from the referenced evidence (or `NOASSERTION`), adds `provenance` and
`extensions["aipe.migration"]`, rewrites only `measurement.properties.tool` `{name, version}` to the
v0.2 `Attribution` `{id: <name>, name, version}`, and validates the output with all three v0.2 layers.
