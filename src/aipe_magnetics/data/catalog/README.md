# Bundled catalogue (`aipe_magnetics/data/catalog`)

`Catalog.default()` loads every `**/*.json` file below this directory. Each file holds **one**
schema-0.2.0 record and is validated strictly on load (JSON syntax, JSON Schema, pydantic
semantics, unique ids, resolvable references). A malformed file fails the whole load with its path.

## Current content: one clearly synthetic triple

| id | kind | purpose |
| --- | --- | --- |
| `manufacturer.example-synthetic` | `manufacturer` | fictional manufacturer so that `manufacturer_ref` resolves |
| `material.example-synthetic-ferrite` | `magnetic_material` | illustrative MnZn-ferrite-like material with **round-number** synthetic parameters |
| `core.example-synthetic-e42` | `core_geometry` | illustrative E-core set with round-number effective parameters (`Ae*le = Ve`) |

All three carry `synthetic: true`, assumption evidence (`review_status: unreviewed`) and the tag
`synthetic`. None of the numbers is a datasheet or measured value; the synthetic Steinmetz set
`k = 1 W/m^3, alpha = 1.5, beta = 2.5` is chosen only so that `Pv(100 kHz, 0.1 T) = 100 kW/m^3`
exactly, which makes it convenient for self-consistency tests. Engineering results computed from
these records must be reported as synthetic / illustrative.

## Adding real entries (maintainer)

1. One record per file, `schema_version: "0.2.0"`, SI `{value, unit}` quantities with `original`
   retained when converted (`Quantity.si`).
2. `synthetic: false` records need non-assumption evidence (`source`, `measurement`, ...) with a
   real citation, `locator` (page / table / figure), `license` and, where available, `url`,
   `version` and `sha256` of the source bytes.
3. Keep `review_status: unreviewed` until an owner has reviewed the values.
4. Place files by kind (`manufacturers/`, `materials/`, `cores/`, ...); file names are free, ids are not.
5. Run `python -m aipe_magnetics.data.schema --validate <file> [...]` (the files on the command line
   plus the bundled catalogue form the reference context; the three layers `[schema]`, `[semantic]`
   and `[physical]` must all pass) and the data tests (`pytest tests/data`);
   `Catalog.default().validate()` must return `[]`.
6. The data-layer tests and the verification check `data.catalog-loads-and-cross-references` assert
   per-record invariants only (synthetic records are tagged `synthetic`; real records cite
   `source`/`measurement` evidence with a `locator` or `url`), never record counts, so adding real
   entries does not break them.

Redistribution rules of third-party data apply (see `references/README.md` and `LICENSING.md`).
