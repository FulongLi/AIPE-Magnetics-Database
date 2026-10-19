# Initial repository audit (v0.1 → v0.2)

Audit date: 2026-10-19. Audited commit: `f73b0cd` (merge of PR #1, branch `codex/aipe-ecosystem-v0.1`).
Git history: 3 commits (`af9c7e5` initial README-only repository, `f3168b2` v0.1 scaffold, `f73b0cd` merge).
History is preserved unchanged; v0.2 work lives on the branch `feature/aipe-magnetics-v0.2`.

## 1. Inventory

| Path | Purpose | Assessment |
| --- | --- | --- |
| `schemas/magnetic-record.schema.json` | JSON Schema 2020-12, v0.1.0, 11 record kinds, SI `{value, unit}` objects | Sound, strict (`additionalProperties: false`). Limited vocabulary: no manufacturer identity, no datasheet parameters, no waveform/B-H/loss tables, no DC bias, no uncertainty, no checksums/licence at record level, no provenance beyond `evidence`. Kept byte-for-byte for backward compatibility. |
| `scripts/validate.py` | Offline validator: schema + duplicate IDs + evidence/reference integrity + finiteness | Correct and useful. Kept unchanged so the documented v0.1 commands keep working. Its rules are re-implemented for v0.2 in `aipe_magnetics.data.records` / `aipe_magnetics.data.schema`. |
| `importers/json_record.py` | Validating JSON import with SHA-256 of raw bytes; never overwrites; refuses existing provenance | Good conservative behaviour; semantics ported to the v0.2 importer (`aipe_magnetics.data.importers.json_record`). Kept unchanged. |
| `tests/test_records.py` | 17 `unittest` negative/positive tests | All pass (`python -m unittest discover -s tests`). Kept; v0.2 adds `pytest` suites. |
| `examples/synthetic-transformer.json` | Identity-only synthetic record | Kept; used as the v0.1 → v0.2 migration fixture. |
| `aipe.yaml` | Registry capability manifest v0.1.0 (`type: database`, `maturity: scaffold`, `integration: mapped`) | Valid against AIPE-Registry schema. Updated for v0.2 (capabilities, tools, extensions) while keeping manifest `schema_version: 0.1.0`, which the Registry schema requires as a constant. |
| `docs/core-integration.md` | Mapping table to AIPE-Core Engineering State | Documentation only; no executable adapter existed. v0.2 adds an executable, tested round-trip adapter (`aipe_magnetics.integrations.aipe_core`). |
| `data/raw`, `data/canonical`, `data/derived` | Ownership boundaries (READMEs only) | Kept. No data was ever bundled. |
| `references/README.md`, `LICENSING.md`, `CONTRIBUTING.md`, `CHANGELOG.md` | Policy documents | Kept; CHANGELOG extended. |
| `.github/workflows/validate.yml` | CI: unittest + validate example on Python 3.12 | Kept as the legacy job; a new matrix workflow adds install, pytest, verification, demo, MCP and packaging checks. |
| `requirements-dev.txt` | `jsonschema[format]>=4.23,<5` | Kept (legacy path); the package declares its own dependencies in `pyproject.toml`. |

There was **no** Python package, no `pyproject.toml`, no engineering calculation, no model, no MCP
server, no plugin and no real data in v0.1. Nothing is obsolete or duplicated; the only
"incomplete" items were explicitly declared as future work in the v0.1 README.

## 2. Dependencies and licences

| Component | Licence | Notes |
| --- | --- | --- |
| This repository | `NOASSERTION` (no licence file; owner decision pending) | v0.2 keeps `NOASSERTION`; no blanket licence is assigned to historical code. |
| jsonschema, referencing | MIT | runtime |
| numpy, scipy | BSD-3-Clause | runtime (v0.2) |
| pydantic | MIT | runtime (v0.2) |
| PyYAML | MIT | runtime (v0.2, manifest tooling) |
| h5py | BSD-3-Clause | optional (`[hdf5]`) |
| mcp (Model Context Protocol Python SDK 2.x) | MIT | optional (`[mcp]`) |
| torch | BSD-3-Clause | optional (`[torch]`), never required |
| AIPE-Core, AIPE-Registry (schemas vendored for offline tests) | MIT | pinned copies with NOTICE files |
| Princeton MagNet (`PrincetonUniversity/magnet`) | MIT (code); dataset terms recorded by the adapter | source adapter only, no bundled data |
| Paderborn `mag-net-hub` | MIT | optional model backend; weights fetched on demand by that package |
| Paderborn `materialdatabase` | GPL-3.0 | **no code copied**; adapter reads a user-provided or on-demand checkout; no redistribution |
| MagNet Challenge 2 (`minjiechen/magnetchallenge-2`) | no licence file | reference + user-supplied import only; nothing downloaded or redistributed by default |

## 3. Compatibility assessment

* **Record compatibility.** v0.2 is a superset of v0.1: every v0.1 `kind` remains valid, the
  `evidence` structure is unchanged (new optional fields only), and `aipe_magnetics.data.migrate`
  upgrades v0.1 files mechanically without inventing values. v0.1 files continue to validate with the
  v0.1 schema and the legacy script.
* **AIPE-Core.** Core v0.1.0 Engineering State (JSON Schema 2020-12, 14 schema files) was pinned at
  commit recorded in `src/aipe_magnetics/integrations/aipe_core/schemas/v0.1.0/NOTICE.md`. The domain
  schema version (0.2.0) is independent from the Core version (0.1.0), as Core's contracts require.
  `integration` is only raised from `mapped` to `native` because an executable round-trip adapter
  (export → Core validation → import) is implemented and tested.
* **AIPE-Registry.** The manifest schema is frozen at `schema_version: 0.1.0` with
  `additionalProperties: false`; new v0.2 metadata (model maturity, verification evidence, platforms,
  limitations, runtime requirements) therefore lives under the namespaced `extensions["aipe.magnetics"]`
  key, which the Registry schema permits.
* **Python.** v0.1 targeted Python 3.10+; v0.2 keeps 3.10 as the minimum.

## 4. Decisions

1. Keep all v0.1 files and commands working; add the package under `src/aipe_magnetics`.
2. Make the v0.2 pydantic models normative and generate JSON Schema from them (tested for
   agreement); keep the schema files in the package and mirror them under `schemas/v0.2/`.
3. Do not bundle third-party datasets. Provide pinned, checksum-verified on-demand downloads where
   the licence permits, and user-supplied import paths otherwise.
4. Record datasheet-derived material parameters only with explicit source locators and
   `review_status: unreviewed` until an owner reviews them.
5. Track verification evidence per capability and operating range (no global flag).
