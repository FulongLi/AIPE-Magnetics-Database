# AIPE Magnetics v0.2 architecture

AIPE Magnetics is one repository with one Python package (`aipe_magnetics`) and thin interaction
layers around it. The rule that shapes everything: **mathematics never depends on an AI host**.
Every calculation is a plain Python function with typed, validated inputs and an
`EngineeringResult` output; the CLI, the MCP server and the Codex / Claude Code plugins only call
that API.

```
                 ┌──────────────────────────── interaction layers ────────────────────────────┐
                 │  CLI (aipe-magnetics)   MCP stdio server (aipe-magnetics-mcp)   plugins/   │
                 └──────────────────────────────────────┬─────────────────────────────────────┘
                                                        │ typed calls only
                 ┌──────────────────────────────────────▼─────────────────────────────────────┐
                 │  api/   list_materials · get_material · compare_materials · estimate_core_loss  │
                 │         evaluate_hysteresis · design_inductor · design_transformer ·           │
                 │         validate_design · generate_report · verification_status                │
                 └───────────────┬───────────────────────┬────────────────────────┬─────────────┘
                                 │                       │                        │
        ┌────────────────────────▼──────┐   ┌────────────▼────────────┐   ┌───────▼──────────────────┐
        │ engineering/                  │   │ models/                 │   │ verification/            │
        │ materials · flux · core_loss  │──▶│ contract (MagneticModel)│◀──│ evidence registry        │
        │ inductor · transformer        │   │ empirical: SE, iGSE     │   │ suite + checks (L1/L2/L3)│
        │ saturation · winding · report │   │ physics: JA, Bertotti   │   │ reports (JSON + MD)      │
        └───────────────┬───────────────┘   │ dynamic: ODE framework  │   └──────────────────────────┘
                        │                   │ data_driven: ANN/LSTM/  │
        ┌───────────────▼───────────────┐   │   Transformer/Neural-ODE│
        │ data/                         │   └─────────────────────────┘
        │ records (v0.2 standard) ·     │
        │ schema · migrate · units ·    │   ┌─────────────────────────┐
        │ hdf5 waveforms · catalog ·    │   │ integrations/           │
        │ importers · source adapters   │   │ aipe_core (round trip)  │
        └───────────────────────────────┘   │ aipe_registry (manifest)│
                                            └─────────────────────────┘
```

## Modules

| Module | Responsibility | Key types |
| --- | --- | --- |
| `common` | errors, SI unit registry, `Quantity`, `EngineeringResult` envelope, strict JSON I/O | `Quantity`, `EngineeringResult`, `SourceReference` |
| `data` | Magnetic Data Standard v0.2 (pydantic normative models + generated JSON Schema), v0.1 migration, HDF5 waveform store, bundled catalogue, importers, external source adapters | `MagneticRecord`, `WaveformStore`, `Catalog`, `SourceAdapter` |
| `models` | Magnetic Model Contract and model families | `MagneticModel`, `CoreLossModel`, `HysteresisModel`, `ValidityDomain`, `ModelOutput` |
| `engineering` | design and estimation calculations with explicit assumptions | functions returning `EngineeringResult` |
| `verification` | evidence per capability and operating range, check suite, report generation | `EvidenceRecord`, `VerificationRegistry`, `@check` |
| `integrations` | AIPE-Core Engineering State export/import and validation, AIPE-Registry manifest validation | `MagneticComponentSummary`, `validate_engineering_state`, `validate_manifest` |
| `api` | stable typed entry points | `aipe_magnetics.api.*` |
| `cli` | `aipe-magnetics` command | argparse subcommands |
| `mcp` | local stdio MCP server (`mcp` SDK 2.x, protocol 2026-07-28) | `build_server()` |

## Contracts

* **Quantities** are `{value, unit}` objects. Records store SI only and keep `original`
  `{value, unit}` when converted. The API accepts any registered unit and normalises explicitly.
* **Model contract**: `identity` (id, version, family, method version), `domain`
  (materials, excitation kinds, frequency/temperature/flux ranges, DC-bias support, history
  requirement, numerical constraints, known limitations), `check_domain()` with a *policy*
  (`reject` raises `OutOfDomainError`; `warn` evaluates and flags) and `evaluate()` returning a
  `ModelOutput` with assumptions, limitations, warnings, sources and diagnostics.
* **Result envelope**: values with units and status (`calculated`, `assumed`, `estimated`,
  `looked_up`), assumptions, sources, model identity/version, verification status derived from the
  evidence registry for the given conditions, warnings, limitations, explicit errors.
* **Verification** levels: L1 software/schema correctness, L2 numerical and physical consistency,
  L3 engineering application validation. Evidence is recorded per capability and operating range; a
  result outside every verified range is flagged `out_of_range`.
* **Errors** are typed (`InvalidInputError`, `OutOfDomainError`, `DataNotFoundError`,
  `NumericalError`, `ConvergenceError`, `LicenseRestrictionError`, `AuthorizationError`). Invalid
  inputs are rejected, never clipped.

## Separation of AI logic from mathematics

`models/`, `engineering/`, `data/` and `verification/` import nothing from `mcp/`, `cli/` or
`plugins/` (the plain-text summary shared by the CLI and the executable demonstration lives in
`common/format.py`). The MCP server and the plugin skills are documentation and transport only;
`tests/mcp/test_server.py` asserts that the server builds without any API key or network access and
that its tool set equals `api.CAPABILITIES` plus `list_capabilities`, one tool per API function.

## Repository layout

```
AIPE-Magnetics-Database/
├── src/aipe_magnetics/        package (data, models, engineering, verification, integrations, api, mcp, cli)
├── schemas/                   v0.1 schema (unchanged) and v0.2 mirror of the packaged schemas
├── skills/                    canonical SKILL.md sources shared by both plugins
├── plugins/claude/            Claude Code plugin + marketplace
├── plugins/codex/             Codex plugin + marketplace
├── examples/                  v0.1 example, synthetic import example, DAB 10 kW demonstration
├── tests/                     pytest suites (data, models, engineering, api, mcp, plugins, integrations)
├── benchmarks/                reproducible fitting / timing scripts
├── docs/                      detailed documentation
├── scripts/, importers/       v0.1 tooling, kept working
├── pyproject.toml, aipe.yaml, README.md, CHANGELOG.md
```

## Capability overview

| Area | What you get |
| --- | --- |
| Data | Magnetic Data Schema v0.2 (materials, manufacturers, datasheet parameters, cores, windings, measurements, B-H loops, loss tables, waveforms, model metadata), SI with original units, checksums, provenance, licence; v0.1 migration; HDF5 waveform store; bundled catalogue (TDK N87 datasheet, 10 ferrites with MagNet-derived Steinmetz sets, 5 TDK cores); source adapters for Princeton MagNet, MagNet Challenge 2, Paderborn materialdatabase and mag-net-hub |
| Models | Steinmetz, iGSE, Bertotti, Jiles-Atherton (ODE framework), data-driven adapter (ANN / LSTM / Transformer / Neural ODE; numpy + optional torch / magnethub backends) behind one Magnetic Model Contract with declared validity domains |
| Engineering | material lookup and comparison, flux density, core-loss estimation, inductor and transformer sizing (every catalogue core offered in the material evaluated and ranked), saturation screening, Dowell winding loss, operating-point validation, Markdown/JSON reports; **waveform-informed transformer design** from the converter's actual waveforms (HDF5 / CSV / JSON import or SPS synthesis: exact RMS and harmonics, flux from the voltage integral, iGSE on the real flux, per-layer harmonic Dowell copper loss, leakage and external inductance, thermal estimate, ranked integer-turn candidates, material comparison) and structured **material measurement requests** |
| Verification | evidence per capability and operating range (L1 software, L2 numerical/physical, L3 engineering), check suite with machine- and human-readable reports, verification status in every result derived from the bundled evidence file (`L0 unverified`, stated explicitly, until the release step records that file) |
| Interfaces | typed Python API, `aipe-magnetics` CLI, stdio MCP server, Codex plugin, Claude Code plugin, AIPE-Core round-trip adapter, AIPE-Registry manifest |

Every result is an envelope with values + units, assumptions, model identity and version, input
conditions, sources, verification status, warnings and limitations. Invalid or out-of-domain inputs
are rejected explicitly, never clipped; uncertainty is reported as unknown when it is unknown.
Version 0.2.0 was developed on `feature/aipe-magnetics-v0.2`; the v0.1 record scaffold is preserved
unchanged (see [audit-v0.1.md](audit-v0.1.md) and the [CHANGELOG](../CHANGELOG.md)).
