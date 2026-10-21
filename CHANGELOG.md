# Changelog

## Unreleased — branch `feat/dab-10kw-workflow` (package version stays 0.2.0 until the release step records new evidence)

### Added
- **Waveform import** (`aipe_magnetics.engineering.waveform_import`): sampled converter waveforms from the AIPE DAB
  HDF5 layout (`/time`, `/channels/<name>` with `unit` / `description` / `sign_convention` attributes, JSON
  `metadata`; optional `h5py`), CSV (explicit units, strict numerals) and JSON documents; window selection, period
  detection (unbiased autocorrelation + zero-crossing refinement, checked against a frequency hint), extraction of the
  last complete period with explicit closure residual and steady-state metrics, exact piecewise-linear waveforms
  (`PiecewiseLinear`), flux linkage by trapezoidal integration of a voltage (drift removed), flux-density waveforms.
- **Waveform-informed DAB transformer design** - public capability `design_transformer_from_waveforms`
  (`WaveformInformedTransformerRequest`, CLI `design waveform-transformer`, MCP tool): actual primary / secondary
  currents and the transformer voltage (or SPS parameters synthesised with `design/transformers/dab.py`) -> exact RMS
  and Fourier harmonics, flux from the voltage integral, iGSE core loss on that flux at the design temperature
  (temperature interpolation / flagged extrapolation), per-layer harmonic Dowell copper loss for litz (strand
  sub-layers) and foil with the MMF profile of the arrangement, 1-D leakage inductance and the external inductance
  still required (`L_total - L_leak`), magnetising inductance (reluctance model with a stated residual gap and the
  datasheet AL where recorded), window fill, current density, thermal network (E cores, electrothermal) or lumped
  surface estimate; ranked integer-turn candidates with explicit violations, comparison with the square-current
  approximation (same geometry with `P / V` RMS and Dowell at the fundamental, actual RMS at the fundamental, and the
  public `design_transformer`), material comparison for the excitation (missing data listed), optional AIPE-Core
  Engineering State export whose analytical evidence derives from the waveform source evidence (path + SHA-256;
  evidence kinds stay assumption / analytical / source).
- **Measurement requests** - public capability `measurement_requests` (`MeasurementRequestSpec`, CLI
  `measurement-requests`, MCP tool): per material the coverage of B-H data, core loss vs f/B/T, DC bias, saturation vs
  temperature, complex permeability, winding AC resistance and thermal conductivity, and one structured request per
  gap with test conditions and priority (schema `aipe.magnetics.measurement-request/0.3`).
- Packaged MES 0.3.0 JSON Schema (`src/aipe_magnetics/schemas/state/v0.3.0/magnetic-engineering-state.schema.json`,
  mirrored under `schemas/state/v0.3.0/`), which `state/schema.py` already referenced.
- Optional-dependency extras `viz` (matplotlib), `mesh` (gmsh), `cad` (cadquery) named by the v0.3 install hints.
- Tests for the v0.3 `design/transformers/dab.py` path (SPS power / inductance / phase-shift relations, exact
  piecewise-linear current against the executed ngspice metrics), the waveform import, the waveform-informed design
  (synthetic sources and, when present, the executed DAB waveform file), the measurement requests, the loss-evaluator
  range check, the packaged state schema, the workflow verification record and the new CLI commands.
- `aipe.yaml`: capabilities `magnetics-waveform-informed-transformer-design`, `magnetics-measurement-requests`,
  inputs / outputs for converter waveforms and measurement requests, optional extras, model maturity entries.

### Fixed
- `database/loss.py`: the flux-density range check tested the lower bound with the two-sided `_covers`, so a value
  above the fitted range also produced the "below the fitted range" message and a value below produced "exceeds";
  upper and lower bounds are now checked separately (array inputs report the smallest non-zero element).
- `workflows/dab_transformer.py`: `WorkflowOptions.created_at` no longer defaults to a literal date; the caller must
  pass the run timestamp. The verification record is built from what was executed (`build_verification_record`):
  the software/schema level is `passed` only after the state validated in that run, the numerical level is at most
  `partial` because the workflow's energy-balance / convergence / mesh checks are self-consistency checks, and the
  reference to a non-existent `docs/release-verification-v0.3.0.md` is gone.
- `integrations/aipe_core/bridge.py`: a design result of any `design_transformer*` capability is exported as a
  transformer (previously only an exact `design_transformer` suffix).

### Known limitations of the new capability
- No evidence record exists yet in the bundled registry for `aipe.magnetics.design_transformer_from_waveforms`;
  its results therefore report `L0 unverified` for the sizing relations (the iGSE loss model keeps its own L2
  status in `data.loss_model_verification`). Adding `@check` evidence changes the bundle fingerprint and needs the
  maintainer's `verify --update-bundled` release step.
- Catalogue materials other than N87 carry loss parameter sets up to 90 degC only; at a 100 degC design temperature
  their losses are extrapolated from the 90 degC set and flagged. Only N87, 3C90 and 3C94 have saturation data; the
  other grades are listed as not evaluable with the missing data named, and the measurement requests ask for it.

## 0.2.0 — AIPE Magnetics v0.2 (unified magnetic engineering & standalone expert plugin)

Branch `feature/aipe-magnetics-v0.2`. The v0.1 files, commands and tests are preserved unchanged.

### Added
- **Python package `aipe-magnetics`** (`src/aipe_magnetics`, Python ≥ 3.10, `pyproject.toml`) with
  five functional modules (`data`, `models`, `engineering`, `verification`, `integrations`) and thin
  interaction layers (`api`, `cli`, `mcp`). All mathematics is callable without a language model.
- **Magnetic Data Schema v0.2** (`aipe_magnetics.data.records`, generated JSON Schema 2020-12 in
  `src/aipe_magnetics/schemas/v0.2` mirrored to `schemas/v0.2`): manufacturers, materials with
  datasheet parameters and Steinmetz parameter sets, core families/geometries, winding geometry,
  components, operating conditions, excitation waveforms, measurements, B-H loops, core-loss tables,
  HDF5 waveform datasets, datasets, model metadata; explicit SI with original units, checksums,
  provenance, quality, uncertainty (never fabricated), licence per record and evidence item.
- **v0.1 → v0.2 migration** (`aipe-magnetics schema migrate`), strict validation of both versions.
- **HDF5 waveform store** (`aipe.magnetics.waveforms/0.2` layout) with per-array units, sampling,
  timebase, sign convention, NaN missing values and SHA-256 checksums.
- **Bundled catalogue** generated by `tools/build_catalog.py` with full provenance: TDK N87 datasheet
  parameters (June 2025 datasheet), ten MnZn ferrites with temperature-specific Steinmetz sets fitted
  reproducibly from the Princeton MagNet scalar databases (MIT), Mühlethaler 2012 N87 parameters,
  five TDK core geometries (E 65/32/27, E 70/33/32, ETD 59/31/22, PM 87/70, UU 93/152/30).
- **Source adapters** for Princeton MagNet, MagNet Challenge 2, Paderborn materialdatabase and
  mag-net-hub (discovery, metadata, extraction, normalisation, integrity, attribution, pinned
  versions, licence checks, caching, explicit errors); no third-party data redistributed.
- **Model framework**: Magnetic Model Contract (identity, validity domain, domain policy),
  classical Steinmetz, iGSE (analytic PWL and sampled waveforms, minor-loop detection), Bertotti
  loss separation, Jiles-Atherton hysteresis on a reusable ODE framework (state, excitation, solver
  configuration, convergence study, periodic steady state, energy accounting), data-driven adapter
  (architecture / weights / inference configuration separation; numpy ANN + LSTM, Neural ODE,
  optional torch and mag-net-hub backends, verified synthetic reference ANN).
- **Engineering engine**: material lookup/comparison, flux density, core-loss estimation,
  inductor and transformer sizing (every catalogue core offered in the material evaluated and the
  feasible, completely evaluated candidates ranked by total estimated loss), saturation screening,
  Dowell winding resistance and copper loss, operating-point validation, Markdown/JSON reports.
- **Verification framework**: evidence per capability and operating range at levels L1/L2/L3,
  check suite with machine- and human-readable reports, release evidence file consulted by every
  API result (`verified` / `partially_verified` / `out_of_range` / `unverified`; `L0 unverified` with an
  explicit statement while the file has not been recorded by the release step).
- **Stable API and CLI** (`aipe_magnetics.api`, `aipe-magnetics`), typed requests and result
  envelopes (`aipe.magnetics.result/0.2`).
- **Local stdio MCP server** (`aipe-magnetics-mcp`, mcp SDK 2.x): read-only by default, gated and
  path-confined report writing, no keys, no cloud, no shell execution.
- **Codex and Claude Code plugins** sharing five skills and one `.mcp.json`; marketplaces at the
  repository root (`.agents/plugins/marketplace.json`, `.claude-plugin/marketplace.json`).
- **AIPE-Core round-trip adapter** (export → offline contract validation → import) with pinned
  Core/Registry schemas; `aipe.yaml` updated to v0.2 capabilities (`integration: native`).
- **10 kW / 100 kHz DAB transformer demonstration** (`aipe-magnetics demo dab-10kw`).
- **CI** workflow (Linux/Windows/macOS × Python 3.10/3.12), clean-environment wheel install test,
  offline check, plugin packaging checks. Its `verify --check-bundled` step passes only once the
  bundled evidence file is committed (release step); no green run is recorded in this branch state.
- Documentation: architecture, v0.1 audit, data standard, models, verification, API, MCP,
  plugins (install/uninstall/troubleshooting), sources and licensing, Core integration, demo.

### Fixed during the pre-release review
- Catalogue: `core.tdk-e70-33-32` mean turn length corrected from 230.5 mm (two-core-set former) to
  209 mm (B66372B1000T001, one core set, AR 15.8 uOhm) per the TDK datasheet coil-former table p.4; AR
  values recorded for every coil former; UU 93/152/30 window height stored as the drawn single-U 48 mm
  with the derived 96 mm marked as derived; Muhlethaler 2012 N87 set validity reduced to the Table I
  fit span (20-50 kHz); unsupported `power` tags and E/U-family standards removed.
- `design_transformer` / `design_inductor`: candidates whose copper loss or window fill cannot be
  evaluated are reported as not feasible (`evaluation: incomplete`) instead of being ranked on core
  loss alone; a winding the Dowell model cannot evaluate or a foil wider than the window makes the
  candidate infeasible; the selected candidate's warnings, Curie/frequency-range/loss-parameter
  checks, a `zero_dc_bias` and (for trapezoidal voltage) a `trapezoidal_voltage_as_square` assumption
  reach the top level; several secondaries share the throughput power (equal split or `power_share`);
  user-supplied currents are labelled `specified`; the no-feasible-candidate error names the reasons.
- `design_inductor`: `ripple_duty` (default 0.5) controls the triangular ripple waveform used by iGSE
  and is recorded as an assumption; a ripple larger than 2 x the peak current is rejected.
- Dowell porosity for round wire uses the equivalent square-conductor width (eta = h / pitch);
  conductor temperatures outside the stated validity of the linear resistivity model are flagged.
- `compare_materials` derives its verification block from the evidence registry (per material in
  `data.comparison[].loss_verification`) instead of asserting L2.
- Bertotti with user coefficients works (`model_parameters` k_h, beta_h, k_c, k_e).
- `validate_design` reports `data.passed = null` (inconclusive) when a required check could not be
  evaluated; the Curie check repeats the record note about the datasheet bound.
- Evidence registry: a condition that is not supplied no longer counts as out of range
  (`in_verified_range: null` with the unchecked dimension named); the missing bundled evidence file is
  named in every status, in `aipe-magnetics info` and in `verification_status()`; `--update-bundled`
  re-runs the suite until the bundle is stable and omits wall-clock durations.
- MCP server: unknown or misspelled tool arguments are refused instead of silently dropped;
  `evaluate_hysteresis` defaults match the API (`periods=10`, `steady_state_tol` exposed); write
  authorisation requires an output directory at construction; `AIPE_MAGNETICS_MCP_ALLOW_WRITE` is parsed
  explicitly; an existing destination is `destination_exists`, not `authorization_required`.
- CLI: `--flux-fraction 0` reaches validation (rejected explicitly); `--duty-rise` alone derives the
  symmetric trapezoid; malformed `--param`, missing files and non-envelope inputs give one
  `ERROR <code>: <message>` line (exit 1); `report` accepts single envelopes, JSON reports and the demo
  `steps[]` output; `mcp serve --allow-write` requires `--output-dir`.
- Typed inputs: numeric strings and booleans are rejected for quantities, integers and floats; an
  `unknown` uncertainty carries no numbers; the trapezoid constraint `2*(duty_rise + duty_flat) == 1`
  is enforced at validation (`duty_flat` derived when omitted); `flux_density_peak` must be positive;
  sampled waveforms declare `endpoint_included`.
- Reports render material comparisons and validation check verdicts, print quantified uncertainties,
  escape table pipes and no longer truncate candidate notes.
- Documentation, manifest, skills and CI wording no longer claim area-product screening, an executed
  ecosystem release run, host-validation runs, a reviewer-run torch test or committed release
  evidence; the state of the bundled evidence file is stated wherever verification is mentioned.

### Known issues (maintainer)
- The bundled evidence `src/aipe_magnetics/verification/evidence/v0.2.0.json` (58 checks, all passing)
  and its Markdown twin `docs/verification-report-v0.2.0.md` were produced by the release run on
  2026-10-19 (`verify --update-bundled`); re-run the release step after changing any check, otherwise
  `verify --check-bundled` and CI fail.
- Verification evidence is L1/L2 only (software, numerical and physical consistency); no L3
  experimental validation is bundled. Linux/macOS CI legs are defined but had not run at release time.
- `.github/workflows/validate.yml` runs its original `python -m unittest discover -s tests -v`
  unchanged: the pytest-only suites live in sub-directories without package markers (and the release
  tests in `tests/release/`), so unittest discovery still finds exactly the 17 v0.1 tests.

### Changed
- `aipe.yaml`: `name` AIPE Magnetics, `type: tool`, capabilities and tools extended, `aipe.core`
  dependency optional, `extensions["aipe.magnetics"]` added; manifest `schema_version` stays 0.1.0.
- `LICENSING.md` extended with the third-party component table; repository licence stays
  `NOASSERTION`.

### Unchanged (preserved)
- `schemas/magnetic-record.schema.json` (v0.1.0), `scripts/validate.py`, `importers/json_record.py`,
  `tests/test_records.py`, `examples/synthetic-transformer.json`, `.github/workflows/validate.yml`.

## 0.1.0 — proposed scaffold

- Establish canonical AIPE Magnetics Database identity and Registry manifest.
- Define strict SI magnetic record kinds with source/evidence metadata.
- Add offline validation, conservative JSON import and negative tests.
- Document Core mappings and raw/canonical/derived provenance boundaries.
- Add one synthetic identity-only example, with no fabricated real data.
