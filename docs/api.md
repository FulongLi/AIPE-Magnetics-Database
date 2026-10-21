# Python API and CLI

```python
import aipe_magnetics.api as api

r = api.estimate_core_loss(material="N87",
                           frequency={"value": 100, "unit": "kHz"},
                           flux_density_peak={"value": 150, "unit": "mT"},
                           temperature={"value": 100, "unit": "degC"},
                           waveform={"kind": "triangular", "duty": 0.5})   # iGSE
print(r.status, r.values["loss_density"].quantity, r.model.id, r.verification.state)
for a in r.assumptions: print("assumption:", a.statement)
for l in r.limitations: print("limitation:", l.statement)
```

All functions live in `aipe_magnetics.api` and accept a typed request object from
`aipe_magnetics.api.types` **or** keyword arguments. They return an
`aipe_magnetics.common.results.EngineeringResult` and never raise for engineering rejections
(`status == "rejected"` with `errors[]`); programming errors still raise.

| Function | Request type | Main values |
| --- | --- | --- |
| `list_materials(query=None, material_class=None)` | - | `data.materials` |
| `get_material(material)` | - | permeability, Bsat(T), Curie temperature, density, resistivity; `data.material` |
| `compare_materials(materials, frequency=None, flux_density_peak=None, temperature=None)` | - | `data.comparison` (loss density, saturation margin) |
| `estimate_core_loss(...)` | `CoreLossRequest` | `loss_density` [W/m^3], `core_loss` [W] when a core is given |
| `evaluate_hysteresis(...)` | `HysteresisRequest` | `flux_density_peak`, `remanence`, `coercivity`, `loop_energy_density` |
| `design_inductor(...)` | `InductorDesignRequest` | `turns`, `air_gap`, `flux_density_peak`, `core_loss`, `copper_loss`, `total_loss` |
| `design_transformer(...)` | `TransformerDesignRequest` | `turns_<winding>`, `flux_density_peak`, `core_loss`, `copper_loss`, `total_loss`, `data.candidates` |
| `design_transformer_from_waveforms(..., created_at=None, export_state=False)` | `WaveformInformedTransformerRequest` | `turns_primary`, `turns_secondary`, `turns_ratio`, `flux_linkage_peak_to_peak`, `flux_density_peak`, `saturation_margin`, `loss_density`, `core_loss`, `copper_loss`, `total_loss`, `leakage_inductance`, `external_inductance_required`, `magnetising_inductance`, `window_fill`, `temperature_rise`, `hotspot_winding`, `hotspot_core`, `current_rms_<winding>`; `data.excitation`, `data.candidates` (ranked), `data.square_current_comparison`, `data.material_comparison`, `data.measurement_requests`, `data.engineering_state` (with `export_state=True`) |
| `measurement_requests(...)` | `MeasurementRequestSpec` | `data.requests[]` (per material: `coverage`, `requests[]` with `category`, `priority`, `test_conditions`, `deliverable`), `data.summary` |
| `validate_design(...)` | `DesignValidationRequest` | `saturation_margin`, `loss_density`; `data.checks`, `data.passed` (`true` / `false` / `null` = inconclusive because a check could not be evaluated) |
| `generate_report(results, ...)` | `ReportRequest` | `data.content`, `data.written_to` |
| `verification_status(capability=None, ...)` | - | `verification`, `data.records` |
| `list_capabilities()` | - | `data.capabilities` |

### Waveform-informed transformer design

`design_transformer_from_waveforms` replaces the `P / V` square-current approximation of `design_transformer` with
the converter's **actual** winding waveforms. The waveform source (`waveforms`) is one of

| `kind` | Fields | Notes |
| --- | --- | --- |
| `hdf5` | `path`, `channels` (`primary_current` required; `secondary_current`, `primary_voltage`, `magnetising_current` optional), `window` `{t0, t1}`, `time_dataset` (`time`), `channel_group` (`channels`) | AIPE DAB layout: `/time` [s], `/channels/<name>` with `unit` / `description` / `sign_convention` attributes, root attribute `metadata` (JSON; `spec.l_ext`, `l_lk_p`, `l_lk_s`, `n_ratio` give the total series inductance when `series_inductance_total` is omitted). Needs `h5py` (`aipe-magnetics[hdf5]`). |
| `csv` | `path`, `channels`, `time_column`, `time_unit`, `units` (unit per mapped column), `delimiter`, `window` | units are never inferred from headers |
| `json` | `path` or inline `document`, `channels`, `window` | `{"time": {"unit", "values"}, "channels": {name: {"unit", "values"}}}` or one uniform period `{"frequency": {...}, "channels": {...}}` |
| `sps` | `sps`: `power`, `primary_voltage`, `secondary_voltage`, `phase_shift` **or** `series_inductance` | lossless single-phase-shift model (`design/transformers/dab.py`) synthesises the trapezoidal current and the triangular flux |

Processing: window -> period detection (autocorrelation, checked against `frequency`) -> last complete period ->
exact RMS / Fourier harmonics (`harmonics`, default 41) of each winding current -> flux linkage from the integral of the
primary voltage (drift removed) -> for every evaluable material (Bsat at `temperature`, loss set covering the
frequency; extrapolated sets are flagged) and every candidate core: minimum primary turns from the flux-limit policy
(and `loss_density_target`), `turn_steps` further turn counts, `N2 = round(N1 / turns_ratio_target)` within
`turns_ratio_tolerance`, iGSE core loss on the actual flux waveform, winding layout (parametric E-core layers or a
generic stack on the coil-former data) for every conductor option (`windings.conductors`, litz strand diameters, foil
thicknesses, current densities) and arrangement (`P-S`, `P-S-P`, ...), per-layer harmonic Dowell copper loss (litz as
strand sub-layers), 1-D leakage inductance, `external_inductance_required = L_total - L_leak`, magnetising inductance,
window fill, current density, thermal estimate (`thermal`: forced convection 40 W/m^2/K at 40 degC by default - an
explicit assumption - or natural convection; hot-spot limits). Feasible candidates are ranked by total loss at the
design temperature; `data.infeasible_summary` counts the violated constraints. Results hold for the supplied operating
point only; everything is analytical (limitations are listed in every result).

`measurement_requests` assesses, per material, the catalogue coverage of B-H data, core loss vs f/B/T, DC bias,
saturation vs temperature, complex permeability, winding AC resistance and thermal conductivity for the given
excitation and returns one request per gap with test conditions and a priority (`high` blocks the design, `medium`
limits accuracy, `low` not consumed by the current models); schema `aipe.magnetics.measurement-request/0.3`.

## Result envelope (`aipe.magnetics.result/0.2`)

| Field | Meaning |
| --- | --- |
| `status` | `ok`; `rejected` for invalid/out-of-domain inputs and missing data, for a design with no feasible candidate (`no_feasible_candidate`, reasons aggregated in the message) and for a `validate_design` whose checks fail (`design_checks_failed`; values are still computed and `data.passed` is `false`); `error` for a nonphysical result (e.g. negative loop energy) |
| `values` | `{key: {label, quantity{value, unit, original?, uncertainty?}, status, derived_from, note?}}`; status is `calculated`, `estimated`, `looked_up` (catalogue data), `specified` (caller input echoed), `assumed` or `measured` |
| `model` | `{id, name, version, family, method_version}` of the model that produced estimates |
| `inputs` | validated, SI-normalised input conditions |
| `assumptions` | `{id, statement, basis, affects}` - every default, policy and engineering judgement |
| `sources` | datasheets, publications, repositories, datasets with locators and licences |
| `verification` | `{level L0-L3, state not_run/verified/partially_verified/out_of_range/unverified/failed, statement, evidence_refs, in_verified_range}` from the evidence registry for *these* conditions; `not_run` is the state of a rejected result (nothing evaluated), `in_verified_range` is `null` when no range was declared or a declared dimension was not supplied |
| `warnings`, `limitations`, `errors` | explicit, machine-readable |
| `data` | structured payloads (tables, records, candidate lists) |

Uncertainty is reported only when a source quantifies it; otherwise `uncertainty` is absent and must
be read as unknown.

## Units

Inputs accept any unit in `aipe_magnetics.common.units` (e.g. `kHz`, `mT`, `degC`, `A/mm^2`,
`kW/m^3`, `uH`); they are converted to coherent SI with the original value retained. Unknown units
and dimension mismatches are rejected. Bare numbers are rejected.

## CLI

```sh
aipe-magnetics info
aipe-magnetics materials list [--query N87] [--class ferrite_mnzn] [--json]
aipe-magnetics materials get N87 [--json]
aipe-magnetics materials compare N87 3C94 -f "100 kHz" -b "150 mT" -t "90 degC"
aipe-magnetics loss estimate N87 -f "100 kHz" -b "150 mT" -t "100 degC" --waveform triangular [--core core.tdk-e65-32-27] [--json]
aipe-magnetics hysteresis evaluate --field-peak "500 A/m" --periods 4
aipe-magnetics design inductor -L "100 uH" --current-peak "20 A" --current-rms "14 A" --ripple "4 A" [--ripple-duty 0.5] -f "100 kHz" --material N87
aipe-magnetics design transformer -P "10 kW" -f "100 kHz" --primary-voltage "800 V" --secondary-voltage "400 V" --material N87 -t "100 degC" --loss-target "300 kW/m^3"
aipe-magnetics design waveform-transformer --source runs/dab/waveforms.h5 --channel primary_current=i_L --channel secondary_current=i_s \
    --channel primary_voltage=v_tp --window "200 us" "240 us" -f "100 kHz" --turns-ratio 2 --loss-target "300 kW/m^3" \
    [--material N87] [--core core.tdk-e65-32-27] [--conductor litz --conductor foil] [--arrangement P-S-P] [--cooling forced --heat-transfer-coefficient "40 W/m^2/K"] \
    [--state-output state.json --created-at 2026-10-21T12:00:00Z] [--json]
aipe-magnetics design waveform-transformer --sps-power "10 kW" --sps-primary-voltage "800 V" --sps-secondary-voltage "400 V" --sps-phase-shift "22.5 deg" -f "100 kHz" --turns-ratio 2
aipe-magnetics measurement-requests -f "100 kHz" -b "178 mT" -t "100 degC" [--material N87 --material 3C94] [--temperature-range "25 degC" "120 degC"] [--output requests.json]
aipe-magnetics validate --material N87 --core core.tdk-e65-32-27 --turns 14 -f "100 kHz" --voltage-peak "800 V" -t "100 degC"
aipe-magnetics report result1.json result2.json --output report.md     # envelopes, JSON reports (results[]) or demo output (steps[].result)
aipe-magnetics verify --output reports/verification --check-bundled     # exits 1 until the maintainer's bundled evidence file exists
aipe-magnetics demo dab-10kw --output reports/demo
aipe-magnetics schema validate <record.json> | migrate <v0.1.json> <v0.2.json> | export <dir> [--check]
aipe-magnetics import json <source.json> <destination.json> [--migrate]
aipe-magnetics mcp serve [--allow-write --output-dir DIR] | self-test
```

Exit codes: 0 ok, 2 rejected result (including a design that fails validation checks), 1 error (one
`ERROR <code>: <message>` line on stderr, never a traceback). `--json` prints the full envelope.
