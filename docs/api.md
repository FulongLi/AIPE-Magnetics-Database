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
| `validate_design(...)` | `DesignValidationRequest` | `saturation_margin`, `loss_density`; `data.checks` |
| `generate_report(results, ...)` | `ReportRequest` | `data.content`, `data.written_to` |
| `verification_status(capability=None, ...)` | - | `verification`, `data.records` |
| `list_capabilities()` | - | `data.capabilities` |

## Result envelope (`aipe.magnetics.result/0.2`)

| Field | Meaning |
| --- | --- |
| `status` | `ok`, `rejected` (invalid/out-of-domain inputs, missing data) or `error` |
| `values` | `{key: {label, quantity{value, unit, original?, uncertainty?}, status, derived_from}}`; status is `calculated`, `estimated`, `looked_up`, `assumed` or `measured` |
| `model` | `{id, name, version, family, method_version}` of the model that produced estimates |
| `inputs` | validated, SI-normalised input conditions |
| `assumptions` | `{id, statement, basis, affects}` - every default, policy and engineering judgement |
| `sources` | datasheets, publications, repositories, datasets with locators and licences |
| `verification` | `{level L0-L3, state verified/partially_verified/out_of_range/unverified/failed, statement, evidence_refs}` from the evidence registry for *these* conditions |
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
aipe-magnetics design inductor -L "100 uH" --current-peak "20 A" --current-rms "14 A" --ripple "4 A" -f "100 kHz" --material N87
aipe-magnetics design transformer -P "10 kW" -f "100 kHz" --primary-voltage "800 V" --secondary-voltage "400 V" --material N87 -t "100 degC" --loss-target "300 kW/m^3"
aipe-magnetics validate --material N87 --core core.tdk-e65-32-27 --turns 14 -f "100 kHz" --voltage-peak "800 V" -t "100 degC"
aipe-magnetics report result1.json result2.json --output report.md
aipe-magnetics verify --output reports/verification --check-bundled
aipe-magnetics demo dab-10kw --output reports/demo
aipe-magnetics schema validate <record.json> | migrate <v0.1.json> <v0.2.json> | export <dir> [--check]
aipe-magnetics import json <source.json> <destination.json> [--migrate]
aipe-magnetics mcp serve [--allow-write --output-dir DIR] | self-test
```

Exit codes: 0 ok, 2 rejected result, 1 error. `--json` prints the full envelope.
