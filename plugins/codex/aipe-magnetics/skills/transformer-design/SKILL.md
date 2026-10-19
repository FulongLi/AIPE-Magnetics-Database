---
name: transformer-design
description: Basic high-frequency transformer design with the AIPE Magnetics engine (e.g. DAB, LLC, phase-shifted full bridge) - turns from volt-seconds, turns ratio, peak flux density, core candidate selection and comparison, saturation screening, core loss (Steinmetz / iGSE for square or trapezoidal flux), winding resistance and copper loss, and a traceable engineering report. Use when a user asks to size a transformer, pick a core, check flux density or estimate transformer losses. All numbers come from aipe-magnetics (design_transformer, validate_design, estimate_core_loss).
---

# Transformer design (basic analytical sizing)

Tools: `design_transformer`, `validate_design`, `estimate_core_loss`, `compare_materials`,
`get_material`, `generate_report` (MCP) or `aipe-magnetics design transformer ...` (CLI). The
repository ships a fully executable reference: `aipe-magnetics demo dab-10kw` (10 kW, 800 V to 400 V,
100 kHz dual-active-bridge transformer).

## 1. Specification to collect

| Input | Unit | Notes |
| --- | --- | --- |
| power | W | nominal throughput |
| primary voltage and waveform | V | square (full bridge, DAB), sinusoidal (LLC approx.), or trapezoidal with duty |
| secondary voltage(s) | V | gives the turns ratio |
| switching frequency | Hz | |
| duty / phase shift | - | for DAB the flux is triangular (square voltage); the tool derives the flux waveform |
| material, core candidates | ids | or `cores: null` to let the tool screen catalogue cores by area product |
| flux density policy | T or fraction | default 70 % of Bsat(T_max) *and* a loss-density target; both recorded as assumptions |
| temperature | degC | selects Bsat(T) and the Steinmetz set |
| winding: conductor type (foil, litz, round), current density, fill factor | - | explicit assumptions |

## 2. Run and interpret

`design_transformer` returns per candidate core: primary/secondary turns (integers), realised turns
ratio, peak flux density and saturation margin, core loss density and core loss (model identity and
version stated; iGSE for non-sinusoidal flux), winding DC/AC resistance and copper loss (Dowell 1-D
model), total estimated loss, window fill estimate, and a ranked comparison table in `data`.

Report *all* candidates the user asked about, including rejected ones with the reason (saturation,
window overflow, no parameter set at the operating point).

## 3. Validate and document

- `validate_design` re-checks saturation, operating point and model domains for the chosen design.
- `generate_report` creates a Markdown/JSON report that separates calculated values, model-based
  estimates and assumptions, lists sources (datasheets, parameter-set references) and the
  verification status per capability.

## 4. What this skill does not claim

- No finite-element leakage/fringing analysis, no thermal model, no insulation/creepage design, no
  manufacturing or hardware qualification. Say so whenever a user asks for "final" numbers.
- Core loss under DC bias or with minor loops is outside the Steinmetz/iGSE validity; the tool flags
  this - repeat the flag.
- Verification levels: results are typically L2 (equations verified against analytical references);
  they are not measured values.
