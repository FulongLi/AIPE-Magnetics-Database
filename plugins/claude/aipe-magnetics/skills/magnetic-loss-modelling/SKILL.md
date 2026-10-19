---
name: magnetic-loss-modelling
description: Core-loss and hysteresis modelling with the AIPE Magnetics model framework - classical Steinmetz, iGSE for non-sinusoidal flux waveforms (triangular, trapezoidal, piecewise-linear, sampled), Bertotti loss separation, Jiles-Atherton hysteresis loops via the ODE framework, and data-driven surrogate inference. Use when a user asks to estimate core loss for a waveform, choose a loss model, evaluate a B-H loop, understand model validity ranges, or run a trained loss model. Guides model selection and interpretation of validity domain and verification status; never computes losses outside the tools.
---

# Magnetic loss modelling

Tools: `estimate_core_loss`, `evaluate_hysteresis`, `get_material`, `verification_status`,
`list_capabilities` (MCP) or `aipe-magnetics loss estimate`, `aipe-magnetics hysteresis evaluate` (CLI).

## Model selection guide

| Situation | Model (`model=`) | Notes |
| --- | --- | --- |
| sinusoidal flux, datasheet-type estimate | `steinmetz` | Pv = k f^α B^β; valid only inside the parameter set's f/B/T ranges |
| square-wave voltage (triangular flux), trapezoidal or PWL flux, no DC bias | `igse` | Venkatachalam et al. 2002; reduces exactly to Steinmetz for sinusoids; ignores DC bias and minor loops (flagged) |
| laminated / conductive materials, loss separation | `bertotti` | hysteresis + classical eddy + excess terms; requires the three coefficients |
| hysteresis loop shape, remanence, coercivity, loop energy | `jiles_atherton` via `evaluate_hysteresis` | scalar, rate-independent; parameters must be identified per material (an illustrative set is shipped, not a material fit) |
| trained surrogate (ANN/LSTM/Neural ODE) | `data_driven:<model-id>` | only the shipped synthetic reference and user-provided weights; verification status states what was checked |

`model="auto"` picks Steinmetz for sinusoidal and iGSE for other waveforms when a parameter set
covers the operating point; otherwise the call is rejected with the reason.

## Workflow

1. Define the operating point: frequency, peak flux density (or peak-to-peak), temperature, waveform
   (`sinusoidal`, `triangular` with duty, `trapezoidal` with rise/flat fractions, `piecewise_linear`
   breakpoints, or sampled B(t) from a user file), DC bias if any.
2. Call `estimate_core_loss`. Read `model` (identity, version), `values.loss_density` (W/m^3), and,
   when a core is given, `values.core_loss` (W). Report `assumptions`, `limitations`, `warnings`
   (e.g. `sampling_sensitivity`, `minor_loops_ignored`, `out_of_domain`) and the `verification` block.
3. For hysteresis, call `evaluate_hysteresis` with the field excitation (amplitude, frequency,
   periods to steady state) and optional parameters; report peak B, remanence, coercivity, loop energy
   density and the solver diagnostics (tolerances, steps). Loop energy must be non-negative; a
   negative value indicates a sign-convention error and must be reported as such.
4. If the user needs a model outside the verified ranges, present the result with its
   `out_of_range` status rather than hiding it, or decline when the policy rejects it.

## Interpretation rules

- A verified model (L2) means the implementation reproduces analytical/independent references; it
  does not validate the *parameters* for the user's specific core.
- Uncertainty: when the tool reports `unknown`, do not invent a tolerance band.
- Temperature: Steinmetz sets are temperature specific; a set is selected only if its temperature
  range contains the request. The tool will not silently use a 100 degC set at 25 degC.
