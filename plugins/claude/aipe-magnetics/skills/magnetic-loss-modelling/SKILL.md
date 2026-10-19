---
name: magnetic-loss-modelling
description: Core-loss and hysteresis modelling with the AIPE Magnetics model framework - classical Steinmetz, iGSE for non-sinusoidal flux waveforms (triangular, trapezoidal, piecewise-linear, sampled), Bertotti loss separation with user coefficients, and Jiles-Atherton hysteresis loops via the ODE framework. Use when a user asks to estimate core loss for a waveform, choose a loss model, evaluate a B-H loop, or understand model validity ranges and verification status. Guides model selection and interpretation of validity domain and verification status; never computes losses outside the tools. Trained data-driven surrogates are reachable only through the Python adapter API, not through these tools.
---

# Magnetic loss modelling

Tools: `estimate_core_loss`, `evaluate_hysteresis`, `get_material`, `verification_status`,
`list_capabilities` (MCP) or `aipe-magnetics loss estimate`, `aipe-magnetics hysteresis evaluate` (CLI).

## Model selection guide

| Situation | Model (`model=`) | Notes |
| --- | --- | --- |
| sinusoidal flux, datasheet-type estimate | `steinmetz` | Pv = k f^α B^β; valid only inside the parameter set's f/B/T ranges |
| square-wave voltage (triangular flux), trapezoidal or PWL flux, no DC bias | `igse` | Venkatachalam et al. 2002; reduces exactly to Steinmetz for sinusoids; ignores DC bias and minor loops (flagged) |
| laminated / conductive materials, loss separation | `bertotti` | hysteresis + classical eddy + excess terms; requires `model_parameters` {k_h, beta_h, k_c, k_e} in SI (no catalogue material carries them); sinusoidal flux only |
| hysteresis loop shape, remanence, coercivity, loop energy | n/a - use the tool `evaluate_hysteresis` | Jiles-Atherton, scalar, rate-independent; parameters must be identified per material (an illustrative set is shipped, not a material fit) |
| trained surrogate (ANN/LSTM/Neural ODE) | not available through `estimate_core_loss` | data-driven inference runs only through the Python adapter API (`aipe_magnetics.models.datadriven`) on the shipped synthetic reference or user-provided weights; `model=` accepts exactly `auto`, `steinmetz`, `igse`, `bertotti` |

`model="auto"` picks Steinmetz for sinusoidal and iGSE for other waveforms when a parameter set
covers the operating point; otherwise the call is rejected with the reason.

## Workflow

1. Define the operating point: frequency, peak flux density (or peak-to-peak), temperature, waveform
   (`sinusoidal`, `triangular` with duty, `trapezoidal` with `duty_rise` - the symmetric trapezoid needs
   `2*(duty_rise + duty_flat) == 1`, so omit `duty_flat` to have it derived -, `piecewise_linear`
   breakpoints, or sampled B(t) from a user file: samples span one closed period `[0, T]` unless
   `endpoint_included: false` is passed for `[0, T)` data), DC bias if any.
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

- Quote the `verification` block as returned. `L0 unverified` with a statement naming a missing bundled
  evidence file means no evidence is available to the result at all; a verified model (L2) means the
  implementation reproduces analytical/independent references; neither validates the *parameters* for
  the user's specific core.
- Uncertainty: when the tool reports `unknown`, do not invent a tolerance band.
- Temperature: Steinmetz sets are temperature specific; a set is selected only if its temperature
  range contains the request. The tool will not silently use a 100 degC set at 25 degC.
