# Models

All models implement the Magnetic Model Contract (`aipe_magnetics.models.contract`): a stable
identity (id, implementation version, numerical-method version), a declared validity domain, a
`check_domain()` honouring the caller's policy (`reject` or `warn`), and a `ModelOutput` with values,
assumptions, limitations, warnings, sources and diagnostics. Models never clip inputs; impossible
inputs raise `InvalidInputError`, overflow or nonphysical results raise `NumericalError`.

| Model id | Family | Equation / method | Parameters | Verified (L2) against |
| --- | --- | --- | --- | --- |
| `aipe.magnetics.model.steinmetz` | empirical | `Pv = k f^α B̂^β` (Steinmetz 1892; sinusoidal flux) | `k` [W/m^3 with f in Hz, B in T], `α`, `β`, validity ranges, fit temperature | scaling laws, zero excitation, invalid inputs, overflow detection; parameter sets carry fit metrics |
| `aipe.magnetics.model.igse` | empirical | `Pv = (1/T)∫ k_i |dB/dt|^α (ΔB)^(β−α) dt`, `k_i = k / ((2π)^(α−1) ∫₀^{2π}|cos θ|^α 2^(β−α) dθ)` (Venkatachalam, Sullivan, Abdallah, Tacca 2002; Mühlethaler et al. 2012 eqs. 2-3) | same Steinmetz set; `k_i` by adaptive quadrature | reduces to Steinmetz for sinusoids (≤1e-9), triangular closed form `k_i f^α (ΔB)^β [D^(1−α)+(1−D)^(1−α)]`, sampled-waveform convergence with N, Venkatachalam's closed-form `k_i` approximation (≤1 %), minor-loop detection |
| `aipe.magnetics.model.bertotti` | physics | `Pv = k_h f B^β_h + k_c f² B² + k_e f^1.5 B^1.5` (Bertotti 1988); classical eddy coefficient `k_c = π²σd²/6` for thin laminations | `k_h`, `β_h`, `k_c`, `k_e` (user supplied) | scaling and limiting cases, invalid parameters, eddy coefficient versus direct computation |
| `aipe.magnetics.model.jiles_atherton` | physics (ODE) | scalar Jiles-Atherton: `He = H + αM`, `M_an = Ms[coth(He/a) − a/He]`, `dM_irr/dH = δ_M (M_an − M_irr)/(δk − α(M_an − M_irr))`, `M = M_irr + c(M_an − M_irr)` (Jiles & Atherton 1986; Jiles, Thoelke & Devine 1992; δ_M after Jiles 1992) integrated with the dynamic framework | `Ms`, `a`, `α`, `k`, `c` (SI); shipped set is **illustrative, not a material** | zero excitation, small-signal initial susceptibility `χ₀ = cMs/(3a) / (1 − αcMs/(3a))`, loop closure and dissipation (∮H dB ≥ 0), branch ordering, solver-tolerance convergence |
| `aipe.magnetics.model.ode_framework` | dynamic | `ODEModel` (state, parameters, initial state, rhs, constraints, outputs), `Excitation` (constant, sinusoidal, triangular, PWL, sampled, from waveform), `SolverConfig` (RK45/DOP853/Radau/BDF/LSODA, tolerances, step limits), `solve`, `convergence_study`, `periodic_steady_state`, energy helpers | - | linear RL closed form, observed convergence order, nonlinear-inductor power balance, explicit solver failure, constraints reported not clipped, steady-state detection |
| `aipe.magnetics.model.datadriven.reference-steinmetz-ann` | data-driven (ANN) | numpy MLP 2×8×8×1 (tanh) on `(log₁₀ f, log₁₀ B̂) → log₁₀ Pv`, weights JSON with SHA-256, normalisation statistics in the inference configuration | trained on a **synthetic** Steinmetz surface by `benchmarks/fit_reference_mlp.py` | reproduces the analytic surface within the recorded tolerance; tampered weights rejected; deterministic |
| data-driven adapters | data-driven | `ArchitectureSpec` (ann, lstm, transformer, neural_ode) / `WeightsBundle` / `InferenceConfig` separation; numpy backend (MLP, LSTM with PyTorch weight layout), Neural ODE (numpy vector field + scipy integration), optional torch backend (state dict / TorchScript), optional `magnethub` backend (Paderborn/Sydney MagNet Challenge models) | user supplied | LSTM vectorised vs step loop, Neural-ODE linear closed form, out-of-domain rejection, explicit error when an optional backend is missing |

## Declared validity and policy

Steinmetz/iGSE validity is the parameter set's frequency, flux-density and temperature window
(datasheet fits and MagNet fits are stored with their ranges and fit metrics). Out-of-range
requests are rejected by default or evaluated with an `out_of_domain` warning under
`policy="warn"`; the result's `verification` block then reports `out_of_range`. iGSE declares
DC bias as *ignored* and refuses waveforms with minor loops (policy `reject`) or flags them
(`warn`). Jiles-Atherton declares `requires_history=True`, supports DC offsets and is rate
independent (no eddy-current or excess loss).

## Parameter provenance

Steinmetz parameter sets in the catalogue carry `basis` (`datasheet_fit`, `measurement_fit`,
`publication`, `manufacturer_supplied`, `synthetic`), a `source_ref` evidence id, the fit
temperature, validity ranges and `fit_metrics`. See `docs/sources.md` for the MagNet-derived fits
and the TDK datasheet points used for N87.

## What is not claimed

No model here is a thermal, finite-element or manufacturing model. Jiles-Atherton parameters are
not identified for any catalogue material. Data-driven inference is demonstrated on a synthetic
reference only; running the Paderborn/Sydney challenge models requires the optional `magnethub`
package (upstream requires Python ≥ 3.13 with torch/jax/equinox, so it cannot be an install extra
for Python 3.10-3.12) and the user's own download of its weights. PyTorch is not installed in the
release environment; the torch backend's "present" branch was exercised by the reviewer in a
throwaway environment with torch 2.5.1+cpu.
