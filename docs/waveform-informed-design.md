# Waveform-informed transformer design

`design_transformer_from_waveforms` (Python API, CLI `aipe-magnetics design waveform-transformer`, MCP tool of the
same name) designs a transformer from the converter's **actual** winding waveforms instead of the `P / V`
square-current approximation used by `design_transformer`. The waveforms come from a sampled record (AIPE DAB HDF5
layout, CSV, JSON) or are synthesised with the lossless single-phase-shift (SPS) DAB model of
`aipe_magnetics.design.transformers.dab`.

## Inputs

| Input | Where | Notes |
| --- | --- | --- |
| primary current, secondary current, transformer primary voltage, (magnetising current) | `waveforms.channels` -> dataset / column names | nothing is inferred from names; every channel carries an explicit unit (HDF5 `unit` attribute, CSV `units`, JSON `unit`) |
| steady-state window | `waveforms.window {t0, t1}` | the **last complete period** of the window is used; cycle-to-cycle RMS / peak spread is reported |
| switching frequency | `frequency` | period-detection hint (autocorrelation check + zero-crossing refinement); the detected mismatch is reported |
| turns-ratio target and tolerance | `turns_ratio_target`, `turns_ratio_tolerance` | `N2 = round(N1 / n_target)` |
| total series inductance | `series_inductance_total` or the HDF5 metadata (`spec.l_ext + l_lk_p + l_lk_s n^2`) | gives `external_inductance_required = L_total - L_leak` |
| materials, cores, offer policy | `materials`, `cores`, `material_offer_policy` | default: every catalogue material with Bsat data and a loss set covering the frequency; cores offered in the material |
| design temperature, flux limit, loss ceiling | `temperature`, `flux_limit`, `loss_density_target` | Bsat(T), iGSE parameter set (interpolated between fits; extrapolation flagged) |
| winding options | `windings`: conductors (litz / foil), strand diameters, foil thicknesses, current densities, arrangements (`P-S`, `P-S-P`, ...), fill and current-density limits | every combination is a candidate |
| thermal | `thermal`: forced (default 40 W/m^2/K, 40 degC ambient) or natural convection, radiation, hot-spot limits | recorded as an assumption |
| isolation | `isolation`: creepage margin, inter-winding insulation, working voltage / class (informational) | the insulation system is **not** designed |

## Computations per candidate (material, core, N1 : N2, conductor option, arrangement)

1. exact RMS, mean and Fourier harmonics (default 41) of each winding current on its piecewise-linear period;
2. flux linkage `lambda(t) = integral v_p dt` (per-period drift removed, zero mean) -> `B(t) = lambda / (N1 A_e)`,
   peak and maximum flux density, saturation margin against `Bsat(T)`;
3. minimum primary turns from the flux-limit policy (and the loss-density ceiling), `turn_steps` further counts;
4. iGSE core loss on the actual flux waveform at the design temperature (`LossEvaluator`, exact per segment);
5. layout: parametric E-core layers (cores with the E outline: E 65/32/27, E 70/33/32) or a generic layer stack on the
   coil-former data (ETD, PM); UU 93/152/30 has no coil former and is reported as not evaluable;
6. per-layer harmonic Dowell copper loss with the MMF profile of the arrangement, each winding with its own spectrum;
   litz bundles are expanded into `sqrt(n_strands)` strand sub-layers, foils are used as such; the 1-D litz
   proximity model (Sullivan) is reported as a cross-reference;
7. 1-D window-field leakage inductance referred to the primary, external inductance still required, magnetising
   inductance (reluctance model with a 10 um residual gap per leg; datasheet AL where recorded);
8. window fill (bare copper / coil-former window), current density, radial fit;
9. thermal estimate: E cores through the thermal network (core and winding nodes, hot spots, electrothermal fixed
   point); other outlines through a lumped surface estimate (labelled, optimistic for hot spots);
10. ranking of the feasible candidates by total loss at the design temperature; every violated constraint is named.

The result also carries `data.square_current_comparison` (same geometry: `P / V` RMS with Dowell at the fundamental
= the `design_transformer` method, the actual RMS at the fundamental, and the full harmonic evaluation; plus the
public `design_transformer` result on the selected core), `data.material_comparison` (every catalogue material in
the reference core with the selected turns: loss density under the actual flux waveform, Bsat, margin, minimum
turns, missing data, extrapolation flags) and `data.measurement_requests` (see `measurement_requests`).
`export_state=True` with a caller-supplied `created_at` adds the AIPE-Core Engineering State 0.1.0 of the selected
design whose analytical evidence lists the waveform source evidence (file path, SHA-256 artifact) as an input.

## Executed example: 10 kW DAB, 800 V : 400 V, 100 kHz, SPS, phi = 22.5 deg, L_total = 35 uH

Source: `runs/dab-l1-nominal-ic/waveforms.h5` (ngspice level-1 run, 24 cycles; window 200-240 us; last period
230-240 us; 490 breakpoints per period). Request: `turns_ratio_target 2`, N87 (the only catalogue material with Bsat
data **and** a 100 degC loss set that is offered in a catalogue core), design temperature 100 degC, flux limit
70 % Bsat, loss ceiling 300 kW/m^3, litz 0.1 mm strands at 4 A/mm^2, arrangements P-S and P-S-P, forced convection
40 W/m^2/K at 40 degC, hot-spot limits 120 degC (winding) / 110 degC (core).

Imported excitation: `i_L` 13.68 A rms / 14.3 A peak (trapezoidal), `i_s` 27.93 A rms, flux-linkage swing
4.000 mV*s per cycle (volt-second imbalance 7e-8 after drift removal), throughput power 10.00 kW, periodicity 0.99999.

Ranked feasible candidates (12 of 35 evaluated; all P-S-P - the P-S variants exceed the winding hot-spot limit):

| Rank | Core | Material | N1 : N2 | B_peak | Core loss | Copper loss | Total | L_leak | L_ext required | L_m | dT (hot spot) | Fill |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | E 70/33/32 | N87 | 20 : 10 | 0.146 T | 19.1 W | 15.9 W | 35.0 W | 1.55 uH | 33.45 uH | 3.91 mH | 63 K (network) | 0.30 |
| 2 | E 65/32/27 | N87 | 22 : 11 | 0.170 T | 19.7 W | 15.9 W | 35.6 W | 1.68 uH | 33.32 uH | 3.75 mH | 65 K (network) | 0.36 |
| 3 | PM 87/70 | N87 | 16 : 8 | 0.137 T | 22.1 W | 13.6 W | 35.7 W | 0.96 uH | 34.04 uH | 3.39 mH | 25 K (lumped surface, optimistic) | 0.17 |
| 4 | E 65/32/27 | N87 | 21 : 10 | 0.178 T | 21.5 W | 14.9 W | 36.4 W | 1.53 uH | 33.47 uH | 3.42 mH | 66 K (network) | 0.33 |

Comparison with the square-current approximation on the selected E 70/33/32, 20 : 10, P-S-P (identical geometry):

| Current model | Primary RMS | Copper loss (both windings) |
| --- | --- | --- |
| `P / V` square approximation, Dowell at the fundamental (`design_transformer` method) | 12.5 A | 12.9 W |
| actual RMS (13.68 A / 27.93 A), Dowell at the fundamental | 13.68 A | 15.8 W |
| actual waveform, per-layer harmonic Dowell (this capability) | 13.68 A | 15.9 W |

The public `design_transformer` on the same core selects 16 : 8 (B_pk 0.183 T, core 29.5 W, copper 15.3 W,
total 44.8 W, datasheet mean turn length for both windings). The documented v0.2 demo design (E 65/32/27, 21 : 10)
reappears here as rank 4 with the same core loss (21.5 W; same flux) and 14.9 W instead of 17.7 W of copper loss
(layer-resolved mean turn lengths instead of the 188 mm coil-former value outweigh the 9 % higher RMS and the
harmonics); it needs an external inductor of 33.5 uH and shows 66 K hot-spot rise under the assumed cooling.

Material comparison in E 70/33/32 with 20 turns: 3C94 (94 kW/m^3) and 3C90 (125 kW/m^3) show lower iGSE loss
densities than N87 (188 kW/m^3), **but** their only loss sets end at 90 degC, so the 100 degC values are flagged
`temperature_extrapolated`; N27, N30, N49, 3E6, 3F4, 77 and 78 are not evaluable (no saturation data), which the
measurement requests list as high-priority gaps.

## Limitations

* Analytical models only: 1-D Dowell / 1-D leakage field, iGSE on the volume-average flux with sinusoidal-fit
  parameters (DC bias, minor loops and relaxation not modelled), thermal network or lumped surface with generic class
  conductivities; no finite elements, no fringing, no insulation-system, mechanical or manufacturing analysis.
* Results hold for the supplied operating point; other voltages, loads, phase shifts or modulation schemes change the
  waveforms and must be evaluated separately. Imported waveforms are taken as given (one period, no smoothing); the
  idealisations of the source simulation carry over.
* The capability has no evidence record in the bundled verification registry yet, so results report `L0 unverified`
  for the sizing relations (the iGSE loss model keeps its own L2 status in `data.loss_model_verification`); the
  pytest suite covers the loader, the SPS model against the executed run, the Dowell / leakage building blocks and
  the end-to-end design. Recording evidence requires the maintainer's `verify --update-bundled` release step.
* No hardware was built or measured.

## Measurement requests

`measurement_requests` (CLI `aipe-magnetics measurement-requests`, MCP tool) assesses per material the catalogue
coverage of B-H data, core loss versus f / B / T, DC bias, saturation versus temperature, complex permeability,
winding AC resistance and thermal conductivity for the excitation and returns one structured request per gap
(schema `aipe.magnetics.measurement-request/0.3`) with test conditions (frequencies around the switching frequency and
its harmonics, flux densities around the design peak capped at 0.9 Bsat, temperatures including the design
temperature, waveforms, DC-bias fields) and a priority: `high` when the gap blocks the design (no saturation data or
no loss set at the design conditions), `medium` when it limits accuracy (non-sinusoidal loss validation, DC bias,
winding AC resistance, grade-specific thermal conductivity), `low` for data the current models do not consume.
