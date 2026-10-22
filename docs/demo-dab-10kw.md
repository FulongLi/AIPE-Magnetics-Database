# Demonstration: 10 kW DAB high-frequency transformer at 100 kHz

Run: `aipe-magnetics demo dab-10kw --output reports/demo` (or `python examples/dab_10kw_transformer/run_demo.py`).
The demonstration calls the public API functions only; no number is hard-coded in the report.

## Electrical operating point (specified inputs)

| Quantity | Value | Source |
| --- | --- | --- |
| Nominal power | 10 kW | task brief; identical to the AIPE-Core DAB example requirements |
| Primary DC-link voltage | 800 V | task brief / Core example |
| Secondary DC-link voltage | 400 V | task brief / Core example |
| Switching frequency | 100 kHz | task brief / Core example |
| Topology | dual active bridge, single-phase-shift modulation | assumption of the demonstration |
| Transformer primary voltage | ±800 V square wave, 50 % duty (full bridge) | follows from topology |
| Nominal turns ratio | 2 : 1 (800 V : 400 V) | unity voltage-conversion ratio at nominal point |
| Maximum core temperature | 100 degC (373.15 K) | assumption; selects Bsat(100 degC) and the 100 degC Steinmetz set |
| Preferred material | TDK/EPCOS N87 (MnZn ferrite) | task brief; datasheet-derived parameters with locators |

## Excitation waveform

With a square voltage of amplitude `V = 800 V` across `N1` turns, Faraday's law gives a
**triangular** flux density with peak

`B_pk = V / (4 f N1 A_e)`.

Core loss is therefore estimated with **iGSE** for a triangular flux waveform (50 % duty), which
reduces exactly to the Steinmetz equation for sinusoidal excitation and uses the same datasheet
parameter set. DC bias is zero in steady state (no DC flux component in a full-bridge DAB with
blocking/balanced control) - this is stated as an assumption.

## Design assumptions (explicit, reported in every result)

* Winding currents: first-order estimate `I_rms ≈ P / V` per winding (square current approximation);
  the real DAB current is trapezoidal and depends on the phase shift and series inductance, so copper
  loss is an **estimate** and is labelled as such.
* Flux density limit: the smaller of 70 % of Bsat(100 degC) and the flux density that meets a core
  loss density ceiling of 300 kW/m^3 (choice recorded as an assumption; both numbers appear in the
  report).
* Windings: copper, current density 4 A/mm^2, window fill factor 0.4, foil or litz as selected,
  Dowell 1-D AC resistance factor at 100 degC conductor temperature.
* Candidate cores: every catalogue core offered in N87 (E 65/32/27, E 70/33/32, ETD 59/31/22, PM 87/70,
  UU 93/152/30) with effective parameters from TDK datasheets (Ae, le, Ve, Amin, window area, mean turn
  length, mass). The UU core record has no coil former (no mean turn length / window area), so its copper
  loss and window fill cannot be evaluated; it is reported as *not feasible* with that reason, never
  ranked against complete totals.

## Steps performed (all through `aipe_magnetics.api`)

1. Material lookup (`get_material("N87")`).
2. Material parameter inspection (Bsat vs temperature, Steinmetz sets with validity ranges, Curie
   temperature, source locators, review status).
3. Core candidate enumeration (`design_transformer` with `cores=None` evaluates every catalogue core
   offered in N87; there is no area-product pre-screening - the window-fill check is the exact form of
   that constraint).
4. Basic turns calculation (Faraday, integer turns, realised turns ratio).
5. Flux density estimation per candidate.
6. Saturation screening (`validate_design`, margin against Bsat at 100 degC).
7. Core-loss estimation (iGSE, triangular flux, 100 degC set; model identity, version and validity
   domain reported).
8. Winding-loss estimation (DC resistance from mean turn length, Dowell AC factor, copper loss per
   winding).
9. Candidate comparison (ranked table: total estimated loss, flux density, fill factor, mass).
10. Engineering report generation (`generate_report`, Markdown + JSON) and export of the selected
    candidate as an AIPE-Core Engineering State fragment.

## What the demonstration does not claim

No hardware was built or measured. No thermal, finite-element, leakage-inductance, insulation or
manufacturing analysis was run. The report separates **calculated** values (turns, flux density from
Faraday's law), **model estimates** (core loss from iGSE with datasheet-fitted Steinmetz parameters,
copper loss from Dowell's model) and **assumptions** (currents, fill factor, current density, flux
limit), and prints the verification status of each capability as derived from the bundled evidence
registry: `L0 unverified` with a statement naming the missing file while the release evidence file
`src/aipe_magnetics/verification/evidence/v0.2.0.json` has not been recorded by the maintainer's release
step, L1/L2 per capability once it has. The zero-DC-bias assumption is recorded as `zero_dc_bias` in the
design result.

## Result summary of the release run

Output of the release run ([demo-dab-10kw-report.md](demo-dab-10kw-report.md), selected core E 65/32/27):

| Quantity | Value | Status |
| --- | --- | --- |
| turns primary : secondary | 21 : 10 | calculated (Faraday, integer turns) |
| peak flux density | 0.178 T (Bsat at 100 degC: 0.39 T, margin 54 %) | calculated |
| core loss density / core loss | 274 kW/m^3 / 21.5 W | estimated (iGSE, TDK 100 degC datasheet-fit set) |
| copper loss (litz, Dowell 1-D) | 17.7 W | estimated |
| total estimated loss | 39.2 W | estimated |
| window fill | 0.30 | estimated |
| verification | L2 (sizing relations and iGSE verified against analytical references) | - |

Other candidates: E 70/33/32 44.8 W, PM 87/70 44.9 W; ETD 59/31/22 rejected (window fill 0.51 > 0.40);
UU 93/152/30 not evaluable (no mean turn length / window area in the datasheet record). See
[release-verification-v0.2.0.md](release-verification-v0.2.0.md) for the record of what was executed for
the release. Values are analytical design estimates with stated assumptions; no hardware was built or measured.
