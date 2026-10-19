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
* Candidate cores: catalogue E and ETD cores offered in N87 with effective parameters from TDK
  datasheets (Ae, le, Ve, Amin, window area, mean turn length, mass).

## Steps performed (all through `aipe_magnetics.api`)

1. Material lookup (`get_material("N87")`).
2. Material parameter inspection (Bsat vs temperature, Steinmetz sets with validity ranges, Curie
   temperature, source locators, review status).
3. Core candidate selection (`design_transformer` with `cores=None` screens the catalogue by area
   product, then evaluates each candidate).
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
limit), and prints the verification status (L1/L2) per capability.
