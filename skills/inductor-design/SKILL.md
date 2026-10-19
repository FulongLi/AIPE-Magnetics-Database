---
name: inductor-design
description: Basic power inductor design with the AIPE Magnetics engine - turns, air gap, peak flux density, saturation screening, DC/AC copper loss and core loss for a given inductance, current and frequency on a selected core and material. Use when a user asks to size an inductor, choose turns or gap, check saturation, or estimate inductor losses. All calculations run in aipe-magnetics (design_inductor, validate_design, estimate_core_loss); the skill guides input collection and honest reporting of assumptions and limitations.
---

# Inductor design (basic analytical sizing)

Tools: `design_inductor`, `validate_design`, `estimate_core_loss`, `get_material`, `list_materials`,
`generate_report` (MCP) or `aipe-magnetics design inductor ...` (CLI).

## 1. Collect the specification (ask for what is missing - do not assume silently)

| Input | Unit | Notes |
| --- | --- | --- |
| inductance | H (uH) | target value at the operating point |
| current: peak and RMS (or DC + ripple) | A | peak sets saturation, RMS sets copper loss |
| ripple / AC flux swing | A or T | needed for core loss; `ripple_duty` (default 0.5, symmetric triangle) sets the ripple shape used by iGSE |
| switching frequency | Hz | |
| material and core | catalogue ids or user records | if the user has no core yet, run with `core: null` to evaluate every catalogue core offered in the material (feasible candidates ranked by total estimated loss) |
| maximum flux density policy | T or fraction of Bsat | default: 70 % of Bsat at the maximum temperature (the tool records it as an assumption) |
| maximum temperature | degC | selects Bsat(T) and Steinmetz set |
| winding: wire type, current density, fill factor | - | defaults are explicit assumptions in the result |

## 2. Run the design

Call `design_inductor` with the specification. The result contains, with units: turns, air gap
length (gapped-core energy method), peak flux density, saturation margin, window fill estimate, DC
resistance, AC resistance factor (Dowell), copper loss, core loss density and core loss, plus the
`assumptions`, `limitations`, `warnings`, `sources` and `verification` sections.

If the result status is `rejected`, report the error verbatim (e.g. the core saturates at the peak
current, or no Steinmetz parameter set covers the operating point) and propose concrete next steps
(larger core, lower flux limit, different material). Do not relax inputs on your own.

## 3. Validate and report

- Call `validate_design` on the returned design to run saturation screening and operating-point
  validation against the material/core limits and the model validity domains.
- Call `generate_report` for a Markdown/JSON engineering report when the user wants a document.
  Writing a file requires the user to give an output path and the server to be started with write
  permission (`--allow-write`); otherwise return the report text inline.

## 4. Communicate honestly

- Separate **calculated** values (turns, flux density from Ampere/Faraday relations) from
  **estimated** values (core loss from Steinmetz/iGSE parameters, AC resistance from Dowell's
  1-D model) and **assumed** parameters (fill factor, current density, flux limit).
- Say explicitly that no thermal, FEM, fringing-field or manufacturing verification has been run
  unless a tool result says it has.
- Quote the verification level/state as returned (`L0 unverified` with an explicit statement while no
  bundled evidence file is available). L2 means the loss-model equations were checked against analytical
  and independent references, not that the design was built and measured.
