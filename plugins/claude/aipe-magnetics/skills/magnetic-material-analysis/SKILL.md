---
name: magnetic-material-analysis
description: Look up, inspect and compare soft magnetic materials (ferrites, powder cores, nanocrystalline, amorphous, steels) from the AIPE Magnetics catalogue - permeability, saturation flux density versus temperature, Curie temperature, Steinmetz parameters, datasheet loss points - and rank candidates for a frequency / flux density / temperature operating point. Use when a user asks which material to use, what a material's parameters are, how two grades compare, or whether a datasheet value is available. Never estimate material data from memory; every number must come from the aipe-magnetics tools with its source reference.
---

# Magnetic material analysis

You have the `aipe-magnetics` engineering engine available either as MCP tools
(`list_materials`, `get_material`, `compare_materials`, `estimate_core_loss`, `verification_status`)
or as the CLI (`aipe-magnetics materials list|get|compare`, `aipe-magnetics loss estimate`).
All numbers you report must come from those calls. Do not compute core loss or interpolate datasheet
values yourself.

## Workflow

1. **Identify the materials.** Call `list_materials` (optionally with a query such as `N87` or a
   class such as `ferrite_mnzn`). If the user names a material that is not in the catalogue, say so
   explicitly and offer the closest catalogue entries; do not invent parameters. The user may import
   their own records (`aipe-magnetics import json <source.json> <destination.json> [--migrate]`) or supply a
   datasheet value with its source.
2. **Inspect parameters.** Call `get_material` for each candidate. Report values *with units and the
   stated conditions* (temperature, field strength, frequency) and with the evidence reference
   (datasheet title, table, review status). Distinguish `datasheet_typical`, `measurement`,
   `publication` and `synthetic` bases.
3. **Compare at the operating point.** Call `compare_materials` with the user's frequency, peak flux
   density and temperature. The tool returns, per material, the Steinmetz-based loss density (when a
   parameter set covers the operating point), the saturation margin, and the model's verification
   status. If a material has no parameter set covering the point, the tool reports it as *not
   available* - keep it that way in your answer.
4. **Explain the ranking** in engineering terms: loss density at the operating point, saturation
   headroom at the maximum temperature, Curie margin, frequency range, and any missing data. Always
   quote the `limitations` and `warnings` arrays of the result.
5. **State what is not verified.** Repeat the `verification` block of each result (level L0-L3, state
   `verified`, `partially_verified`, `out_of_range`, `unverified`, `failed`). An `out_of_range` status means the
   model was evaluated outside its verified operating range; `L0 unverified` with a statement naming a
   missing bundled evidence file means no evidence is available at all; say so.

## Rules

- Units: report SI plus the common engineering unit in parentheses (e.g. `0.39 T (390 mT)`,
  `375 kW/m^3`). Temperatures in degC with kelvin available on request.
- Do not round away significant digits before the final presentation.
- If the user asks for a value that the tools cannot provide (e.g. loss at a DC bias for a model that
  ignores DC bias), say that it is not available and why, and point to the data import path.
- Cite the datasheet/source reference ids returned by the tools.

## Example

> User: "Compare N87 and the synthetic example ferrite at 100 kHz, 200 mT, 100 degC."

Call `compare_materials(materials=["N87", "example-synthetic-ferrite"], frequency={"value":100,"unit":"kHz"},
flux_density_peak={"value":200,"unit":"mT"}, temperature={"value":100,"unit":"degC"})`, then present the
returned table, the saturation margins, the verification status and the limitations verbatim.
