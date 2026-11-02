---
name: magnetic-data-lookup
description: Retrieve standardised magnetic material, core-geometry and measurement data (core loss, complex permeability, B-H curves, permittivity) from the AIPE Magnetics Database with full provenance - which materials exist, what a grade's datasheet values are, which measured loss tables cover a frequency / temperature / flux density / DC-bias / waveform condition, and where each number comes from. Use for data questions only; never estimate, interpolate or extrapolate magnetic data from memory. For loss prediction or component design use the separate AIPE Magnetics Design plugin.
---

# Magnetic data lookup

You have the AIPE Magnetics Database as MCP tools (`database_summary`, `list_sources`, `list_materials`,
`get_material`, `material_properties`, `list_cores`, `get_core`, `find_measurements`, `get_table`,
`provenance`) or as the CLI `aipe-magnetics-db`. Every number you report must come from those calls.

## Rules

1. **Units always.** Give conditions with units: `"100 kHz"`, `"90 degC"`, `"0.1 T..0.2 T"`. Results are SI
   (Hz, T, A/m, K, W/m^3); convert for the user only when asked and say that you converted.
2. **Keep data classes apart.** Each record has `data_class`: `experimental_measurement`,
   `manufacturer_datasheet`, `manufacturer_tool_export`, `derived_from_measurement` or `fitted_parameters`.
   Never present datasheet curves or fitted coefficients as measurements.
3. **Quote quantity definitions.** `flux_density_peak` is the AC amplitude (max-min)/2; loss is time-averaged
   W/m^3 unless the column is `energy_density_per_cycle` (J/m^3 per cycle). Read the column `definition`.
4. **Report provenance and flags.** Name the source, version and licence, and repeat quality flags such as
   `third_party_unreviewed`, `temperature_not_stated`, `curie_temperature_may_be_bound`, `unit_reinterpreted_mT`.
   Schema validation is not an experimental verification.
5. **Duplicates.** `find_measurements` hides re-publications of the same experiment (e.g. MAS copies of
   Princeton MagNet). Mention them only via `include_duplicates` when the user asks for every source.
6. **No modelling.** Do not compute Steinmetz losses, interpolate between rows or extrapolate outside the
   recorded ranges. Point to AIPE Magnetics Design for that.

## Typical workflow

1. `list_materials(query="N87")` -> `get_material("TDK N87")` for the canonical record and its member sources.
2. `find_measurements(material="TDK N87", quantity="core_loss", frequency="100 kHz", temperature="90 degC")`.
3. `get_table(record_id, frequency="95 kHz..105 kHz", waveform="sinusoidal", limit=50)`; MagNet frequencies are
   10 Hz Welch estimates (99,950 Hz), so query ranges rather than exact values.
4. `provenance(record_id)` before quoting a number in a report.
