# MAS normalised catalogue (first import batch)

This folder contains an initial data-only integration of the [MAS / OpenMagnetics catalogue](https://github.com/Power-Supply-Manufacturers-Association/MAS), pinned at commit \`c33ccfec8e00e156d632c9a52b02bc1cfdf714dc\` on 2026-11-02.

## Scope and counts

- \`materials/*.jsonl\`: 1,137 commercial magnetic-material records.
- \`shapes/*.jsonl\`: 2,294 standard core-shape records.
- \`datasets/*.json\`: two source records; \`index.json\`: machine-readable shard index.
- \`../../raw/mas/SOURCES.json\`: exact upstream Git commit, blob SHA-1 and licence metadata.

Every non-index JSON object validates against the existing **AIPE Magnetic Data Schema v0.2** (\`aipe_magnetics_database.data.records.MagneticRecord\`). JSONL is newline-delimited JSON: each line is one complete v0.2 JSON object, so large catalogues remain sharded rather than forcing a multi-GB JSON array.

## Scientific mapping and limitations

### Material records

Normalised fields: manufacturer-qualified name, grade, material class, certain initial-permeability points, temperature-dependent permeability points, saturation/remanence/coercivity conditions and stated frequency recommendation ranges. Source temperatures are converted from °C to K with original units retained. Source B and H values already in SI units are copied without modifying measured/quoted values.

**Not yet integrated as typed measurement tables:** all MAS complex permeability/loss curves, Steinmetz coefficients, thermal and electrical material characteristics, model fitting details and datasheet-specific boundary statements. They remain in the immutable upstream source. Do not interpret a source limit (e.g., ≥ Curie temperature) as an exact material measurement. Missing parameters are not filled, zeroed or guessed. These records are **not** a complete high-fidelity material-characterisation data set.

### Core shape records

Each record identifies the standard shape and its named dimension tolerance ranges in \`extensions["aipe.mas"]["dimensions_range_m"]\` using explicit \`minimum_m\` and \`maximum_m\`. No midpoints, effective magnetic dimensions (\`A_e\`, \`l_e\`, \`V_e\`) or parts have been fabricated. Three upstream rows state a minimum larger than the maximum (P 4.6/3.1 \`H\`, EC 120 \`T\`, U 30/25/16 \`D\`); those bounds are not swapped or corrected but kept verbatim in \`inconsistent_dimension_ranges_m\`, excluded from \`dimensions_range_m\`, and the record is flagged \`upstream_dimension_range_inverted\`. These shapes **must not** be used to compute transformer geometry without a suitable verified part specification.

### Source and quality

Every row carries the pinned upstream source commit, filename and line, third-party licence, source evidence, \`third_party_unreviewed\` flag and manufacturer information when available. Stable row IDs are disambiguated with source row numbers; do not confuse identity with independent verification.

The upstream catalogue is Apache-2.0 (copy in \`../../raw/mas/LICENSE.upstream.md\`); original manufacturer datasheets can have separate rights and are **not** copied. No original MAS source bytes are vendored here: they are retrievable at the pinned commit and identified by Git blob hashes. Before production use, check the manufacturer datasheet, licence and the measurement conditions.

## Validation

\`\`\`sh
python -m pytest -q tests/data/test_mas_catalogue.py
\`\`\`

The test checks all 3,433 schema records (including two dataset records), unique IDs, cross-record dataset references, declared totals and existence of explicit SI range units. It does not independently verify physical specifications.

## Next data batches

1. Bring the approved Princeton MagNet CC-BY-4.0 steady-state and transient data through the existing source adapters, without pushing large raw MATLAB archives into Git.
2. Add a full MAS field-mapping adapter for complex permeability, core loss, provenance and uncertainty (including upper/lower bounds); keep numerical arrays in appropriately sized table/waveform artifacts.
3. Add Paderborn and PSMA data behind explicit licence and measurement-condition checks, avoiding duplicate records derived from the same MagNet source.
4. Once data services are stable, move the existing \`engineering/\`, \`models/\` and design interfaces to the separate future **AIPE-Magnetics-Design** project. That extraction is outside this data-ingestion PR.
