# AIPE Magnetic Data Schema v0.2

Normative definition: `aipe_magnetics.data.records` (pydantic). Machine-readable schema:
`src/aipe_magnetics/schemas/v0.2/magnetic-record.schema.json` (JSON Schema 2020-12), mirrored
under `schemas/v0.2/`. Large numerical arrays use HDF5 (`aipe_magnetics.data.hdf5.WaveformStore`,
layout `aipe.magnetics.waveforms/0.2`), never JSON.

## 1. Record kinds

| Kind | Content | New in 0.2 |
| --- | --- | --- |
| `manufacturer` | identity of a material/core manufacturer | yes |
| `magnetic_material` | datasheet parameters: permeabilities, Bsat(T, H), remanence, coercivity, Curie temperature, density, resistivity, frequency range, Steinmetz parameter sets, loss reference points, temperature dependence, DC-bias points | extended |
| `core_material` | v0.1 kind, material as used in a core | kept |
| `core_family`, `core_geometry` | shape family; effective parameters Ae, le, Ve, Amin, Σl/A, window areas, mean turn length, mass, dimensions, gap options | extended |
| `winding`, `winding_geometry` | turns, conductor type/area, strands, foil dimensions, layers, material | extended |
| `inductor`, `transformer`, `magnetic_component` | component records: core/material/winding references, inductances, turns ratio, gap, ratings | extended |
| `operating_condition` | frequency, peak/DC flux density, DC field, temperatures, currents, voltages, duty, excitation reference | extended |
| `excitation_waveform` | parametric (sinusoidal, triangular, trapezoidal, square, piecewise-linear, DC) or sampled time-domain excitation with controlled quantity, timebase and sign convention | yes |
| `measurement` | v0.1 artifact-based measurement with instrument/method/specimen detail | extended |
| `core_loss_measurement` | table of (f, Bpk, T, DC bias, waveform) → Pv points on one specimen | yes |
| `bh_loop` | B-H hysteresis loop summary (peaks, remanence, coercivity, loop energy) pointing at sampled data | yes |
| `waveform_dataset` | description of an HDF5 container: arrays, units, sampling, timebase, conventions, checksums | yes |
| `dataset` | collection-level source, version, licence, citation, checksums | yes |
| `loss_model` | v0.1 kind | kept |
| `model_metadata` | model identity, family, architecture, parameters/weights artifact, inference configuration, validity, verification references | yes |

## 2. Envelope (every record)

```
schema_version  "0.2.0"
id              stable lowercase id, e.g. material.tdk-n87, core.tdk-e65-32-27
kind, name, synthetic (bool), description?
license         SPDX identifier/expression or NOASSERTION (record level)
properties      kind-specific object (validated; unknown fields rejected)
evidence_refs   ids into evidence (min 1)
evidence[]      {id, kind: assumption|source|measurement|analytical|simulation, reference, description,
                 created_at, review_status, license, locator?, url?, version?, sha256?, accessed_at?}
provenance?     {created_at?, activity?, tool{id,version}?, agent?, inputs[], source_commit?,
                 dataset_version?, normalization?}
quality?        {status: raw|normalized|curated|reviewed|rejected, completeness?, flags[], notes[]}
artifacts[]     {uri, media_type, role, sha256?, size_bytes?, description?}
dataset_ref?, tags[], extensions{ namespaced keys }
```

Semantic rules (enforced): unique evidence ids; every `evidence_ref` resolves and is not `rejected`;
non-synthetic records need at least one non-assumption evidence item; measurement-type records need
measurement evidence; cross-record references resolve to the right kind (`validate_catalogue`).

## 3. Explicit requirements

| Topic | Requirement |
| --- | --- |
| SI units | Stored quantities are coherent SI with canonical spellings: `T`, `A/m`, `Hz`, `K`, `W/m^3`, `J/m^3`, `m`, `m^2`, `m^3`, `kg`, `kg/m^3`, `Ohm*m`, `H`, `V`, `A`, `W`, `s`, `1`. Temperatures are kelvin (`degC` is converted). |
| Original units | A converted quantity keeps `original: {value, unit}`; arrays keep `original_unit` attributes in HDF5. |
| Data conversion | Conversions use the fixed factors of `aipe_magnetics.common.units` (exact rationals); `provenance.normalization` describes the applied normalisation; the importer records tool version and source checksum, so normalisation is reproducible. No unit inference: importers require an explicit column→unit mapping. |
| Sign conventions | B, H, voltage and current are positive along the stated winding reference direction (right-hand rule); a B-H loop traverses counter-clockwise in the (H, B) plane for a dissipative cycle so ∮H dB ≥ 0. Each waveform record states `sign_convention` / `reference_direction`. |
| Coordinate definitions | Scalar quantities along the magnetic path (`coordinate_definition` in waveform datasets); vector fields are out of scope for 0.2. |
| Sampling rate and timebase | `sampling_rate` (Hz) or explicit time arrays; `timebase` ∈ {`uniform_per_period`, `explicit_time_array`, `uniform_absolute`}; `samples_per_sequence`; periodic sequences start at t = 0 of one steady-state period. |
| Measurement conditions | `operating_condition` / per-sequence condition arrays: frequency, temperature, DC bias field/flux, duty, excitation. |
| Specimen geometry | `core_loss_measurement.specimen_core_ref` / `specimen_description`; `measurement.specimen`. |
| Manufacturer and material identifiers | `manufacturer` records and `manufacturer_ref`; `grade` for the manufacturer designation; ids are stable and lowercase. |
| Source references | `evidence` items with `reference`, `locator` (page/table/figure), `url`, `version`, `sha256`. |
| Dataset versions | `dataset.version`, `dataset.source_commit`, `provenance.dataset_version`. |
| Checksums | SHA-256 of original bytes (`artifacts[].sha256`, `dataset.checksums`, `extensions["aipe.import"].source_sha256`); HDF5 arrays carry per-array SHA-256 of their canonical little-endian float64 bytes. |
| Missing values | Omitted in JSON (never 0 or null); `NaN` in HDF5 arrays with `missing_value: "NaN"` declared. |
| Uncertainty | `Quantity.uncertainty` with `kind` ∈ {standard, expanded, bounds, unknown}; `unknown` is the default and is reported as unknown, never estimated. |
| Data quality | `quality.status` and `quality.flags`; `review_status` per evidence item. |
| Licence information | `license` on the record, on every evidence item and on `dataset`; `dataset.redistribution_permitted` is tri-state (true/false/unknown). |

## 4. Original data preservation

Original source bytes are never modified. Importers read them, record their SHA-256, write a
*new* normalized record/HDF5 file, and refuse to overwrite existing destinations. Raw downloads
live in the local cache (`.aipe-cache/` by default) with a manifest of URLs, pinned commits and
checksums; they are not committed.

## 5. Compatibility and migration

v0.1 records remain valid under the v0.1 schema. `aipe_magnetics.data.migrate` (CLI:
`aipe-magnetics schema migrate <in> <out>`) upgrades a v0.1 record to v0.2: it bumps the version,
keeps every field, derives the record licence from the referenced evidence when unambiguous (else
`NOASSERTION`), and records the migration in `provenance` and `extensions["aipe.migration"]`.
No physical value is added or guessed during migration.

## 6. HDF5 waveform layout `aipe.magnetics.waveforms/0.2`

```
/  attrs: layout_version, schema_version, created_by, tool_version, dataset_version, license,
          source_id, sign_convention, coordinate_definition, timebase, samples_per_sequence
/sequences/<quantity>   float64 [n_sequences, n_samples]  attrs: quantity, unit, original_unit, missing_value
/conditions/<name>      float64 [n_sequences]             attrs: unit, original_unit
/checksums              attrs: <path> -> sha256 of canonical bytes
```

Quantities: `flux_density` (T), `field` (A/m), `voltage` (V), `current` (A), optional `time` (s).
Conditions: `frequency` (Hz), `temperature` (K), `dc_bias_field` (A/m), `loss_density` (W/m^3) when
measured. The companion `waveform_dataset` JSON record describes the file and carries evidence,
licence and provenance.
