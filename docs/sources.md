# External data sources, adapters and attribution

AIPE Magnetics does **not** redistribute third-party datasets. Each source below has a reusable
adapter in `aipe_magnetics.data.sources` that performs discovery, metadata inspection, extraction,
format conversion, unit normalisation, integrity validation (SHA-256), original-source attribution,
version tracking (pinned commit), licence / redistribution checks, local caching and explicit error
handling. Where the licence permits, the adapter downloads on demand from a pinned upstream commit;
otherwise it only reads a checkout or files the user supplies. Usage, exact unit mappings, cache
behaviour and the executable import examples (`examples/import_princeton_magnet_db.py`,
`examples/import_magnet_hub_fixture.py`) are documented in [sources-adapters.md](sources-adapters.md).
The Paderborn materialdatabase licence is recorded with the SPDX identifier `GPL-3.0-only`.

| Source | Pinned commit | Licence | What it provides | AIPE adapter policy |
| --- | --- | --- | --- | --- |
| [Princeton MagNet](https://github.com/PrincetonUniversity/magnet) | `8a6892e91ac622e42f8b9182a6dfe9094ceea97a` | MIT (code and packaged data; the website download page states MIT-style terms of access; citation of the TPEL "MagNet" papers is requested) | packaged scalar core-loss databases for 10 materials (`src/magnet/data/<material>_database.h5`: f, B amplitude, DC bias, duty cycles, temperature, Pv), Steinmetz/iGSE constants (`src/magnet/constants.py`), tutorial datasets, trained LSTM/Transformer weights; full single-cycle waveform dataset hosted externally | **on-demand, pinned, checksum-verified** read of the packaged HDF5 databases (`princeton_magnet_db.read_magnet_database`); waveform zips are fetched only when the user requests them; nothing is bundled except *derived* Steinmetz fits with full provenance |
| [MagNet Challenge 2](https://github.com/minjiechen/magnetchallenge-2) | `ab4913ca63ebda1f8eab429c64d3e93f96204713` | **no licence file** (all rights reserved by default); data hosted on external links | challenge rules, CSV data formats (B, H in SI, temperature in degC, 1024 or 128 samples per period), baseline notebooks | **reference and user-supplied import only**: the adapter documents the formats and imports files the user has obtained under the challenge terms; no download, no redistribution |
| [Paderborn LEA materialdatabase](https://github.com/upb-lea/materialdatabase) | `04fff8e98ae3ddb72809223c42e8c939ed1073bb` | GPL-3.0 | CSV complex permeability / permittivity versus f, T, B from LEA test bench, MagNet and TDK MDT; digitised datasheet curves (B-H, Pv, general information) for 3C95, 3F46, N27, N30, N49, N87, N95, T37 | **no code copied, no data bundled**: the adapter reads a user checkout (or an explicit on-demand clone into the cache) and converts CSV rows into v0.2 records marked with the GPL-3.0 licence; derived records inherit the licence notice |
| [Paderborn mag-net-hub](https://github.com/upb-lea/mag-net-hub) | `1ed635d7bfac3c2aa024471af363beadc05f8932` | MIT | trained loss models (Paderborn HARDCORE residual CNN, TorchScript; Sydney MMINet state dicts) for 15 materials; sequence (B→H) models; test fixtures | **optional model backend**: if the user installs `magnethub`, `aipe_magnetics.models.datadriven.magnethub_backend` wraps `LossModel(material, team)` with explicit unit mapping (B in T, 1024 samples/period for Paderborn, 128 for Sydney, f in Hz, T in degC → Pv in W/m^3, H in A/m); no weights are vendored |

## Unit and convention notes captured during the audit

* MagNet scalar databases: `Frequency` Hz, `Flux_Density` T (amplitude), `DC_Bias` A/m, `Duty_P`/`Duty_N`
  per-unit (`-1` = sinusoidal), `Temperature` degC, `Power_Loss` W/m^3; measured on one toroid per
  material (e.g. N87 R34.0×20.5×12.5, Ae 82.6 mm^2, le 82.06 mm, N1 = N2 = 5), oil bath, 2022 setup.
* MagNet single-cycle waveform files: 1024 samples per period (challenge/tutorial variants use 128 by
  8× decimation), B in T, H in A/m, sampling time in s, temperature in degC.
* mag-net-hub: Paderborn loss model expects exactly 1024 samples per period and is phase sensitive in
  its waveform classifier; Sydney expects 128; the sequence model assumes a 16 MHz sample rate.
  The Paderborn graph reports |∮H dB|·f while Sydney's trapezoid has no absolute value.
* materialdatabase: some B-H files store B in mT despite the convention (N27/N30/T37) and headers
  carry no units; the adapter requires an explicit column→unit mapping per file.

## Derived data bundled with this repository

`src/aipe_magnetics/data/derived/princeton-magnet-steinmetz-fits.json` — classical Steinmetz
parameter sets fitted by `benchmarks/fit_steinmetz_from_magnet.py` (ordinary least squares in log
space) on sinusoidal, zero-DC-bias rows of the MagNet scalar databases, per measured temperature
(25/50/70/90 degC), restricted to 50–500 kHz and 10–300 mT. Each set records the source file SHA-256,
the upstream commit, the fitted ranges and fit metrics (max / mean relative error). Cross-check
against the constants published upstream in `src/magnet/constants.py` (`material_steinmetz_param`,
documented there as `(k_i, alpha, beta)`): for **N87** and **3C90** our 25 degC fits reproduce α and
β to four significant digits and the derived iGSE coefficient k_i within 0.2 %; for **3C94** α and β
match but the upstream first entry equals our classical *k* (2.009 vs 2.0046), i.e. it appears to be
k rather than k_i; for the remaining materials α/β differ by a few percent, so upstream evidently used
different point selections. The bundled sets are therefore labelled as AIPE fits with their own
metrics, not as upstream values.

## Citation

Princeton MagNet: D. Serrano et al., "Why MagNet", IEEE TPEL 2023, doi:10.1109/TPEL.2023.3291084;
H. Li et al., "How MagNet", IEEE TPEL 2023, doi:10.1109/TPEL.2023.3309232; H. Li, D. Serrano, S. Wang,
M. Chen, "MagNet-AI", IEEE TPEL 2023, doi:10.1109/TPEL.2023.3309233.
Paderborn HARDCORE model: T. Förster et al., arXiv:2401.11488. Paderborn materialdatabase: LEA,
Paderborn University (GPL-3.0). MagNet Challenge 2: M. Chen et al., IEEE PELS MagNet Challenge 2025.
