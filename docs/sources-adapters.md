# External source adapters

`aipe_magnetics.data.sources` turns the four audited upstream sources into v0.2 records without
bundling or inventing anything. Every adapter implements the same contract
(`aipe_magnetics.data.sources.base.SourceAdapter`):

| Method | Purpose |
| --- | --- |
| `discover(root=None, allow_download=False)` | what is available: files / materials / sizes in a user checkout, an installed package, the local cache, or upstream at the pinned commit (`SourceItem` list; `available=False` means a download would be needed) |
| `inspect(item)` | metadata without loading whole arrays where the container allows it: units, columns, shapes, counts, SHA-256 (and whether it equals the pinned file) |
| `extract(item, **filters)` | `ExtractedData`: numpy arrays with **explicit** units (as stated upstream), provenance, notes, quality flags |
| `validate(extracted)` | list of integrity problems (finite values, monotone time, unit ranges, consistent counts, checksum match); nothing is "fixed" silently |
| `to_records(extracted, destination_dir=..., dataset_id=None, allow_unpinned=False)` | writes **new** JSON records and HDF5 artifacts (never overwrites) and returns an `ImportResult` |
| `attribution()` | `SourceReference` list (repository, pinned commit, licence, citation) |
| `check_license(action)` | `read` / `cache` / `bundle` / `redistribute`; raises `LicenseRestrictionError` when not permitted |

Every produced record carries: source URL, SHA-256 of the original bytes, the licence on the record
and on every evidence item, `review_status: unreviewed`, `extensions["aipe.import"]` (importer,
normalisation, source digest, column mapping) and `extensions["aipe.sources"]` (adapter facts, unit
mapping, filters, pinning status). Large arrays go to HDF5, never to JSON. Records produced for the
same material by several imports share one deterministic identity record
(`material.<source>-<grade>`, grade only, no datasheet values) unless `material_ref=` points at a
curated material record that is present in `context=`. When `dataset_id="dataset.<token>"` is given,
every sibling record and artifact of that import is named from `<token>`; because the token becomes
part of file names it must match `[a-z0-9]+(-[a-z0-9]+)*` and must not be a Windows device name
(`InvalidInputError` before anything is written; the same rule applies to `file_prefix`).

**Pinning policy.** The pinned commit is asserted (`properties.version`, `properties.source_commit`,
`provenance.source_commit`, file evidence `version`) only when the bytes that were read are verified
to be the upstream bytes at that commit: a SHA-256 match with the digest recorded at ingest (Princeton
databases, MagNet-Hub fixtures), a git blob id match, or a checkout that `git` confirms to be the
pristine pinned commit (materialdatabase). `extensions["aipe.sources"]["pinned"]` records the
outcome: `true` (verified), `false` (verified **different** - only importable with
`allow_unpinned=True`; every record then carries the quality flag
`source_differs_from_pinned_commit`, `pinned_mismatch` with both digests, and the version label
`unpinned copy (sha256 <actual>)` instead of the commit) or `null` (not verifiable, e.g. a material
not packaged at the pinned commit or user-supplied MagNet Challenge files - the version label is then
`unverified copy (sha256 ...)` / `user-supplied files (sha256 ...)` and `source_commit` is omitted).
The audited commit itself stays documented as `extensions["aipe.sources"]["commit"]` and in the
repository evidence in every case.

**Raw artifacts.** `Artifact.uri` is "a URI or a path relative to the record file". For an original
source file it is the pinned raw URL (`raw.githubusercontent.com/.../<commit>/...`) when the bytes
are verified pinned, otherwise the POSIX path of the file relative to the destination directory (an
absolute `file:` URI only when no relative path exists, e.g. another Windows drive - reported in
`ImportResult.warnings`), exactly like the HDF5 waveform importer; `sha256` and `size_bytes` are
always those of the file as read.

Table-only data (scalar operating points, datasheet curves) does not fit the waveform layout, so
`TableStore` writes `aipe.magnetics.tables/0.2` with the same conventions as the waveform layout:

```
/  attrs: layout_version, schema_version, created_by_tool/version, dataset_version, license, source_id,
          checksum_method, missing_value "NaN", n_rows, description?, material_ref?, source_dataset_ref?
/conditions/<name>  float64 [n_rows]  attrs: quantity, unit (canonical SI), original_unit, missing_value, source_column?
/categories/<name>  int64 [n_rows]    attrs: categories (JSON list), missing_code -1, source_column?
/checksums          attrs: "conditions/<name>" / "categories/<name>" -> SHA-256 of canonical bytes
```

## Policy summary

| Source | Pinned commit | Licence | `check_license` | What is downloaded, and when |
| --- | --- | --- | --- | --- |
| Princeton MagNet (`princeton_magnet`) | `8a6892e91ac622e42f8b9182a6dfe9094ceea97a` | MIT | all four actions allowed | packaged `<material>_database.h5` from `raw.githubusercontent.com` at the pinned commit, **only** with `allow_download=True`, SHA-256 verified against the values recorded at ingest; website zips never by default (no stable URL or checksum upstream) |
| MagNet Challenge 2 (`magnet_challenge2`) | `ab4913ca63ebda1f8eab429c64d3e93f96204713` | NOASSERTION | `read` only | **nothing, ever**: `fetch()`, `download()` and `discover(allow_download=True)` raise `LicenseRestrictionError` with the manual path |
| Paderborn materialdatabase (`paderborn_materialdatabase`) | `04fff8e98ae3ddb72809223c42e8c939ed1073bb` | GPL-3.0-only | `read`, `cache` | a shallow `git` clone of exactly the pinned commit into the cache, only with `allow_download=True`; HEAD is verified; `bundle` / `redistribute` refused |
| Paderborn MagNet-Hub (`paderborn_magnethub`) | `1ed635d7bfac3c2aa024471af363beadc05f8932` | MIT | all four actions allowed | the two test fixtures from `raw.githubusercontent.com` at the pinned commit with recorded SHA-256, only with `allow_download=True`; model weights are never fetched by AIPE (install `mag-net-hub` yourself) |

## Local cache

`LocalCache(root=None)` uses, in this order, the `root` argument, the environment variable
`AIPE_MAGNETICS_CACHE`, or `.aipe-cache/` in the current directory. `manifest.json` records one entry
per URL (`url, path, filename, sha256, size, commit, license, source_id, retrieved_at` - the
timestamp only when the caller supplies one) and one per pinned checkout. `fetch(url,
expected_sha256, allow_download)` never opens the network unless `allow_download=True`
(`DownloadNotAllowedError`, code `download_not_allowed`), streams the download to a temporary file,
verifies the SHA-256 and deletes the file on a mismatch (`DataIntegrityError`), serves cached files
without re-downloading, and refuses `file://` URLs and URLs with embedded credentials - for the
initial URL **and for every redirect target** (`SafeRedirectHandler`; an `https` -> `http` downgrade
is refused too, and the final response URL is checked again). Cached file names that would be Windows
device names (`CON`, `NUL`, `COM1` ...) are prefixed with `file-`. `verify()` re-hashes every cached
file and, for every checkout, re-checks `HEAD` **and** that no tracked file is modified or deleted
(`git status --porcelain --untracked-files=no` with `--no-optional-locks`; untracked additions are
deliberately not detected). Cached files are never modified by the adapters (they always write new
files elsewhere). `clone_pinned(url, commit)` runs `git init`, `git fetch --depth 1 origin <commit>`,
`git checkout --detach FETCH_HEAD` with `GIT_TERMINAL_PROMPT=0` and nothing else, and serves an
existing cached checkout only after the same `checkout_problems()` verification passes
(`find_checkout()` is a plain manifest lookup without checks); nothing from a checkout is ever
imported or executed.

## Princeton MagNet

```python
from aipe_magnetics.data.sources import PrincetonMagnetAdapter, LocalCache

adapter = PrincetonMagnetAdapter(LocalCache())
items = adapter.discover("path/to/magnet-checkout")          # or discover() for an installed package / cache / remote list
item = next(i for i in items if i.material == "N87")
adapter.inspect(item)                                        # shape, columns, embedded text metadata, sha256 vs pinned
data = adapter.extract(item, temperature_degC=25.0, excitation="sinusoidal", dc_bias_range_am=(0, 0))
adapter.validate(data)                                       # []
result = adapter.to_records(data, destination_dir="out", max_json_points=200)
```

Discovery: (a) a checkout or any directory with `<material>_database.h5` (a directory without such
files is not an error: the cached / remote items are listed instead, each with an `extra["note"]`;
a non-existent directory raises `DataNotFoundError`); (b) the installed `magnet` package, located with
`importlib.util.find_spec` only - the package is never imported (it pins numpy 1.22 / pandas 1.2.4 /
torch 1.9 / streamlit); (c) remote items for the ten packaged files with their pinned URL and SHA-256
(`extract(item, allow_download=True)` or `fetch(item, allow_download=True)`). `inspect()` reports
`matches_pinned_commit`; `to_records()` refuses a file that differs from the pinned one unless
`allow_unpinned=True` (see the pinning policy above). Range filters, including a
`temperature_degC=(low, high)` pair, reject reversed bounds with `InvalidInputError`.

Exact unit mapping of the packaged scalar database (`COLUMN_MAP`):

| upstream column | AIPE column | unit upstream | stored |
| --- | --- | --- | --- |
| `Frequency` | `frequency` | Hz (Welch estimate, 10 Hz quantised) | Hz |
| `Flux_Density` | `flux_density_peak` | T, AC amplitude `(max B - min B)/2` | T |
| `DC_Bias` | `dc_bias_field` | A/m, `round(mean H)` | A/m |
| `Duty_P`, `Duty_N` | `duty_p`, `duty_n` | per-unit; `-1` = sinusoidal | `1`; the `-1` sentinel becomes **NaN** (not applicable), excitation kept as a categorical column |
| `Temperature` | `temperature` | degC (oil-bath set point 25/50/70/90) | K (original degC retained) |
| `Power_Loss` | `loss_density` | W/m^3 | W/m^3 |

Excitation: `Duty_P == -1` sinusoidal; `|Duty_P + Duty_N - 1| <= 1e-6` triangular; otherwise
trapezoidal (`d2 = d4 = (1 - d1 - d3)/2`). Filters (`temperature_degC`, `frequency_range_hz`,
`flux_density_range_t`, `dc_bias_range_am`, `excitation`) are expressed in the source units named in
the argument.

Records: `dataset.princeton-magnet-<material>` (MIT, citation of the three TPEL papers, checksums of
the source file and the table artifact), `material.princeton-magnet-<material>` (identity; manufacturer
as listed in upstream `constants.py`), `measurement.princeton-magnet-<material>-database` (specimen
from `info_core`, e.g. `N87 toroid R34.0X20.5X12.5; Ae 8.26e-05 m^2; le 0.08206 m; Ve 6.778e-06 m^3; N1 5; N2 5`,
`info_setup` as instrument text, `info_processing` as method, upstream MATLAB pipeline as tool
attribution) and `core-loss-measurement.princeton-magnet-<material>-subset` with at most
`max_json_points` (default 200) points: the sinusoidal zero-bias rows at the lowest temperature sorted
by frequency then flux density, evenly thinned; the full table (hundreds of thousands of rows) is the
HDF5 artifact `princeton-magnet-<material>-core-loss-table.h5`.

Website products (documented in `WEBSITE_PRODUCTS`, not downloaded by default): `<material>_cycle.zip`
(`b_cycle.csv` / `h_cycle.csv` rows x 1024, `sampling_time.csv` = `1/(f*1024)` s, `temperature.csv` degC,
`test_info.txt`) and `<material>_measurements.zip` (`voltage.csv` / `current.csv` rows x 10000 at 8 ns).
`import_cycle_zip(path, destination_dir=..., material=...)` imports a zip the user downloaded: B in T,
H in A/m, frequency from the sampling time and the row length (1024 or 128 samples accepted), degC -> K;
the zip's SHA-256 is recorded because the upstream product is unpinned. Pass
`website_zip_urls={"cycle": "https://.../{material}_cycle.zip"}` to the adapter and call
`fetch_website_zip(..., allow_download=True)` to let the cache fetch it.

The existing reader `princeton_magnet_db.read_magnet_database` / `find_database_files` /
`fit_steinmetz_by_temperature` is unchanged and reused.

## MagNet Challenge 2

```python
from aipe_magnetics.data.sources import MagnetChallenge2Adapter

adapter = MagnetChallenge2Adapter()
adapter.discover("path/with/downloaded/csvs")                 # groups <Material>_<k>_{B,H,T}.csv and testing files
result = adapter.import_user_files("N87_4_B.csv", "N87_4_H.csv", "N87_4_T.csv", material="N87",
                                   frequency_or_sampling_time=(62.5, "ns"),          # 16 MHz MagNetX data
                                   column_units={"B": "T", "H": "A/m", "T": "degC"},  # REQUIRED, never inferred
                                   destination_dir="out", nominal_level=4)
```

`column_units` must name all three files with dimensionally correct units (`UnitError` otherwise).
`frequency_or_sampling_time` is a `(value, unit)` pair: a sampling time (`s`-compatible) selects
`timebase="uniform_absolute"` with `sampling_rate = 1/Ts` (the MagNetX convention: rows are transient
records of 2016-32016 samples, 100 duty-cycle sub-cycles, no per-row frequency); a frequency
(`Hz`-compatible, scalar or one per row) selects `timebase="uniform_per_period"` and requires 128 or
1024 samples per row (MagNet Challenge 1 / tutorial convention) unless `allow_any_length=True`.
Per-row sampling times with `timebase="uniform_per_period"` give `f = 1/(Ts * n_samples)` (Princeton
cycle convention). Rows must be rectangular, B and T finite, H may hold NaN (padded / unknown part of
the test segments) and NaN stays NaN - never 0. `validate()` flags flux-density peaks above 2 T and
temperatures outside -60..250 degC (`TEMPERATURE_PLAUSIBLE_K`, the challenge's 25/50/70 degC with a
wide margin), which catches kelvin values declared as `degC` (298.15 "degC" = 571 K) and vice versa.
Records: `dataset.magnet-challenge2-<stem>`, `waveform-dataset.magnet-challenge2-<stem>`, identity
`material.magnet-challenge2-<material>`; all `license: NOASSERTION`, `redistribution_permitted:
false`, versioned `user-supplied files (sha256 ...)` with `pinned: null` (the pinned commit refers to
the documentation repository only) and raw artifacts relative to the destination directory. Formats
and locations (two Dropbox folders, `docs/README.pdf`, the MagNetX mirror) are in
`DESCRIPTOR.data_locations`; the refusal error names the manual path and the organisers' contact.

## Paderborn materialdatabase (GPL-3.0)

```python
from aipe_magnetics.data.sources import PaderbornMaterialDatabaseAdapter, LocalCache

adapter = PaderbornMaterialDatabaseAdapter(LocalCache())
items = adapter.discover("path/to/materialdatabase-checkout")   # or discover(allow_download=True) to clone the pinned commit
by_id = {i.item_id: i for i in items}
lea = adapter.extract(by_id["complex_permeability/LEA_MTB/N95.csv"])
adapter.to_records(lea, destination_dir="out")                  # table artifact + dataset + measurement (+ identity material)
info = adapter.extract(by_id["datasheet_curves/N87/general_information.csv"])
adapter.to_records(info, destination_dir="out")                 # magnetic_material record with ConditionedValues
```

Discovery: `complex_permeability/{LEA_MTB,MagNet,TDK_MDT,datasheet}/<Material>.csv`,
`complex_permittivity/LEA_MTB/<Material>.csv`, `datasheet_curves/<Material>/*.csv`. The `datasheet`
folder duplicates `TDK_MDT` (same rows, no `h_offset`, other float formatting) and is skipped unless
`include_duplicates=True`. Provenance per source (no per-row flag upstream): LEA_MTB measured on the
Paderborn test bench (measurement evidence, `measurement` record with probe codes); MagNet derived by
upstream from Princeton sinusoidal cycles assuming elliptical hysteresis (measurement evidence,
`measurement` record stating the derivation); TDK_MDT computed from TDK Magnetic Design Tool curves
(source evidence, no measurement record); `datasheet_curves` digitised from manufacturer datasheets.

Exact unit mappings (`UNIT_MAPS`; headers carry no units):

| file type | columns -> (AIPE column, unit) |
| --- | --- |
| `complex_permeability` | `probe` categorical; `f` -> frequency Hz; `T` -> temperature degC; `b` -> flux_density_peak T; `mu_real` -> permeability_real 1; `mu_imag` -> permeability_imag 1 (positive, `mu = mu' - j mu''`); `h_offset` -> dc_bias_field A/m |
| `complex_permittivity` | `probe`; `f` Hz; `T` degC; `eps_real` -> permittivity_real 1; `eps_imag` -> permittivity_imag 1 (positive) |
| `b_over_h_at_f_T` | `h` -> field A/m; `b` -> flux_density T; `T` degC; `f` Hz; `branch` categorical (`r`/`f`) |
| `mu_amplitude_over_b_at_T` | `b` T; `mu` -> amplitude_permeability 1; `T` degC |
| `p_v_over_{b_at_f_T,f_at_b_T,T_at_f_b}` | `b` T; `f` Hz; `T` degC; `p_v` -> loss_density, **unit must be confirmed**: `confirm_units={"p_v": "W/m^3"}` |
| `small_signal_mu_real/imag_over_f_at_T` | `f` Hz; `mu_real`/`mu_imag` 1 (no temperature column despite the name) |
| `small_signal_mu_initial_over_T` | `T` degC; `mu` -> initial_permeability 1 |
| `general_information` | initial permeability 1, Curie temperature degC -> K, density kg/m^3, resistivity Ohm*m, Bsat at 25 / 100 degC (T, with temperature condition), application and material class (MnZn -> `ferrite_mnzn`) |

`p_v` is W/m^3 according to the upstream code (`functions_TDK_MDT.py` converts mW/cm^3 x 1000;
`plot.py` divides by 1000 to label kW/m^3; magnitudes 7e2-7e6) but the CSV states nothing, so the
caller confirms it explicitly; the confirmed unit is recorded in provenance.

Flux-density plausibility: values above 2 T when read with the documented unit are rejected
(`DataIntegrityError`) instead of guessed. `unit_overrides={"b": "mT"}` accepts a whole column in
millitesla (needed for `N27`, `N30`, `T37` `b_over_h_at_f_T.csv`); `quarantine_implausible_rows=True`
drops rows of genuinely mixed-unit columns (`N87/p_v_over_f_at_b_T.csv` has 45 of 252 rows with
`b = 25`; `N87/mu_amplitude_over_b_at_T.csv` one row with `b = 125`) and rows with non-positive
frequency (the 4-field stray row of `3C95/b_over_h_at_f_T.csv`). Quarantine is **refused** (the error
says why and points to `unit_overrides`) unless the column really holds two unit populations: at
most 25 % of the rows implausible (`QUARANTINE_MAX_FRACTION`), the surviving rows at most 0.6 T
(`FLUX_DENSITY_FERRITE_MAX_T`; every flux-density column of the pinned repository peaks at 0.526 T)
and the smallest implausible value at least 10x the largest plausible one
(`QUARANTINE_MIN_SCALE_GAP`; the genuine files show 125x and 312x). A continuous millitesla curve
that merely crosses 2 "T" therefore cannot be turned into a "few bad rows" case. Every quarantine is
documented with row count, fraction, surviving maximum, quarantined minimum and scale gap in
`quality.notes`, `ExtractedData.metadata["quarantine"]` and `extensions["aipe.sources"]["quarantine"]`.
Other known anomalies reported by `validate()`: two negative `mu_imag` rows in `MagNet/N27.csv`
(warning), one `mu_real` outlier in `LEA_MTB/N49.csv`, `T37` B-H equals `N30` B-H, `3C95` B-H has an
empty `branch` column. `inspect()` reports the git blob SHA-1 of the LF-normalised bytes; for five
files it is compared with the value verified at the pinned commit (Windows checkouts are CRLF).

Pinning for this source: `discover(root)` asks `git` (read-only, `--no-optional-locks`) whether the
enclosing work tree is the pristine pinned commit and records the answer as
`extra["checkout_verified"]`; the cached clone is verified by `LocalCache.clone_pinned`. Records
assert the commit only when the checkout or the file's blob id was verified; otherwise they are
versioned `unverified copy (sha256 ...)`, and a verified blob mismatch is refused unless
`allow_unpinned=True`. `general_information.csv` always produces its own record
`material.paderborn-materialdatabase-<grade>-datasheet`; `material_ref=` is never used as that
record's id - it must name a material record present in `context=` and is recorded as
`extensions["aipe.sources"]["related_material_ref"]` (plus a quality note), so a GPL-licensed record
can never take over a curated id.

Every record, evidence item and HDF5 file produced from this source carries `GPL-3.0-only`;
`check_license("bundle")` and `check_license("redistribute")` raise `LicenseRestrictionError`
because the copyleft terms are incompatible with the NOASSERTION/MIT distribution of AIPE Magnetics;
`read` and `cache` are allowed. Never run the upstream `Data()` class against a cached copy: it
rewrites CSVs and writes `*_backup` files.

## Paderborn MagNet-Hub

```python
from aipe_magnetics.data.sources import MagnetHubAdapter, LocalCache

adapter = MagnetHubAdapter(LocalCache())
adapter.discover("path/to/mag-net-hub")                      # fixtures + model directories (opaque weights, names and sizes only)
adapter.discover()                                           # installed magnethub (find_spec only, never imported) + remote fixtures
result = adapter.import_test_fixture("path/to/mag-net-hub", "out")   # genuine measured data, MIT
out = adapter.run_model("N87", "paderborn", b_1024_in_tesla, (100e3, "Hz"), (25, "degC"))  # needs pip install mag-net-hub
```

Facts recorded from the audit: 15 loss-model materials (`3C90 3C92 3C94 3C95 3E6 3F4 77 78 79 ML95S N27 N30 N49 N87 T37`),
teams `paderborn` (1024 samples per period, HARDCORE residual dilated CNN, TorchScript) and `sydney`
(128 samples, MMINN state dicts); inputs B T, f Hz, T degC; outputs p W/m^3, h A/m; the Paderborn
graph reports `|oint H dB| f` while Sydney's trapezoid has no absolute value; the sequence models
(GRU8 `.eqx`, 15 materials incl. FEC007/FEC014) take no frequency and assume the 16 MHz training
sample rate; every sequence normaliser has `T_max = 70 degC`.

Fixture mapping (`tests/test_files/unit_test_data_ploss_at_450kWpm3.csv`, sha256 `b728a796...5047`):
`B_t_0..B_t_1023` -> flux_density T (one zero-mean period per row), `freq` -> frequency Hz, `temp` ->
temperature degC -> K, `ploss` -> loss_density W/m^3, `material` -> one waveform file and one
`waveform_dataset` record per material (`paderborn-magnethub-ploss-fixture-<material>.h5`), plus
`dataset.paderborn-magnethub-ploss-fixture` and identity material records. The measurement campaign is
not identified upstream (evidence kind `source`, note in the records). `SourceItem.sha256` of a local
fixture is its computed digest (the pinned digest is `extra["expected_sha256"]`); a fixture that
differs from the pinned file is refused by `to_records` / `import_test_fixture` unless
`allow_unpinned=True`. `run_model` delegates to
`aipe_magnetics.models.datadriven.magnethub_backend`; frequency and temperature must be `Quantity` or
`(value, unit)` pairs (bare numbers are rejected); it raises `UnsupportedOperationError` when
`magnethub` is not installed.

## Limitations found in the audit

* **Princeton MagNet.** `constants.material_steinmetz_param` is `(k_i, alpha, beta)` - the iGSE
  coefficient fitted on sinusoidal 25 degC zero-bias rows, not the classical `k` (for 3C94 the first
  entry equals the classical k; see `docs/sources.md`). Temperature is the oil-bath set point (core up
  to ~5 degC hotter); Bdc is not measured (only Hdc); one core per material; frequencies are quantised
  to 10 Hz and losses were cast to float32 upstream; N49 reaches 794 kHz, 3E6 has no DC-bias sweep.
  Single-cycle files hold 1024 samples per period while challenge/tutorial variants hold 128 (8x
  decimation); the website zips are unpinned with no published checksums.
* **MagNet Challenge 2.** No licence anywhere; nominal sequence lengths disagree between documents
  (8015 vs 8016, 5008 vs 4975) - always use the actual column count; data are mutable on Dropbox
  (Material C was replaced); padded test files mark unknown H as NaN.
* **Paderborn materialdatabase.** mT-valued B-H files (N27/N30/T37), mixed-unit columns in two N87
  curve files, empty branch column and a 4-field row in 3C95 B-H, T37 B-H copied from N30, no
  temperature in `small_signal_*_over_f_at_T`, unit-less `p_v` header, TDK_MDT "mu_imag" is a model
  output (fit from 25/100 degC curves), MagNet subsets assume elliptical loops with set-point
  temperatures and kHz-rounded frequencies; pyproject MIT classifier contradicts the GPL LICENSE.
* **MagNet-Hub.** The Paderborn loss model is phase sensitive (waveform classifier uses fixed sample
  indices and `|b[0]| < 0.1 max b` for sines): keep the upstream zero-crossing start convention; it
  needs exactly 1024 samples per period (Sydney 128) - the backend rejects other lengths instead of
  resampling silently; the sequence model needs 16 MHz input and the sequence fixture rows are
  multi-period with DC offsets; the 2023 challenge letters A-E have a different meaning than the 2025
  ones; model weights were trained including held-out test data (upstream changelog).

## Verification checks (L1, offline)

`aipe_magnetics.verification.checks.check_sources` registers `sources.descriptors-valid`,
`sources.cache-tamper-detected`, `sources.license-policy-enforced` and `sources.unit-mapping-explicit`
(capability `aipe.magnetics.data.sources`). They use temporary directories and an in-memory transport;
no network and no upstream checkout is touched.

## Examples and tests

```
python examples/import_princeton_magnet_db.py <checkout_or_data_dir> <out_dir> [--material N87] [--max-json-points 200] [--allow-unpinned]
python examples/import_magnet_hub_fixture.py <checkout_root> <out_dir> [--materials N87 3C90] [--allow-unpinned]
```

Both print whether the records assert the pinned commit (`pinned: True`) and the raw artifact URI.

Offline unit tests: `tests/data/test_sources_base.py`, `tests/data/test_sources_adapters.py`. The
external tests in the latter run only when `AIPE_MAGNET_CHECKOUT`, `AIPE_MAGNETHUB_CHECKOUT`,
`AIPE_MATERIALDATABASE_CHECKOUT`, `AIPE_MAGNETCHALLENGE2_CHECKOUT` point at checkouts of the pinned
commits, and the download / clone tests only with `AIPE_ALLOW_DOWNLOAD=1`; otherwise they skip with
exactly that reason.
