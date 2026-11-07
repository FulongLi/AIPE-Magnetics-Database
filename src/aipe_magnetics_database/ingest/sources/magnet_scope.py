"""Princeton MagNet raw oscilloscope captures (MIT), PrincetonUniversity/magnet @ ``8a6892e9``.

``scripts/data_processing_files_and_example/N87/R34.0X20.5X12.5/Dataset1`` holds two example acquisition runs of
the MagNet test bench on a TDK N87 R34.0X20.5X12.5 toroid (5:5 turns): ``25/0_1_1`` (100 sinusoidal captures,
no DC bias) and ``70/1_2_6`` (100 triangular captures, d0 = 0, dP = 0.6, Hdc = 15 A/m). Each capture is 10,000
samples of the secondary voltage [V] and primary current [A] at 8 ns (Tektronix DPO4054, record length 100,000
averaged to 10,000 in software), with the commanded set points (Hdc [A/m], d0, dP [per unit], f [Hz], B_peak [T])
in ``*_Parameters.csv`` and the specimen / bench description in ``Test_Info.xlsx``. Units are those of MagNet's
own processing scripts (``ReadRaw_1.m``: ``Volt % V``, ``Curr % A``, ``Time % s``).

The captures are written unchanged (float64 from the published decimal text) into HDF5 files of the
``aipe.magnetics.waveforms/0.2`` layout with a ``uniform_absolute`` 125 MHz timebase. The time CSV is printed to three
significant digits (consecutive samples repeat), so it is not stored as an axis; instead the importer verifies that
every one of its values equals ``k * 8 ns`` printed that way; nothing is filtered, integrated,
resampled or cut to a cycle (B and H are *not* derived). The current and time CSVs end with all-empty rows
(spreadsheet padding: 282 / 291 in the 25 degC run, 292 time rows in the 70 degC run); they are dropped and
counted - every non-empty row has exactly 10,000 values and the 100 time rows of each run are identical. The HDF5
files are large and live in ``waveforms-local/`` (not in Git; regenerated from the pinned raw files and published to
Hugging Face).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aipe_magnetics_database.common.errors import DataIntegrityError
from aipe_magnetics_database.ingest.acquire import RawFile, obtain
from aipe_magnetics_database.ingest.common import (
    ImportSummary,
    SourceWriter,
    dataset_record,
    evidence,
    finish_source,
    importer_attribution,
    sha256_path,
    validated,
)

SOURCE_ID = "princeton_magnet_scope"
DIRECTORY = "princeton-magnet-scope"
COMMIT = "8a6892e91ac622e42f8b9182a6dfe9094ceea97a"
COMMIT_TIME = "2026-05-12T00:00:00Z"
LICENSE = "MIT"
REPOSITORY = "https://github.com/PrincetonUniversity/magnet"
BASE = "scripts/data_processing_files_and_example/N87/R34.0X20.5X12.5/Dataset1"
CITATION = ("H. Li, D. Serrano, T. Guillod, S. Wang, E. Dogariu, A. Nadler, M. Luo, V. Bansal, N. K. Jha, Y. Chen, C. R. Sullivan, "
            "M. Chen, 'How MagNet: Machine Learning Framework for Modeling Power Magnetic Material Characteristics', IEEE TPEL, 2023; "
            "data: github.com/PrincetonUniversity/magnet (MIT)")
_FILES = {
    "25/0_1_1_Data_Curr.csv": ("c613cceeb39f458d56ef9643fdad1416bc859e7f15d7f31289be58f21b7c4ba2", 15203426),
    "25/0_1_1_Data_Time.csv": ("61af372393c0c935fbe42f2befa7c6de37d852fa123d3c9edfcaf2a3e2cd5219", 11909691),
    "25/0_1_1_Data_Volt.csv": ("c938b72923c656dd0b11128d8f9181251e85a7ad83eabece40e4b9791ef4ad13", 5913012),
    "25/0_1_1_Parameters.csv": ("824422ac58c2cb141380dc2db3e1ffb78a3b24522fef3abe52fb46c460bb5596", 3468),
    "70/1_2_6_Data_Curr.csv": ("32c2c2c6b496180c42c960644c9abbf7257134ca6f3351769212f17801b974e8", 12001402),
    "70/1_2_6_Data_Time.csv": ("eb1a013caced6dad5a008ac5c3eab02327a7974653e2f4a991a312330a133872", 11919692),
    "70/1_2_6_Data_Volt.csv": ("1c2338dc25f0df06dc2a892128e26bf74da5278ac2cf974b8fbf60e906eb8147", 5748830),
    "70/1_2_6_Parameters.csv": ("2a46daf04bb0713a40078ab4d878da5b963576da1677760dd19e8a241b20e954", 3279),
    "Test_Info.xlsx": ("209e434c89b2a58d00d9013089ae8c24a0ca89bd92b9d41abaf5627a3515ded6", 10440),
}
SAMPLE_INTERVAL = 8e-9  # s; proven against every value of the time CSV (see run)
RUNS = {"25/0_1_1": (25, "sinusoidal"), "70/1_2_6": (70, "triangular (d0 = 0, dP = 0.6)")}


def raw_files() -> list[RawFile]:
    return [RawFile(f"{BASE}/{name}", f"https://raw.githubusercontent.com/PrincetonUniversity/magnet/{COMMIT}/{BASE}/{name}", sha, size, LICENSE)
            for name, (sha, size) in _FILES.items()]


def _matrix(path: Path) -> tuple[np.ndarray, int]:
    """Rows of a comma-separated numeric file; all-empty trailing rows are dropped and counted (``nan`` stays NaN)."""
    rows, empty = [], 0
    for line in path.read_bytes().replace(b"\r\n", b"\n").split(b"\n"):
        if not line.strip():
            continue
        cells = line.split(b",")
        if all(not c.strip() for c in cells):
            empty += 1
            continue
        if any(not c.strip() for c in cells):
            raise DataIntegrityError(f"{path.name}: row with missing values among filled ones")
        if empty:
            raise DataIntegrityError(f"{path.name}: data row after empty padding rows")
        rows.append([float(c) for c in cells])
    return np.asarray(rows, dtype=np.float64), empty


def _test_info(path: Path) -> dict[str, Any]:
    from aipe_magnetics_database.ingest.readers import read_excel
    sheet = read_excel(path, path.name)[0]
    info = {}
    for row in [sheet.header, *sheet.rows]:
        if row and row[0] is not None:
            info[str(row[0]).strip()] = row[1]
    return info


def run(output_root: Path, *, local_dir: str | Path | None = None, allow_download: bool = False, cache: Any = None) -> ImportSummary:
    from aipe_magnetics_database.common.quantity import Quantity
    from aipe_magnetics_database.data.hdf5 import WaveformMeta, WaveformStore

    files = raw_files()
    paths = obtain(SOURCE_ID, files, local_dir=local_dir, allow_download=allow_download, cache=cache, commit=COMMIT)
    summary = ImportSummary(SOURCE_ID, COMMIT, LICENSE, "permitted (MIT; cite the MagNet TPEL papers)")
    summary.inputs = [{"path": f.name, "url": f.url, "sha256": f.sha256, "size": f.size} for f in files]
    writer = SourceWriter(output_root / DIRECTORY, summary)
    info = _test_info(paths[f"{BASE}/Test_Info.xlsx"])
    specimen = (f"{info.get('Material')} {info.get('Shape')} toroid, N1 = {info.get('N1')} (current), N2 = {info.get('N2')} (voltage); "
                f"Le = {info.get('Le (m)')} m, Ae = {info.get('Ae (m2)')} m^2, Ve = {info.get('Ve (m3)')} m^3 (Test_Info.xlsx)")
    bench = {k: info.get(k) for k in ("Date", "Place", "Setup trapezoidal info", "Setup sinusoidal info", "Setup DC bias info",
                                      "Setup temp info", "Measurement info", "Acquisition info") if info.get(k) is not None}

    def ev(locator: str, measured: bool = False) -> list[dict[str, Any]]:
        items = [evidence("evidence.magnet-scope-source", "source", "PrincetonUniversity/magnet", "MIT-licensed MagNet repository.", COMMIT_TIME,
                          LICENSE, url=f"{REPOSITORY}/tree/{COMMIT}/{BASE}", version=COMMIT, locator=locator)]
        if measured:
            items.append(evidence("evidence.magnet-scope-measurement", "measurement", "Princeton Power Electronics Research Lab (MagNet)",
                                  f"Oscilloscope capture of the MagNet bench ({info.get('Acquisition info', '')[:160]}).", COMMIT_TIME, LICENSE,
                                  url="https://mag-net.princeton.edu/", version=COMMIT, locator=locator))
        return items

    prov = {"activity": "normalize_princeton_magnet_scope", "source_commit": COMMIT, "tool": importer_attribution("magnet_scope"),
            "normalization": "CSV decimal text -> float64 unchanged (V, A as in MagNet's ReadRaw_1.m); all-empty padding rows dropped; "
                             "uniform 8 ns sampling (125 MHz) verified against every value of the 3-significant-digit time CSV, which "
                             "is kept only in raw/; "
                             "temperature from the run folder name (degC -> K); commanded set points from *_Parameters.csv; no filtering, "
                             "integration, resampling or cycle extraction."}
    mid = "material.princeton-magnet-scope-tdk-n87"
    material = validated({
        "schema_version": "0.2.0", "id": mid, "kind": "magnetic_material", "name": "TDK N87", "synthetic": False, "license": LICENSE,
        "properties": {"grade": "N87", "description": "N87 specimen of the MagNet raw oscilloscope example runs (Test_Info.xlsx)."},
        "evidence_refs": ["evidence.magnet-scope-source"], "evidence": ev(f"{BASE}/Test_Info.xlsx"), "dataset_ref": "dataset.princeton-magnet-scope",
        "quality": {"status": "normalized", "flags": ["identity_only", "third_party_unreviewed"]}, "provenance": prov,
        "extensions": {"aipe.data": {"data_class": "identity", "source_id": SOURCE_ID}, "aipe.princeton": {"manufacturer": "TDK"}}})
    store = WaveformStore()
    waveforms = []
    out_dir = output_root / DIRECTORY / "waveforms-local"
    out_dir.mkdir(parents=True, exist_ok=True)
    for run_key, (temp_c, excitation) in RUNS.items():
        mats = {}
        for kind in ("Volt", "Curr", "Time"):
            arr, empty = _matrix(paths[f"{BASE}/{run_key}_Data_{kind}.csv"])
            if empty:
                summary.exclude(f"{run_key}_Data_{kind}.csv", "all-empty padding row (no values)", empty)
            mats[kind] = arr
        params, _ = _matrix(paths[f"{BASE}/{run_key}_Parameters.csv"])
        n = {k: v.shape for k, v in mats.items()} | {"Parameters": params.shape}
        if not (mats["Volt"].shape == mats["Curr"].shape == mats["Time"].shape and params.shape == (mats["Volt"].shape[0], 5)):
            raise DataIntegrityError(f"{run_key}: inconsistent shapes {n}")
        # The time CSV is printed to 3 significant digits (1.02E-06 repeats), so it cannot serve as an increasing time axis.
        # It is used to *prove* the sampling grid instead: every value must equal k * 8 ns printed the same way.
        grid = np.arange(mats["Time"].shape[1]) * SAMPLE_INTERVAL
        printed = np.array([float(f"{v:.2E}") if v else 0.0 for v in grid])
        if not all(np.array_equal(row, printed) for row in mats["Time"]):
            raise DataIntegrityError(f"{run_key}: time column is not a uniform {SAMPLE_INTERVAL} s grid printed to 3 significant digits")
        tag = f"{temp_c}c-{run_key.split('/')[1].replace('_', '-')}"
        rid = f"waveform-dataset.princeton-magnet-scope-n87-{tag}"
        rel = f"waveforms-local/{rid.split('.', 1)[1]}.h5"
        meta = WaveformMeta(source_id=SOURCE_ID, dataset_version=COMMIT, license=LICENSE, timebase="uniform_absolute",
                            sampling_rate=Quantity.parse((125.0, "MHz"), name="sampling_rate"),
                            description=f"MagNet raw captures, N87 R34.0X20.5X12.5, {temp_c} degC, {excitation}; voltage on the "
                                        "secondary (N2), current in the primary (N1); conditions are commanded set points.",
                            material_ref=mid, source_dataset_ref="dataset.princeton-magnet-scope")
        conditions = {"frequency": (params[:, 3], "Hz"), "flux_density_peak": (params[:, 4], "T"), "dc_bias_field": (params[:, 0], "A/m"),
                      "temperature": (np.full(params.shape[0], float(temp_c)), "degC")}
        props = store.write(out_dir / Path(rel).name, {"voltage": (mats["Volt"], "V"), "current": (mats["Curr"], "A")},
                            conditions, meta=meta, overwrite=True)
        h5 = out_dir / Path(rel).name
        digest = sha256_path(h5)
        rec = validated({
            "schema_version": "0.2.0", "id": rid, "kind": "waveform_dataset", "name": f"MagNet N87 raw captures, {temp_c} degC, {excitation}",
            "synthetic": False, "license": LICENSE, "properties": json.loads(props.model_dump_json(exclude_none=True)),
            "evidence_refs": ["evidence.magnet-scope-source", "evidence.magnet-scope-measurement"],
            "evidence": ev(f"{BASE}/{run_key}_Data_*.csv", True), "dataset_ref": "dataset.princeton-magnet-scope",
            "artifacts": [{"uri": f"../{rel}", "media_type": "application/x-hdf5", "role": "waveform_data", "sha256": digest,
                           "size_bytes": h5.stat().st_size, "description": "AIPE waveform file (layout aipe.magnetics.waveforms/0.2)"}],
            "quality": {"status": "normalized", "flags": ["commanded_conditions", "raw_capture_unprocessed", "third_party_unreviewed"]},
            "provenance": prov | {"inputs": [mid]},
            "extensions": {"aipe.data": {"data_class": "experimental_measurement", "source_id": SOURCE_ID, "material_ref": mid,
                                         "underlying_source": f"princeton-magnet-scope:n87-{tag}"},
                           "aipe.magnet_scope": {"run": run_key, "excitation": excitation, "specimen": specimen, "bench": bench,
                                                 "commanded_parameters_columns": ["Hdc [A/m]", "d0 [1]", "dP [1]", "f [Hz]", "B_peak [T]"],
                                                 "commanded_parameters": [[None if np.isnan(v) else float(v) for v in row] for row in params],
                                                 "rows_file": rel, "rows_sha256": digest}}})
        waveforms.append(rec)
        summary.count("waveform_dataset")
        summary.rows("waveform_samples", int(mats["Volt"].size))
        summary.rows("waveform_sequences", int(mats["Volt"].shape[0]))
    summary.count("magnetic_material", 1)
    dataset = dataset_record(
        rid="dataset.princeton-magnet-scope", name="Princeton MagNet raw oscilloscope example runs", source_id=SOURCE_ID,
        title="MagNet raw oscilloscope captures, N87 R34.0X20.5X12.5 (data processing example)", license=LICENSE,
        url=f"{REPOSITORY}/tree/{COMMIT}/{BASE}", version=COMMIT, source_commit=COMMIT, publisher="Princeton University (Minjie Chen group)",
        redistribution=True, record_count=len(waveforms), citation=CITATION, checksums={k: v[0] for k, v in _FILES.items()}, materials=["TDK N87"],
        description="200 unprocessed voltage / current captures (10,000 samples at 8 ns each) with commanded set points; the only raw "
                    "waveforms MagNet publishes with a pinnable URL.", evidence_items=ev(f"{BASE}/Test_Info.xlsx"), provenance=prov)
    shard_files = {"materials": writer.shards([material], "materials", "materials", 100),
                   "waveforms": writer.shards(waveforms, "waveforms", "waveforms", 100)}
    writer.json("datasets/princeton-magnet-scope.json", dataset)
    return finish_source(writer, summary, shard_files, ["datasets/princeton-magnet-scope.json"])


__all__ = ["COMMIT", "DIRECTORY", "SOURCE_ID", "raw_files", "run"]
