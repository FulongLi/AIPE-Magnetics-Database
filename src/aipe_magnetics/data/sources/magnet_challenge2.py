"""Adapter for the IEEE PELS **MagNet Challenge 2** (https://github.com/minjiechen/magnetchallenge-2).

Pinned commit ``ab4913ca63ebda1f8eab429c64d3e93f96204713``. The repository has **no licence file**
(SPDX ``NOASSERTION``; only ``tutorials/tutorial-2`` is MPL-2.0) and the MagNetX transient data are
hosted on external Dropbox shares without a stated data licence. Therefore this adapter

* never downloads anything (:meth:`MagnetChallenge2Adapter.fetch` / :meth:`download` raise
  :class:`~aipe_magnetics.common.errors.LicenseRestrictionError` naming the manual path),
* only imports files the user obtained under the challenge terms
  (:meth:`MagnetChallenge2Adapter.import_user_files`), with an **explicit** column -> unit mapping,
* marks every produced record ``license: NOASSERTION`` and ``redistribution_permitted: false``.

Formats documented upstream (``docs/README.pdf``, ``README.md`` lines 221-222, MATLAB templates):
training CSVs ``<Material>_<k>_{B,H,T}.csv`` (k = 1..7 = nominal 50/80/125/200/320/500/800 kHz),
rows = independent sequences sampled at a fixed 16 MHz (62.5 ns), nominal lengths 32016 / 20016 /
12816 / 8015 (8016 in the tools) / 5008 / 3216 / 2016 samples, temperatures 25/50/70 degC; final-test
CSVs ``<Material>_Testing_True_{B_seq,H_seq,T,Loss}.csv`` and ``<Material>_Testing_Padded_H_seq.csv``
(1000-step segments, unknown part of H = NaN). The MagNet Challenge 1 / tutorial single-period format
uses 1024 or 128 samples per period with a per-row frequency; both conventions are accepted here.

Unit facts: B in tesla, H in A/m, T in degC (Test/Webinar-6 slides; glyphs partially lost in the
PDFs, confirmed by value ranges). Missing H values stay ``NaN`` (never 0).
"""

from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from aipe_magnetics.common.errors import (
    DataIntegrityError,
    DataNotFoundError,
    InvalidInputError,
    LicenseRestrictionError,
    UnitError,
)
from aipe_magnetics.common.jsonio import canonical_sha256, sha256_file
from aipe_magnetics.common.units import units
from aipe_magnetics.data.hdf5 import WaveformMeta, WaveformStore
from aipe_magnetics.data.importers.base import IMPORT_EXTENSION, ImportResult, is_file_uri, source_artifact
from aipe_magnetics.data.records import MagneticRecord
from aipe_magnetics.data.sources.base import (
    SOURCES_EXTENSION,
    ExtractedData,
    SourceAdapter,
    SourceDescriptor,
    SourceItem,
    cross_validate,
    dataset_token,
    make_evidence,
    record_envelope,
    slug,
    source_uri,
    sources_extension,
    write_records,
)

SOURCE_ID = "magnet_challenge2"
REPOSITORY_URL = "https://github.com/minjiechen/magnetchallenge-2"
COMMIT = "ab4913ca63ebda1f8eab429c64d3e93f96204713"
COMMIT_DATE = "2026-08-03T03:06:52-07:00"
LICENSE = "NOASSERTION"
CONTACT = "pelsmagnet@gmail.com (README.md lines 104 and 135)"
TRAINING_DATA_URL = "https://www.dropbox.com/scl/fo/o07x7fa6d5elev19pzd1w/AISuBEWf8LQVszFeoaTlqY4?rlkey=8ec7mqukfr9no61h193zb4qdb&dl=0"
FINAL_EVALUATION_URL = "https://www.dropbox.com/scl/fo/en5z0mef5cub7q574cknf/AA7moyzEJDOC398WwhvLzJs?rlkey=n9r8jqk6azh461tmxy1ixr3vm&dl=0"
CHALLENGE1_URL = "https://www.princeton.edu/~minjie/magnet.html"
MAGNETX_MIRROR_URL = "https://github.com/PaulShuk/MagNetX"
DATA_LOCATIONS = [
    f"{TRAINING_DATA_URL} ('MagNet Challenge 2 Data': training CSVs per material and frequency level; subfolder "
    "MagNet_Challenge_2_Pretest holds <Material>_Testing_{True,Padded}.h5; README.md lines 115/147/213)",
    f"{FINAL_EVALUATION_URL} (final evaluation data for materials A-E and all submitted results; README.md lines 5/30)",
    f"{REPOSITORY_URL}/blob/{COMMIT}/docs/README.pdf (MagNetX training-data format specification)",
    f"{REPOSITORY_URL}/tree/{COMMIT}/tutorials/tutorial-2/data (tiny N87 level-5 extract, 128 x 5008, MPL-2.0 folder)",
    f"{CHALLENGE1_URL} (MagNet Challenge 1 steady-state data, 128 samples per period)",
    f"{MAGNETX_MIRROR_URL} (MagNetX mirror named in the slides; not audited)",
]
MANUAL_PATH = ("Download the CSV / HDF5 files yourself from the Dropbox folders linked in the upstream README under the challenge "
               "terms, keep them outside this package, and import them with MagnetChallenge2Adapter.import_user_files(b_csv, h_csv, "
               "t_csv, material=..., frequency_or_sampling_time=(62.5, 'ns'), column_units={'B': 'T', 'H': 'A/m', 'T': 'degC'}). "
               f"For a data licence statement contact {CONTACT}.")
CITATION = ("M. Chen et al., 'MagNet Challenge for Data-Driven Power Magnetics Modeling', IEEE Open J. Power Electron., 2024, "
            "doi:10.1109/OJPEL.2024.3469916; IEEE PELS MagNet Challenge 2 (MagNetX database), Princeton University and Dartmouth College")
SAMPLING_RATE_HZ = 16e6
#: frequency level k -> (nominal frequency Hz, nominal sequence lengths documented upstream)
NOMINAL_LEVELS: dict[int, tuple[float, tuple[int, ...]]] = {
    1: (50e3, (32016,)), 2: (80e3, (20016,)), 3: (125e3, (12816,)), 4: (200e3, (8015, 8016)), 5: (320e3, (5008, 4975)),
    6: (500e3, (3216,)), 7: (800e3, (2016,)),
}
TEMPERATURES_DEGC: tuple[float, ...] = (25.0, 50.0, 70.0)
#: plausibility window for declared temperatures: the challenge range (25..70 degC) with a wide margin (-60..250 degC) so that
#: kelvin values declared as degC (298.15 "degC" = 571 K) are caught
TEMPERATURE_PLAUSIBLE_K: tuple[float, float] = (213.15, 523.15)
MATERIALS: tuple[str, ...] = ("3C90", "3C94", "3E6", "3F4", "77", "78", "N27", "N30", "N49", "N87", "A", "B", "C", "D", "E")
#: core shapes per material as read from Pretest/Pretest_Save_Template.m lines 15-24
CORE_SHAPES: dict[str, str] = {"3C90": "TX-25-15-10", "N87": "R34.0X20.5X12.5", "3E6": "TX-22-14-6.4", "3C94": "TX-20-10-7",
                               "3F4": "E-32-6-20-R", "77": "0014", "78": "0076", "N27": "R20.0X10.0X7.0", "N30": "22.1X13.7X6.35",
                               "N49": "R16.0X9.6X6.3"}
ACCEPTED_SAMPLES_PER_PERIOD: tuple[int, ...] = (128, 1024)
#: required keys of the ``column_units`` mapping and their SI targets
COLUMN_KEYS: dict[str, tuple[str, str]] = {"B": ("flux_density", "T"), "H": ("field", "A/m"), "T": ("temperature", "K")}
_TRAIN_RE = re.compile(r"^(?P<material>[A-Za-z0-9]+)_(?P<level>[1-7])_(?P<kind>[BHT])\.csv(\.gz)?$")
_TEST_RE = re.compile(r"^(?P<material>[A-Za-z0-9]+)_Testing_(?P<variant>True|Padded)_(?P<kind>B_seq|H_seq|T|Loss)\.csv(\.gz)?$")

DESCRIPTOR = SourceDescriptor(
    source_id=SOURCE_ID, title="IEEE PELS MagNet Challenge 2 (MagNetX transient B-H database)", repository_url=REPOSITORY_URL,
    pinned_commit=COMMIT, pinned_commit_date=COMMIT_DATE, license=LICENSE, redistribution_permitted=False, citation=CITATION,
    publisher="IEEE PELS MagNet Challenge organisers, Princeton University (M. Chen, S. Wang, H. Kwon, H. Li) and Dartmouth College (T. Guillod, C. R. Sullivan)",
    data_locations=DATA_LOCATIONS,
    notes=["No LICENSE file at the repository root and no data licence on the Dropbox shares: all rights reserved by default; "
           "only tutorials/tutorial-2 is MPL-2.0 (code) and its slides CC BY-ND 4.0.",
           "The adapter never downloads or redistributes; it imports user-supplied files with explicit units only.",
           "Transient data: 16 MHz sampling (62.5 ns), 7 frequency levels encoded only by the file index k, 100 duty-cycle "
           "sub-cycles per sequence, temperatures 25/50/70 degC; nominal lengths differ between documents (8015 vs 8016, 5008 vs 4975) "
           "- always read the actual column count.",
           "Padded test files mark unknown H samples as NaN; AIPE keeps them NaN (never 0).",
           MANUAL_PATH])


def _load_numeric_csv(path: Path, *, delimiter: str, missing_allowed: bool, label: str) -> np.ndarray:
    """Fast path ``np.loadtxt`` (literal 'nan' accepted), fallback ``np.genfromtxt`` for empty cells; ragged rows rejected."""
    try:
        array = np.loadtxt(path, delimiter=delimiter, ndmin=2, dtype=np.float64)
    except ValueError:
        try:
            array = np.genfromtxt(path, delimiter=delimiter, dtype=np.float64, missing_values=["", "NA", "NaN", "nan", "NAN"],
                                  filling_values=np.nan, ndmin=2)
        except ValueError as error:
            raise DataIntegrityError(f"{label}: cannot parse {path.name} as a rectangular numeric CSV: {str(error)[:300]}",
                                     details={"path": str(path)}) from None
    if array.size == 0:
        raise DataIntegrityError(f"{label}: {path.name} holds no numeric data", details={"path": str(path)})
    if np.isinf(array).any():
        raise DataIntegrityError(f"{label}: {path.name} contains infinite values", details={"path": str(path)})
    if not missing_allowed and np.isnan(array).any():
        raise DataIntegrityError(f"{label}: {path.name} contains {int(np.isnan(array).sum())} missing/NaN cells where none are allowed",
                                 details={"path": str(path)})
    return array


def validate_column_units(column_units: Mapping[str, str] | None) -> dict[str, str]:
    """The explicit ``{'B': 'T', 'H': 'A/m', 'T': 'degC'}`` mapping is REQUIRED; units must be dimensionally right."""
    if not isinstance(column_units, Mapping) or not column_units:
        raise InvalidInputError("column_units is required: an explicit mapping {'B': <unit of B>, 'H': <unit of H>, 'T': <unit of T>}; "
                                "units are never inferred from the files", details={"required_keys": list(COLUMN_KEYS)})
    unknown = sorted(set(column_units) - set(COLUMN_KEYS))
    if unknown:
        raise InvalidInputError(f"column_units has unknown keys {unknown}; expected exactly {list(COLUMN_KEYS)}")
    missing = [k for k in COLUMN_KEYS if k not in column_units]
    if missing:
        raise InvalidInputError(f"column_units lacks {missing}; every file needs an explicit unit", details={"missing": missing})
    out: dict[str, str] = {}
    for key, (quantity, si_unit) in COLUMN_KEYS.items():
        unit = column_units[key]
        if not isinstance(unit, str):
            raise InvalidInputError(f"column_units[{key!r}] must be a unit string")
        if not units.compatible(unit, si_unit):  # raises UnitError for unknown units
            raise UnitError(f"column_units[{key!r}] = {unit!r} is not a unit of {quantity} (expected something compatible with {si_unit!r})",
                            details={"key": key, "unit": unit, "quantity": quantity})
        out[key] = unit
    return out


class MagnetChallenge2Adapter(SourceAdapter):
    """Reference-only source: documents formats, imports user-supplied files, never downloads."""

    descriptor = DESCRIPTOR
    importer_id = "aipe.magnetics.sources.magnet_challenge2"
    _license_policy = {"read": True, "cache": False, "bundle": False, "redistribute": False}

    # -- licence-driven refusals -----------------------------------------------------------------------
    def refuse_download(self, what: str = "the MagNet Challenge 2 data") -> None:
        raise LicenseRestrictionError(
            f"{what} cannot be downloaded by AIPE Magnetics: the upstream repository {REPOSITORY_URL} has no licence file and the "
            "Dropbox-hosted dataset states no data licence (all rights reserved by default). " + MANUAL_PATH,
            details={"source_id": SOURCE_ID, "license": LICENSE, "data_locations": DATA_LOCATIONS, "manual_path": MANUAL_PATH,
                     "contact": CONTACT})

    def fetch(self, *args: Any, **kwargs: Any) -> Path:
        self.refuse_download()
        raise AssertionError("unreachable")  # pragma: no cover

    def download(self, *args: Any, **kwargs: Any) -> Path:
        self.refuse_download()
        raise AssertionError("unreachable")  # pragma: no cover

    # -- discovery ------------------------------------------------------------------------------------
    def discover(self, root: str | Path | None = None, allow_download: bool = False, **options: Any) -> list[SourceItem]:
        if allow_download:
            self.refuse_download()
        if root is None:
            return [SourceItem(source_id=SOURCE_ID, item_id=f"remote-{i}", kind="remote", url=location.split(" ", 1)[0], available=False,
                               license=LICENSE, description=location, extra={"manual_download_required": True})
                    for i, location in enumerate(DATA_LOCATIONS, start=1)]
        root = Path(root)
        if not root.is_dir():
            raise DataNotFoundError(f"directory not found: {root}")
        groups: dict[tuple[str, str, str], dict[str, Path]] = {}
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            match = _TRAIN_RE.match(path.name)
            if match:
                key = (match["material"], "training", f"level-{match['level']}")
                groups.setdefault(key, {})[match["kind"]] = path
                continue
            match = _TEST_RE.match(path.name)
            if match:
                key = (match["material"], "testing", match["variant"].lower())
                groups.setdefault(key, {})[match["kind"]] = path
                continue
            if path.suffix.lower() in (".h5", ".hdf5") and "_Testing_" in path.name:
                material = path.name.split("_Testing_")[0]
                variant = "padded" if "Padded" in path.name else "true"
                groups.setdefault((material, "testing-hdf5", variant), {})["h5"] = path
        items: list[SourceItem] = []
        for (material, family, detail), files in sorted(groups.items()):
            size = sum(p.stat().st_size for p in files.values())
            extra: dict[str, Any] = {"family": family, "detail": detail, "files": {k: str(p) for k, p in sorted(files.items())},
                                     "core_shape": CORE_SHAPES.get(material)}
            if family == "training":
                level = int(detail.split("-")[1])
                extra["nominal_frequency_hz"], extra["nominal_lengths"] = NOMINAL_LEVELS[level]
                extra["complete_triplet"] = set(files) == {"B", "H", "T"}
            items.append(SourceItem(source_id=SOURCE_ID, item_id=f"{material}/{family}/{detail}", kind="file", material=material,
                                    path=str(next(iter(files.values())).parent), size_bytes=size, available=True, license=LICENSE,
                                    description=f"user-supplied MagNet Challenge 2 {family} files for {material} ({detail})", extra=extra))
        return items

    # -- inspection -----------------------------------------------------------------------------------
    def inspect(self, item: SourceItem, **options: Any) -> dict[str, Any]:
        files = {k: Path(p) for k, p in item.extra.get("files", {}).items()}
        if not files:
            return {"item_id": item.item_id, "available": item.available, "description": item.description,
                    "manual_path": MANUAL_PATH, "units": {"B": "T", "H": "A/m", "T": "degC"}, "sampling_rate_hz": SAMPLING_RATE_HZ}
        info: dict[str, Any] = {"item_id": item.item_id, "material": item.material, "files": {}, "units_documented_upstream":
                                {"B": "T", "H": "A/m", "T": "degC"}, "sampling_rate_hz": SAMPLING_RATE_HZ, "extra": dict(item.extra)}
        for key, path in sorted(files.items()):
            if path.suffix.lower() in (".h5", ".hdf5"):
                info["files"][key] = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "format": "hdf5"}
                continue
            rows, widths, nan_cells = 0, set(), 0
            opener = _open_text(path)
            with opener as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    cells = line.rstrip("\r\n").split(",")
                    rows += 1
                    widths.add(len(cells))
                    nan_cells += sum(1 for c in cells if c.strip().lower() in ("", "nan", "na"))
            info["files"][key] = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "rows": rows,
                                  "columns": sorted(widths), "rectangular": len(widths) == 1, "missing_cells": nan_cells}
        return info

    # -- extraction -----------------------------------------------------------------------------------
    def load_user_files(self, b_csv: str | Path, h_csv: str | Path, t_csv: str | Path, *, material: str,
                        frequency_or_sampling_time: tuple[Any, str], column_units: Mapping[str, str] | None,
                        timebase: str | None = None, delimiter: str = ",", nominal_level: int | None = None,
                        samples_per_period: Sequence[int] = ACCEPTED_SAMPLES_PER_PERIOD, allow_any_length: bool = False) -> ExtractedData:
        """Read user-supplied B / H / T CSV files into :class:`ExtractedData` (sequences) with strict validation."""
        mapping = validate_column_units(column_units)
        if not isinstance(material, str) or not material.strip():
            raise InvalidInputError("material must be a non-empty label, e.g. 'N87' or 'A'")
        paths = {"B": Path(b_csv), "H": Path(h_csv), "T": Path(t_csv)}
        for key, path in paths.items():
            if not path.is_file():
                raise DataNotFoundError(f"{key} file not found: {path}")
        if not isinstance(frequency_or_sampling_time, (tuple, list)) or len(frequency_or_sampling_time) != 2 \
                or not isinstance(frequency_or_sampling_time[1], str):
            raise InvalidInputError("frequency_or_sampling_time must be a (value, unit) pair, e.g. (62.5, 'ns') for the 16 MHz MagNetX "
                                    "data or (200e3, 'Hz') / (array_of_Hz, 'Hz') for single-period rows; a bare number is ambiguous")
        value, unit = frequency_or_sampling_time
        if units.compatible(unit, "Hz"):
            mode = "frequency"
        elif units.compatible(unit, "s"):
            mode = "sampling_time"
        else:
            raise UnitError(f"frequency_or_sampling_time unit {unit!r} is neither a frequency nor a time unit")
        if timebase is None:
            timebase = "uniform_per_period" if mode == "frequency" else "uniform_absolute"
        if timebase not in ("uniform_per_period", "uniform_absolute"):
            raise InvalidInputError("timebase must be 'uniform_per_period' (rows are single periods) or 'uniform_absolute' (fixed sample rate)")
        b = _load_numeric_csv(paths["B"], delimiter=delimiter, missing_allowed=False, label="B")
        h = _load_numeric_csv(paths["H"], delimiter=delimiter, missing_allowed=True, label="H")
        t = _load_numeric_csv(paths["T"], delimiter=delimiter, missing_allowed=False, label="T")
        if b.shape != h.shape:
            raise DataIntegrityError(f"B {b.shape} and H {h.shape} differ in shape (rows x samples must match)")
        n_rows, n_samples = b.shape
        if n_samples < 2:
            raise DataIntegrityError(f"sequences need at least 2 samples, got {n_samples} columns")
        t_flat = t.reshape(-1) if t.shape[1] == 1 or t.shape[0] == 1 else None
        if t_flat is None or t_flat.shape[0] != n_rows:
            raise DataIntegrityError(f"T must hold one temperature per sequence ({n_rows}); got shape {t.shape}")
        values = np.asarray(value, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise InvalidInputError("frequency_or_sampling_time values must be finite and positive")
        if values.size not in (1, n_rows):
            raise InvalidInputError(f"frequency_or_sampling_time must be a scalar or one value per sequence ({n_rows}), got {values.size}")
        notes: list[str] = []
        conditions: dict[str, tuple[np.ndarray, str]] = {"temperature": (t_flat, mapping["T"])}
        sampling_rate: tuple[float, str] | None = None
        if timebase == "uniform_per_period":
            if n_samples not in samples_per_period and not allow_any_length:
                raise DataIntegrityError(f"single-period rows must have {tuple(samples_per_period)} samples, got {n_samples}; pass "
                                         "allow_any_length=True or use timebase='uniform_absolute' for transient records")
            if mode == "frequency":
                frequency = np.broadcast_to(np.asarray([units.convert(v, unit, "Hz") for v in values]), (n_rows,)).copy()
            else:
                ts_s = np.asarray([units.convert(v, unit, "s") for v in values])
                frequency = np.broadcast_to(1.0 / (ts_s * n_samples), (n_rows,)).copy()
                notes.append(f"frequency derived as 1/(sampling_time * {n_samples} samples) from the stated sampling time")
            conditions["frequency"] = (frequency, "Hz")
        else:
            if mode == "frequency":
                if values.size != 1:
                    raise InvalidInputError("uniform_absolute needs a single sampling rate / sampling time, not one per row")
                sampling_rate = (float(values[0]), unit)
            else:
                if values.size != 1 and not np.allclose(values, values[0]):
                    raise InvalidInputError("uniform_absolute needs one common sampling time; per-row sampling times differ "
                                            "(use timebase='uniform_per_period' for single-period rows)")
                # exact rational arithmetic: 62.5 ns -> 16 000 000 Hz, not 15 999 999.999999998 Hz
                seconds = Fraction(float(values[0])) * units.get(unit).factor
                sampling_rate = (float(1 / seconds), "Hz")
            notes.append("rows are transient records at a fixed sample rate (not single periods); no per-row frequency exists")
        nan_h = int(np.isnan(h).sum())
        flags: list[str] = []
        if nan_h:
            flags.append("h_padded_nan")
            notes.append(f"{nan_h} H samples are missing (NaN) - padded/unknown part of the segments; kept as NaN, never 0")
            all_nan_rows = int(np.all(np.isnan(h), axis=1).sum())
            if all_nan_rows:
                notes.append(f"{all_nan_rows} H row(s) are entirely missing")
        if nominal_level is not None:
            if nominal_level not in NOMINAL_LEVELS:
                raise InvalidInputError(f"nominal_level must be 1..7, got {nominal_level}")
            nominal_f, lengths = NOMINAL_LEVELS[nominal_level]
            if n_samples not in lengths:
                notes.append(f"level {nominal_level} nominal lengths are {lengths}; file has {n_samples} columns")
        source_files = {p.name: sha256_file(p) for p in paths.values()}
        item = SourceItem(source_id=SOURCE_ID, item_id=f"{material}/user-files/{paths['B'].stem}", kind="file", material=material,
                          path=str(paths["B"].parent), size_bytes=sum(p.stat().st_size for p in paths.values()), available=True,
                          license=LICENSE, description="user-supplied MagNet Challenge 2 B/H/T files",
                          extra={"files": {k: str(p) for k, p in paths.items()}, "core_shape": CORE_SHAPES.get(material)})
        return ExtractedData(
            source_id=SOURCE_ID, item=item, kind="sequences", material=material,
            arrays={"flux_density": (b, mapping["B"]), "field": (h, mapping["H"])}, conditions=conditions,
            timebase=timebase, sampling_rate=sampling_rate,  # type: ignore[arg-type]
            source_files=source_files, source_paths={p.name: str(p) for p in paths.values()},
            column_sources={"flux_density": paths["B"].name, "field": paths["H"].name, "temperature": paths["T"].name},
            metadata={"n_sequences": n_rows, "n_samples": n_samples, "nominal_level": nominal_level, "mode": mode,
                      "frequency_or_sampling_time": [float(values[0]) if values.size == 1 else values.tolist(), unit],
                      "missing_h_samples": nan_h, "core_shape": CORE_SHAPES.get(material),
                      "temperatures_present": sorted(set(np.round(t_flat, 3).tolist()))},
            notes=notes, quality_flags=flags)

    def extract(self, item: SourceItem, **filters: Any) -> ExtractedData:
        files = {k: Path(p) for k, p in item.extra.get("files", {}).items()}
        if not {"B", "H", "T"} <= set(files):
            raise InvalidInputError("extract() needs a discovered training triplet item (B, H and T files); testing/HDF5 items are "
                                    "imported with import_user_files on the individual files")
        required = ("frequency_or_sampling_time", "column_units")
        missing = [k for k in required if k not in filters]
        if missing:
            raise InvalidInputError(f"extract() requires the explicit arguments {missing} (no unit inference)")
        level = item.extra.get("detail", "").split("-")[-1]
        return self.load_user_files(files["B"], files["H"], files["T"], material=item.material or "unknown",
                                    nominal_level=int(level) if level.isdigit() else None, **filters)

    # -- validation -----------------------------------------------------------------------------------
    def validate(self, extracted: ExtractedData) -> list[str]:
        problems = super().validate(extracted)
        if extracted.kind != "sequences":
            return problems
        b, b_unit = extracted.arrays.get("flux_density", (None, "T"))
        if b is not None:
            peak = float(np.nanmax(np.abs(units.convert(1.0, b_unit, "T") * np.asarray(b))))
            if peak > 2.0:
                problems.append(f"flux_density: peak {peak:g} T exceeds any soft-magnetic saturation; check the stated unit {b_unit!r}")
        h, h_unit = extracted.arrays.get("field", (None, "A/m"))
        if h is not None and np.all(np.isnan(h)):
            problems.append("field: every H sample is missing")
        t, t_unit = extracted.conditions.get("temperature", (None, "K"))
        if t is not None:
            kelvin = np.asarray([units.convert(v, t_unit, "K") for v in np.asarray(t, dtype=np.float64)])
            lo, hi = TEMPERATURE_PLAUSIBLE_K
            outside = int(((kelvin < lo) | (kelvin > hi)).sum())
            if outside:
                problems.append(f"temperature: {outside} value(s) outside {lo - 273.15:g}..{hi - 273.15:g} degC ({lo}..{hi} K; the challenge "
                                f"measured at {TEMPERATURES_DEGC} degC) when read as {t_unit!r} - most likely kelvin declared as degC or vice versa; "
                                "check column_units['T']")
        return problems

    # -- records --------------------------------------------------------------------------------------
    def to_records(self, extracted: ExtractedData, *, destination_dir: str | Path, dataset_id: str | None = None,
                   material_ref: str | None = None, context: Sequence[MagneticRecord] = (), retrieved_at: str | None = None,
                   synthetic: bool = False, source_description: str | None = None, **options: Any) -> ImportResult:
        if options:
            raise InvalidInputError(f"unknown options {sorted(options)}")
        if extracted.kind != "sequences" or extracted.source_id != SOURCE_ID:
            raise InvalidInputError("to_records expects sequences loaded by MagnetChallenge2Adapter.load_user_files")
        problems = self.validate(extracted)
        if problems:
            raise DataIntegrityError("user files failed validation: " + "; ".join(problems), details={"problems": problems})
        material = extracted.material or "unknown"
        token = slug(material)
        stem = slug(_file_stem(extracted.item.item_id.rsplit("/", 1)[-1]))
        destination_dir = Path(destination_dir)
        base = dataset_token(dataset_id) if dataset_id else f"magnet-challenge2-{stem if stem.startswith(token) else f'{token}-{stem}'}"
        dataset_id = dataset_id or f"dataset.{base}"
        h5_path = destination_dir / f"{base}.h5"
        if h5_path.exists():
            raise FileExistsError(f"{h5_path} exists; adapters never overwrite")
        files_digest = canonical_sha256(dict(sorted(extracted.source_files.items())))
        repo_evidence = self._repository_evidence(
            description="MagNet Challenge 2 repository (rules, data-format documentation, baseline notebooks) at the pinned commit; "
                        "no licence file; data hosted externally without a stated licence.")
        files_evidence = make_evidence(
            evidence_id=f"evidence.{base}-user-files", kind="source",
            reference=f"User-supplied MagNet Challenge 2 files for {material}: " + ", ".join(sorted(extracted.source_files)),
            description=(source_description or "B(t)/H(t)/temperature files obtained by the user from the challenge data release "
                         "(Dropbox); imported as supplied with the explicit column units "
                         f"{extracted.unit_mapping()}; no licence statement exists for the data (NOASSERTION)."),
            created_at=COMMIT_DATE, license=LICENSE, locator="; ".join(f"{k}: {v}" for k, v in sorted(extracted.column_sources.items())),
            url=TRAINING_DATA_URL, version="unpinned Dropbox release", sha256=files_digest, accessed_at=retrieved_at)
        evidence = [repo_evidence, files_evidence]
        normalization = ("arrays converted to canonical SI (" + ", ".join(f"{k}: {u}" for k, u in extracted.unit_mapping().items())
                         + ") by WaveformStore; original units kept as HDF5 attributes; missing H samples kept as NaN")
        data_version = f"user-supplied files (sha256 {files_digest})"  # the data never come from the pinned repository
        provenance = self._provenance(f"imported {extracted.n_rows} user-supplied sequences ({extracted.metadata.get('n_samples')} samples)",
                                      inputs=[e.id for e in evidence], normalization=normalization, retrieved_at=retrieved_at, version=data_version)
        raw_uris = {name: source_uri(extracted.source_paths[name], destination_dir) for name in sorted(extracted.source_files)}
        warnings = list(extracted.notes) + [f"{name}: no relative path from {destination_dir}; the raw artifact uses the absolute URI {uri}"
                                            for name, uri in raw_uris.items() if is_file_uri(uri)]
        if material_ref is None:
            material_record: MagneticRecord | None = self._material_identity_record(
                material=material, evidence=[repo_evidence], synthetic=synthetic,
                description=f"Identity record for MagNet Challenge 2 material label {material!r}"
                            + (f" (core shape {CORE_SHAPES[material]} per Pretest_Save_Template.m)" if material in CORE_SHAPES else "")
                            + "; no datasheet parameters.")
            material_ref = material_record.id
        else:
            material_record = None
        sampling_rate = None
        if extracted.sampling_rate is not None:
            from aipe_magnetics.common.quantity import Quantity

            sampling_rate = Quantity.si(extracted.sampling_rate[0], extracted.sampling_rate[1])
        meta = WaveformMeta(source_id=SOURCE_ID, dataset_version=f"user files sha256:{files_digest}", license=LICENSE,
                            timebase=extracted.timebase, sampling_rate=sampling_rate, material_ref=material_ref,
                            source_dataset_ref=dataset_id,
                            description=f"MagNet Challenge 2 user-supplied B/H sequences of {material}; {extracted.metadata.get('n_samples')} samples per row")
        store = WaveformStore()
        try:
            properties = store.write(h5_path, extracted.arrays, extracted.conditions, meta=meta)
            h5_artifact = store.artifact(h5_path, uri=h5_path.name, description="AIPE waveform file (layout aipe.magnetics.waveforms/0.2)")
            raw_artifacts = [source_artifact(Path(extracted.source_paths[name]), media_type="text/csv", sha256=digest, role="raw", uri=raw_uris[name],
                                             description="original user-supplied file (unmodified)")
                             for name, digest in sorted(extracted.source_files.items())]
            extensions = {
                IMPORT_EXTENSION: self._import_extension(source_name=", ".join(sorted(extracted.source_files)), source_sha256=files_digest,
                                                         normalization=normalization,
                                                         extra={"source_sha256_method": "sha256 of the canonical JSON {file name: sha256} of the three files",
                                                                "source_files": dict(sorted(extracted.source_files.items())),
                                                                "column_units": extracted.unit_mapping(),
                                                                "frequency_or_sampling_time": extracted.metadata.get("frequency_or_sampling_time"),
                                                                "timebase": extracted.timebase, "rows": extracted.n_rows}),
                SOURCES_EXTENSION: sources_extension(self.descriptor, extracted.item, extracted, waveform_file=h5_path.name,
                                                     redistribution="not permitted (NOASSERTION)", pinned=None,
                                                     pinned_note="user-supplied data; the pinned commit refers to the documentation repository only"),
            }
            quality = {"status": "normalized", "flags": list(extracted.quality_flags), "notes": list(extracted.notes)}
            waveform_record = record_envelope(
                record_id=f"waveform-dataset.{base}", kind="waveform_dataset",
                name=f"MagNet Challenge 2 sequences - {material} ({stem})", properties=properties.model_dump(mode="json", exclude_none=True),
                evidence=evidence, license=LICENSE, provenance=provenance, artifacts=[h5_artifact, *raw_artifacts], dataset_ref=dataset_id,
                tags=["external-source", SOURCE_ID, "user-supplied"], extensions=extensions, quality=quality, synthetic=synthetic,
                description=f"{extracted.n_rows} user-supplied B(t)/H(t) sequences of {material}; licence NOASSERTION; not redistributable.")
            dataset_record = self._dataset_record(
                dataset_id=dataset_id, name=f"MagNet Challenge 2 user-supplied data - {material} ({stem})", evidence=evidence,
                checksums={**extracted.source_files, h5_path.name: h5_artifact.sha256 or ""}, materials=[material], record_count=extracted.n_rows,
                provenance=provenance, artifacts=[h5_artifact, *raw_artifacts], extensions=extensions, retrieved_at=retrieved_at,
                title=f"MagNet Challenge 2 (MagNetX) data for {material} as supplied by the user", url=TRAINING_DATA_URL, synthetic=synthetic,
                version=data_version,
                description=("User-supplied challenge files imported without redistribution rights (NOASSERTION); "
                             f"waveform file {h5_path.name}. Formats per upstream docs/README.pdf; {MANUAL_PATH}"))
            records = [dataset_record] + ([material_record] if material_record is not None else []) + [waveform_record]
            cross_validate(records, context)
            written = [str(h5_path)] + write_records(records, destination_dir, reusable_ids=[material_ref])
        except Exception:
            if h5_path.exists():
                h5_path.unlink()
            raise
        return self._import_result(records=records, artifacts=[h5_artifact, *raw_artifacts], provenance=provenance, warnings=warnings,
                                   written=written, source_sha256=files_digest)

    def import_user_files(self, b_csv: str | Path, h_csv: str | Path, t_csv: str | Path, *, material: str,
                          frequency_or_sampling_time: tuple[Any, str], column_units: Mapping[str, str] | None,
                          destination_dir: str | Path, dataset_id: str | None = None, timebase: str | None = None, delimiter: str = ",",
                          nominal_level: int | None = None, allow_any_length: bool = False, material_ref: str | None = None,
                          context: Sequence[MagneticRecord] = (), retrieved_at: str | None = None, synthetic: bool = False,
                          source_description: str | None = None) -> ImportResult:
        """Import user-supplied B / H / T CSV files (see module docstring); ``column_units`` is mandatory."""
        extracted = self.load_user_files(b_csv, h_csv, t_csv, material=material, frequency_or_sampling_time=frequency_or_sampling_time,
                                         column_units=column_units, timebase=timebase, delimiter=delimiter, nominal_level=nominal_level,
                                         allow_any_length=allow_any_length)
        return self.to_records(extracted, destination_dir=destination_dir, dataset_id=dataset_id, material_ref=material_ref, context=context,
                               retrieved_at=retrieved_at, synthetic=synthetic, source_description=source_description)


def _file_stem(name: str) -> str:
    """``'N87_5_B.csv.gz' -> 'N87_5'``, ``'N87_Testing_True_B_seq.csv' -> 'N87_Testing_True'``."""
    stem = re.sub(r"(\.csv)?(\.gz)?$", "", name, flags=re.IGNORECASE)
    return re.sub(r"_(B|H|T|B_seq|H_seq)$", "", stem) or stem


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        import gzip

        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


__all__ = ["SOURCE_ID", "DESCRIPTOR", "DATA_LOCATIONS", "MANUAL_PATH", "NOMINAL_LEVELS", "SAMPLING_RATE_HZ", "TEMPERATURES_DEGC",
           "TEMPERATURE_PLAUSIBLE_K", "MATERIALS", "CORE_SHAPES", "ACCEPTED_SAMPLES_PER_PERIOD", "COLUMN_KEYS", "MagnetChallenge2Adapter",
           "validate_column_units"]
