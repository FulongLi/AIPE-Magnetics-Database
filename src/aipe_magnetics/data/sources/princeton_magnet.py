"""Adapter for the Princeton **MagNet** database (https://github.com/PrincetonUniversity/magnet, MIT).

Pinned commit ``8a6892e91ac622e42f8b9182a6dfe9094ceea97a``. Three discovery paths, none of which
imports or executes upstream code:

(a) a user checkout of the repository (``src/magnet/data/<material>_database.h5``) or any directory
    holding those files;
(b) the data directory of an installed ``magnet`` package, located with
    ``importlib.util.find_spec`` only (the package itself is **never imported** - it pins numpy 1.22,
    pandas 1.2.4, torch 1.9 and streamlit);
(c) an on-demand, pinned, checksum-verified download of the packaged ``<material>_database.h5`` from
    ``raw.githubusercontent.com`` into the :class:`~aipe_magnetics.data.sources.cache.LocalCache`
    (``allow_download=True`` required; SHA-256 values below were recorded from the pinned files).

Column -> unit mapping of the packaged scalar database (explicit, no inference; see
:mod:`aipe_magnetics.data.sources.princeton_magnet_db`)::

    Frequency     -> frequency         Hz     (Welch estimate, 10 Hz quantised)
    Flux_Density  -> flux_density_peak T      (AC amplitude = (max B - min B)/2)
    DC_Bias       -> dc_bias_field     A/m    (round(mean H))
    Duty_P/Duty_N -> duty_p/duty_n     1      (per-unit; upstream sentinel -1 = sinusoidal -> NaN here)
    Temperature   -> temperature       degC -> K
    Power_Loss    -> loss_density      W/m^3

Excitation classification (from upstream ``app/ui_db.py`` conventions): ``Duty_P == -1`` sinusoidal;
``|Duty_P + Duty_N - 1| <= tolerance`` triangular (d0 = 0); otherwise trapezoidal with
``d2 = d4 = (1 - Duty_P - Duty_N)/2``.

Records produced by :meth:`PrincetonMagnetAdapter.to_records`: a ``dataset`` record (source, commit,
licence MIT, citation of the three TPEL MagNet papers, checksums), the full table as an HDF5 table
artifact (layout ``aipe.magnetics.tables/0.2``: ``/conditions`` arrays with units and per-array
SHA-256), a ``core_loss_measurement`` JSON record with at most ``max_json_points`` points chosen
deterministically (large arrays never go to JSON) and a ``measurement`` record with specimen, setup
and processing text taken from the file's embedded metadata (parsed as text, never unpickled).

The full single-cycle waveform products of the MagNet website (``<material>_cycle.zip``,
``<material>_measurements.zip``) are documented in :data:`WEBSITE_PRODUCTS`; they are unpinned and
not downloaded by default. :meth:`PrincetonMagnetAdapter.import_cycle_zip` imports a zip the user
downloaded into the waveform layout.

Note on the upstream Steinmetz constants (``src/magnet/constants.py``): ``material_steinmetz_param``
is documented upstream as ``(k_i, alpha, beta)`` - the *iGSE* coefficient, not the classical ``k``
- fitted on sinusoidal 25 degC zero-bias rows; see ``docs/sources.md`` for the cross-check.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from aipe_magnetics.common.errors import DataIntegrityError, DataNotFoundError, InvalidInputError, UnsupportedOperationError
from aipe_magnetics.common.jsonio import sha256_file
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.data.hdf5 import HDF5_MEDIA_TYPE, WaveformMeta, WaveformStore, h5py_available
from aipe_magnetics.data.importers.base import IMPORT_EXTENSION, ImportResult, is_file_uri, source_artifact
from aipe_magnetics.data.records import Artifact, Attribution, Evidence, MagneticRecord
from aipe_magnetics.data.sources.base import (
    SOURCES_EXTENSION,
    ExtractedData,
    SourceAdapter,
    SourceDescriptor,
    SourceItem,
    TableMeta,
    TableStore,
    cross_validate,
    dataset_token,
    file_safe_token,
    make_evidence,
    quantity_json,
    range_problems,
    record_envelope,
    slug,
    source_uri,
    sources_extension,
    write_records,
)
from aipe_magnetics.data.sources.cache import LocalCache
from aipe_magnetics.data.sources.princeton_magnet_db import (
    COLUMN_UNITS,
    KNOWN_MATERIALS,
    UPSTREAM_COMMIT,
    UPSTREAM_LICENSE,
    UPSTREAM_REPOSITORY,
    MagnetDatabase,
    _parse_protocol0_text,
    find_database_files,
    read_magnet_database,
)

SOURCE_ID = "princeton_magnet"
COMMIT_DATE = "2026-05-11T21:26:57-04:00"
RAW_URL_TEMPLATE = "https://raw.githubusercontent.com/PrincetonUniversity/magnet/{commit}/src/magnet/data/{material}_database.h5"
WEBSITE_URL = "https://mag-net.princeton.edu/"
CHALLENGE_DROPBOX_URL = "https://www.dropbox.com/sh/yk3rsinvsj831a7/AAAC6vPwXSJgruxmq0EbNvrVa?dl=0"
CITATION = (
    "D. Serrano et al., 'Why MagNet: Quantifying the Complexity of Modeling Power Magnetic Material Characteristics', "
    "IEEE Trans. Power Electron., 2023, doi:10.1109/TPEL.2023.3291084; "
    "H. Li et al., 'How MagNet: Machine Learning Framework for Modeling Power Magnetic Material Characteristics', "
    "IEEE Trans. Power Electron., 2023, doi:10.1109/TPEL.2023.3309232; "
    "H. Li, D. Serrano, S. Wang, M. Chen, 'MagNet-AI: Neural Network as Datasheet for Magnetics Modeling and Material "
    "Recommendation', IEEE Trans. Power Electron., 2023, doi:10.1109/TPEL.2023.3309233"
)
TERMS_OF_ACCESS = ("Website download page (app/ui_raw.py, 'Terms of Access'): MIT permission text; repository LICENSE: MIT, "
                   "Copyright (c) 2021 Minjie Chen; citation of the three TPEL papers is requested")
#: material -> manufacturer as listed upstream in ``src/magnet/constants.py`` (``manufacturers``, lines 42-53)
MANUFACTURERS: dict[str, str] = {"3C90": "Ferroxcube", "3C94": "Ferroxcube", "3E6": "Ferroxcube", "3F4": "Ferroxcube",
                                 "77": "Fair-Rite", "78": "Fair-Rite", "N27": "TDK", "N30": "TDK", "N49": "TDK", "N87": "TDK"}
#: packaged database files at the pinned commit: SHA-256 (recorded at ingest), byte size, row count
PACKAGED_DATABASES: dict[str, dict[str, Any]] = {
    "3C90": {"sha256": "421d9fd33e4c1626dd3a22ca26755b2b98a98f4175f44df2622d7ead8d5417f2", "size_bytes": 6953888, "n_points": 108494},
    "3C94": {"sha256": "7859a037a0d1225e3d60c61d285c506445b32fbcf642309a5fd18f07c71fdbd5", "size_bytes": 7286488, "n_points": 113691},
    "3E6": {"sha256": "7198d2cfddde46f50e85a9d2e320b402b6322677632ff3697f8916e497e47921", "size_bytes": 458056, "n_points": 6996},
    "3F4": {"sha256": "bc166e60edf37150d30224f785378c0f1b0160f580402f447cd770617ee20de5", "size_bytes": 3250616, "n_points": 50630},
    "77": {"sha256": "7e2a9044b92ab6ebea40fa971747a3b3af491c8f7f66c030e53f5e85ee099a39", "size_bytes": 1929376, "n_points": 29986},
    "78": {"sha256": "160643789e6d210cacbe60364ab64a1decab87f5ab321e6cc1d9f083a9434fcd", "size_bytes": 1552096, "n_points": 24091},
    "N27": {"sha256": "1fbfe3e94511c70c5c95a3d9bb768d91040321685c6b3cb8e8bd24e3305093c8", "size_bytes": 2758992, "n_points": 42948},
    "N30": {"sha256": "f41f08b0516e8bff35968b6b5e0d6730ed713400dd06bdf784e7a09fb8d41b02", "size_bytes": 914896, "n_points": 14134},
    "N49": {"sha256": "860319862452636277260a83f214179f4b964b431b63b9f7d630eb312f1c3f82", "size_bytes": 2645024, "n_points": 41168},
    "N87": {"sha256": "263c0087f6f43ccfd1b338d105abb6b9432be2cb85c59528539317f99b25bd63", "size_bytes": 9154016, "n_points": 142871},
}
#: upstream column -> (AIPE column, unit as stored upstream)
COLUMN_MAP: dict[str, tuple[str, str]] = {
    "Frequency": ("frequency", "Hz"), "Flux_Density": ("flux_density_peak", "T"), "DC_Bias": ("dc_bias_field", "A/m"),
    "Duty_P": ("duty_p", "1"), "Duty_N": ("duty_n", "1"), "Temperature": ("temperature", "degC"), "Power_Loss": ("loss_density", "W/m^3"),
}
EXCITATIONS: tuple[str, ...] = ("sinusoidal", "triangular", "trapezoidal")
#: website zip products (documentation; URLs are not pinned upstream - see ``website_zip_urls``)
WEBSITE_PRODUCTS: dict[str, dict[str, Any]] = {
    "cycle": {
        "file_pattern": "<material>_cycle.zip",
        "members": {"b_cycle.csv": "rows x 1024 single-cycle flux density [T]", "h_cycle.csv": "rows x 1024 field [A/m]",
                    "sampling_time.csv": "one value per row, sampling time [s] = 1/(f*1024)",
                    "temperature.csv": "one value per row [degC]", "test_info.txt": "setup text"},
        "units": {"b_cycle.csv": "T", "h_cycle.csv": "A/m", "sampling_time.csv": "s", "temperature.csv": "degC"},
        "samples_per_period": (1024, 128),
        "note": "upstream prose is inconsistent (128 vs 1024 samples); the importer accepts both and records which was found",
    },
    "measurements": {
        "file_pattern": "<material>_measurements.zip",
        "members": {"voltage.csv": "rows x 10000 secondary voltage [V] at 8 ns", "current.csv": "rows x 10000 primary current [A]",
                    "sampling_time.csv": "sampling time [s] (8e-9)", "temperature.csv": "[degC]", "test_info.txt": "setup text"},
        "units": {"voltage.csv": "V", "current.csv": "A", "sampling_time.csv": "s", "temperature.csv": "degC"},
        "samples_per_record": 10000,
        "note": "raw oscilloscope records (several switching cycles), not single periods",
    },
}
DEFAULT_CYCLE_MEMBERS = {"flux_density": "b_cycle.csv", "field": "h_cycle.csv", "sampling_time": "sampling_time.csv",
                         "temperature": "temperature.csv", "info": "test_info.txt"}

DESCRIPTOR = SourceDescriptor(
    source_id=SOURCE_ID, title="Princeton MagNet database (PrincetonUniversity/magnet)", repository_url=UPSTREAM_REPOSITORY,
    pinned_commit=UPSTREAM_COMMIT, pinned_commit_date=COMMIT_DATE, license=UPSTREAM_LICENSE, redistribution_permitted=True,
    citation=CITATION, publisher="Princeton Power Electronics Research Lab (Minjie Chen), with Dartmouth College and Plexim",
    data_locations=[
        f"{UPSTREAM_REPOSITORY}/tree/{UPSTREAM_COMMIT}/src/magnet/data (<material>_database.h5, 10 materials, ~40 MB)",
        RAW_URL_TEMPLATE,
        f"{WEBSITE_URL} (MagNet Download page: <material>_cycle.zip and <material>_measurements.zip, unpinned)",
        f"{CHALLENGE_DROPBOX_URL} (MagNet Challenge 2023 data for 10 materials, unpinned)",
    ],
    notes=[TERMS_OF_ACCESS,
           "Scalar database columns: Frequency Hz, Flux_Density T (amplitude), DC_Bias A/m, Duty_P/Duty_N per-unit (-1 = sinusoidal), "
           "Temperature degC, Power_Loss W/m^3; one toroid per material; oil-bath temperature set point; Bdc not measured.",
           "constants.material_steinmetz_param is (k_i, alpha, beta) - an iGSE coefficient fitted on sinusoidal 25 degC zero-bias rows.",
           "Website zips: cycle files hold 1024 samples per period (challenge/tutorial variants 128 by 8x decimation); measurements "
           "hold 10000 samples at 8 ns; URLs are not pinned and no checksums are published."])


def _installed_package_data_dirs() -> list[Path]:
    """Data directories of an installed ``magnet`` package found without importing it."""
    try:
        spec = importlib.util.find_spec("magnet")
    except (ImportError, ValueError):  # pragma: no cover - broken installations
        return []
    if spec is None:
        return []
    locations: list[str] = []
    if spec.submodule_search_locations:
        locations.extend(str(p) for p in spec.submodule_search_locations)
    elif spec.origin:
        locations.append(str(Path(spec.origin).parent))
    dirs: list[Path] = []
    for location in locations:
        candidate = Path(location) / "data"
        if candidate.is_dir() and any(candidate.glob("*_database.h5")):
            dirs.append(candidate)
    return dirs


def classify_excitation(duty_p: np.ndarray, duty_n: np.ndarray, *, tolerance: float = 1e-6) -> np.ndarray:
    """Codes into :data:`EXCITATIONS`: 0 sinusoidal (Duty_P == -1), 1 triangular (d1 + d3 == 1), 2 trapezoidal."""
    p, n = np.asarray(duty_p, dtype=np.float64), np.asarray(duty_n, dtype=np.float64)
    codes = np.full(p.shape, 2, dtype=np.int64)
    codes[np.abs(p + n - 1.0) <= tolerance] = 1
    codes[p == -1.0] = 0
    return codes


def _item_for_path(path: Path, material: str, *, kind: str = "file") -> SourceItem:
    known = PACKAGED_DATABASES.get(material, {})
    return SourceItem(source_id=SOURCE_ID, item_id=path.name, kind=kind, material=material, path=str(path),  # type: ignore[arg-type]
                      url=RAW_URL_TEMPLATE.format(commit=UPSTREAM_COMMIT, material=material),
                      size_bytes=path.stat().st_size if path.is_file() else None, available=True, license=UPSTREAM_LICENSE,
                      description=f"packaged MagNet scalar core-loss database for {material}",
                      extra={"expected_sha256": known.get("sha256"), "expected_size_bytes": known.get("size_bytes"),
                             "expected_n_points": known.get("n_points"), "manufacturer": MANUFACTURERS.get(material)})


class PrincetonMagnetAdapter(SourceAdapter):
    """Discovery, inspection, extraction and record generation for the MagNet scalar databases."""

    descriptor = DESCRIPTOR
    importer_id = "aipe.magnetics.sources.princeton_magnet"
    _license_policy = {"read": True, "cache": True, "bundle": True, "redistribute": True}

    def __init__(self, cache: LocalCache | None = None, *, website_zip_urls: Mapping[str, str] | None = None):
        self.cache = cache or LocalCache()
        #: optional ``{"cycle": "https://.../{material}_cycle.zip", "measurements": ...}`` templates (unpinned upstream)
        self.website_zip_urls = dict(website_zip_urls or {})

    # -- discovery ----------------------------------------------------------------------------------
    def discover(self, root: str | Path | None = None, allow_download: bool = False, *, materials: Sequence[str] | None = None,
                 **options: Any) -> list[SourceItem]:
        wanted = [m for m in (materials or KNOWN_MATERIALS)]
        unknown = sorted(set(wanted) - set(KNOWN_MATERIALS))
        if unknown:
            raise InvalidInputError(f"unknown MagNet materials {unknown}; known: {list(KNOWN_MATERIALS)}")
        items: list[SourceItem] = []
        found: set[str] = set()
        note: str | None = None
        if root is not None:
            root = Path(root)
            if not root.is_dir():
                raise DataNotFoundError(f"directory not found: {root}", details={"root": str(root)})
            try:
                local = find_database_files(root)
            except DataNotFoundError:
                # a directory without database files is not an error for discovery: report what a download would provide
                local = {}
                note = f"no <material>_database.h5 files under {root}; listing cached / remote items instead"
            for material, path in sorted(local.items()):
                if material in wanted:
                    items.append(_item_for_path(path, material))
                    found.add(material)
        else:
            for data_dir in _installed_package_data_dirs():
                for path in sorted(data_dir.glob("*_database.h5")):
                    material = path.name.split("_database")[0]
                    if material in wanted and material not in found:
                        items.append(_item_for_path(path, material, kind="package_data"))
                        found.add(material)
        for material in wanted:
            if material in found:
                continue
            url = RAW_URL_TEMPLATE.format(commit=UPSTREAM_COMMIT, material=material)
            known = PACKAGED_DATABASES[material]
            cached = self.cache.lookup(url)
            if cached is not None and self.cache.path_of(cached).is_file():
                item = _item_for_path(self.cache.path_of(cached), material)
                item.extra["cached"] = True
                if note:
                    item.extra["note"] = note
                items.append(item)
                continue
            items.append(SourceItem(
                source_id=SOURCE_ID, item_id=f"{material}_database.h5", kind="remote", material=material, url=url,
                size_bytes=known["size_bytes"], sha256=known["sha256"], available=False, license=UPSTREAM_LICENSE,
                description=f"packaged MagNet database for {material}; download requires allow_download=True "
                            f"({'authorised for this call' if allow_download else 'not authorised'})",
                extra={"expected_sha256": known["sha256"], "expected_n_points": known["n_points"], "downloadable": True,
                       "manufacturer": MANUFACTURERS.get(material), **({"note": note} if note else {})}))
        return items

    def fetch(self, item: SourceItem, *, allow_download: bool = False, retrieved_at: str | None = None, timeout: float = 120.0) -> Path:
        """Local path of an item; remote items are fetched through the cache only with ``allow_download=True``."""
        if item.path and Path(item.path).is_file():
            return Path(item.path)
        if not item.url:
            raise DataNotFoundError(f"item {item.item_id} has neither a local path nor a URL")
        self.check_license("cache")
        expected = item.sha256 or item.extra.get("expected_sha256")
        return self.cache.fetch(item.url, expected, allow_download, filename=item.item_id, commit=UPSTREAM_COMMIT,
                                license=UPSTREAM_LICENSE, source_id=SOURCE_ID, retrieved_at=retrieved_at, timeout=timeout)

    # -- inspection ---------------------------------------------------------------------------------
    def inspect(self, item: SourceItem, *, compute_sha256: bool = True, with_ranges: bool = False, **options: Any) -> dict[str, Any]:
        """Header-level metadata of a database file (shapes, columns, embedded text metadata, digest)."""
        if not h5py_available():
            raise UnsupportedOperationError("inspecting MagNet HDF5 databases requires h5py (pip install 'aipe-magnetics[hdf5]')")
        import h5py  # type: ignore

        path = self.fetch(item, allow_download=False)
        info: dict[str, Any] = {"item_id": item.item_id, "material": item.material, "path": str(path), "size_bytes": path.stat().st_size,
                                "units": dict(COLUMN_UNITS), "column_map": {k: list(v) for k, v in COLUMN_MAP.items()}}
        with h5py.File(path, "r") as handle:
            group = handle.get("dataframe")
            if group is None:
                raise DataIntegrityError(f"{path.name}: missing /dataframe group (not a pandas fixed-format store)")
            values = group.get("block0_values")
            items = group.get("block0_items")
            info["columns"] = [x.decode() if isinstance(x, bytes) else str(x) for x in items[()]] if items is not None else []
            info["shape"] = [int(s) for s in values.shape] if values is not None else None
            info["n_points"] = int(values.shape[0]) if values is not None else None
            info["dtype"] = str(values.dtype) if values is not None else None
            raw_meta = group.attrs.get("metadata", b"")
            if isinstance(raw_meta, bytes):
                raw_meta = raw_meta.decode("utf-8", "replace")
            info["metadata"] = _parse_protocol0_text(raw_meta) if raw_meta else {}
            if with_ranges and values is not None:
                block = np.asarray(values[()], dtype=np.float64)
                info["ranges"] = {name: [float(block[:, i].min()), float(block[:, i].max())] for i, name in enumerate(info["columns"])}
        if compute_sha256:
            info["sha256"] = sha256_file(path)
            expected = item.extra.get("expected_sha256") or item.sha256
            info["matches_pinned_commit"] = None if expected is None else (info["sha256"] == expected)
        info["expected_n_points"] = item.extra.get("expected_n_points")
        return info

    # -- extraction ---------------------------------------------------------------------------------
    def extract(self, item: SourceItem, *, temperature_degC: float | tuple[float, float] | None = None,
                frequency_range_hz: tuple[float, float] | None = None, flux_density_range_t: tuple[float, float] | None = None,
                dc_bias_range_am: tuple[float, float] | None = None, excitation: str | None = None,
                allow_download: bool = False, duty_tolerance: float = 1e-6, **filters: Any) -> ExtractedData:
        if filters:
            raise InvalidInputError(f"unknown filters {sorted(filters)}; supported: temperature_degC, frequency_range_hz, "
                                    "flux_density_range_t, dc_bias_range_am, excitation")
        path = self.fetch(item, allow_download=allow_download)
        db = read_magnet_database(path, material=item.material)
        return self.extract_database(db, item=item, temperature_degC=temperature_degC, frequency_range_hz=frequency_range_hz,
                                     flux_density_range_t=flux_density_range_t, dc_bias_range_am=dc_bias_range_am,
                                     excitation=excitation, duty_tolerance=duty_tolerance)

    def extract_database(self, db: MagnetDatabase, *, item: SourceItem | None = None, temperature_degC: float | tuple[float, float] | None = None,
                         frequency_range_hz: tuple[float, float] | None = None, flux_density_range_t: tuple[float, float] | None = None,
                         dc_bias_range_am: tuple[float, float] | None = None, excitation: str | None = None,
                         duty_tolerance: float = 1e-6) -> ExtractedData:
        """Build :class:`ExtractedData` from an already-read :class:`MagnetDatabase` (filters in source units)."""
        if excitation is not None and excitation not in EXCITATIONS:
            raise InvalidInputError(f"excitation must be one of {EXCITATIONS}, got {excitation!r}")
        item = item or _item_for_path(db.path, db.material)
        f, b, hdc = db.column("Frequency"), db.column("Flux_Density"), db.column("DC_Bias")
        dp, dn, t, p = db.column("Duty_P"), db.column("Duty_N"), db.column("Temperature"), db.column("Power_Loss")
        codes = classify_excitation(dp, dn, tolerance=duty_tolerance)
        mask = np.ones(db.n_points, dtype=bool)
        applied: dict[str, Any] = {}
        if temperature_degC is not None:
            if isinstance(temperature_degC, (tuple, list)):
                if len(temperature_degC) != 2:
                    raise InvalidInputError("temperature_degC must be a single set point or a (low, high) pair")
                lo, hi = float(temperature_degC[0]), float(temperature_degC[1])
                if lo > hi:
                    raise InvalidInputError(f"temperature_degC: lower bound {lo} exceeds upper bound {hi}")
                mask &= (t >= lo) & (t <= hi)
            else:
                lo = hi = float(temperature_degC)
                mask &= np.abs(t - lo) <= 0.5
            applied["temperature_degC"] = [lo, hi]
        for key, array, rng in (("frequency_range_hz", f, frequency_range_hz), ("flux_density_range_t", b, flux_density_range_t),
                                ("dc_bias_range_am", hdc, dc_bias_range_am)):
            if rng is not None:
                lo, hi = float(rng[0]), float(rng[1])
                if lo > hi:
                    raise InvalidInputError(f"{key}: lower bound {lo} exceeds upper bound {hi}")
                mask &= (array >= lo) & (array <= hi)
                applied[key] = [lo, hi]
        if excitation is not None:
            mask &= codes == EXCITATIONS.index(excitation)
            applied["excitation"] = excitation
        if not mask.any():
            raise DataNotFoundError(f"{db.material}: no rows match the filters {applied}", details={"filters": applied})
        idx = np.flatnonzero(mask)
        duty_p = dp[idx].copy()
        duty_n = dn[idx].copy()
        sinus = codes[idx] == 0
        duty_p[sinus] = np.nan  # upstream sentinel -1 means 'not applicable'; AIPE stores missing as NaN
        duty_n[sinus] = np.nan
        columns = {
            "frequency": (f[idx], "Hz"), "flux_density_peak": (b[idx], "T"), "dc_bias_field": (hdc[idx], "A/m"),
            "duty_p": (duty_p, "1"), "duty_n": (duty_n, "1"), "temperature": (t[idx], "degC"), "loss_density": (p[idx], "W/m^3"),
        }
        counts = {name: int((codes[idx] == code).sum()) for code, name in enumerate(EXCITATIONS)}
        return ExtractedData(
            source_id=SOURCE_ID, item=item, kind="table", material=db.material, columns=columns,
            categorical={"excitation": (codes[idx], list(EXCITATIONS))},
            source_files={db.path.name: db.sha256}, source_paths={db.path.name: str(db.path)},
            column_sources={aipe: upstream for upstream, (aipe, _unit) in COLUMN_MAP.items()},
            metadata={"info_core": db.metadata.get("info_core"), "info_setup": db.metadata.get("info_setup"),
                      "info_processing": db.metadata.get("info_processing"), "core": dict(db.core_info),
                      "n_points_total": db.n_points, "n_points_selected": int(idx.size), "excitation_counts": counts,
                      "row_indices_selected": idx.tolist() if idx.size <= 1000 else None,
                      "manufacturer": MANUFACTURERS.get(db.material), "expected_sha256": PACKAGED_DATABASES.get(db.material, {}).get("sha256")},
            filters=applied,
            notes=["Duty_P/Duty_N of sinusoidal rows (upstream sentinel -1) stored as NaN; excitation kept as a categorical column",
                   "temperature is the oil-bath set point (25/50/70/90 degC); DC bias is round(mean H) of the measured field"])

    # -- validation ---------------------------------------------------------------------------------
    def validate(self, extracted: ExtractedData) -> list[str]:
        problems = super().validate(extracted)
        if extracted.kind != "table":
            return problems
        cols = extracted.columns
        if "frequency" in cols:
            problems += range_problems("frequency", cols["frequency"][0], "Hz", minimum=1e3, maximum=1e7)
        if "flux_density_peak" in cols:
            problems += range_problems("flux_density_peak", cols["flux_density_peak"][0], "T", minimum=0.0, maximum=1.0, strict_min=True)
        if "loss_density" in cols:
            problems += range_problems("loss_density", cols["loss_density"][0], "W/m^3", minimum=0.0, strict_min=True)
        if "temperature" in cols:
            problems += range_problems("temperature", cols["temperature"][0], "degC", minimum=-50.0, maximum=200.0)
        for name in ("duty_p", "duty_n"):
            if name in cols:
                problems += range_problems(name, cols[name][0], "1", minimum=0.0, maximum=1.0, strict_min=True)
        if "excitation" in extracted.categorical and "duty_p" in cols:
            codes = np.asarray(extracted.categorical["excitation"][0])
            nan_duty = np.isnan(np.asarray(cols["duty_p"][0]))
            if np.any((codes == 0) != nan_duty):
                problems.append("excitation codes disagree with the duty columns (sinusoidal rows must have missing duty)")
        expected = extracted.metadata.get("expected_sha256")
        for name, digest in extracted.source_files.items():
            if expected and digest != expected:
                problems.append(f"{name}: sha256 {digest[:12]}... differs from the pinned upstream file {expected[:12]}... "
                                "(different version or modified copy)")
        return problems

    # -- records ------------------------------------------------------------------------------------
    @staticmethod
    def select_json_points(extracted: ExtractedData, max_points: int) -> tuple[np.ndarray, str]:
        """Deterministic subset: sinusoidal zero-bias rows at the lowest temperature sorted by (f, B); evenly thinned."""
        if max_points < 1:
            raise InvalidInputError("max_json_points must be >= 1")
        f = np.asarray(extracted.columns["frequency"][0])
        b = np.asarray(extracted.columns["flux_density_peak"][0])
        t = np.asarray(extracted.columns["temperature"][0])
        hdc = np.asarray(extracted.columns["dc_bias_field"][0])
        codes = np.asarray(extracted.categorical["excitation"][0])
        sel = np.flatnonzero((codes == 0) & (hdc == 0.0))
        rule = "sinusoidal zero-bias rows at the lowest temperature, sorted by frequency then flux density"
        if sel.size:
            t_min = t[sel].min()
            sel = sel[np.abs(t[sel] - t_min) <= 0.5]
        else:
            sel = np.arange(f.size)
            rule = "no sinusoidal zero-bias rows in the extraction: all rows sorted by temperature, frequency, flux density"
        order = np.lexsort((b[sel], f[sel], t[sel]))
        sel = sel[order]
        if sel.size > max_points:
            sel = sel[np.unique(np.linspace(0, sel.size - 1, max_points).round().astype(int))]
            rule += f"; evenly thinned to {max_points} points"
        return sel, rule

    def to_records(self, extracted: ExtractedData, *, destination_dir: str | Path, dataset_id: str | None = None,
                   max_json_points: int = 200, material_ref: str | None = None, context: Sequence[MagneticRecord] = (),
                   retrieved_at: str | None = None, file_prefix: str | None = None, allow_unpinned: bool = False,
                   **options: Any) -> ImportResult:
        """Write ``dataset`` / identity ``magnetic_material`` / ``measurement`` / ``core_loss_measurement`` records and the table artifact.

        ``allow_unpinned``: a source file that is not byte-identical to the packaged file at the pinned
        commit is refused with :class:`DataIntegrityError`; with ``True`` it is imported, every record is
        flagged ``source_differs_from_pinned_commit``, versioned ``'unpinned copy (sha256 ...)'`` instead
        of the commit, and ``extensions['aipe.sources']['pinned_mismatch']`` holds both digests.
        ``dataset_id`` / ``file_prefix`` tokens must be file-safe (``[a-z0-9]+(-[a-z0-9]+)*``).
        """
        if options:
            raise InvalidInputError(f"unknown options {sorted(options)}")
        if extracted.kind != "table" or extracted.source_id != SOURCE_ID:
            raise InvalidInputError("to_records expects table data extracted by PrincetonMagnetAdapter")
        problems = self.validate(extracted)
        blocking = [p for p in problems if "differs from the pinned upstream file" not in p]
        if blocking:
            raise DataIntegrityError("extracted data failed validation: " + "; ".join(blocking), details={"problems": problems})
        warnings = [p for p in problems if p not in blocking]
        pin = self._unpinned_policy(extracted, allow_unpinned=allow_unpinned)
        material = extracted.material or "unknown"
        token = slug(material)
        destination_dir = Path(destination_dir)
        base = dataset_token(dataset_id) if dataset_id else f"princeton-magnet-{token}"
        dataset_id = dataset_id or f"dataset.{base}"
        prefix = file_safe_token(file_prefix, label="file_prefix") if file_prefix else f"{base}-core-loss"
        source_name, source_digest = next(iter(extracted.source_files.items()))
        source_path = extracted.source_paths.get(source_name)
        raw_uri = source_uri(source_path or source_name, destination_dir, pinned_url=extracted.item.url if pin.pinned else None)
        if is_file_uri(raw_uri):
            warnings.append(f"{source_name}: no relative path from {destination_dir} to the source file; the raw artifact uses the absolute URI {raw_uri}")
        file_version = pin.version or UPSTREAM_COMMIT
        origin = (f"commit {UPSTREAM_COMMIT}" if pin.pinned else
                  f"an UNPINNED copy (sha256 {source_digest}; the pinned commit {UPSTREAM_COMMIT[:12]} holds {pin.mismatch['expected_sha256'][:12]}...)"
                  if pin.mismatch else f"a copy without a pinned digest (sha256 {source_digest}; material not packaged at commit {UPSTREAM_COMMIT[:12]})")
        if pin.pinned is None:
            warnings.append(f"{source_name}: no pinned digest is known for material {material!r}; records are versioned by the file's SHA-256")
        core = extracted.metadata.get("core") or {}
        core_text = core.get("core")
        specimen = (f"{material} toroid {core_text}" if core_text else material) + "".join(
            f"; {k} {core[k]} {core.get('units', {}).get(k, '')}".rstrip() for k in ("Ae", "le", "Ve") if k in core) + "".join(
            f"; {k} {core[k]}" for k in ("N1", "N2", "CoreN", "Dataset") if k in core)
        info_setup = extracted.metadata.get("info_setup") or ""
        info_processing = extracted.metadata.get("info_processing") or ""
        acquired = info_setup.split("---")[0].strip() if info_setup else None

        # evidence
        repo_evidence = self._repository_evidence(
            description="Princeton MagNet repository at the pinned commit (MIT, Copyright (c) 2021 Minjie Chen); packaged scalar "
                        "core-loss databases under src/magnet/data; citation of the TPEL MagNet papers requested.")
        file_evidence = make_evidence(
            evidence_id=f"evidence.princeton-magnet-{token}-database", kind="measurement",
            reference=f"PrincetonUniversity/magnet src/magnet/data/{source_name} (pandas fixed-format HDF5), commit {UPSTREAM_COMMIT}",
            description=(f"Measured scalar core-loss points of {material} ({specimen}); two-winding V-I method in an oil bath; "
                         f"info_setup: {info_setup[:300]}" + (" ..." if len(info_setup) > 300 else "")),
            created_at=COMMIT_DATE, license=UPSTREAM_LICENSE, locator="/dataframe/block0_values (columns Frequency, Flux_Density, "
            "DC_Bias, Duty_P, Duty_N, Temperature, Power_Loss)", url=extracted.item.url, version=file_version, sha256=source_digest,
            accessed_at=retrieved_at)
        evidence = [repo_evidence, file_evidence]
        normalization = ("columns converted to canonical SI with aipe_magnetics.common.units (temperature degC -> K; other columns "
                         "already SI); original units kept as HDF5 'original_unit' attributes and Quantity.original; sinusoidal duty "
                         "sentinel -1 -> NaN; excitation classified from Duty_P/Duty_N")
        provenance = self._provenance(f"extracted {extracted.n_rows} of {extracted.metadata.get('n_points_total')} rows from {source_name}"
                                      + (f" with filters {extracted.filters}" if extracted.filters else ""),
                                      inputs=[repo_evidence.id, file_evidence.id], normalization=normalization, retrieved_at=retrieved_at,
                                      version=pin.version)
        pin_quality = {"status": "normalized", "flags": list(pin.flags), "notes": list(warnings)} if (pin.flags or warnings) else None
        if material_ref is None:
            material_record = self._material_identity_record(
                material=material, evidence=[repo_evidence], manufacturer=extracted.metadata.get("manufacturer"),
                extensions={SOURCES_EXTENSION: dict(sources_extension(self.descriptor, extracted.item), item_id=f"{material}_database.h5",
                                                    locator="src/magnet/constants.py (manufacturers)")})
            material_ref = material_record.id
        else:
            material_record = None

        # table artifact
        table_path = destination_dir / f"{prefix}-table.h5"
        if table_path.exists():
            raise FileExistsError(f"{table_path} exists; adapters never overwrite")
        meta = TableMeta(source_id=SOURCE_ID, dataset_version=file_version, license=UPSTREAM_LICENSE, material_ref=material_ref,
                         source_dataset_ref=dataset_id,
                         description=f"Princeton MagNet scalar core-loss table of {material} ({extracted.n_rows} rows); {specimen}")
        store = TableStore()
        try:
            description = store.write(table_path, extracted.columns, meta=meta, categorical=extracted.categorical,
                                      source_columns=extracted.column_sources)
            table_artifact = store.artifact(table_path, uri=table_path.name,
                                            description="full extracted table (layout aipe.magnetics.tables/0.2, per-array SHA-256)")
            raw_artifact = source_artifact(Path(source_path) if source_path else Path(source_name), media_type=HDF5_MEDIA_TYPE,
                                           sha256=source_digest, role="raw", uri=raw_uri,
                                           description="original upstream file as read (pandas fixed-format HDF5, unmodified)")
            import_ext = self._import_extension(source_name=source_name, source_sha256=source_digest, normalization=normalization,
                                                extra={"columns": {aipe: {"source_column": src, "unit": extracted.columns[aipe][1]}
                                                                   for aipe, src in extracted.column_sources.items() if aipe in extracted.columns},
                                                       "filters": dict(extracted.filters), "rows": extracted.n_rows})
            sources_ext = sources_extension(self.descriptor, extracted.item, extracted, table_artifact=table_path.name,
                                            excitation_counts=extracted.metadata.get("excitation_counts"), **pin.extension())
            extensions = {IMPORT_EXTENSION: import_ext, SOURCES_EXTENSION: sources_ext}

            # subset for JSON
            sel, rule = self.select_json_points(extracted, max_json_points)
            points = self._loss_points(extracted, sel, file_evidence.id)
            dataset_record = self._dataset_record(
                dataset_id=dataset_id, name=f"Princeton MagNet scalar core-loss database - {material}", evidence=evidence,
                checksums={source_name: source_digest, table_path.name: table_artifact.sha256 or ""}, materials=[material],
                record_count=extracted.n_rows, provenance=provenance, artifacts=[table_artifact, raw_artifact], extensions=extensions,
                retrieved_at=retrieved_at, version=pin.version, quality=pin_quality,
                description=(f"{extracted.n_rows} measured operating points ({extracted.metadata.get('excitation_counts')}) of {material} "
                             f"from {source_name} at {origin}; licence MIT; the full table is the HDF5 artifact "
                             f"{table_path.name}, the JSON core_loss_measurement record holds a {len(points)}-point subset."))
            measurement_record = record_envelope(
                record_id=f"measurement.{base}-database", kind="measurement",
                name=f"Princeton MagNet measurement campaign - {material}",
                properties={"material_ref": material_ref, "artifact": table_path.name,
                            "tool": Attribution(id="github.com/PrincetonUniversity/magnet/scripts/data_processing_files_and_example",
                                                version=UPSTREAM_COMMIT, name="MagNet MATLAB processing pipeline (ReadRaw_1 ... Webpage_6)").model_dump(mode="json", exclude_none=True),
                            "instrument": info_setup or "not stated in the file",
                            "method": ("two-winding voltage-current method (secondary voltage integrated to B, primary current to H), "
                                       "single-cycle averaging and discard rules per upstream info_processing: " + (info_processing or "not stated")),
                            "specimen": specimen, **({"acquired_at": acquired} if acquired else {}),
                            "description": f"Measurement provenance of the {material} scalar database; values are in the table artifact."},
                evidence=evidence, license=UPSTREAM_LICENSE, provenance=provenance, artifacts=[table_artifact], dataset_ref=dataset_id,
                tags=["external-source", SOURCE_ID], extensions=extensions, quality=pin_quality,
                description=f"Setup, specimen and processing description embedded in {source_name} (parsed as text).")
            loss_properties: dict[str, Any] = {
                "material_ref": material_ref, "specimen_description": specimen,
                "method": "two-winding V-I method, oil bath, single-cycle averaged waveforms (see measurement record)",
                "points": points, "dataset_ref": dataset_id,
                "description": (f"Subset rule: {rule}. The complete table ({extracted.n_rows} rows) is the HDF5 artifact "
                                f"{table_path.name}; large arrays are never written to JSON."),
            }
            if info_setup:
                loss_properties["instrument"] = (info_setup[:200] + " ...") if len(info_setup) > 200 else info_setup
            loss_record = record_envelope(
                record_id=f"core-loss-measurement.{base}-subset", kind="core_loss_measurement",
                name=f"Princeton MagNet core-loss points - {material} ({len(points)}-point JSON subset)",
                properties=loss_properties, evidence=evidence, license=UPSTREAM_LICENSE, provenance=provenance,
                artifacts=[table_artifact], dataset_ref=dataset_id, tags=["external-source", SOURCE_ID, "subset"], extensions=extensions,
                quality={"status": "normalized", "flags": (["json_subset"] if len(points) < extracted.n_rows else []) + list(pin.flags),
                         "notes": [f"{len(points)} of {extracted.n_rows} rows in JSON; selection rule: {rule}", *warnings]})
            records = [dataset_record] + ([material_record] if material_record is not None else []) + [measurement_record, loss_record]
            cross_validate(records, context)
            written = [str(table_path)] + write_records(records, destination_dir, reusable_ids=[material_ref])
        except Exception:
            if table_path.exists():
                table_path.unlink()
            raise
        return self._import_result(records=records, artifacts=[table_artifact, raw_artifact], provenance=provenance, warnings=warnings,
                                   written=written, source_sha256=source_digest)

    @staticmethod
    def _loss_points(extracted: ExtractedData, indices: np.ndarray, evidence_id: str) -> list[dict[str, Any]]:
        cols = extracted.columns
        codes = np.asarray(extracted.categorical["excitation"][0])
        points: list[dict[str, Any]] = []
        for i in indices.tolist():
            code = int(codes[i])
            point: dict[str, Any] = {
                "frequency": quantity_json(cols["frequency"][0][i], "Hz"),
                "flux_density_peak": quantity_json(cols["flux_density_peak"][0][i], "T"),
                "temperature": quantity_json(cols["temperature"][0][i], "degC"),
                "loss_density": quantity_json(cols["loss_density"][0][i], "W/m^3"),
                "dc_bias_field": quantity_json(cols["dc_bias_field"][0][i], "A/m"),
                "excitation": EXCITATIONS[code], "source_ref": evidence_id, "basis": "measurement",
            }
            dp, dn = float(cols["duty_p"][0][i]), float(cols["duty_n"][0][i])
            if code != 0 and 0.0 < dp < 1.0:
                point["duty"] = dp
                d0 = (1.0 - dp - dn) / 2.0
                point["note"] = f"Duty_P {dp:g} (rising interval), Duty_N {dn:g} (falling interval)" + (f", d2 = d4 = {d0:g}" if code == 2 else "")
            points.append(point)
        return points

    # -- website zip products -----------------------------------------------------------------------
    def website_zip_url(self, product: str, material: str) -> str:
        if product not in WEBSITE_PRODUCTS:
            raise InvalidInputError(f"product must be one of {sorted(WEBSITE_PRODUCTS)}")
        template = self.website_zip_urls.get(product)
        if not template:
            raise DataNotFoundError(
                f"no URL configured for the MagNet website product {product!r}; the upstream repository publishes no stable URL or "
                f"checksum for <material>_{product}.zip - download it manually from {WEBSITE_URL} (or the Dropbox share) and call "
                "import_cycle_zip(path), or pass website_zip_urls={...} with a '{material}' placeholder",
                details={"product": product, "website": WEBSITE_URL})
        return template.format(material=material)

    def fetch_website_zip(self, product: str, material: str, *, allow_download: bool = False, expected_sha256: str | None = None,
                          retrieved_at: str | None = None, timeout: float = 600.0) -> Path:
        """Fetch a website zip through the cache (explicit opt-in; unpinned upstream, digest recorded at download)."""
        url = self.website_zip_url(product, material)
        self.check_license("cache")
        return self.cache.fetch(url, expected_sha256, allow_download, filename=f"{material}_{product}.zip", commit=None,
                                license=UPSTREAM_LICENSE, source_id=SOURCE_ID, retrieved_at=retrieved_at, timeout=timeout)

    def import_cycle_zip(self, path: str | Path, *, destination_dir: str | Path, material: str, dataset_id: str | None = None,
                         members: Mapping[str, str] | None = None, samples_per_period: int | None = None, source_url: str | None = None,
                         material_ref: str | None = None, context: Sequence[MagneticRecord] = (), retrieved_at: str | None = None,
                         max_sequences: int | None = None) -> ImportResult:
        """Import a user-downloaded ``<material>_cycle.zip`` (b_cycle.csv / h_cycle.csv / sampling_time.csv / temperature.csv).

        B [T] and H [A/m] rows are single steady-state periods of 1024 (or 128) samples; the per-row
        sampling time [s] gives ``f = 1/(Ts * n_samples)``; temperature [degC] -> K. Output: one
        waveform file (layout ``aipe.magnetics.waveforms/0.2``) plus ``waveform_dataset`` and ``dataset``
        records with licence MIT and the zip's SHA-256.
        """
        zip_path = Path(path)
        if not zip_path.is_file():
            raise DataNotFoundError(f"zip file not found: {zip_path}")
        if material not in KNOWN_MATERIALS:
            raise InvalidInputError(f"material must be one of {list(KNOWN_MATERIALS)}, got {material!r}")
        names = dict(DEFAULT_CYCLE_MEMBERS)
        names.update(members or {})
        digest = sha256_file(zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            present = set(archive.namelist())
            missing = [names[k] for k in ("flux_density", "field", "sampling_time", "temperature") if names[k] not in present]
            if missing:
                raise DataIntegrityError(f"{zip_path.name}: missing members {missing}; present: {sorted(present)[:12]}")
            b = _read_csv_matrix(archive, names["flux_density"], max_rows=max_sequences)
            h = _read_csv_matrix(archive, names["field"], max_rows=max_sequences)
            ts = _read_csv_matrix(archive, names["sampling_time"], max_rows=max_sequences).reshape(-1)
            temp = _read_csv_matrix(archive, names["temperature"], max_rows=max_sequences).reshape(-1)
            info_text = archive.read(names["info"]).decode("utf-8", "replace") if names.get("info") in present else None
        if b.ndim != 2 or h.shape != b.shape:
            raise DataIntegrityError(f"b_cycle {b.shape} and h_cycle {h.shape} must be 2-D arrays of identical shape")
        n_rows, n_samples = b.shape
        accepted = (samples_per_period,) if samples_per_period else WEBSITE_PRODUCTS["cycle"]["samples_per_period"]
        if n_samples not in accepted:
            raise DataIntegrityError(f"{n_samples} samples per row; expected one of {accepted} (pass samples_per_period= to override)")
        if ts.shape[0] != n_rows or temp.shape[0] != n_rows:
            raise DataIntegrityError(f"row counts differ: B/H {n_rows}, sampling_time {ts.shape[0]}, temperature {temp.shape[0]}")
        if not (np.all(np.isfinite(b)) and np.all(np.isfinite(h)) and np.all(np.isfinite(ts)) and np.all(ts > 0)):
            raise DataIntegrityError("non-finite samples or non-positive sampling times in the cycle files")
        frequency = 1.0 / (ts * n_samples)
        destination_dir = Path(destination_dir)
        token = slug(material)
        base = dataset_token(dataset_id) if dataset_id else f"princeton-magnet-{token}-cycle"
        dataset_id = dataset_id or f"dataset.{base}"
        h5_path = destination_dir / f"{base}.h5"
        if h5_path.exists():
            raise FileExistsError(f"{h5_path} exists; adapters never overwrite")
        item = SourceItem(source_id=SOURCE_ID, item_id=zip_path.name, kind="file", material=material, path=str(zip_path), url=source_url,
                          size_bytes=zip_path.stat().st_size, sha256=digest, available=True, license=UPSTREAM_LICENSE,
                          description="user-downloaded MagNet website cycle zip")
        repo_evidence = self._repository_evidence()
        zip_evidence = make_evidence(
            evidence_id=f"evidence.princeton-magnet-{token}-cycle-zip", kind="measurement",
            reference=f"MagNet website product {material}_cycle.zip (single-cycle B/H waveforms, {n_samples} samples per period)",
            description=("Measured single-cycle flux density and field waveforms of the Princeton MagNet database as downloaded by the user; "
                         "Terms of Access on the download page are MIT-style; the product is unpinned upstream, its SHA-256 is recorded here."
                         + (f" test_info.txt: {info_text[:300]}" if info_text else "")),
            created_at=COMMIT_DATE, license=UPSTREAM_LICENSE, locator=", ".join(names[k] for k in ("flux_density", "field", "sampling_time", "temperature")),
            url=source_url or WEBSITE_URL, version=f"unpinned website product (sha256 {digest})", sha256=digest, accessed_at=retrieved_at)
        evidence = [repo_evidence, zip_evidence]
        normalization = ("B [T], H [A/m] stored as given; frequency derived as 1/(sampling_time * n_samples); temperature degC -> K; "
                         "original units recorded as HDF5 attributes")
        zip_version = f"unpinned website product (sha256 {digest})"
        provenance = self._provenance(f"imported {n_rows} single-cycle sequences from {zip_path.name}", inputs=[e.id for e in evidence],
                                      normalization=normalization, retrieved_at=retrieved_at, version=zip_version)
        raw_uri = source_uri(zip_path, destination_dir)
        warnings = [f"{zip_path.name}: no relative path from {destination_dir}; the raw artifact uses the absolute URI {raw_uri}"] if is_file_uri(raw_uri) else []
        if material_ref is None:
            material_record: MagneticRecord | None = self._material_identity_record(
                material=material, evidence=[repo_evidence], manufacturer=MANUFACTURERS.get(material))
            material_ref = material_record.id
        else:
            material_record = None
        meta = WaveformMeta(source_id=SOURCE_ID, dataset_version=f"website-zip sha256:{digest}", license=UPSTREAM_LICENSE,
                            timebase="uniform_per_period", material_ref=material_ref, source_dataset_ref=dataset_id,
                            description=f"Princeton MagNet single-cycle B/H waveforms of {material}; one period per row, {n_samples} samples")
        store = WaveformStore()
        try:
            properties = store.write(h5_path, {"flux_density": (b, "T"), "field": (h, "A/m")},
                                     {"frequency": (frequency, "Hz"), "temperature": (temp, "degC")}, meta=meta)
            h5_artifact = store.artifact(h5_path, uri=h5_path.name, description="AIPE waveform file (layout aipe.magnetics.waveforms/0.2)")
            raw_artifact = source_artifact(zip_path, media_type="application/zip", sha256=digest, role="raw", uri=raw_uri,
                                           description="original zip as downloaded by the user (unmodified)")
            extensions = {IMPORT_EXTENSION: self._import_extension(source_name=zip_path.name, source_sha256=digest, normalization=normalization,
                                                                   extra={"members": names, "samples_per_period": n_samples, "rows": n_rows,
                                                                          "units": WEBSITE_PRODUCTS["cycle"]["units"]}),
                          SOURCES_EXTENSION: sources_extension(self.descriptor, item, product="cycle", waveform_file=h5_path.name, pinned=None,
                                                               pinned_note="website zips are unpinned upstream; the zip's SHA-256 is the version")}
            waveform_record = record_envelope(
                record_id=f"waveform-dataset.{base}", kind="waveform_dataset",
                name=f"Princeton MagNet single-cycle waveforms - {material}", properties=properties.model_dump(mode="json", exclude_none=True),
                evidence=evidence, license=UPSTREAM_LICENSE, provenance=provenance, artifacts=[h5_artifact, raw_artifact], dataset_ref=dataset_id,
                tags=["external-source", SOURCE_ID, "cycle"], extensions=extensions,
                description=f"{n_rows} measured single-cycle B(t)/H(t) sequences ({n_samples} samples) with frequency and temperature per row.")
            dataset_record = self._dataset_record(
                dataset_id=dataset_id, name=f"Princeton MagNet website cycle product - {material}", evidence=evidence,
                checksums={zip_path.name: digest, h5_path.name: h5_artifact.sha256 or ""}, materials=[material], record_count=n_rows,
                provenance=provenance, artifacts=[h5_artifact, raw_artifact], extensions=extensions, retrieved_at=retrieved_at,
                title=f"Princeton MagNet {material}_cycle.zip (website product)", url=source_url or WEBSITE_URL, version=zip_version,
                description=f"User-downloaded {zip_path.name} (sha256 {digest}); unpinned upstream product imported into {h5_path.name}.")
            records = [dataset_record] + ([material_record] if material_record is not None else []) + [waveform_record]
            cross_validate(records, context)
            written = [str(h5_path)] + write_records(records, destination_dir, reusable_ids=[material_ref])
        except Exception:
            if h5_path.exists():
                h5_path.unlink()
            raise
        return self._import_result(records=records, artifacts=[h5_artifact, raw_artifact], provenance=provenance, warnings=warnings,
                                   written=written, source_sha256=digest)


def _read_csv_matrix(archive: zipfile.ZipFile, member: str, *, max_rows: int | None = None) -> np.ndarray:
    rows: list[list[float]] = []
    with archive.open(member) as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
        for number, cells in enumerate(reader, start=1):
            if not cells or all(not c.strip() for c in cells):
                continue
            try:
                rows.append([float(c) for c in cells])
            except ValueError:
                raise DataIntegrityError(f"{member}: row {number} contains a non-numeric cell") from None
            if rows and len(rows[-1]) != len(rows[0]):
                raise DataIntegrityError(f"{member}: row {number} has {len(rows[-1])} cells, row 1 has {len(rows[0])} (ragged file)")
            if max_rows is not None and len(rows) >= max_rows:
                break
    if not rows:
        raise DataIntegrityError(f"{member}: no numeric rows")
    return np.asarray(rows, dtype=np.float64)


def known_file_checksums() -> dict[str, str]:
    """``<material>_database.h5 -> sha256`` of the packaged files at the pinned commit."""
    return {f"{m}_database.h5": info["sha256"] for m, info in PACKAGED_DATABASES.items()}


__all__ = [
    "SOURCE_ID", "DESCRIPTOR", "CITATION", "COLUMN_MAP", "EXCITATIONS", "MANUFACTURERS", "PACKAGED_DATABASES", "RAW_URL_TEMPLATE",
    "WEBSITE_PRODUCTS", "WEBSITE_URL", "PrincetonMagnetAdapter", "classify_excitation", "known_file_checksums",
]
