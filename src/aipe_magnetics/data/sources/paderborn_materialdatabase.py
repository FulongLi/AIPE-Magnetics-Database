"""Adapter for the Paderborn LEA **materialdatabase** (https://github.com/upb-lea/materialdatabase, GPL-3.0).

Pinned commit ``04fff8e98ae3ddb72809223c42e8c939ed1073bb``. The only licence grant in the repository is
the GPL-3.0 ``LICENSE`` text (the ``pyproject.toml`` classifier says MIT - metadata only, not a grant).
Consequently nothing from the repository is bundled, copied or imported: the adapter **reads a user
checkout** (``root``) or an **explicitly requested** shallow clone of the pinned commit into the local
cache (``allow_download=True``), converts CSV rows into v0.2 records marked ``GPL-3.0-only`` and
refuses ``check_license('bundle')`` / ``check_license('redistribute')``.

Discovery (``<root>/materialdatabase/data`` or the data directory itself)::

    complex_permeability/<source>/<Material>.csv   source in LEA_MTB | MagNet | TDK_MDT | datasheet
    complex_permittivity/LEA_MTB/<Material>.csv
    datasheet_curves/<Material>/<curve_type>.csv   general_information, b_over_h_at_f_T, mu_amplitude_over_b_at_T,
                                                   p_v_over_{b_at_f_T,f_at_b_T,T_at_f_b}, small_signal_mu_*

``datasheet/<Material>.csv`` duplicates ``TDK_MDT/<Material>.csv`` (same rows, no ``h_offset`` column,
different float formatting); duplicates are excluded from discovery unless ``include_duplicates=True``.

Explicit column -> unit mappings (:data:`UNIT_MAPS`, nothing inferred from headers, which carry no
units): ``f`` Hz, ``T`` degC, ``b`` T (peak amplitude), ``h`` A/m, ``mu*`` / ``eps*`` 1 (imaginary parts
stored positive, convention ``mu = mu' - j mu''``), ``h_offset`` A/m, ``p_v`` **W/m^3 per upstream code**
(``functions_TDK_MDT.py`` converts mW/cm^3 x1000 and ``plot.py`` divides by 1000 to label kW/m^3) but
the header is unit-less, so the caller must confirm it with ``confirm_units={'p_v': 'W/m^3'}``.

Known anomalies (audit): ``datasheet_curves/{N27,N30,T37}/b_over_h_at_f_T.csv`` hold **millitesla**
values in the ``b`` column (max 378-485); 45 rows of ``N87/p_v_over_f_at_b_T.csv`` carry ``b = 25``
and ``N87/mu_amplitude_over_b_at_T.csv`` line 38 ``b = 125`` (mixed units within one column); ``3C95``
B-H has an empty ``branch`` column and one 4-field row; ``T37`` B-H equals ``N30`` B-H. Implausible
flux-density magnitudes (> :data:`FLUX_DENSITY_PLAUSIBLE_MAX_T`) are therefore **rejected** with a clear
error; the caller may pass ``unit_overrides={'b': 'mT'}`` (whole column) or
``quarantine_implausible_rows=True`` (mixed columns; rows are dropped and listed in ``quality``).
Quarantine is accepted only when the column really holds two unit populations: at most
:data:`QUARANTINE_MAX_FRACTION` of the rows are implausible, the surviving rows stay below the ferrite
bound :data:`FLUX_DENSITY_FERRITE_MAX_T` (every column of the pinned repository peaks at 0.526 T) and
the smallest implausible value is at least :data:`QUARANTINE_MIN_SCALE_GAP` times the largest plausible
one (the genuine cases show gaps of 125x and 312x); a continuous millitesla curve that merely crosses
2 "T" fails all three tests and needs ``unit_overrides``.

Pinning: the pinned commit is asserted only for files whose git blob id was verified or whose checkout
was verified (cached pinned clone, or a user checkout that ``git`` confirms to be the pristine pinned
commit); otherwise records are versioned ``'unverified copy (sha256 ...)'``; a verified *mismatch* is
refused unless ``allow_unpinned=True`` (records are then flagged ``source_differs_from_pinned_commit``).
"""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from aipe_magnetics.common.errors import DataIntegrityError, DataNotFoundError, InvalidInputError, UnitError
from aipe_magnetics.common.jsonio import sha256_bytes, sha256_file
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.units import units
from aipe_magnetics.data.importers.base import IMPORT_EXTENSION, ImportResult, is_file_uri, source_artifact
from aipe_magnetics.data.records import MagneticRecord
from aipe_magnetics.data.sources.base import (
    SOURCES_EXTENSION,
    ExtractedData,
    PinStatus,
    SourceAdapter,
    SourceDescriptor,
    SourceItem,
    TableMeta,
    TableStore,
    cross_validate,
    dataset_token,
    make_evidence,
    range_problems,
    record_envelope,
    slug,
    source_uri,
    sources_extension,
    write_records,
)
from aipe_magnetics.data.sources.cache import LocalCache

SOURCE_ID = "paderborn_materialdatabase"
REPOSITORY_URL = "https://github.com/upb-lea/materialdatabase"
COMMIT = "04fff8e98ae3ddb72809223c42e8c939ed1073bb"
COMMIT_DATE = "2026-08-27T13:35:09+02:00"
LICENSE = "GPL-3.0-only"
DATA_SUBDIR = Path("materialdatabase") / "data"
FAMILIES: tuple[str, ...] = ("complex_permeability", "complex_permittivity", "datasheet_curves")
PERMEABILITY_SOURCES: tuple[str, ...] = ("LEA_MTB", "MagNet", "TDK_MDT", "datasheet")
#: provenance class per data source (there is no per-row flag upstream)
SOURCE_PROVENANCE: dict[str, str] = {
    "LEA_MTB": "measured on the Paderborn LEA Material Test Bench (capacitive compensated two-winding method, IEEE 6648460)",
    "MagNet": "derived by upstream from Princeton MagNet sinusoidal time series assuming elliptical hysteresis (setpoint T, kHz-rounded f)",
    "TDK_MDT": "computed by upstream from TDK Magnetic Design Tool loss and amplitude-permeability curves (datasheet-derived, not measured)",
    "datasheet": "stale duplicate of TDK_MDT (same rows without h_offset)",
    "datasheet_curves": "digitised / transcribed by upstream from manufacturer datasheets (TDK, Ferroxcube)",
}
#: hard plausibility limit: no soft-magnetic material exceeds 2 T; larger values read as tesla are a unit error
FLUX_DENSITY_PLAUSIBLE_MAX_T = 2.0
#: survivors of a quarantine must look like a ferrite tesla column (the pinned repository holds MnZn/NiZn ferrites only; max 0.526 T)
FLUX_DENSITY_FERRITE_MAX_T = 0.6
#: quarantine is refused when more than this fraction of the rows is implausible (genuine mixed-unit files: 1.4 % and 17.9 %)
QUARANTINE_MAX_FRACTION = 0.25
#: ... or when the smallest implausible value is less than this factor above the largest plausible one (genuine cases: 125x, 312x)
QUARANTINE_MIN_SCALE_GAP = 10.0
#: file type -> upstream column -> (AIPE column, unit as stored upstream); None = ambiguous, confirmation required
UNIT_MAPS: dict[str, dict[str, tuple[str, str | None]]] = {
    "complex_permeability": {"f": ("frequency", "Hz"), "T": ("temperature", "degC"), "b": ("flux_density_peak", "T"),
                             "mu_real": ("permeability_real", "1"), "mu_imag": ("permeability_imag", "1"), "h_offset": ("dc_bias_field", "A/m")},
    "complex_permittivity": {"f": ("frequency", "Hz"), "T": ("temperature", "degC"), "eps_real": ("permittivity_real", "1"),
                             "eps_imag": ("permittivity_imag", "1")},
    "b_over_h_at_f_T": {"h": ("field", "A/m"), "b": ("flux_density", "T"), "T": ("temperature", "degC"), "f": ("frequency", "Hz")},
    "mu_amplitude_over_b_at_T": {"b": ("flux_density_peak", "T"), "mu": ("amplitude_permeability", "1"), "T": ("temperature", "degC")},
    "p_v_over_b_at_f_T": {"b": ("flux_density_peak", "T"), "p_v": ("loss_density", None), "f": ("frequency", "Hz"), "T": ("temperature", "degC")},
    "p_v_over_f_at_b_T": {"f": ("frequency", "Hz"), "p_v": ("loss_density", None), "T": ("temperature", "degC"), "b": ("flux_density_peak", "T")},
    "p_v_over_T_at_f_b": {"T": ("temperature", "degC"), "p_v": ("loss_density", None), "f": ("frequency", "Hz"), "b": ("flux_density_peak", "T")},
    "small_signal_mu_real_over_f_at_T": {"f": ("frequency", "Hz"), "mu_real": ("permeability_real", "1")},
    "small_signal_mu_imag_over_f_at_T": {"f": ("frequency", "Hz"), "mu_imag": ("permeability_imag", "1")},
    "small_signal_mu_initial_over_T": {"T": ("temperature", "degC"), "mu": ("initial_permeability", "1")},
}
CATEGORICAL_COLUMNS: dict[str, dict[str, str]] = {"complex_permeability": {"probe": "probe"}, "complex_permittivity": {"probe": "probe"},
                                                  "b_over_h_at_f_T": {"branch": "branch"}}
#: ambiguous units: what the CSV most likely contains and the evidence; the caller must confirm
AMBIGUOUS_UNITS: dict[str, dict[str, str]] = {
    "p_v": {"expected": "W/m^3",
            "evidence": "header has no unit; upstream data_loader/TDK_MDT/functions_TDK_MDT.py line 106 converts mW/cm^3 x1000 to W/m^3 and "
                        "processing/plot.py divides p_v by 1000 to label kW/m^3; magnitudes 7e2..7e6 are consistent with W/m^3 for MnZn ferrites"},
}
GENERAL_INFORMATION_COLUMNS: dict[str, tuple[str, str | None]] = {
    "application": ("application", None), "material": ("material_class", None), "initial permeability": ("initial_permeability", "1"),
    "curie temperature in C": ("curie_temperature", "degC"), "density in kg/m^3": ("density", "kg/m^3"),
    "resistivity in Ohm m": ("resistivity", "Ohm*m"), "saturation flux density in T at 25 C": ("saturation_flux_density_25C", "T"),
    "saturation flux density in T at 100 C": ("saturation_flux_density_100C", "T"),
}
#: git blob SHA-1 (LF-normalised content) of files verified in the audit at the pinned commit
KNOWN_BLOB_SHA1: dict[str, str] = {
    "complex_permeability/LEA_MTB/N95.csv": "6bce3a8f954d2835456952a46337fa1e6bfcce73",
    "complex_permeability/MagNet/N87.csv": "ece2a1e0c209701b2af9b4a55ba83d737ca5d962",
    "complex_permeability/TDK_MDT/N87.csv": "29265ecf2e3bf590c74604fe3c484872a4f00d10",
    "complex_permeability/datasheet/N87.csv": "8e041852a48cbeb33e0fabb7773377fd4515ed9f",
    "datasheet_curves/N87/general_information.csv": "733ca561480a5a5ab1076f302914a5ed7628a94c",
}
MATERIAL_CLASS_MAP = {"MnZn": "ferrite_mnzn", "NiZn": "ferrite_nizn"}

DESCRIPTOR = SourceDescriptor(
    source_id=SOURCE_ID, title="Paderborn University LEA materialdatabase (upb-lea/materialdatabase)", repository_url=REPOSITORY_URL,
    pinned_commit=COMMIT, pinned_commit_date=COMMIT_DATE, license=LICENSE, redistribution_permitted=False,
    citation="UPB-LEA, materialdatabase 0.4.0 (GPL-3.0), https://github.com/upb-lea/materialdatabase; LEA_MTB method: IEEE Xplore 6648460; "
             "MagNet subsets derived from https://mag-net.princeton.edu/; TDK_MDT subsets derived from the TDK Magnetic Design Tool",
    publisher="Department of Power Electronics and Electrical Drives (LEA), Paderborn University",
    data_locations=[f"{REPOSITORY_URL}/tree/{COMMIT}/materialdatabase/data (79 CSV files, ~4 MB: complex_permeability, complex_permittivity, datasheet_curves)",
                    "PyPI 'materialdatabase' wheel (ships the CSVs; not used by this adapter)"],
    notes=["Licence: LICENSE is the verbatim GPL-3.0 text (no 'or later'); pyproject classifier says MIT (metadata only, not a grant); no data "
           "licence statement; third-party provenance (Princeton MagNet, TDK MDT, manufacturer datasheets) undocumented upstream.",
           "AIPE policy: read a user checkout or an explicit on-demand clone only; never bundle or redistribute (copyleft incompatible with "
           "the NOASSERTION/MIT distribution); produced records carry GPL-3.0-only on record and evidence.",
           "Units: f Hz, T degC, b T (peak), h A/m, mu/eps dimensionless with positive imaginary parts, h_offset A/m; p_v is W/m^3 per upstream "
           "code but headers are unit-less -> caller confirmation required.",
           "Anomalies: N27/N30/T37 B-H files in mT; N87 p_v_over_f b=25 rows and mu_amplitude b=125 row; 3C95 B-H empty branch + one 4-field row; "
           "T37 B-H is a copy of N30 B-H; small_signal_*_over_f_at_T files carry no temperature; N27 MagNet has 2 rows with negative mu_imag; "
           "LEA_MTB N49 has one mu_real outlier (25603).",
           "Never run the upstream Data() class against a cached copy: it rewrites CSVs and writes *_backup files."])


def git_blob_sha1_lf(raw: bytes) -> str:
    """git blob SHA-1 of the LF-normalised bytes (the repository index stores LF; Windows checkouts are CRLF)."""
    content = raw.replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()  # noqa: S324 - git object id, not security


def _parse_csv(raw: bytes, path_label: str) -> tuple[list[str], list[list[str]], list[str]]:
    text = raw.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = [h.strip() for h in next(reader)]
    except StopIteration:
        raise DataIntegrityError(f"{path_label}: empty file") from None
    rows: list[list[str]] = []
    notes: list[str] = []
    for number, cells in enumerate(reader, start=2):
        if not cells or all(not c.strip() for c in cells):
            continue
        cells = [c.strip() for c in cells]
        if len(cells) < len(header):
            notes.append(f"row {number}: {len(cells)} fields for {len(header)} header columns (missing fields read as empty)")
            cells += [""] * (len(header) - len(cells))
        elif len(cells) > len(header):
            raise DataIntegrityError(f"{path_label}: row {number} has {len(cells)} fields but the header has {len(header)}")
        rows.append(cells)
    return header, rows, notes


class PaderbornMaterialDatabaseAdapter(SourceAdapter):
    """Reads the GPL-3.0 materialdatabase CSVs from a checkout or an explicit pinned clone; never bundles them."""

    descriptor = DESCRIPTOR
    importer_id = "aipe.magnetics.sources.paderborn_materialdatabase"
    _license_policy = {"read": True, "cache": True, "bundle": False, "redistribute": False}

    def __init__(self, cache: LocalCache | None = None):
        self.cache = cache or LocalCache()

    # -- root resolution --------------------------------------------------------------------------------
    def resolve_data_dir(self, root: str | Path | None = None, allow_download: bool = False, *, retrieved_at: str | None = None) -> Path:
        """``materialdatabase/data`` of a checkout (``root``), the data dir itself, or the cached pinned clone (verified)."""
        return self._data_dir(root, allow_download, retrieved_at=retrieved_at)[0]

    def _data_dir(self, root: str | Path | None, allow_download: bool, *, retrieved_at: str | None = None) -> tuple[Path, bool | None]:
        """``(data_dir, checkout_verified)``.

        ``checkout_verified`` is True when the directory belongs to a pristine checkout of the pinned commit:
        the cached clone (HEAD and tracked files verified by :meth:`LocalCache.clone_pinned`) or a user
        checkout that ``git`` confirms read-only (:meth:`LocalCache.checkout_problems`); ``None`` when that
        could not be established (no git, not a repository, another commit, modified files).
        """
        if root is not None:
            root = Path(root)
            for candidate in (root / DATA_SUBDIR, root / "data", root):
                if candidate.is_dir() and (candidate / "complex_permeability").is_dir():
                    return candidate, self._user_checkout_verified(candidate)
            raise DataNotFoundError(f"{root} is neither a materialdatabase checkout nor its data directory "
                                    "(expected <root>/materialdatabase/data/complex_permeability)")
        self.check_license("cache")
        checkout = self.cache.clone_pinned(REPOSITORY_URL, COMMIT, allow_download=allow_download, license=LICENSE, source_id=SOURCE_ID,
                                           retrieved_at=retrieved_at)
        data_dir = checkout / DATA_SUBDIR
        if not data_dir.is_dir():
            raise DataIntegrityError(f"pinned clone {checkout} lacks {DATA_SUBDIR}")
        return data_dir, True

    @staticmethod
    def _user_checkout_verified(data_dir: Path) -> bool | None:
        """True only when git reports the enclosing work tree at the pinned commit with no modified tracked file."""
        try:
            problems = LocalCache.checkout_problems(data_dir, COMMIT)
        except Exception:  # noqa: BLE001 - verification is best effort; failure means "not verified", never an import error
            return None
        return True if not problems else None

    # -- discovery ------------------------------------------------------------------------------------
    def discover(self, root: str | Path | None = None, allow_download: bool = False, *, include_duplicates: bool = False,
                 materials: Sequence[str] | None = None, **options: Any) -> list[SourceItem]:
        data_dir, checkout_verified = self._data_dir(root, allow_download)
        wanted = set(materials) if materials else None
        items: list[SourceItem] = []
        for family in FAMILIES:
            base = data_dir / family
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("*.csv")):
                rel = path.relative_to(data_dir).as_posix()
                parts = rel.split("/")
                if family == "datasheet_curves":
                    material, file_type = parts[1], path.stem
                    data_source = "datasheet_curves"
                else:
                    data_source, material, file_type = parts[1], path.stem, family
                if "_backup" in path.stem or (family == "complex_permeability" and "Am-1" in path.stem):
                    continue  # upstream tool artefacts / merged DC-bias variants
                if wanted is not None and material not in wanted:
                    continue
                duplicate_of = None
                if data_source == "datasheet":
                    twin = data_dir / family / "TDK_MDT" / path.name
                    if twin.is_file():
                        duplicate_of = twin.relative_to(data_dir).as_posix()
                        if not include_duplicates:
                            continue
                known = file_type in UNIT_MAPS or file_type == "general_information"
                items.append(SourceItem(
                    source_id=SOURCE_ID, item_id=rel, kind="parameters" if file_type == "general_information" else "table", material=material,
                    path=str(path), url=f"{REPOSITORY_URL}/blob/{COMMIT}/materialdatabase/data/{rel}", size_bytes=path.stat().st_size,
                    available=True, license=LICENSE, description=f"{family}/{data_source} {file_type} for {material}: {SOURCE_PROVENANCE.get(data_source, '')}",
                    extra={"family": family, "data_source": data_source, "file_type": file_type, "relative_path": rel,
                           "duplicate_of": duplicate_of, "unit_map_known": known,
                           "ambiguous_units": sorted(c for c, (_a, u) in UNIT_MAPS.get(file_type, {}).items() if u is None),
                           "known_blob_sha1": KNOWN_BLOB_SHA1.get(rel), "checkout_verified": checkout_verified}))
        if not items:
            raise DataNotFoundError(f"no materialdatabase CSV files found under {data_dir}")
        return items

    # -- inspection -----------------------------------------------------------------------------------
    def inspect(self, item: SourceItem, **options: Any) -> dict[str, Any]:
        path = Path(item.path or "")
        if not path.is_file():
            raise DataNotFoundError(f"file not found: {path}")
        raw = path.read_bytes()
        header, rows, notes = _parse_csv(raw, item.item_id)
        file_type = item.extra.get("file_type", "")
        unit_map = UNIT_MAPS.get(file_type, {})
        info: dict[str, Any] = {
            "item_id": item.item_id, "material": item.material, "path": str(path), "size_bytes": len(raw), "sha256": sha256_bytes(raw),
            "sha256_lf_normalised": sha256_bytes(raw.replace(b"\r\n", b"\n")), "git_blob_sha1_lf": git_blob_sha1_lf(raw),
            "known_blob_sha1": KNOWN_BLOB_SHA1.get(item.item_id), "header": header, "n_rows": len(rows), "line_endings": "CRLF" if b"\r\n" in raw else "LF",
            "unit_map": {c: {"aipe_column": a, "unit": u if u else f"ambiguous - confirm (expected {AMBIGUOUS_UNITS.get(c, {}).get('expected')})"}
                         for c, (a, u) in unit_map.items()},
            "categorical": CATEGORICAL_COLUMNS.get(file_type, {}), "provenance": SOURCE_PROVENANCE.get(item.extra.get("data_source", ""), ""),
            "parse_notes": notes, "license": LICENSE,
        }
        if info["known_blob_sha1"]:
            info["matches_pinned_commit"] = info["git_blob_sha1_lf"] == info["known_blob_sha1"]
        if file_type == "general_information":
            info["values"] = dict(zip(header, rows[0])) if rows else {}
            return info
        ranges: dict[str, Any] = {}
        for i, column in enumerate(header):
            if column in CATEGORICAL_COLUMNS.get(file_type, {}):
                ranges[column] = sorted({r[i] for r in rows})
                continue
            values = np.array([float(r[i]) if r[i] not in ("", "nan", "NaN") else np.nan for r in rows], dtype=np.float64)
            present = values[~np.isnan(values)]
            ranges[column] = {"min": float(present.min()), "max": float(present.max()), "distinct": int(np.unique(present).size),
                              "missing": int(np.isnan(values).sum())} if present.size else {"missing": int(values.size)}
        info["columns"] = ranges
        if "b" in header:
            b_values = ranges["b"]
            if isinstance(b_values, dict) and b_values.get("max", 0) > FLUX_DENSITY_PLAUSIBLE_MAX_T:
                info["flux_density_anomaly"] = (f"column 'b' reaches {b_values['max']:g}; tesla values above {FLUX_DENSITY_PLAUSIBLE_MAX_T} T are implausible "
                                                "(probably millitesla); extraction rejects this file without unit_overrides / quarantine")
        return info

    # -- extraction -----------------------------------------------------------------------------------
    def extract(self, item: SourceItem, *, unit_overrides: Mapping[str, str] | None = None, confirm_units: Mapping[str, str] | None = None,
                quarantine_implausible_rows: bool = False, **filters: Any) -> ExtractedData:
        if filters:
            raise InvalidInputError(f"unknown filters {sorted(filters)}; supported: unit_overrides, confirm_units, quarantine_implausible_rows")
        path = Path(item.path or "")
        if not path.is_file():
            raise DataNotFoundError(f"file not found: {path}")
        raw = path.read_bytes()
        digest = sha256_bytes(raw)
        header, rows, parse_notes = _parse_csv(raw, item.item_id)
        file_type = item.extra.get("file_type", "")
        if file_type == "general_information":
            return self._extract_general_information(item, header, rows, digest, parse_notes, raw)
        unit_map = UNIT_MAPS.get(file_type)
        if unit_map is None:
            raise InvalidInputError(f"{item.item_id}: no documented column->unit mapping for file type {file_type!r}; known types: "
                                    f"{sorted(UNIT_MAPS)}", details={"file_type": file_type})
        overrides = dict(unit_overrides or {})
        confirmations = dict(confirm_units or {})
        categorical_cols = CATEGORICAL_COLUMNS.get(file_type, {})
        unknown_columns = [c for c in header if c not in unit_map and c not in categorical_cols]
        if unknown_columns:
            raise DataIntegrityError(f"{item.item_id}: unexpected columns {unknown_columns} (documented: {sorted(unit_map) + sorted(categorical_cols)})")
        for key in list(overrides) + list(confirmations):
            if key not in unit_map:
                raise InvalidInputError(f"{key!r} is not a numeric column of {file_type} ({sorted(unit_map)})")
        columns: dict[str, tuple[np.ndarray, str]] = {}
        column_sources: dict[str, str] = {}
        notes = list(parse_notes)
        flags: list[str] = []
        quarantined: list[int] = []
        quarantine_summary: dict[str, dict[str, Any]] = {}
        n = len(rows)
        parsed: dict[str, np.ndarray] = {}
        for i, column in enumerate(header):
            if column in categorical_cols:
                continue
            try:
                parsed[column] = np.array([float(r[i]) if r[i] not in ("", "nan", "NaN") else np.nan for r in rows], dtype=np.float64)
            except ValueError as error:
                raise DataIntegrityError(f"{item.item_id}: column {column!r} is not numeric: {error}") from None
        for column, (aipe_name, documented_unit) in unit_map.items():
            if column not in parsed:
                continue
            values = parsed[column]
            unit = overrides.get(column)
            if unit is None:
                if documented_unit is None:
                    confirmed = confirmations.get(column)
                    expected = AMBIGUOUS_UNITS[column]["expected"]
                    if confirmed is None:
                        raise InvalidInputError(
                            f"{item.item_id}: the unit of column {column!r} is not stated in the file; evidence: {AMBIGUOUS_UNITS[column]['evidence']}. "
                            f"Confirm it explicitly with confirm_units={{'{column}': '{expected}'}} (or another unit of the same dimension)",
                            details={"column": column, "expected": expected})
                    unit = confirmed
                    if confirmed != expected:
                        notes.append(f"{column}: caller-confirmed unit {confirmed!r} differs from the documented expectation {expected!r}")
                else:
                    unit = documented_unit
            si_unit = units.si_unit_of(unit)
            if (documented_unit and not units.compatible(unit, documented_unit)) or (documented_unit is None and not units.compatible(unit, "W/m^3")):
                raise UnitError(f"{item.item_id}: unit {unit!r} for column {column!r} has the wrong dimension", details={"column": column, "unit": unit})
            if aipe_name in ("flux_density", "flux_density_peak"):
                tesla = np.abs(np.asarray([units.convert(v, unit, "T") if not np.isnan(v) else np.nan for v in values]))
                bad = np.flatnonzero(tesla > FLUX_DENSITY_PLAUSIBLE_MAX_T)
                if bad.size:
                    # Quarantine is only legitimate for a column holding two unit populations (a few rows typed in mT
                    # among tesla rows). A continuous mT curve that merely crosses 2 "T" must be overridden as a whole.
                    survivors = tesla[tesla <= FLUX_DENSITY_PLAUSIBLE_MAX_T]
                    fraction = bad.size / n
                    surviving_max = float(survivors.max()) if survivors.size else float("nan")
                    quarantined_min = float(tesla[bad].min())
                    scale_gap = quarantined_min / surviving_max if survivors.size and surviving_max > 0 else float("inf")
                    reasons: list[str] = []
                    if fraction > QUARANTINE_MAX_FRACTION:
                        reasons.append(f"{bad.size} of {n} rows ({fraction:.0%}) are implausible (quarantine limit {QUARANTINE_MAX_FRACTION:.0%})")
                    if not survivors.size or surviving_max > FLUX_DENSITY_FERRITE_MAX_T:
                        reasons.append(f"the remaining rows reach {surviving_max:g} T, above the ferrite bound {FLUX_DENSITY_FERRITE_MAX_T} T")
                    if scale_gap < QUARANTINE_MIN_SCALE_GAP:
                        reasons.append(f"the smallest implausible value is only {scale_gap:.2g}x the largest plausible one "
                                       f"(two unit populations need >= {QUARANTINE_MIN_SCALE_GAP:g}x)")
                    mixed_units = not reasons
                    if not mixed_units or not quarantine_implausible_rows:
                        hint = ("pass quarantine_implausible_rows=True to drop the listed rows (two unit populations in one column)" if mixed_units
                                else f"pass unit_overrides={{'{column}': 'mT'}} if the whole column is in millitesla (quarantine is refused because "
                                     + "; ".join(reasons) + ")")
                        raise DataIntegrityError(
                            f"{item.item_id}: {bad.size} of {n} '{column}' values exceed {FLUX_DENSITY_PLAUSIBLE_MAX_T} T when read as {unit!r} "
                            f"(max {np.nanmax(tesla):g} T, e.g. rows {bad[:5].tolist()}); refusing to guess the unit - {hint}",
                            details={"column": column, "rows": bad[:50].tolist(), "max_T": float(np.nanmax(tesla)), "fraction": fraction,
                                     "surviving_max_T": surviving_max, "scale_gap": scale_gap, "quarantine_permitted": mixed_units})
                    quarantined.extend(bad.tolist())
                    quarantine_summary[column] = {"rows": int(bad.size), "fraction": round(fraction, 4), "surviving_max_T": surviving_max,
                                                  "quarantined_min_T": quarantined_min, "scale_gap": round(scale_gap, 2), "unit_read_as": unit}
                    notes.append(f"{column}: {bad.size} of {n} rows ({fraction:.1%}) quarantined as implausible when read as {unit!r}; "
                                 f"surviving maximum {surviving_max:g} T, quarantined minimum {quarantined_min:g} T (scale gap {scale_gap:.0f}x)")
            columns[aipe_name] = (values, unit)
            column_sources[aipe_name] = column
            if unit != documented_unit:
                notes.append(f"{column} -> {aipe_name}: unit {unit!r} taken from the caller ({'override' if column in overrides else 'confirmation'})")
        categorical: dict[str, tuple[np.ndarray, list[str]]] = {}
        for column, aipe_name in categorical_cols.items():
            if column not in header:
                continue
            i = header.index(column)
            labels = [r[i] for r in rows]
            cats = sorted({lab for lab in labels if lab != ""})
            codes = np.array([cats.index(lab) if lab != "" else -1 for lab in labels], dtype=np.int64)
            if (codes < 0).any():
                flags.append(f"{aipe_name}_missing")
                notes.append(f"{column}: {int((codes < 0).sum())} empty labels (stored as missing code -1)")
            categorical[aipe_name] = (codes, cats)
            column_sources[aipe_name] = column
        if quarantine_implausible_rows and "frequency" in columns:
            bad_f = np.flatnonzero(~(np.asarray(columns["frequency"][0]) > 0))
            if bad_f.size:
                quarantined.extend(bad_f.tolist())
                notes.append(f"{bad_f.size} row(s) with non-positive frequency quarantined (0-based data rows {bad_f[:20].tolist()})")
        if quarantined:
            keep = np.setdiff1d(np.arange(n), np.asarray(sorted(set(quarantined))))
            columns = {k: (v[keep], u) for k, (v, u) in columns.items()}
            categorical = {k: (c[keep], cats) for k, (c, cats) in categorical.items()}
            flags.append("rows_quarantined")
            notes.append(f"{len(set(quarantined))} row(s) quarantined in total (0-based data rows {sorted(set(quarantined))[:50]})")
        if not columns:
            raise DataIntegrityError(f"{item.item_id}: no numeric columns found")
        data_source = item.extra.get("data_source", "")
        return ExtractedData(
            source_id=SOURCE_ID, item=item, kind="table", material=item.material, columns=columns, categorical=categorical,
            source_files={item.item_id: digest}, source_paths={item.item_id: str(path)}, column_sources=column_sources,
            metadata={"family": item.extra.get("family"), "data_source": data_source, "file_type": file_type, "provenance": SOURCE_PROVENANCE.get(data_source),
                      "header": header, "n_rows_file": n, "git_blob_sha1_lf": git_blob_sha1_lf(raw), "known_blob_sha1": KNOWN_BLOB_SHA1.get(item.item_id),
                      "checkout_verified": item.extra.get("checkout_verified"), "quarantine": quarantine_summary,
                      "imaginary_sign_convention": "positive imaginary parts; mu = mu' - j mu'', eps = eps' - j eps''"},
            filters={"unit_overrides": overrides, "confirm_units": confirmations, "quarantine_implausible_rows": quarantine_implausible_rows},
            notes=notes, quality_flags=flags)

    def _extract_general_information(self, item: SourceItem, header: list[str], rows: list[list[str]], digest: str, notes: list[str],
                                     raw: bytes) -> ExtractedData:
        if len(rows) != 1:
            raise DataIntegrityError(f"{item.item_id}: expected exactly one data row, found {len(rows)}")
        unknown = [c for c in header if c not in GENERAL_INFORMATION_COLUMNS]
        if unknown:
            raise DataIntegrityError(f"{item.item_id}: unexpected general_information columns {unknown}")
        parameters: dict[str, Any] = {}
        for column, value in zip(header, rows[0]):
            name, unit = GENERAL_INFORMATION_COLUMNS[column]
            if unit is None:
                parameters[name] = {"value": value, "source_column": column}
                continue
            if value == "":
                continue  # missing stays missing
            try:
                number = float(value)
            except ValueError:
                raise DataIntegrityError(f"{item.item_id}: {column!r} = {value!r} is not numeric") from None
            entry: dict[str, Any] = {"value": number, "unit": unit, "source_column": column}
            if name.endswith("_25C"):
                entry["temperature"] = (25.0, "degC")
            elif name.endswith("_100C"):
                entry["temperature"] = (100.0, "degC")
            parameters[name] = entry
        return ExtractedData(source_id=SOURCE_ID, item=item, kind="parameters", material=item.material, parameters=parameters,
                             source_files={item.item_id: digest}, source_paths={item.item_id: item.path or ""},
                             column_sources={name: col for col, (name, _u) in GENERAL_INFORMATION_COLUMNS.items() if col in header},
                             metadata={"family": "datasheet_curves", "data_source": "datasheet_curves", "file_type": "general_information",
                                       "provenance": SOURCE_PROVENANCE["datasheet_curves"], "header": header,
                                       "known_blob_sha1": KNOWN_BLOB_SHA1.get(item.item_id), "git_blob_sha1_lf": git_blob_sha1_lf(raw),
                                       "checkout_verified": item.extra.get("checkout_verified")}, notes=notes)

    # -- validation -----------------------------------------------------------------------------------
    def validate(self, extracted: ExtractedData) -> list[str]:
        problems = super().validate(extracted)
        known = extracted.metadata.get("known_blob_sha1")
        if known and extracted.metadata.get("git_blob_sha1_lf") and extracted.metadata["git_blob_sha1_lf"] != known:
            problems.append(f"{extracted.item.item_id}: git blob SHA-1 differs from the pinned commit ({known[:12]}...)")
        if extracted.kind == "table":
            cols = extracted.columns
            if "frequency" in cols:
                problems += range_problems("frequency", cols["frequency"][0], cols["frequency"][1], minimum=0.0, strict_min=True)
            if "temperature" in cols:
                kelvin = np.asarray([units.convert(v, cols["temperature"][1], "K") if not np.isnan(v) else np.nan for v in cols["temperature"][0]])
                problems += range_problems("temperature", kelvin, "K", minimum=0.0, maximum=1500.0, strict_min=True)
            for name in ("flux_density_peak",):
                if name in cols:
                    problems += range_problems(name, cols[name][0], cols[name][1], minimum=0.0, strict_min=True)
            for name in ("permeability_real", "permittivity_real", "amplitude_permeability", "initial_permeability"):
                if name in cols:
                    problems += range_problems(name, cols[name][0], "1", minimum=0.0, strict_min=True)
            if "permeability_imag" in cols and extracted.metadata.get("file_type") == "complex_permeability":
                negative = int((np.asarray(cols["permeability_imag"][0]) < 0).sum())
                if negative:
                    problems.append(f"permeability_imag: {negative} negative value(s) (upstream convention stores positive loss terms)")
            if "loss_density" in cols:
                problems += range_problems("loss_density", cols["loss_density"][0], cols["loss_density"][1], minimum=0.0, strict_min=True)
        elif extracted.kind == "parameters":
            for name, entry in extracted.parameters.items():
                if isinstance(entry, dict) and "unit" in entry and not np.isfinite(entry["value"]):
                    problems.append(f"{name}: non-finite value")
        return problems

    # -- records --------------------------------------------------------------------------------------
    def to_records(self, extracted: ExtractedData, *, destination_dir: str | Path, dataset_id: str | None = None,
                   material_ref: str | None = None, context: Sequence[MagneticRecord] = (), retrieved_at: str | None = None,
                   allow_unpinned: bool = False, **options: Any) -> ImportResult:
        """Write the table artifact plus ``dataset`` (+ ``measurement`` for measured sources, + identity material) or, for
        ``general_information.csv``, a ``magnetic_material`` record ``material.paderborn-materialdatabase-<grade>-datasheet``.

        ``material_ref`` is always a *reference*: for table files it is the ``material_ref`` of the measurement
        record (it must resolve in ``context``); for ``general_information.csv`` it is recorded as
        ``extensions['aipe.sources']['related_material_ref']`` and never changes the produced record id.
        ``allow_unpinned``: a file whose verified git blob id differs from the pinned commit is refused unless
        ``True`` (records are then flagged ``source_differs_from_pinned_commit``).
        """
        if options:
            raise InvalidInputError(f"unknown options {sorted(options)}")
        if extracted.source_id != SOURCE_ID:
            raise InvalidInputError("to_records expects data extracted by PaderbornMaterialDatabaseAdapter")
        problems = self.validate(extracted)
        blocking = [p for p in problems if "negative value" not in p and "differs from the pinned commit" not in p]
        if blocking:
            raise DataIntegrityError("extracted data failed validation: " + "; ".join(blocking), details={"problems": problems})
        warnings = [p for p in problems if p not in blocking]
        pin = self._unpinned_policy(extracted, allow_unpinned=allow_unpinned)
        material = extracted.material or "unknown"
        item = extracted.item
        rel = item.item_id
        stem = slug(rel.replace(".csv", ""))
        destination_dir = Path(destination_dir)
        source_name, source_digest = next(iter(extracted.source_files.items()))
        data_source = extracted.metadata.get("data_source", "")
        provenance_text = extracted.metadata.get("provenance") or SOURCE_PROVENANCE.get(data_source, "")
        measured = data_source in ("LEA_MTB", "MagNet")
        base = dataset_token(dataset_id) if dataset_id else f"paderborn-materialdatabase-{stem}"
        dataset_id = dataset_id or f"dataset.{base}"
        repo_evidence = self._repository_evidence(
            description="Paderborn LEA materialdatabase repository at the pinned commit; the GPL-3.0 LICENSE is the only licence grant "
                        "(pyproject classifier MIT is metadata only); records derived from it inherit GPL-3.0-only.")
        file_evidence = make_evidence(
            evidence_id=f"evidence.paderborn-materialdatabase-{stem}", kind="measurement" if measured else "source",
            reference=f"upb-lea/materialdatabase materialdatabase/data/{rel}, commit {COMMIT}",
            description=f"{provenance_text}. Columns: {extracted.metadata.get('header')}; licence GPL-3.0-only (repository licence).",
            created_at=COMMIT_DATE, license=LICENSE, locator=rel, url=item.url, version=pin.version or COMMIT, sha256=source_digest,
            accessed_at=retrieved_at)
        evidence = [repo_evidence, file_evidence]
        raw_uri = source_uri(extracted.source_paths[source_name], destination_dir)
        if is_file_uri(raw_uri):
            warnings.append(f"{rel}: no relative path from {destination_dir} to the source file; the raw artifact uses the absolute URI {raw_uri}")
        if pin.pinned is None:
            warnings.append(f"{rel}: neither the file's git blob nor the checkout could be verified against the pinned commit; records are "
                            "versioned by the file's SHA-256")
        if extracted.kind == "parameters":
            return self._parameters_to_records(extracted, destination_dir=destination_dir, dataset_id=dataset_id, evidence=evidence,
                                               material_ref=material_ref, context=context, retrieved_at=retrieved_at, warnings=warnings,
                                               pin=pin, raw_uri=raw_uri)
        normalization = ("columns converted to canonical SI with aipe_magnetics.common.units (temperature degC -> K" +
                         (f"; caller units: {extracted.filters}" if any(extracted.filters.get(k) for k in ("unit_overrides", "confirm_units")) else "") +
                         "); original units kept as HDF5 'original_unit' attributes; imaginary parts kept positive as upstream")
        provenance = self._provenance(f"extracted {extracted.n_rows} rows from {rel}", inputs=[e.id for e in evidence],
                                      normalization=normalization, retrieved_at=retrieved_at, version=pin.version)
        if material_ref is None:
            material_record: MagneticRecord | None = self._material_identity_record(material=material, evidence=[repo_evidence])
            material_ref = material_record.id
        else:
            material_record = None
        table_path = destination_dir / f"{base}-table.h5"
        if table_path.exists():
            raise FileExistsError(f"{table_path} exists; adapters never overwrite")
        meta = TableMeta(source_id=SOURCE_ID, dataset_version=pin.version or COMMIT, license=LICENSE, material_ref=material_ref,
                         source_dataset_ref=dataset_id, description=f"materialdatabase {rel}: {provenance_text}")
        store = TableStore()
        try:
            store.write(table_path, extracted.columns, meta=meta, categorical=extracted.categorical, source_columns=extracted.column_sources)
            table_artifact = store.artifact(table_path, uri=table_path.name, description="extracted table (layout aipe.magnetics.tables/0.2, per-array SHA-256)")
            raw_artifact = source_artifact(Path(extracted.source_paths[source_name]), media_type="text/csv", sha256=source_digest, role="raw",
                                           uri=raw_uri, description="original upstream CSV as read (unmodified)")
            source_facts: dict[str, Any] = dict(pin.extension(), data_source=data_source, checkout_verified=extracted.metadata.get("checkout_verified"))
            if extracted.metadata.get("quarantine"):
                source_facts["quarantine"] = extracted.metadata["quarantine"]
            extensions = {
                IMPORT_EXTENSION: self._import_extension(source_name=rel, source_sha256=source_digest, normalization=normalization,
                                                         extra={"columns": {a: {"source_column": extracted.column_sources.get(a), "unit": u}
                                                                            for a, (_v, u) in extracted.columns.items()},
                                                                "categorical": {a: cats for a, (_c, cats) in extracted.categorical.items()},
                                                                "filters": dict(extracted.filters), "rows": extracted.n_rows,
                                                                "git_blob_sha1_lf": extracted.metadata.get("git_blob_sha1_lf")}),
                SOURCES_EXTENSION: sources_extension(self.descriptor, item, extracted, table_artifact=table_path.name,
                                                     license_notice="GPL-3.0-only: derived records may be used and modified under the GPL; "
                                                                    "AIPE Magnetics does not bundle or redistribute them", **source_facts),
            }
            quality = {"status": "normalized", "flags": list(extracted.quality_flags) + list(pin.flags), "notes": list(extracted.notes) + warnings}
            dataset_record = self._dataset_record(
                dataset_id=dataset_id, name=f"materialdatabase {rel}", evidence=evidence,
                checksums={rel: source_digest, table_path.name: table_artifact.sha256 or ""}, materials=[material], record_count=extracted.n_rows,
                provenance=provenance, artifacts=[table_artifact, raw_artifact], extensions=extensions, retrieved_at=retrieved_at,
                title=f"Paderborn materialdatabase - {rel}", url=item.url, quality=quality, version=pin.version,
                description=f"{extracted.n_rows} rows of {rel} ({provenance_text}); GPL-3.0-only; table artifact {table_path.name}.")
            records = [dataset_record] + ([material_record] if material_record is not None else [])
            if measured:
                instrument = ("LEA Material Test Bench (upstream enum lists LeCroy HDO4104, Wayne Kerr 6500B, ZES Zimmer LMG640)" if data_source == "LEA_MTB"
                              else "Princeton MagNet two-winding V-I setup (values derived by upstream, see method)")
                method = ("capacitive compensated two-winding permeability measurement (upstream README cites IEEE Xplore 6648460); probe codes in the table"
                          if data_source == "LEA_MTB" else
                          "complex permeability derived from MagNet sinusoidal cycles assuming elliptical hysteresis: mu_r from B/H amplitudes, "
                          "loss angle from measured p_v (upstream data_loader/MagNet)")
                probes = extracted.categorical.get("probe")
                specimen = ("probes " + ", ".join(probes[1])) if probes else f"{material} core as measured by Princeton MagNet"
                records.append(record_envelope(
                    record_id=f"measurement.{base}", kind="measurement",
                    name=f"materialdatabase measurement - {rel}",
                    properties={"material_ref": material_ref, "artifact": table_path.name, "instrument": instrument, "method": method,
                                "specimen": specimen, "description": f"{provenance_text}; values are in the table artifact."},
                    evidence=evidence, license=LICENSE, provenance=provenance, artifacts=[table_artifact], dataset_ref=dataset_id,
                    tags=["external-source", SOURCE_ID, data_source], extensions=extensions, quality=quality,
                    description=f"Measurement provenance of {rel} (GPL-3.0-only)."))
            cross_validate(records, context)
            written = [str(table_path)] + write_records(records, destination_dir, reusable_ids=[material_ref])
        except Exception:
            if table_path.exists():
                table_path.unlink()
            raise
        return self._import_result(records=records, artifacts=[table_artifact, raw_artifact], provenance=provenance, warnings=warnings,
                                   written=written, source_sha256=source_digest)

    def _parameters_to_records(self, extracted: ExtractedData, *, destination_dir: Path, dataset_id: str, evidence: Sequence[Any],
                               material_ref: str | None, context: Sequence[MagneticRecord], retrieved_at: str | None, warnings: list[str],
                               pin: PinStatus, raw_uri: str) -> ImportResult:
        material = extracted.material or "unknown"
        token = slug(material)
        item = extracted.item
        rel = item.item_id
        source_name, source_digest = next(iter(extracted.source_files.items()))
        params = extracted.parameters
        # material_ref is a *reference* to a curated record, never the id of the GPL record written here
        record_id = f"material.paderborn-materialdatabase-{token}-datasheet"
        related: str | None = None
        if material_ref is not None:
            if material_ref == record_id:
                raise InvalidInputError(f"material_ref must name a curated record to relate to, not the datasheet record {record_id} itself")
            target = next((r for r in context if r.id == material_ref), None)
            if target is None:
                raise DataIntegrityError(
                    f"material_ref {material_ref!r} is not in the supplied context; general_information.csv always produces its own "
                    f"GPL-3.0-only record {record_id} and records material_ref only as the curated record it relates to "
                    "(extensions['aipe.sources']['related_material_ref']) - pass the curated record in context=[...]",
                    details={"material_ref": material_ref, "record_id": record_id})
            if target.kind not in ("magnetic_material", "core_material"):
                raise InvalidInputError(f"material_ref {material_ref!r} points to a {target.kind} record, not a material",
                                        details={"material_ref": material_ref, "kind": target.kind})
            related = material_ref
        normalization = ("scalar datasheet values converted to SI with Quantity.si (Curie temperature degC -> K; Bsat temperatures degC -> K); "
                         "originals retained; nothing interpolated or added")
        provenance = self._provenance(f"transcribed datasheet scalars of {material} from {rel}", inputs=[e.id for e in evidence],
                                      normalization=normalization, retrieved_at=retrieved_at, version=pin.version)
        ref = evidence[-1].id
        properties: dict[str, Any] = {"grade": material}
        material_class = params.get("material_class", {}).get("value")
        if material_class in MATERIAL_CLASS_MAP:
            properties["material_class"] = MATERIAL_CLASS_MAP[material_class]

        def conditioned(name: str, note: str | None = None) -> dict[str, Any] | None:
            entry = params.get(name)
            if not entry or "unit" not in entry:
                return None
            out: dict[str, Any] = {"value": Quantity.si(entry["value"], entry["unit"]).model_dump(mode="json", exclude_none=True), "source_ref": ref}
            if "temperature" in entry:
                out["temperature"] = Quantity.si(*entry["temperature"]).model_dump(mode="json", exclude_none=True)
            if note:
                out["note"] = note
            return out

        for key, name in (("initial_permeability", "initial_permeability"), ("curie_temperature", "curie_temperature"), ("density", "density"),
                          ("resistivity", "resistivity")):
            value = conditioned(key, "transcribed by upstream from the manufacturer datasheet; conditions not stated in the CSV")
            if value is not None:
                properties[name] = value
        bsat = [v for v in (conditioned("saturation_flux_density_25C", "datasheet value; field strength H not stated in the CSV"),
                            conditioned("saturation_flux_density_100C", "datasheet value; field strength H not stated in the CSV")) if v]
        if bsat:
            properties["saturation_flux_density"] = bsat
        application = params.get("application", {}).get("value")
        properties["description"] = (f"Datasheet parameters of {material} as digitised/transcribed by the upstream materialdatabase (general_information.csv; "
                                     f"application: {application or 'not stated'}; material class: {material_class or 'not stated'}). Upstream digitised the "
                                     "manufacturer datasheet; manufacturer identity and revision are not stated in the file. Licence GPL-3.0-only."
                                     + (f" Relates to the curated record {related}." if related else ""))
        source_facts: dict[str, Any] = dict(pin.extension(), data_source="datasheet_curves", checkout_verified=extracted.metadata.get("checkout_verified"))
        if related:
            source_facts["related_material_ref"] = related
        extensions = {
            IMPORT_EXTENSION: self._import_extension(source_name=rel, source_sha256=source_digest, normalization=normalization,
                                                     extra={"columns": {n: {"source_column": extracted.column_sources.get(n), "unit": e.get("unit")}
                                                                        for n, e in params.items()}}),
            SOURCES_EXTENSION: sources_extension(self.descriptor, item, extracted, license_notice="GPL-3.0-only: not bundled or redistributed by AIPE Magnetics",
                                                 **source_facts),
        }
        raw_artifact = source_artifact(Path(extracted.source_paths[source_name]), media_type="text/csv", sha256=source_digest, role="raw",
                                       uri=raw_uri, description="original upstream CSV as read (unmodified)")
        quality_notes = ["upstream digitised the manufacturer datasheet; values unreviewed"] + warnings
        if related:
            quality_notes.append(f"related curated record: {related} (this GPL-3.0-only record is kept separate from it)")
        material_record = record_envelope(
            record_id=record_id, kind="magnetic_material", name=f"{material} datasheet parameters (Paderborn materialdatabase)",
            properties=properties, evidence=evidence, license=LICENSE, provenance=provenance, artifacts=[raw_artifact], dataset_ref=dataset_id,
            tags=["external-source", SOURCE_ID, "datasheet-derived"], extensions=extensions,
            quality={"status": "normalized", "flags": list(pin.flags), "notes": quality_notes},
            description=properties["description"])
        dataset_record = self._dataset_record(
            dataset_id=dataset_id, name=f"materialdatabase {rel}", evidence=evidence, checksums={rel: source_digest}, materials=[material],
            record_count=len([k for k, v in params.items() if "unit" in v]), provenance=provenance, artifacts=[raw_artifact], extensions=extensions,
            retrieved_at=retrieved_at, title=f"Paderborn materialdatabase - {rel}", url=item.url, version=pin.version,
            quality={"status": "normalized", "flags": list(pin.flags), "notes": list(warnings)} if (pin.flags or warnings) else None,
            description=f"General datasheet information of {material} transcribed upstream; GPL-3.0-only.")
        records = [dataset_record, material_record]
        cross_validate(records, context)
        written = write_records(records, destination_dir)
        return self._import_result(records=records, artifacts=[raw_artifact], provenance=provenance, warnings=warnings, written=written,
                                   source_sha256=source_digest)


__all__ = ["SOURCE_ID", "DESCRIPTOR", "COMMIT", "LICENSE", "DATA_SUBDIR", "FAMILIES", "PERMEABILITY_SOURCES", "SOURCE_PROVENANCE", "UNIT_MAPS",
           "CATEGORICAL_COLUMNS", "AMBIGUOUS_UNITS", "GENERAL_INFORMATION_COLUMNS", "KNOWN_BLOB_SHA1", "FLUX_DENSITY_PLAUSIBLE_MAX_T",
           "FLUX_DENSITY_FERRITE_MAX_T", "QUARANTINE_MAX_FRACTION", "QUARANTINE_MIN_SCALE_GAP", "PaderbornMaterialDatabaseAdapter", "git_blob_sha1_lf"]
