"""Common contract of the external data-source adapters.

Every adapter (:mod:`princeton_magnet`, :mod:`magnet_challenge2`, :mod:`paderborn_materialdatabase`,
:mod:`paderborn_magnethub`) implements :class:`SourceAdapter`:

``discover``   what is available (files / materials / sizes) in a user checkout, an installed package,
               the local cache or - only with ``allow_download=True`` and only when the licence allows
               it - upstream at the pinned commit;
``inspect``    metadata of one item (units, columns, shapes, counts, checksums) without loading whole
               arrays where the container allows it;
``extract``    normalised in-memory data (:class:`ExtractedData`: numpy arrays + explicit units +
               provenance), optionally filtered;
``validate``   integrity problems of extracted data (finite values, monotone time, unit ranges,
               checksum match) - returned, never silently fixed;
``to_records`` v0.2 JSON records (``dataset`` + ``waveform_dataset`` / ``core_loss_measurement`` /
               ``measurement`` / ``magnetic_material``) and HDF5 artifacts with SHA-256 digests, written
               as *new* files into a destination directory;
``attribution`` the :class:`~aipe_magnetics.common.results.SourceReference` list to cite;
``check_license`` the redistribution policy (``read`` / ``cache`` / ``bundle`` / ``redistribute``).

Engineering-integrity rules shared by all adapters: no data is invented; original bytes are never
modified (adapters only read sources and write new files); every produced record carries the source
URL, pinned commit, SHA-256, licence and ``review_status: unreviewed``; there is no unit inference -
every column / array has an explicit, documented unit mapping; SI normalisation goes through
:mod:`aipe_magnetics.common.units` and the original unit is retained.

Table-only data (scalar operating points, datasheet curves) does not fit the waveform layout, which
requires 2-D sequences. :class:`TableStore` therefore writes the sibling layout
``aipe.magnetics.tables/0.2`` with the *same* conventions (canonical SI unit + ``original_unit`` per
array, ``NaN`` for missing values, per-array SHA-256 under ``/checksums``, deterministic files).
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, NamedTuple, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from aipe_magnetics import TOOL_ID, __version__
from aipe_magnetics.common.errors import (
    DataIntegrityError,
    InvalidInputError,
    LicenseRestrictionError,
    UnitError,
    UnsupportedOperationError,
)
from aipe_magnetics.common.jsonio import dump, sha256_file
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import SourceReference
from aipe_magnetics.common.units import units
from aipe_magnetics.data.hdf5 import (
    CHECKSUM_METHOD,
    HDF5_MEDIA_TYPE,
    MISSING_VALUE,
    array_sha256,
    canonicalize_nan,
    convert_array,
    h5py_available,
)
from aipe_magnetics.data.importers.base import ImportResult, import_extension, relative_uri, tool_attribution
from aipe_magnetics.data.records import (
    SCHEMA_VERSION,
    Artifact,
    Evidence,
    MagneticRecord,
    Provenance,
    validate_catalogue,
)
from aipe_magnetics.data.sources.cache import is_windows_reserved_name

TABLE_LAYOUT_VERSION = "aipe.magnetics.tables/0.2"
SOURCES_EXTENSION = "aipe.sources"
#: quality flag of every record produced from a source file that is not byte-identical to the pinned upstream file
UNPINNED_FLAG = "source_differs_from_pinned_commit"
LicenseAction = Literal["read", "cache", "bundle", "redistribute"]
LICENSE_ACTIONS: tuple[str, ...] = ("read", "cache", "bundle", "redistribute")
#: SPDX identifiers the adapters use (plus ``NOASSERTION``); anything else must be an SPDX expression.
KNOWN_LICENSES: tuple[str, ...] = ("MIT", "GPL-3.0-only", "GPL-3.0-or-later", "MPL-2.0", "CC-BY-4.0", "NOASSERTION")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_COLUMN_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SPDX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+\-]*( (AND|OR|WITH) [A-Za-z0-9][A-Za-z0-9.+\-]*)*$")
#: tokens used verbatim in file names (dataset ids, file prefixes): no separators, drive letters, dots or traversal
_FILE_TOKEN_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


# --------------------------------------------------------------------------------------------------
# Descriptors and items
# --------------------------------------------------------------------------------------------------
class SourceDescriptor(BaseModel):
    """Identity, pinned version, licence and redistribution policy of one external source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(description="adapter id, e.g. princeton_magnet")
    title: str = Field(min_length=1)
    repository_url: str = Field(min_length=1)
    pinned_commit: str = Field(description="40-hex git commit the adapter was written and audited against")
    pinned_commit_date: str | None = Field(default=None, description="RFC 3339 committer date of the pinned commit")
    license: str = Field(description="SPDX identifier / expression of the upstream repository, or NOASSERTION")
    redistribution_permitted: bool | None = Field(
        description="True: AIPE may bundle/redistribute with notice; False: never; None: unknown / not assessed")
    citation: str | None = None
    publisher: str | None = None
    data_locations: list[str] = Field(default_factory=list, description="where the data lives (URLs, repo paths)")
    notes: list[str] = Field(default_factory=list)

    @field_validator("source_id")
    @classmethod
    def _source_id(cls, v: str) -> str:
        if not _SOURCE_ID_RE.match(v):
            raise ValueError(f"source_id {v!r} must match {_SOURCE_ID_RE.pattern}")
        return v

    @field_validator("pinned_commit")
    @classmethod
    def _commit(cls, v: str) -> str:
        if not _COMMIT_RE.match(v):
            raise ValueError("pinned_commit must be a 40-character lowercase hexadecimal git commit hash")
        return v

    @field_validator("repository_url")
    @classmethod
    def _url(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError("repository_url must be an https:// URL")
        return v

    @field_validator("license")
    @classmethod
    def _license(cls, v: str) -> str:
        if v != "NOASSERTION" and not _SPDX_RE.match(v):
            raise ValueError(f"license {v!r} must be an SPDX identifier/expression or NOASSERTION")
        return v

    def problems(self) -> list[str]:
        """Consistency problems beyond field validation (used by the verification suite)."""
        out: list[str] = []
        if self.license == "NOASSERTION" and self.redistribution_permitted:
            out.append(f"{self.source_id}: redistribution cannot be permitted without a licence")
        if not self.data_locations:
            out.append(f"{self.source_id}: no data locations documented")
        if self.license not in KNOWN_LICENSES:
            out.append(f"{self.source_id}: licence {self.license!r} is not one of the audited identifiers {KNOWN_LICENSES}")
        if self.license != "NOASSERTION" and not self.citation:
            out.append(f"{self.source_id}: licensed source without citation text")
        return out

    def record_id_token(self) -> str:
        return self.source_id.replace("_", "-")


class SourceItem(BaseModel):
    """One discoverable unit of a source (a file, a material, a package data directory, a remote object)."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    item_id: str = Field(min_length=1, description="stable within the source, e.g. 'N87_database.h5'")
    kind: Literal["file", "directory", "checkout", "package_data", "remote", "table", "sequences", "parameters"]
    material: str | None = None
    path: str | None = Field(default=None, description="local path when available")
    url: str | None = Field(default=None, description="upstream location (pinned) when known")
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$",
                               description="known or computed digest of the original bytes")
    available: bool = Field(description="True when the bytes are readable locally without any download")
    license: str = "NOASSERTION"
    description: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ExtractedData:
    """Normalised in-memory data of one item: numpy arrays with *explicit* units plus provenance.

    ``kind='table'`` uses ``columns`` (name -> (1-D values, unit as in the source)) and optional
    ``categorical`` columns (name -> (int codes, categories)); ``kind='sequences'`` uses ``arrays``
    (quantity -> (2-D values, unit)) and ``conditions`` (name -> (1-D values, unit));
    ``kind='parameters'`` carries scalar datasheet parameters in ``parameters``. Units are the units
    the source states (documented per adapter); SI normalisation happens in :meth:`SourceAdapter.to_records`.
    """

    source_id: str
    item: SourceItem
    kind: Literal["table", "sequences", "parameters"]
    material: str | None = None
    columns: dict[str, tuple[np.ndarray, str]] = field(default_factory=dict)
    categorical: dict[str, tuple[np.ndarray, list[str]]] = field(default_factory=dict)
    arrays: dict[str, tuple[np.ndarray, str]] = field(default_factory=dict)
    conditions: dict[str, tuple[np.ndarray, str]] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    timebase: Literal["uniform_per_period", "explicit_time_array", "uniform_absolute"] = "uniform_per_period"
    sampling_rate: tuple[float, str] | None = None
    source_files: dict[str, str] = field(default_factory=dict, metadata={"doc": "file name -> sha256 of original bytes"})
    source_paths: dict[str, str] = field(default_factory=dict, metadata={"doc": "file name -> local path"})
    column_sources: dict[str, str] = field(default_factory=dict, metadata={"doc": "AIPE column -> upstream column"})
    metadata: dict[str, Any] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        if self.kind == "table":
            first = next(iter(self.columns.values()), None)
            return 0 if first is None else int(np.asarray(first[0]).shape[0])
        if self.kind == "sequences":
            first = next(iter(self.arrays.values()), None)
            return 0 if first is None else int(np.asarray(first[0]).shape[0])
        return len(self.parameters)

    def unit_mapping(self) -> dict[str, str]:
        """Explicit column / array -> unit mapping as used for this extraction."""
        out = {name: unit for name, (_values, unit) in self.columns.items()}
        out.update({name: unit for name, (_values, unit) in self.arrays.items()})
        out.update({name: unit for name, (_values, unit) in self.conditions.items()})
        return out


# --------------------------------------------------------------------------------------------------
# Table store (aipe.magnetics.tables/0.2)
# --------------------------------------------------------------------------------------------------
class TableMeta(BaseModel):
    """File-level metadata of a table file (nothing inferred)."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1, description="upstream commit / release / date")
    license: str = Field(default="NOASSERTION", min_length=1)
    description: str | None = None
    material_ref: str | None = None
    source_dataset_ref: str | None = None


class TableColumnLayout(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    name: str
    kind: Literal["numeric", "categorical"] = "numeric"
    unit: str | None = Field(default=None, description="canonical SI unit (numeric columns)")
    original_unit: str | None = None
    dtype: str = "float64"
    shape: list[int] = Field(default_factory=list)
    missing_value: str = MISSING_VALUE
    source_column: str | None = None
    categories: list[str] | None = None


class TableDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")
    layout_version: str = TABLE_LAYOUT_VERSION
    n_rows: int = Field(ge=0)
    columns: list[TableColumnLayout]
    material_ref: str | None = None
    source_dataset_ref: str | None = None
    description: str | None = None


@dataclass
class TableData:
    columns: dict[str, np.ndarray]
    units: dict[str, str]
    original_units: dict[str, str]
    categorical: dict[str, np.ndarray]
    categories: dict[str, list[str]]
    attrs: dict[str, Any]
    checksums: dict[str, str] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        first = next(iter(self.columns.values()), None)
        if first is None:
            first = next(iter(self.categorical.values()), None)
        return 0 if first is None else int(first.shape[0])

    def labels(self, name: str) -> list[str | None]:
        codes, cats = self.categorical[name], self.categories[name]
        return [None if c < 0 else cats[int(c)] for c in codes.tolist()]


def _require_h5py() -> Any:
    if not h5py_available():
        raise UnsupportedOperationError("HDF5 table support needs h5py: install aipe-magnetics[hdf5]", details={"extra": "hdf5"})
    import h5py  # type: ignore

    return h5py


def _py(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def codes_sha256(codes: Any) -> str:
    """SHA-256 of int64 little-endian C-order bytes (categorical code columns)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(codes, dtype="<i8")).tobytes(order="C")).hexdigest()


def _validate_table_columns(columns: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, str], int]:
    if not isinstance(columns, Mapping) or not columns:
        raise InvalidInputError("columns must be a non-empty mapping name -> (values, unit)")
    converted: dict[str, np.ndarray] = {}
    si_units: dict[str, str] = {}
    original: dict[str, str] = {}
    n_rows: int | None = None
    for name, item in columns.items():
        if not isinstance(name, str) or not _COLUMN_RE.match(name):
            raise InvalidInputError(f"column name {name!r} must match {_COLUMN_RE.pattern}")
        if not isinstance(item, (tuple, list)) or len(item) != 2 or not isinstance(item[1], str):
            raise InvalidInputError(f"{name}: supply columns as (values, unit) pairs with an explicit unit string",
                                    details={"column": name})
        values, unit = item
        array = np.asarray(values)
        if array.dtype.kind not in "iuf":
            raise InvalidInputError(f"{name}: numeric column required, got dtype {array.dtype}", details={"column": name})
        array = np.asarray(array, dtype=np.float64)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1 or array.shape[0] < 1:
            raise InvalidInputError(f"{name}: columns must be 1-D with at least one row, got shape {array.shape}")
        if np.isinf(array).any():
            raise InvalidInputError(f"{name}: infinite values are not allowed (missing values must be NaN)")
        if n_rows is None:
            n_rows = int(array.shape[0])
        elif array.shape[0] != n_rows:
            raise InvalidInputError(f"{name}: {array.shape[0]} rows but other columns have {n_rows}")
        si_unit = units.si_unit_of(unit)  # raises UnitError for unknown units
        converted[name] = canonicalize_nan(convert_array(array, unit, si_unit))
        si_units[name] = si_unit
        original[name] = unit
    assert n_rows is not None
    return converted, si_units, original, n_rows


def _validate_categorical(categorical: Mapping[str, Any] | None, n_rows: int) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    codes_out: dict[str, np.ndarray] = {}
    cats_out: dict[str, list[str]] = {}
    if not categorical:
        return codes_out, cats_out
    for name, item in categorical.items():
        if not isinstance(name, str) or not _COLUMN_RE.match(name):
            raise InvalidInputError(f"categorical column name {name!r} must match {_COLUMN_RE.pattern}")
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise InvalidInputError(f"{name}: supply categorical columns as (codes, categories)")
        codes = np.asarray(item[0])
        categories = [str(c) for c in item[1]]
        if codes.dtype.kind not in "iu" or codes.ndim != 1 or codes.shape[0] != n_rows:
            raise InvalidInputError(f"{name}: codes must be a 1-D integer array of {n_rows} rows, got dtype {codes.dtype} shape {codes.shape}")
        if len(set(categories)) != len(categories) or not categories:
            raise InvalidInputError(f"{name}: categories must be a non-empty list of unique strings")
        if codes.min() < -1 or codes.max() >= len(categories):
            raise InvalidInputError(f"{name}: codes must lie in -1 (missing) .. {len(categories) - 1}")
        codes_out[name] = np.asarray(codes, dtype=np.int64)
        cats_out[name] = categories
    return codes_out, cats_out


class TableStore:
    """Writer / reader / verifier of the ``aipe.magnetics.tables/0.2`` HDF5 layout.

    Layout::

        /  attrs: layout_version, schema_version, created_by_tool, created_by_version, dataset_version,
                  license, source_id, checksum_method, missing_value, n_rows, description?, material_ref?
        /conditions/<name>   float64 [n_rows]  attrs: quantity, unit (canonical SI), original_unit,
                                               missing_value "NaN", axis0 "row", source_column?
        /categories/<name>   int64 [n_rows]    attrs: categories (JSON list), missing_code -1, source_column?
        /checksums           attrs: "conditions/<name>" / "categories/<name>" -> SHA-256 of canonical bytes
    """

    layout_version = TABLE_LAYOUT_VERSION

    def __init__(self, *, compression: str = "gzip", compression_level: int = 4):
        if compression_level < 0 or compression_level > 9:
            raise InvalidInputError("compression_level must be in 0..9")
        self.compression, self.compression_level = compression, int(compression_level)

    def write(self, path: str | Path, columns: Mapping[str, Any], *, meta: TableMeta,
              categorical: Mapping[str, Any] | None = None, source_columns: Mapping[str, str] | None = None,
              overwrite: bool = False) -> TableDescription:
        lib = _require_h5py()
        if not isinstance(meta, TableMeta):
            raise InvalidInputError("meta must be a TableMeta instance")
        path = Path(path)
        numeric, si_units, original, n_rows = _validate_table_columns(columns)
        codes, categories = _validate_categorical(categorical, n_rows)
        overlap = set(numeric) & set(codes)
        if overlap:
            raise InvalidInputError(f"column names used both numerically and categorically: {sorted(overlap)}")
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        sources = dict(source_columns or {})
        checksums: dict[str, str] = {}
        with lib.File(path, "w") as handle:
            handle.attrs["layout_version"] = TABLE_LAYOUT_VERSION
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["created_by_tool"] = TOOL_ID
            handle.attrs["created_by_version"] = __version__
            handle.attrs["dataset_version"] = meta.dataset_version
            handle.attrs["license"] = meta.license
            handle.attrs["source_id"] = meta.source_id
            handle.attrs["checksum_method"] = CHECKSUM_METHOD
            handle.attrs["missing_value"] = MISSING_VALUE
            handle.attrs["n_rows"] = np.int64(n_rows)
            if meta.description:
                handle.attrs["description"] = meta.description
            if meta.material_ref:
                handle.attrs["material_ref"] = meta.material_ref
            if meta.source_dataset_ref:
                handle.attrs["source_dataset_ref"] = meta.source_dataset_ref
            group = handle.create_group("conditions")
            chunk = (min(n_rows, 1 << 16),)
            for name, array in numeric.items():
                dataset = group.create_dataset(name, data=array, dtype="<f8", chunks=chunk, compression=self.compression,
                                               compression_opts=self.compression_level, shuffle=True, track_times=False)
                dataset.attrs["quantity"] = name
                dataset.attrs["unit"] = si_units[name]
                dataset.attrs["original_unit"] = original[name]
                dataset.attrs["missing_value"] = MISSING_VALUE
                dataset.attrs["axis0"] = "row"
                if name in sources:
                    dataset.attrs["source_column"] = sources[name]
                checksums[f"conditions/{name}"] = array_sha256(array)
            cgroup = handle.create_group("categories")
            for name, array in codes.items():
                dataset = cgroup.create_dataset(name, data=array, dtype="<i8", chunks=chunk, compression=self.compression,
                                                compression_opts=self.compression_level, shuffle=True, track_times=False)
                dataset.attrs["quantity"] = name
                dataset.attrs["categories"] = json.dumps(categories[name], ensure_ascii=False)
                dataset.attrs["missing_code"] = np.int64(-1)
                dataset.attrs["axis0"] = "row"
                if name in sources:
                    dataset.attrs["source_column"] = sources[name]
                checksums[f"categories/{name}"] = codes_sha256(array)
            sums = handle.create_group("checksums")
            sums.attrs["method"] = CHECKSUM_METHOD
            for key, digest in checksums.items():
                sums.attrs[key] = digest
        return self._description(meta, numeric, si_units, original, codes, categories, sources, n_rows)

    @staticmethod
    def _description(meta: TableMeta, numeric: Mapping[str, np.ndarray], si_units: Mapping[str, str],
                     original: Mapping[str, str], codes: Mapping[str, np.ndarray], categories: Mapping[str, list[str]],
                     sources: Mapping[str, str], n_rows: int) -> TableDescription:
        # HDF5 lists group members alphabetically, so describe() and write() agree on the order
        layouts = [TableColumnLayout(path=f"/conditions/{name}", name=name, kind="numeric", unit=si_units[name],
                                     original_unit=original[name], dtype="float64", shape=[n_rows],
                                     source_column=sources.get(name)) for name in sorted(numeric)]
        layouts += [TableColumnLayout(path=f"/categories/{name}", name=name, kind="categorical", dtype="int64",
                                      shape=[n_rows], categories=list(categories[name]), source_column=sources.get(name))
                    for name in sorted(codes)]
        return TableDescription(n_rows=n_rows, columns=layouts, material_ref=meta.material_ref,
                                source_dataset_ref=meta.source_dataset_ref, description=meta.description)

    def read(self, path: str | Path, *, verify: bool = True) -> TableData:
        lib = _require_h5py()
        path = Path(path)
        if not path.is_file():
            raise DataIntegrityError(f"{path}: file not found", details={"path": str(path)})
        with lib.File(path, "r") as handle:
            attrs = {key: _py(value) for key, value in handle.attrs.items()}
            if attrs.get("layout_version") != TABLE_LAYOUT_VERSION:
                raise DataIntegrityError(f"{path}: not an {TABLE_LAYOUT_VERSION!r} file (layout_version={attrs.get('layout_version')!r})")
            columns: dict[str, np.ndarray] = {}
            si_units: dict[str, str] = {}
            original: dict[str, str] = {}
            if "conditions" in handle:
                for name in sorted(handle["conditions"]):
                    dataset = handle["conditions"][name]
                    columns[name] = np.asarray(dataset[()], dtype=np.float64)
                    si_units[name] = str(_py(dataset.attrs.get("unit", "")))
                    original[name] = str(_py(dataset.attrs.get("original_unit", si_units[name])))
            codes: dict[str, np.ndarray] = {}
            categories: dict[str, list[str]] = {}
            if "categories" in handle:
                for name in sorted(handle["categories"]):
                    dataset = handle["categories"][name]
                    codes[name] = np.asarray(dataset[()], dtype=np.int64)
                    categories[name] = [str(c) for c in json.loads(str(_py(dataset.attrs.get("categories", "[]"))))]
            checksums = {}
            if "checksums" in handle:
                checksums = {key: str(_py(value)) for key, value in handle["checksums"].attrs.items() if key != "method"}
        if not columns and not codes:
            raise DataIntegrityError(f"{path}: table holds no columns")
        data = TableData(columns=columns, units=si_units, original_units=original, categorical=codes, categories=categories,
                         attrs=attrs, checksums=checksums)
        if verify:
            mismatches = self._mismatches(data)
            if mismatches:
                raise DataIntegrityError(f"{path}: checksum verification failed: " + "; ".join(mismatches),
                                         details={"path": str(path), "mismatches": mismatches})
        return data

    @staticmethod
    def _mismatches(data: TableData) -> list[str]:
        problems: list[str] = []
        stored = dict(data.checksums)
        if not stored:
            return ["no checksums recorded in the file"]
        for name, array in sorted(data.columns.items()):
            expected = stored.pop(f"conditions/{name}", None)
            actual = array_sha256(array)
            if expected is None:
                problems.append(f"conditions/{name}: no checksum recorded")
            elif expected != actual:
                problems.append(f"conditions/{name}: expected {expected}, computed {actual}")
        for name, array in sorted(data.categorical.items()):
            expected = stored.pop(f"categories/{name}", None)
            actual = codes_sha256(array)
            if expected is None:
                problems.append(f"categories/{name}: no checksum recorded")
            elif expected != actual:
                problems.append(f"categories/{name}: expected {expected}, computed {actual}")
        for key in sorted(stored):
            problems.append(f"{key}: checksum recorded but array missing")
        return problems

    def verify_checksums(self, path: str | Path) -> list[str]:
        return self._mismatches(self.read(path, verify=False))

    def describe(self, path: str | Path) -> TableDescription:
        """Layout description from attributes only (arrays are not loaded)."""
        lib = _require_h5py()
        path = Path(path)
        if not path.is_file():
            raise DataIntegrityError(f"{path}: file not found", details={"path": str(path)})
        with lib.File(path, "r") as handle:
            attrs = {key: _py(value) for key, value in handle.attrs.items()}
            if attrs.get("layout_version") != TABLE_LAYOUT_VERSION:
                raise DataIntegrityError(f"{path}: not an {TABLE_LAYOUT_VERSION!r} file")
            layouts: list[TableColumnLayout] = []
            n_rows = int(attrs.get("n_rows", 0))
            if "conditions" in handle:
                for name in sorted(handle["conditions"]):
                    dataset = handle["conditions"][name]
                    layouts.append(TableColumnLayout(
                        path=f"/conditions/{name}", name=name, kind="numeric", unit=str(_py(dataset.attrs["unit"])),
                        original_unit=str(_py(dataset.attrs.get("original_unit", dataset.attrs["unit"]))),
                        dtype=str(dataset.dtype.name), shape=[int(s) for s in dataset.shape],
                        missing_value=str(_py(dataset.attrs.get("missing_value", MISSING_VALUE))),
                        source_column=_py(dataset.attrs.get("source_column")) if "source_column" in dataset.attrs else None))
            if "categories" in handle:
                for name in sorted(handle["categories"]):
                    dataset = handle["categories"][name]
                    layouts.append(TableColumnLayout(
                        path=f"/categories/{name}", name=name, kind="categorical", dtype=str(dataset.dtype.name),
                        shape=[int(s) for s in dataset.shape],
                        categories=[str(c) for c in json.loads(str(_py(dataset.attrs.get("categories", "[]"))))],
                        source_column=_py(dataset.attrs.get("source_column")) if "source_column" in dataset.attrs else None))
        return TableDescription(n_rows=n_rows, columns=layouts, material_ref=attrs.get("material_ref"),
                                source_dataset_ref=attrs.get("source_dataset_ref"), description=attrs.get("description"))

    @staticmethod
    def artifact(path: str | Path, *, uri: str | None = None, description: str | None = None) -> Artifact:
        path = Path(path)
        if not path.is_file():
            raise DataIntegrityError(f"{path}: file not found", details={"path": str(path)})
        return Artifact(uri=uri or path.name, media_type=HDF5_MEDIA_TYPE, role="normalized", sha256=sha256_file(path),
                        size_bytes=path.stat().st_size, description=description)


# --------------------------------------------------------------------------------------------------
# Generic integrity helpers
# --------------------------------------------------------------------------------------------------
def finite_problems(name: str, values: Any, *, allow_nan: bool = False) -> list[str]:
    array = np.asarray(values, dtype=np.float64)
    problems: list[str] = []
    if np.isinf(array).any():
        problems.append(f"{name}: {int(np.isinf(array).sum())} infinite value(s)")
    if not allow_nan and np.isnan(array).any():
        problems.append(f"{name}: {int(np.isnan(array).sum())} NaN value(s) where none are allowed")
    return problems


def range_problems(name: str, values: Any, unit: str, *, minimum: float | None = None, maximum: float | None = None,
                   strict_min: bool = False) -> list[str]:
    """Report values outside ``[minimum, maximum]`` (in the stated ``unit``), ignoring NaN."""
    array = np.asarray(values, dtype=np.float64)
    present = array[~np.isnan(array)]
    problems: list[str] = []
    if present.size == 0:
        return problems
    if minimum is not None:
        bad = present <= minimum if strict_min else present < minimum
        if bad.any():
            problems.append(f"{name}: {int(bad.sum())} value(s) {'<=' if strict_min else '<'} {minimum} {unit} (min {present.min():g})")
    if maximum is not None and (present > maximum).any():
        problems.append(f"{name}: {int((present > maximum).sum())} value(s) > {maximum} {unit} (max {present.max():g})")
    return problems


def monotone_time_problems(time_values: Any) -> list[str]:
    array = np.asarray(time_values, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    bad_rows = np.where(~np.all(np.diff(array, axis=1) > 0, axis=1))[0]
    return [f"time: {len(bad_rows)} sequence(s) are not strictly increasing (first row {int(bad_rows[0])})"] if bad_rows.size else []


def checksum_problems(source_files: Mapping[str, str], source_paths: Mapping[str, str]) -> list[str]:
    """Recompute the digest of every still-present source file and compare with the recorded one."""
    problems: list[str] = []
    for name, digest in sorted(source_files.items()):
        path = source_paths.get(name)
        if path is None or not Path(path).is_file():
            continue
        actual = sha256_file(path)
        if actual != digest:
            problems.append(f"{name}: source bytes changed since extraction (recorded {digest[:12]}..., now {actual[:12]}...)")
    return problems


def unit_problems(mapping: Mapping[str, str], expected: Mapping[str, str]) -> list[str]:
    """Every column of ``expected`` must be present with a unit compatible with the expected SI unit."""
    problems: list[str] = []
    for name, si_unit in expected.items():
        unit = mapping.get(name)
        if unit is None:
            problems.append(f"{name}: no explicit unit recorded")
            continue
        try:
            if not units.compatible(unit, si_unit):
                problems.append(f"{name}: unit {unit!r} is not compatible with {si_unit!r}")
        except UnitError as error:
            problems.append(f"{name}: {error.message}")
    return problems


# --------------------------------------------------------------------------------------------------
# Record helpers shared by the adapters
# --------------------------------------------------------------------------------------------------
def slug(text: str) -> str:
    """Lower-case id token: ``'R34.0X20.5X12.5' -> 'r34-0x20-5x12-5'``, ``'3C90' -> '3c90'``."""
    token = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    if not token:
        raise InvalidInputError(f"cannot build an id token from {text!r}")
    return token


def make_evidence(*, evidence_id: str, kind: str, reference: str, description: str, created_at: str, license: str,
                  locator: str | None = None, url: str | None = None, version: str | None = None,
                  sha256: str | None = None, accessed_at: str | None = None) -> Evidence:
    return Evidence(id=evidence_id, kind=kind, reference=reference, description=description, created_at=created_at,  # type: ignore[arg-type]
                    review_status="unreviewed", license=license, locator=locator, url=url, version=version,
                    sha256=sha256, accessed_at=accessed_at)


def record_envelope(*, record_id: str, kind: str, name: str, properties: Mapping[str, Any], evidence: Sequence[Evidence],
                    license: str, provenance: Provenance, description: str | None = None, synthetic: bool = False,
                    artifacts: Sequence[Artifact] = (), dataset_ref: str | None = None, tags: Iterable[str] = (),
                    extensions: Mapping[str, Any] | None = None, quality: Mapping[str, Any] | None = None) -> MagneticRecord:
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "id": record_id, "kind": kind, "name": name, "synthetic": synthetic,
        "license": license, "properties": dict(properties),
        "evidence_refs": [e.id for e in evidence], "evidence": [e.model_dump(mode="json", exclude_none=True) for e in evidence],
        "provenance": provenance.model_dump(mode="json", exclude_none=True),
        "artifacts": [a.model_dump(mode="json", exclude_none=True) for a in artifacts],
        "tags": list(tags), "extensions": dict(extensions or {}),
    }
    if description is not None:
        data["description"] = description
    if dataset_ref is not None:
        data["dataset_ref"] = dataset_ref
    if quality is not None:
        data["quality"] = dict(quality)
    try:
        return MagneticRecord.model_validate(data)
    except Exception as error:  # noqa: BLE001 - re-raised as an explicit input problem
        raise InvalidInputError(f"adapter produced an invalid {kind} record {record_id}: {error}") from None


def file_safe_token(token: Any, *, label: str = "token") -> str:
    """A token that may be used verbatim in file names on every platform.

    Accepted: ``^[a-z0-9]+(-[a-z0-9]+)*$`` (so no path separators, drive letters such as ``a:``, dots or
    ``..`` traversal) and not a Windows reserved device name (``CON``, ``NUL``, ``COM1`` ...). The check
    runs *before* any file is created; a violation is an :class:`InvalidInputError`, never a raw OS error.
    """
    if not isinstance(token, str) or not _FILE_TOKEN_RE.match(token):
        raise InvalidInputError(f"{label} {token!r} must match {_FILE_TOKEN_RE.pattern} (lower-case letters, digits and single hyphens; "
                                "it is used verbatim in file names)", details={"token": token})
    if is_windows_reserved_name(token):
        raise InvalidInputError(f"{label} {token!r} is a reserved device name on Windows and cannot name a file", details={"token": token})
    return token


def dataset_token(dataset_id: str) -> str:
    """``'dataset.<token>' -> '<token>'``; the token names every sibling record and artifact of one import.

    The token is validated with :func:`file_safe_token` because it becomes part of HDF5 / JSON file names.
    """
    if not isinstance(dataset_id, str) or not dataset_id.startswith("dataset.") or len(dataset_id) <= len("dataset."):
        raise InvalidInputError(f"dataset_id must look like 'dataset.<token>', got {dataset_id!r}")
    return file_safe_token(dataset_id[len("dataset."):], label="dataset_id token")


def source_uri(source_path: str | Path, destination_dir: str | Path, *, pinned_url: str | None = None) -> str:
    """Artifact URI of an original (raw) source file.

    ``Artifact.uri`` is "a URI or a path relative to the record file". When the local bytes are verified to
    equal the pinned upstream file, ``pinned_url`` (the raw URL at the pinned commit) is the most durable
    reference and is returned. Otherwise the POSIX path of the file relative to ``destination_dir`` is
    returned (the same convention as :mod:`aipe_magnetics.data.importers.hdf5_waveforms`); when no relative
    path exists (another Windows drive) it is an absolute ``file:`` URI.
    """
    if pinned_url:
        return pinned_url
    return relative_uri(Path(source_path), Path(destination_dir) / "record.json")


class PinStatus(NamedTuple):
    """Outcome of :meth:`SourceAdapter._unpinned_policy` (see there)."""

    mismatch: dict[str, str] | None
    version: str | None
    flags: list[str]
    pinned: bool | None

    def extension(self) -> dict[str, Any]:
        """Keys for ``extensions['aipe.sources']``: ``pinned`` (True / False / None) and the mismatch facts."""
        out: dict[str, Any] = {"pinned": self.pinned}
        if self.mismatch is not None:
            out["pinned_mismatch"] = dict(self.mismatch)
        return out


def pinned_mismatch(extracted: ExtractedData) -> dict[str, str] | None:
    """Describe how the extracted source file differs from the pinned upstream file, or ``None`` when it matches.

    Adapters record the pinned digest in ``metadata['expected_sha256']`` (byte identity) or, for files whose
    git blob id was verified at the pinned commit, in ``metadata['known_blob_sha1']`` together with the
    computed ``metadata['git_blob_sha1_lf']``. When neither is available the file cannot be checked and
    ``None`` is returned (the adapter then documents that in ``extensions['aipe.sources']['pinned']``).
    """
    expected = extracted.metadata.get("expected_sha256")
    if expected:
        for name, digest in extracted.source_files.items():
            if digest != expected:
                return {"file": name, "expected_sha256": expected, "actual_sha256": digest}
        return None
    known = extracted.metadata.get("known_blob_sha1")
    actual_blob = extracted.metadata.get("git_blob_sha1_lf")
    if known and actual_blob and known != actual_blob:
        name, digest = next(iter(extracted.source_files.items()), ("", ""))
        return {"file": name, "expected_git_blob_sha1": known, "actual_git_blob_sha1": actual_blob, "actual_sha256": digest}
    return None


def write_records(records: Sequence[MagneticRecord], destination_dir: str | Path, *, reusable_ids: Iterable[str] = ()) -> list[str]:
    """Write ``<id>.json`` per record; never overwrites.

    A record whose id is in ``reusable_ids`` (deterministic identity records) may already exist with
    *identical* content; it is then kept and not reported as written. Any other existing target, or an
    identity record with different content, raises :class:`FileExistsError` before anything is written.
    """
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    reusable = set(reusable_ids)
    to_write: list[tuple[MagneticRecord, Path]] = []
    conflicts: list[str] = []
    for record in records:
        path = destination_dir / f"{record.id}.json"
        if not path.exists():
            to_write.append((record, path))
            continue
        if record.id in reusable:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = None
            if existing == record.to_json_dict():
                continue  # identical identity record already present: reuse
        conflicts.append(str(path))
    if conflicts:
        raise FileExistsError(f"adapters never overwrite records; existing: {conflicts}")
    return [str(dump(record.to_json_dict(), path, overwrite=False)) for record, path in to_write]


def cross_validate(records: Sequence[MagneticRecord], context: Iterable[MagneticRecord] = ()) -> None:
    errors = validate_catalogue([*context, *records])
    if errors:
        raise DataIntegrityError("produced records do not cross-validate: " + "; ".join(errors), details={"errors": errors})


def quantity_json(value: float, unit: str) -> dict[str, Any]:
    """SI ``{value, unit, original?}`` object through :meth:`Quantity.si`."""
    return Quantity.si(float(value), unit).model_dump(mode="json", exclude_none=True)


def sources_extension(descriptor: SourceDescriptor, item: SourceItem, extracted: ExtractedData | None = None,
                      **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_id": descriptor.source_id, "repository_url": descriptor.repository_url, "commit": descriptor.pinned_commit,
        "license": descriptor.license, "redistribution_permitted": descriptor.redistribution_permitted,
        "item_id": item.item_id, "adapter_version": __version__,
    }
    if item.url:
        payload["item_url"] = item.url
    if extracted is not None:
        payload["unit_mapping"] = extracted.unit_mapping()
        payload["column_sources"] = dict(extracted.column_sources)
        payload["source_files"] = dict(sorted(extracted.source_files.items()))
        if extracted.filters:
            payload["filters"] = dict(extracted.filters)
        if extracted.notes:
            payload["notes"] = list(extracted.notes)
    payload.update(extra)
    return payload


# --------------------------------------------------------------------------------------------------
# The adapter contract
# --------------------------------------------------------------------------------------------------
class SourceAdapter(ABC):
    """Abstract base of every external source adapter (see module docstring)."""

    descriptor: SourceDescriptor
    importer_id: str = "aipe.magnetics.sources"
    #: actions permitted by default (adapters with special policies override :meth:`check_license`)
    _license_policy: dict[str, bool] = {"read": True, "cache": True, "bundle": False, "redistribute": False}

    # -- abstract --------------------------------------------------------------------------------
    @abstractmethod
    def discover(self, root: str | Path | None = None, allow_download: bool = False, **options: Any) -> list[SourceItem]:
        """List what is available (local checkout / package data / cache / remote at the pinned commit)."""

    @abstractmethod
    def inspect(self, item: SourceItem, **options: Any) -> dict[str, Any]:
        """Metadata of one item: units, columns, shapes, sample counts, checksums (no bulk array loading)."""

    @abstractmethod
    def extract(self, item: SourceItem, **filters: Any) -> ExtractedData:
        """Normalised in-memory data of one item (numpy arrays + explicit units + metadata)."""

    @abstractmethod
    def to_records(self, extracted: ExtractedData, *, destination_dir: str | Path, dataset_id: str | None = None,
                   **options: Any) -> ImportResult:
        """Write JSON records and HDF5 artifacts (never overwriting) and return the :class:`ImportResult`."""

    # -- shared behaviour --------------------------------------------------------------------------
    def validate(self, extracted: ExtractedData) -> list[str]:
        """Integrity problems: finite values, monotone time, explicit compatible units, consistent counts, checksums."""
        problems: list[str] = []
        if extracted.source_id != self.descriptor.source_id:
            problems.append(f"extracted data belongs to {extracted.source_id!r}, not {self.descriptor.source_id!r}")
        if extracted.kind == "table":
            n_rows = None
            for name, (values, unit) in extracted.columns.items():
                array = np.asarray(values, dtype=np.float64)
                if array.ndim != 1:
                    problems.append(f"{name}: table columns must be 1-D, got shape {array.shape}")
                    continue
                n_rows = array.shape[0] if n_rows is None else n_rows
                if array.shape[0] != n_rows:
                    problems.append(f"{name}: {array.shape[0]} rows, expected {n_rows}")
                problems += finite_problems(name, array, allow_nan=True)
                if not units.known(unit):
                    problems.append(f"{name}: unknown unit {unit!r}")
            for name, (codes, categories) in extracted.categorical.items():
                codes_array = np.asarray(codes)
                if n_rows is not None and codes_array.shape != (n_rows,):
                    problems.append(f"{name}: categorical column has shape {codes_array.shape}, expected ({n_rows},)")
                if codes_array.size and (codes_array.min() < -1 or codes_array.max() >= len(categories)):
                    problems.append(f"{name}: categorical codes outside -1..{len(categories) - 1}")
        elif extracted.kind == "sequences":
            shape = None
            for name, (values, unit) in extracted.arrays.items():
                array = np.asarray(values, dtype=np.float64)
                if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 2:
                    problems.append(f"{name}: sequence arrays must be 2-D [n_sequences, n_samples>=2], got {array.shape}")
                    continue
                shape = array.shape if shape is None else shape
                if array.shape != shape:
                    problems.append(f"{name}: shape {array.shape} differs from {shape}")
                problems += finite_problems(name, array, allow_nan=(name != "time"))
                if name == "time":
                    problems += monotone_time_problems(array)
                if not units.known(unit):
                    problems.append(f"{name}: unknown unit {unit!r}")
            for name, (values, unit) in extracted.conditions.items():
                array = np.asarray(values, dtype=np.float64)
                if shape is not None and array.shape != (shape[0],):
                    problems.append(f"{name}: condition has shape {array.shape}, expected ({shape[0]},)")
                problems += finite_problems(name, array, allow_nan=True)
                if name == "frequency":
                    problems += range_problems(name, array, unit, minimum=0.0, strict_min=True)
                if not units.known(unit):
                    problems.append(f"{name}: unknown unit {unit!r}")
            if extracted.timebase == "uniform_per_period" and "frequency" not in extracted.conditions and extracted.sampling_rate is None:
                problems.append("uniform_per_period sequences need a per-sequence 'frequency' condition or a sampling rate")
            if extracted.timebase == "uniform_absolute" and extracted.sampling_rate is None:
                problems.append("uniform_absolute sequences need a sampling rate")
        elif extracted.kind == "parameters" and not extracted.parameters:
            problems.append("no parameters extracted")
        problems += checksum_problems(extracted.source_files, extracted.source_paths)
        return problems

    def attribution(self) -> list[SourceReference]:
        d = self.descriptor
        return [SourceReference(id=f"src.{d.record_id_token()}", kind="repository", reference=d.title, url=d.repository_url,
                                version=d.pinned_commit, license=d.license, note=d.citation)]

    def check_license(self, action: str) -> None:
        """Raise :class:`LicenseRestrictionError` when ``action`` is not permitted for this source."""
        if action not in LICENSE_ACTIONS:
            raise InvalidInputError(f"action must be one of {LICENSE_ACTIONS}, got {action!r}")
        permitted = self._license_policy.get(action, False)
        if action in ("bundle", "redistribute"):
            permitted = permitted and self.descriptor.redistribution_permitted is True
        if not permitted:
            raise LicenseRestrictionError(
                f"{self.descriptor.source_id}: action {action!r} is not permitted (licence {self.descriptor.license}, "
                f"redistribution_permitted={self.descriptor.redistribution_permitted})",
                details={"source_id": self.descriptor.source_id, "action": action, "license": self.descriptor.license,
                         "notes": list(self.descriptor.notes)})

    # -- helpers for subclasses --------------------------------------------------------------------
    def _repository_evidence(self, *, suffix: str = "repository", description: str | None = None, locator: str | None = None,
                             url: str | None = None, sha256: str | None = None, kind: str = "source") -> Evidence:
        d = self.descriptor
        return make_evidence(
            evidence_id=f"evidence.{d.record_id_token()}-{suffix}", kind=kind,
            reference=f"{d.title} ({d.repository_url}, commit {d.pinned_commit})",
            description=description or f"Upstream repository of the {d.title} at the pinned commit; licence {d.license}.",
            created_at=d.pinned_commit_date or "1970-01-01T00:00:00Z", license=d.license, locator=locator,
            url=url or d.repository_url, version=d.pinned_commit, sha256=sha256)

    def _unpinned_policy(self, extracted: ExtractedData, *, allow_unpinned: bool) -> PinStatus:
        """Decide how the relation between the extracted source file and the pinned upstream file is recorded.

        * verified identical (pinned digest / blob id matches, or the whole checkout was verified to be the
          pristine pinned commit): ``PinStatus(pinned=True, version=None)`` - the pinned commit is recorded;
        * verified different: a :class:`DataIntegrityError` unless ``allow_unpinned=True``; then
          ``pinned=False``, the honest version label ``'unpinned copy (sha256 <actual>)'`` replaces the
          commit in dataset properties, provenance and file evidence, and every record is flagged
          :data:`UNPINNED_FLAG`;
        * not verifiable (no pinned digest known): ``pinned=None`` and the label ``'unverified copy
          (sha256 <actual>)'`` - the commit is never asserted for bytes that were not checked.
        """
        mismatch = pinned_mismatch(extracted)
        if mismatch is not None:
            if not allow_unpinned:
                raise DataIntegrityError(
                    f"{mismatch['file']} is not byte-identical to the file at the pinned commit {self.descriptor.pinned_commit[:12]} of "
                    f"{self.descriptor.title} ({', '.join(f'{k} {v[:12]}...' for k, v in mismatch.items() if k != 'file')}); refusing to "
                    f"stamp records with that commit. Pass allow_unpinned=True to import the copy anyway: the records are then flagged "
                    f"{UNPINNED_FLAG!r} and versioned by the file's SHA-256 instead of the commit", details=dict(mismatch))
            return PinStatus(mismatch, f"unpinned copy (sha256 {mismatch['actual_sha256']})", [UNPINNED_FLAG], False)
        meta = extracted.metadata
        if meta.get("expected_sha256") or meta.get("checkout_verified") or (meta.get("known_blob_sha1") and meta.get("git_blob_sha1_lf")):
            return PinStatus(None, None, [], True)
        digest = next(iter(extracted.source_files.values()), None)
        return PinStatus(None, f"unverified copy (sha256 {digest})" if digest else "unverified copy", [], None)

    def _provenance(self, activity: str, *, inputs: Sequence[str], normalization: str, retrieved_at: str | None = None,
                    version: str | None = None) -> Provenance:
        """Import provenance; ``version`` replaces the pinned commit when the data are not the pinned upstream bytes."""
        pinned = version is None
        return Provenance(created_at=retrieved_at, activity=activity, tool=tool_attribution(), inputs=list(inputs),
                          source_commit=self.descriptor.pinned_commit if pinned else None,
                          dataset_version=self.descriptor.pinned_commit if pinned else version, normalization=normalization)

    def _dataset_record(self, *, dataset_id: str, name: str, evidence: Sequence[Evidence], checksums: Mapping[str, str],
                        materials: Sequence[str], record_count: int | None, description: str, provenance: Provenance,
                        artifacts: Sequence[Artifact] = (), extensions: Mapping[str, Any] | None = None,
                        retrieved_at: str | None = None, title: str | None = None, url: str | None = None,
                        synthetic: bool = False, tags: Iterable[str] = (), quality: Mapping[str, Any] | None = None,
                        version: str | None = None) -> MagneticRecord:
        """``dataset`` record; ``version=None`` means the data are the pinned upstream bytes (commit recorded as
        ``version`` and ``source_commit``), otherwise ``version`` is the honest label and ``source_commit`` is omitted."""
        d = self.descriptor
        properties: dict[str, Any] = {
            "source_id": d.source_id, "title": title or d.title, "url": url or d.repository_url, "version": version or d.pinned_commit,
            "license": d.license, "redistribution_permitted": d.redistribution_permitted,
            "materials": list(materials), "checksums": dict(sorted(checksums.items())), "description": description,
        }
        if version is None:
            properties["source_commit"] = d.pinned_commit
        if d.publisher:
            properties["publisher"] = d.publisher
        if d.citation:
            properties["citation"] = d.citation
        if record_count is not None:
            properties["record_count"] = int(record_count)
        if retrieved_at:
            properties["retrieved_at"] = retrieved_at
        return record_envelope(record_id=dataset_id, kind="dataset", name=name, properties=properties, evidence=evidence,
                               license=d.license, provenance=provenance, description=description, synthetic=synthetic,
                               artifacts=artifacts, tags=["external-source", d.source_id, *tags], extensions=extensions, quality=quality)

    def material_identity_id(self, material: str) -> str:
        return f"material.{self.descriptor.record_id_token()}-{slug(material)}"

    def _material_identity_record(self, *, material: str, evidence: Sequence[Evidence], provenance: Provenance | None = None,
                                  manufacturer: str | None = None, material_class: str | None = None,
                                  description: str | None = None, extensions: Mapping[str, Any] | None = None,
                                  synthetic: bool = False) -> MagneticRecord:
        """Minimal identity record for ``material_ref`` targets (grade only; no datasheet value is invented).

        The record is deterministic for a given source and material (its provenance does not mention the
        importing file), so several imports into one directory share it (see :func:`write_records`).
        """
        d = self.descriptor
        properties: dict[str, Any] = {"grade": material}
        if material_class:
            properties["material_class"] = material_class
        text = description or (f"Identity record for material {material!r} as labelled by {d.title}"
                               + (f" (manufacturer stated upstream: {manufacturer})" if manufacturer else "")
                               + ". It carries no datasheet parameters; it exists so that measurement records can reference the material.")
        properties["description"] = text
        identity_provenance = Provenance(activity=f"identity record for the material label {material!r} of {d.title}", tool=tool_attribution(),
                                         inputs=[e.id for e in evidence], source_commit=d.pinned_commit, dataset_version=d.pinned_commit,
                                         normalization="none; identity only")
        identity_extensions: dict[str, Any] = {SOURCES_EXTENSION: {
            "source_id": d.source_id, "repository_url": d.repository_url, "commit": d.pinned_commit, "license": d.license,
            "redistribution_permitted": d.redistribution_permitted, "material_label": material, "identity_only": True,
            "adapter_version": __version__}}
        for key, value in (extensions or {}).items():
            identity_extensions[key] = {**identity_extensions.get(key, {}), **value} if isinstance(value, Mapping) else value
        return record_envelope(record_id=self.material_identity_id(material), kind="magnetic_material",
                               name=f"{material} ({d.title} label)", properties=properties, evidence=evidence, license=d.license,
                               provenance=identity_provenance, description=text, synthetic=synthetic,
                               tags=["external-source", d.source_id, "identity-only"], extensions=identity_extensions)

    def _import_result(self, *, records: Sequence[MagneticRecord], artifacts: Sequence[Artifact], provenance: Provenance,
                       warnings: Sequence[str], written: Sequence[str], source_sha256: str | None) -> ImportResult:
        return ImportResult(importer=self.importer_id, records=list(records), artifacts=list(artifacts), provenance=provenance,
                            warnings=list(warnings), written=list(written), source_sha256=source_sha256)

    def _import_extension(self, *, source_name: str, source_sha256: str, normalization: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return import_extension(source_name=source_name, source_sha256=source_sha256, importer=self.importer_id,
                                normalization=normalization, extra=extra)


__all__ = [
    "TABLE_LAYOUT_VERSION", "SOURCES_EXTENSION", "UNPINNED_FLAG", "LICENSE_ACTIONS", "KNOWN_LICENSES", "LicenseAction",
    "SourceDescriptor", "SourceItem", "ExtractedData", "SourceAdapter", "PinStatus",
    "TableMeta", "TableColumnLayout", "TableDescription", "TableData", "TableStore", "codes_sha256",
    "finite_problems", "range_problems", "monotone_time_problems", "checksum_problems", "unit_problems",
    "slug", "file_safe_token", "dataset_token", "source_uri", "pinned_mismatch", "make_evidence", "record_envelope", "write_records",
    "cross_validate", "quantity_json", "sources_extension",
]
