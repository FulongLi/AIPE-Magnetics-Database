"""HDF5 waveform store - layout ``aipe.magnetics.waveforms/0.2``.

Large numerical arrays (sampled excitation waveforms, B-H loops) never go into JSON. A
``waveform_dataset`` record describes an HDF5 file with the layout below; this module writes,
reads, describes and verifies such files.

Layout
------
File attributes (strings unless noted)::

    layout_version        "aipe.magnetics.waveforms/0.2"
    schema_version        "0.2.0" (record standard the file belongs to)
    created_by_tool       "aipe.magnetics"
    created_by_version    package version that wrote the file
    dataset_version       caller-supplied dataset version (upstream release tag, commit, date)
    sign_convention       e.g. "B, H, v, i positive along the winding reference direction (right-hand rule)"
    coordinate_definition e.g. "scalar quantities along the magnetic path; no vector components"
    timebase              "uniform_per_period" | "explicit_time_array" | "uniform_absolute"
    sampling_rate_hz      float64, only when one uniform sampling rate applies to every sequence
    sampling_rate_original_value, sampling_rate_original_unit
                          the sampling rate exactly as supplied (float64, unit symbol), only when it
                          was converted to Hz, so the normalisation is reproducible from the file
    license               SPDX expression or "NOASSERTION"
    source_id             source / adapter identifier, e.g. "aipe.synthetic", "princeton_magnet"
    description           optional free text
    checksum_method       "sha256 of float64 little-endian C-order bytes (NaN canonicalised)"

    /sequences/<quantity>   2-D float64 [n_sequences, n_samples]; gzip + shuffle, chunked by rows;
                            attrs: quantity, unit (canonical SI), original_unit (as supplied),
                            missing_value "NaN", axis0 "sequence", axis1 "sample"
                            quantities: flux_density [T], field [A/m], voltage [V], current [A], time [s]
    /conditions/<name>      1-D float64 [n_sequences] per-sequence scalar conditions; same attrs
                            names: frequency [Hz], temperature [K], dc_bias_field [A/m],
                            loss_density [W/m^3], flux_density_peak [T], field_peak [A/m], duty [1]
    /checksums              group whose attributes map "sequences/<quantity>" and "conditions/<name>"
                            to the SHA-256 hex digest of that array's canonical bytes

Conventions
-----------
* **Units.** Arrays are converted to the canonical SI unit on write with exactly the affine rule of
  the scalar registry (``value_SI = factor * value + offset`` in float64, see
  :func:`convert_array`); the supplied unit is kept in ``original_unit`` so the normalisation is
  reproducible. Unknown or dimensionally incompatible units raise :class:`UnitError`. Nothing is
  inferred from names. A value that overflows float64 *during* the conversion (e.g. ``1e306 kA/m``)
  is rejected with :class:`InvalidInputError`; it is never stored as ``inf``.
* **Missing values** are ``NaN`` - never 0. :func:`missing_mask` returns the boolean mask.
  ``+/-inf`` is rejected. Time arrays may not contain missing samples and must be strictly
  increasing within each sequence.
* **Numeric input.** Integer arrays are accepted only when every element is exactly representable
  in float64 (``|v| <= 2**53``); larger integers are rejected instead of being rounded silently.
  Boolean, complex, object and string arrays, and ragged nested lists, are rejected.
* **Shapes.** Every sequence array must have the identical shape ``[n_sequences, n_samples]`` with
  ``n_sequences >= 1`` and ``n_samples >= 2``; every condition must have ``n_sequences`` entries
  (shape ``(n,)`` or the equivalent column ``(n, 1)``). Anything else raises
  :class:`InvalidInputError`; there is no broadcasting, padding or truncation.
* **Timebase.** ``explicit_time_array`` requires (and is required by) a ``time`` array;
  ``uniform_absolute`` requires ``sampling_rate``; ``uniform_per_period`` (exactly one steady-state
  period per row) requires ``sampling_rate`` or the ``frequency`` condition so the sample spacing
  ``1 / (f * n_samples)`` is recoverable. When the ``frequency`` condition is the only timebase
  information it must be finite in every row; when both are given, every finite row frequency must
  satisfy ``f * n_samples == sampling_rate`` (relative tolerance 1e-9), otherwise the two statements
  contradict each other and the write is refused.
* **Checksums.** ``sha256(np.ascontiguousarray(a, '<f8').tobytes())`` after NaN canonicalisation
  (every NaN written with the canonical quiet-NaN bit pattern); verified on :meth:`WaveformStore.read`
  and by :meth:`WaveformStore.verify_checksums`. The per-array checksums cover the array **content**
  only: dataset and file attributes (units, timebase, sign convention ...) are not part of them. The
  file-level SHA-256 that :meth:`WaveformStore.artifact` records in a record's ``artifacts`` covers
  the whole file, attributes included.
* **Determinism.** Datasets are created with ``track_times=False``; identical inputs give identical
  files, so the file-level SHA-256 recorded in a record's ``artifacts`` is reproducible.
* **Nothing is written on invalid input.** :meth:`WaveformStore.prepare` performs every check and
  conversion without touching the file system; :meth:`WaveformStore.write` is ``prepare`` followed by
  :meth:`WaveformStore.write_prepared`.

h5py is optional: importing this module never requires it. Using :class:`WaveformStore` without
h5py raises :class:`UnsupportedOperationError` (``install aipe-magnetics[hdf5]``).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aipe_magnetics import TOOL_ID, __version__
from aipe_magnetics.common.errors import DataIntegrityError, InvalidInputError, UnitError, UnsupportedOperationError
from aipe_magnetics.common.jsonio import sha256_file
from aipe_magnetics.common.quantity import Original, Quantity
from aipe_magnetics.common.units import units
from aipe_magnetics.data.records import SCHEMA_VERSION, Artifact, ArrayLayout, WaveformDatasetProperties

try:  # optional dependency
    import h5py  # type: ignore
except ImportError:  # pragma: no cover - exercised only when the extra is missing
    h5py = None

LAYOUT_VERSION = "aipe.magnetics.waveforms/0.2"
HDF5_MEDIA_TYPE = "application/x-hdf5"
CHECKSUM_METHOD = "sha256 of float64 little-endian C-order bytes (NaN canonicalised)"
MISSING_VALUE = "NaN"
#: largest integer magnitude that float64 represents exactly
EXACT_INTEGER_LIMIT = 2 ** 53
#: relative tolerance of the ``f * n_samples == sampling_rate`` consistency rule (uniform_per_period)
SAMPLING_RATE_RTOL = 1e-9

#: sequence quantity -> canonical SI unit
SEQUENCE_QUANTITIES: dict[str, str] = {"flux_density": "T", "field": "A/m", "voltage": "V", "current": "A", "time": "s"}
#: per-sequence scalar condition -> canonical SI unit
CONDITION_QUANTITIES: dict[str, str] = {
    "frequency": "Hz", "temperature": "K", "dc_bias_field": "A/m", "loss_density": "W/m^3",
    "flux_density_peak": "T", "field_peak": "A/m", "duty": "1",
}
#: physical constraints applied to the non-missing entries of a condition
_CONDITION_CONSTRAINTS: dict[str, str] = {
    "frequency": "positive", "temperature": "positive", "loss_density": "nonnegative",
    "flux_density_peak": "nonnegative", "field_peak": "nonnegative", "duty": "open_unit_interval",
}
Timebase = Literal["uniform_per_period", "explicit_time_array", "uniform_absolute"]
_COMPRESSIONS: tuple[str | None, ...] = ("gzip", "lzf", None)

DEFAULT_SIGN_CONVENTION = "B, H, v, i positive along the winding reference direction (right-hand rule)"
DEFAULT_COORDINATE_DEFINITION = "scalar quantities along the magnetic path; no vector components"


def _require_h5py() -> Any:
    if h5py is None:
        raise UnsupportedOperationError("HDF5 waveform support needs h5py: install aipe-magnetics[hdf5]",
                                        details={"extra": "hdf5"})
    return h5py


# --------------------------------------------------------------------------------------------------
# Array helpers (no h5py needed)
# --------------------------------------------------------------------------------------------------
def convert_array(values: Any, from_unit: str, to_unit: str) -> np.ndarray:
    """Vectorised unit conversion with the same float64 affine arithmetic as ``units.convert``.

    Element-wise results are bit-identical to the scalar registry conversion. NaN propagates
    (missing stays missing). Raises :class:`UnitError` for unknown or incompatible units. Values that
    overflow float64 become ``inf`` here (no warning is emitted); the store rejects them explicitly.
    """
    src, dst = units.get(from_unit), units.get(to_unit)
    if src.dimension != dst.dimension:
        raise UnitError(f"cannot convert {from_unit!r} to {to_unit!r}: incompatible dimensions",
                        details={"from": from_unit, "to": to_unit})
    array = np.array(values, dtype=np.float64, copy=True)
    if from_unit == to_unit:
        return array
    if {from_unit, to_unit} == {"rad", "deg"}:
        return array * (math.pi / 180.0 if from_unit == "deg" else 180.0 / math.pi)
    if (from_unit in ("rad", "deg")) != (to_unit in ("rad", "deg")):
        raise UnitError(f"angles ({from_unit!r}) are not interchangeable with plain ratios ({to_unit!r})")
    with np.errstate(over="ignore"):
        si = array * float(src.factor) + float(src.offset)
        return (si - float(dst.offset)) / float(dst.factor)


def canonicalize_nan(array: np.ndarray) -> np.ndarray:
    """Copy of ``array`` (float64) with every NaN rewritten as the canonical quiet NaN."""
    out = np.array(array, dtype=np.float64, copy=True)
    out[np.isnan(out)] = np.nan
    return out


def array_sha256(array: Any) -> str:
    """SHA-256 of the canonical float64 little-endian C-order bytes of ``array`` (NaN canonicalised)."""
    canonical = np.ascontiguousarray(canonicalize_nan(np.asarray(array)), dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def missing_mask(array: Any) -> np.ndarray:
    """Boolean mask of missing (NaN) entries."""
    return np.isnan(np.asarray(array, dtype=np.float64))


# --------------------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------------------
class WaveformMeta(BaseModel):
    """File-level metadata written as HDF5 attributes (everything the layout requires, nothing inferred)."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1, description="source / adapter identifier")
    dataset_version: str = Field(min_length=1, description="dataset version, release tag, commit or date")
    license: str = Field(default="NOASSERTION", min_length=1)
    timebase: Timebase = "uniform_per_period"
    sampling_rate: Quantity | None = Field(default=None, description="uniform sampling rate (any Hz-compatible unit)")
    sign_convention: str = Field(default=DEFAULT_SIGN_CONVENTION, min_length=1)
    coordinate_definition: str = Field(default=DEFAULT_COORDINATE_DEFINITION, min_length=1)
    description: str | None = None
    material_ref: str | None = None
    source_dataset_ref: str | None = None

    @model_validator(mode="after")
    def _sampling_rate(self) -> "WaveformMeta":
        if self.sampling_rate is not None:
            if not units.compatible(self.sampling_rate.unit, "Hz"):
                raise ValueError(f"sampling_rate must have the dimension of Hz, got {self.sampling_rate.unit!r}")
            if self.sampling_rate.magnitude("Hz") <= 0:
                raise ValueError("sampling_rate must be positive")
            if self.sampling_rate.uncertainty is not None:
                raise ValueError("sampling_rate uncertainty cannot be stored in the waveform layout (only value, unit "
                                 "and original are file attributes); state it in the record instead")
        return self

    def sampling_rate_hz(self) -> Quantity | None:
        """The sampling rate normalised to Hz with the supplied value / unit retained as ``original``."""
        return None if self.sampling_rate is None else self.sampling_rate.to("Hz")


@dataclass
class WaveformData:
    """In-memory content of a waveform file (SI arrays, units, original units, attributes, checksums)."""

    arrays: dict[str, np.ndarray]
    array_units: dict[str, str]
    array_original_units: dict[str, str]
    conditions: dict[str, np.ndarray]
    condition_units: dict[str, str]
    condition_original_units: dict[str, str]
    attrs: dict[str, Any]
    checksums: dict[str, str] = field(default_factory=dict)

    @property
    def n_sequences(self) -> int:
        first = next(iter(self.arrays.values()))
        return int(first.shape[0])

    @property
    def n_samples(self) -> int:
        first = next(iter(self.arrays.values()))
        return int(first.shape[1])

    def mask(self, name: str) -> np.ndarray:
        """Missing-value mask of a sequence array or condition."""
        if name in self.arrays:
            return missing_mask(self.arrays[name])
        if name in self.conditions:
            return missing_mask(self.conditions[name])
        raise InvalidInputError(f"unknown array or condition {name!r}", details={"name": name})


@dataclass
class PreparedWaveforms:
    """Validated, SI-normalised content of a waveform file before it is written (:meth:`WaveformStore.prepare`)."""

    meta: WaveformMeta
    sequences: dict[str, np.ndarray]
    sequence_original_units: dict[str, str]
    conditions: dict[str, np.ndarray]
    condition_original_units: dict[str, str]
    n_sequences: int
    n_samples: int
    warnings: list[str] = field(default_factory=list)

    @property
    def properties(self) -> WaveformDatasetProperties:
        """The ``waveform_dataset`` properties that describe the file once written."""
        return WaveformStore._properties(self.meta, self.sequences, self.sequence_original_units, self.conditions,
                                         self.n_sequences, self.n_samples)


# --------------------------------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------------------------------
def _as_pair(name: str, item: Any) -> tuple[Any, str]:
    if not isinstance(item, (tuple, list)) or len(item) != 2 or not isinstance(item[1], str):
        raise InvalidInputError(f"{name}: supply arrays as (values, unit) pairs with an explicit unit string",
                                details={"name": name})
    return item[0], item[1]


def _as_float_array(name: str, values: Any) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (ValueError, TypeError) as error:
        raise InvalidInputError(f"{name}: values cannot be interpreted as a rectangular numeric array ({error})",
                                details={"name": name}) from None
    if array.dtype.kind not in "iuf":
        raise InvalidInputError(f"{name}: numeric array required, got dtype {array.dtype} (bool/complex/object/str are rejected)",
                                details={"name": name, "dtype": str(array.dtype)})
    if array.dtype.kind in "iu" and array.size:
        largest = max(abs(int(array.max())), abs(int(array.min())))
        if largest > EXACT_INTEGER_LIMIT:
            raise InvalidInputError(f"{name}: integer values up to |{largest}| exceed 2**53 and are not exactly representable in "
                                    "float64; convert them explicitly instead of relying on a silent rounding",
                                    details={"name": name, "largest": largest})
    array = np.asarray(array, dtype=np.float64)
    if np.isinf(array).any():
        raise InvalidInputError(f"{name}: infinite values are not allowed (missing values must be NaN)",
                                details={"name": name, "count": int(np.isinf(array).sum())})
    return array


def _to_si(name: str, array: np.ndarray, unit: str, si_unit: str) -> np.ndarray:
    if not units.compatible(unit, si_unit):
        raise UnitError(f"{name}: unit {unit!r} is not compatible with {si_unit!r}", details={"name": name, "unit": unit})
    si_array = convert_array(array, unit, si_unit)
    overflow = int(np.isinf(si_array).sum())
    if overflow:
        raise InvalidInputError(f"{name}: {overflow} value(s) overflow float64 when converted from {unit!r} to {si_unit!r}",
                                details={"name": name, "unit": unit, "count": overflow})
    return si_array


def _validate_sequences(arrays: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, str], tuple[int, int]]:
    if not isinstance(arrays, Mapping) or not arrays:
        raise InvalidInputError("arrays must be a non-empty mapping quantity -> (values, unit)")
    unknown = sorted(set(arrays) - set(SEQUENCE_QUANTITIES))
    if unknown:
        raise InvalidInputError(f"unknown sequence quantities {unknown}; allowed: {sorted(SEQUENCE_QUANTITIES)}",
                                details={"unknown": unknown})
    if not (set(arrays) - {"time"}):
        raise InvalidInputError("at least one physical sequence quantity (flux_density, field, voltage, current) is required")
    converted: dict[str, np.ndarray] = {}
    original_units: dict[str, str] = {}
    shape: tuple[int, int] | None = None
    for name in SEQUENCE_QUANTITIES:  # fixed order
        if name not in arrays:
            continue
        values, unit = _as_pair(name, arrays[name])
        array = _as_float_array(name, values)
        if array.ndim != 2:
            raise InvalidInputError(f"{name}: sequence arrays must be 2-D [n_sequences, n_samples], got shape {array.shape}",
                                    details={"name": name, "shape": list(array.shape)})
        if shape is None:
            shape = (int(array.shape[0]), int(array.shape[1]))
            if shape[0] < 1 or shape[1] < 2:
                raise InvalidInputError(f"{name}: need n_sequences >= 1 and n_samples >= 2, got shape {array.shape}",
                                        details={"name": name, "shape": list(array.shape)})
        elif tuple(array.shape) != shape:
            raise InvalidInputError(f"{name}: shape {array.shape} differs from the other sequence arrays {shape}",
                                    details={"name": name, "shape": list(array.shape), "expected": list(shape)})
        si_array = _to_si(name, array, unit, SEQUENCE_QUANTITIES[name])
        if name == "time":
            if np.isnan(si_array).any():
                raise InvalidInputError("time: missing samples are not allowed in a time array")
            if not np.all(np.diff(si_array, axis=1) > 0):
                raise InvalidInputError("time: every sequence must be strictly increasing")
        converted[name] = canonicalize_nan(si_array)
        original_units[name] = unit
    assert shape is not None
    return converted, original_units, shape


def _validate_conditions(conditions: Mapping[str, Any] | None, n_sequences: int) -> tuple[dict[str, np.ndarray], dict[str, str], list[str]]:
    converted: dict[str, np.ndarray] = {}
    original_units: dict[str, str] = {}
    warnings: list[str] = []
    if conditions is None:
        return converted, original_units, warnings
    if not isinstance(conditions, Mapping):
        raise InvalidInputError("conditions must be a mapping name -> (values, unit)")
    unknown = sorted(set(conditions) - set(CONDITION_QUANTITIES))
    if unknown:
        raise InvalidInputError(f"unknown condition names {unknown}; allowed: {sorted(CONDITION_QUANTITIES)}",
                                details={"unknown": unknown})
    for name in CONDITION_QUANTITIES:
        if name not in conditions:
            continue
        values, unit = _as_pair(name, conditions[name])
        array = _as_float_array(name, values)
        if array.ndim == 2 and array.shape[1] == 1:
            warnings.append(f"{name}: column array of shape {array.shape} flattened to ({array.shape[0]},)")
            array = array[:, 0]
        if array.ndim != 1 or array.shape[0] != n_sequences:
            raise InvalidInputError(f"{name}: conditions must be 1-D with one entry per sequence ({n_sequences}), got shape {array.shape}",
                                    details={"name": name, "shape": list(array.shape), "expected": [n_sequences]})
        si_unit = CONDITION_QUANTITIES[name]
        si_array = _to_si(name, array, unit, si_unit)
        present = si_array[~np.isnan(si_array)]
        rule = _CONDITION_CONSTRAINTS.get(name)
        if rule == "positive" and (present <= 0).any():
            raise InvalidInputError(f"{name}: values must be positive in {si_unit}", details={"name": name})
        if rule == "nonnegative" and (present < 0).any():
            raise InvalidInputError(f"{name}: values must be non-negative in {si_unit}", details={"name": name})
        if rule == "open_unit_interval" and ((present <= 0) | (present >= 1)).any():
            raise InvalidInputError(f"{name}: values must lie strictly between 0 and 1", details={"name": name})
        converted[name] = canonicalize_nan(si_array)
        original_units[name] = unit
    return converted, original_units, warnings


def _validate_timebase(meta: WaveformMeta, arrays: Mapping[str, np.ndarray], conditions: Mapping[str, np.ndarray],
                       n_samples: int) -> None:
    has_time = "time" in arrays
    if meta.timebase == "explicit_time_array" and not has_time:
        raise InvalidInputError("timebase 'explicit_time_array' requires a 'time' sequence array")
    if meta.timebase != "explicit_time_array" and has_time:
        raise InvalidInputError(f"a 'time' array was supplied but timebase is {meta.timebase!r}; use 'explicit_time_array'")
    if meta.timebase == "uniform_absolute" and meta.sampling_rate is None:
        raise InvalidInputError("timebase 'uniform_absolute' requires meta.sampling_rate")
    if meta.timebase != "uniform_per_period":
        return
    frequency = conditions.get("frequency")
    if meta.sampling_rate is None:
        if frequency is None:
            raise InvalidInputError("timebase 'uniform_per_period' requires meta.sampling_rate or the 'frequency' condition "
                                    "so that the sample spacing (1/(f*n_samples)) is recoverable")
        missing = int(np.isnan(frequency).sum())
        if missing:
            raise InvalidInputError(f"timebase 'uniform_per_period' without meta.sampling_rate: the 'frequency' condition is the "
                                    f"only timebase information and must be finite in every row ({missing} missing value(s))",
                                    details={"missing": missing})
        return
    if frequency is None:
        return
    rate = meta.sampling_rate.magnitude("Hz")
    finite = frequency[~np.isnan(frequency)]
    mismatch = np.abs(finite * n_samples - rate) > SAMPLING_RATE_RTOL * rate
    if mismatch.any():
        offending = float(finite[mismatch][0])
        raise InvalidInputError(
            f"timebase 'uniform_per_period' (one period of {n_samples} samples per row) with meta.sampling_rate = {rate!r} Hz "
            f"requires frequency * n_samples == sampling_rate for every row, but a row states {offending!r} Hz "
            f"({offending * n_samples!r} Hz); either drop the sampling rate or use timebase 'uniform_absolute'",
            details={"sampling_rate_hz": rate, "frequency_hz": offending, "n_samples": n_samples})


def _py(value: Any) -> Any:
    """Convert an HDF5 attribute value to a plain Python object."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# --------------------------------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------------------------------
class WaveformStore:
    """Writer / reader / verifier for the ``aipe.magnetics.waveforms/0.2`` HDF5 layout."""

    layout_version = LAYOUT_VERSION

    def __init__(self, *, compression: str | None = "gzip", compression_level: int = 4, max_chunk_bytes: int = 1 << 19):
        if compression not in _COMPRESSIONS:
            raise InvalidInputError(f"compression must be one of {_COMPRESSIONS}, got {compression!r}",
                                    details={"compression": compression})
        if isinstance(compression_level, bool) or not isinstance(compression_level, int) or not 0 <= compression_level <= 9:
            raise InvalidInputError("compression_level must be an integer in 0..9")
        if not isinstance(max_chunk_bytes, int) or max_chunk_bytes < 8:
            raise InvalidInputError("max_chunk_bytes must be an integer >= 8")
        self.compression = compression
        self.compression_level = int(compression_level)
        self.max_chunk_bytes = int(max_chunk_bytes)

    def _dataset_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"shuffle": True, "track_times": False}
        if self.compression is not None:
            options["compression"] = self.compression
            if self.compression == "gzip":  # lzf takes no options
                options["compression_opts"] = self.compression_level
        return options

    # -- prepare / write --------------------------------------------------------------------------
    def prepare(self, arrays: Mapping[str, Any], conditions: Mapping[str, Any] | None = None, *,
                meta: WaveformMeta) -> PreparedWaveforms:
        """Validate and normalise everything **without writing**: the complete input check of :meth:`write`.

        Raises :class:`InvalidInputError` / :class:`UnitError`; needs no h5py.
        """
        if not isinstance(meta, WaveformMeta):
            raise InvalidInputError("meta must be a WaveformMeta instance")
        sequences, seq_original, (n_sequences, n_samples) = _validate_sequences(arrays)
        condition_arrays, cond_original, warnings = _validate_conditions(conditions, n_sequences)
        _validate_timebase(meta, sequences, condition_arrays, n_samples)
        return PreparedWaveforms(meta=meta, sequences=sequences, sequence_original_units=seq_original,
                                 conditions=condition_arrays, condition_original_units=cond_original,
                                 n_sequences=n_sequences, n_samples=n_samples, warnings=warnings)

    def write(self, path: str | Path, arrays: Mapping[str, Any], conditions: Mapping[str, Any] | None = None, *,
              meta: WaveformMeta, overwrite: bool = False) -> WaveformDatasetProperties:
        """Validate, normalise to SI and write a waveform file; returns the describing properties.

        ``arrays``/``conditions`` map a quantity name to ``(values, unit)``. Raises
        :class:`InvalidInputError` / :class:`UnitError` before anything is written and
        :class:`FileExistsError` when ``path`` exists and ``overwrite`` is false.
        """
        _require_h5py()
        prepared = self.prepare(arrays, conditions, meta=meta)
        return self.write_prepared(path, prepared, overwrite=overwrite)

    def write_prepared(self, path: str | Path, prepared: PreparedWaveforms, *, overwrite: bool = False) -> WaveformDatasetProperties:
        """Write the output of :meth:`prepare` (no further validation; ``FileExistsError`` unless ``overwrite``)."""
        lib = _require_h5py()
        if not isinstance(prepared, PreparedWaveforms):
            raise InvalidInputError("prepared must come from WaveformStore.prepare()")
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
        meta = prepared.meta
        n_sequences, n_samples = prepared.n_sequences, prepared.n_samples
        path.parent.mkdir(parents=True, exist_ok=True)
        checksums: dict[str, str] = {}
        options = self._dataset_options()
        with lib.File(path, "w") as handle:
            handle.attrs["layout_version"] = LAYOUT_VERSION
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["created_by_tool"] = TOOL_ID
            handle.attrs["created_by_version"] = __version__
            handle.attrs["dataset_version"] = meta.dataset_version
            handle.attrs["sign_convention"] = meta.sign_convention
            handle.attrs["coordinate_definition"] = meta.coordinate_definition
            handle.attrs["timebase"] = meta.timebase
            handle.attrs["license"] = meta.license
            handle.attrs["source_id"] = meta.source_id
            handle.attrs["checksum_method"] = CHECKSUM_METHOD
            handle.attrs["missing_value"] = MISSING_VALUE
            rate = meta.sampling_rate_hz()
            if rate is not None:
                handle.attrs["sampling_rate_hz"] = np.float64(rate.value)
                if rate.original is not None:
                    handle.attrs["sampling_rate_original_value"] = np.float64(rate.original.value)
                    handle.attrs["sampling_rate_original_unit"] = rate.original.unit
            if meta.description:
                handle.attrs["description"] = meta.description
            if meta.material_ref:
                handle.attrs["material_ref"] = meta.material_ref
            if meta.source_dataset_ref:
                handle.attrs["source_dataset_ref"] = meta.source_dataset_ref
            seq_group = handle.create_group("sequences")
            rows = max(1, min(n_sequences, self.max_chunk_bytes // (8 * n_samples)))
            for name, array in prepared.sequences.items():
                dataset = seq_group.create_dataset(name, data=array, dtype="<f8", chunks=(rows, n_samples), **options)
                dataset.attrs["quantity"] = name
                dataset.attrs["unit"] = SEQUENCE_QUANTITIES[name]
                dataset.attrs["original_unit"] = prepared.sequence_original_units[name]
                dataset.attrs["missing_value"] = MISSING_VALUE
                dataset.attrs["axis0"] = "sequence"
                dataset.attrs["axis1"] = "sample"
                checksums[f"sequences/{name}"] = array_sha256(array)
            cond_group = handle.create_group("conditions")
            for name, array in prepared.conditions.items():
                dataset = cond_group.create_dataset(name, data=array, dtype="<f8", chunks=(min(n_sequences, 1 << 16),), **options)
                dataset.attrs["quantity"] = name
                dataset.attrs["unit"] = CONDITION_QUANTITIES[name]
                dataset.attrs["original_unit"] = prepared.condition_original_units[name]
                dataset.attrs["missing_value"] = MISSING_VALUE
                dataset.attrs["axis0"] = "sequence"
                checksums[f"conditions/{name}"] = array_sha256(array)
            sums = handle.create_group("checksums")
            sums.attrs["method"] = CHECKSUM_METHOD
            for key, digest in checksums.items():
                sums.attrs[key] = digest
        return prepared.properties

    @staticmethod
    def _properties(meta: WaveformMeta, sequences: Mapping[str, np.ndarray], seq_original: Mapping[str, str],
                    conditions: Mapping[str, np.ndarray], n_sequences: int, n_samples: int) -> WaveformDatasetProperties:
        layouts = [ArrayLayout(path=f"/sequences/{name}", quantity=name, unit=SEQUENCE_QUANTITIES[name],
                               original_unit=seq_original[name], dtype="float64", shape=[n_sequences, n_samples],
                               axis_semantics=["sequence", "sample"], missing_value=MISSING_VALUE)
                   for name in sequences]
        return WaveformDatasetProperties(
            material_ref=meta.material_ref, source_dataset_ref=meta.source_dataset_ref, format="hdf5",
            layout_version=LAYOUT_VERSION, sampling_rate=meta.sampling_rate_hz(), samples_per_sequence=n_samples,
            timebase=meta.timebase, sign_convention=meta.sign_convention, coordinate_definition=meta.coordinate_definition,
            arrays=layouts, conditions={name: f"/conditions/{name}" for name in conditions}, n_sequences=n_sequences,
            description=meta.description)

    # -- read -------------------------------------------------------------------------------------
    @staticmethod
    def _open_layout(lib: Any, handle: Any, path: Path) -> dict[str, Any]:
        """File attributes of an ``aipe.magnetics.waveforms/0.2`` file; :class:`DataIntegrityError` for anything else."""
        attrs = {key: _py(value) for key, value in handle.attrs.items()}
        if attrs.get("layout_version") != LAYOUT_VERSION:
            raise DataIntegrityError(f"{path}: not an {LAYOUT_VERSION!r} file (layout_version={attrs.get('layout_version')!r})",
                                     details={"path": str(path), "layout_version": attrs.get("layout_version")})
        if "sequences" not in handle or not isinstance(handle["sequences"], lib.Group):
            raise DataIntegrityError(f"{path}: missing /sequences group", details={"path": str(path)})
        if "conditions" in handle and not isinstance(handle["conditions"], lib.Group):
            raise DataIntegrityError(f"{path}: /conditions is not a group", details={"path": str(path)})
        return attrs

    @staticmethod
    def _dataset(lib: Any, handle: Any, group: str, name: str, ndim: int, path: Path) -> Any:
        dataset = handle[group][name]
        if not isinstance(dataset, lib.Dataset) or dataset.ndim != ndim:
            raise DataIntegrityError(f"{path}: /{group}/{name} must be a {ndim}-D dataset", details={"path": str(path), "name": name})
        if "unit" not in dataset.attrs:
            raise DataIntegrityError(f"{path}: /{group}/{name} lacks the required 'unit' attribute", details={"path": str(path), "name": name})
        return dataset

    def read(self, path: str | Path, *, verify: bool = True) -> WaveformData:
        """Load every array (SI float64) with units, original units, attributes and stored checksums.

        With ``verify=True`` every stored checksum is recomputed; a mismatch raises
        :class:`DataIntegrityError`. Structural deviations from the layout (missing groups, wrong
        dimensionality, missing unit attributes, inconsistent shapes) raise :class:`DataIntegrityError`.
        """
        lib = _require_h5py()
        path = Path(path)
        if not path.is_file():
            raise DataIntegrityError(f"{path}: file not found", details={"path": str(path)})
        with lib.File(path, "r") as handle:
            attrs = self._open_layout(lib, handle, path)
            arrays: dict[str, np.ndarray] = {}
            array_units: dict[str, str] = {}
            array_original: dict[str, str] = {}
            for name in sorted(handle["sequences"]):
                dataset = self._dataset(lib, handle, "sequences", name, 2, path)
                arrays[name] = np.asarray(dataset[()], dtype=np.float64)
                array_units[name] = str(_py(dataset.attrs["unit"]))
                array_original[name] = str(_py(dataset.attrs.get("original_unit", array_units[name])))
            if not arrays:
                raise DataIntegrityError(f"{path}: /sequences holds no arrays", details={"path": str(path)})
            shapes = {array.shape for array in arrays.values()}
            if len(shapes) != 1:
                raise DataIntegrityError(f"{path}: sequence arrays have different shapes {sorted(shapes)}", details={"path": str(path)})
            n_sequences = next(iter(shapes))[0]
            conditions: dict[str, np.ndarray] = {}
            condition_units: dict[str, str] = {}
            condition_original: dict[str, str] = {}
            if "conditions" in handle:
                for name in sorted(handle["conditions"]):
                    dataset = self._dataset(lib, handle, "conditions", name, 1, path)
                    if dataset.shape[0] != n_sequences:
                        raise DataIntegrityError(f"{path}: /conditions/{name} has {dataset.shape[0]} entries for {n_sequences} sequences",
                                                 details={"path": str(path), "name": name})
                    conditions[name] = np.asarray(dataset[()], dtype=np.float64)
                    condition_units[name] = str(_py(dataset.attrs["unit"]))
                    condition_original[name] = str(_py(dataset.attrs.get("original_unit", condition_units[name])))
            checksums = {}
            if "checksums" in handle:
                checksums = {key: str(_py(value)) for key, value in handle["checksums"].attrs.items() if key != "method"}
        data = WaveformData(arrays=arrays, array_units=array_units, array_original_units=array_original,
                            conditions=conditions, condition_units=condition_units,
                            condition_original_units=condition_original, attrs=attrs, checksums=checksums)
        if verify:
            mismatches = self._mismatches(data)
            if mismatches:
                raise DataIntegrityError(f"{path}: checksum verification failed: " + "; ".join(mismatches),
                                         details={"path": str(path), "mismatches": mismatches})
        return data

    @staticmethod
    def _mismatches(data: WaveformData) -> list[str]:
        problems: list[str] = []
        stored = dict(data.checksums)
        if not stored:
            return ["no checksums recorded in the file"]
        present: dict[str, np.ndarray] = {f"sequences/{k}": v for k, v in data.arrays.items()}
        present.update({f"conditions/{k}": v for k, v in data.conditions.items()})
        for key in sorted(present):
            expected = stored.pop(key, None)
            actual = array_sha256(present[key])
            if expected is None:
                problems.append(f"{key}: no checksum recorded")
            elif expected != actual:
                problems.append(f"{key}: expected {expected}, computed {actual}")
        for key in sorted(stored):
            problems.append(f"{key}: checksum recorded but array missing")
        return problems

    def verify_checksums(self, path: str | Path) -> list[str]:
        """Recompute every array checksum; returns the list of mismatches (empty = array content intact).

        Attributes are not covered (see the module docstring); compare the file-level SHA-256 of the
        record's artifact to detect metadata changes.
        """
        data = self.read(path, verify=False)
        return self._mismatches(data)

    # -- describe ---------------------------------------------------------------------------------
    def describe(self, path: str | Path) -> WaveformDatasetProperties:
        """``WaveformDatasetProperties`` reconstructed from the file's metadata (arrays are not loaded).

        For a file written by :meth:`write` the result equals the properties returned by ``write``
        (including the ``original`` of a converted sampling rate).
        """
        lib = _require_h5py()
        path = Path(path)
        if not path.is_file():
            raise DataIntegrityError(f"{path}: file not found", details={"path": str(path)})
        with lib.File(path, "r") as handle:
            attrs = self._open_layout(lib, handle, path)
            layouts: list[ArrayLayout] = []
            shapes: set[tuple[int, ...]] = set()
            for name in sorted(handle["sequences"]):
                dataset = self._dataset(lib, handle, "sequences", name, 2, path)
                shape = tuple(int(s) for s in dataset.shape)
                shapes.add(shape)
                layouts.append(ArrayLayout(path=f"/sequences/{name}", quantity=str(_py(dataset.attrs.get("quantity", name))),
                                           unit=str(_py(dataset.attrs["unit"])),
                                           original_unit=str(_py(dataset.attrs.get("original_unit", dataset.attrs["unit"]))),
                                           dtype=str(dataset.dtype.name), shape=list(shape),
                                           axis_semantics=["sequence", "sample"],
                                           missing_value=str(_py(dataset.attrs.get("missing_value", MISSING_VALUE)))))
            conditions: dict[str, str] = {}
            condition_lengths: dict[str, int] = {}
            if "conditions" in handle:
                for name in sorted(handle["conditions"]):
                    dataset = self._dataset(lib, handle, "conditions", name, 1, path)
                    conditions[name] = f"/conditions/{name}"
                    condition_lengths[name] = int(dataset.shape[0])
        if not layouts:
            raise DataIntegrityError(f"{path}: /sequences holds no arrays", details={"path": str(path)})
        if len(shapes) != 1:
            raise DataIntegrityError(f"{path}: sequence arrays have different shapes {sorted(shapes)}", details={"path": str(path)})
        shape = next(iter(shapes))
        for name, length in condition_lengths.items():
            if length != shape[0]:
                raise DataIntegrityError(f"{path}: /conditions/{name} has {length} entries for {shape[0]} sequences",
                                         details={"path": str(path), "name": name})
        sampling_rate = None
        if "sampling_rate_hz" in attrs:
            original = None
            if "sampling_rate_original_unit" in attrs and "sampling_rate_original_value" in attrs:
                original = Original(value=float(attrs["sampling_rate_original_value"]), unit=str(attrs["sampling_rate_original_unit"]))
            sampling_rate = Quantity(value=float(attrs["sampling_rate_hz"]), unit="Hz", original=original)
        return WaveformDatasetProperties(
            material_ref=attrs.get("material_ref"), source_dataset_ref=attrs.get("source_dataset_ref"), format="hdf5",
            layout_version=LAYOUT_VERSION, sampling_rate=sampling_rate, samples_per_sequence=shape[1],
            timebase=attrs.get("timebase", "uniform_per_period"), sign_convention=str(attrs.get("sign_convention", DEFAULT_SIGN_CONVENTION)),
            coordinate_definition=str(attrs.get("coordinate_definition", DEFAULT_COORDINATE_DEFINITION)), arrays=layouts,
            conditions=conditions, n_sequences=shape[0], description=attrs.get("description"))

    # -- artifacts --------------------------------------------------------------------------------
    @staticmethod
    def artifact(path: str | Path, *, uri: str | None = None, description: str | None = None) -> Artifact:
        """``Artifact`` entry (role ``waveform_data``) with the file's SHA-256 and size for a record."""
        path = Path(path)
        if not path.is_file():
            raise DataIntegrityError(f"{path}: file not found", details={"path": str(path)})
        return Artifact(uri=uri or path.name, media_type=HDF5_MEDIA_TYPE, role="waveform_data",
                        sha256=sha256_file(path), size_bytes=path.stat().st_size, description=description)


def h5py_available() -> bool:
    return h5py is not None


__all__ = [
    "LAYOUT_VERSION", "HDF5_MEDIA_TYPE", "CHECKSUM_METHOD", "MISSING_VALUE", "EXACT_INTEGER_LIMIT", "SAMPLING_RATE_RTOL",
    "SEQUENCE_QUANTITIES", "CONDITION_QUANTITIES", "WaveformMeta", "WaveformData", "PreparedWaveforms", "WaveformStore",
    "convert_array", "canonicalize_nan", "array_sha256", "missing_mask", "h5py_available",
]
