"""Common adapter for data-driven magnetic models (ANN, LSTM, Transformer, Neural ODE).

Three strictly separated layers make up a :class:`ModelPackage`:

* :class:`ArchitectureSpec` (``architectures.py``) - the network descriptor, no trained numbers;
* :class:`WeightsBundle` - the trained parameters: a file (or external package) in a declared
  ``format`` with an optional SHA-256 digest and a mandatory :class:`SourceReference`;
* :class:`InferenceConfig` - normalisation statistics, dtype, batch size, device and the mapping
  from input features to validity-domain conditions.

:class:`DataDrivenModel` implements the :class:`MagneticModel` contract on top of a package and a
:class:`Backend`. Backends are looked up by weights format and family through a registry
(:func:`get_backend`); the numpy backend is always available, the torch and magnethub backends are
optional and import their dependency lazily, raising :class:`UnsupportedOperationError` with an
installation hint when it is missing (never a silent fallback).

Normalisation (per feature, elementwise; ``mean``/``std`` default to 0/1)::

    none    : x' = x                              inverse  y = y'
    zscore  : x' = (x - mean) / std               inverse  y = y' * std + mean
    minmax  : x' = (x - min) / (max - min)        inverse  y = y' * (max - min) + min      (maps [min, max] -> [0, 1])
    log10   : x' = (log10(x) - mean) / std        inverse  y = 10 ** (y' * std + mean)     (x must be > 0)

Integrity policy (applied in :class:`DataDrivenModel.__init__` through :meth:`ModelPackage.read_weights`):

* the weights file is read **once**; the declared ``sha256`` (and ``size_bytes``, when given) are
  verified on that buffer and the backends parse the same buffer, so no unverified bytes can be
  substituted between the check and the load; a mismatch raises :class:`DataIntegrityError`;
* a missing digest yields the warning ``weights_unverified`` on the model and on every output;
* executable formats (``torchscript``) are refused without a digest, because loading them executes
  content, and the project never executes unverified repository content;
* a relative ``WeightsBundle.path`` must stay inside the package directory (``..`` traversal is
  rejected); an absolute path is accepted as a trusted-caller feature for locally stored models.

Validity domain: ``ValidityRange`` bounds are SI by contract (``Hz``, ``K``, ``T``, ``A/m``);
:class:`ModelPackage` refuses a domain declared in any other unit instead of comparing SI-converted
inputs against non-SI bounds. ``predict(..., policy="reject")`` raises :class:`OutOfDomainError` when
any sample lies outside the declared :class:`ValidityDomain`; ``policy="warn"`` evaluates anyway and
marks the output (``domain.inside=False`` plus an ``out_of_domain`` warning). Inputs are never clipped.
Conditions carried by an input feature (``inference.domain_map``) are always derived from the actual
inputs; ``conditions=`` may only add conditions the feature signature does not carry (for example a
temperature for a temperature-blind surrogate), as one value or one value per sample. Naming a
mapped condition explicitly is an :class:`InvalidInputError`, never a silent override.

Input conventions: ``predict`` takes a mapping ``feature -> value`` where a value is a number, a
sequence/array or a :class:`Quantity` (converted to the feature's declared unit, dimension-checked).
Scalar features have batch shape ``(N,)``, sequence features ``(N, L)``; unbatched values are
accepted and broadcast; ``N = 0`` and ``L = 0`` are rejected. Outputs: for a single sample every
scalar output is returned as a :class:`Quantity` in ``values``; all arrays (batches, sequences) are
placed in ``diagnostics``. When a backend cannot honour ``InferenceConfig.dtype`` (the numpy backend
always computes in float64) the model carries the warning ``dtype_ignored`` instead of staying silent.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from aipe_magnetics.common.errors import (
    DataIntegrityError,
    DataNotFoundError,
    InvalidInputError,
    NumericalError,
    UnitError,
    UnsupportedOperationError,
)
from aipe_magnetics.common.jsonio import dump, load as load_json, sha256_bytes
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import Assumption, Limitation, ModelIdentity, SourceReference, Warning_
from aipe_magnetics.common.units import units
from aipe_magnetics.models.contract import DomainCheck, DomainPolicy, MagneticModel, ModelOutput, ValidityDomain, ValidityRange
from aipe_magnetics.models.datadriven.architectures import FAMILIES, ArchitectureSpec
from aipe_magnetics.models.datadriven.numpy_backend import NumpyBackend

WeightsFormat = Literal["aipe-json", "npz", "torch-state-dict", "torchscript", "equinox", "external-package"]
WEIGHTS_FORMATS: tuple[str, ...] = ("aipe-json", "npz", "torch-state-dict", "torchscript", "equinox", "external-package")
EXECUTABLE_FORMATS: tuple[str, ...] = ("torchscript",)
NormalizationKind = Literal["log10", "zscore", "minmax", "none"]
DomainCondition = Literal["frequency", "temperature", "flux_density_peak", "field_peak"]
CONDITION_SI_UNITS: dict[str, str] = {"frequency": "Hz", "temperature": "K", "flux_density_peak": "T", "field_peak": "A/m"}
PACKAGE_SCHEMA = "aipe.magnetics.model-package/0.2"
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


# --------------------------------------------------------------------------------------------------
# Unit helper for arrays
# --------------------------------------------------------------------------------------------------
def convert_array(values: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    """Elementwise, dimension-checked unit conversion of a float array (exact factor/offset arithmetic)."""
    src, dst = units.get(from_unit), units.get(to_unit)
    array = np.asarray(values, dtype=np.float64)
    if src.dimension != dst.dimension:
        raise UnitError(f"cannot convert {from_unit!r} to {to_unit!r}: incompatible dimensions",
                        details={"from": from_unit, "to": to_unit})
    if from_unit == to_unit:
        return array.copy()
    if from_unit in ("rad", "deg") or to_unit in ("rad", "deg"):
        return np.vectorize(lambda v: units.convert(float(v), from_unit, to_unit), otypes=[np.float64])(array)
    si = array * float(src.factor) + float(src.offset)
    return (si - float(dst.offset)) / float(dst.factor)


# --------------------------------------------------------------------------------------------------
# Inference configuration
# --------------------------------------------------------------------------------------------------
class NormalizationSpec(BaseModel):
    """Elementwise feature scaling; see the module docstring for the formulas."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: NormalizationKind = "none"
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "NormalizationSpec":
        for name in ("mean", "std", "min", "max"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.kind == "zscore":
            if self.mean is None or self.std is None:
                raise ValueError("zscore normalisation needs mean and std")
        if self.kind in ("zscore", "log10") and self.std is not None and self.std <= 0:
            raise ValueError("std must be positive")
        if self.kind == "minmax":
            if self.min is None or self.max is None:
                raise ValueError("minmax normalisation needs min and max")
            if not self.max > self.min:
                raise ValueError("minmax needs max > min")
        if self.kind == "none" and any(v is not None for v in (self.mean, self.std, self.min, self.max)):
            raise ValueError("kind 'none' takes no statistics")
        if self.kind in ("zscore", "log10") and (self.min is not None or self.max is not None):
            raise ValueError(f"kind {self.kind!r} does not use min/max")
        if self.kind == "minmax" and (self.mean is not None or self.std is not None):
            raise ValueError("kind 'minmax' does not use mean/std")
        return self

    @property
    def is_affine(self) -> bool:
        return self.kind in ("none", "zscore", "minmax")

    def forward(self, x: np.ndarray, *, name: str = "feature") -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.kind == "none":
            return x
        if self.kind == "zscore":
            return (x - float(self.mean)) / float(self.std)  # type: ignore[arg-type]
        if self.kind == "minmax":
            return (x - float(self.min)) / (float(self.max) - float(self.min))  # type: ignore[arg-type]
        if np.any(x <= 0):
            raise InvalidInputError(f"{name}: log10 normalisation needs strictly positive values "
                                    f"(min={float(np.min(x)):g})", details={"feature": name})
        mean = 0.0 if self.mean is None else float(self.mean)
        std = 1.0 if self.std is None else float(self.std)
        return (np.log10(x) - mean) / std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        if self.kind == "none":
            return z
        if self.kind == "zscore":
            return z * float(self.std) + float(self.mean)  # type: ignore[arg-type]
        if self.kind == "minmax":
            return z * (float(self.max) - float(self.min)) + float(self.min)  # type: ignore[arg-type]
        mean = 0.0 if self.mean is None else float(self.mean)
        std = 1.0 if self.std is None else float(self.std)
        with np.errstate(over="ignore"):
            return np.power(10.0, z * std + mean)


class InferenceConfig(BaseModel):
    """Normalisation statistics and runtime settings (nothing about the architecture or the weights)."""

    model_config = ConfigDict(extra="forbid")
    input_normalization: dict[str, NormalizationSpec] = Field(default_factory=dict)
    output_denormalization: dict[str, NormalizationSpec] = Field(default_factory=dict)
    dtype: Literal["float32", "float64"] = "float64"
    batch_size: int = Field(default=1024, ge=1)
    device: str = Field(default="cpu", min_length=1)
    domain_map: dict[DomainCondition, str] = Field(
        default_factory=dict, description="validity-domain condition -> input feature that carries it")

    def input_scaler(self, feature: str) -> NormalizationSpec:
        return self.input_normalization.get(feature, NormalizationSpec())

    def output_scaler(self, feature: str) -> NormalizationSpec:
        return self.output_denormalization.get(feature, NormalizationSpec())


# --------------------------------------------------------------------------------------------------
# Trained parameters
# --------------------------------------------------------------------------------------------------
class WeightsBundle(BaseModel):
    """Where the trained parameters are, in which format, with which digest, and where they came from."""

    model_config = ConfigDict(extra="forbid")
    format: WeightsFormat
    path: str | None = Field(default=None, description="file path, absolute or relative to the package metadata file")
    uri: str | None = Field(default=None, description="URI or package identifier (e.g. pkg:pypi/mag-net-hub)")
    sha256: str | None = Field(default=None, description="SHA-256 of the weights file bytes (lowercase hex)")
    size_bytes: int | None = Field(default=None, ge=0)
    source: SourceReference
    note: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "WeightsBundle":
        if self.path is None and self.uri is None:
            raise ValueError("weights need a path or a uri")
        if self.sha256 is not None and not _SHA256_RE.match(self.sha256):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        if self.format == "external-package" and self.path is not None:
            raise ValueError("external-package weights are owned by the package; do not give a path")
        if self.format != "external-package" and self.path is None:
            raise ValueError(f"format {self.format!r} needs a local path")
        return self


# --------------------------------------------------------------------------------------------------
# The package
# --------------------------------------------------------------------------------------------------
_REAL_SOURCE_KINDS = ("datasheet", "publication", "repository", "dataset", "standard", "measurement_record")


class ModelPackage(BaseModel):
    """Architecture + weights + inference configuration + identity, validity domain and provenance."""

    model_config = ConfigDict(extra="forbid")

    package_schema: Literal["aipe.magnetics.model-package/0.2"] = PACKAGE_SCHEMA
    identity: ModelIdentity
    architecture: ArchitectureSpec
    weights: WeightsBundle
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    domain: ValidityDomain = Field(default_factory=ValidityDomain)
    sources: list[SourceReference] = Field(min_length=1)
    synthetic: bool = Field(description="True when the training data / parameters are synthetic or illustrative")
    description: str = ""
    training: dict[str, Any] = Field(default_factory=dict, description="training record: data, optimiser, seed, metrics")

    _base_dir: Path | None = PrivateAttr(default=None)
    _weights_cache: tuple[bytes, str] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _consistent(self) -> "ModelPackage":
        if self.identity.family != "data_driven":
            raise ValueError("identity.family must be 'data_driven'")
        spec = self.architecture
        for label, mapping, names in (("input_normalization", self.inference.input_normalization, spec.input_features),
                                      ("output_denormalization", self.inference.output_denormalization, spec.output_features)):
            unknown = [n for n in mapping if n not in names]
            if unknown:
                raise ValueError(f"inference.{label} refers to unknown features: {unknown}")
        for condition, feature in self.inference.domain_map.items():
            if feature not in spec.input_features:
                raise ValueError(f"inference.domain_map[{condition!r}] = {feature!r} is not an input feature")
            if not units.compatible(spec.unit_of(feature), CONDITION_SI_UNITS[condition]):
                raise ValueError(f"feature {feature!r} ({spec.unit_of(feature)}) cannot carry the condition "
                                 f"{condition!r} ({CONDITION_SI_UNITS[condition]})")
        for condition, si_unit in CONDITION_SI_UNITS.items():
            rng: ValidityRange | None = getattr(self.domain, condition)
            if rng is None:
                continue
            units.get(rng.unit)  # unknown unit -> UnitError (a ValueError)
            if rng.unit != si_unit:
                raise ValueError(f"domain.{condition} is declared in {rng.unit!r}; ValidityRange bounds are SI by contract, "
                                 f"so convert them to {si_unit!r} (inputs are compared in {si_unit})")
        if len({s.id for s in self.sources}) != len(self.sources):
            raise ValueError("source ids must be unique")
        labelled = any("synthetic" in f"{s.reference} {s.note or ''}".lower() for s in self.sources)
        if self.synthetic and not labelled:
            raise ValueError("synthetic packages must say so in at least one source reference or note")
        if not self.synthetic and not any(s.kind in _REAL_SOURCE_KINDS for s in self.sources):
            raise ValueError("a non-synthetic package needs at least one datasheet/publication/dataset/repository source")
        return self

    # -- location -----------------------------------------------------------------------------------
    @property
    def base_dir(self) -> Path | None:
        return self._base_dir

    def with_base_dir(self, directory: str | Path) -> "ModelPackage":
        clone = self.model_copy()
        clone._base_dir = Path(directory)
        return clone

    def weights_path(self) -> Path:
        """Resolved weights file path.

        A relative ``WeightsBundle.path`` is resolved against the package directory and must stay
        inside it (``..`` traversal and symlinks escaping it are rejected). An absolute path is
        accepted unchanged: a trusted-caller feature for models stored elsewhere on the local disk.
        """
        if self.weights.format == "external-package":
            raise UnsupportedOperationError("external-package weights are owned by the external package; no local file",
                                            details={"uri": self.weights.uri})
        assert self.weights.path is not None
        path = Path(self.weights.path)
        if path.is_absolute():
            return path
        if self._base_dir is None:
            raise InvalidInputError(f"weights path {self.weights.path!r} is relative but the package has no base "
                                    "directory; load it with ModelPackage.load() or call with_base_dir()")
        base = self._base_dir.resolve()
        full = (base / path).resolve()
        try:
            full.relative_to(base)
        except ValueError:
            raise InvalidInputError(
                f"weights path {self.weights.path!r} escapes the package directory {str(base)!r}; relative weights "
                "paths must stay inside the package (use an absolute path for files stored elsewhere)",
                details={"path": self.weights.path, "base_dir": str(base)}) from None
        return full

    def read_weights(self) -> tuple[bytes, str]:
        """Read the weights file once and return ``(bytes, sha256)`` after verifying the declaration.

        The declared ``sha256`` and ``size_bytes`` are checked against the buffer that is returned,
        so a backend parsing these bytes never works on anything other than the verified content.
        The result is cached on the package until :meth:`release_weights` is called.
        """
        if self._weights_cache is not None:
            return self._weights_cache
        path = self.weights_path()
        if not path.is_file():
            raise DataNotFoundError(f"weights file not found: {path}", details={"path": str(path)})
        try:
            data = path.read_bytes()
        except OSError as error:
            raise DataNotFoundError(f"weights file not readable: {path} ({error})", details={"path": str(path)}) from None
        digest = sha256_bytes(data)
        declared = self.weights.sha256
        if declared is not None and digest != declared:
            raise DataIntegrityError(
                f"weights file {path.name} does not match its declared sha256 (declared {declared[:12]}..., "
                f"actual {digest[:12]}...); refusing to load tampered or corrupted parameters",
                details={"path": str(path), "expected": declared, "actual": digest})
        if self.weights.size_bytes is not None and len(data) != self.weights.size_bytes:
            raise DataIntegrityError(
                f"weights file {path.name} has {len(data)} bytes but the package declares size_bytes="
                f"{self.weights.size_bytes}; refusing to load a file that does not match its declaration",
                details={"path": str(path), "expected": self.weights.size_bytes, "actual": len(data)})
        self._weights_cache = (data, digest)
        return self._weights_cache

    def release_weights(self) -> None:
        """Drop the buffer cached by :meth:`read_weights` (called once the backend has loaded)."""
        self._weights_cache = None

    # -- persistence --------------------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "ModelPackage":
        path = Path(path)
        if not path.is_file():
            raise DataNotFoundError(f"package metadata not found (or not a file): {path}", details={"path": str(path)})
        try:
            data = load_json(path)
        except OSError as error:
            raise DataNotFoundError(f"package metadata not readable: {path} ({error})", details={"path": str(path)}) from None
        except ValueError as error:  # malformed/duplicate-key/non-finite JSON, invalid UTF-8
            raise DataIntegrityError(f"package metadata {path.name} is not valid strict JSON: {error}") from None
        try:
            package = cls.model_validate(data)
        except ValueError as error:
            raise DataIntegrityError(f"package metadata {path.name} is invalid: {error}") from None
        package._base_dir = path.resolve().parent
        return package

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        return dump(self.model_dump(mode="json", exclude_none=True), path, overwrite=overwrite)

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


# --------------------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------------------
@runtime_checkable
class Backend(Protocol):
    """A loaded inference engine. ``run`` maps normalised per-feature inputs to normalised outputs."""

    name: str

    def load(self, package: ModelPackage) -> None: ...

    def run(self, package: ModelPackage, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]: ...


BackendFactory = Callable[[], Backend]
_BACKENDS: dict[str, tuple[BackendFactory, dict[str, frozenset[str]]]] = {}


def register_backend(name: str, factory: BackendFactory, supports: Mapping[str, Iterable[str]], *,
                     replace: bool = False) -> None:
    """Register ``factory`` for the ``{format: families}`` combinations it supports.

    Formats and families must be known (:data:`WEIGHTS_FORMATS`, :data:`FAMILIES`); re-registering an
    existing ``name`` is an error unless ``replace=True`` makes the override deliberate.
    """
    if not isinstance(name, str) or not name:
        raise InvalidInputError("backend name must be a non-empty string")
    if not callable(factory):
        raise InvalidInputError(f"backend {name!r}: factory must be callable")
    table = {fmt: frozenset(fams) for fmt, fams in supports.items()}
    if not table:
        raise InvalidInputError(f"backend {name!r} must support at least one (format, family) combination")
    for fmt, fams in table.items():
        if fmt not in WEIGHTS_FORMATS:
            raise InvalidInputError(f"unknown weights format {fmt!r}; known: {', '.join(WEIGHTS_FORMATS)}")
        unknown = sorted(set(fams) - set(FAMILIES))
        if unknown or not fams:
            raise InvalidInputError(f"backend {name!r}: format {fmt!r} lists unknown or no families {unknown}; "
                                    f"known families: {', '.join(FAMILIES)}", details={"unknown": unknown})
    if name in _BACKENDS and not replace:
        raise InvalidInputError(f"backend {name!r} is already registered; pass replace=True to override it deliberately",
                                details={"name": name})
    _BACKENDS[name] = (factory, table)


def unregister_backend(name: str) -> bool:
    """Remove a registered backend; returns whether it existed."""
    return _BACKENDS.pop(name, None) is not None


def available_backends() -> dict[str, dict[str, list[str]]]:
    return {name: {fmt: sorted(fams) for fmt, fams in table.items()} for name, (_, table) in _BACKENDS.items()}


def get_backend(format: str, family: str) -> Backend:
    """Return a fresh backend instance for ``(format, family)`` or raise an explicit error."""
    if format not in WEIGHTS_FORMATS:
        raise InvalidInputError(f"unknown weights format {format!r}; known: {', '.join(WEIGHTS_FORMATS)}")
    candidates = [(name, factory) for name, (factory, table) in _BACKENDS.items() if family in table.get(format, ())]
    if candidates:
        return candidates[0][1]()
    if family == "transformer" and format in ("aipe-json", "npz"):
        raise UnsupportedOperationError(
            "transformer inference is not implemented in the numpy backend; export the model to TorchScript "
            "(format 'torchscript') and use the optional torch backend (install aipe-magnetics[torch])",
            details={"format": format, "family": family})
    if format == "equinox":
        raise UnsupportedOperationError(
            "no backend executes 'equinox' weights in AIPE Magnetics v0.2 (JAX/Equinox models such as the "
            "MagNet-Hub sequence models); re-export to npz/aipe-json or TorchScript",
            details={"format": format, "family": family})
    raise UnsupportedOperationError(f"no backend supports format {format!r} for family {family!r}; "
                                    f"available: {available_backends()}", details={"format": format, "family": family})


# --------------------------------------------------------------------------------------------------
# Input coercion
# --------------------------------------------------------------------------------------------------
def _to_array(name: str, value: Any, unit: str) -> np.ndarray:
    if isinstance(value, Quantity):
        try:
            return np.asarray(value.magnitude(unit), dtype=np.float64)
        except UnitError as error:
            raise InvalidInputError(f"{name}: {error.message}", details={"feature": name}) from None
    if isinstance(value, bool) or (isinstance(value, np.ndarray) and value.dtype == bool):
        raise InvalidInputError(f"{name}: booleans are not numeric inputs", details={"feature": name})
    if isinstance(value, dict):
        return _to_array(name, Quantity.parse(value, name=name), unit)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise InvalidInputError(f"{name}: cannot interpret value as an array ({error})", details={"feature": name}) from None
    if array.dtype.kind not in "iuf":
        raise InvalidInputError(f"{name}: inputs must be real numbers, got dtype {array.dtype}", details={"feature": name})
    array = array.astype(np.float64)
    if not np.all(np.isfinite(array)):
        raise InvalidInputError(f"{name}: inputs must be finite", details={"feature": name})
    return array


def coerce_inputs(spec: ArchitectureSpec, inputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Validate ``inputs`` against the feature signature; returns float64 arrays ``(N,)`` / ``(N, L)``."""
    if not isinstance(inputs, Mapping):
        raise InvalidInputError(f"inputs must be a mapping feature -> value, got {type(inputs).__name__}")
    missing = [n for n in spec.input_features if n not in inputs]
    unexpected = [n for n in inputs if n not in spec.input_features]
    if missing or unexpected:
        raise InvalidInputError(f"input features mismatch: missing {missing}, unexpected {unexpected}; "
                                f"expected exactly {spec.input_features}", details={"missing": missing, "unexpected": unexpected})
    arrays: dict[str, np.ndarray] = {}
    lengths: set[int] = set()
    for name in spec.input_features:
        array = _to_array(name, inputs[name], spec.unit_of(name))
        if name in spec.sequence_features:
            if array.ndim == 1:
                array = array[None, :]
            if array.ndim != 2:
                raise InvalidInputError(f"{name}: sequence feature needs shape (L,) or (N, L), got {array.shape}")
            lengths.add(int(array.shape[1]))
        else:
            if array.ndim == 0:
                array = array.reshape(1)
            if array.ndim != 1:
                raise InvalidInputError(f"{name}: scalar feature needs shape () or (N,), got {array.shape}")
        arrays[name] = array
    if len(lengths) > 1:
        raise InvalidInputError(f"sequence features have different lengths {sorted(lengths)}")
    if 0 in lengths:
        raise InvalidInputError("sequence features must have at least one sample (L >= 1); a recurrent model applied "
                                "to zero time steps would only report its initial state", details={"sequence_length": 0})
    if lengths and spec.sequence_length is not None and lengths != {spec.sequence_length}:
        raise InvalidInputError(f"sequence length {lengths.pop()} does not match the architecture's "
                                f"sequence_length={spec.sequence_length}; resample explicitly before calling the model")
    n_batch = max(int(a.shape[0]) for a in arrays.values())
    if n_batch == 0:
        raise InvalidInputError("empty batch: every feature has zero samples (N = 0); nothing to evaluate",
                                details={"batch_size": 0})
    for name, array in arrays.items():
        if array.shape[0] == n_batch:
            continue
        if array.shape[0] != 1:
            raise InvalidInputError(f"{name}: batch size {array.shape[0]} is incompatible with {n_batch}")
        arrays[name] = np.ascontiguousarray(np.broadcast_to(array, (n_batch,) + array.shape[1:]))
    return arrays


def _outside_mask(rng: ValidityRange, values: np.ndarray) -> np.ndarray:
    mask = ~np.isfinite(values)
    if rng.min is not None:
        mask |= values < rng.min
    if rng.max is not None:
        mask |= values > rng.max
    return mask


# --------------------------------------------------------------------------------------------------
# Explicit validity conditions (shared by DataDrivenModel and NeuralODEModel)
# --------------------------------------------------------------------------------------------------
def condition_array(name: str, value: Any, *, n_batch: int | None = None) -> np.ndarray:
    """Coerce an explicit validity condition to a 1-D float array in its SI unit.

    ``value`` is a number, a :class:`Quantity` (dimension-checked, converted) or a 1-D sequence with
    one value per sample. With ``n_batch`` given, a single value is repeated for the batch and any
    other length is rejected; nothing is truncated.
    """
    if name not in CONDITION_SI_UNITS:
        raise InvalidInputError(f"unknown domain condition {name!r}; known: {', '.join(CONDITION_SI_UNITS)}",
                                details={"condition": name})
    array = _to_array(name, value, CONDITION_SI_UNITS[name])
    if array.ndim > 1:
        raise InvalidInputError(f"{name}: a validity condition is a scalar or a 1-D per-sample array, got shape {array.shape}",
                                details={"condition": name})
    array = np.ascontiguousarray(array.reshape(-1))
    if array.size == 0:
        raise InvalidInputError(f"{name}: empty condition array", details={"condition": name})
    if n_batch is not None:
        if array.size == 1 and n_batch != 1:
            array = np.full(n_batch, float(array[0]))
        elif array.size != n_batch:
            raise InvalidInputError(f"{name}: expected one value or {n_batch} per-sample values, got {array.size}",
                                    details={"condition": name, "batch_size": n_batch, "values": int(array.size)})
    return array


def reject_impossible_conditions(conditions: Mapping[str, np.ndarray]) -> None:
    """Physically impossible conditions are invalid inputs, not merely out of domain."""
    for condition, raw in conditions.items():
        values = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise InvalidInputError(f"{condition} must be finite", details={"condition": condition})
        if condition in ("frequency", "temperature") and np.any(values <= 0):
            raise InvalidInputError(f"{condition} must be positive ({CONDITION_SI_UNITS[condition]}), "
                                    f"got min {float(np.min(values)):g}", details={"condition": condition})
        if condition in ("flux_density_peak", "field_peak") and np.any(values < 0):
            raise InvalidInputError(f"{condition} is a peak magnitude and cannot be negative, "
                                    f"got min {float(np.min(values)):g}", details={"condition": condition})


# --------------------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------------------
class DataDrivenModel(MagneticModel):
    """A :class:`MagneticModel` backed by a :class:`ModelPackage` and an inference :class:`Backend`."""

    def __init__(self, package: ModelPackage, *, backend: Backend | None = None,
                 require_sha256_for_executable_formats: bool = True) -> None:
        self.package = package
        self.identity = package.identity
        self.domain = package.domain
        self.sources = list(package.sources)
        self.parameters = []
        self.load_warnings: list[Warning_] = []
        fmt = package.weights.format
        self.integrity: dict[str, Any] = {"format": fmt, "sha256_declared": package.weights.sha256,
                                          "sha256_actual": None, "verified": False}
        if fmt == "external-package":
            self.load_warnings.append(Warning_(
                code="weights_unverified",
                message=f"weights are owned by the external package {package.weights.uri!r}; AIPE Magnetics did not "
                        "verify their integrity"))
        else:
            declared = package.weights.sha256
            if declared is None and fmt in EXECUTABLE_FORMATS and require_sha256_for_executable_formats:
                raise DataIntegrityError(  # decided from the declaration alone, before the file is even read
                    f"refusing to load executable {fmt!r} weights without a declared sha256 "
                    "(loading executes the archive; declare WeightsBundle.sha256)",
                    details={"path": str(package.weights.path), "format": fmt})
            _, actual = package.read_weights()  # existence, digest and size verified on the cached buffer
            self.integrity["sha256_actual"] = actual
            if declared is not None:
                self.integrity["verified"] = True
            else:
                self.load_warnings.append(Warning_(
                    code="weights_unverified",
                    message=f"weights file {package.weights_path().name} has no declared sha256; integrity was not "
                            f"verified (actual digest {actual})"))
        try:
            self.backend: Backend = backend if backend is not None else get_backend(fmt, package.architecture.family)
            self.backend.load(package)  # backends parse the buffer verified above
        finally:
            package.release_weights()
        backend_dtype = getattr(self.backend, "dtype", None)
        if isinstance(backend_dtype, str) and backend_dtype != package.inference.dtype:
            self.load_warnings.append(Warning_(
                code="dtype_ignored",
                message=f"backend {self.backend.name!r} computes in {backend_dtype}; the configured inference dtype "
                        f"{package.inference.dtype!r} is not honoured by this backend"))

    # -- construction helpers -----------------------------------------------------------------------
    @classmethod
    def from_path(cls, metadata_path: str | Path, **kwargs: Any) -> "DataDrivenModel":
        return cls(ModelPackage.load(metadata_path), **kwargs)

    # -- domain -------------------------------------------------------------------------------------
    def _domain_conditions(self, arrays: Mapping[str, np.ndarray], conditions: Mapping[str, Any] | None,
                           n_batch: int) -> dict[str, np.ndarray]:
        """SI condition arrays: mapped conditions come from the inputs, explicit ones only add to them."""
        spec = self.package.architecture
        mapped = dict(self.package.inference.domain_map)
        out: dict[str, np.ndarray] = {}
        for condition, feature in mapped.items():
            values = convert_array(arrays[feature], spec.unit_of(feature), CONDITION_SI_UNITS[condition])
            if feature in spec.sequence_features:
                values = np.max(np.abs(values), axis=1)
            out[condition] = values
        for condition, value in dict(conditions or {}).items():
            if condition in mapped:
                raise InvalidInputError(
                    f"condition {condition!r} is derived from the input feature {mapped[condition]!r} "
                    "(inference.domain_map) and cannot be overridden through conditions=; change the input instead",
                    details={"condition": condition, "feature": mapped[condition]})
            out[condition] = condition_array(condition, value, n_batch=n_batch)
        reject_impossible_conditions(out)
        return out

    def check_domain_arrays(self, conditions: Mapping[str, np.ndarray], *, material: str | None = None,
                            excitation: str | None = None, dc_bias: bool | None = None) -> DomainCheck:
        """Vectorised :meth:`check_domain`: every sample of every SI condition array is checked."""
        base = self.check_domain(material=material, excitation=excitation, dc_bias=dc_bias)
        violations = list(base.violations)
        checked = dict(base.checked)
        for condition in CONDITION_SI_UNITS:
            values = conditions.get(condition)
            rng: ValidityRange | None = getattr(self.domain, condition)
            if values is None or rng is None:
                continue
            if rng.unit != CONDITION_SI_UNITS[condition]:  # packages validate this; model_copy() can bypass validation
                raise DataIntegrityError(
                    f"validity range for {condition} is declared in {rng.unit!r}; ValidityRange bounds must be SI "
                    f"({CONDITION_SI_UNITS[condition]}) - the package is inconsistent",
                    details={"condition": condition, "unit": rng.unit})
            values = np.asarray(values, dtype=np.float64)
            checked[condition] = rng.describe()
            mask = _outside_mask(rng, values)
            if np.any(mask):
                bad = values[mask]
                violations.append(f"{condition}: {int(mask.sum())} of {values.size} sample(s) outside {rng.describe()} "
                                  f"(min {float(np.min(bad)):g}, max {float(np.max(bad)):g} {rng.unit})")
        return DomainCheck(inside=not violations, violations=violations, checked=checked)

    # -- evaluation ---------------------------------------------------------------------------------
    def predict(self, inputs: Mapping[str, Any], *, policy: DomainPolicy = "reject", material: str | None = None,
                excitation: str | None = None, dc_bias: bool | None = None,
                conditions: Mapping[str, Any] | None = None) -> ModelOutput:
        """Run inference. See the module docstring for input conventions and the domain policy."""
        if policy not in ("reject", "warn"):
            raise InvalidInputError(f"policy must be 'reject' or 'warn', got {policy!r}")
        spec = self.package.architecture
        if spec.family == "neural_ode":
            raise UnsupportedOperationError(
                "neural_ode packages are integrated in time: use NeuralODEModel.simulate() "
                "(the raw vector field is NeuralODEModel.vector_field())", details={"family": "neural_ode"})
        config = self.package.inference
        arrays = coerce_inputs(spec, inputs)
        n_batch = int(next(iter(arrays.values())).shape[0])
        length = next((int(a.shape[1]) for n, a in arrays.items() if n in spec.sequence_features), None)

        domain = self.check_domain_arrays(self._domain_conditions(arrays, conditions, n_batch), material=material,
                                          excitation=excitation, dc_bias=dc_bias)
        if not domain.inside and policy == "reject":
            domain.raise_if_outside(self.identity.id)

        normalised = {name: config.input_scaler(name).forward(array, name=name) for name, array in arrays.items()}
        raw = self.backend.run(self.package, normalised)

        outputs: dict[str, np.ndarray] = {}
        for name in spec.output_features:
            if name not in raw:
                raise DataIntegrityError(f"backend {self.backend.name!r} did not return output {name!r}",
                                         details={"missing": name})
            y = np.asarray(raw[name], dtype=np.float64)
            expected = (n_batch, length) if (name in spec.sequence_features and length is not None) else (n_batch,)
            if name in spec.sequence_features and length is None and y.ndim == 2 and y.shape[0] == n_batch:
                expected = y.shape  # external backends may define the output sequence length themselves
            if y.shape != expected:
                raise DataIntegrityError(f"backend output {name!r} has shape {y.shape}, expected {expected}")
            y = config.output_scaler(name).inverse(y)
            if not np.all(np.isfinite(y)):
                raise NumericalError(f"output {name!r} is not finite; inputs drive the network outside its numeric "
                                     "range", details={"output": name})
            outputs[name] = y

        values: dict[str, Quantity] = {}
        if n_batch == 1:
            for name, y in outputs.items():
                if name not in spec.sequence_features:
                    values[name] = Quantity(value=float(y[0]), unit=spec.unit_of(name))

        warnings = list(self.load_warnings)
        if not domain.inside:
            warnings.append(Warning_(code="out_of_domain", severity="warning",
                                     message="inputs outside the declared validity domain were evaluated under "
                                             "policy='warn': " + "; ".join(domain.violations)))
        assumptions = [Assumption(
            id="input_units", basis="model_assumption", affects=list(spec.output_features),
            statement="Inputs are interpreted in the declared feature units: " +
                      ", ".join(f"{n} [{spec.unit_of(n)}]" for n in spec.input_features) +
                      "; outputs are returned in " + ", ".join(f"{n} [{spec.unit_of(n)}]" for n in spec.output_features) + ".")]
        limitations = [Limitation(code="data_driven_surrogate", statement=(
            "Data-driven surrogate: outputs interpolate the training data and carry no physical guarantee "
            "outside the declared validity domain; uncertainty is not quantified (unknown)."))]
        if self.package.synthetic:
            limitations.append(Limitation(code="synthetic_parameters", statement=(
                "The trained parameters derive from SYNTHETIC/illustrative data; results do not describe a real material.")))
        for i, statement in enumerate(self.domain.known_limitations, start=1):
            limitations.append(Limitation(code=f"known_limitation_{i}", statement=statement))

        diagnostics: dict[str, Any] = {
            "backend": self.backend.name,
            "batch_size": n_batch,
            "sequence_length": length,
            "dtype": getattr(self.backend, "dtype", None) or "unknown",  # the precision actually used
            "dtype_configured": config.dtype,
            "weights": dict(self.integrity),
            "outputs": {name: y.tolist() for name, y in outputs.items()},
            "domain_conditions_checked": sorted(domain.checked),
        }
        return ModelOutput(model=self.identity, values=values, domain=domain, assumptions=assumptions,
                           limitations=limitations, warnings=warnings, sources=list(self.sources), diagnostics=diagnostics)

    def evaluate(self, *args: Any, policy: DomainPolicy = "reject", **kwargs: Any) -> ModelOutput:
        if len(args) > 1:
            raise InvalidInputError("evaluate() takes at most one positional argument: the inputs mapping")
        inputs = args[0] if args else kwargs.pop("inputs", None)
        if inputs is None:
            raise InvalidInputError("evaluate() needs the inputs mapping (feature -> value)")
        return self.predict(inputs, policy=policy, **kwargs)

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["package"] = {
            "architecture": self.package.architecture.model_dump(mode="json", exclude_none=True),
            "weights": self.package.weights.model_dump(mode="json", exclude_none=True),
            "inference": self.package.inference.model_dump(mode="json", exclude_none=True),
            "synthetic": self.package.synthetic,
            "integrity": dict(self.integrity),
            "backend": self.backend.name,
        }
        return info


# --------------------------------------------------------------------------------------------------
# Default backend registrations (optional backends import their dependency lazily inside load())
# --------------------------------------------------------------------------------------------------
register_backend("numpy", NumpyBackend, {"aipe-json": NumpyBackend.families, "npz": NumpyBackend.families})

from aipe_magnetics.models.datadriven.torch_backend import TorchBackend  # noqa: E402  (no torch import here)

register_backend("torch", TorchBackend, {"torch-state-dict": ("ann", "lstm"), "torchscript": ("ann", "lstm", "transformer")})

from aipe_magnetics.models.datadriven.magnethub_backend import MagnetHubBackend  # noqa: E402  (no magnethub import here)

register_backend("magnethub", MagnetHubBackend, {"external-package": ("ann", "lstm")})

__all__ = [
    "Backend", "BackendFactory", "CONDITION_SI_UNITS", "DataDrivenModel", "DomainCondition", "EXECUTABLE_FORMATS",
    "InferenceConfig", "ModelPackage", "NormalizationKind", "NormalizationSpec", "PACKAGE_SCHEMA", "WEIGHTS_FORMATS",
    "WeightsBundle", "WeightsFormat", "available_backends", "coerce_inputs", "condition_array", "convert_array",
    "get_backend", "register_backend", "reject_impossible_conditions", "unregister_backend", "ArchitectureSpec",
]
