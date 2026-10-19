"""Pure-numpy inference backend (float64, deterministic, no optional dependencies).

Equations and conventions
-------------------------

Dense layer ``i`` (PyTorch ``nn.Linear`` layout, ``W`` of shape ``(out, in)``)::

    y = act(x @ W.T + b)

Activations::

    relu(x)     = max(x, 0)
    tanh(x)     = tanh(x)
    gelu(x)     = 0.5 * x * (1 + erf(x / sqrt(2)))          (exact form, torch default approximate='none')
    sigmoid(x)  = 1 / (1 + exp(-x))                         (scipy.special.expit, overflow-safe)
    identity(x) = x

LSTM cell (identical to ``torch.nn.LSTM``, one layer, batch-first), with gate pre-activations
stacked in the order ``i, f, g, o`` along the first axis of ``W_ih`` ``(4H, in)``, ``W_hh`` ``(4H, H)``,
``b_ih`` and ``b_hh`` ``(4H,)``::

    a_t = x_t @ W_ih.T + b_ih + h_{t-1} @ W_hh.T + b_hh
    i_t = sigmoid(a_t[0H:1H])      input gate
    f_t = sigmoid(a_t[1H:2H])      forget gate
    g_t = tanh(a_t[2H:3H])         cell candidate
    o_t = sigmoid(a_t[3H:4H])      output gate
    c_t = f_t * c_{t-1} + i_t * g_t
    h_t = o_t * tanh(c_t)
    h_0 = c_0 = 0

Stacked layers feed ``h_t`` of layer ``k`` as ``x_t`` of layer ``k+1``. A dense head is applied to
``h_T`` (``output_mode='last'``) or to every ``h_t`` (``output_mode='sequence'``). Because the
weight layout equals PyTorch's, a ``torch.nn.LSTM`` state dict exported to ``npz`` or ``aipe-json``
loads unchanged.

Weight file formats
-------------------

``aipe-json`` (UTF-8 JSON, lists of numbers, no binary blobs)::

    {"format": "aipe-json", "format_version": "0.2", "family": "ann",
     "layers": [{"W": [[...], ...], "b": [...], "activation": "tanh"}, ...]}

    {"format": "aipe-json", "format_version": "0.2", "family": "lstm",
     "lstm": [{"W_ih": [[...]], "W_hh": [[...]], "b_ih": [...], "b_hh": [...]}, ...],
     "head": [{"W": [[...]], "b": [...], "activation": "identity"}, ...]}

The ``family`` of a Neural-ODE vector field file is ``"neural_ode"`` with the ``layers`` layout.
The header fields ``family`` and ``architecture`` and the activation tags inside the file are
*redundant* with the architecture descriptor and are cross-checked on load: a disagreement is a
:class:`DataIntegrityError` (a weights file written for another family or architecture never loads
silently, even when the dense shapes happen to match).

``npz`` (``numpy.savez``, loaded with ``allow_pickle=False``): arrays named exactly like the PyTorch
state dict (``layers.{i}.weight``, ``layers.{i}.bias``, ``lstm.weight_ih_l{k}``, ..., ``head.{i}.weight``,
``head.{i}.bias``). Activations come from the descriptor only. A ``.npy`` single array, an empty or
truncated file and pickled content are all rejected with :class:`DataIntegrityError`.

Both loaders accept a path *or the file bytes*; :class:`NumpyBackend` always parses the bytes
returned by :meth:`ModelPackage.read_weights`, i.e. the very buffer whose SHA-256 and size were
verified (no second, unverified read of the file). They produce a *parameter dictionary*
``{name: float64 array}``; :func:`build_network` validates names and shapes against
:meth:`ArchitectureSpec.expected_parameter_shapes` and refuses missing, unexpected, mis-shaped or
non-finite parameters.

The backend computes in float64 regardless of ``InferenceConfig.dtype`` and says so through its
``dtype`` attribute; :class:`DataDrivenModel` turns a mismatch into the warning ``dtype_ignored``.

Limitation: the ``transformer`` family is a descriptor only; this backend raises
:class:`UnsupportedOperationError` for it (inference via TorchScript + torch backend).
"""

from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
from scipy.special import erf, expit

from aipe_magnetics.common.errors import (
    DataIntegrityError,
    DataNotFoundError,
    InvalidInputError,
    NumericalError,
    UnsupportedOperationError,
)
from aipe_magnetics.common.jsonio import loads as loads_json
from aipe_magnetics.models.datadriven.architectures import (
    ACTIVATIONS,
    ArchitectureSpec,
    LSTMHyperparameters,
    MLPHyperparameters,
    NeuralODEHyperparameters,
    stack_inputs,
    unstack_outputs,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from aipe_magnetics.models.datadriven.adapter import ModelPackage

AIPE_JSON_FORMAT_VERSION = "0.2"
RESERVED_AIPE_JSON_KEYS: frozenset[str] = frozenset({"format", "format_version", "family", "architecture", "layers", "lstm", "head"})
_SQRT2 = math.sqrt(2.0)


# --------------------------------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------------------------------
def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + erf(x / _SQRT2))


def _identity(x: np.ndarray) -> np.ndarray:
    return x


ACTIVATION_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "relu": lambda x: np.maximum(x, 0.0),
    "tanh": np.tanh,
    "gelu": _gelu,
    "sigmoid": expit,
    "identity": _identity,
}


def activation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    try:
        return ACTIVATION_FUNCTIONS[name]
    except KeyError:
        raise InvalidInputError(f"unknown activation {name!r}; supported: {', '.join(ACTIVATIONS)}",
                                details={"activation": name}) from None


# --------------------------------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DenseLayer:
    """``y = act(x @ W.T + b)`` with ``weight`` of shape ``(out, in)``."""

    weight: np.ndarray
    bias: np.ndarray
    activation: str = "identity"

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64)
        if weight.ndim != 2 or bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise DataIntegrityError(f"dense layer shapes inconsistent: W {weight.shape}, b {bias.shape}")
        if not (np.all(np.isfinite(weight)) and np.all(np.isfinite(bias))):
            raise DataIntegrityError("dense layer contains non-finite parameters")
        activation(self.activation)  # validates the name
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)

    @property
    def in_features(self) -> int:
        return int(self.weight.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.weight.shape[0])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return activation(self.activation)(x @ self.weight.T + self.bias)


@dataclass(frozen=True)
class LSTMLayer:
    """One LSTM layer in PyTorch layout (gates ``i, f, g, o`` stacked along axis 0)."""

    weight_ih: np.ndarray
    weight_hh: np.ndarray
    bias_ih: np.ndarray
    bias_hh: np.ndarray

    def __post_init__(self) -> None:
        arrays = {name: np.asarray(getattr(self, name), dtype=np.float64)
                  for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")}
        w_ih, w_hh, b_ih, b_hh = arrays["weight_ih"], arrays["weight_hh"], arrays["bias_ih"], arrays["bias_hh"]
        if w_ih.ndim != 2 or w_hh.ndim != 2 or b_ih.ndim != 1 or b_hh.ndim != 1:
            raise DataIntegrityError("LSTM parameters must be W_ih (4H, in), W_hh (4H, H), b_ih (4H,), b_hh (4H,)")
        four_h = w_ih.shape[0]
        if four_h % 4 != 0 or w_hh.shape != (four_h, four_h // 4) or b_ih.shape != (four_h,) or b_hh.shape != (four_h,):
            raise DataIntegrityError(f"LSTM parameter shapes inconsistent: W_ih {w_ih.shape}, W_hh {w_hh.shape}, "
                                     f"b_ih {b_ih.shape}, b_hh {b_hh.shape}")
        if not all(np.all(np.isfinite(a)) for a in arrays.values()):
            raise DataIntegrityError("LSTM layer contains non-finite parameters")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    @property
    def hidden_size(self) -> int:
        return int(self.weight_hh.shape[1])

    @property
    def input_size(self) -> int:
        return int(self.weight_ih.shape[1])


# --------------------------------------------------------------------------------------------------
# Forward passes
# --------------------------------------------------------------------------------------------------
def mlp_forward(layers: Sequence[DenseLayer], x: np.ndarray) -> np.ndarray:
    """Apply a dense stack to ``x`` of shape ``(..., in)``; returns ``(..., out)`` in float64."""
    h = np.asarray(x, dtype=np.float64)
    if not layers:
        raise InvalidInputError("an MLP needs at least one dense layer")
    if h.shape[-1] != layers[0].in_features:
        raise InvalidInputError(f"input has {h.shape[-1]} features, the network expects {layers[0].in_features}")
    for layer in layers:
        h = layer(h)
    return h


def lstm_step(layer: LSTMLayer, x_t: np.ndarray, h: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One time step for a batch: ``x_t (N, in)``, ``h, c (N, H)`` -> new ``(h, c)``."""
    hidden = layer.hidden_size
    gates = x_t @ layer.weight_ih.T + layer.bias_ih + h @ layer.weight_hh.T + layer.bias_hh
    i = expit(gates[:, 0 * hidden:1 * hidden])
    f = expit(gates[:, 1 * hidden:2 * hidden])
    g = np.tanh(gates[:, 2 * hidden:3 * hidden])
    o = expit(gates[:, 3 * hidden:4 * hidden])
    c_new = f * c + i * g
    h_new = o * np.tanh(c_new)
    return h_new, c_new


def lstm_forward(layers: Sequence[LSTMLayer], x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Run stacked LSTM layers over ``x (N, L, in)``.

    Returns ``(outputs (N, L, H_last), [h_T per layer], [c_T per layer])``. The input projection of
    every layer is computed for all time steps at once; only the recurrence is a Python loop.
    """
    seq = np.asarray(x, dtype=np.float64)
    if seq.ndim != 3:
        raise InvalidInputError(f"LSTM input must have shape (N, L, in), got {seq.shape}")
    if not layers:
        raise InvalidInputError("an LSTM needs at least one layer")
    if seq.shape[-1] != layers[0].input_size:
        raise InvalidInputError(f"input has {seq.shape[-1]} features, the LSTM expects {layers[0].input_size}")
    n_batch, length, _ = seq.shape
    h_final: list[np.ndarray] = []
    c_final: list[np.ndarray] = []
    layer_input = seq
    for layer in layers:
        hidden = layer.hidden_size
        projected = layer_input @ layer.weight_ih.T + layer.bias_ih + layer.bias_hh  # (N, L, 4H)
        h = np.zeros((n_batch, hidden))
        c = np.zeros((n_batch, hidden))
        outputs = np.empty((n_batch, length, hidden))
        w_hh_t = layer.weight_hh.T
        for t in range(length):
            gates = projected[:, t, :] + h @ w_hh_t
            i = expit(gates[:, 0 * hidden:1 * hidden])
            f = expit(gates[:, 1 * hidden:2 * hidden])
            g = np.tanh(gates[:, 2 * hidden:3 * hidden])
            o = expit(gates[:, 3 * hidden:4 * hidden])
            c = f * c + i * g
            h = o * np.tanh(c)
            outputs[:, t, :] = h
        h_final.append(h)
        c_final.append(c)
        layer_input = outputs
    return layer_input, h_final, c_final


# --------------------------------------------------------------------------------------------------
# Networks built from (descriptor, parameter dictionary)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MLPNetwork:
    layers: tuple[DenseLayer, ...]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return mlp_forward(self.layers, x)


@dataclass(frozen=True)
class LSTMNetwork:
    lstm_layers: tuple[LSTMLayer, ...]
    head: tuple[DenseLayer, ...]
    output_mode: str = "last"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        outputs, h_final, _ = lstm_forward(self.lstm_layers, x)
        features = outputs if self.output_mode == "sequence" else h_final[-1]
        return mlp_forward(self.head, features)


Network = MLPNetwork | LSTMNetwork


def validate_parameters(spec: ArchitectureSpec, params: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Check names, shapes and finiteness of ``params`` against the descriptor; returns float64 copies."""
    expected = spec.expected_parameter_shapes()
    missing = sorted(set(expected) - set(params))
    unexpected = sorted(set(params) - set(expected))
    if missing or unexpected:
        raise DataIntegrityError(
            f"parameter names do not match architecture {spec.name!r}: missing {missing}, unexpected {unexpected}",
            details={"missing": missing, "unexpected": unexpected})
    checked: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        try:
            array = np.asarray(params[name], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise DataIntegrityError(f"parameter {name!r} is not numeric: {error}", details={"parameter": name}) from None
        if array.shape != shape:
            raise DataIntegrityError(f"parameter {name!r} has shape {array.shape}, architecture expects {shape}",
                                     details={"parameter": name, "actual": list(array.shape), "expected": list(shape)})
        if not np.all(np.isfinite(array)):
            raise DataIntegrityError(f"parameter {name!r} contains non-finite values", details={"parameter": name})
        checked[name] = array
    return checked


def build_network(spec: ArchitectureSpec, params: Mapping[str, np.ndarray],
                  activation_tags: Mapping[str, str] | None = None) -> Network:
    """Instantiate the numpy network for ``spec`` from a validated parameter dictionary.

    ``activation_tags`` (``{"layers.0": "tanh", ...}`` / ``{"head.0": ...}``) are the redundant tags
    carried by ``aipe-json`` files; they must agree with the descriptor.
    """
    if spec.family == "transformer":
        raise UnsupportedOperationError(
            "transformer inference is not implemented in the numpy backend; export the model to TorchScript "
            "(format 'torchscript') and use the optional torch backend (install aipe-magnetics[torch])",
            details={"family": "transformer", "backend": "numpy"})
    if spec.is_external:
        raise UnsupportedOperationError("external (opaque) architectures cannot be executed by the numpy backend",
                                        details={"kind": spec.kind, "backend": "numpy"})
    checked = validate_parameters(spec, params)
    hp = spec.typed_hyperparameters()
    tags = dict(activation_tags or {})

    def dense_stack(prefix: str) -> tuple[DenseLayer, ...]:
        layers = []
        for i, (_, _, act) in enumerate(spec.dense_layer_sizes()):
            tag = tags.get(f"{prefix}.{i}")
            if tag is not None and tag != act:
                raise DataIntegrityError(
                    f"weights file tags {prefix}.{i} with activation {tag!r} but the architecture declares {act!r}",
                    details={"layer": f"{prefix}.{i}", "file": tag, "architecture": act})
            layers.append(DenseLayer(checked[f"{prefix}.{i}.weight"], checked[f"{prefix}.{i}.bias"], act))
        return tuple(layers)

    if isinstance(hp, (MLPHyperparameters, NeuralODEHyperparameters)):
        return MLPNetwork(dense_stack("layers"))
    if isinstance(hp, LSTMHyperparameters):
        lstm_layers = tuple(
            LSTMLayer(checked[f"lstm.weight_ih_l{k}"], checked[f"lstm.weight_hh_l{k}"],
                      checked[f"lstm.bias_ih_l{k}"], checked[f"lstm.bias_hh_l{k}"])
            for k in range(hp.num_layers))
        return LSTMNetwork(lstm_layers, dense_stack("head"), hp.output_mode)
    raise UnsupportedOperationError(f"numpy backend cannot build family {spec.family!r}")  # pragma: no cover


# --------------------------------------------------------------------------------------------------
# Loaders: file -> parameter dictionary (+ activation tags)
# --------------------------------------------------------------------------------------------------
def _as_array(value: Any, where: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DataIntegrityError(f"{where}: not a numeric array ({error})", details={"where": where}) from None
    return array


def _read_source(source: str | Path | bytes) -> tuple[bytes, str]:
    """Return ``(file bytes, label)`` for a path or an in-memory buffer (the verified weights bytes)."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source), "<weights bytes>"
    path = Path(source)
    try:
        return path.read_bytes(), path.name
    except OSError as error:
        raise DataNotFoundError(f"weights file not readable: {path} ({error})", details={"path": str(path)}) from None


def load_aipe_json(source: str | Path | bytes) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    """Read an ``aipe-json`` weights file (path or bytes) -> ``(parameters, activation_tags, header)``."""
    raw, label = _read_source(source)
    try:
        data = loads_json(raw.decode("utf-8"))
    except ValueError as error:  # duplicate keys, NaN/Inf, malformed JSON, invalid UTF-8
        raise DataIntegrityError(f"weights file {label} is not valid strict JSON: {error}") from None
    if not isinstance(data, dict) or data.get("format") != "aipe-json":
        raise DataIntegrityError(f"weights file {label} is not an aipe-json weights file")
    if str(data.get("format_version")) != AIPE_JSON_FORMAT_VERSION:
        raise DataIntegrityError(f"unsupported aipe-json format_version {data.get('format_version')!r}; "
                                 f"expected {AIPE_JSON_FORMAT_VERSION!r}")
    params: dict[str, np.ndarray] = {}
    tags: dict[str, str] = {}

    def read_dense(entries: Any, prefix: str) -> None:
        if not isinstance(entries, list):
            raise DataIntegrityError(f"aipe-json: {prefix!r} must be a list of dense layers")
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict) or "W" not in entry or "b" not in entry:
                raise DataIntegrityError(f"aipe-json: dense layer {prefix}.{i} needs 'W' and 'b'")
            params[f"{prefix}.{i}.weight"] = _as_array(entry["W"], f"{prefix}.{i}.W")
            params[f"{prefix}.{i}.bias"] = _as_array(entry["b"], f"{prefix}.{i}.b")
            if "activation" in entry:
                if entry["activation"] not in ACTIVATIONS:
                    raise DataIntegrityError(f"aipe-json: unknown activation {entry['activation']!r} in {prefix}.{i}")
                tags[f"{prefix}.{i}"] = str(entry["activation"])

    family = data.get("family")
    if family in ("ann", "neural_ode"):
        read_dense(data.get("layers"), "layers")
    elif family == "lstm":
        cells = data.get("lstm")
        if not isinstance(cells, list) or not cells:
            raise DataIntegrityError("aipe-json: 'lstm' must be a non-empty list of layers")
        for k, cell in enumerate(cells):
            if not isinstance(cell, dict) or any(key not in cell for key in ("W_ih", "W_hh", "b_ih", "b_hh")):
                raise DataIntegrityError(f"aipe-json: lstm layer {k} needs W_ih, W_hh, b_ih, b_hh")
            for key, name in (("W_ih", "weight_ih"), ("W_hh", "weight_hh"), ("b_ih", "bias_ih"), ("b_hh", "bias_hh")):
                params[f"lstm.{name}_l{k}"] = _as_array(cell[key], f"lstm.{k}.{key}")
        read_dense(data.get("head"), "head")
    else:
        raise DataIntegrityError(f"aipe-json: unsupported or missing family {family!r}")
    header = {key: value for key, value in data.items() if key not in ("layers", "lstm", "head")}
    return params, tags, header


def check_aipe_json_header(header: Mapping[str, Any], spec: ArchitectureSpec) -> None:
    """Cross-check the redundant ``family`` / ``architecture`` header fields against the descriptor."""
    family = header.get("family")
    if family != spec.family:
        raise DataIntegrityError(
            f"weights file declares family {family!r} but the package architecture is family {spec.family!r}",
            details={"file": family, "architecture": spec.family})
    name = header.get("architecture")
    if name is not None and str(name) != spec.name:
        raise DataIntegrityError(
            f"weights file was written for architecture {name!r} but the package declares {spec.name!r}",
            details={"file": name, "architecture": spec.name})


_NPZ_ERRORS = (OSError, ValueError, EOFError, TypeError, zipfile.BadZipFile)


def load_npz(source: str | Path | bytes) -> dict[str, np.ndarray]:
    """Read an ``npz`` weights file (path or bytes; PyTorch state-dict names) with ``allow_pickle=False``."""
    raw, label = _read_source(source)
    try:
        archive = np.load(io.BytesIO(raw), allow_pickle=False)
    except _NPZ_ERRORS as error:  # empty/truncated file, bad zip, pickled payload, ...
        raise DataIntegrityError(f"weights file {label} is not a readable npz archive: {error}") from None
    if not isinstance(archive, np.lib.npyio.NpzFile):  # e.g. a single-array .npy file
        raise DataIntegrityError(f"weights file {label} is not an npz archive (np.load returned {type(archive).__name__})")
    try:
        with archive:
            return {name: np.asarray(archive[name], dtype=np.float64) for name in archive.files}
    except _NPZ_ERRORS as error:  # object/string arrays, truncated members
        raise DataIntegrityError(f"weights file {label} holds non-numeric or unreadable arrays: {error}") from None


def load_parameters(source: str | Path | bytes, fmt: str) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Dispatch on the weights format; returns ``(parameters, activation_tags)``."""
    if fmt == "aipe-json":
        params, tags, _ = load_aipe_json(source)
        return params, tags
    if fmt == "npz":
        return load_npz(source), {}
    raise UnsupportedOperationError(f"the numpy backend reads 'aipe-json' and 'npz' weights, not {fmt!r}",
                                    details={"format": fmt, "backend": "numpy"})


def write_aipe_json(path: str | Path, spec: ArchitectureSpec, params: Mapping[str, np.ndarray], *,
                    extra: Mapping[str, Any] | None = None, overwrite: bool = False) -> Path:
    """Serialise a parameter dictionary for ``spec`` in the ``aipe-json`` format (sorted keys, LF, UTF-8).

    ``extra`` adds informational header fields; the keys reserved by the format
    (:data:`RESERVED_AIPE_JSON_KEYS`) cannot be overridden.
    """
    from aipe_magnetics.common.jsonio import dump

    clash = sorted(set(extra or {}) & RESERVED_AIPE_JSON_KEYS)
    if clash:
        raise InvalidInputError(f"extra header keys {clash} are reserved by the aipe-json format",
                                details={"reserved": sorted(RESERVED_AIPE_JSON_KEYS), "clash": clash})
    checked = validate_parameters(spec, params)
    hp = spec.typed_hyperparameters()
    document: dict[str, Any] = {"format": "aipe-json", "format_version": AIPE_JSON_FORMAT_VERSION,
                                "family": spec.family, "architecture": spec.name}
    document.update(dict(extra or {}))

    def dense_entries(prefix: str) -> list[dict[str, Any]]:
        return [{"W": checked[f"{prefix}.{i}.weight"].tolist(), "b": checked[f"{prefix}.{i}.bias"].tolist(),
                 "activation": act} for i, (_, _, act) in enumerate(spec.dense_layer_sizes())]

    if isinstance(hp, (MLPHyperparameters, NeuralODEHyperparameters)):
        document["layers"] = dense_entries("layers")
    elif isinstance(hp, LSTMHyperparameters):
        document["lstm"] = [{"W_ih": checked[f"lstm.weight_ih_l{k}"].tolist(), "W_hh": checked[f"lstm.weight_hh_l{k}"].tolist(),
                             "b_ih": checked[f"lstm.bias_ih_l{k}"].tolist(), "b_hh": checked[f"lstm.bias_hh_l{k}"].tolist()}
                            for k in range(hp.num_layers)]
        document["head"] = dense_entries("head")
    else:  # pragma: no cover - validate_parameters already rejects opaque families
        raise UnsupportedOperationError(f"cannot serialise family {spec.family!r} as aipe-json")
    return dump(document, path, overwrite=overwrite)


# --------------------------------------------------------------------------------------------------
# The backend object
# --------------------------------------------------------------------------------------------------
class NumpyBackend:
    """Backend for ``aipe-json`` / ``npz`` weights of the ``ann``, ``lstm`` and ``neural_ode`` families."""

    name = "numpy"
    dtype = "float64"  # computational precision, independent of InferenceConfig.dtype
    formats: tuple[str, ...] = ("aipe-json", "npz")
    families: tuple[str, ...] = ("ann", "lstm", "neural_ode")

    def __init__(self) -> None:
        self.network: Network | None = None
        self.parameter_count: int = 0

    def load(self, package: "ModelPackage") -> None:
        spec = package.architecture
        if spec.family not in self.families:
            raise UnsupportedOperationError(
                f"the numpy backend does not execute family {spec.family!r} "
                "(transformer: use a TorchScript export with the torch backend)",
                details={"family": spec.family, "backend": self.name})
        if package.weights.format not in self.formats:
            raise UnsupportedOperationError(f"the numpy backend reads {self.formats}, not {package.weights.format!r}",
                                            details={"format": package.weights.format})
        data, _ = package.read_weights()  # digest/size verified on this very buffer
        if package.weights.format == "aipe-json":
            params, tags, header = load_aipe_json(data)
            check_aipe_json_header(header, spec)
        else:
            params, tags = load_npz(data), {}
        self.network = build_network(spec, params, tags)
        self.parameter_count = spec.parameter_count()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Raw tensor forward pass (normalised space); used by the Neural-ODE integrator."""
        if self.network is None:
            raise InvalidInputError("backend not loaded; call load(package) first")
        return self.network(np.asarray(x, dtype=np.float64))

    def run(self, package: "ModelPackage", x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Map normalised per-feature inputs to normalised per-feature outputs (float64)."""
        spec = package.architecture
        tensor = stack_inputs(spec, x)  # rejects N = 0 and L = 0
        if tensor.shape[0] == 0:  # pragma: no cover - defensive, stack_inputs already refuses empty batches
            raise InvalidInputError("empty batch (N = 0): nothing to evaluate", details={"batch_size": 0})
        chunk = max(1, int(package.inference.batch_size))
        pieces = [self.forward(tensor[start:start + chunk]) for start in range(0, tensor.shape[0], chunk)]
        y = np.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]
        if not np.all(np.isfinite(y)):
            raise NumericalError("network output is not finite", details={"backend": self.name})
        return unstack_outputs(spec, y)


__all__ = [
    "ACTIVATION_FUNCTIONS", "AIPE_JSON_FORMAT_VERSION", "RESERVED_AIPE_JSON_KEYS", "DenseLayer", "LSTMLayer",
    "LSTMNetwork", "MLPNetwork", "Network", "NumpyBackend", "activation", "build_network", "check_aipe_json_header",
    "load_aipe_json", "load_npz", "load_parameters", "lstm_forward", "lstm_step", "mlp_forward", "validate_parameters",
    "write_aipe_json",
]
