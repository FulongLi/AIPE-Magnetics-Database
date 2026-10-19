"""Model ARCHITECTURE descriptors for the data-driven model family.

This module is the first of the three strictly separated layers of a data-driven model package:

1. **architecture** (this module) - *what* the network is: family, layer sizes, activations,
   the ordered input/output feature signature with physical units, and which features are
   sequences. It carries **no trained numbers**.
2. **trained parameters** (:class:`~aipe_magnetics.models.datadriven.adapter.WeightsBundle`) -
   a file in a declared format with a SHA-256 digest and a :class:`SourceReference`.
3. **inference configuration** (:class:`~aipe_magnetics.models.datadriven.adapter.InferenceConfig`) -
   normalisation statistics, dtype, batch size, device and the mapping from input features to
   validity-domain conditions.

Supported families and their hyper-parameter schemas
-----------------------------------------------------

``ann`` (``kind='mlp'``, default)
    Multi-layer perceptron. ``hidden_sizes=[h1, h2, ...]`` (may be empty -> affine model),
    ``activation`` for hidden layers, ``output_activation`` for the last layer. Dense layer
    ``i`` computes ``y = act(W_i x + b_i)`` with ``W_i`` of shape ``(out, in)`` (PyTorch
    ``nn.Linear`` convention). Parameter names: ``layers.{i}.weight``, ``layers.{i}.bias``.

``lstm`` (``kind='lstm'``, default)
    Stacked LSTM with ``num_layers`` layers of ``hidden_size`` units followed by a dense head
    (``head_hidden_sizes``, ``head_activation``, ``head_output_activation``). Gate layout and
    parameter names follow ``torch.nn.LSTM``: ``lstm.weight_ih_l{k}`` ``(4H, in)``,
    ``lstm.weight_hh_l{k}`` ``(4H, H)``, ``lstm.bias_ih_l{k}`` and ``lstm.bias_hh_l{k}`` ``(4H,)``
    with gates stacked in the order ``i, f, g, o``. ``output_mode='last'`` applies the head to the
    final hidden state (sequence-to-one), ``'sequence'`` to every step (sequence-to-sequence).
    Head parameters: ``head.{i}.weight``, ``head.{i}.bias``.

``transformer`` (``kind='transformer'``, default)
    Descriptor only in v0.2 (``d_model``, ``nhead``, ``num_layers``, ``dim_feedforward``,
    ``dropout``, ``pooling``). Inference is available solely through a TorchScript export and the
    optional torch backend; the numpy backend rejects it explicitly.

``neural_ode`` (``kind='mlp_vector_field'``, default)
    Vector field ``dy/dt = f_theta(y, u)`` given by an MLP whose input is the concatenation of the
    ``state_size`` state variables and ``input_size`` exogenous inputs, and whose output has
    ``state_size`` components. ``input_features`` lists the state names followed by the exogenous
    input names; ``output_features`` lists the state names. Parameter names as for ``ann``.

Any family may use ``kind='external'`` for an opaque architecture owned by an external package
(e.g. the MagNet-Hub models); no parameter shapes can be derived for it.

Hyper-parameter values are validated **strictly** (pydantic strict mode): ``8.0``, ``"8"`` and
``True`` are not layer sizes, and ``[0]`` is rejected; tuples are accepted and stored as lists. The
validated mapping with every default made explicit is written back into ``hyperparameters``, so the
descriptor, :meth:`ArchitectureSpec.typed_hyperparameters` and the JSON round trip always agree.

Feature signature conventions
-----------------------------

* ``input_features`` / ``output_features`` are ordered, unique names. ``input_units`` and
  ``output_units`` give the physical unit of each feature (default ``"1"``, dimensionless); units
  must exist in :mod:`aipe_magnetics.common.units`.
* ``sequence_features`` names the features that vary along the sequence (time) axis. A scalar
  feature has batch shape ``(N,)``, a sequence feature ``(N, L)``; ``sequence_length`` fixes ``L``
  when the architecture requires it.
* :func:`stack_inputs` builds the backend tensor: ``(N, n_in)`` for feed-forward families and
  ``(N, L, n_in)`` for sequence families, where scalar features are repeated along the sequence
  axis (the usual way scalar conditions such as frequency or temperature condition a recurrent
  model). :func:`unstack_outputs` performs the inverse mapping for the outputs. An empty batch
  (``N = 0``) or an empty sequence (``L = 0``) is rejected with :class:`InvalidInputError`: a
  recurrent model applied to zero time steps would only report its initial state.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aipe_magnetics.common.errors import InvalidInputError, UnsupportedOperationError
from aipe_magnetics.common.units import units

Family = Literal["ann", "lstm", "transformer", "neural_ode"]
Activation = Literal["relu", "tanh", "gelu", "sigmoid", "identity"]
ACTIVATIONS: tuple[str, ...] = ("relu", "tanh", "gelu", "sigmoid", "identity")

FAMILIES: tuple[str, ...] = ("ann", "lstm", "transformer", "neural_ode")
DEFAULT_KIND: dict[str, str] = {
    "ann": "mlp",
    "lstm": "lstm",
    "transformer": "transformer",
    "neural_ode": "mlp_vector_field",
}
EXTERNAL_KIND = "external"


# --------------------------------------------------------------------------------------------------
# Family hyper-parameter schemas (no trained numbers)
# --------------------------------------------------------------------------------------------------
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)


class MLPHyperparameters(BaseModel):
    """Hyper-parameters of a multi-layer perceptron (family ``ann``)."""

    model_config = _STRICT
    kind: Literal["mlp"] = "mlp"
    hidden_sizes: list[int] = Field(default_factory=list)
    activation: Activation = "tanh"
    output_activation: Activation = "identity"

    @model_validator(mode="after")
    def _sizes(self) -> "MLPHyperparameters":
        if any((not isinstance(h, int)) or isinstance(h, bool) or h < 1 for h in self.hidden_sizes):
            raise ValueError("hidden_sizes must be positive integers")
        return self


class LSTMHyperparameters(BaseModel):
    """Hyper-parameters of a stacked LSTM with a dense head (family ``lstm``)."""

    model_config = _STRICT
    kind: Literal["lstm"] = "lstm"
    hidden_size: int = Field(ge=1)
    num_layers: int = Field(default=1, ge=1)
    output_mode: Literal["last", "sequence"] = "last"
    head_hidden_sizes: list[int] = Field(default_factory=list)
    head_activation: Activation = "tanh"
    head_output_activation: Activation = "identity"

    @model_validator(mode="after")
    def _sizes(self) -> "LSTMHyperparameters":
        if any((not isinstance(h, int)) or isinstance(h, bool) or h < 1 for h in self.head_hidden_sizes):
            raise ValueError("head_hidden_sizes must be positive integers")
        return self


class TransformerHyperparameters(BaseModel):
    """Descriptor of an encoder-only transformer (family ``transformer``); inference via TorchScript only."""

    model_config = _STRICT
    kind: Literal["transformer"] = "transformer"
    d_model: int = Field(ge=1)
    nhead: int = Field(ge=1)
    num_layers: int = Field(ge=1)
    dim_feedforward: int = Field(ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    pooling: Literal["last", "mean", "cls"] = "mean"

    @model_validator(mode="after")
    def _heads(self) -> "TransformerHyperparameters":
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        return self


class NeuralODEHyperparameters(BaseModel):
    """Hyper-parameters of an MLP vector field ``dy/dt = f_theta(y, u)`` (family ``neural_ode``)."""

    model_config = _STRICT
    kind: Literal["mlp_vector_field"] = "mlp_vector_field"
    state_size: int = Field(ge=1)
    input_size: int = Field(default=0, ge=0)
    hidden_sizes: list[int] = Field(default_factory=list)
    activation: Activation = "tanh"
    output_activation: Activation = "identity"

    @model_validator(mode="after")
    def _sizes(self) -> "NeuralODEHyperparameters":
        if any((not isinstance(h, int)) or isinstance(h, bool) or h < 1 for h in self.hidden_sizes):
            raise ValueError("hidden_sizes must be positive integers")
        return self


class ExternalHyperparameters(BaseModel):
    """Opaque architecture owned by an external package (``kind='external'``)."""

    model_config = ConfigDict(extra="allow", frozen=True, strict=True)
    kind: Literal["external"] = "external"
    package: str = Field(min_length=1, description="name of the external package that owns the architecture")


TypedHyperparameters = (MLPHyperparameters | LSTMHyperparameters | TransformerHyperparameters
                        | NeuralODEHyperparameters | ExternalHyperparameters)

_SCHEMAS: dict[tuple[str, str], type[BaseModel]] = {
    ("ann", "mlp"): MLPHyperparameters,
    ("lstm", "lstm"): LSTMHyperparameters,
    ("transformer", "transformer"): TransformerHyperparameters,
    ("neural_ode", "mlp_vector_field"): NeuralODEHyperparameters,
}


def parse_hyperparameters(family: str, hyperparameters: Mapping[str, Any]) -> TypedHyperparameters:
    """Validate a raw hyper-parameter mapping against the schema of ``family`` (strict types).

    Values are not coerced: ``8.0``, ``"8"`` and ``True`` are rejected as layer sizes. Tuples are
    the only accepted alternative spelling of a list (they are stored as lists).
    """
    if not isinstance(hyperparameters, Mapping):
        raise ValueError(f"hyperparameters must be a mapping, got {type(hyperparameters).__name__}")
    data = {key: list(value) if isinstance(value, tuple) else value for key, value in hyperparameters.items()}
    kind = data.get("kind", DEFAULT_KIND.get(family))
    if kind == EXTERNAL_KIND:
        return ExternalHyperparameters.model_validate(data)
    schema = _SCHEMAS.get((family, str(kind)))
    if schema is None:
        raise ValueError(f"family {family!r} does not support hyperparameters.kind={kind!r}; "
                         f"use {DEFAULT_KIND[family]!r} or 'external'")
    data.setdefault("kind", kind)
    return schema.model_validate(data)


# --------------------------------------------------------------------------------------------------
# The architecture descriptor
# --------------------------------------------------------------------------------------------------
class ArchitectureSpec(BaseModel):
    """Family, hyper-parameters and feature signature of a data-driven model (no trained numbers)."""

    model_config = ConfigDict(extra="forbid")

    family: Family
    name: str = Field(min_length=1)
    hyperparameters: dict[str, Any] = Field(default_factory=dict, description="family-specific layer/hyper-parameter dict")
    input_features: list[str] = Field(min_length=1)
    output_features: list[str] = Field(min_length=1)
    sequence_features: list[str] = Field(default_factory=list, description="features that vary along the sequence axis")
    sequence_length: int | None = Field(default=None, ge=1, description="required sequence length L, if fixed")
    input_units: dict[str, str] = Field(default_factory=dict, description="feature -> unit (default '1')")
    output_units: dict[str, str] = Field(default_factory=dict, description="feature -> unit (default '1')")
    description: str = ""

    # -- validation ---------------------------------------------------------------------------------
    @model_validator(mode="after")
    def _consistent(self) -> "ArchitectureSpec":
        for label, names in (("input_features", self.input_features), ("output_features", self.output_features)):
            if len(set(names)) != len(names):
                raise ValueError(f"{label} must be unique")
            if any((not isinstance(n, str)) or not n for n in names):
                raise ValueError(f"{label} must be non-empty strings")
        overlap = set(self.input_features) & set(self.output_features)
        if overlap and self.family != "neural_ode":
            raise ValueError(f"features cannot be both input and output: {sorted(overlap)}")
        if self.family == "neural_ode":
            for name, unit in self.output_units.items():
                if name in self.input_units and self.input_units[name] != unit:
                    raise ValueError(f"neural_ode: state {name!r} has different input and output units")
        known = set(self.input_features) | set(self.output_features)
        unknown = [n for n in self.sequence_features if n not in known]
        if unknown:
            raise ValueError(f"sequence_features not in input/output features: {unknown}")
        if len(set(self.sequence_features)) != len(self.sequence_features):
            raise ValueError("sequence_features must be unique")
        for label, mapping, names in (("input_units", self.input_units, self.input_features),
                                      ("output_units", self.output_units, self.output_features)):
            bad = [n for n in mapping if n not in names]
            if bad:
                raise ValueError(f"{label} refers to unknown features: {bad}")
            for name, unit in mapping.items():
                units.get(unit)  # raises UnitError (a ValueError) for unknown units
        hp = parse_hyperparameters(self.family, self.hyperparameters)  # raises ValueError on schema violations
        self._validate_family(hp)
        # store the validated, fully explicit form so that `hyperparameters`, `typed_hyperparameters()`
        # and the JSON round trip agree (no lax-coerced copy diverging from the raw mapping)
        self.hyperparameters = hp.model_dump()
        return self

    def _validate_family(self, hp: TypedHyperparameters) -> None:
        seq_inputs = self.sequence_input_features
        seq_outputs = self.sequence_output_features
        if isinstance(hp, ExternalHyperparameters):
            return
        if isinstance(hp, MLPHyperparameters):
            if self.sequence_features or self.sequence_length is not None:
                raise ValueError("family 'ann' (mlp) takes scalar features only; use 'lstm' for sequences")
            return
        if isinstance(hp, NeuralODEHyperparameters):
            if self.sequence_features or self.sequence_length is not None:
                raise ValueError("family 'neural_ode' describes the vector field; sequences arise from integration")
            if len(self.input_features) != hp.state_size + hp.input_size:
                raise ValueError("neural_ode: len(input_features) must equal state_size + input_size "
                                 "(state names followed by exogenous input names)")
            if self.output_features != self.input_features[:hp.state_size]:
                raise ValueError("neural_ode: output_features must equal the first state_size input_features "
                                 "(the state names; the model outputs state trajectories)")
            return
        if isinstance(hp, (LSTMHyperparameters, TransformerHyperparameters)):
            if not seq_inputs:
                raise ValueError(f"family {self.family!r} needs at least one sequence input feature")
            if isinstance(hp, LSTMHyperparameters):
                if hp.output_mode == "sequence" and set(seq_outputs) != set(self.output_features):
                    raise ValueError("lstm output_mode='sequence': every output feature must be a sequence feature")
                if hp.output_mode == "last" and seq_outputs:
                    raise ValueError("lstm output_mode='last': output features are per-sample scalars, not sequences")
            return

    # -- introspection ------------------------------------------------------------------------------
    def typed_hyperparameters(self) -> TypedHyperparameters:
        return parse_hyperparameters(self.family, self.hyperparameters)

    @property
    def kind(self) -> str:
        return str(self.hyperparameters.get("kind", DEFAULT_KIND[self.family]))

    @property
    def is_external(self) -> bool:
        return self.kind == EXTERNAL_KIND

    @property
    def sequence_input_features(self) -> list[str]:
        return [n for n in self.input_features if n in self.sequence_features]

    @property
    def sequence_output_features(self) -> list[str]:
        return [n for n in self.output_features if n in self.sequence_features]

    @property
    def is_sequence_model(self) -> bool:
        return bool(self.sequence_input_features)

    def unit_of(self, feature: str) -> str:
        if feature in self.input_features:
            return self.input_units.get(feature, "1")
        if feature in self.output_features:
            return self.output_units.get(feature, "1")
        raise InvalidInputError(f"unknown feature {feature!r}", details={"feature": feature})

    def dense_layer_sizes(self) -> list[tuple[int, int, str]]:
        """``[(in, out, activation), ...]`` of the dense stack (ann / neural_ode / lstm head)."""
        hp = self.typed_hyperparameters()
        if isinstance(hp, MLPHyperparameters):
            return _dense_sizes(len(self.input_features), hp.hidden_sizes, len(self.output_features),
                                hp.activation, hp.output_activation)
        if isinstance(hp, NeuralODEHyperparameters):
            return _dense_sizes(hp.state_size + hp.input_size, hp.hidden_sizes, hp.state_size,
                                hp.activation, hp.output_activation)
        if isinstance(hp, LSTMHyperparameters):
            return _dense_sizes(hp.hidden_size, hp.head_hidden_sizes, len(self.output_features),
                                hp.head_activation, hp.head_output_activation)
        raise UnsupportedOperationError(f"no dense layer stack is defined for family {self.family!r} kind {self.kind!r}",
                                        details={"family": self.family, "kind": self.kind})

    def expected_parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        """Parameter name -> shape (PyTorch naming), derived from the descriptor alone."""
        hp = self.typed_hyperparameters()
        shapes: dict[str, tuple[int, ...]] = {}
        if isinstance(hp, (MLPHyperparameters, NeuralODEHyperparameters)):
            for i, (n_in, n_out, _) in enumerate(self.dense_layer_sizes()):
                shapes[f"layers.{i}.weight"] = (n_out, n_in)
                shapes[f"layers.{i}.bias"] = (n_out,)
            return shapes
        if isinstance(hp, LSTMHyperparameters):
            n_in = len(self.input_features)
            for k in range(hp.num_layers):
                layer_in = n_in if k == 0 else hp.hidden_size
                shapes[f"lstm.weight_ih_l{k}"] = (4 * hp.hidden_size, layer_in)
                shapes[f"lstm.weight_hh_l{k}"] = (4 * hp.hidden_size, hp.hidden_size)
                shapes[f"lstm.bias_ih_l{k}"] = (4 * hp.hidden_size,)
                shapes[f"lstm.bias_hh_l{k}"] = (4 * hp.hidden_size,)
            for i, (d_in, d_out, _) in enumerate(self.dense_layer_sizes()):
                shapes[f"head.{i}.weight"] = (d_out, d_in)
                shapes[f"head.{i}.bias"] = (d_out,)
            return shapes
        raise UnsupportedOperationError(
            f"parameter shapes cannot be derived for family {self.family!r} kind {self.kind!r} "
            "(transformer and external architectures are opaque descriptors in v0.2)",
            details={"family": self.family, "kind": self.kind})

    def parameter_count(self) -> int:
        return int(sum(int(np.prod(shape)) for shape in self.expected_parameter_shapes().values()))


def _dense_sizes(n_in: int, hidden: list[int], n_out: int, activation: str, output_activation: str
                 ) -> list[tuple[int, int, str]]:
    sizes = [n_in, *hidden, n_out]
    layers: list[tuple[int, int, str]] = []
    for i in range(len(sizes) - 1):
        act = output_activation if i == len(sizes) - 2 else activation
        layers.append((sizes[i], sizes[i + 1], act))
    return layers


# --------------------------------------------------------------------------------------------------
# Tensor layout helpers shared by the numpy and torch backends
# --------------------------------------------------------------------------------------------------
def stack_inputs(spec: ArchitectureSpec, x: Mapping[str, np.ndarray]) -> np.ndarray:
    """Stack per-feature arrays into the backend input tensor.

    Scalar features are ``(N,)``, sequence features ``(N, L)``. Feed-forward families give
    ``(N, n_in)``; sequence families give ``(N, L, n_in)`` with scalars repeated along the axis.
    """
    missing = [n for n in spec.input_features if n not in x]
    if missing:
        raise InvalidInputError(f"missing input features: {missing}", details={"missing": missing})
    arrays = [np.asarray(x[n], dtype=np.float64) for n in spec.input_features]
    if not spec.is_sequence_model:
        for name, arr in zip(spec.input_features, arrays):
            if arr.ndim != 1:
                raise InvalidInputError(f"feature {name!r} must have shape (N,), got {arr.shape}")
        if arrays[0].shape[0] == 0:
            raise InvalidInputError("empty batch (N = 0): nothing to evaluate", details={"batch_size": 0})
        return np.stack(arrays, axis=-1)
    seq = [a for n, a in zip(spec.input_features, arrays) if n in spec.sequence_features]
    lengths = {a.shape[-1] for a in seq}
    if any(a.ndim != 2 for a in seq) or len(lengths) != 1:
        raise InvalidInputError("sequence features must all have shape (N, L) with one common L")
    n_batch, length = seq[0].shape
    if n_batch == 0:
        raise InvalidInputError("empty batch (N = 0): nothing to evaluate", details={"batch_size": 0})
    if length == 0:
        raise InvalidInputError("sequence features must have at least one sample (L >= 1); a recurrent model applied "
                                "to zero time steps would only report its initial state", details={"sequence_length": 0})
    columns = []
    for name, arr in zip(spec.input_features, arrays):
        if name in spec.sequence_features:
            if arr.shape != (n_batch, length):
                raise InvalidInputError(f"feature {name!r} has shape {arr.shape}, expected {(n_batch, length)}")
            columns.append(arr)
        else:
            if arr.shape != (n_batch,):
                raise InvalidInputError(f"scalar feature {name!r} has shape {arr.shape}, expected {(n_batch,)}")
            columns.append(np.broadcast_to(arr[:, None], (n_batch, length)))
    return np.stack(columns, axis=-1)


def unstack_outputs(spec: ArchitectureSpec, y: np.ndarray) -> dict[str, np.ndarray]:
    """Split the backend output tensor ``(N, n_out)`` or ``(N, L, n_out)`` into per-feature arrays."""
    y = np.asarray(y, dtype=np.float64)
    n_out = len(spec.output_features)
    if y.ndim not in (2, 3) or y.shape[-1] != n_out:
        raise InvalidInputError(f"backend output has shape {y.shape}; expected (N, {n_out}) or (N, L, {n_out})")
    return {name: np.ascontiguousarray(y[..., j]) for j, name in enumerate(spec.output_features)}


__all__ = [
    "ACTIVATIONS", "Activation", "ArchitectureSpec", "DEFAULT_KIND", "EXTERNAL_KIND", "ExternalHyperparameters",
    "FAMILIES", "Family", "LSTMHyperparameters", "MLPHyperparameters", "NeuralODEHyperparameters",
    "TransformerHyperparameters", "TypedHyperparameters", "parse_hyperparameters", "stack_inputs", "unstack_outputs",
]
