"""Data-driven model family: common adapter for ANN, LSTM, Transformer and Neural-ODE models.

Public surface
--------------

* :class:`ArchitectureSpec` - network descriptor (family, hyper-parameters, feature signature, units).
* :class:`WeightsBundle` - trained parameters: file/URI, format, SHA-256, :class:`SourceReference`.
* :class:`InferenceConfig` / :class:`NormalizationSpec` - scaling statistics and runtime settings.
* :class:`ModelPackage` - the three layers plus identity, validity domain, sources and training record.
* :class:`DataDrivenModel` - :class:`MagneticModel` implementation (``predict`` / ``evaluate``).
* :class:`NeuralODEModel` - executable Neural-ODE member (MLP vector field + ``solve_ivp``).
* Backends: :class:`NumpyBackend` (always), :class:`TorchBackend` and :class:`MagnetHubBackend`
  (optional, lazy imports with explicit errors); :func:`get_backend` / :func:`register_backend`.
* :data:`MODELS` - factories of the bundled reference models; :func:`reference_steinmetz_ann`
  loads the SYNTHETIC Steinmetz-surface surrogate shipped in ``reference/``.

Importing this package never imports torch, jax or magnethub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from aipe_magnetics.models.contract import MagneticModel
from aipe_magnetics.models.datadriven.adapter import (
    CONDITION_SI_UNITS,
    EXECUTABLE_FORMATS,
    WEIGHTS_FORMATS,
    Backend,
    DataDrivenModel,
    InferenceConfig,
    ModelPackage,
    NormalizationSpec,
    WeightsBundle,
    available_backends,
    coerce_inputs,
    condition_array,
    convert_array,
    get_backend,
    register_backend,
    reject_impossible_conditions,
    unregister_backend,
)
from aipe_magnetics.models.datadriven.architectures import (
    ACTIVATIONS,
    FAMILIES,
    ArchitectureSpec,
    LSTMHyperparameters,
    MLPHyperparameters,
    NeuralODEHyperparameters,
    TransformerHyperparameters,
    parse_hyperparameters,
)
from aipe_magnetics.models.datadriven.magnethub_backend import MagnetHubBackend, magnethub_available, magnethub_package
from aipe_magnetics.models.datadriven.neural_ode import NeuralODEModel, linear_forced_response
from aipe_magnetics.models.datadriven.numpy_backend import NumpyBackend, write_aipe_json
from aipe_magnetics.models.datadriven.torch_backend import TorchBackend, torch_available

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
REFERENCE_METADATA_NAME = "metadata.json"
REFERENCE_MODEL_ID = "aipe.magnetics.model.datadriven.reference-steinmetz-ann"


def reference_package_path() -> Path:
    """Path of the bundled reference package metadata (``reference/metadata.json``)."""
    return REFERENCE_DIR / REFERENCE_METADATA_NAME


def load_reference_package() -> ModelPackage:
    """Load (and validate) the bundled SYNTHETIC Steinmetz-surface ANN package without running it."""
    return ModelPackage.load(reference_package_path())


def reference_steinmetz_ann(**kwargs: Any) -> DataDrivenModel:
    """Factory: the bundled reference ANN (SYNTHETIC Steinmetz surface), integrity-checked on load."""
    return DataDrivenModel(load_reference_package(), **kwargs)


reference_steinmetz_ann.model_id = REFERENCE_MODEL_ID  # type: ignore[attr-defined]
reference_steinmetz_ann.synthetic = True  # type: ignore[attr-defined]

MODELS: list[Callable[..., MagneticModel]] = [reference_steinmetz_ann]
MODEL_IDS: tuple[str, ...] = (REFERENCE_MODEL_ID,)

__all__ = [
    "ACTIVATIONS", "ArchitectureSpec", "Backend", "CONDITION_SI_UNITS", "DataDrivenModel", "EXECUTABLE_FORMATS",
    "FAMILIES", "InferenceConfig", "LSTMHyperparameters", "MLPHyperparameters", "MODELS", "MODEL_IDS",
    "MagnetHubBackend", "ModelPackage", "NeuralODEHyperparameters", "NeuralODEModel", "NormalizationSpec",
    "NumpyBackend", "REFERENCE_DIR", "REFERENCE_MODEL_ID", "TorchBackend", "TransformerHyperparameters",
    "WEIGHTS_FORMATS", "WeightsBundle", "available_backends", "coerce_inputs", "condition_array", "convert_array",
    "get_backend", "linear_forced_response", "load_reference_package", "magnethub_available", "magnethub_package",
    "parse_hyperparameters", "reference_package_path", "reference_steinmetz_ann", "register_backend",
    "reject_impossible_conditions", "torch_available", "unregister_backend", "write_aipe_json",
]
