"""Optional PyTorch backend (``torch-state-dict`` and ``torchscript`` weights).

PyTorch is **never imported at module import time**. :func:`import_torch` imports it lazily and
raises :class:`UnsupportedOperationError` with the installation hint ``install aipe-magnetics[torch]``
when it is missing; there is no silent fallback to another backend.

Conventions
-----------

* ``torch-state-dict``: a ``torch.save`` file holding only tensors. It is read with
  ``torch.load(..., map_location="cpu", weights_only=True)`` so that no arbitrary pickled code can
  execute, and loaded with ``strict=True`` into a module built from the :class:`ArchitectureSpec`
  by :func:`build_module` (``ann``: ``layers.{i}`` ``nn.Linear`` stack; ``lstm``: ``nn.LSTM(batch_first=True)``
  named ``lstm`` plus a ``head.{i}`` ``nn.Linear`` stack). Parameter names therefore coincide with
  the numpy backend's parameter dictionary, and :func:`state_dict_to_numpy` /
  :func:`numpy_to_state_dict` convert between the two without any renaming.
* ``torchscript``: ``torch.jit.load`` of an exported program. Because this executes the archive,
  :class:`DataDrivenModel` refuses TorchScript bundles that carry no SHA-256 digest. The
  ``transformer`` family is served only through this path in v0.2.
* Both formats are parsed from the bytes returned by :meth:`ModelPackage.read_weights`, i.e. the
  buffer whose SHA-256/size were verified, never from a second read of the file.
* Inputs are stacked exactly like the numpy backend (:func:`stack_inputs`), cast to the configured
  dtype and evaluated under ``torch.inference_mode()``; outputs are returned as float64 numpy arrays.
  The precision actually used is exposed as ``TorchBackend.dtype`` (a TorchScript archive keeps its
  own parameter dtype), so :class:`DataDrivenModel` can report ``dtype_ignored`` when it differs from
  the configuration.
* A device that this torch build cannot use (``cuda`` without CUDA support, an accelerator that is
  not present) raises :class:`UnsupportedOperationError` naming the device; an unparsable device
  string is an :class:`InvalidInputError`. There is no silent fallback to the CPU.
* Activations: ``relu`` -> ``torch.relu``, ``tanh`` -> ``torch.tanh``, ``gelu`` -> ``torch.nn.functional.gelu``
  (exact erf form), ``sigmoid`` -> ``torch.sigmoid``, ``identity`` -> no-op; identical to the numpy
  definitions.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from aipe_magnetics.common.errors import DataIntegrityError, InvalidInputError, NumericalError, UnsupportedOperationError
from aipe_magnetics.models.datadriven.architectures import (
    ArchitectureSpec,
    LSTMHyperparameters,
    MLPHyperparameters,
    stack_inputs,
    unstack_outputs,
)

if TYPE_CHECKING:  # pragma: no cover
    from aipe_magnetics.models.datadriven.adapter import ModelPackage

TORCH_INSTALL_HINT = "install aipe-magnetics[torch]"


def import_torch() -> Any:
    """Import torch lazily; explicit :class:`UnsupportedOperationError` when it is not installed."""
    try:
        import torch  # noqa: WPS433 - optional dependency, imported on demand
    except ImportError as error:
        raise UnsupportedOperationError(
            f"the torch backend needs PyTorch, which is not installed: {TORCH_INSTALL_HINT} "
            "(pip install 'aipe-magnetics[torch]')",
            details={"backend": "torch", "import_error": str(error), "hint": TORCH_INSTALL_HINT}) from None
    return torch


def torch_available() -> bool:
    try:
        import_torch()
    except UnsupportedOperationError:
        return False
    return True


def _torch_dtype(torch: Any, name: str) -> Any:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _activate(torch: Any, name: str, x: Any) -> Any:
    if name == "identity":
        return x
    if name == "relu":
        return torch.relu(x)
    if name == "tanh":
        return torch.tanh(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    if name == "gelu":
        return torch.nn.functional.gelu(x)
    raise InvalidInputError(f"unknown activation {name!r}")


def build_module(spec: ArchitectureSpec, torch: Any | None = None) -> Any:
    """Build an un-initialised-by-us ``torch.nn.Module`` whose state-dict names match the descriptor."""
    torch = torch or import_torch()
    hp = spec.typed_hyperparameters()
    nn = torch.nn
    if isinstance(hp, MLPHyperparameters):
        sizes = spec.dense_layer_sizes()

        class MLPModule(nn.Module):  # type: ignore[misc,valid-type]
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(n_in, n_out) for n_in, n_out, _ in sizes])
                self.activations = [act for _, _, act in sizes]

            def forward(self, x: Any) -> Any:
                for layer, act in zip(self.layers, self.activations):
                    x = _activate(torch, act, layer(x))
                return x

        return MLPModule()
    if isinstance(hp, LSTMHyperparameters):
        head_sizes = spec.dense_layer_sizes()
        n_in = len(spec.input_features)

        class LSTMModule(nn.Module):  # type: ignore[misc,valid-type]
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(n_in, hp.hidden_size, num_layers=hp.num_layers, batch_first=True)
                self.head = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out, _ in head_sizes])
                self.head_activations = [act for _, _, act in head_sizes]
                self.output_mode = hp.output_mode

            def forward(self, x: Any) -> Any:
                outputs, (h_n, _) = self.lstm(x)
                z = outputs if self.output_mode == "sequence" else h_n[-1]
                for layer, act in zip(self.head, self.head_activations):
                    z = _activate(torch, act, layer(z))
                return z

        return LSTMModule()
    raise UnsupportedOperationError(
        f"the torch backend builds 'ann' (mlp) and 'lstm' modules from a descriptor; family {spec.family!r} "
        f"kind {spec.kind!r} needs a TorchScript export (format 'torchscript')",
        details={"family": spec.family, "kind": spec.kind})


def state_dict_to_numpy(state_dict: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Convert a torch state dict to the float64 parameter dictionary used by the numpy backend."""
    return {name: np.asarray(tensor.detach().cpu().numpy(), dtype=np.float64) for name, tensor in state_dict.items()}


def numpy_to_state_dict(params: Mapping[str, np.ndarray], torch: Any | None = None, dtype: str = "float64") -> dict[str, Any]:
    torch = torch or import_torch()
    return {name: torch.as_tensor(np.asarray(array, dtype=np.float64)).to(_torch_dtype(torch, dtype))
            for name, array in params.items()}


def save_state_dict(module_or_state_dict: Any, path: str | Path, torch: Any | None = None) -> Path:
    """``torch.save`` of a plain tensor state dict (loadable with ``weights_only=True``)."""
    torch = torch or import_torch()
    state = module_or_state_dict.state_dict() if hasattr(module_or_state_dict, "state_dict") else dict(module_or_state_dict)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in state.items()}, str(path))
    return path


class TorchBackend:
    """Backend for ``torch-state-dict`` (ann, lstm) and ``torchscript`` (ann, lstm, transformer) weights."""

    name = "torch"
    formats: tuple[str, ...] = ("torch-state-dict", "torchscript")

    def __init__(self) -> None:
        self.module: Any = None
        self.dtype: str | None = None  # precision actually used, e.g. "float32"; set by load()
        self._torch: Any = None
        self._dtype: Any = None
        self._device: Any = None

    def load(self, package: "ModelPackage") -> None:
        torch = import_torch()
        self._torch = torch
        spec = package.architecture
        fmt = package.weights.format
        if fmt not in self.formats:
            raise UnsupportedOperationError(f"the torch backend reads {self.formats}, not {fmt!r}", details={"format": fmt})
        device_name = package.inference.device
        self._dtype = _torch_dtype(torch, package.inference.dtype)
        try:
            self._device = torch.device(device_name)
        except (RuntimeError, TypeError) as error:
            raise InvalidInputError(f"unknown torch device {device_name!r}: {error}") from None
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise UnsupportedOperationError(
                f"device {device_name!r} requested but this PyTorch build has no usable CUDA device "
                "(torch.cuda.is_available() is False); use device='cpu' or install a CUDA-enabled build",
                details={"device": device_name, "backend": self.name})
        data, _ = package.read_weights()  # digest/size verified on this very buffer
        name = Path(package.weights.path or "").name
        if fmt == "torch-state-dict":
            module = build_module(spec, torch)
            try:
                state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            except Exception as error:  # noqa: BLE001 - any unpickling problem is an integrity problem
                raise DataIntegrityError(f"cannot read state dict {name} with weights_only=True: {error}",
                                         details={"path": package.weights.path}) from None
            if not isinstance(state, dict):
                raise DataIntegrityError(f"{name} does not contain a state dict", details={"path": package.weights.path})
            try:
                module.load_state_dict(state, strict=True)
            except RuntimeError as error:
                raise DataIntegrityError(f"state dict {name} does not match architecture {spec.name!r}: {error}",
                                         details={"path": package.weights.path}) from None
            module = module.to(self._dtype)
        else:
            try:
                module = torch.jit.load(io.BytesIO(data), map_location="cpu")
            except Exception as error:  # noqa: BLE001
                raise DataIntegrityError(f"cannot load TorchScript archive {name}: {error}",
                                         details={"path": package.weights.path}) from None
            try:
                first = next(iter(module.parameters()))
                self._dtype = first.dtype
            except StopIteration:
                pass
        try:
            self.module = module.to(self._device).eval()
        except (RuntimeError, AssertionError) as error:  # e.g. "Torch not compiled with CUDA enabled", missing accelerator
            raise UnsupportedOperationError(f"cannot place the model on device {device_name!r}: {error}",
                                            details={"device": device_name, "backend": self.name}) from None
        self.dtype = str(self._dtype).replace("torch.", "")

    def run(self, package: "ModelPackage", x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.module is None:
            raise InvalidInputError("backend not loaded; call load(package) first")
        torch = self._torch
        spec = package.architecture
        tensor = stack_inputs(spec, x)  # rejects N = 0 and L = 0
        if tensor.shape[0] == 0:  # pragma: no cover - defensive, stack_inputs already refuses empty batches
            raise InvalidInputError("empty batch (N = 0): nothing to evaluate", details={"batch_size": 0})
        chunk = max(1, int(package.inference.batch_size))
        pieces = []
        with torch.inference_mode():
            for start in range(0, tensor.shape[0], chunk):
                batch = torch.as_tensor(tensor[start:start + chunk]).to(dtype=self._dtype, device=self._device)
                out = self.module(batch)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                pieces.append(np.asarray(out.detach().cpu().numpy(), dtype=np.float64))
        y = np.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]
        if not np.all(np.isfinite(y)):
            raise NumericalError("network output is not finite", details={"backend": self.name})
        return unstack_outputs(spec, y)


__all__ = ["TORCH_INSTALL_HINT", "TorchBackend", "build_module", "import_torch", "numpy_to_state_dict",
           "save_state_dict", "state_dict_to_numpy", "torch_available"]
