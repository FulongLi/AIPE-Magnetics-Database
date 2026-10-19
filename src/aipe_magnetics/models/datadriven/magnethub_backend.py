"""Optional adapter to the Paderborn **MagNet-Hub** package (``magnethub``, MIT licence).

Upstream: https://github.com/upb-lea/mag-net-hub - pip name ``mag-net-hub``; the adapter was written
against version 0.0.11, commit ``1ed635d7bfac3c2aa024471af363beadc05f8932`` (read-only study of
``src_py/magnethub/loss.py``; no code was copied). Note that this upstream release declares
``requires-python >= 3.13`` and depends on torch, pandas, jax and equinox; AIPE Magnetics neither
bundles nor downloads it. ``magnethub`` is imported lazily; when it is missing
:class:`UnsupportedOperationError` is raised with an installation hint.

Upstream call convention (``magnethub.loss.LossModel``)::

    model = LossModel(material="N87", team="paderborn")        # material names as in the MagNet Challenge
    p, h = model(b_field, frequency, temperature)

    b_field     : (N, L) array, flux density in T, one full period per row
    frequency   : scalar or (N,) in Hz
    temperature : scalar or (N,) in degC
    p           : (N,) volumetric loss in W/m^3
    h           : (N, L) field strength in A/m

``LossModel`` interpolates ``b_field`` to the team model's ``expected_seq_len`` (1024 samples for the
Paderborn CNN, 128 for the Sydney model; module constant ``magnethub.loss.L = 1024``). This adapter
reads ``expected_seq_len`` **at runtime** and *rejects* sequences of any other length instead of
letting them be resampled silently, so callers resample explicitly and knowingly.

AIPE feature signature of a MagNet-Hub package (:func:`magnethub_package`)::

    inputs : flux_density [T] sequence (N, L), frequency [Hz] (N,), temperature [K] (N,)
    outputs: loss_density [W/m^3] (N,), field [A/m] sequence (N, L)

Temperature is converted from kelvin to degC (``T_degC = T_K - 273.15``) through the unit registry;
a temperature <= 0 K is rejected. The MagNet-Hub models are opaque architectures
(``hyperparameters.kind = "external"``); their validity ranges are those of the MagNet Challenge
training data and are *not* declared numerically here (see the package's ``known_limitations``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from aipe_magnetics.common.errors import DataIntegrityError, InvalidInputError, NumericalError, UnsupportedOperationError
from aipe_magnetics.common.units import units

if TYPE_CHECKING:  # pragma: no cover
    from aipe_magnetics.models.datadriven.adapter import ModelPackage

MAGNETHUB_IMPORT_NAME = "magnethub"
MAGNETHUB_PIP_NAME = "mag-net-hub"
MAGNETHUB_REFERENCE_VERSION = "0.0.11"
MAGNETHUB_REFERENCE_COMMIT = "1ed635d7bfac3c2aa024471af363beadc05f8932"
MAGNETHUB_URL = "https://github.com/upb-lea/mag-net-hub"
MAGNETHUB_INSTALL_HINT = (f"pip install {MAGNETHUB_PIP_NAME} (upstream requires Python >= 3.13 plus torch, pandas, "
                          "jax and equinox); AIPE Magnetics does not bundle or download it")
DEFAULT_SEQUENCE_LENGTH = 1024  # upstream magnethub.loss.L; the real value is read from the package at runtime
INPUT_FEATURES: tuple[str, ...] = ("flux_density", "frequency", "temperature")
OUTPUT_FEATURES: tuple[str, ...] = ("loss_density", "field")
TEAMS: tuple[str, ...] = ("paderborn", "sydney")
# Material identifiers accepted by magnethub.loss.MATERIALS at the reference commit (documentation only;
# the installed package is the authority and validates the material itself).
MATERIALS_AT_REFERENCE_COMMIT: tuple[str, ...] = ("ML95S", "T37", "3C90", "3C92", "3C94", "3C95", "3E6", "3F4",
                                                   "77", "78", "79", "N27", "N30", "N49", "N87")


def import_magnethub_loss() -> Any:
    """Import ``magnethub.loss`` lazily; explicit error when the package (or its torch stack) is missing."""
    try:
        import importlib

        return importlib.import_module(f"{MAGNETHUB_IMPORT_NAME}.loss")
    except ImportError as error:
        raise UnsupportedOperationError(
            f"the MagNet-Hub backend needs the optional package {MAGNETHUB_IMPORT_NAME!r}, which could not be "
            f"imported ({error}); {MAGNETHUB_INSTALL_HINT}",
            details={"backend": "magnethub", "import_error": str(error), "hint": MAGNETHUB_INSTALL_HINT}) from None


def magnethub_available() -> bool:
    try:
        import_magnethub_loss()
    except UnsupportedOperationError:
        return False
    return True


def kelvin_to_celsius(temperature_k: np.ndarray) -> np.ndarray:
    """``T_degC = T_K - 273.15`` via the unit registry; rejects non-positive absolute temperatures."""
    values = np.asarray(temperature_k, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise InvalidInputError("temperature must be a finite absolute temperature > 0 K",
                                details={"feature": "temperature"})
    degc = units.get("degC")  # value_in_SI = factor * value + offset  ->  degC = (K - offset) / factor
    return (values - float(degc.offset)) / float(degc.factor)


def prepare_inputs(x: Mapping[str, np.ndarray], expected_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map AIPE feature arrays to the ``(b_field[T], frequency[Hz], temperature[degC])`` upstream arguments."""
    missing = [n for n in INPUT_FEATURES if n not in x]
    if missing:
        raise InvalidInputError(f"MagNet-Hub inputs missing: {missing}", details={"missing": missing})
    b = np.asarray(x["flux_density"], dtype=np.float64)
    f = np.asarray(x["frequency"], dtype=np.float64).reshape(-1)
    t_k = np.asarray(x["temperature"], dtype=np.float64).reshape(-1)
    if b.ndim == 1:
        b = b[None, :]
    if b.ndim != 2:
        raise InvalidInputError(f"flux_density must have shape (N, L), got {b.shape}")
    n_batch, length = b.shape
    if length != expected_length:
        raise InvalidInputError(
            f"flux_density sequences have {length} samples but the installed MagNet-Hub model expects exactly "
            f"{expected_length} samples per period; resample explicitly (upstream would interpolate silently)",
            details={"length": length, "expected": expected_length})
    if f.shape != (n_batch,) or t_k.shape != (n_batch,):
        raise InvalidInputError(f"frequency and temperature must have shape ({n_batch},), got {f.shape} and {t_k.shape}")
    if np.any(~np.isfinite(b)) or np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise InvalidInputError("flux_density must be finite and frequency finite and positive")
    return b, f, kelvin_to_celsius(t_k)


def collect_outputs(p: Any, h: Any, n_batch: int) -> dict[str, np.ndarray]:
    """Validate upstream outputs ``p (N,) [W/m^3]`` and ``h (N, L) [A/m]`` and return float64 arrays."""
    loss = np.asarray(p, dtype=np.float64).reshape(-1)
    if loss.shape != (n_batch,):
        raise DataIntegrityError(f"MagNet-Hub returned loss of shape {np.shape(p)}, expected ({n_batch},)")
    if h is None:
        raise DataIntegrityError("MagNet-Hub returned no field sequence")
    field = np.asarray(h, dtype=np.float64)
    if field.ndim == 1:
        field = field[None, :]
    if field.ndim != 2 or field.shape[0] != n_batch:
        raise DataIntegrityError(f"MagNet-Hub returned field of shape {field.shape}, expected ({n_batch}, L)")
    if not (np.all(np.isfinite(loss)) and np.all(np.isfinite(field))):
        raise NumericalError("MagNet-Hub returned non-finite values", details={"backend": "magnethub"})
    return {"loss_density": loss, "field": field}


class MagnetHubBackend:
    """Backend wrapping ``magnethub.loss.LossModel`` for ``external-package`` weights."""

    name = "magnethub"
    formats: tuple[str, ...] = ("external-package",)

    def __init__(self) -> None:
        self.model: Any = None
        self.expected_sequence_length: int | None = None
        self.material: str | None = None
        self.team: str | None = None

    def load(self, package: "ModelPackage") -> None:
        spec = package.architecture
        if package.weights.format != "external-package":
            raise UnsupportedOperationError("the MagNet-Hub backend serves 'external-package' weights only",
                                            details={"format": package.weights.format})
        hp = spec.hyperparameters
        if hp.get("kind") != "external" or hp.get("package") != MAGNETHUB_IMPORT_NAME:
            raise InvalidInputError("MagNet-Hub packages need hyperparameters {'kind': 'external', 'package': 'magnethub', "
                                    "'material': ..., 'team': ...}", details={"hyperparameters": dict(hp)})
        if tuple(spec.input_features) != INPUT_FEATURES or tuple(spec.output_features) != OUTPUT_FEATURES:
            raise InvalidInputError(f"MagNet-Hub packages use inputs {INPUT_FEATURES} and outputs {OUTPUT_FEATURES}")
        material, team = str(hp.get("material", "")), str(hp.get("team", "paderborn"))
        loss = import_magnethub_loss()
        try:
            self.model = loss.LossModel(material=material, team=team)
        except ValueError as error:  # upstream rejects unknown materials/teams with ValueError
            raise InvalidInputError(f"MagNet-Hub rejected material={material!r}, team={team!r}: {error}",
                                    details={"material": material, "team": team}) from None
        self.material, self.team = material, team
        expected = getattr(getattr(self.model, "mdl", None), "expected_seq_len", getattr(loss, "L", DEFAULT_SEQUENCE_LENGTH))
        self.expected_sequence_length = int(expected)
        if spec.sequence_length is not None and spec.sequence_length != self.expected_sequence_length:
            raise DataIntegrityError(
                f"package declares sequence_length={spec.sequence_length} but the installed MagNet-Hub model for "
                f"{material}/{team} expects {self.expected_sequence_length} samples per period",
                details={"declared": spec.sequence_length, "runtime": self.expected_sequence_length})

    def run(self, package: "ModelPackage", x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.model is None or self.expected_sequence_length is None:
            raise InvalidInputError("backend not loaded; call load(package) first")
        b, f, t_degc = prepare_inputs(x, self.expected_sequence_length)
        p, h = self.model(b, f, t_degc)
        return collect_outputs(p, h, b.shape[0])


def magnethub_source(*, note: str | None = None) -> Any:
    from aipe_magnetics.common.results import SourceReference

    return SourceReference(
        id="src.upb-lea-mag-net-hub", kind="repository", url=MAGNETHUB_URL, license="MIT",
        reference="upb-lea/mag-net-hub: MagNet Toolkit - Certified Models of the MagNet Challenge (Paderborn University LEA)",
        version=f"{MAGNETHUB_REFERENCE_VERSION} @ {MAGNETHUB_REFERENCE_COMMIT}",
        note=note or "Adapter written against src_py/magnethub/loss.py; models and data are not redistributed by AIPE Magnetics.")


def magnethub_package(material: str, team: str = "paderborn", *, sequence_length: int | None = DEFAULT_SEQUENCE_LENGTH) -> Any:
    """Build the :class:`ModelPackage` descriptor for a MagNet-Hub loss model (no download, no import)."""
    from aipe_magnetics.common.results import ModelIdentity, SourceReference
    from aipe_magnetics.models.contract import ValidityDomain
    from aipe_magnetics.models.datadriven.adapter import InferenceConfig, ModelPackage, WeightsBundle
    from aipe_magnetics.models.datadriven.architectures import ArchitectureSpec

    if team not in TEAMS:
        raise InvalidInputError(f"team must be one of {TEAMS}, got {team!r}")
    if not material:
        raise InvalidInputError("material must be a non-empty MagNet Challenge material name, e.g. 'N87'")
    paper = SourceReference(
        id="src.foerster-2024-hardcore", kind="publication", url="https://arxiv.org/abs/2401.11488",
        reference="N. Foerster, W. Kirchgaessner, T. Piepenbrock, O. Schweins, O. Wallscheid, 'HARDCORE: H-field and power "
                  "loss estimation for arbitrary waveforms with residual, dilated convolutional neural networks in ferrite "
                  "cores', arXiv:2401.11488, 2024", note="Paderborn team model cited by the upstream source.")
    spec = ArchitectureSpec(
        family="ann", name=f"magnethub-{team}-{material}",
        hyperparameters={"kind": "external", "package": MAGNETHUB_IMPORT_NAME, "material": material, "team": team},
        input_features=list(INPUT_FEATURES), output_features=list(OUTPUT_FEATURES),
        sequence_features=["flux_density", "field"], sequence_length=sequence_length,
        input_units={"flux_density": "T", "frequency": "Hz", "temperature": "K"},
        output_units={"loss_density": "W/m^3", "field": "A/m"},
        description=f"MagNet-Hub {team} loss model for material {material} (opaque external architecture).")
    return ModelPackage(
        identity=ModelIdentity(id=f"aipe.magnetics.model.datadriven.magnethub-{team}-{material.lower()}",
                               name=f"MagNet-Hub {team} model ({material})", version="0.2.0", family="data_driven"),
        architecture=spec,
        weights=WeightsBundle(format="external-package", uri=f"pkg:pypi/{MAGNETHUB_PIP_NAME}@{MAGNETHUB_REFERENCE_VERSION}",
                              source=magnethub_source(), note="weights ship inside the magnethub package"),
        inference=InferenceConfig(domain_map={"frequency": "frequency", "temperature": "temperature",
                                              "flux_density_peak": "flux_density"}),
        domain=ValidityDomain(
            materials=[material], excitation=["sampled"], dc_bias="unsupported", requires_history=False,
            numerical_constraints=[f"flux_density must contain exactly one period sampled at the model's expected length "
                                   f"({sequence_length if sequence_length else 'read at runtime'} samples)"],
            known_limitations=[
                "Validity range equals the MagNet Challenge training data of the material; frequency, temperature and "
                "flux-density limits are not declared numerically here and are not checked.",
                "Requires the external mag-net-hub package (MIT) with its torch stack; nothing is downloaded by AIPE.",
            ]),
        sources=[magnethub_source(), paper], synthetic=False,
        description="Adapter descriptor for a MagNet-Hub certified loss model; inference runs in the external package.")


__all__ = [
    "DEFAULT_SEQUENCE_LENGTH", "INPUT_FEATURES", "MAGNETHUB_IMPORT_NAME", "MAGNETHUB_INSTALL_HINT", "MAGNETHUB_PIP_NAME",
    "MAGNETHUB_REFERENCE_COMMIT", "MAGNETHUB_REFERENCE_VERSION", "MAGNETHUB_URL", "MATERIALS_AT_REFERENCE_COMMIT",
    "MagnetHubBackend", "OUTPUT_FEATURES", "TEAMS", "collect_outputs", "import_magnethub_loss", "kelvin_to_celsius",
    "magnethub_available", "magnethub_package", "magnethub_source", "prepare_inputs",
]
