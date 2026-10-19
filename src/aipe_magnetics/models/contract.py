"""The Magnetic Model Contract.

Every model family (empirical, physics-based, dynamic/ODE, data-driven) implements the same
contract so that the engineering layer, the verification framework, the API and the MCP server
treat them uniformly:

* :class:`ModelIdentity`  - stable id, name, implementation version, family, numerical-method version.
* :class:`ValidityDomain` - supported materials, excitation kinds, frequency/temperature/flux ranges,
  DC-bias handling, whether magnetic history (state) is required, numerical constraints and known
  limitations.
* :meth:`MagneticModel.check_domain` - returns a :class:`DomainCheck`; the caller's *policy* decides
  whether out-of-domain inputs are rejected (:class:`OutOfDomainError`) or evaluated with an
  ``out_of_range`` flag.
* :class:`ModelOutput` - values with units plus assumptions, limitations, warnings and diagnostics.

Models never clip or silently alter inputs. They raise :class:`InvalidInputError` for impossible
inputs (negative frequency, non-finite values) and :class:`NumericalError` for overflow or
nonphysical outputs.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aipe_magnetics.common.errors import InvalidInputError, NumericalError, OutOfDomainError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import Assumption, Limitation, ModelIdentity, SourceReference, Warning_

ExcitationKind = Literal["sinusoidal", "triangular", "trapezoidal", "square", "piecewise_linear", "sampled", "dc"]
DomainPolicy = Literal["reject", "warn"]
DCBiasSupport = Literal["unsupported", "ignored", "supported"]


class ValidityRange(BaseModel):
    """Closed interval in SI units. ``None`` bounds are open."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    min: float | None = None
    max: float | None = None
    unit: str

    def contains(self, value: float) -> bool:
        if not math.isfinite(value):
            return False
        if self.min is not None and value < self.min:
            return False
        if self.max is not None and value > self.max:
            return False
        return True

    def describe(self) -> str:
        lo = "-inf" if self.min is None else f"{self.min:g}"
        hi = "+inf" if self.max is None else f"{self.max:g}"
        return f"[{lo}, {hi}] {self.unit}"


class ValidityDomain(BaseModel):
    """Declared validity of a model (what the model *claims* to cover, independent of verification)."""

    model_config = ConfigDict(extra="forbid")

    materials: list[str] | None = Field(default=None, description="material ids; None = parametrised/material-agnostic")
    excitation: list[ExcitationKind] = Field(default_factory=list)
    frequency: ValidityRange | None = None
    temperature: ValidityRange | None = None
    flux_density_peak: ValidityRange | None = None
    field_peak: ValidityRange | None = None
    dc_bias: DCBiasSupport = "unsupported"
    requires_history: bool = Field(default=False, description="True if the model needs prior magnetic state")
    numerical_constraints: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)


class DomainCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inside: bool
    violations: list[str] = Field(default_factory=list)
    checked: dict[str, str] = Field(default_factory=dict, description="condition -> range description")

    def raise_if_outside(self, model_id: str) -> None:
        if not self.inside:
            raise OutOfDomainError(f"{model_id}: inputs outside declared validity domain: " + "; ".join(self.violations),
                                   details={"model": model_id, "violations": list(self.violations)})


class ModelParameter(BaseModel):
    """A named, unit-bearing model parameter with its provenance."""

    model_config = ConfigDict(extra="forbid")
    name: str
    value: float
    unit: str = "1"
    description: str = ""
    source_ref: str | None = Field(default=None, description="id of a SourceReference")


class ModelOutput(BaseModel):
    """Output of a single model evaluation."""

    model_config = ConfigDict(extra="forbid")
    model: ModelIdentity
    values: dict[str, Quantity] = Field(default_factory=dict)
    domain: DomainCheck
    assumptions: list[Assumption] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)
    warnings: list[Warning_] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict, description="solver statistics, intermediate quantities")

    def value(self, key: str) -> Quantity:
        return self.values[key]


class MagneticModel(ABC):
    """Base class of every model. Subclasses set ``identity`` and ``domain`` and implement ``evaluate``."""

    identity: ModelIdentity
    domain: ValidityDomain
    sources: list[SourceReference] = []
    parameters: list[ModelParameter] = []

    # -- domain -----------------------------------------------------------------------------------
    def check_domain(self, *, frequency: float | None = None, temperature: float | None = None,
                     flux_density_peak: float | None = None, field_peak: float | None = None,
                     material: str | None = None, excitation: str | None = None,
                     dc_bias: bool | None = None) -> DomainCheck:
        """Check SI-valued conditions against :attr:`domain`. Unknown (None) conditions are not checked."""
        violations: list[str] = []
        checked: dict[str, str] = {}
        pairs = (("frequency", frequency, self.domain.frequency),
                 ("temperature", temperature, self.domain.temperature),
                 ("flux_density_peak", flux_density_peak, self.domain.flux_density_peak),
                 ("field_peak", field_peak, self.domain.field_peak))
        for name, value, rng in pairs:
            if value is None or rng is None:
                continue
            checked[name] = rng.describe()
            if not rng.contains(float(value)):
                violations.append(f"{name}={float(value):g} {rng.unit} outside {rng.describe()}")
        if material is not None and self.domain.materials is not None:
            checked["material"] = ", ".join(self.domain.materials)
            if material not in self.domain.materials:
                violations.append(f"material {material!r} not in supported set")
        if excitation is not None and self.domain.excitation:
            checked["excitation"] = ", ".join(self.domain.excitation)
            if excitation not in self.domain.excitation:
                violations.append(f"excitation {excitation!r} not supported")
        if dc_bias:
            checked["dc_bias"] = self.domain.dc_bias
            if self.domain.dc_bias == "unsupported":
                violations.append("DC bias present but model does not support DC bias")
        return DomainCheck(inside=not violations, violations=violations, checked=checked)

    # -- evaluation ---------------------------------------------------------------------------------
    @abstractmethod
    def evaluate(self, *args: Any, policy: DomainPolicy = "reject", **kwargs: Any) -> ModelOutput:
        """Evaluate the model. Must call :meth:`check_domain` and honour ``policy``."""

    # -- helpers for subclasses ---------------------------------------------------------------------
    @staticmethod
    def require_finite_positive(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidInputError(f"{name} must be a real number")
        if not math.isfinite(value):
            raise InvalidInputError(f"{name} must be finite, got {value}")
        if value <= 0:
            raise InvalidInputError(f"{name} must be positive, got {value}")
        return float(value)

    @staticmethod
    def require_finite(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidInputError(f"{name} must be a real number")
        if not math.isfinite(value):
            raise InvalidInputError(f"{name} must be finite, got {value}")
        return float(value)

    @staticmethod
    def check_output_finite(name: str, value: float, *, nonnegative: bool = True) -> float:
        if not math.isfinite(value):
            raise NumericalError(f"{name} is not finite ({value}); inputs produce numerical overflow",
                                 details={"quantity": name})
        if nonnegative and value < 0:
            raise NumericalError(f"{name} is negative ({value}); nonphysical result", details={"quantity": name})
        return float(value)

    def describe(self) -> dict[str, Any]:
        return {
            "identity": self.identity.model_dump(mode="json"),
            "domain": self.domain.model_dump(mode="json", exclude_none=True),
            "parameters": [p.model_dump(mode="json", exclude_none=True) for p in self.parameters],
            "sources": [s.model_dump(mode="json", exclude_none=True) for s in self.sources],
        }


class CoreLossModel(MagneticModel):
    """Contract for volumetric core-loss models: returns ``loss_density`` in W/m^3."""

    @abstractmethod
    def loss_density(self, *, frequency: float, flux_density_peak: float, temperature: float | None = None,
                     policy: DomainPolicy = "reject", **kwargs: Any) -> ModelOutput:
        """Time-average volumetric loss for a periodic excitation. SI inputs: Hz, T (peak), K."""

    def evaluate(self, *args: Any, policy: DomainPolicy = "reject", **kwargs: Any) -> ModelOutput:
        return self.loss_density(*args, policy=policy, **kwargs)


class HysteresisModel(MagneticModel):
    """Contract for constitutive/hysteresis models producing B(H) or M(H) trajectories."""

    @abstractmethod
    def trajectory(self, field: Any, *, policy: DomainPolicy = "reject", **kwargs: Any) -> ModelOutput:
        """Given a sampled H(t) (A/m), return B(t) (T), M(t) (A/m) and loop energy diagnostics."""

    def evaluate(self, *args: Any, policy: DomainPolicy = "reject", **kwargs: Any) -> ModelOutput:
        return self.trajectory(*args, policy=policy, **kwargs)
