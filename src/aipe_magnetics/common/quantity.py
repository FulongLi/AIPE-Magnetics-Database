"""Typed physical quantities.

A :class:`Quantity` is the single way physical values travel through the API. It always carries a
unit, optionally keeps the *original* value/unit before normalisation (reproducible conversion),
and may carry an uncertainty statement. Uncertainty is never fabricated: when unknown it is
``None`` and reported as unknown.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aipe_magnetics.common.errors import InvalidInputError, UnitError
from aipe_magnetics.common.units import units


def _real_number(value: Any, *, name: str = "value") -> Any:
    """Reject booleans and strings *before* pydantic's lax coercion turns them into numbers."""
    if value is None:
        return value
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {type(value).__name__} {value!r}")
    return value


class Uncertainty(BaseModel):
    """Uncertainty statement attached to a quantity.

    ``kind='unknown'`` is the honest default when no quantitative basis exists.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["standard", "expanded", "bounds", "unknown"] = "unknown"
    value: float | None = Field(default=None, description="standard/expanded uncertainty in the quantity unit")
    lower: float | None = Field(default=None, description="lower bound (kind='bounds')")
    upper: float | None = Field(default=None, description="upper bound (kind='bounds')")
    coverage_factor: float | None = Field(default=None, description="k for expanded uncertainty")
    basis: str | None = Field(default=None, description="how the uncertainty was obtained")

    @field_validator("value", "lower", "upper", "coverage_factor", mode="before")
    @classmethod
    def _numbers(cls, v: Any, info: Any) -> Any:
        return _real_number(v, name=info.field_name)

    @model_validator(mode="after")
    def _consistent(self) -> "Uncertainty":
        if self.kind == "unknown" and any(x is not None for x in (self.value, self.lower, self.upper, self.coverage_factor)):
            raise ValueError("kind='unknown' cannot carry value, lower, upper or coverage_factor")
        if self.kind in ("standard", "expanded") and self.value is None:
            raise ValueError("standard/expanded uncertainty needs a value")
        if self.kind == "bounds" and (self.lower is None or self.upper is None):
            raise ValueError("bounds uncertainty needs lower and upper")
        if self.kind == "bounds" and self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("lower bound exceeds upper bound")
        if self.value is not None and (self.value < 0 or not math.isfinite(self.value)):
            raise ValueError("uncertainty value must be finite and non-negative")
        return self


class Original(BaseModel):
    """The value as supplied before unit normalisation."""

    model_config = ConfigDict(extra="forbid")
    value: float
    unit: str

    @field_validator("value", mode="before")
    @classmethod
    def _numeric(cls, v: Any) -> Any:
        return _real_number(v)


class Quantity(BaseModel):
    """``{value, unit}`` with optional provenance of the original unit and an uncertainty statement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float
    unit: str
    original: Original | None = None
    uncertainty: Uncertainty | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _numeric(cls, v: Any) -> Any:
        # strings such as "100" and booleans are rejected explicitly; pydantic's lax mode would coerce them
        return _real_number(v)

    @field_validator("value")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("value must be finite")
        return float(v)

    @field_validator("unit")
    @classmethod
    def _known_unit(cls, v: str) -> str:
        units.get(v)  # raises UnitError (a ValueError) when unknown
        return v

    # -- construction helpers -------------------------------------------------------------------
    @classmethod
    def si(cls, value: float, unit: str, **kwargs: Any) -> "Quantity":
        """Create a quantity normalised to the canonical SI unit, remembering the original."""
        si_value, si_unit = units.to_si(value, unit)
        original = None if si_unit == unit else Original(value=float(value), unit=unit)
        return cls(value=si_value, unit=si_unit, original=original, **kwargs)

    @classmethod
    def parse(cls, data: Any, *, name: str = "quantity") -> "Quantity":
        """Parse ``Quantity``, ``{value, unit}`` dict or ``(value, unit)`` tuple; never a bare number."""
        if isinstance(data, Quantity):
            return data
        if isinstance(data, dict):
            try:
                return cls.model_validate(data)
            except (ValueError, UnitError) as error:
                raise InvalidInputError(f"{name}: {error}") from None
        if isinstance(data, (tuple, list)) and len(data) == 2:
            try:
                return cls.si(_real_number(data[0]), str(data[1]))
            except (ValueError, UnitError) as error:
                raise InvalidInputError(f"{name}: {error}") from None
        raise InvalidInputError(
            f"{name}: physical values must be given as {{'value': ..., 'unit': ...}}, not {type(data).__name__}")

    # -- conversion -----------------------------------------------------------------------------
    def to(self, unit: str) -> "Quantity":
        if unit == self.unit:
            return self
        value = units.convert(self.value, self.unit, unit)
        unc = None
        if self.uncertainty is not None:
            unc = self.uncertainty
            if unc.kind in ("standard", "expanded") and unc.value is not None:
                # offsets cancel for differences, so a pure factor applies
                factor = units.convert(1.0, self.unit, unit) - units.convert(0.0, self.unit, unit)
                unc = unc.model_copy(update={"value": abs(unc.value * factor)})
            elif unc.kind == "bounds" and unc.lower is not None and unc.upper is not None:
                lo, hi = units.convert(unc.lower, self.unit, unit), units.convert(unc.upper, self.unit, unit)
                unc = unc.model_copy(update={"lower": min(lo, hi), "upper": max(lo, hi)})
        return Quantity(value=value, unit=unit, original=self.original or Original(value=self.value, unit=self.unit),
                        uncertainty=unc)

    def to_si(self) -> "Quantity":
        return self.to(units.si_unit_of(self.unit))

    @property
    def si_value(self) -> float:
        return units.convert(self.value, self.unit, units.si_unit_of(self.unit))

    def magnitude(self, unit: str) -> float:
        """Numeric value expressed in ``unit`` (dimension-checked)."""
        return units.convert(self.value, self.unit, unit)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.value:g} {self.unit}"


def require_positive(q: Quantity, name: str) -> Quantity:
    if q.si_value <= 0:
        raise InvalidInputError(f"{name} must be positive, got {q.value} {q.unit}", details={"name": name})
    return q


def require_nonnegative(q: Quantity, name: str) -> Quantity:
    if q.si_value < 0:
        raise InvalidInputError(f"{name} must be non-negative, got {q.value} {q.unit}", details={"name": name})
    return q


def require_dimension(q: Quantity, unit: str, name: str) -> Quantity:
    if not units.compatible(q.unit, unit):
        raise UnitError(f"{name} must have the dimension of {unit!r}, got {q.unit!r}", details={"name": name})
    return q
