"""Explicit, dimension-checked unit handling.

The registry is deliberately small and transparent: every unit is defined by a scale factor (and an
offset for temperature) relative to coherent SI, together with an integer dimension vector
``(m, kg, s, A, K)``. Conversions are exact rational/decimal factors where possible so that
normalisation is reproducible bit-for-bit across platforms.

Canonical SI spellings follow AIPE-Core ``common.schema.json`` (``Ohm``, ``m^2``, ``W/m/K``, ``1``) and
extend them with magnetics quantities (``A/m``, ``W/m^3``, ``Ohm*m``, ``kg/m^3``, ``J/m^3``).

Nothing here guesses a unit. An unknown unit string raises :class:`UnitError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

from aipe_magnetics.common.errors import UnitError

Dim = tuple[int, int, int, int, int]  # (m, kg, s, A, K)

_DIMLESS: Dim = (0, 0, 0, 0, 0)


@dataclass(frozen=True)
class UnitDef:
    symbol: str
    dimension: Dim
    factor: Fraction  # value_in_SI = factor * value + offset
    offset: Fraction = Fraction(0)
    si: bool = False  # canonical coherent SI spelling for this dimension
    description: str = ""


def _d(m=0, kg=0, s=0, A=0, K=0) -> Dim:
    return (m, kg, s, A, K)


_F = Fraction


def _defs() -> Iterable[UnitDef]:
    # dimensionless
    yield UnitDef("1", _DIMLESS, _F(1), si=True, description="dimensionless")
    yield UnitDef("%", _DIMLESS, _F(1, 100), description="percent")
    yield UnitDef("ppm", _DIMLESS, _F(1, 1_000_000), description="parts per million")
    yield UnitDef("rad", _DIMLESS, _F(1), description="plane angle (treated as dimensionless)")
    yield UnitDef("deg", _DIMLESS, _F(1), description="degree of arc; converted numerically by convert()")
    # length
    yield UnitDef("m", _d(m=1), _F(1), si=True)
    yield UnitDef("cm", _d(m=1), _F(1, 100))
    yield UnitDef("mm", _d(m=1), _F(1, 1000))
    yield UnitDef("um", _d(m=1), _F(1, 10**6))
    yield UnitDef("nm", _d(m=1), _F(1, 10**9))
    yield UnitDef("in", _d(m=1), _F(254, 10000))
    yield UnitDef("mil", _d(m=1), _F(254, 10**7))
    # area / volume
    yield UnitDef("m^2", _d(m=2), _F(1), si=True)
    yield UnitDef("cm^2", _d(m=2), _F(1, 10**4))
    yield UnitDef("mm^2", _d(m=2), _F(1, 10**6))
    yield UnitDef("m^3", _d(m=3), _F(1), si=True)
    yield UnitDef("cm^3", _d(m=3), _F(1, 10**6))
    yield UnitDef("mm^3", _d(m=3), _F(1, 10**9))
    yield UnitDef("m^4", _d(m=4), _F(1), si=True, description="area product / second moment of area")
    yield UnitDef("cm^4", _d(m=4), _F(1, 10**8))
    yield UnitDef("mm^4", _d(m=4), _F(1, 10**12))
    yield UnitDef("L", _d(m=3), _F(1, 1000))
    # mass / density
    yield UnitDef("kg", _d(kg=1), _F(1), si=True)
    yield UnitDef("g", _d(kg=1), _F(1, 1000))
    yield UnitDef("kg/m^3", _d(m=-3, kg=1), _F(1), si=True)
    yield UnitDef("g/cm^3", _d(m=-3, kg=1), _F(1000))
    # time / frequency
    yield UnitDef("s", _d(s=1), _F(1), si=True)
    yield UnitDef("ms", _d(s=1), _F(1, 1000))
    yield UnitDef("us", _d(s=1), _F(1, 10**6))
    yield UnitDef("ns", _d(s=1), _F(1, 10**9))
    yield UnitDef("min", _d(s=1), _F(60))
    yield UnitDef("h", _d(s=1), _F(3600))
    yield UnitDef("Hz", _d(s=-1), _F(1), si=True)
    yield UnitDef("kHz", _d(s=-1), _F(1000))
    yield UnitDef("MHz", _d(s=-1), _F(10**6))
    yield UnitDef("1/s", _d(s=-1), _F(1))
    yield UnitDef("1/m", _d(m=-1), _F(1), si=True, description="reciprocal length (core constant sum(l/A))")
    yield UnitDef("1/mm", _d(m=-1), _F(1000))
    # electrical
    yield UnitDef("A", _d(A=1), _F(1), si=True)
    yield UnitDef("mA", _d(A=1), _F(1, 1000))
    yield UnitDef("kA", _d(A=1), _F(1000))
    yield UnitDef("V", _d(m=2, kg=1, s=-3, A=-1), _F(1), si=True)
    yield UnitDef("mV", _d(m=2, kg=1, s=-3, A=-1), _F(1, 1000))
    yield UnitDef("kV", _d(m=2, kg=1, s=-3, A=-1), _F(1000))
    yield UnitDef("W", _d(m=2, kg=1, s=-3), _F(1), si=True)
    yield UnitDef("mW", _d(m=2, kg=1, s=-3), _F(1, 1000))
    yield UnitDef("kW", _d(m=2, kg=1, s=-3), _F(1000))
    yield UnitDef("MW", _d(m=2, kg=1, s=-3), _F(10**6))
    yield UnitDef("VA", _d(m=2, kg=1, s=-3), _F(1), description="apparent power (same dimension as W)")
    yield UnitDef("kVA", _d(m=2, kg=1, s=-3), _F(1000))
    yield UnitDef("J", _d(m=2, kg=1, s=-2), _F(1), si=True)
    yield UnitDef("mJ", _d(m=2, kg=1, s=-2), _F(1, 1000))
    yield UnitDef("kJ", _d(m=2, kg=1, s=-2), _F(1000))
    yield UnitDef("Wh", _d(m=2, kg=1, s=-2), _F(3600))
    yield UnitDef("Ohm", _d(m=2, kg=1, s=-3, A=-2), _F(1), si=True)
    yield UnitDef("mOhm", _d(m=2, kg=1, s=-3, A=-2), _F(1, 1000))
    yield UnitDef("kOhm", _d(m=2, kg=1, s=-3, A=-2), _F(1000))
    yield UnitDef("S", _d(m=-2, kg=-1, s=3, A=2), _F(1), si=True)
    yield UnitDef("Ohm*m", _d(m=3, kg=1, s=-3, A=-2), _F(1), si=True, description="resistivity")
    yield UnitDef("Ohm*cm", _d(m=3, kg=1, s=-3, A=-2), _F(1, 100))
    yield UnitDef("uOhm*cm", _d(m=3, kg=1, s=-3, A=-2), _F(1, 10**8))
    yield UnitDef("S/m", _d(m=-3, kg=-1, s=3, A=2), _F(1), si=True, description="conductivity")
    yield UnitDef("MS/m", _d(m=-3, kg=-1, s=3, A=2), _F(10**6))
    yield UnitDef("H", _d(m=2, kg=1, s=-2, A=-2), _F(1), si=True)
    yield UnitDef("mH", _d(m=2, kg=1, s=-2, A=-2), _F(1, 1000))
    yield UnitDef("uH", _d(m=2, kg=1, s=-2, A=-2), _F(1, 10**6))
    yield UnitDef("nH", _d(m=2, kg=1, s=-2, A=-2), _F(1, 10**9))
    yield UnitDef("H/m", _d(m=1, kg=1, s=-2, A=-2), _F(1), si=True, description="permeability")
    yield UnitDef("F", _d(m=-2, kg=-1, s=4, A=2), _F(1), si=True)
    yield UnitDef("uF", _d(m=-2, kg=-1, s=4, A=2), _F(1, 10**6))
    yield UnitDef("nF", _d(m=-2, kg=-1, s=4, A=2), _F(1, 10**9))
    yield UnitDef("pF", _d(m=-2, kg=-1, s=4, A=2), _F(1, 10**12))
    yield UnitDef("V*s", _d(m=2, kg=1, s=-2, A=-1), _F(1), si=True, description="flux linkage / volt-seconds")
    yield UnitDef("Wb", _d(m=2, kg=1, s=-2, A=-1), _F(1), description="magnetic flux")
    yield UnitDef("A*turn", _d(A=1), _F(1), description="magnetomotive force (ampere-turns)")
    # magnetic
    yield UnitDef("T", _d(kg=1, s=-2, A=-1), _F(1), si=True)
    yield UnitDef("mT", _d(kg=1, s=-2, A=-1), _F(1, 1000))
    yield UnitDef("uT", _d(kg=1, s=-2, A=-1), _F(1, 10**6))
    yield UnitDef("G", _d(kg=1, s=-2, A=-1), _F(1, 10**4), description="gauss")
    yield UnitDef("kG", _d(kg=1, s=-2, A=-1), _F(1, 10))
    yield UnitDef("A/m", _d(m=-1, A=1), _F(1), si=True)
    yield UnitDef("kA/m", _d(m=-1, A=1), _F(1000))
    yield UnitDef("Oe", _d(m=-1, A=1), _F(1000, 4) / _F(314159265358979323846, 10**20), description="oersted (1000/(4*pi) A/m)")
    # power density, energy density
    yield UnitDef("W/m^3", _d(m=-1, kg=1, s=-3), _F(1), si=True, description="volumetric loss density")
    yield UnitDef("kW/m^3", _d(m=-1, kg=1, s=-3), _F(1000))
    yield UnitDef("mW/cm^3", _d(m=-1, kg=1, s=-3), _F(1000))
    yield UnitDef("W/cm^3", _d(m=-1, kg=1, s=-3), _F(10**6))
    yield UnitDef("W/kg", _d(m=2, s=-3), _F(1), si=True, description="mass-specific loss")
    yield UnitDef("J/m^3", _d(m=-1, kg=1, s=-2), _F(1), si=True, description="energy density")
    yield UnitDef("kJ/m^3", _d(m=-1, kg=1, s=-2), _F(1000))
    yield UnitDef("W/m^2", _d(kg=1, s=-3), _F(1), si=True)
    # thermal
    yield UnitDef("K", _d(K=1), _F(1), si=True)
    yield UnitDef("degC", _d(K=1), _F(1), _F(27315, 100), description="degree Celsius")
    yield UnitDef("degF", _d(K=1), _F(5, 9), _F(45967, 180), description="degree Fahrenheit")
    yield UnitDef("K/W", _d(m=-2, kg=-1, s=3, K=1), _F(1), si=True)
    yield UnitDef("W/m/K", _d(m=1, kg=1, s=-3, K=-1), _F(1), si=True)
    yield UnitDef("1/K", _d(K=-1), _F(1), si=True, description="temperature coefficient")
    yield UnitDef("W/m^2/K", _d(kg=1, s=-3, K=-1), _F(1), si=True, description="heat transfer coefficient")
    yield UnitDef("J/kg/K", _d(m=2, s=-2, K=-1), _F(1), si=True, description="specific heat capacity")
    yield UnitDef("m^2/s", _d(m=2, s=-1), _F(1), si=True, description="kinematic viscosity / diffusivity")
    # velocity, misc
    yield UnitDef("m/s", _d(m=1, s=-1), _F(1), si=True)
    yield UnitDef("T/s", _d(kg=1, s=-3, A=-1), _F(1), si=True, description="flux density slew rate")
    yield UnitDef("A/s", _d(s=-1, A=1), _F(1), si=True)
    yield UnitDef("V/m", _d(m=1, kg=1, s=-3, A=-1), _F(1), si=True)
    yield UnitDef("A/m^2", _d(m=-2, A=1), _F(1), si=True, description="current density")
    yield UnitDef("A/mm^2", _d(m=-2, A=1), _F(10**6))


class UnitRegistry:
    """Immutable registry of known units with dimension-checked conversion."""

    def __init__(self, definitions: Iterable[UnitDef] | None = None):
        self._units: dict[str, UnitDef] = {}
        for item in definitions if definitions is not None else _defs():
            if item.symbol in self._units:
                raise ValueError(f"duplicate unit definition {item.symbol}")
            self._units[item.symbol] = item
        self._si_by_dim: dict[Dim, str] = {}
        for item in self._units.values():
            if item.si:
                self._si_by_dim.setdefault(item.dimension, item.symbol)

    # -- lookup ---------------------------------------------------------------------------------
    def get(self, symbol: str) -> UnitDef:
        if not isinstance(symbol, str):
            raise UnitError(f"unit must be a string, got {type(symbol).__name__}")
        try:
            return self._units[symbol]
        except KeyError:
            raise UnitError(f"unknown unit {symbol!r}; known units: {', '.join(sorted(self._units))}",
                            details={"unit": symbol}) from None

    def known(self, symbol: str) -> bool:
        return symbol in self._units

    def symbols(self) -> list[str]:
        return sorted(self._units)

    def dimension(self, symbol: str) -> Dim:
        return self.get(symbol).dimension

    def is_si(self, symbol: str) -> bool:
        return self.get(symbol).si

    def si_unit_of(self, symbol: str) -> str:
        """Return the canonical coherent SI spelling for the dimension of ``symbol``."""
        unit = self.get(symbol)
        try:
            return self._si_by_dim[unit.dimension]
        except KeyError:  # pragma: no cover - every dimension in _defs has an SI spelling
            raise UnitError(f"no canonical SI unit registered for dimension of {symbol!r}") from None

    def compatible(self, a: str, b: str) -> bool:
        return self.get(a).dimension == self.get(b).dimension

    # -- conversion -----------------------------------------------------------------------------
    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert ``value`` between units of the same dimension. Raises :class:`UnitError` otherwise."""
        src, dst = self.get(from_unit), self.get(to_unit)
        if src.dimension != dst.dimension:
            raise UnitError(f"cannot convert {from_unit!r} to {to_unit!r}: incompatible dimensions",
                            details={"from": from_unit, "to": to_unit})
        if from_unit == to_unit:
            return float(value)
        if {from_unit, to_unit} == {"rad", "deg"}:
            import math
            return float(value) * (math.pi / 180.0 if from_unit == "deg" else 180.0 / math.pi)
        if from_unit in ("rad", "deg") or to_unit in ("rad", "deg"):
            if (from_unit in ("rad", "deg")) != (to_unit in ("rad", "deg")):
                raise UnitError(f"angles ({from_unit!r}) are not interchangeable with plain ratios ({to_unit!r})")
        si_value = float(value) * float(src.factor) + float(src.offset)
        return (si_value - float(dst.offset)) / float(dst.factor)

    def to_si(self, value: float, unit: str) -> tuple[float, str]:
        """Return ``(value_in_SI, canonical_SI_unit)``."""
        si = self.si_unit_of(unit)
        return self.convert(value, unit, si), si

    def factor_to_si(self, unit: str) -> float:
        """Multiplicative factor to SI (offset units such as degC are rejected)."""
        definition = self.get(unit)
        if definition.offset != 0:
            raise UnitError(f"{unit!r} has an offset; use convert() instead of a pure factor")
        return float(definition.factor)


units = UnitRegistry()


def convert(value: float, from_unit: str, to_unit: str) -> float:
    return units.convert(value, from_unit, to_unit)


def to_si(value: float, unit: str) -> tuple[float, str]:
    return units.to_si(value, unit)


def si_unit_of(unit: str) -> str:
    return units.si_unit_of(unit)


# Physical constants (CODATA 2018 / SI 2019). mu_0 is no longer exact; the value below is the
# CODATA 2018 recommended value, relative uncertainty 1.5e-10.
MU_0 = 1.25663706212e-6  # H/m
COPPER_RESISTIVITY_20C = 1.68e-8  # Ohm*m, annealed copper, commonly tabulated value (IACS 100% = 1.7241e-8)
COPPER_TEMPERATURE_COEFFICIENT = 0.00393  # 1/K, annealed copper near 20 degC
