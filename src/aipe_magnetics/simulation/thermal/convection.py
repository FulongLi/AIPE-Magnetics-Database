"""Surface heat-transfer coefficients: natural convection (Churchill-Chu), linearised radiation, forced convection.

* Natural convection from a vertical isothermal surface of height ``L`` (Churchill & Chu 1975, all Ra)::

      Nu = {0.825 + 0.387 Ra^(1/6) / [1 + (0.492/Pr)^(9/16)]^(8/27)}^2,   Ra = g beta dT L^3 / (nu alpha)

  applied to every exposed surface of the component with ``L`` = component height (an engineering simplification for
  the horizontal top / bottom faces; stated in the results).
* Radiation to surroundings at ambient temperature: ``h_rad = eps sigma (T_s^2 + T_a^2)(T_s + T_a)``.
* Forced convection: a user-supplied heat-transfer coefficient (no correlation is assumed for an unknown air flow).

Air properties at the film temperature use power-law fits through the 300 K handbook values of the property library
(``k ~ T^0.81``, ``nu ~ T^1.72``, Pr = 0.71; within about 3 % of Incropera Table A.4 between 280 K and 420 K).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from aipe_magnetics.common.errors import InvalidInputError

STEFAN_BOLTZMANN = 5.670374419e-8
GRAVITY = 9.80665


def air_properties(temperature: float) -> dict[str, float]:
    from aipe_magnetics.database.properties import physical_property
    k300 = physical_property("air", "thermal_conductivity").value
    nu300 = physical_property("air", "kinematic_viscosity").value
    pr = physical_property("air", "prandtl_number").value
    t = max(temperature, 200.0)
    return {"k": k300 * (t / 300.0) ** 0.81, "nu": nu300 * (t / 300.0) ** 1.72, "Pr": pr}


def natural_convection_h(surface_temperature: float, ambient: float, height: float) -> float:
    """Churchill-Chu vertical-plate coefficient (W/m^2/K); 0 for a surface not warmer than ambient."""
    if height <= 0:
        raise InvalidInputError("characteristic height must be positive")
    dt = surface_temperature - ambient
    if dt <= 1e-9:
        return 0.0
    film = 0.5 * (surface_temperature + ambient)
    p = air_properties(film)
    alpha = p["nu"] / p["Pr"]
    ra = GRAVITY * (1.0 / film) * dt * height ** 3 / (p["nu"] * alpha)
    nu = (0.825 + 0.387 * ra ** (1.0 / 6.0) / (1.0 + (0.492 / p["Pr"]) ** (9.0 / 16.0)) ** (8.0 / 27.0)) ** 2
    return nu * p["k"] / height


def radiation_h(surface_temperature: float, ambient: float, emissivity: float) -> float:
    ts, ta = surface_temperature, ambient
    return emissivity * STEFAN_BOLTZMANN * (ts * ts + ta * ta) * (ts + ta)


@dataclass(frozen=True)
class Cooling:
    """Cooling boundary condition of the component."""

    mode: Literal["natural", "forced"] = "natural"
    ambient: float = 313.15                 # K
    forced_h: float | None = None           # W/m^2/K (mode='forced')
    radiation: bool = True
    emissivity: float = 0.9
    note: str | None = None

    def __post_init__(self) -> None:
        if self.mode == "forced" and (self.forced_h is None or self.forced_h <= 0):
            raise InvalidInputError("forced cooling needs a positive forced_h (W/m^2/K)")
        if not (0 < self.ambient < 600):
            raise InvalidInputError("ambient temperature must be given in K")

    def h_total(self, surface_temperature: float, height: float) -> float:
        h = self.forced_h if self.mode == "forced" else natural_convection_h(surface_temperature, self.ambient, height)
        if self.radiation:
            h += radiation_h(surface_temperature, self.ambient, self.emissivity)
        return float(h)

    def describe(self) -> str:
        base = (f"forced convection h = {self.forced_h:g} W/m^2/K" if self.mode == "forced"
                else "natural convection (Churchill-Chu vertical plate, L = component height)")
        if self.radiation:
            base += f" + radiation (emissivity {self.emissivity:g})"
        return base + f", ambient {self.ambient - 273.15:.1f} degC"

    def to_json(self) -> dict:
        return {"mode": self.mode, "ambient_K": self.ambient, "forced_h_W_per_m2K": self.forced_h, "radiation": self.radiation,
                "emissivity": self.emissivity, "description": self.describe()}


__all__ = ["Cooling", "air_properties", "natural_convection_h", "radiation_h", "STEFAN_BOLTZMANN"]
