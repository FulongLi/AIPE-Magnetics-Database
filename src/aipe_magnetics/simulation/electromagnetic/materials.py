"""Material laws of the electromagnetic solvers.

* Linear reluctivity ``nu = 1 / (mu_0 mu_r)`` per region material.
* :class:`SaturationCurve` - an *assumed* single-valued anhysteretic curve
  ``B(H) = mu_0 H + J_s tanh(mu_0 (mu_i - 1) H / J_s)`` calibrated to two catalogue numbers: the initial
  permeability ``mu_i`` (slope at the origin) and the saturation polarisation ``J_s = B_sat - mu_0 H_sat``
  (``B_sat`` measured at ``H_sat``). It reproduces the datasheet point by construction only approximately
  (the tanh knee is a modelling choice); results that depend on the knee are flagged as such. The curve is
  used for local saturation assessment by the nonlinear magnetostatic solver.
* Copper conductivity ``sigma(T) = 1 / rho(T)`` from the v0.2 linear resistivity model
  (:func:`aipe_magnetics.engineering.formulas.copper_resistivity`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from aipe_magnetics.common.units import MU_0


@dataclass(frozen=True)
class SaturationCurve:
    mu_i: float
    b_sat: float
    h_sat: float = 1200.0
    note: str = "tanh anhysteretic curve calibrated to mu_i and Bsat (assumption)"
    _table: tuple[np.ndarray, np.ndarray] = field(default=None, repr=False, compare=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not (self.mu_i > 1 and self.b_sat > 0):
            raise ValueError("SaturationCurve needs mu_i > 1 and b_sat > 0")
        h = np.concatenate([[0.0], np.geomspace(1e-3, 1e7, 4000)])
        b = self.b_of_h(h)
        object.__setattr__(self, "_table", (b, h))

    @property
    def j_s(self) -> float:
        return self.b_sat - MU_0 * self.h_sat

    def b_of_h(self, h: np.ndarray | float) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        return MU_0 * h + self.j_s * np.tanh(MU_0 * (self.mu_i - 1.0) * h / self.j_s)

    def h_of_b(self, b: np.ndarray | float) -> np.ndarray:
        bt, ht = self._table
        b = np.abs(np.asarray(b, dtype=float))
        h = np.interp(b, bt, ht)
        beyond = b > bt[-1]
        if np.any(beyond):
            h = np.where(beyond, ht[-1] + (b - bt[-1]) / MU_0, h)
        return h

    def reluctivity(self, b: np.ndarray | float) -> np.ndarray:
        """Secant reluctivity ``nu(|B|) = H / B`` (limit ``1 / (mu_0 mu_i)`` at B -> 0)."""
        b = np.abs(np.asarray(b, dtype=float))
        nu0 = 1.0 / (MU_0 * self.mu_i)
        with np.errstate(divide="ignore", invalid="ignore"):
            nu = np.where(b > 1e-9, self.h_of_b(b) / np.maximum(b, 1e-300), nu0)
        return nu

    def to_json(self) -> dict[str, float | str]:
        return {"mu_i": self.mu_i, "b_sat_T": self.b_sat, "h_sat_A_per_m": self.h_sat, "j_s_T": self.j_s, "model": self.note}


def copper_conductivity(temperature: float) -> float:
    from aipe_magnetics.engineering.formulas import copper_resistivity
    return 1.0 / copper_resistivity(temperature)


def skin_depth(frequency: float, temperature: float) -> float:
    return math.sqrt(1.0 / (math.pi * frequency * MU_0 * copper_conductivity(temperature)))


__all__ = ["SaturationCurve", "copper_conductivity", "skin_depth"]
