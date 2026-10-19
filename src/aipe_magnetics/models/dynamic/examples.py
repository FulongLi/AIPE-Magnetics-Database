"""Reference dynamic models used to verify the ODE framework.

LinearRL
--------
Series R-L circuit driven by a voltage source ``v(t)``::

    di/dt = (v(t) - R i) / L,    tau = L / R

Closed-form solutions (used as analytical references):

* constant ``v(t) = V``:  ``i(t) = V/R + (i0 - V/R) exp(-t/tau)``
* sinusoid ``v(t) = V0 + V sin(w t + phi)``:  with ``|Z| = sqrt(R^2 + (wL)^2)`` and
  ``theta = atan2(wL, R)``, the particular solution is ``i_p(t) = V0/R + (V/|Z|) sin(w t + phi - theta)``
  and ``i(t) = i_p(t) + (i0 - i_p(0)) exp(-t/tau)``.

Outputs: ``v``, ``p_in = v i``, ``p_R = R i^2``, ``w_L = L i^2 / 2``.  Optional constraint ``|i| <= i_max``
is *reported* on the solution and never enforced.

NonlinearInductor
-----------------
Flux-linkage state ``lambda`` with a saturating, explicit current relation (arctangent flux
linkage, its inverse is a tangent)::

    lambda(i) = L0 * I_sat * atan(i / I_sat)          (small-signal inductance L0 at i = 0)
    i(lambda) = I_sat * tan(lambda / (L0 * I_sat))     (defined for |lambda| < lambda_max = pi/2 * L0 * I_sat)
    dlambda/dt = v(t) - R * i(lambda)

Stored (co-)energy, analytic::

    W(lambda) = int_0^lambda i(l') dl' = -L0 * I_sat^2 * ln cos(lambda / (L0 * I_sat))   (>= 0)

Power balance used for the energy-consistency check::

    int v i dt = int R i^2 dt + W(lambda(t1)) - W(lambda(t0))

Inside the ODE right-hand side a flux linkage at or beyond ``lambda_max`` has no finite current; the
rhs then returns ``nan`` so that the adaptive step controller rejects the trial step (the state
constraint ``|lambda| < lambda_max`` is reported on accepted solutions).

FiniteTimeBlowUp
----------------
``dy/dt = y^2`` with ``y(0) = y0 > 0`` blows up at ``t* = 1/y0``; integrating past ``t*`` must make the
solver fail explicitly (used by the "solver failure is explicit" check).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from aipe_magnetics.common.errors import InvalidInputError, UnsupportedOperationError
from aipe_magnetics.models.contract import ModelParameter
from aipe_magnetics.models.dynamic.excitation import Excitation
from aipe_magnetics.models.dynamic.ode import ODEModel

__all__ = ["LinearRL", "NonlinearInductor", "FiniteTimeBlowUp"]


def _finite_positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise InvalidInputError(f"{name} must be a real number, got {type(value).__name__}", details={"name": name})
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise InvalidInputError(f"{name} must be finite and positive, got {value}", details={"name": name})
    return value


def _finite_nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise InvalidInputError(f"{name} must be a real number, got {type(value).__name__}", details={"name": name})
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise InvalidInputError(f"{name} must be finite and non-negative, got {value}", details={"name": name})
    return value


class LinearRL(ODEModel):
    """Series R-L circuit, state ``i`` (A), excitation ``v(t)`` (V)."""

    state_names = ("i",)
    default_initial_state = {"i": 0.0}  # de-energised circuit; documented physical default
    name = "linear_rl"

    def __init__(self, R: float, L: float, *, i_max: float | None = None):
        self.R = _finite_positive("R", R)
        self.L = _finite_positive("L", L)
        self.i_max = None if i_max is None else _finite_positive("i_max", i_max)
        self.parameters = {
            "R": ModelParameter(name="R", value=self.R, unit="Ohm", description="series resistance"),
            "L": ModelParameter(name="L", value=self.L, unit="H", description="inductance"),
        }
        if self.i_max is not None:
            self.parameters["i_max"] = ModelParameter(name="i_max", value=self.i_max, unit="A",
                                                      description="reported current limit (constraint)")
        super().__init__()

    @property
    def tau(self) -> float:
        return self.L / self.R

    def rhs(self, t: float, y: np.ndarray, u: float, du_dt: float | None) -> list[float]:  # noqa: ARG002
        return [(u - self.R * y[0]) / self.L]

    def outputs(self, t: np.ndarray, y: np.ndarray, u: np.ndarray) -> dict[str, np.ndarray]:  # noqa: ARG002
        i = y[0]
        return {"v": np.asarray(u, dtype=float), "p_in": u * i, "p_R": self.R * i * i, "w_L": 0.5 * self.L * i * i}

    def constraints(self, t: np.ndarray, y: np.ndarray) -> list[str]:  # noqa: ARG002
        if self.i_max is None:
            return []
        peak = float(np.max(np.abs(y[0])))
        if peak > self.i_max:
            idx = int(np.argmax(np.abs(y[0])))
            return [f"|i| = {peak:.6g} A exceeds i_max = {self.i_max:.6g} A at t = {float(t[idx]):.6g} s"]
        return []

    def closed_form(self, excitation: Excitation, t: Any, i0: float = 0.0) -> np.ndarray:
        """Analytic ``i(t)`` for constant or sinusoidal excitations (``t`` measured from 0)."""
        if not isinstance(excitation, Excitation):
            raise InvalidInputError("excitation must be an Excitation")
        tt = np.asarray(t, dtype=float)
        if not np.all(np.isfinite(tt)):
            raise InvalidInputError("t must be finite")
        if isinstance(i0, bool) or not math.isfinite(float(i0)):
            raise InvalidInputError("i0 must be a finite number")
        i0 = float(i0)
        tau = self.tau
        if excitation.kind == "constant":
            v = excitation.parameters["value"]
            return v / self.R + (i0 - v / self.R) * np.exp(-tt / tau)
        if excitation.kind == "sinusoidal":
            p = excitation.parameters
            w = 2.0 * math.pi * p["frequency"]
            z = math.hypot(self.R, w * self.L)
            theta = math.atan2(w * self.L, self.R)

            def particular(x: np.ndarray) -> np.ndarray:
                return p["offset"] / self.R + (p["amplitude"] / z) * np.sin(w * x + p["phase"] - theta)

            return particular(tt) + (i0 - float(particular(np.asarray(0.0)))) * np.exp(-tt / tau)
        raise UnsupportedOperationError(f"no closed form for excitation kind {excitation.kind!r}")


class NonlinearInductor(ODEModel):
    """Saturating inductor with flux-linkage state ``lambda`` (V*s), excitation ``v(t)`` (V)."""

    state_names = ("lambda",)
    default_initial_state = {"lambda": 0.0}  # demagnetised / zero-current start; documented default
    name = "nonlinear_inductor"

    def __init__(self, L0: float, I_sat: float, R: float):
        self.L0 = _finite_positive("L0", L0)
        self.I_sat = _finite_positive("I_sat", I_sat)
        self.R = _finite_nonnegative("R", R)
        self.lambda_scale = self.L0 * self.I_sat
        self.lambda_max = 0.5 * math.pi * self.lambda_scale
        self.parameters = {
            "L0": ModelParameter(name="L0", value=self.L0, unit="H", description="small-signal inductance at i=0"),
            "I_sat": ModelParameter(name="I_sat", value=self.I_sat, unit="A", description="saturation current scale"),
            "R": ModelParameter(name="R", value=self.R, unit="Ohm", description="winding resistance"),
        }
        super().__init__()

    # -- constitutive relation ------------------------------------------------------------------
    def current(self, flux_linkage: Any) -> np.ndarray:
        lam = np.asarray(flux_linkage, dtype=float)
        x = lam / self.lambda_scale
        with np.errstate(invalid="ignore"):
            out = np.where(np.abs(x) < 0.5 * math.pi, self.I_sat * np.tan(x), np.nan)
        return out

    def flux_linkage(self, current: Any) -> np.ndarray:
        return self.lambda_scale * np.arctan(np.asarray(current, dtype=float) / self.I_sat)

    def stored_energy(self, flux_linkage: Any) -> np.ndarray:
        lam = np.asarray(flux_linkage, dtype=float)
        x = lam / self.lambda_scale
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(np.abs(x) < 0.5 * math.pi, -self.lambda_scale * self.I_sat * np.log(np.cos(x)), np.nan)
        return out

    # -- ODE contract ---------------------------------------------------------------------------
    def rhs(self, t: float, y: np.ndarray, u: float, du_dt: float | None) -> list[float]:  # noqa: ARG002
        x = y[0] / self.lambda_scale
        if abs(x) >= 0.5 * math.pi:
            return [math.nan]  # no finite current beyond lambda_max; trial step is rejected by the controller
        return [u - self.R * self.I_sat * math.tan(x)]

    def outputs(self, t: np.ndarray, y: np.ndarray, u: np.ndarray) -> dict[str, np.ndarray]:  # noqa: ARG002
        i = self.current(y[0])
        return {"i": i, "v": np.asarray(u, dtype=float), "p_in": u * i, "p_R": self.R * i * i,
                "w": self.stored_energy(y[0])}

    def constraints(self, t: np.ndarray, y: np.ndarray) -> list[str]:  # noqa: ARG002
        peak = float(np.max(np.abs(y[0])))
        if peak >= self.lambda_max:
            return [f"|lambda| = {peak:.6g} V*s reaches lambda_max = {self.lambda_max:.6g} V*s (no finite current)"]
        return []


class FiniteTimeBlowUp(ODEModel):
    """``dy/dt = y^2``: finite-time blow-up at ``t = 1/y0`` (solver-failure test case)."""

    state_names = ("y",)
    default_initial_state = {"y": 1.0}
    name = "finite_time_blow_up"

    def __init__(self) -> None:
        self.parameters = {}
        super().__init__()

    def rhs(self, t: float, y: np.ndarray, u: float, du_dt: float | None) -> list[float]:  # noqa: ARG002
        return [y[0] * y[0]]

    @staticmethod
    def exact(t: Any, y0: float = 1.0) -> np.ndarray:
        tt = np.asarray(t, dtype=float)
        return 1.0 / (1.0 / y0 - tt)
