"""Bertotti loss separation (statistical theory of losses).

Model (G. Bertotti, IEEE Trans. Magn. 24 (1988) 621-630), time-average volumetric loss for a
sinusoidal flux density of peak value ``Bpk`` (T) and frequency ``f`` (Hz)::

    Pv = k_h * f * Bpk^beta_h  +  k_c * f^2 * Bpk^2  +  k_e * f^1.5 * Bpk^1.5      [W/m^3]
         ----- hysteresis -----   -- classical eddy --   ------ excess ------

* ``k_h`` [W/(m^3 Hz T^beta_h)] and ``beta_h`` [1]: quasi-static hysteresis energy per cycle,
  ``W_h = k_h Bpk^beta_h`` (J/m^3), proportional to frequency.
* ``k_c`` [W/(m^3 Hz^2 T^2)]: classical eddy-current coefficient.
* ``k_e`` [W/(m^3 Hz^1.5 T^1.5)]: excess (anomalous) loss coefficient of the statistical theory
  (``k_e = 8 sqrt(sigma G S V0)`` in Bertotti's notation for a sinusoidal induction).

All coefficients are empirical fits for one material at one temperature; the model is therefore
*parametrised* and has no built-in material or temperature dependence.

Classical eddy-current coefficient from lamination geometry
-----------------------------------------------------------
For a lamination of thickness ``d`` (m) and conductivity ``sigma`` (S/m) with a *uniform* sinusoidal
flux density ``B(t) = Bpk sin(wt)`` across the thickness (thin lamination, no skin effect, in-plane
flux, lamination width >> thickness, no inter-laminar currents), Faraday's law gives the in-plane
electric field ``E(x, t) = -x dB/dt`` at distance ``x`` from the mid-plane.  Averaging the local
Joule loss density ``sigma E^2`` over the thickness and over one period::

    P_c = (1/d) int_{-d/2}^{d/2} sigma <E^2> dx = sigma d^2 w^2 Bpk^2 / 24 = pi^2 sigma d^2 f^2 Bpk^2 / 6

hence ``k_c = pi^2 sigma d^2 / 6`` (:meth:`BertottiLossModel.classical_eddy_coefficient`).  The
assumptions make ``k_c`` an upper-bound-type classical estimate valid while the skin depth is much
larger than ``d``.

Validation and limits
---------------------
Coefficients must be finite and non-negative, ``beta_h`` finite and positive (Python or numpy real
scalars; booleans are rejected).  ``f`` and ``Bpk`` must be finite and non-negative; zero excitation
yields exactly zero loss.  Overflow (non-finite result) raises :class:`NumericalError`.  Inputs
outside the declared validity domain are rejected or flagged according to ``policy``.  Lamination
provenance (``lamination={'conductivity': sigma, 'thickness': d}``) is validated: both keys are
required, positive and finite, no other keys are accepted, and ``k_c`` must equal
``pi^2 sigma d^2 / 6`` (relative tolerance 1e-9) because the assumption attached to the result
states exactly that; use :meth:`BertottiLossModel.from_lamination` to derive ``k_c``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from aipe_magnetics import __version__
from aipe_magnetics.common.errors import InvalidInputError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import Assumption, Limitation, ModelIdentity, SourceReference, Warning_
from aipe_magnetics.models.contract import (
    CoreLossModel,
    DomainPolicy,
    ModelOutput,
    ModelParameter,
    ValidityDomain,
    ValidityRange,
)
from aipe_magnetics.models.physics.reference_parameters import BERTOTTI_1988

__all__ = ["BertottiLossModel", "BERTOTTI_MODEL_ID"]

BERTOTTI_MODEL_ID = "aipe.magnetics.model.bertotti"


def _finite(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise InvalidInputError(f"{name} must be a real number, got {type(value).__name__}", details={"name": name})
    value = float(value)
    if not math.isfinite(value):
        raise InvalidInputError(f"{name} must be finite, got {value}", details={"name": name})
    return value


def _lamination(lamination: Any, k_c: float) -> dict[str, float] | None:
    if lamination is None:
        return None
    if not isinstance(lamination, dict):
        raise InvalidInputError("lamination must be a dict {'conductivity': S/m, 'thickness': m} or None",
                                details={"type": type(lamination).__name__})
    missing = sorted({"conductivity", "thickness"} - set(lamination))
    unknown = sorted(set(lamination) - {"conductivity", "thickness"})
    if missing or unknown:
        raise InvalidInputError(f"lamination needs exactly the keys 'conductivity' and 'thickness'; missing {missing}, "
                                f"unknown {unknown}", details={"missing": missing, "unknown": unknown})
    sigma = _positive("lamination['conductivity']", lamination["conductivity"])
    d = _positive("lamination['thickness']", lamination["thickness"])
    expected = math.pi ** 2 * sigma * d * d / 6.0
    if abs(k_c - expected) > 1e-9 * expected:
        raise InvalidInputError(
            f"k_c = {k_c:g} is inconsistent with the lamination data (pi^2 sigma d^2 / 6 = {expected:g}); the "
            "thin-lamination assumption attached to the result would be false - use from_lamination() or drop "
            "the lamination argument", details={"k_c": k_c, "expected": expected})
    return {"conductivity": sigma, "thickness": d}


def _nonnegative(name: str, value: Any) -> float:
    value = _finite(name, value)
    if value < 0.0:
        raise InvalidInputError(f"{name} must be non-negative, got {value}", details={"name": name})
    return value


def _positive(name: str, value: Any) -> float:
    value = _finite(name, value)
    if value <= 0.0:
        raise InvalidInputError(f"{name} must be positive, got {value}", details={"name": name})
    return value


def _range(name: str, bounds: tuple[float, float] | None, unit: str) -> ValidityRange | None:
    if bounds is None:
        return None
    try:
        lo, hi = bounds
    except (TypeError, ValueError):
        raise InvalidInputError(f"{name} must be a (min, max) pair") from None
    lo, hi = _finite(f"{name}[0]", lo), _finite(f"{name}[1]", hi)
    if lo > hi:
        raise InvalidInputError(f"{name} must satisfy min <= max, got ({lo}, {hi})")
    return ValidityRange(min=lo, max=hi, unit=unit)


class BertottiLossModel(CoreLossModel):
    """``Pv = k_h f Bpk^beta_h + k_c f^2 Bpk^2 + k_e f^1.5 Bpk^1.5`` (W/m^3), SI parameters."""

    identity = ModelIdentity(id=BERTOTTI_MODEL_ID, name="Bertotti loss separation", version=__version__,
                             family="physics", method_version="closed-form/1")

    def __init__(self, k_h: float, beta_h: float, k_c: float, k_e: float, *,
                 frequency_range: tuple[float, float] | None = None,
                 flux_density_range: tuple[float, float] | None = None,
                 temperature_range: tuple[float, float] | None = None,
                 material: str | None = None, source: SourceReference | None = None,
                 lamination: dict[str, float] | None = None):
        self.k_h = _nonnegative("k_h", k_h)
        self.beta_h = _positive("beta_h", beta_h)
        self.k_c = _nonnegative("k_c", k_c)
        self.k_e = _nonnegative("k_e", k_e)
        self.material = material
        self.parameter_source = source
        self.lamination = _lamination(lamination, self.k_c)
        ref = source.id if source is not None else None
        self.parameters = [
            ModelParameter(name="k_h", value=self.k_h, unit="W/(m^3*Hz*T^beta_h)",
                           description="hysteresis coefficient (energy per cycle per T^beta_h)", source_ref=ref),
            ModelParameter(name="beta_h", value=self.beta_h, unit="1", description="hysteresis flux-density exponent",
                           source_ref=ref),
            ModelParameter(name="k_c", value=self.k_c, unit="W/(m^3*Hz^2*T^2)",
                           description="classical eddy-current coefficient", source_ref=ref),
            ModelParameter(name="k_e", value=self.k_e, unit="W/(m^3*Hz^1.5*T^1.5)",
                           description="excess (anomalous) loss coefficient", source_ref=ref),
        ]
        self.sources = [BERTOTTI_1988] + ([source] if source is not None else [])
        self.domain = ValidityDomain(
            materials=[material] if material else None,
            excitation=["sinusoidal"],
            frequency=_range("frequency_range", frequency_range, "Hz"),
            flux_density_peak=_range("flux_density_range", flux_density_range, "T"),
            temperature=_range("temperature_range", temperature_range, "K"),
            dc_bias="unsupported",
            requires_history=False,
            numerical_constraints=["coefficients finite and >= 0, beta_h > 0", "f >= 0, Bpk >= 0 (finite)"],
            known_limitations=[
                "sinusoidal flux density only (no waveform-shape correction)",
                "coefficients are a per-material, per-temperature fit; no built-in temperature dependence",
                "no DC-bias or minor-loop effects",
                "classical term assumes uniform flux (no skin effect) when derived from lamination geometry",
            ],
        )

    # -- construction helpers -------------------------------------------------------------------
    @staticmethod
    def classical_eddy_coefficient(conductivity: float, thickness: float) -> float:
        """``k_c = pi^2 sigma d^2 / 6`` for a thin lamination (sigma in S/m, d in m); see module docstring."""
        sigma = _positive("conductivity", conductivity)
        d = _positive("thickness", thickness)
        k_c = math.pi ** 2 * sigma * d * d / 6.0
        if not math.isfinite(k_c):
            raise InvalidInputError("conductivity * thickness^2 overflows", details={"sigma": sigma, "d": d})
        return k_c

    @classmethod
    def from_lamination(cls, k_h: float, beta_h: float, k_e: float, *, conductivity: float, thickness: float,
                        **kwargs: Any) -> "BertottiLossModel":
        k_c = cls.classical_eddy_coefficient(conductivity, thickness)
        return cls(k_h, beta_h, k_c, k_e, lamination={"conductivity": float(conductivity),
                                                      "thickness": float(thickness)}, **kwargs)

    # -- evaluation -----------------------------------------------------------------------------
    def components(self, frequency: float, flux_density_peak: float) -> dict[str, float]:
        """Hysteresis, classical and excess loss densities (W/m^3) with finiteness checks."""
        f = _nonnegative("frequency", frequency)
        b = _nonnegative("flux_density_peak", flux_density_peak)
        if f == 0.0 or b == 0.0:
            return {"hysteresis": 0.0, "classical": 0.0, "excess": 0.0, "total": 0.0}
        try:
            hyst = self.k_h * f * b ** self.beta_h
            classical = self.k_c * f * f * b * b
            excess = self.k_e * f ** 1.5 * b ** 1.5
        except OverflowError:
            hyst = classical = excess = math.inf
        hyst = self.check_output_finite("hysteresis_loss_density", hyst)
        classical = self.check_output_finite("classical_loss_density", classical)
        excess = self.check_output_finite("excess_loss_density", excess)
        total = self.check_output_finite("loss_density", hyst + classical + excess)
        return {"hysteresis": hyst, "classical": classical, "excess": excess, "total": total}

    def loss_density(self, *, frequency: float, flux_density_peak: float, temperature: float | None = None,
                     policy: DomainPolicy = "reject", **kwargs: Any) -> ModelOutput:
        if kwargs:
            raise InvalidInputError(f"unexpected arguments {sorted(kwargs)}")
        if policy not in ("reject", "warn"):
            raise InvalidInputError(f"policy must be 'reject' or 'warn', got {policy!r}")
        f = _nonnegative("frequency", frequency)
        b = _nonnegative("flux_density_peak", flux_density_peak)
        temp = None if temperature is None else _positive("temperature", temperature)
        check = self.check_domain(frequency=f, flux_density_peak=b, temperature=temp, excitation="sinusoidal")
        if policy == "reject":
            check.raise_if_outside(self.identity.id)
        parts = self.components(f, b)
        values = {
            "loss_density": Quantity(value=parts["total"], unit="W/m^3"),
            "hysteresis_loss_density": Quantity(value=parts["hysteresis"], unit="W/m^3"),
            "classical_loss_density": Quantity(value=parts["classical"], unit="W/m^3"),
            "excess_loss_density": Quantity(value=parts["excess"], unit="W/m^3"),
        }
        if f > 0.0:
            values["energy_per_cycle_density"] = Quantity(value=parts["total"] / f, unit="J/m^3")
        warnings: list[Warning_] = []
        if not check.inside:
            warnings.append(Warning_(code="out_of_domain", message="; ".join(check.violations), severity="warning"))
        if temp is None and self.domain.temperature is not None:
            warnings.append(Warning_(code="temperature_unknown",
                                     message="temperature not supplied; coefficients are a fit at "
                                             f"{self.domain.temperature.describe()}", severity="info"))
        assumptions = [
            Assumption(id="bertotti.sinusoidal", statement="Flux density is sinusoidal with the given peak value.",
                       basis="model_assumption", affects=list(values)),
            Assumption(id="bertotti.fit", statement="Coefficients k_h, beta_h, k_c, k_e are a fit for one material and "
                       "temperature and are used unchanged.", basis="model_assumption", affects=list(values)),
        ]
        if self.lamination:
            assumptions.append(Assumption(
                id="bertotti.thin-lamination",
                statement=(f"k_c = pi^2 sigma d^2 / 6 with sigma = {self.lamination['conductivity']:g} S/m, "
                           f"d = {self.lamination['thickness']:g} m: uniform sinusoidal flux across the thickness, "
                           "no skin effect, no inter-laminar currents."),
                basis="model_assumption", affects=["classical_loss_density", "loss_density"]))
        limitations = [Limitation(code=f"bertotti.limit.{i}", statement=s)
                       for i, s in enumerate(self.domain.known_limitations)]
        return ModelOutput(
            model=self.identity, values=values, domain=check, assumptions=assumptions, limitations=limitations,
            warnings=warnings, sources=list(self.sources),
            diagnostics={"frequency": f, "flux_density_peak": b, "temperature": temp,
                         "fractions": {k: (parts[k] / parts["total"] if parts["total"] > 0 else 0.0)
                                       for k in ("hysteresis", "classical", "excess")}},
        )
