"""Fast, vectorised core-loss evaluation for design loops and field post-processing.

The engineering capability :func:`aipe_magnetics.engineering.core_loss.estimate_core_loss` returns a full
result envelope per call. Design optimisation and finite-element post-processing evaluate the same
material model thousands of times (one value per mesh element, per candidate, per electrothermal
iteration), so this module evaluates the *same equations* without the envelope:

* sinusoidal flux: classical Steinmetz ``Pv = k f^alpha Bpk^beta``;
* piecewise-linear flux (triangular, symmetric trapezoid, general piecewise linear): iGSE in its exact
  closed form ``Pv = k_i (dB)^(beta-alpha) f^alpha sum_i |dB_i|^alpha x_i^(1-alpha)`` with ``k_i`` from
  :func:`aipe_magnetics.models.empirical.igse.igse_coefficient` (the same production coefficient).

Equality with the envelope path is asserted by ``tests/database/test_loss_evaluator.py`` and by the
verification check ``database.loss-evaluator-matches-engine``.

Temperature handling
--------------------
:meth:`LossEvaluator.loss_density` uses the parameter set valid at the temperature (the catalogue rule:
fit temperature within 5 K or a stated range). With ``interpolate_temperature=True`` (the design default)
every temperature inside the span of the fit temperatures is evaluated with the two bracketing sets and the loss
density is interpolated **linearly in temperature** (exact at each fit temperature, continuous in between) - an explicit, reported approximation (ferrite loss versus
temperature is smooth and U-shaped; the catalogue sets are 10-25 K apart). Temperatures outside the span of
the fit temperatures are rejected (``policy='reject'``) or evaluated with the nearest set and flagged
``temperature_extrapolated`` (``policy='warn'``). Frequency / flux-density coverage of the chosen sets is
checked the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from aipe_magnetics.common.errors import DataNotFoundError, InvalidInputError, OutOfDomainError
from aipe_magnetics.data.records import SteinmetzParameterSet

FIT_TEMPERATURE_TOLERANCE_K = 5.0


@dataclass(frozen=True)
class FluxShape:
    """Normalised periodic flux shape: segments ``(fraction_of_period, fraction_of_peak_to_peak)``.

    ``sinusoidal=True`` selects the classical Steinmetz path (segments ignored).
    """

    kind: str
    segments: tuple[tuple[float, float], ...] = ()
    sinusoidal: bool = False

    @classmethod
    def sinusoid(cls) -> "FluxShape":
        return cls(kind="sinusoidal", sinusoidal=True)

    @classmethod
    def triangular(cls, duty: float = 0.5) -> "FluxShape":
        if not (0 < duty < 1):
            raise InvalidInputError(f"triangular duty must be in (0, 1), got {duty}")
        return cls(kind="triangular", segments=((duty, 1.0), (1.0 - duty, -1.0)))

    @classmethod
    def trapezoidal(cls, duty_rise: float) -> "FluxShape":
        """Symmetric trapezoid (rise, flat, fall, flat) with rising fraction ``duty_rise`` (<= 0.5)."""
        if not (0 < duty_rise <= 0.5):
            raise InvalidInputError(f"trapezoidal duty_rise must be in (0, 0.5], got {duty_rise}")
        flat = 0.5 - duty_rise
        segs = [(duty_rise, 1.0), (flat, 0.0), (duty_rise, -1.0), (flat, 0.0)] if flat > 0 else [(duty_rise, 1.0), (duty_rise, -1.0)]
        return cls(kind="trapezoidal", segments=tuple(segs))

    @classmethod
    def piecewise_linear(cls, breakpoints: Sequence[tuple[float, float]]) -> "FluxShape":
        """From normalised breakpoints ``(t/T, B/B_pk)`` covering ``[0, 1]``; peak-to-peak normalised internally."""
        pts = [(float(t), float(b)) for t, b in breakpoints]
        if len(pts) < 3 or abs(pts[0][0]) > 1e-12 or abs(pts[-1][0] - 1.0) > 1e-12:
            raise InvalidInputError("breakpoints must start at t=0, end at t=1 and have at least 3 points")
        if abs(pts[0][1] - pts[-1][1]) > 1e-9:
            raise InvalidInputError("piecewise-linear flux must be periodic (B(0) == B(T))")
        values = [b for _, b in pts]
        pp = max(values) - min(values)
        if pp <= 0:
            raise InvalidInputError("piecewise-linear flux has zero peak-to-peak")
        segs = []
        for (t0, b0), (t1, b1) in zip(pts, pts[1:]):
            if t1 <= t0:
                raise InvalidInputError("breakpoint times must increase strictly")
            segs.append((t1 - t0, (b1 - b0) / pp))
        return cls(kind="piecewise_linear", segments=tuple(segs))

    def igse_shape_factor(self, alpha: float) -> float:
        """``sum_i |dB_i/dB|^alpha x_i^(1-alpha)`` (dimensionless)."""
        return float(sum(abs(db) ** alpha * x ** (1.0 - alpha) for x, db in self.segments if db != 0.0))


@dataclass
class LossEvaluation:
    """Loss density with the provenance of the evaluation."""

    loss_density: float | np.ndarray
    sets: list[str]
    weights: list[float]
    model: Literal["steinmetz", "igse"]
    temperature: float
    interpolated: bool = False
    warnings: list[str] = field(default_factory=list)


def _set_temperature(s: SteinmetzParameterSet) -> float | None:
    return None if s.temperature is None else s.temperature.magnitude("K")


def _covers(rng, value: float, unit: str) -> bool:
    if rng is None:
        return True
    from aipe_magnetics.common.units import units
    lo = None if rng.min is None else units.convert(float(rng.min), rng.unit, unit)
    hi = None if rng.max is None else units.convert(float(rng.max), rng.unit, unit)
    return (lo is None or value >= lo * (1 - 1e-9)) and (hi is None or value <= hi * (1 + 1e-9))


class LossEvaluator:
    """Core-loss density of one material with temperature interpolation between Steinmetz parameter sets."""

    def __init__(self, material_id: str, sets: Sequence[SteinmetzParameterSet]):
        if not sets:
            raise DataNotFoundError(f"{material_id} has no Steinmetz parameter set; core loss cannot be evaluated")
        self.material_id = material_id
        self.sets = list(sets)

    @classmethod
    def for_material(cls, query: str, catalog=None) -> "LossEvaluator":
        from aipe_magnetics.database.accessors import material_data
        data = material_data(query, catalog)
        return cls(data.id, data.steinmetz_sets)

    # -- set selection ----------------------------------------------------------------------------------
    def _frequency_candidates(self, frequency: float) -> list[SteinmetzParameterSet]:
        covering = [s for s in self.sets if _set_temperature(s) is not None and _covers(s.frequency_range, frequency, "Hz")]
        return covering

    def select(self, frequency: float, temperature: float, *, interpolate_temperature: bool = True,
               policy: Literal["reject", "warn"] = "warn") -> tuple[list[SteinmetzParameterSet], list[float], list[str]]:
        """``(sets, weights, warnings)`` for the conditions (weights sum to 1)."""
        warnings: list[str] = []
        candidates = self._frequency_candidates(frequency)
        if not candidates:
            message = f"{self.material_id}: no Steinmetz parameter set covers {frequency:g} Hz"
            if policy == "reject":
                raise OutOfDomainError(message, details={"frequency": frequency})
            warnings.append(message + "; the set closest in temperature is used outside its frequency range")
            candidates = [s for s in self.sets if _set_temperature(s) is not None]
            if not candidates:
                raise DataNotFoundError(f"{self.material_id}: no parameter set states a fit temperature")
        temps = sorted({_set_temperature(s) for s in candidates})
        exact = sorted((abs(_set_temperature(s) - temperature), i, s) for i, s in enumerate(candidates)
                       if abs(_set_temperature(s) - temperature) <= FIT_TEMPERATURE_TOLERANCE_K)
        if interpolate_temperature and temps[0] <= temperature <= temps[-1]:
            # continuous piecewise-linear interpolation over the whole fitted span (exact at every fit temperature); the
            # catalogue +-5 K rule would make the loss discontinuous in temperature (electrothermal iterations oscillate)
            on_fit = [s for s in candidates if abs(_set_temperature(s) - temperature) < 1e-9]
            if on_fit:
                return [on_fit[0]], [1.0], warnings
        elif exact:  # catalogue rule: a set fitted within 5 K wins (closest first)
            return [exact[0][2]], [1.0], warnings
        if interpolate_temperature and temps[0] < temperature < temps[-1]:
            lo_t = max(t for t in temps if t < temperature)
            hi_t = min(t for t in temps if t > temperature)
            lo = next(s for s in candidates if _set_temperature(s) == lo_t)
            hi = next(s for s in candidates if _set_temperature(s) == hi_t)
            w = (temperature - lo_t) / (hi_t - lo_t)
            if lo.basis != hi.basis:
                warnings.append(f"temperature interpolation between sets of different basis ({lo.id}: {lo.basis}, {hi.id}: {hi.basis})")
            return [lo, hi], [1.0 - w, w], warnings
        if temps[0] < temperature < temps[-1]:  # inside the span, interpolation disabled
            message = (f"{self.material_id}: no parameter set is fitted within {FIT_TEMPERATURE_TOLERANCE_K:g} K of "
                       f"{temperature - 273.15:.1f} degC and temperature interpolation is disabled")
        else:
            message = (f"{self.material_id}: temperature {temperature - 273.15:.1f} degC is outside the fit temperatures "
                       f"{temps[0] - 273.15:.0f}..{temps[-1] - 273.15:.0f} degC of the parameter sets covering {frequency:g} Hz")
        if policy == "reject":
            raise OutOfDomainError(message, details={"temperature": temperature, "fit_temperatures": temps})
        nearest = min(candidates, key=lambda s: abs(_set_temperature(s) - temperature))
        warnings.append(message + f"; temperature_extrapolated: set {nearest.id} is used")
        return [nearest], [1.0], warnings

    # -- evaluation -------------------------------------------------------------------------------------
    @staticmethod
    def _single(s: SteinmetzParameterSet, frequency: float, peak: np.ndarray, shape: FluxShape) -> np.ndarray:
        k, a, b = s.k.magnitude("W/m^3"), s.alpha, s.beta
        if shape.sinusoidal:
            return k * frequency ** a * np.power(peak, b)
        from aipe_magnetics.models.empirical.igse import igse_coefficient
        ki = igse_coefficient(k, a, b)
        return ki * np.power(2.0 * peak, b) * frequency ** a * shape.igse_shape_factor(a)

    def loss_density(self, frequency: float, flux_density_peak: float | np.ndarray, temperature: float, shape: FluxShape, *,
                     interpolate_temperature: bool = True, policy: Literal["reject", "warn"] = "warn") -> LossEvaluation:
        """Loss density (W/m^3) for scalar or array ``flux_density_peak`` (T; half the peak-to-peak swing)."""
        if not (frequency > 0 and math.isfinite(frequency)):
            raise InvalidInputError(f"frequency must be positive, got {frequency}")
        if not (temperature > 0 and math.isfinite(temperature)):
            raise InvalidInputError(f"temperature must be positive (K), got {temperature}")
        peak = np.asarray(flux_density_peak, dtype=float)
        if np.any(peak < 0) or not np.all(np.isfinite(peak)):
            raise InvalidInputError("flux_density_peak must be finite and non-negative")
        sets, weights, warnings = self.select(frequency, temperature, interpolate_temperature=interpolate_temperature, policy=policy)
        bmax = float(np.max(peak)) if peak.size else 0.0
        bmin_pos = float(np.min(peak[peak > 0])) if np.any(peak > 0) else 0.0
        for s in sets:
            if s.flux_density_range is not None and bmax > 0:
                hi_ok = _covers(s.flux_density_range, bmax, "T")
                if not hi_ok:
                    warnings.append(f"{s.id}: peak flux density {bmax:.3f} T exceeds the set's fitted range")
                if s.flux_density_range.min is not None and peak.ndim == 0 and 0 < bmin_pos and not _covers(s.flux_density_range, bmin_pos, "T"):
                    warnings.append(f"{s.id}: peak flux density {bmin_pos:.3f} T is below the set's fitted range")
        value = sum(w * self._single(s, frequency, peak, shape) for s, w in zip(sets, weights))
        out = value if peak.ndim else float(value)
        return LossEvaluation(loss_density=out, sets=[s.id for s in sets], weights=list(weights),
                              model="steinmetz" if shape.sinusoidal else "igse", temperature=temperature,
                              interpolated=len(sets) > 1, warnings=warnings)


__all__ = ["FluxShape", "LossEvaluation", "LossEvaluator"]
