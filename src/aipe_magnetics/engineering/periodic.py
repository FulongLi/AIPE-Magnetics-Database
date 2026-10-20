"""Periodic piecewise-linear waveforms with exact RMS, mean and Fourier coefficients.

Converter currents (DAB trapezoidal currents, inductor ripple, PWM currents) are piecewise linear over one
period. :class:`PiecewiseLinear` stores breakpoints ``(t_k / T, value_k)`` and evaluates, *in closed form*:

* mean and RMS (exact integration of the linear segments),
* complex Fourier coefficients ``c_n = (1/T) integral x(t) exp(-j 2 pi n t / T) dt`` per segment
  (``integral (a + b t) exp(-j w t) dt`` analytically), from which the amplitude of harmonic ``n`` is
  ``2 |c_n|`` (``n >= 1``),
* the energy fraction captured by the first ``n_max`` harmonics (used to justify a truncation).

Used by the harmonic winding-loss models (Dowell per harmonic, finite-element harmonic solves) so that the
loss of non-sinusoidal currents is computed from their actual spectrum instead of an RMS-at-fundamental
assumption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aipe_magnetics.common.errors import InvalidInputError


@dataclass(frozen=True)
class PiecewiseLinear:
    """Periodic waveform through ``points`` ``[(t/T, value), ...]`` with ``t`` from 0 to 1 and ``value(0) == value(1)``."""

    points: tuple[tuple[float, float], ...]
    unit: str = "A"
    label: str = ""

    def __post_init__(self) -> None:
        pts = self.points
        if len(pts) < 2:
            raise InvalidInputError("a periodic waveform needs at least two breakpoints")
        if abs(pts[0][0]) > 1e-12 or abs(pts[-1][0] - 1.0) > 1e-12:
            raise InvalidInputError("breakpoints must start at t/T = 0 and end at t/T = 1")
        if any(b[0] < a[0] for a, b in zip(pts, pts[1:])):
            raise InvalidInputError("breakpoint times must not decrease")
        if abs(pts[0][1] - pts[-1][1]) > 1e-9 * max(1.0, max(abs(v) for _, v in pts)):
            raise InvalidInputError("waveform must be periodic: value(0) == value(T)")

    @classmethod
    def from_lists(cls, t: list[float], v: list[float], unit: str = "A", label: str = "") -> "PiecewiseLinear":
        return cls(points=tuple((float(a), float(b)) for a, b in zip(t, v)), unit=unit, label=label)

    def segments(self):
        for (t0, v0), (t1, v1) in zip(self.points, self.points[1:]):
            if t1 > t0:
                yield t0, t1, v0, v1

    def mean(self) -> float:
        return float(sum(0.5 * (v0 + v1) * (t1 - t0) for t0, t1, v0, v1 in self.segments()))

    def rms(self) -> float:
        s = sum((v0 * v0 + v0 * v1 + v1 * v1) / 3.0 * (t1 - t0) for t0, t1, v0, v1 in self.segments())
        return math.sqrt(max(s, 0.0))

    def ac_rms(self) -> float:
        return math.sqrt(max(self.rms() ** 2 - self.mean() ** 2, 0.0))

    def peak(self) -> float:
        return max(abs(v) for _, v in self.points)

    def peak_to_peak(self) -> float:
        vals = [v for _, v in self.points]
        return max(vals) - min(vals)

    def value(self, t: np.ndarray | float) -> np.ndarray:
        tt = np.mod(np.asarray(t, dtype=float), 1.0)
        ts = np.array([p[0] for p in self.points])
        vs = np.array([p[1] for p in self.points])
        return np.interp(tt, ts, vs)

    def fourier(self, n: int) -> complex:
        """Complex coefficient ``c_n`` (period normalised to 1)."""
        if n == 0:
            return complex(self.mean())
        w = 2.0 * math.pi * n
        c = 0j
        for t0, t1, v0, v1 in self.segments():
            dt = t1 - t0
            b = (v1 - v0) / dt
            # integral_{t0}^{t1} (v0 + b (t - t0)) e^{-j w t} dt
            e0, e1 = np.exp(-1j * w * t0), np.exp(-1j * w * t1)
            c += (v0 * e0 - v1 * e1) / (1j * w) + b * (e1 - e0) / (w * w)
        return complex(c)

    def harmonic_amplitudes(self, n_max: int) -> list[tuple[int, float, float]]:
        """``[(n, amplitude, phase_rad), ...]`` for n = 1..n_max (amplitude = 2|c_n|)."""
        out = []
        for n in range(1, n_max + 1):
            c = self.fourier(n)
            out.append((n, 2.0 * abs(c), float(np.angle(c))))
        return out

    def harmonic_energy_fraction(self, n_max: int) -> float:
        """Share of the AC mean-square value carried by harmonics 1..n_max (Parseval)."""
        ac2 = self.ac_rms() ** 2
        if ac2 == 0:
            return 1.0
        captured = sum(a * a / 2.0 for _, a, _ in self.harmonic_amplitudes(n_max))
        return float(min(1.0, captured / ac2))

    def scaled(self, factor: float, unit: str | None = None, label: str | None = None) -> "PiecewiseLinear":
        return PiecewiseLinear(points=tuple((t, v * factor) for t, v in self.points), unit=unit or self.unit, label=label or self.label)

    def to_json(self, period: float | None = None) -> dict:
        out = {"breakpoints": [[t, v] for t, v in self.points], "unit": self.unit, "label": self.label, "rms": self.rms(),
               "mean": self.mean(), "peak": self.peak(), "peak_to_peak": self.peak_to_peak()}
        if period is not None:
            out["period_s"] = period
        return out


def triangular(peak_to_peak: float, duty: float = 0.5, dc: float = 0.0, unit: str = "A", label: str = "") -> PiecewiseLinear:
    """Triangle rising for ``duty`` of the period, centred on ``dc``."""
    if not (0 < duty < 1):
        raise InvalidInputError("duty must be in (0, 1)")
    lo, hi = dc - peak_to_peak / 2, dc + peak_to_peak / 2
    return PiecewiseLinear(points=((0.0, lo), (duty, hi), (1.0, lo)), unit=unit, label=label)


def square(amplitude: float, duty: float = 0.5, unit: str = "V", label: str = "") -> PiecewiseLinear:
    """Bipolar square wave approximated with 1e-6 T transitions (exact enough for RMS/harmonic use)."""
    e = 1e-6
    return PiecewiseLinear(points=((0.0, -amplitude), (e, amplitude), (duty, amplitude), (duty + e, -amplitude), (1.0, -amplitude)),
                           unit=unit, label=label)


__all__ = ["PiecewiseLinear", "square", "triangular"]
