"""ODE solution container and energy / loop utilities.

:class:`ODESolution` holds the time grid ``t``, the state trajectory ``y`` (``n_states x n_t``), the
excitation samples ``u``, model-defined derived ``outputs``, solver ``diagnostics`` and the list of
physical ``constraint_violations`` (reported, never clipped).  Optional dense output allows
interpolation at arbitrary times.

Energy conventions
------------------
* :func:`trapezoid` / :func:`cumulative_trapezoid` integrate sampled signals with the composite
  trapezoidal rule (second-order accurate on smooth data).
* :func:`cycle_energy_density` evaluates the closed-loop integral

      w = oint H dB  [J/m^3]

  with the trapezoidal rule, including the closing segment from the last to the first sample.  With
  ``H`` on the abscissa and ``B`` on the ordinate a dissipative hysteresis loop is traversed
  counter-clockwise (ascending branch below the descending branch), so ``w > 0`` is the energy
  absorbed by the material per unit volume per cycle (Green's theorem: ``oint x dy = +Area`` for a
  counter-clockwise contour).  ``oint B dH = -w``.
* :func:`interpolated_crossings` locates level crossings of a sampled signal by linear interpolation;
  it is used for remanence (``B`` at ``H = 0``) and coercivity (``H`` at ``B = 0``).  A crossing
  requires a *strict sign change*: a tangency (``[1, 0, 1]``), a run of level samples at either end
  of the record and identically-level data are not crossings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from aipe_magnetics.common.errors import InvalidInputError, UnsupportedOperationError

__all__ = ["ODESolution", "trapezoid", "cumulative_trapezoid", "cycle_energy_density", "loop_closure_error",
           "interpolated_crossings"]

_np_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _as_1d(name: str, data: Any, *, min_size: int = 2) -> np.ndarray:
    try:
        arr = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as error:
        raise InvalidInputError(f"{name} must be numeric: {error}") from None
    if arr.ndim != 1:
        raise InvalidInputError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size < min_size:
        raise InvalidInputError(f"{name} needs at least {min_size} samples, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise InvalidInputError(f"{name} contains non-finite values")
    return arr


def trapezoid(y: Any, x: Any) -> float:
    """Composite trapezoidal integral of ``y`` over ``x`` (``x`` strictly increasing)."""
    yy, xx = _as_1d("y", y), _as_1d("x", x)
    if yy.shape != xx.shape:
        raise InvalidInputError(f"y and x must have the same length, got {yy.size} and {xx.size}")
    if not np.all(np.diff(xx) > 0):
        raise InvalidInputError("x must be strictly increasing")
    return float(_np_trapezoid(yy, xx))


def cumulative_trapezoid(y: Any, x: Any) -> np.ndarray:
    """Running trapezoidal integral with ``result[0] = 0`` (same length as ``x``)."""
    yy, xx = _as_1d("y", y), _as_1d("x", x)
    if yy.shape != xx.shape:
        raise InvalidInputError(f"y and x must have the same length, got {yy.size} and {xx.size}")
    if not np.all(np.diff(xx) > 0):
        raise InvalidInputError("x must be strictly increasing")
    increments = 0.5 * (yy[1:] + yy[:-1]) * np.diff(xx)
    return np.concatenate([[0.0], np.cumsum(increments)])


def cycle_energy_density(H: Any, B: Any, *, close: bool = True) -> float:
    """Closed-loop integral ``oint H dB`` (J/m^3) of a sampled loop, see module docstring for the sign."""
    h, b = _as_1d("H", H, min_size=3), _as_1d("B", B, min_size=3)
    if h.shape != b.shape:
        raise InvalidInputError(f"H and B must have the same length, got {h.size} and {b.size}")
    if close:
        h = np.concatenate([h, h[:1]])
        b = np.concatenate([b, b[:1]])
    return float(np.sum(0.5 * (h[1:] + h[:-1]) * np.diff(b)))


def loop_closure_error(x: Any) -> float:
    """``|x[-1] - x[0]| / max|x|`` (0 when the signal is identically zero): how well a loop closes."""
    arr = _as_1d("x", x)
    scale = float(np.max(np.abs(arr)))
    if scale == 0.0:
        return 0.0
    return float(abs(arr[-1] - arr[0]) / scale)


def interpolated_crossings(x: Any, y: Any, level: float = 0.0) -> list[dict[str, float]]:
    """Points where ``x`` crosses ``level``; returns ``{'index', 'x', 'y', 'direction'}`` per crossing.

    A crossing is a *strict sign change* of ``x - level``: between two consecutive samples of opposite
    sign the crossing index is linearly interpolated, and a run of samples lying exactly on the level
    counts as one crossing (at the middle of the run) only when the nearest off-level samples on both
    sides have opposite signs.  A tangency, a level run touching either end of the record and
    identically-level data yield no crossing.  ``y`` is linearly interpolated at the crossing index;
    ``direction`` is ``+1`` when ``x`` increases through the level and ``-1`` when it decreases.
    """
    xx, yy = _as_1d("x", x), _as_1d("y", y)
    if xx.shape != yy.shape:
        raise InvalidInputError("x and y must have the same length")
    level = float(level)
    if not math.isfinite(level):
        raise InvalidInputError("level must be finite")
    d = xx - level
    sign = np.sign(d)
    n = d.size
    out: list[dict[str, float]] = []
    i = 0
    while i < n:
        if sign[i] == 0.0:
            j = i
            while j < n and sign[j] == 0.0:
                j += 1
            if i > 0 and j < n and sign[i - 1] * sign[j] < 0.0:
                mid = 0.5 * (i + j - 1)
                out.append({"index": float(mid), "x": level, "y": float(np.interp(mid, np.arange(n), yy)),
                            "direction": 1.0 if sign[j] > 0.0 else -1.0})
            i = j
            continue
        if i + 1 < n and sign[i] * sign[i + 1] < 0.0:
            frac = float(d[i] / (d[i] - d[i + 1]))
            out.append({"index": float(i + frac), "x": level, "y": float(yy[i] + frac * (yy[i + 1] - yy[i])),
                        "direction": 1.0 if d[i + 1] > 0.0 else -1.0})
        i += 1
    return out


@dataclass
class ODESolution:
    """Result of :func:`aipe_magnetics.models.dynamic.ode.solve` (immutable by convention)."""

    t: np.ndarray
    y: np.ndarray
    u: np.ndarray
    state_names: tuple[str, ...]
    outputs: dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    constraint_violations: list[str] = field(default_factory=list)
    dense: Callable[[np.ndarray], np.ndarray] | None = field(default=None, repr=False)

    # -- access ---------------------------------------------------------------------------------
    @property
    def n_states(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.t.size)

    @property
    def t_span(self) -> tuple[float, float]:
        return float(self.t[0]), float(self.t[-1])

    @property
    def final_state(self) -> np.ndarray:
        return np.array(self.y[:, -1], dtype=float)

    @property
    def initial_state(self) -> np.ndarray:
        return np.array(self.y[:, 0], dtype=float)

    def state(self, name: str) -> np.ndarray:
        try:
            idx = self.state_names.index(name)
        except ValueError:
            raise InvalidInputError(f"unknown state {name!r}; states are {self.state_names}") from None
        return self.y[idx]

    def output(self, name: str) -> np.ndarray:
        try:
            return self.outputs[name]
        except KeyError:
            raise InvalidInputError(f"unknown output {name!r}; outputs are {tuple(self.outputs)}") from None

    def interpolate(self, t: Any) -> np.ndarray:
        """State at arbitrary times inside the solved span (needs ``dense_output=True``)."""
        if self.dense is None:
            raise UnsupportedOperationError("dense output was not requested (SolverConfig.dense_output=False)")
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        lo, hi = self.t_span
        tol = 1e-12 * max(1.0, abs(lo), abs(hi))
        if np.any(tt < lo - tol) or np.any(tt > hi + tol):
            raise InvalidInputError(f"interpolation times must lie inside [{lo:g}, {hi:g}]")
        return np.asarray(self.dense(np.clip(tt, lo, hi)), dtype=float)

    def __call__(self, t: Any) -> np.ndarray:
        return self.interpolate(t)

    @property
    def satisfies_constraints(self) -> bool:
        return not self.constraint_violations

    def summary(self) -> dict[str, Any]:
        """JSON-safe summary (no arrays)."""
        return {
            "state_names": list(self.state_names),
            "n_points": self.n_points,
            "t_span": list(self.t_span),
            "initial_state": [float(v) for v in self.initial_state],
            "final_state": [float(v) for v in self.final_state],
            "state_abs_max": [float(np.max(np.abs(row))) for row in self.y],
            "outputs": sorted(self.outputs),
            "constraint_violations": list(self.constraint_violations),
            "diagnostics": {k: v for k, v in self.diagnostics.items()},
        }
