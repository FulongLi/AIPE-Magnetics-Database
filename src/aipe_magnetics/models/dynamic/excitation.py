"""Time-varying excitation ``u(t)`` for the dynamic (ODE) model framework.

An :class:`Excitation` is a validated, deterministic callable ``u(t)`` (scalar or vectorised over a
numpy array of times) with, where it exists analytically, its time derivative ``du/dt`` (models
such as Jiles-Atherton need ``sign(dH/dt)``).  Each constructor documents its exact waveform
definition and conventions:

* ``constant(value)``: ``u(t) = value``; ``du/dt = 0``.
* ``sinusoidal(amplitude, frequency, phase=0, offset=0)``:
  ``u(t) = offset + amplitude * sin(2*pi*frequency*t + phase)``, ``du/dt`` analytic.
* ``triangular(amplitude, frequency, duty=0.5, offset=0)``: period ``T = 1/frequency``; with
  ``tau = t mod T`` the signal rises linearly from ``offset - amplitude`` at ``tau = 0`` to
  ``offset + amplitude`` at ``tau = duty*T`` and falls linearly back to ``offset - amplitude`` at
  ``tau = T``.  ``du/dt`` is piecewise constant (``+2A/(duty T)`` rising, ``-2A/((1-duty) T)``
  falling); at the corners the *right-hand* derivative is returned.
* ``piecewise_linear(times, values, periodic)``: linear interpolation between knots.  Non-periodic
  excitations are defined only on ``[times[0], times[-1]]`` and raise :class:`InvalidInputError`
  outside it (no extrapolation).  Periodic excitations repeat with period ``times[-1] - times[0]``
  and the table must *close*: ``|values[-1] - values[0]| <= closure_tolerance * (max - min)``
  (default ``1e-6``); the last value is then set equal to the first so that the periodic
  continuation is continuous (a table with a value jump at the wrap is rejected, exactly as value
  jumps inside the period are impossible because ``times`` must be strictly increasing).
* ``sampled(times, values, interpolation='linear'|'cubic', periodic=False)``: measured or tabulated
  samples.  ``'linear'`` has a piecewise-constant derivative; ``'cubic'`` uses a
  ``scipy.interpolate.CubicSpline`` (``bc_type='periodic'`` when periodic) with an analytic spline
  derivative.  The same closure rule as for ``piecewise_linear`` applies to periodic tables.  A
  record of samples is a finite, non-periodic signal unless ``periodic=True`` is stated explicitly.
* ``from_waveform(waveform)``: adapter for ``aipe_magnetics.models.waveforms.Waveform`` objects (one
  period of a periodic flux-density waveform): sinusoidal -> ``sinusoidal``, triangular / trapezoidal /
  piecewise-linear -> periodic ``piecewise_linear`` built from the breakpoints, sampled -> periodic
  linear ``sampled``.  Values are carried over unchanged.  The module is imported lazily; when it is
  not installed the adapter raises :class:`UnsupportedOperationError` instead of failing at import
  time.
* ``from_callable(func, derivative=None, period=None)``: arbitrary user function (kind ``'custom'``).
  The callable is evaluated through a guard: a result that is not numeric, has the wrong shape or
  an exception raised by the callable surfaces as :class:`InvalidInputError` naming the time at
  which it happened (a scalar result is broadcast over an array of times).

Breakpoints: excitations whose derivative is discontinuous (triangular, piecewise-linear and
linearly interpolated samples) expose :meth:`Excitation.breakpoints` so that the solver can
restart the integration at every corner instead of stepping across a derivative jump.  Breakpoints
are stored as *absolute* times: for a periodic excitation the corner times ``t_i`` of one period
(``times[0] <= t_i < times[0] + period``) which repeat as ``t_i + k * period``; for a non-periodic
table the interior knot times.

All parameters are validated once at construction: non-finite numbers, non-positive frequencies,
duty cycles outside ``(0, 1)``, non-increasing times, length mismatches and non-closing periodic
tables raise :class:`InvalidInputError`.  Nothing is clipped or silently corrected.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np

from aipe_magnetics.common.errors import AIPEMagneticsError, InvalidInputError, UnsupportedOperationError

__all__ = ["Excitation", "EXCITATION_KINDS", "DEFAULT_CLOSURE_TOLERANCE"]

EXCITATION_KINDS = ("constant", "sinusoidal", "triangular", "piecewise_linear", "sampled", "custom")

#: relative (to the value excursion) mismatch tolerated between the first and last value of a
#: periodic table; the same default as ``models.waveforms.Waveform.sampled``
DEFAULT_CLOSURE_TOLERANCE = 1e-6

_TWO_PI = 2.0 * math.pi


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise InvalidInputError(f"{name} must be a real number, got {type(value).__name__}", details={"name": name})
    value = float(value)
    if not math.isfinite(value):
        raise InvalidInputError(f"{name} must be finite, got {value}", details={"name": name})
    return value


def _positive(name: str, value: Any) -> float:
    value = _finite(name, value)
    if value <= 0.0:
        raise InvalidInputError(f"{name} must be positive, got {value}", details={"name": name})
    return value


def _nonnegative(name: str, value: Any) -> float:
    value = _finite(name, value)
    if value < 0.0:
        raise InvalidInputError(f"{name} must be non-negative, got {value}", details={"name": name})
    return value


def _array(name: str, data: Any) -> np.ndarray:
    try:
        arr = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as error:
        raise InvalidInputError(f"{name} must be a 1-D numeric array: {error}", details={"name": name}) from None
    if arr.ndim != 1:
        raise InvalidInputError(f"{name} must be 1-D, got shape {arr.shape}", details={"name": name})
    if arr.size < 2:
        raise InvalidInputError(f"{name} needs at least two samples, got {arr.size}", details={"name": name})
    if not np.all(np.isfinite(arr)):
        raise InvalidInputError(f"{name} contains non-finite values", details={"name": name})
    return arr


def _describe_time(t: np.ndarray) -> str:
    if t.ndim == 0:
        return f"{float(t):g}"
    if t.size == 0:
        return "[]"
    return f"[{float(t.min()):g} .. {float(t.max()):g}] ({t.size} values)"


class Excitation:
    """Validated time-varying input ``u(t)`` with optional analytic derivative and breakpoints."""

    __slots__ = ("kind", "parameters", "period", "_func", "_deriv", "_breaks", "_domain", "split_hint",
                 "_extrema")

    def __init__(self, kind: str, func: Callable[[np.ndarray], np.ndarray],
                 derivative: Callable[[np.ndarray], np.ndarray] | None = None, *,
                 period: float | None = None, breakpoints: Sequence[float] = (),
                 parameters: dict[str, Any] | None = None, domain: tuple[float, float] | None = None,
                 split_hint: bool = False, extrema: tuple[float, float] | None = None):
        if kind not in EXCITATION_KINDS:
            raise InvalidInputError(f"unknown excitation kind {kind!r}; expected one of {EXCITATION_KINDS}")
        self.kind = kind
        self._func = func
        self._deriv = derivative
        self.period = None if period is None else _positive("period", period)
        self._breaks = np.array(sorted(float(b) for b in breakpoints), dtype=float)
        self.parameters = dict(parameters or {})
        self._domain = domain
        self.split_hint = bool(split_hint)
        self._extrema = extrema

    # -- constructors ---------------------------------------------------------------------------
    @classmethod
    def constant(cls, value: float) -> "Excitation":
        value = _finite("value", value)

        def func(t: np.ndarray) -> np.ndarray:
            return np.full(np.shape(t), value, dtype=float)

        def deriv(t: np.ndarray) -> np.ndarray:
            return np.zeros(np.shape(t), dtype=float)

        return cls("constant", func, deriv, parameters={"value": value}, extrema=(value, value))

    @classmethod
    def sinusoidal(cls, amplitude: float, frequency: float, phase: float = 0.0, offset: float = 0.0) -> "Excitation":
        amplitude = _nonnegative("amplitude", amplitude)
        frequency = _positive("frequency", frequency)
        phase = _finite("phase", phase)
        offset = _finite("offset", offset)
        omega = _TWO_PI * frequency

        def func(t: np.ndarray) -> np.ndarray:
            return offset + amplitude * np.sin(omega * np.asarray(t, dtype=float) + phase)

        def deriv(t: np.ndarray) -> np.ndarray:
            return amplitude * omega * np.cos(omega * np.asarray(t, dtype=float) + phase)

        return cls("sinusoidal", func, deriv, period=1.0 / frequency,
                   parameters={"amplitude": amplitude, "frequency": frequency, "phase": phase, "offset": offset},
                   extrema=(offset - amplitude, offset + amplitude))

    @classmethod
    def triangular(cls, amplitude: float, frequency: float, duty: float = 0.5, offset: float = 0.0) -> "Excitation":
        amplitude = _nonnegative("amplitude", amplitude)
        frequency = _positive("frequency", frequency)
        duty = _finite("duty", duty)
        offset = _finite("offset", offset)
        if not 0.0 < duty < 1.0:
            raise InvalidInputError(f"duty must lie strictly inside (0, 1), got {duty}", details={"name": "duty"})
        period = 1.0 / frequency
        t_rise = duty * period
        t_fall = period - t_rise
        slope_up = 2.0 * amplitude / t_rise
        slope_down = -2.0 * amplitude / t_fall

        def func(t: np.ndarray) -> np.ndarray:
            tau = np.mod(np.asarray(t, dtype=float), period)
            rising = tau < t_rise
            out = np.where(rising, offset - amplitude + slope_up * tau,
                           offset + amplitude + slope_down * (tau - t_rise))
            return out

        def deriv(t: np.ndarray) -> np.ndarray:
            tau = np.mod(np.asarray(t, dtype=float), period)
            return np.where(tau < t_rise, slope_up, slope_down)

        return cls("triangular", func, deriv, period=period, breakpoints=(0.0, t_rise),
                   parameters={"amplitude": amplitude, "frequency": frequency, "duty": duty, "offset": offset},
                   split_hint=True, extrema=(offset - amplitude, offset + amplitude))

    @classmethod
    def piecewise_linear(cls, times: Sequence[float], values: Sequence[float], periodic: bool, *,
                         closure_tolerance: float = DEFAULT_CLOSURE_TOLERANCE) -> "Excitation":
        return cls._interpolated("piecewise_linear", times, values, "linear", bool(periodic), split_hint=True,
                                 closure_tolerance=closure_tolerance)

    @classmethod
    def sampled(cls, times: Sequence[float], values: Sequence[float], interpolation: str = "linear",
                periodic: bool = False, *, closure_tolerance: float = DEFAULT_CLOSURE_TOLERANCE) -> "Excitation":
        if interpolation not in ("linear", "cubic"):
            raise InvalidInputError(f"interpolation must be 'linear' or 'cubic', got {interpolation!r}")
        return cls._interpolated("sampled", times, values, interpolation, bool(periodic),
                                 split_hint=(interpolation == "linear"), closure_tolerance=closure_tolerance)

    @classmethod
    def _interpolated(cls, kind: str, times: Sequence[float], values: Sequence[float], interpolation: str,
                      periodic: bool, *, split_hint: bool, closure_tolerance: float) -> "Excitation":
        t = _array("times", times)
        v = _array("values", values).copy()
        if t.shape != v.shape:
            raise InvalidInputError(f"times and values must have the same length, got {t.size} and {v.size}")
        if not np.all(np.diff(t) > 0):
            raise InvalidInputError("times must be strictly increasing")
        closure_tolerance = _nonnegative("closure_tolerance", closure_tolerance)
        t0, t1 = float(t[0]), float(t[-1])
        span = t1 - t0
        params: dict[str, Any] = {"n_samples": int(t.size), "t_start": t0, "t_end": t1, "periodic": periodic,
                                  "interpolation": interpolation}
        if periodic:
            excursion = float(np.max(v) - np.min(v))
            mismatch = abs(float(v[-1] - v[0]))
            if mismatch > closure_tolerance * excursion:
                raise InvalidInputError(
                    f"periodic {kind} table does not close: values[0] = {v[0]:g} but values[-1] = {v[-1]:g} "
                    f"(mismatch {mismatch:g} exceeds closure_tolerance {closure_tolerance:g} of the {excursion:g} "
                    "excursion); the last sample must repeat the first (append it for [t0, T) data) or pass "
                    "periodic=False", details={"first": float(v[0]), "last": float(v[-1]), "mismatch": mismatch,
                                               "closure_tolerance": closure_tolerance})
            v[-1] = v[0]  # within tolerance: make the periodic continuation exactly continuous
            params["closure_mismatch"] = mismatch

        def fold(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            if periodic:
                return t0 + np.mod(x - t0, span)
            if np.any(x < t0 - 1e-12 * max(1.0, abs(t0))) or np.any(x > t1 + 1e-12 * max(1.0, abs(t1))):
                raise InvalidInputError(
                    f"non-periodic {kind} excitation is defined only on [{t0:g}, {t1:g}] s; no extrapolation",
                    details={"t_start": t0, "t_end": t1})
            return np.clip(x, t0, t1)  # only removes round-off outside the checked tolerance

        if interpolation == "linear":
            slopes = np.diff(v) / np.diff(t)

            def func(x: np.ndarray) -> np.ndarray:
                return np.interp(fold(x), t, v)

            def deriv(x: np.ndarray) -> np.ndarray:
                xf = fold(x)
                idx = np.searchsorted(t, xf, side="right") - 1
                idx = np.clip(idx, 0, slopes.size - 1)
                return slopes[idx]

            if periodic:
                breaks = [float(b) for b in t[:-1]]  # absolute corner times of one period (t[-1] == t0 + span)
            else:
                breaks = [float(b) for b in t[1:-1]]
        else:
            from scipy.interpolate import CubicSpline

            spline = CubicSpline(t, v, bc_type="periodic" if periodic else "not-a-knot")
            dspline = spline.derivative()

            def func(x: np.ndarray) -> np.ndarray:
                return spline(fold(x))

            def deriv(x: np.ndarray) -> np.ndarray:
                return dspline(fold(x))

            breaks = []
        extrema = (float(np.min(v)), float(np.max(v))) if interpolation == "linear" else None
        return cls(kind, func, deriv, period=(span if periodic else None), breakpoints=breaks, parameters=params,
                   domain=(None if periodic else (t0, t1)), split_hint=split_hint, extrema=extrema)

    @classmethod
    def from_callable(cls, func: Callable[[Any], Any], derivative: Callable[[Any], Any] | None = None, *,
                      period: float | None = None, name: str = "custom") -> "Excitation":
        if not callable(func):
            raise InvalidInputError("func must be callable")
        if derivative is not None and not callable(derivative):
            raise InvalidInputError("derivative must be callable or None")
        label = str(name)

        def guarded(callable_: Callable[[Any], Any], what: str) -> Callable[[np.ndarray], np.ndarray]:
            def wrapped(t: np.ndarray) -> np.ndarray:
                tt = np.asarray(t, dtype=float)
                try:
                    out = np.asarray(callable_(tt), dtype=float)
                except AIPEMagneticsError:
                    raise
                except Exception as error:  # noqa: BLE001 - any failure of a user callable is an invalid excitation
                    raise InvalidInputError(
                        f"custom excitation {label!r} {what} failed at t = {_describe_time(tt)}: "
                        f"{type(error).__name__}: {error}",
                        details={"name": label, "what": what, "error": type(error).__name__}) from error
                if out.shape != tt.shape:
                    if out.ndim == 0:
                        out = np.full(tt.shape, float(out), dtype=float)  # scalar result broadcast over times
                    else:
                        raise InvalidInputError(
                            f"custom excitation {label!r} {what} returned shape {out.shape} for t of shape "
                            f"{tt.shape}; it must return one value per time",
                            details={"name": label, "what": what, "shape": list(out.shape)})
                return out
            return wrapped

        d = None if derivative is None else guarded(derivative, "derivative")
        return cls("custom", guarded(func, "value"), d, period=period, parameters={"name": label})

    @classmethod
    def from_waveform(cls, waveform: Any) -> "Excitation":
        """Adapt one period of a ``models.waveforms.Waveform`` to a periodic excitation.

        Mapping (exact, no resampling of analytic kinds): ``sinusoidal`` -> :meth:`sinusoidal` with
        ``amplitude = peak`` and ``offset = dc_component``; ``triangular`` / ``trapezoidal`` /
        ``piecewise_linear`` -> periodic :meth:`piecewise_linear` from the ``(x, B)`` breakpoints with
        ``t = x / frequency``; ``sampled`` -> periodic linear :meth:`sampled` of the stored samples.  The
        ``Waveform`` has already validated closure of its table with its own tolerance, so the last
        value is set equal to the first before the table is handed over.  The values are carried over
        unchanged (tesla for a flux-density waveform); which physical quantity the excitation drives
        is decided by the consuming model.  The module is imported lazily; when it is unavailable
        :class:`UnsupportedOperationError` is raised.
        """
        try:
            import importlib
            waveforms = importlib.import_module("aipe_magnetics.models.waveforms")
        except ImportError as error:
            raise UnsupportedOperationError(
                "aipe_magnetics.models.waveforms is not available in this installation; build the "
                "excitation with Excitation.sinusoidal/triangular/piecewise_linear/sampled instead",
                details={"import_error": str(error)}) from None
        waveform_cls = getattr(waveforms, "Waveform", None)
        if waveform_cls is None or not isinstance(waveform, waveform_cls):
            raise UnsupportedOperationError(
                f"from_waveform expects a models.waveforms.Waveform instance, got {type(waveform).__name__}")
        kind = str(waveform.kind)
        frequency = float(waveform.frequency)
        if kind == "sinusoidal":
            exc = cls.sinusoidal(float(waveform.peak), frequency, 0.0, float(waveform.dc_component))
        elif kind in ("triangular", "trapezoidal", "piecewise_linear"):
            points = np.array(waveform.breakpoints, dtype=float)
            values = points[:, 1].copy()
            values[-1] = values[0]
            exc = cls.piecewise_linear(points[:, 0] / frequency, values, periodic=True)
        elif kind == "sampled":
            times, values = waveform.samples()
            values = np.array(values, dtype=float)
            values[-1] = values[0]
            exc = cls.sampled(times, values, interpolation="linear", periodic=True)
        else:
            raise UnsupportedOperationError(f"cannot adapt waveform kind {kind!r} to an Excitation")
        exc.parameters["waveform_kind"] = kind
        describe = getattr(waveform, "describe", None)
        if callable(describe):
            exc.parameters["waveform"] = describe()
        return exc

    # -- evaluation -----------------------------------------------------------------------------
    def __call__(self, t: Any) -> Any:
        return self.value(t)

    def value(self, t: Any) -> Any:
        """``u(t)``; returns a float for scalar ``t`` and an ndarray for array ``t``."""
        scalar = np.ndim(t) == 0
        out = self._func(np.asarray(t, dtype=float))
        return float(out) if scalar else np.asarray(out, dtype=float)

    @property
    def has_derivative(self) -> bool:
        return self._deriv is not None

    def derivative(self, t: Any) -> Any:
        """``du/dt``; raises :class:`UnsupportedOperationError` when no analytic derivative exists."""
        if self._deriv is None:
            raise UnsupportedOperationError(
                f"excitation of kind {self.kind!r} has no analytic derivative; supply one via from_callable()")
        scalar = np.ndim(t) == 0
        out = self._deriv(np.asarray(t, dtype=float))
        return float(out) if scalar else np.asarray(out, dtype=float)

    @property
    def domain(self) -> tuple[float, float] | None:
        """``(t_min, t_max)`` where the excitation is defined; ``None`` means the whole real line."""
        return self._domain

    def breakpoints(self, t0: float, t1: float) -> np.ndarray:
        """Times strictly inside ``(t0, t1)`` where the derivative (or value) is discontinuous."""
        t0, t1 = float(t0), float(t1)
        if self._breaks.size == 0 or t1 <= t0:
            return np.empty(0, dtype=float)
        if self.period is None:
            pts = self._breaks
        else:
            period = self.period
            # the stored corners lie in [b_min, b_max]; every corner t_i + k*period inside (t0, t1) has
            # (t0 - b_max)/period - 1 <= k <= (t1 - b_min)/period + 1 whatever the table's origin is
            b_min, b_max = float(self._breaks[0]), float(self._breaks[-1])
            k0 = math.floor((t0 - b_max) / period) - 1
            k1 = math.ceil((t1 - b_min) / period) + 1
            n_periods = k1 - k0 + 1
            if n_periods * self._breaks.size > 5_000_000:
                raise InvalidInputError(
                    "too many excitation breakpoints in the requested span; shorten t_span or disable "
                    "split_at_breakpoints", details={"estimated": n_periods * self._breaks.size})
            ks = np.arange(k0, k1 + 1, dtype=float)
            pts = (ks[:, None] * period + self._breaks[None, :]).ravel()
        tol = 1e-12 * max(1.0, abs(t0), abs(t1))
        pts = pts[(pts > t0 + tol) & (pts < t1 - tol)]
        return np.unique(pts)

    def extrema(self, t0: float, t1: float, samples: int = 4096) -> tuple[float, float]:
        """``(min u, max u)`` on ``[t0, t1]``.

        Exact for a constant, for a periodic excitation over at least one period, for a tabulated
        excitation over its whole domain and for piecewise-linear kinds on any interval (the sample
        grid is augmented with the knots inside the interval).  For a smooth kind on a sub-period
        interval the result is a dense sampled estimate (``samples`` points).
        """
        t0, t1 = float(t0), float(t1)
        if not (math.isfinite(t0) and math.isfinite(t1)) or t1 < t0:
            raise InvalidInputError(f"extrema needs a finite interval with t1 >= t0, got ({t0}, {t1})")
        if self._extrema is not None:
            tol = 1e-12 * max(1.0, abs(t0), abs(t1))
            if self.kind == "constant":
                return self._extrema
            if self.period is not None and (t1 - t0) >= self.period - tol:
                return self._extrema
            if self.period is None and self._domain is not None \
                    and t0 <= self._domain[0] + tol and t1 >= self._domain[1] - tol:
                return self._extrema
        if int(samples) < 2:
            raise InvalidInputError("samples must be >= 2")
        grid = np.linspace(t0, t1, int(samples))
        if self._breaks.size:
            grid = np.unique(np.concatenate([grid, self.breakpoints(t0, t1)]))
        vals = self.value(grid)
        return float(np.min(vals)), float(np.max(vals))

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {"kind": self.kind, "parameters": dict(self.parameters),
                                "has_derivative": self.has_derivative}
        if self.period is not None:
            info["period"] = self.period
        return info

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        params = ", ".join(f"{k}={v!r}" for k, v in self.parameters.items())
        return f"Excitation.{self.kind}({params})"
