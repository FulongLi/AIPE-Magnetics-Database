"""Periodic flux-density waveform descriptions used by core-loss models and the engineering layer.

Conventions (all SI)
--------------------
* Flux density ``B`` in tesla, time ``t`` in seconds, frequency ``f`` in hertz, period ``T = 1/f``.
* A :class:`Waveform` describes exactly one period of a periodic flux density ``B(t)``. Time is
  measured from the start of the period; piecewise-linear and sampled descriptions use the period
  fraction ``x = t / T`` in ``[0, 1]`` and must close: ``B(0) == B(T)``.
* ``peak`` is the amplitude of the AC component, ``(max B - min B) / 2``, i.e. half the major-loop
  excursion ``peak_to_peak``. For the symmetric constructors it equals the ``peak`` argument. This is
  the ``Bpk`` that Steinmetz-type models expect.
* ``dc_component`` is the time average of ``B(t)`` over one period. For the symmetric constructors it
  equals the ``dc`` argument. ``maximum`` and ``minimum`` are the absolute extremes.
* A *reversal* is a sign change of ``dB/dt`` around the cyclic period. A single major hysteresis loop
  has exactly two reversals (one maximum, one minimum); more than two reversals per period indicates
  minor loops. Flat segments (``dB/dt == 0``) are not reversals. Sample-to-sample changes whose
  magnitude does not exceed ``tolerance * peak_to_peak`` (default :data:`DEFAULT_REVERSAL_TOLERANCE`
  ``= 1e-9``) are treated as flat; this suppresses rounding noise only. The tolerance is a
  classification aid: it never filters the waveform itself, so measured waveforms with real noise must
  be smoothed or decimated by the caller before they are used in a loss integral.

Analytic definitions (``theta = 2*pi*f*t``, ``x = t/T``)
----------------------------------------------------------
* ``sinusoidal``:  ``B(t) = dc + peak * sin(theta)``.
* ``triangular``:  linear rise from ``dc - peak`` at ``x = 0`` to ``dc + peak`` at ``x = duty``, linear
  fall back to ``dc - peak`` at ``x = 1``.
* ``trapezoidal`` (symmetric): linear rise over ``duty_rise``, flat at ``dc + peak`` over ``duty_flat``,
  linear fall over ``duty_rise``, flat at ``dc - peak`` over ``duty_flat``; requires
  ``2 * (duty_rise + duty_flat) == 1`` (within 1e-9).
* ``piecewise_linear``: linear interpolation between user breakpoints ``(x_i, B_i)`` with ``x_0 == 0``,
  ``x_last == 1``, strictly increasing ``x`` whose spacing is resolvable in time
  (``(x_{i+1} - x_i) * T > 0`` in double precision - a flux change over zero time would need an infinite
  ``dB/dt``), and ``B_0 == B_last`` (within ``closure_tolerance`` of the peak-to-peak value).
* ``sampled``: uniform samples interpreted as the piecewise-linear interpolant between consecutive
  samples. With ``endpoint_included=True`` (default) the ``n >= 8`` samples ``B_i`` lie at
  ``x_i = i / (n - 1)``, ``i = 0 .. n-1`` (both ends of the period included, so ``B_0 ~= B_{n-1}`` within
  ``closure_tolerance`` of the peak-to-peak value). With ``endpoint_included=False`` the ``n >= 8``
  samples lie at ``x_i = i / n`` (the ``[0, T)`` convention of MagNet-style datasets): the period is
  closed by appending ``B_0`` and the stored representation has ``n + 1`` samples. The two conventions
  cannot be told apart from the data when the waveform is flat at both ends, so the caller must state
  the convention: passing ``[0, T)`` data as ``[0, T]`` stretches every sampling interval by
  ``n / (n - 1)``, which biases an iGSE loss by the factor ``(n / (n - 1))**(1 - alpha)`` (about
  ``-(alpha - 1) / n``, e.g. -0.05 % for ``alpha = 1.5`` and ``n = 1024``) plus whatever the missing
  last interval contributes.

A square *voltage* produces a triangular *flux* waveform; a square flux waveform would require an
infinite ``dB/dt`` and is therefore not constructible here.

Inputs are validated strictly: non-finite, non-real, non-positive (where required) or inconsistent
inputs raise :class:`~aipe_magnetics.common.errors.InvalidInputError`; nothing is clipped or coerced.
Derived descriptors (excursion, peak, DC component) are also required to be finite, so a waveform whose
excursion overflows double precision is rejected and :meth:`Waveform.describe` is always
JSON-serialisable. Direct construction through ``Waveform(...)`` re-validates the descriptors as a last
line of defence (positive frequency and peak, ``maximum > minimum``, ``peak == (maximum - minimum) / 2``,
``minimum <= dc_component <= maximum``, breakpoints/samples matching the kind).
"""

from __future__ import annotations

import math
import numbers
import sys
from typing import Any, Sequence

import numpy as np

from aipe_magnetics.common.errors import InvalidInputError, UnsupportedOperationError
from aipe_magnetics.models.contract import ExcitationKind

__all__ = ["DEFAULT_REVERSAL_TOLERANCE", "PIECEWISE_LINEAR_KINDS", "Waveform", "validate_bool", "validate_real"]

PIECEWISE_LINEAR_KINDS: tuple[str, ...] = ("triangular", "trapezoidal", "piecewise_linear")
DEFAULT_REVERSAL_TOLERANCE = 1e-9
"""Default flatness tolerance of :meth:`Waveform.count_reversals`, as a fraction of the peak-to-peak value."""
_KNOWN_KINDS: tuple[str, ...] = ("sinusoidal",) + PIECEWISE_LINEAR_KINDS + ("sampled",)
_DEFAULT_SAMPLES = 1024
_EPS = sys.float_info.epsilon


def validate_real(name: str, value: Any, *, positive: bool = False, nonnegative: bool = False) -> float:
    """Return ``value`` as a finite ``float`` or raise :class:`InvalidInputError`.

    Accepts Python and NumPy real scalars (not ``bool``, not complex, not strings). ``positive``
    requires ``value > 0``; ``nonnegative`` requires ``value >= 0``. Nothing is clipped.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise InvalidInputError(f"{name} must be a real number, got {type(value).__name__}", details={"name": name})
    result = float(value)
    if not math.isfinite(result):
        raise InvalidInputError(f"{name} must be finite, got {result}", details={"name": name})
    if positive and result <= 0.0:
        raise InvalidInputError(f"{name} must be positive, got {result:g}", details={"name": name, "value": result})
    if nonnegative and result < 0.0:
        raise InvalidInputError(f"{name} must be non-negative, got {result:g}", details={"name": name, "value": result})
    return result


def validate_bool(name: str, value: Any) -> bool:
    """Return ``value`` as a ``bool``; only ``bool``/``numpy.bool_`` are accepted (no truthiness coercion)."""
    if not isinstance(value, (bool, np.bool_)):
        raise InvalidInputError(f"{name} must be a bool, got {type(value).__name__}", details={"name": name})
    return bool(value)


def _validate_count(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise InvalidInputError(f"{name} must be an integer, got {type(value).__name__}", details={"name": name})
    count = int(value)
    if count < minimum:
        raise InvalidInputError(f"{name} must be >= {minimum}, got {count}", details={"name": name, "value": count})
    return count


def _real_array(name: str, values: Any) -> np.ndarray:
    """1-D finite float array from a sequence/array; rejects bool, complex, object and non-finite data."""
    try:
        array = np.asarray(values)
    except Exception as error:  # noqa: BLE001 - any conversion failure is an invalid input
        raise InvalidInputError(f"{name} could not be converted to an array: {error}", details={"name": name}) from None
    if array.dtype == object or array.dtype.kind not in "iuf":
        raise InvalidInputError(f"{name} must contain real numbers, got dtype {array.dtype}", details={"name": name})
    if array.ndim != 1:
        raise InvalidInputError(f"{name} must be one-dimensional, got shape {array.shape}", details={"name": name})
    array = array.astype(float, copy=True)
    if not np.all(np.isfinite(array)):
        raise InvalidInputError(f"{name} must be finite everywhere", details={"name": name})
    return array


def _finite_excursion(name: str, maximum: float, minimum: float) -> float:
    """``maximum - minimum`` of a flux-density description; rejects overflow and zero excursion."""
    excursion = maximum - minimum
    if not math.isfinite(excursion):
        raise InvalidInputError(
            f"flux-density excursion (max B - min B) of {name} overflows double precision "
            f"(max {maximum:g} T, min {minimum:g} T); values of this magnitude are not a physical flux density",
            details={"name": name, "maximum": maximum, "minimum": minimum})
    if excursion <= 0.0:
        raise InvalidInputError(f"{name} describe a constant flux density (zero peak-to-peak excursion); "
                                "a static field is not a periodic excitation", details={"name": name})
    return excursion


class Waveform:
    """Immutable description of one period of a periodic flux-density waveform (see module docstring)."""

    __slots__ = ("_kind", "_frequency", "_peak", "_dc", "_maximum", "_minimum", "_breakpoints", "_samples",
                 "_parameters")

    def __init__(self, kind: str, frequency: float, *, peak: float, dc_component: float, maximum: float,
                 minimum: float, breakpoints: np.ndarray | None = None, samples: np.ndarray | None = None,
                 parameters: dict[str, Any] | None = None):
        # Internal constructor: the classmethods validate user inputs. The checks below re-validate the
        # descriptors so that direct construction cannot produce a nonphysical or inconsistent waveform.
        if kind not in _KNOWN_KINDS:
            raise InvalidInputError(f"unknown waveform kind {kind!r}; known kinds: {', '.join(_KNOWN_KINDS)}")
        f = validate_real("frequency", frequency, positive=True)
        p = validate_real("peak", peak, positive=True)
        d = validate_real("dc_component", dc_component)
        hi = validate_real("maximum", maximum)
        lo = validate_real("minimum", minimum)
        excursion = _finite_excursion("descriptors", hi, lo)
        rounding = 4.0 * _EPS * (abs(hi) + abs(lo))  # cancellation in maximum - minimum for large offsets
        if abs(p - 0.5 * excursion) > 1e-12 * excursion + rounding:
            raise InvalidInputError(
                f"peak ({p:g} T) must equal half the excursion (maximum - minimum) / 2 = {0.5 * excursion:g} T",
                details={"peak": p, "maximum": hi, "minimum": lo})
        if d < lo - 1e-9 * excursion - rounding or d > hi + 1e-9 * excursion + rounding:
            raise InvalidInputError(
                f"dc_component ({d:g} T) must lie within [minimum, maximum] = [{lo:g}, {hi:g}] T",
                details={"dc_component": d, "maximum": hi, "minimum": lo})
        if (breakpoints is not None) != (kind in PIECEWISE_LINEAR_KINDS):
            raise InvalidInputError(f"breakpoints are required for the kinds {PIECEWISE_LINEAR_KINDS} and not "
                                    f"allowed otherwise (kind {kind!r})")
        if (samples is not None) != (kind == "sampled"):
            raise InvalidInputError(f"samples are required for kind 'sampled' and not allowed otherwise (kind {kind!r})")
        self._kind = kind
        self._frequency = f
        self._peak = p
        self._dc = d
        self._maximum = hi
        self._minimum = lo
        if breakpoints is not None:
            breakpoints = np.array(breakpoints, dtype=float, copy=True)
            breakpoints.flags.writeable = False
        if samples is not None:
            samples = np.array(samples, dtype=float, copy=True)
            samples.flags.writeable = False
        self._breakpoints = breakpoints
        self._samples = samples
        self._parameters = dict(parameters or {})

    # ------------------------------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------------------------------
    @classmethod
    def sinusoidal(cls, peak: float, frequency: float, dc: float = 0.0) -> "Waveform":
        """``B(t) = dc + peak * sin(2*pi*f*t)`` with ``peak > 0`` (T), ``frequency > 0`` (Hz)."""
        p = validate_real("peak", peak, positive=True)
        f = validate_real("frequency", frequency, positive=True)
        d = validate_real("dc", dc)
        return cls("sinusoidal", f, peak=p, dc_component=d, maximum=d + p, minimum=d - p, parameters={"dc": d})

    @classmethod
    def triangular(cls, peak: float, frequency: float, duty: float = 0.5, dc: float = 0.0) -> "Waveform":
        """Triangular flux: rise over the fraction ``duty`` of the period, fall over ``1 - duty``."""
        p = validate_real("peak", peak, positive=True)
        f = validate_real("frequency", frequency, positive=True)
        duty_value = validate_real("duty", duty)
        d = validate_real("dc", dc)
        if not 0.0 < duty_value < 1.0:
            raise InvalidInputError(f"duty must lie strictly between 0 and 1, got {duty_value:g}",
                                    details={"name": "duty", "value": duty_value})
        breakpoints = np.array([[0.0, d - p], [duty_value, d + p], [1.0, d - p]])
        return cls("triangular", f, peak=p, dc_component=d, maximum=d + p, minimum=d - p, breakpoints=breakpoints,
                   parameters={"duty": duty_value, "dc": d})

    @classmethod
    def trapezoidal(cls, peak: float, frequency: float, duty_rise: float, duty_flat: float,
                    dc: float = 0.0) -> "Waveform":
        """Symmetric trapezoidal flux: rise, flat, fall, flat with ``2*(duty_rise + duty_flat) == 1``."""
        p = validate_real("peak", peak, positive=True)
        f = validate_real("frequency", frequency, positive=True)
        rise = validate_real("duty_rise", duty_rise, positive=True)
        flat = validate_real("duty_flat", duty_flat, nonnegative=True)
        d = validate_real("dc", dc)
        if abs(2.0 * (rise + flat) - 1.0) > 1e-9:
            raise InvalidInputError(
                "symmetric trapezoid requires 2*(duty_rise + duty_flat) == 1 (rise, flat, fall, flat); "
                f"got duty_rise={rise:g}, duty_flat={flat:g}, sum of fractions {2.0 * (rise + flat):.12g}",
                details={"duty_rise": rise, "duty_flat": flat})
        xs = [0.0, rise, rise + flat, 2.0 * rise + flat, 1.0]
        bs = [d - p, d + p, d + p, d - p, d - p]
        points: list[list[float]] = []
        for x, b in zip(xs, bs):
            if points and x <= points[-1][0]:
                continue  # zero-length flat (duty_flat == 0) or rounding onto the previous breakpoint
            points.append([min(x, 1.0), b])
        points[-1][0] = 1.0
        return cls("trapezoidal", f, peak=p, dc_component=d, maximum=d + p, minimum=d - p,
                   breakpoints=np.array(points), parameters={"duty_rise": rise, "duty_flat": flat, "dc": d})

    @classmethod
    def piecewise_linear(cls, frequency: float, breakpoints: Sequence[Sequence[float]] | np.ndarray,
                         closure_tolerance: float = 1e-9) -> "Waveform":
        """Arbitrary piecewise-linear flux from breakpoints ``(x_i, B_i)``, ``x`` = period fraction.

        Requirements: at least two breakpoints, ``x_0 == 0.0`` and ``x_last == 1.0`` exactly, ``x``
        strictly increasing with every segment duration ``(x_{i+1} - x_i) / f`` representable as a
        positive double, all values finite, a finite non-zero excursion, and closure
        ``|B_0 - B_last| <= closure_tolerance * (max B - min B)``.
        """
        f = validate_real("frequency", frequency, positive=True)
        tolerance = validate_real("closure_tolerance", closure_tolerance, nonnegative=True)
        try:
            array = np.asarray(breakpoints)
        except Exception as error:  # noqa: BLE001
            raise InvalidInputError(f"breakpoints could not be converted to an array: {error}") from None
        if array.dtype == object or array.dtype.kind not in "iuf" or array.ndim != 2 or array.shape[1] != 2:
            raise InvalidInputError("breakpoints must be a sequence of (t_fraction, B) real pairs",
                                    details={"shape": list(array.shape)})
        if array.shape[0] < 2:
            raise InvalidInputError("breakpoints need at least two points (0, B0) ... (1, B0)")
        array = array.astype(float, copy=True)
        if not np.all(np.isfinite(array)):
            raise InvalidInputError("breakpoints must be finite")
        xs, bs = array[:, 0], array[:, 1]
        if xs[0] != 0.0 or xs[-1] != 1.0:
            raise InvalidInputError(f"breakpoint period fractions must start at 0.0 and end at 1.0, got "
                                    f"{xs[0]!r} .. {xs[-1]!r}", details={"first": float(xs[0]), "last": float(xs[-1])})
        dx = np.diff(xs)
        if not np.all(dx > 0.0):
            raise InvalidInputError("breakpoint period fractions must be strictly increasing")
        with np.errstate(under="ignore"):
            durations = dx * (1.0 / f)  # the same expression segments() uses
        if not np.all(durations > 0.0):
            raise InvalidInputError(
                f"breakpoint spacing is not resolvable in time at {f:g} Hz: a segment duration underflows to 0 s "
                "(a flux change over zero time would need an infinite dB/dt)",
                details={"frequency": f, "min_fraction": float(dx.min())})
        maximum, minimum = float(bs.max()), float(bs.min())
        excursion = _finite_excursion("breakpoints", maximum, minimum)
        if abs(float(bs[0] - bs[-1])) > tolerance * excursion:
            raise InvalidInputError(
                f"waveform does not close: B(0)={bs[0]:g} T but B(1)={bs[-1]:g} T "
                f"(tolerance {tolerance:g} of the {excursion:g} T excursion)",
                details={"b_start": float(bs[0]), "b_end": float(bs[-1])})
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            try:
                dc = float(np.sum(0.5 * (bs[:-1] + bs[1:]) * dx))  # exact time average of the linear interpolant
            except FloatingPointError:
                raise InvalidInputError("time average of the breakpoints overflows double precision",
                                        details={"maximum": maximum, "minimum": minimum}) from None
        return cls("piecewise_linear", f, peak=0.5 * excursion, dc_component=dc, maximum=maximum, minimum=minimum,
                   breakpoints=array, parameters={"n_breakpoints": int(array.shape[0])})

    @classmethod
    def sampled(cls, frequency: float, b_samples: Sequence[float] | np.ndarray, closure_tolerance: float = 1e-6, *,
                endpoint_included: bool = True) -> "Waveform":
        """One period of uniformly sampled flux density (``n >= 8`` samples).

        ``endpoint_included=True`` (default): the samples span ``[0, T]`` inclusive (``x_i = i/(n-1)``) and
        must close, ``|B_0 - B_{n-1}| <= closure_tolerance * (max B - min B)``.
        ``endpoint_included=False``: the samples span ``[0, T)`` (``x_i = i/n``, MagNet-style); the period
        is closed by appending ``B_0`` (``n + 1`` stored samples) and ``closure_tolerance`` is not used.
        In both cases the samples must be finite real numbers with a finite, non-zero excursion.
        """
        f = validate_real("frequency", frequency, positive=True)
        tolerance = validate_real("closure_tolerance", closure_tolerance, nonnegative=True)
        closed = validate_bool("endpoint_included", endpoint_included)
        samples = _real_array("b_samples", b_samples)
        n_input = int(samples.size)
        if n_input < 8:
            raise InvalidInputError(f"b_samples needs at least 8 samples per period, got {n_input}",
                                    details={"n_samples": n_input})
        if not closed:
            samples = np.append(samples, samples[0])  # [0, T) convention: B(T) = B(0) closes the period
        maximum, minimum = float(samples.max()), float(samples.min())
        excursion = _finite_excursion("b_samples", maximum, minimum)
        if closed and abs(float(samples[0] - samples[-1])) > tolerance * excursion:
            raise InvalidInputError(
                f"sampled waveform does not close: B[0]={samples[0]:g} T but B[-1]={samples[-1]:g} T "
                f"(tolerance {tolerance:g} of the {excursion:g} T excursion); samples must span one full period "
                "including both end points, or pass endpoint_included=False for [0, T) data",
                details={"b_start": float(samples[0]), "b_end": float(samples[-1])})
        n = int(samples.size)
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            try:
                dc = float((samples.sum() - 0.5 * (samples[0] + samples[-1])) / (n - 1))  # trapezoid mean, exact for PWL
            except FloatingPointError:
                raise InvalidInputError("time average of b_samples overflows double precision",
                                        details={"maximum": maximum, "minimum": minimum}) from None
        return cls("sampled", f, peak=0.5 * excursion, dc_component=dc, maximum=maximum, minimum=minimum,
                   samples=samples, parameters={"n_samples": n, "n_input_samples": n_input, "endpoint_included": closed})

    # ------------------------------------------------------------------------------------------
    # descriptive properties
    # ------------------------------------------------------------------------------------------
    @property
    def kind(self) -> ExcitationKind:
        return self._kind  # type: ignore[return-value]

    @property
    def frequency(self) -> float:
        """Fundamental frequency in Hz."""
        return self._frequency

    @property
    def period(self) -> float:
        """Period ``T = 1/f`` in s."""
        return 1.0 / self._frequency

    @property
    def peak(self) -> float:
        """AC amplitude ``(max B - min B) / 2`` in T."""
        return self._peak

    @property
    def peak_to_peak(self) -> float:
        """Major-loop excursion ``max B - min B`` in T."""
        return self._maximum - self._minimum

    @property
    def dc_component(self) -> float:
        """Time average of ``B(t)`` over one period in T."""
        return self._dc

    @property
    def maximum(self) -> float:
        return self._maximum

    @property
    def minimum(self) -> float:
        return self._minimum

    @property
    def parameters(self) -> dict[str, Any]:
        """Constructor parameters (duty, dc, n_samples, endpoint_included ...), copied."""
        return dict(self._parameters)

    @property
    def n_samples(self) -> int | None:
        """Number of stored samples spanning ``[0, T]`` inclusive for the ``sampled`` kind, else ``None``."""
        return None if self._samples is None else int(self._samples.size)

    @property
    def is_piecewise_linear(self) -> bool:
        return self._kind in PIECEWISE_LINEAR_KINDS

    @property
    def breakpoints(self) -> np.ndarray:
        """``(M, 2)`` array of ``(x, B)`` breakpoints for piecewise-linear kinds (read-only)."""
        if self._breakpoints is None:
            raise UnsupportedOperationError(
                f"breakpoints are defined only for piecewise-linear kinds, not {self._kind!r}")
        return self._breakpoints

    # ------------------------------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------------------------------
    def samples(self, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(t, B)`` on the uniform grid ``t_i = i*T/(n-1)``, ``i = 0..n-1`` (both ends included).

        Analytic kinds are evaluated exactly (``n`` defaults to 1024). For the ``sampled`` kind the stored
        samples are returned when ``n`` is omitted or equal to their count; any other ``n`` resamples the
        piecewise-linear interpolant (linear interpolation, stated explicitly because it changes the
        waveform's representation).
        """
        period = self.period
        if self._samples is not None:
            native = int(self._samples.size)
            count = native if n is None else _validate_count("n", n, 2)
            x_native = np.linspace(0.0, 1.0, native)
            if count == native:
                return x_native * period, self._samples.copy()
            x = np.linspace(0.0, 1.0, count)
            return x * period, np.interp(x, x_native, self._samples)
        count = _DEFAULT_SAMPLES if n is None else _validate_count("n", n, 2)
        x = np.linspace(0.0, 1.0, count)
        if self._kind == "sinusoidal":
            b = self._dc + self._peak * np.sin(2.0 * np.pi * x)
        else:
            assert self._breakpoints is not None
            b = np.interp(x, self._breakpoints[:, 0], self._breakpoints[:, 1])
        return x * period, b

    def segments(self) -> list[tuple[float, float]]:
        """Analytic ``(dt, dB)`` segments (s, T) of a piecewise-linear kind, in time order over one period.

        Raises :class:`UnsupportedOperationError` for ``sinusoidal`` and ``sampled`` waveforms, whose slopes
        are not analytic piecewise constants.
        """
        if self._breakpoints is None:
            raise UnsupportedOperationError(
                f"segments() is defined only for piecewise-linear kinds {PIECEWISE_LINEAR_KINDS}, not {self._kind!r}")
        period = self.period
        xs, bs = self._breakpoints[:, 0], self._breakpoints[:, 1]
        return [(float((xs[i + 1] - xs[i]) * period), float(bs[i + 1] - bs[i])) for i in range(len(xs) - 1)]

    def count_reversals(self, tolerance: float = DEFAULT_REVERSAL_TOLERANCE) -> int:
        """Number of sign changes of ``dB/dt`` around one cyclic period.

        Intervals whose ``|dB|`` does not exceed ``tolerance * peak_to_peak`` are treated as flat and ignored.
        The default suppresses rounding noise only. A larger tolerance can classify a waveform with a known
        sub-threshold ripple, but it is not a remedy for measured noise: the tolerance affects the loop count
        only, never the waveform, and a smooth waveform's per-sample change shrinks with the sample count, so
        noisy data must be smoothed or decimated before evaluation. The result is even: ``2`` means a single
        major loop, ``> 2`` means minor loops, ``0`` means no change above the tolerance.
        """
        tol = validate_real("tolerance", tolerance, nonnegative=True)
        if self._kind == "sinusoidal":
            return 2
        values = self._samples if self._samples is not None else self._breakpoints[:, 1]  # type: ignore[index]
        deltas = np.diff(values)
        signs = np.sign(deltas[np.abs(deltas) > tol * self.peak_to_peak])
        if signs.size == 0:
            return 0
        return int(np.count_nonzero(signs != np.roll(signs, 1)))

    @property
    def has_minor_loops(self) -> bool:
        """True when more than one local maximum/minimum pair occurs per period (default tolerance)."""
        return self.count_reversals() > 2

    def describe(self, reversal_tolerance: float = DEFAULT_REVERSAL_TOLERANCE) -> dict[str, Any]:
        """JSON-serialisable summary (SI units); ``reversal_tolerance`` is forwarded to :meth:`count_reversals`."""
        n_reversals = self.count_reversals(reversal_tolerance)
        return {
            "kind": self._kind,
            "frequency": self._frequency,
            "period": self.period,
            "peak": self._peak,
            "peak_to_peak": self.peak_to_peak,
            "dc_component": self._dc,
            "maximum": self._maximum,
            "minimum": self._minimum,
            "n_reversals": n_reversals,
            "has_minor_loops": n_reversals > 2,
            "reversal_tolerance": float(reversal_tolerance),
            "n_samples": self.n_samples,
            "parameters": dict(self._parameters),
            "units": {"flux_density": "T", "frequency": "Hz", "time": "s"},
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        extra = ", ".join(f"{key}={value:g}" if isinstance(value, float) else f"{key}={value}"
                          for key, value in self._parameters.items() if key != "dc")
        return (f"Waveform({self._kind}, f={self._frequency:g} Hz, peak={self._peak:g} T, "
                f"dc={self._dc:g} T{', ' + extra if extra else ''})")
