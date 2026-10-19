"""Reusable ODE modelling framework (``scipy.integrate.solve_ivp`` underneath).

Contract
--------
A dynamic model is an :class:`ODEModel` with

* ``state_names``: ordered state vector names;
* ``parameters``: ``dict[str, ModelParameter]`` (unit-bearing, with provenance);
* ``initial_state(**overrides)``: explicit, validated initial state.  Each state either has a
  documented physical default (``default_initial_state[name]`` is a number) or is *required*
  (``None``) and must be supplied; unknown or non-finite values raise :class:`InvalidInputError`;
* ``rhs(t, y, u, du_dt)``: continuous-time equations ``dy/dt``; ``u`` is the excitation value and
  ``du_dt`` its derivative (``None`` unless ``requires_du_dt`` is True);
* ``constraints(t, y)``: physical constraints evaluated on the *solution*, returned as a list of
  human-readable violations (never clipped or enforced);
* ``outputs(t, y, u)``: derived arrays (voltages, powers, flux density ...).

:func:`solve` integrates a model over ``t_span`` with a validated :class:`SolverConfig`.  The
integration is split at the excitation's breakpoints (corners of triangular / piecewise-linear
inputs) so that no step straddles a derivative discontinuity.  A solve either returns a complete
:class:`ODESolution` or raises: solver failure (status != 0), non-finite states, an exceeded step
budget (``max_steps``, checked after each segment), an exceeded right-hand-side evaluation budget
(``max_rhs_evaluations``, enforced *during* integration as a hard run-time guard) and arithmetic
failures inside the model raise :class:`ConvergenceError`; a right-hand side or excitation callable
that returns non-numeric values, the wrong number of values or raises a programming error
(``TypeError``, ``KeyError`` ...) raises :class:`InvalidInputError` naming the time of the failure.
No partial result is ever marked as success.

Numerical method
----------------
Methods are the ``solve_ivp`` integrators ``RK45`` (Dormand-Prince 5(4), the 5th-order solution is
propagated), ``DOP853``, ``Radau``, ``BDF`` and ``LSODA``.  Output on a user grid (``t_eval``) is
obtained from the solver's own dense output (``dense_output=True``), exactly as ``solve_ivp``
does internally, so the natural step count is always known.

:func:`convergence_study` reruns a problem with a sequence of fixed ``max_step`` values (and
``first_step = max_step``) and reports the error against the finest run or an analytic reference,
successive differences, the observed order of convergence and a ``converged`` flag.  Note that
``max_step`` caps the step, it does not disable the error controller; choose tolerances loose
enough (or steps small enough) that the controller never rejects a step when a clean order
measurement is wanted.

Periodic steady state
---------------------
:func:`periodic_steady_state` integrates period by period, i.e. it iterates the period map
``y_{k+1} = P(y_k)`` (Poincare map), until the boundary state has settled.  After every period the
relative boundary change is

    change_k = max_i |y_{k+1,i} - y_{k,i}| / scale_i,
    scale_i  = max( max_t y_i(t) - min_t y_i(t) over the period,  atol )

i.e. it is measured against the *peak-to-peak excursion* of each state during that period (not
against its magnitude, which would hide a transient under a DC offset), with the solver's absolute
tolerance as a floor.  For a geometrically contracting transient ``change_k ~ r^k`` the distance
still to go after period ``k`` is ``change_k * r / (1 - r)`` (sum of the remaining geometric
series), so the iteration is declared converged only when

    change_k <= tol   and   change_k * r / (1 - r) <= tol,     r = change_k / change_{k-1} < 1,

with ``r`` measured from two consecutive plain iterations.  Without a measured ratio (first period,
or the period right after an extrapolation) a change is accepted only when it is at most
``tol * 1e-3``, which bounds the remaining distance by ``tol`` for any contraction ratio up to
``0.999``.  A change of exactly zero is an exactly periodic state.

With ``accelerate=True`` the fixed point of the period map is approached with Steffensen's method:
after three consecutive plain iterates ``a, b, c`` the componentwise Aitken delta-squared
extrapolation ``c - (c - b)^2 / ((c - b) - (b - a))`` (applied only to components whose ratio
``(c - b)/(b - a)`` lies strictly inside ``(-1, 1)``) replaces the current state and plain iteration
resumes from it.  For an affine period map (linear time-invariant model) the extrapolation is the
exact fixed point; for slowly contracting nonlinear maps (ratio close to 1, e.g. Jiles-Atherton
minor loops) it replaces hundreds of periods by a few.  Convergence is always *verified* by plain
iterations with the criterion above; an extrapolated state from which the model cannot be
integrated (``ConvergenceError``) is discarded and plain iteration continues from the last plain
iterate (counted in ``rejected_extrapolations``).  The returned period is always an integrated
period: its end state lies within ``change * r / (1 - r) <= tol`` of the periodic orbit and its start
state within ``change / (1 - r)``.  Failing to converge within ``max_periods`` raises
:class:`ConvergenceError` whose message reports the measured contraction ratio and the estimated
number of additional periods.  A ``tol`` below the solver's period-to-period noise level (the
boundary change of an exactly periodic state is not zero but of the order of ``rtol`` accumulated
over the steps of a period) cannot be verified: the changes then plateau with a ratio around 1 and
the error says so; tighten ``rtol``/``atol`` or loosen ``tol`` in that case.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from scipy.integrate import solve_ivp

from aipe_magnetics.common.errors import AIPEMagneticsError, ConvergenceError, InvalidInputError
from aipe_magnetics.models.contract import ModelParameter
from aipe_magnetics.models.dynamic.excitation import Excitation
from aipe_magnetics.models.dynamic.solution import ODESolution

__all__ = ["ODEModel", "SolverConfig", "solve", "convergence_study", "periodic_steady_state", "PeriodicSteadyState",
           "SOLVER_METHODS", "RATIO_FREE_FRACTION"]

SOLVER_METHODS = ("RK45", "DOP853", "Radau", "BDF", "LSODA")
SolverMethod = Literal["RK45", "DOP853", "Radau", "BDF", "LSODA"]
_EPS = float(np.finfo(float).eps)
_TINY = float(np.finfo(float).tiny)

#: a boundary change this far below ``tol`` is accepted without a measured contraction ratio:
#: the remaining distance ``change * r / (1 - r)`` then stays below ``tol`` for every ``r <= 0.999``
RATIO_FREE_FRACTION = 1e-3


# ---------------------------------------------------------------------------------------------
# model contract
# ---------------------------------------------------------------------------------------------
class ODEModel(ABC):
    """Abstract continuous-time model ``dy/dt = rhs(t, y, u(t), du/dt)``."""

    state_names: tuple[str, ...] = ()
    parameters: dict[str, ModelParameter]
    requires_du_dt: bool = False
    #: per-state default initial value; ``None`` marks a required state (no silent default)
    default_initial_state: dict[str, float | None] = {}
    name: str = "ode_model"

    def __init__(self) -> None:
        if not isinstance(self.state_names, tuple) or not self.state_names:
            raise InvalidInputError(f"{type(self).__name__}.state_names must be a non-empty tuple")
        if not hasattr(self, "parameters"):
            self.parameters = {}

    @property
    def n_states(self) -> int:
        return len(self.state_names)

    def initial_state(self, **overrides: float) -> np.ndarray:
        """Explicit, validated initial state vector in ``state_names`` order."""
        unknown = sorted(set(overrides) - set(self.state_names))
        if unknown:
            raise InvalidInputError(f"unknown state(s) {unknown}; states are {self.state_names}",
                                    details={"unknown": unknown})
        values: list[float] = []
        for name in self.state_names:
            if name in overrides:
                value = overrides[name]
            else:
                value = self.default_initial_state.get(name, None)
                if value is None:
                    raise InvalidInputError(
                        f"initial state {name!r} is required for {type(self).__name__} and has no default",
                        details={"state": name})
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise InvalidInputError(f"initial state {name!r} must be a real number")
            value = float(value)
            if not math.isfinite(value):
                raise InvalidInputError(f"initial state {name!r} must be finite, got {value}")
            values.append(value)
        return np.array(values, dtype=float)

    @abstractmethod
    def rhs(self, t: float, y: np.ndarray, u: float, du_dt: float | None) -> Sequence[float] | np.ndarray:
        """Return ``dy/dt`` (length ``n_states``)."""

    def constraints(self, t: np.ndarray, y: np.ndarray) -> list[str]:  # noqa: ARG002 - default: none
        return []

    def outputs(self, t: np.ndarray, y: np.ndarray, u: np.ndarray) -> dict[str, np.ndarray]:  # noqa: ARG002
        return {}

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "states": list(self.state_names), "requires_du_dt": self.requires_du_dt,
                "default_initial_state": dict(self.default_initial_state),
                "parameters": {k: p.model_dump(mode="json", exclude_none=True) for k, p in self.parameters.items()}}


# ---------------------------------------------------------------------------------------------
# solver configuration
# ---------------------------------------------------------------------------------------------
class SolverConfig(BaseModel):
    """Validated solver settings (all fields are explicit; nothing is clipped silently)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: SolverMethod = "RK45"
    rtol: float = Field(default=1e-6, description="relative tolerance (>= 100*eps, < 1)")
    atol: float = Field(default=1e-9, description="absolute tolerance in state units (>= 0)")
    max_step: float | None = Field(default=None, description="maximum step size (s); None = unlimited")
    first_step: float | None = Field(default=None, description="initial step size hint (s); None = automatic")
    dense_output: bool = Field(default=False, description="keep the dense interpolant in the ODESolution")
    max_steps: int = Field(default=1_000_000, description="budget of accepted steps per solve")
    max_rhs_evaluations: int = Field(default=20_000_000, description="hard run-time guard on rhs calls per solve")
    split_at_breakpoints: bool = Field(default=True, description="restart the integrator at excitation corners")
    max_segments: int = Field(default=20_000, description="refuse to split into more segments than this")

    @field_validator("rtol")
    @classmethod
    def _rtol(cls, v: float) -> float:
        if isinstance(v, bool) or not math.isfinite(v):
            raise ValueError("rtol must be finite")
        if v < 100.0 * _EPS:
            raise ValueError(f"rtol must be >= 100*eps = {100.0 * _EPS:.3e} (scipy would silently raise it)")
        if v >= 1.0:
            raise ValueError("rtol must be < 1")
        return float(v)

    @field_validator("atol")
    @classmethod
    def _atol(cls, v: float) -> float:
        if isinstance(v, bool) or not math.isfinite(v) or v < 0.0:
            raise ValueError("atol must be finite and non-negative")
        return float(v)

    @field_validator("max_step", "first_step")
    @classmethod
    def _steps(cls, v: float | None) -> float | None:
        if v is None:
            return None
        if isinstance(v, bool) or not math.isfinite(v) or v <= 0.0:
            raise ValueError("step sizes must be finite and positive (or None)")
        return float(v)

    @field_validator("max_steps", "max_rhs_evaluations", "max_segments")
    @classmethod
    def _budgets(cls, v: int) -> int:
        if isinstance(v, bool) or int(v) != v or v <= 0:
            raise ValueError("budgets must be positive integers")
        return int(v)

    @model_validator(mode="after")
    def _consistent(self) -> "SolverConfig":
        if self.first_step is not None and self.max_step is not None and self.first_step > self.max_step:
            raise ValueError("first_step must not exceed max_step")
        return self


@dataclass
class PeriodicSteadyState:
    """Result of :func:`periodic_steady_state`.

    ``periods`` counts the integrated periods (the cost); ``history`` holds the relative boundary
    change of every integrated period and ``boundary_states`` the initial state followed by the end
    state of every integrated period (with ``accelerate=True`` the start state of a period can differ
    from the previous end state: ``extrapolations`` counts those restarts).  ``contraction_ratio`` is
    the ratio of the last two consecutive plain changes (``None`` when not measurable) and
    ``estimated_remaining_change`` the geometric-series estimate ``change * r / (1 - r)``.
    ``scale`` is the per-state peak-to-peak excursion used to normalise the last change.
    """

    solution: ODESolution
    periods: int
    converged: bool
    boundary_change: float
    boundary_states: list[np.ndarray]
    tolerance: float
    history: list[float]
    contraction_ratio: float | None = None
    estimated_remaining_change: float | None = None
    accelerated: bool = False
    extrapolations: int = 0
    rejected_extrapolations: int = 0
    scale: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
class _Abort(Exception):
    """Internal: abort the integration from inside the right-hand side with a typed package error."""

    def __init__(self, error_type: type, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.details = dict(details or {})


def _as_model(model: Any) -> ODEModel:
    if not isinstance(model, ODEModel):
        raise InvalidInputError(f"model must be an ODEModel instance, got {type(model).__name__}")
    return model


def _as_excitation(excitation: Any) -> Excitation:
    if isinstance(excitation, Excitation):
        return excitation
    if callable(excitation):
        return Excitation.from_callable(excitation)
    raise InvalidInputError(f"excitation must be an Excitation or a callable u(t), got {type(excitation).__name__}")


def _as_config(config: Any) -> SolverConfig:
    if config is None:
        return SolverConfig()
    if isinstance(config, SolverConfig):
        return config
    if isinstance(config, dict):
        try:
            return SolverConfig(**config)
        except ValidationError as error:
            raise InvalidInputError(f"invalid solver configuration: {error}") from None
    raise InvalidInputError(f"config must be a SolverConfig, dict or None, got {type(config).__name__}")


def _as_span(t_span: Any) -> tuple[float, float]:
    try:
        t0, t1 = (float(v) for v in t_span)
    except (TypeError, ValueError):
        raise InvalidInputError("t_span must be a pair (t0, t1) of numbers") from None
    if not (math.isfinite(t0) and math.isfinite(t1)):
        raise InvalidInputError(f"t_span must be finite, got ({t0}, {t1})")
    if t1 <= t0:
        raise InvalidInputError(f"t_span must satisfy t1 > t0, got ({t0}, {t1})")
    return t0, t1


def _as_y0(model: ODEModel, y0: Any) -> np.ndarray:
    if y0 is None:
        return model.initial_state()
    try:
        arr = np.asarray(y0, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise InvalidInputError(f"y0 must be numeric: {error}") from None
    if arr.size != model.n_states:
        raise InvalidInputError(f"y0 must have {model.n_states} entries ({model.state_names}), got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise InvalidInputError("y0 contains non-finite values", details={"y0": [float(v) for v in arr]})
    return arr


def _as_t_eval(t_eval: Any, t0: float, t1: float) -> np.ndarray | None:
    if t_eval is None:
        return None
    arr = np.asarray(t_eval, dtype=float).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise InvalidInputError("t_eval must be a non-empty array of finite times")
    if np.any(np.diff(arr) < 0):
        raise InvalidInputError("t_eval must be non-decreasing")
    tol = 1e-12 * max(1.0, abs(t0), abs(t1))
    if arr[0] < t0 - tol or arr[-1] > t1 + tol:
        raise InvalidInputError(f"t_eval must lie inside t_span [{t0:g}, {t1:g}]")
    return np.clip(arr, t0, t1)


def _segments(excitation: Excitation, t0: float, t1: float, config: SolverConfig) -> list[tuple[float, float]]:
    if not config.split_at_breakpoints or not excitation.split_hint:
        return [(t0, t1)]
    breaks = excitation.breakpoints(t0, t1)
    if breaks.size + 1 > config.max_segments:
        raise InvalidInputError(
            f"excitation has {breaks.size} breakpoints in the span, exceeding max_segments={config.max_segments}; "
            "shorten t_span, raise SolverConfig.max_segments or set split_at_breakpoints=False",
            details={"breakpoints": int(breaks.size), "max_segments": config.max_segments})
    edges = np.concatenate([[t0], breaks, [t1]])
    return [(float(edges[i]), float(edges[i + 1])) for i in range(edges.size - 1)]


def _probe_excitation(exc: Excitation, t: float, needs_du: bool) -> None:
    """Evaluate the excitation once at ``t`` and turn every failure into a typed error."""
    try:
        value = exc.value(t)
        derivative = exc.derivative(t) if needs_du else 0.0
    except AIPEMagneticsError:
        raise
    except Exception as error:  # noqa: BLE001 - a failing excitation is an invalid input, never a crash
        raise InvalidInputError(f"excitation failed at t={t:g}: {type(error).__name__}: {error}",
                                details={"t": t, "error": type(error).__name__}) from error
    if not math.isfinite(value):
        raise InvalidInputError(f"excitation is non-finite at t={t:g}", details={"t": t})
    if not math.isfinite(derivative):
        raise InvalidInputError(f"excitation derivative is non-finite at t={t:g}", details={"t": t})


# ---------------------------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------------------------
def solve(model: ODEModel, excitation: Excitation | Callable[[Any], Any], t_span: Sequence[float],
          config: SolverConfig | dict[str, Any] | None = None, y0: Any = None, *,
          t_eval: Any = None) -> ODESolution:
    """Integrate ``model`` driven by ``excitation`` over ``t_span``; complete solution or a typed error."""
    model = _as_model(model)
    exc = _as_excitation(excitation)
    cfg = _as_config(config)
    t0, t1 = _as_span(t_span)
    y = _as_y0(model, y0)
    grid = _as_t_eval(t_eval, t0, t1)
    n = model.n_states
    if exc.domain is not None and (t0 < exc.domain[0] - 1e-12 * max(1.0, abs(t0)) or
                                   t1 > exc.domain[1] + 1e-12 * max(1.0, abs(t1))):
        raise InvalidInputError(f"excitation is defined only on [{exc.domain[0]:g}, {exc.domain[1]:g}] s, "
                                f"t_span {(t0, t1)} lies outside it")
    needs_du = bool(model.requires_du_dt)
    if needs_du and not exc.has_derivative:
        raise InvalidInputError(f"{type(model).__name__} requires du/dt but the {exc.kind!r} excitation has no "
                                "analytic derivative; supply one (from_callable(func, derivative=...))")
    for t_check in (t0, t1):
        _probe_excitation(exc, t_check, needs_du)
    segments = _segments(exc, t0, t1, cfg)

    counter = {"nfev": 0, "nonfinite": 0}
    budget = cfg.max_rhs_evaluations
    model_name = type(model).__name__

    def fun(t: float, yv: np.ndarray) -> np.ndarray:
        counter["nfev"] += 1
        if counter["nfev"] > budget:
            raise _Abort(ConvergenceError,
                         f"right-hand-side evaluation budget max_rhs_evaluations={budget} exceeded at t={t:g}")
        try:
            u = exc.value(t)
            du = exc.derivative(t) if needs_du else None
        except AIPEMagneticsError as error:
            raise _Abort(InvalidInputError, f"excitation failed at t={t:g}: {error}", getattr(error, "details", None)) \
                from error
        except Exception as error:  # noqa: BLE001 - untyped failure of a user excitation callable
            raise _Abort(InvalidInputError, f"excitation raised {type(error).__name__} at t={t:g}: {error}") from error
        try:
            dy = model.rhs(t, yv, u, du)
        except AIPEMagneticsError as error:
            raise _Abort(ConvergenceError, f"rhs failed at t={t:g}, y={np.asarray(yv).tolist()}: {error}") from error
        except ArithmeticError as error:  # ZeroDivisionError, OverflowError, FloatingPointError
            raise _Abort(ConvergenceError, f"{model_name}.rhs raised {type(error).__name__} at t={t:g}, "
                                           f"y={np.asarray(yv).tolist()}: {error}") from error
        except Exception as error:  # noqa: BLE001 - programming error inside a model: invalid model, not a crash
            raise _Abort(InvalidInputError, f"{model_name}.rhs raised {type(error).__name__} at t={t:g}: {error}") \
                from error
        try:
            dy_arr = np.asarray(dy, dtype=float).reshape(-1)
        except (TypeError, ValueError) as error:
            raise _Abort(InvalidInputError, f"{model_name}.rhs returned non-numeric values at t={t:g}: {error}") \
                from error
        if dy_arr.size != n:
            raise _Abort(InvalidInputError, f"{model_name}.rhs returned {dy_arr.size} values at t={t:g}, expected {n}")
        if not np.all(np.isfinite(dy_arr)):
            counter["nonfinite"] += 1  # trial stage outside the model domain: left to the step controller
        return dy_arr

    ts: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    pieces: list[tuple[float, float, Any]] = []
    stats = {"nfev": 0, "njev": 0, "nlu": 0, "n_steps": 0}
    for index, (ta, tb) in enumerate(segments):
        kwargs: dict[str, Any] = {"method": cfg.method, "rtol": cfg.rtol, "atol": cfg.atol, "dense_output": True}
        if cfg.max_step is not None:
            kwargs["max_step"] = cfg.max_step
        if cfg.first_step is not None:
            kwargs["first_step"] = min(cfg.first_step, tb - ta)
        try:
            sol = solve_ivp(fun, (ta, tb), y, **kwargs)
        except _Abort as error:
            details = {"segment": index, "t_span": [ta, tb], "nfev": counter["nfev"], "method": cfg.method}
            details.update(error.details)
            if error.error_type is ConvergenceError:
                raise ConvergenceError(f"integration aborted: {error.message}", details=details) from error.__cause__
            raise InvalidInputError(f"integration aborted: {error.message}", details=details) from error.__cause__
        if sol.status != 0 or not sol.success:
            raise ConvergenceError(
                f"{cfg.method} failed on [{ta:g}, {tb:g}] s: {sol.message} (status {sol.status}); reached "
                f"t={float(sol.t[-1]):g} after {int(sol.t.size) - 1} steps",
                details={"status": int(sol.status), "message": str(sol.message), "segment": index,
                         "t_reached": float(sol.t[-1]), "nfev": int(sol.nfev), "method": cfg.method,
                         "n_nonfinite_rhs": counter["nonfinite"]})
        if not np.all(np.isfinite(sol.y)):
            raise ConvergenceError(f"non-finite state produced on [{ta:g}, {tb:g}] s",
                                   details={"segment": index, "method": cfg.method})
        stats["nfev"] += int(sol.nfev)
        stats["njev"] += int(sol.njev)
        stats["nlu"] += int(sol.nlu)
        stats["n_steps"] += int(sol.t.size) - 1
        if stats["n_steps"] > cfg.max_steps:
            raise ConvergenceError(
                f"step budget max_steps={cfg.max_steps} exceeded ({stats['n_steps']} accepted steps after "
                f"segment {index + 1}/{len(segments)})",
                details={"max_steps": cfg.max_steps, "n_steps": stats["n_steps"], "method": cfg.method})
        start = 1 if index > 0 else 0
        ts.append(np.asarray(sol.t[start:], dtype=float))
        ys.append(np.asarray(sol.y[:, start:], dtype=float))
        pieces.append((ta, tb, sol.sol))
        y = np.asarray(sol.y[:, -1], dtype=float)

    t_nat = np.concatenate(ts)
    y_nat = np.concatenate(ys, axis=1)
    starts = np.array([p[0] for p in pieces], dtype=float)

    def dense(times: np.ndarray) -> np.ndarray:
        times = np.atleast_1d(np.asarray(times, dtype=float))
        out = np.empty((n, times.size), dtype=float)
        which = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, len(pieces) - 1)
        for i, (_, _, interp) in enumerate(pieces):
            mask = which == i
            if np.any(mask):
                out[:, mask] = np.asarray(interp(times[mask]), dtype=float).reshape(n, -1)
        return out

    if grid is None:
        t_out, y_out = t_nat, y_nat
    else:
        t_out = grid
        y_out = dense(grid)
        # exact end points come from the accepted steps, not the interpolant
        if t_out[0] == t0:
            y_out[:, 0] = y_nat[:, 0]
        if t_out[-1] == t1:
            y_out[:, -1] = y_nat[:, -1]
    if not np.all(np.isfinite(y_out)):
        raise ConvergenceError("non-finite values in the interpolated solution", details={"method": cfg.method})
    u_out = np.asarray(exc.value(t_out), dtype=float)

    outputs_raw = model.outputs(t_out, y_out, u_out)
    if not isinstance(outputs_raw, dict):
        raise InvalidInputError(f"{model_name}.outputs must return a dict of arrays")
    outputs: dict[str, np.ndarray] = {}
    for key, value in outputs_raw.items():
        arr = np.asarray(value, dtype=float)
        if arr.shape != t_out.shape:
            raise InvalidInputError(f"output {key!r} has shape {arr.shape}, expected {t_out.shape}")
        outputs[str(key)] = arr
    violations = [str(v) for v in model.constraints(t_out, y_out)]

    diagnostics: dict[str, Any] = {
        "method": cfg.method, "rtol": cfg.rtol, "atol": cfg.atol, "max_step": cfg.max_step,
        "first_step": cfg.first_step, "nfev": stats["nfev"], "njev": stats["njev"], "nlu": stats["nlu"],
        "status": 0, "message": "The solver successfully reached the end of the integration interval.",
        "success": True, "n_steps": stats["n_steps"], "n_segments": len(segments),
        "n_nonfinite_rhs": counter["nonfinite"], "t_span": [t0, t1], "n_points": int(t_out.size),
        "excitation": exc.describe(),
    }
    return ODESolution(t=t_out, y=y_out, u=u_out, state_names=tuple(model.state_names), outputs=outputs,
                       diagnostics=diagnostics, constraint_violations=violations,
                       dense=dense if cfg.dense_output else None)


# ---------------------------------------------------------------------------------------------
# convergence study
# ---------------------------------------------------------------------------------------------
def convergence_study(model: ODEModel, excitation: Excitation | Callable[[Any], Any], t_span: Sequence[float],
                      config: SolverConfig | dict[str, Any] | None,
                      max_steps: Sequence[float] = (1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5), *,
                      y0: Any = None, tolerance: float = 1e-6, reference: Callable[[np.ndarray], Any] | None = None,
                      n_eval: int = 201) -> dict[str, Any]:
    """Time-step convergence study with fixed ``max_step`` values (see module docstring)."""
    cfg = _as_config(config)
    t0, t1 = _as_span(t_span)
    try:
        hs = sorted({float(h) for h in max_steps}, reverse=True)
    except (TypeError, ValueError):
        raise InvalidInputError("max_steps must be a sequence of positive numbers") from None
    if len(hs) < 2 or any((not math.isfinite(h)) or h <= 0.0 for h in hs):
        raise InvalidInputError("max_steps needs at least two distinct positive finite step sizes")
    if hs[0] > (t1 - t0):
        raise InvalidInputError("the coarsest max_step exceeds the integration span")
    if isinstance(tolerance, bool) or not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise InvalidInputError("tolerance must be a positive finite number")
    if int(n_eval) < 2:
        raise InvalidInputError("n_eval must be >= 2")
    grid = np.linspace(t0, t1, int(n_eval))
    runs: list[ODESolution] = []
    for h in hs:
        first = h if cfg.first_step is None else min(cfg.first_step, h)
        run_cfg = cfg.model_copy(update={"max_step": h, "first_step": first})
        runs.append(solve(model, excitation, (t0, t1), run_cfg, y0, t_eval=grid))

    if reference is None:
        ref = runs[-1].y
        ref_label = "finest_run"
        compared = runs[:-1]
    else:
        ref = np.asarray(reference(grid), dtype=float)
        if ref.ndim == 1:
            ref = ref.reshape(1, -1)
        if ref.shape != runs[-1].y.shape:
            raise InvalidInputError(f"reference must return shape {runs[-1].y.shape}, got {ref.shape}")
        if not np.all(np.isfinite(ref)):
            raise InvalidInputError("reference solution contains non-finite values")
        ref_label = "analytic_reference"
        compared = runs
    scale = float(np.max(np.abs(ref)))
    if scale == 0.0:
        scale = 1.0  # reference identically zero: absolute errors
    errors = [float(np.max(np.abs(run.y - ref)) / scale) for run in compared]
    diffs = [float(np.max(np.abs(runs[i].y - runs[i + 1].y)) / scale) for i in range(len(runs) - 1)]

    def orders_from(values: list[float], steps: list[float]) -> list[float]:
        out: list[float] = []
        for i in range(len(values) - 1):
            if values[i] > 0.0 and values[i + 1] > 0.0 and steps[i] != steps[i + 1]:
                out.append(float(math.log(values[i] / values[i + 1]) / math.log(steps[i] / steps[i + 1])))
        return out

    orders_diff = orders_from(diffs, hs[:-1])
    orders_err = orders_from(errors, hs[:len(errors)])
    observed = float(np.median(orders_diff)) if orders_diff else None
    observed_vs_ref = float(np.median(orders_err)) if orders_err else None
    monotone = all(errors[i] > errors[i + 1] for i in range(len(errors) - 1)) if len(errors) > 1 else True
    finest_error = errors[-1] if errors else 0.0
    return {
        "max_steps": hs,
        "n_steps": [int(run.diagnostics["n_steps"]) for run in runs],
        "nfev": [int(run.diagnostics["nfev"]) for run in runs],
        "method": cfg.method,
        "reference": ref_label,
        "norm": "max over states and grid, relative to max|reference|",
        "scale": scale,
        "errors": errors,
        "successive_differences": diffs,
        "orders_successive": orders_diff,
        "orders_vs_reference": orders_err,
        "observed_order": observed,
        "observed_order_vs_reference": observed_vs_ref,
        "errors_monotone_decreasing": bool(monotone),
        "finest_error": float(finest_error),
        "tolerance": float(tolerance),
        "converged": bool(finest_error <= float(tolerance)),
        "n_eval": int(n_eval),
    }


# ---------------------------------------------------------------------------------------------
# periodic steady state
# ---------------------------------------------------------------------------------------------
def _aitken(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray | None:
    """Componentwise Aitken delta-squared extrapolation of three consecutive iterates.

    A component is extrapolated only when its ratio ``(c - b) / (b - a)`` lies strictly inside
    ``(-1, 1)`` (contracting); other components keep ``c``.  Returns ``None`` when no component can be
    extrapolated or the result is not finite.
    """
    d1 = b - a
    d2 = c - b
    out = np.array(c, dtype=float)
    applied = False
    for i in range(c.size):
        if d1[i] == 0.0 or d2[i] == 0.0:
            continue
        ratio = d2[i] / d1[i]
        if not (-1.0 < ratio < 1.0):
            continue
        out[i] = c[i] + d2[i] * ratio / (1.0 - ratio)  # == c - d2^2 / (d2 - d1)
        applied = True
    if not applied or not np.all(np.isfinite(out)):
        return None
    return out


def periodic_steady_state(model: ODEModel, excitation: Excitation | Callable[[Any], Any], period: float,
                          config: SolverConfig | dict[str, Any] | None = None, max_periods: int = 50,
                          tol: float = 1e-6, *, y0: Any = None, points_per_period: int | None = None,
                          t_start: float = 0.0, accelerate: bool = False) -> PeriodicSteadyState:
    """Iterate the period map until the boundary state has settled (criterion in the module docstring)."""
    model = _as_model(model)
    exc = _as_excitation(excitation)
    cfg = _as_config(config)
    if isinstance(period, bool) or not isinstance(period, (int, float, np.integer, np.floating)) \
            or not math.isfinite(float(period)) or float(period) <= 0.0:
        raise InvalidInputError(f"period must be a positive finite number, got {period!r}")
    period = float(period)
    if exc.period is not None:
        ratio = period / exc.period
        if abs(ratio - round(ratio)) > 1e-9 * max(1.0, ratio) or round(ratio) < 1:
            raise InvalidInputError(
                f"period {period:g} s is not an integer multiple of the excitation period {exc.period:g} s",
                details={"period": period, "excitation_period": exc.period})
    if isinstance(max_periods, bool) or int(max_periods) != max_periods or max_periods < 1:
        raise InvalidInputError("max_periods must be a positive integer")
    if isinstance(tol, bool) or not math.isfinite(float(tol)) or float(tol) <= 0.0:
        raise InvalidInputError("tol must be a positive finite number")
    if points_per_period is not None and (int(points_per_period) != points_per_period or points_per_period < 2):
        raise InvalidInputError("points_per_period must be an integer >= 2 or None")
    t_start = float(t_start)
    if not math.isfinite(t_start):
        raise InvalidInputError("t_start must be finite")
    tol = float(tol)
    accelerate = bool(accelerate)
    floor = max(cfg.atol, _TINY)

    y = _as_y0(model, y0)
    boundary_states = [y.copy()]
    history: list[float] = []
    plain: list[np.ndarray] = [y.copy()]  # consecutive plain iterates since the last extrapolation
    prev_change: float | None = None
    change = math.inf
    ratio: float | None = None
    remaining: float | None = None
    scale = np.full(model.n_states, floor)
    n_extrapolations = 0
    n_rejected = 0
    n_steps_last = 0
    fallback: np.ndarray | None = None  # last plain iterate while integrating from an extrapolated state
    for k in range(int(max_periods)):
        ta = t_start + k * period
        tb = t_start + (k + 1) * period
        grid = None if points_per_period is None else np.linspace(ta, tb, int(points_per_period) + 1)
        try:
            sol = solve(model, exc, (ta, tb), cfg, y0=y, t_eval=grid)
        except ConvergenceError:
            if fallback is None:
                raise
            # the extrapolated state could not be integrated: discard it, resume plain iteration
            n_rejected += 1
            y = fallback
            fallback = None
            plain = [y.copy()]
            prev_change = None
            sol = solve(model, exc, (ta, tb), cfg, y0=y, t_eval=grid)
        fallback = None
        n_steps_last = int(sol.diagnostics["n_steps"])
        y_new = sol.final_state
        scale = np.maximum(np.max(sol.y, axis=1) - np.min(sol.y, axis=1), floor)
        change = float(np.max(np.abs(y_new - y) / scale))
        history.append(change)
        boundary_states.append(y_new.copy())
        ratio = (change / prev_change) if (prev_change is not None and prev_change > 0.0) else None
        remaining = change * ratio / (1.0 - ratio) if (ratio is not None and ratio < 1.0) else None
        converged = change <= tol and (change == 0.0 or change <= RATIO_FREE_FRACTION * tol
                                       or (remaining is not None and remaining <= tol))
        if converged:
            return PeriodicSteadyState(solution=sol, periods=k + 1, converged=True, boundary_change=change,
                                       boundary_states=boundary_states, tolerance=tol, history=history,
                                       contraction_ratio=ratio, estimated_remaining_change=remaining,
                                       accelerated=accelerate, extrapolations=n_extrapolations,
                                       rejected_extrapolations=n_rejected, scale=[float(s) for s in scale])
        plain.append(y_new.copy())
        prev_change = change
        y = y_new
        if accelerate and len(plain) >= 3:
            y_acc = _aitken(plain[-3], plain[-2], plain[-1])
            if y_acc is not None:
                fallback = y.copy()
                y = y_acc
                plain = [y.copy()]
                prev_change = None  # the change after a restart is not comparable with the previous one
                n_extrapolations += 1

    ratio_txt = "not measurable" if ratio is None else f"{ratio:.4g}"
    remaining_txt = "unknown" if remaining is None else f"{remaining:.3e}"
    more: int | None = None
    if ratio is not None and 0.0 < ratio < 1.0 and remaining is not None and remaining > tol:
        more = int(math.ceil(math.log(tol / remaining) / math.log(ratio)))
    hint = ""
    accel_hint = "" if accelerate else " or pass accelerate=True (Steffensen extrapolation of the period map)"
    if (ratio is None or ratio >= 1.0) and change >= 0.5:
        hint = ("the state still drifts monotonically through each period (the boundary change equals the per-period "
                "excursion), i.e. the transient is far from decayed; raise max_periods" + accel_hint)
    elif ratio is None or ratio >= 1.0:
        hint = ("the boundary changes are not contracting (ratio >= 1): either the transient has not started to decay "
                "or the changes have reached the solver's period-to-period noise level (relative tolerance "
                f"rtol={cfg.rtol:g}, {n_steps_last} steps per period, change plateau in details['history']) below "
                "which tol cannot be verified - tighten rtol/atol or loosen tol")
    elif ratio >= 0.5:
        hint = "the transient contracts slowly (ratio close to 1)"
        if more is not None:
            hint += f", about {more} more periods are needed at this ratio"
        hint += "; raise max_periods" + accel_hint
    else:
        hint = "raise max_periods"
    raise ConvergenceError(
        f"periodic steady state not reached within {int(max_periods)} periods: last relative boundary change "
        f"{change:.3e} (tol {tol:.1e}), contraction ratio per period {ratio_txt}, estimated remaining change "
        f"{remaining_txt}; {hint}",
        details={"max_periods": int(max_periods), "tol": tol, "history": history, "contraction_ratio": ratio,
                 "estimated_remaining_change": remaining, "estimated_additional_periods": more,
                 "accelerate": accelerate, "extrapolations": n_extrapolations, "rejected_extrapolations": n_rejected,
                 "rtol": cfg.rtol, "n_steps_last_period": n_steps_last})
