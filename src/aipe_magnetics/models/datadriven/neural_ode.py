"""Neural ODE reference implementation: ``dy/dt = f_theta(y, u(t))`` with an MLP vector field.

Model
-----

State ``y(t)`` in ``R^S``, exogenous input ``u(t)`` in ``R^I`` (``I`` may be 0), and a vector field
given by a multi-layer perceptron (numpy backend, float64)::

    z_in  = [norm_y(y), norm_u(u)]            per-feature input normalisation (affine kinds only)
    z_out = MLP_theta(z_in)                   dense stack from the architecture descriptor
    dy/dt = denorm(z_out)                     per-state output de-normalisation (affine kinds only)

Only affine normalisations (``none``, ``zscore``, ``minmax``) are allowed for the vector field,
because derivatives and states may be negative; ``log10`` is rejected at construction.

Integration
-----------

``scipy.integrate.solve_ivp`` with an explicit Runge-Kutta method (default ``DOP853``, ``rtol=1e-8``,
``atol=1e-10``) over ``[t_eval[0], t_eval[-1]]``; the solution is reported at ``t_eval``. Solver
settings are validated before the integrator runs (``method`` one of :data:`SOLVE_IVP_METHODS` for
``solve_ivp``, ``rtol``/``atol`` finite and positive, ``max_step`` positive or ``inf``); anything the
integrator itself rejects surfaces as :class:`InvalidInputError`. A solver failure raises
:class:`ConvergenceError`; non-finite states raise :class:`NumericalError`. The integrator is
injectable (``integrator=`` callable with the ``solve_ivp`` signature) so the dynamic/ODE framework
of ``aipe_magnetics.models.dynamic`` can supply its checked solver; whether that module is importable
is reported in ``diagnostics["dynamic_ode_module_available"]`` (its API is not assumed).

Exogenous input ``u``: a callable ``u(t) -> array(I,)`` or a pair ``(t_u, U)`` of sampled values
(``U`` of shape ``(len(t_u), I)``) that is linearly interpolated; ``t_u`` must cover the integration
interval (no extrapolation).

Validity domain: a Neural ODE has no feature-to-condition mapping, so the operating conditions are
given explicitly (``conditions={"temperature": Quantity(...)}``, ``material=``, ``excitation=``,
``dc_bias=``) with exactly the conventions of :class:`DataDrivenModel`: one scalar (or a
:class:`Quantity`, dimension-checked and converted to SI) per condition, physically impossible
values rejected as :class:`InvalidInputError`, and ``policy`` deciding what happens outside the
declared domain.

Reference case (used by the verification check ``datadriven.neural-ode-linear-case-closed-form``)
-------------------------------------------------------------------------------------------------

With identity activations and weights chosen so that ``MLP([y, u]) = -a*y + b*u`` the model *is*
the linear ODE ``dy/dt = -a y + b u(t)``. For ``u(t) = U0 + U1 sin(w t)`` the closed form is::

    y(t) = C exp(-a t) + b U0 / a + b U1 (a sin(w t) - w cos(w t)) / (a^2 + w^2)
    C    = y0 - b U0 / a + b U1 w / (a^2 + w^2)

and the numerical trajectory must agree with it to better than 1e-6 (relative to the state scale).
"""

from __future__ import annotations

import importlib
import math
import numbers
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from aipe_magnetics.common.errors import AIPEMagneticsError, ConvergenceError, InvalidInputError, NumericalError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import Assumption, Limitation, Warning_
from aipe_magnetics.models.contract import DomainPolicy, MagneticModel, ModelOutput
from aipe_magnetics.models.datadriven.adapter import (
    Backend,
    DataDrivenModel,
    ModelPackage,
    _to_array,
    condition_array,
    reject_impossible_conditions,
)
from aipe_magnetics.models.datadriven.architectures import NeuralODEHyperparameters

Integrator = Callable[..., Any]
DEFAULT_METHOD = "DOP853"
DEFAULT_RTOL = 1e-8
DEFAULT_ATOL = 1e-10
SOLVE_IVP_METHODS: tuple[str, ...] = ("RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA")


def _validate_solver_settings(method: Any, rtol: Any, atol: Any, max_step: Any, *, integrator: Integrator
                              ) -> tuple[str, float, float, float]:
    """Validate solver settings up front; returns ``(method, rtol, atol, max_step)`` as plain types."""
    if not isinstance(method, str) or not method:
        raise InvalidInputError(f"method must be a non-empty solver name, got {method!r}")
    if integrator is solve_ivp and method not in SOLVE_IVP_METHODS:
        raise InvalidInputError(f"unknown solve_ivp method {method!r}; choose one of {', '.join(SOLVE_IVP_METHODS)}",
                                details={"method": method})
    checked: dict[str, float] = {}
    for name, value in (("rtol", rtol), ("atol", atol), ("max_step", max_step)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise InvalidInputError(f"{name} must be a real number, got {value!r}")
        number = float(value)
        if math.isnan(number) or number <= 0:
            raise InvalidInputError(f"{name} must be positive" + (" (or inf)" if name == "max_step" else ""), details={name: number})
        if name != "max_step" and math.isinf(number):
            raise InvalidInputError(f"{name} must be finite and positive", details={name: number})
        checked[name] = number
    return method, checked["rtol"], checked["atol"], checked["max_step"]


def dynamic_ode_module_available() -> bool:
    """True when ``aipe_magnetics.models.dynamic.ode`` can be imported (its API is not relied upon)."""
    try:
        importlib.import_module("aipe_magnetics.models.dynamic.ode")
    except Exception:  # noqa: BLE001 - absence or a broken sibling module both mean "not available"
        return False
    return True


def linear_forced_response(t: np.ndarray, *, y0: float, a: float, b: float, u0: float, u1: float = 0.0,
                           omega: float = 0.0) -> np.ndarray:
    """Closed-form solution of ``dy/dt = -a y + b (u0 + u1 sin(omega t))`` (see module docstring)."""
    t = np.asarray(t, dtype=np.float64)
    if a == 0:
        raise InvalidInputError("the closed form needs a != 0")
    denom = a * a + omega * omega
    c = y0 - b * u0 / a + b * u1 * omega / denom
    return c * np.exp(-a * t) + b * u0 / a + b * u1 * (a * np.sin(omega * t) - omega * np.cos(omega * t)) / denom


class NeuralODEModel(MagneticModel):
    """Executable Neural-ODE family member built on a ``neural_ode`` :class:`ModelPackage`."""

    def __init__(self, package: ModelPackage, *, backend: Backend | None = None, method: str = DEFAULT_METHOD,
                 rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL, max_step: float = math.inf,
                 integrator: Integrator | None = None) -> None:
        if package.architecture.family != "neural_ode":
            raise InvalidInputError(f"NeuralODEModel needs a 'neural_ode' package, got family {package.architecture.family!r}")
        hp = package.architecture.typed_hyperparameters()
        if not isinstance(hp, NeuralODEHyperparameters):
            raise InvalidInputError("NeuralODEModel needs an MLP vector field (hyperparameters.kind='mlp_vector_field')")
        config = package.inference
        for name in package.architecture.input_features:
            if not config.input_scaler(name).is_affine:
                raise InvalidInputError(f"neural_ode: input normalisation of {name!r} must be affine (none/zscore/minmax)")
        for name in package.architecture.output_features:
            if not config.output_scaler(name).is_affine:
                raise InvalidInputError(f"neural_ode: output de-normalisation of {name!r} must be affine")
        self._core = DataDrivenModel(package, backend=backend)  # integrity check + backend.load
        if not hasattr(self._core.backend, "forward"):
            raise InvalidInputError("the Neural-ODE integrator needs a backend exposing forward(x) (numpy backend)")
        self.package = package
        self.identity = package.identity
        self.domain = package.domain
        self.sources = list(package.sources)
        self.parameters = []
        self.hyperparameters = hp
        self.state_features = list(package.architecture.input_features[:hp.state_size])
        self.input_features = list(package.architecture.input_features[hp.state_size:])
        self.integrator: Integrator = integrator if integrator is not None else solve_ivp
        self.method, self.rtol, self.atol, self.max_step = _validate_solver_settings(
            method, rtol, atol, max_step, integrator=self.integrator)
        self.integrity = self._core.integrity
        self.load_warnings = list(self._core.load_warnings)

    # -- vector field -------------------------------------------------------------------------------
    def vector_field(self, y: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        """``dy/dt`` in physical units for state ``y (S,)`` or batch ``(N, S)`` and input ``u (I,)`` / ``(N, I)``."""
        config = self.package.inference
        y = np.asarray(y, dtype=np.float64)
        batched = y.ndim == 2
        y2 = y if batched else y[None, :]
        if y2.shape[1] != self.hyperparameters.state_size:
            raise InvalidInputError(f"state has {y2.shape[1]} components, the model has {self.hyperparameters.state_size}")
        n_in = self.hyperparameters.input_size
        if n_in:
            if u is None:
                raise InvalidInputError(f"the vector field needs {n_in} exogenous input(s)")
            u2 = np.asarray(u, dtype=np.float64)
            u2 = u2 if u2.ndim == 2 else np.broadcast_to(u2.reshape(1, -1), (y2.shape[0], u2.size))
            if u2.shape != (y2.shape[0], n_in):
                raise InvalidInputError(f"input has shape {u2.shape}, expected ({y2.shape[0]}, {n_in})")
        else:
            u2 = np.zeros((y2.shape[0], 0))
        columns = [config.input_scaler(name).forward(y2[:, j], name=name) for j, name in enumerate(self.state_features)]
        columns += [config.input_scaler(name).forward(u2[:, j], name=name) for j, name in enumerate(self.input_features)]
        z = self._core.backend.forward(np.stack(columns, axis=-1))  # type: ignore[attr-defined]
        dydt = np.stack([config.output_scaler(name).inverse(z[:, j]) for j, name in enumerate(self.state_features)], axis=-1)
        return dydt if batched else dydt[0]

    # -- excitation ---------------------------------------------------------------------------------
    def _input_function(self, u: Any, t0: float, t1: float) -> Callable[[float], np.ndarray]:
        n_in = self.hyperparameters.input_size
        if n_in == 0:
            if u is not None:
                raise InvalidInputError("this Neural ODE is autonomous (input_size=0); do not pass u")
            return lambda t: np.zeros(0)
        if u is None:
            raise InvalidInputError(f"u(t) is required: {n_in} exogenous input(s) {self.input_features}")
        if callable(u):
            def evaluate(t: float) -> np.ndarray:
                value = np.asarray(u(t), dtype=np.float64).reshape(-1)
                if value.shape != (n_in,) or not np.all(np.isfinite(value)):
                    raise InvalidInputError(f"u(t) must return {n_in} finite value(s), got {value}")
                return value
            return evaluate
        if isinstance(u, (tuple, list)) and len(u) == 2:
            t_u = _to_array("t_u", u[0], "s").reshape(-1)
            values = _to_array("u", u[1], "1")
            values = values.reshape(-1, 1) if values.ndim == 1 else values
            if t_u.ndim != 1 or t_u.size < 2 or np.any(np.diff(t_u) <= 0):
                raise InvalidInputError("t_u must be a strictly increasing 1-D array with at least two samples")
            if values.shape != (t_u.size, n_in):
                raise InvalidInputError(f"sampled u must have shape ({t_u.size}, {n_in}), got {values.shape}")
            if t_u[0] > t0 or t_u[-1] < t1:
                raise InvalidInputError("sampled u must cover the whole integration interval (no extrapolation)")
            return lambda t: np.array([np.interp(t, t_u, values[:, j]) for j in range(n_in)])
        raise InvalidInputError("u must be a callable t -> array(I,) or a pair (t_u, U) of sampled values")

    # -- simulation ---------------------------------------------------------------------------------
    def simulate(self, *, y0: Sequence[float] | np.ndarray, t_eval: Sequence[float] | np.ndarray, u: Any = None,
                 policy: DomainPolicy = "reject", conditions: Mapping[str, Any] | None = None,
                 material: str | None = None, excitation: str | None = None, dc_bias: bool | None = None,
                 method: str | None = None, rtol: float | None = None, atol: float | None = None,
                 max_step: float | None = None) -> ModelOutput:
        """Integrate the Neural ODE from ``y0`` and report the trajectory at ``t_eval``.

        ``conditions`` gives the operating conditions (SI numbers or :class:`Quantity`) that the
        validity domain is checked against; the model has no input feature carrying them.
        """
        if policy not in ("reject", "warn"):
            raise InvalidInputError(f"policy must be 'reject' or 'warn', got {policy!r}")
        state0 = _to_array("y0", y0, "1").reshape(-1)
        if state0.shape != (self.hyperparameters.state_size,):
            raise InvalidInputError(f"y0 must have {self.hyperparameters.state_size} component(s), got {state0.shape}")
        times = _to_array("t_eval", t_eval, "s").reshape(-1)
        if times.size < 2 or np.any(np.diff(times) <= 0):
            raise InvalidInputError("t_eval must be strictly increasing with at least two samples")
        # same conventions as DataDrivenModel: SI/Quantity conversion, one value per condition, impossible values rejected
        explicit = {key: condition_array(key, value, n_batch=1) for key, value in dict(conditions or {}).items()}
        reject_impossible_conditions(explicit)
        cond = {key: float(values[0]) for key, values in explicit.items()}
        domain = self.check_domain(material=material, excitation=excitation, dc_bias=dc_bias, **cond)
        if not domain.inside and policy == "reject":
            domain.raise_if_outside(self.identity.id)
        u_fn = self._input_function(u, float(times[0]), float(times[-1]))
        method, rtol, atol, max_step = _validate_solver_settings(
            self.method if method is None else method, self.rtol if rtol is None else rtol,
            self.atol if atol is None else atol, self.max_step if max_step is None else max_step,
            integrator=self.integrator)

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return self.vector_field(y, u_fn(t))

        try:
            solution = self.integrator(rhs, (float(times[0]), float(times[-1])), state0, t_eval=times, method=method,
                                       rtol=rtol, atol=atol, max_step=max_step)
        except AIPEMagneticsError:
            raise  # typed errors raised inside the right-hand side (e.g. by u(t)) pass through unchanged
        except ValueError as error:  # an argument the integrator itself refuses
            raise InvalidInputError(f"the integrator rejected its arguments: {error}",
                                    details={"method": method, "rtol": rtol, "atol": atol, "max_step": max_step}) from None
        if not getattr(solution, "success", False):
            raise ConvergenceError(f"ODE integration failed: {getattr(solution, 'message', 'no message')}",
                                   details={"method": method, "rtol": rtol, "atol": atol})
        trajectory = np.asarray(solution.y, dtype=np.float64).T  # (len(t), S)
        if trajectory.shape != (times.size, self.hyperparameters.state_size) or not np.all(np.isfinite(trajectory)):
            raise NumericalError("ODE solution is incomplete or non-finite", details={"shape": list(trajectory.shape)})

        spec = self.package.architecture
        values = {name: Quantity(value=float(trajectory[-1, j]), unit=spec.unit_of(name))
                  for j, name in enumerate(self.state_features)}
        warnings = list(self.load_warnings)
        if not domain.inside:
            warnings.append(Warning_(code="out_of_domain", message="; ".join(domain.violations)))
        diagnostics: dict[str, Any] = {
            "integrator": "scipy.integrate.solve_ivp" if self.integrator is solve_ivp else getattr(self.integrator, "__name__", "custom"),
            "method": method, "rtol": rtol, "atol": atol, "max_step": None if math.isinf(max_step) else max_step,
            "nfev": int(getattr(solution, "nfev", 0)), "njev": int(getattr(solution, "njev", 0) or 0),
            "nlu": int(getattr(solution, "nlu", 0) or 0), "status": int(getattr(solution, "status", 0)),
            "message": str(getattr(solution, "message", "")),
            "dynamic_ode_module_available": dynamic_ode_module_available(),
            "weights": dict(self.integrity),
            "trajectory": {"t": times.tolist(), **{name: trajectory[:, j].tolist() for j, name in enumerate(self.state_features)}},
        }
        assumptions = [Assumption(id="initial_state", basis="user_input", affects=list(self.state_features),
                                  statement=f"Initial state y0 = {state0.tolist()} at t = {float(times[0]):g} s."),
                       Assumption(id="excitation", basis="user_input", affects=list(self.state_features),
                                  statement="u(t) supplied by the caller (callable or linearly interpolated samples)."
                                  if self.hyperparameters.input_size else "Autonomous system (no exogenous input).")]
        limitations = [Limitation(code="data_driven_surrogate", statement=(
            "Neural-ODE vector field learned from data; no physical guarantee outside the training domain; "
            "uncertainty unknown."))]
        if self.package.synthetic:
            limitations.append(Limitation(code="synthetic_parameters",
                                          statement="Vector-field parameters are synthetic/illustrative."))
        for i, statement in enumerate(self.domain.known_limitations, start=1):
            limitations.append(Limitation(code=f"known_limitation_{i}", statement=statement))
        return ModelOutput(model=self.identity, values=values, domain=domain, assumptions=assumptions,
                           limitations=limitations, warnings=warnings, sources=list(self.sources), diagnostics=diagnostics)

    def evaluate(self, *args: Any, policy: DomainPolicy = "reject", **kwargs: Any) -> ModelOutput:
        if args:
            raise InvalidInputError("evaluate() takes keyword arguments only: y0=, t_eval=, u=, ...")
        return self.simulate(policy=policy, **kwargs)


__all__ = ["DEFAULT_ATOL", "DEFAULT_METHOD", "DEFAULT_RTOL", "NeuralODEModel", "SOLVE_IVP_METHODS",
           "dynamic_ode_module_available", "linear_forced_response"]
