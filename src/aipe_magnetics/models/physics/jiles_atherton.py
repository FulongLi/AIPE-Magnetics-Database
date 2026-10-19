"""Scalar Jiles-Atherton hysteresis model as an ODE driven by ``H(t)``.

References
----------
* D. C. Jiles and D. L. Atherton, "Theory of ferromagnetic hysteresis", J. Magn. Magn. Mater. 61
  (1986) 48-60 (effective field, Langevin anhysteretic, irreversible/reversible decomposition).
* D. C. Jiles, J. B. Thoelke and M. K. Devine, "Numerical determination of hysteresis parameters
  for the modeling of magnetic properties using the theory of ferromagnetic hysteresis", IEEE
  Trans. Magn. 28 (1992) 27-35 (standard field-driven differential form, ``k`` in A/m).
* D. C. Jiles, "A self consistent generalized model for the calculation of minor loop excursions
  in the theory of hysteresis", IEEE Trans. Magn. 28 (1992) 2602-2604 (the ``delta_M`` factor).

Equations implemented (SI; ``H``, ``M`` in A/m, ``B`` in T)
-----------------------------------------------------------
Parameters: ``Ms`` saturation magnetisation (A/m), ``a`` anhysteretic shape parameter (A/m),
``alpha`` inter-domain coupling (1), ``k`` pinning parameter (A/m), ``c`` reversibility (1, 0 <= c < 1).

::

    He      = H + alpha * M                                   effective field
    M_an    = Ms * L(He / a),   L(x) = coth x - 1/x           Langevin anhysteretic (anhysteretic.py)
    chi_e   = dM_an/dHe = (Ms / a) * L'(He / a)
    M       = M_irr + M_rev,    M_rev = c * (M_an - M_irr)    =>  M = (1 - c) M_irr + c M_an
    M_irr   = (M - c * M_an) / (1 - c)                        (explicit in the state M)
    g       = M_an - M_irr = (M_an - M) / (1 - c)
    delta   = sign(dH/dt)        (+1 ascending, -1 descending; dH/dt = 0 gives dM/dt = 0 exactly)
    delta_M = 1 if g * delta > 0 else 0                       (no irreversible change against the drive)

    dM_irr/dH = delta_M * g / (delta * k - alpha * g)         (Jiles-Thoelke-Devine 1992 form)
    dM_an/dH  = chi_e * (1 + alpha * dM/dH)                   (chain rule through He)
    dM/dH     = (1 - c) * dM_irr/dH + c * dM_an/dH

Solving the last three relations for ``dM/dH`` gives the closed form that is integrated::

    dM/dH = [ (1 - c) * delta_M * g / (delta * k - alpha * g) + c * chi_e ] / (1 - alpha * c * chi_e)

or, expressed through the state ``M`` (``(1 - c) g = M_an - M``)::

    dM/dH = [ delta_M * (1 - c) * (M_an - M) / ((1 - c) * delta * k - alpha * (M_an - M)) + c * chi_e ]
            / (1 - alpha * c * chi_e)

    dM/dt = (dM/dH) * dH/dt,        B = mu_0 * (H + M)

Remarks on the closed form (which variant is implemented)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* The irreversible term is the Jiles-Thoelke-Devine (1992) expression
  ``dM_irr/dH = (M_an - M_irr) / (delta k - alpha (M_an - M_irr))``, whose denominator follows from
  ``dHe/dH = 1 + alpha dM_irr/dH``, i.e. the effective-field feedback of the *irreversible*
  component alone.  The reversible term ``c dM_an/dH`` uses the full chain rule
  ``dHe/dH = 1 + alpha dM/dH`` with the total magnetisation.  This mixed form is the one most often
  implemented in the literature; the two feedbacks are *not* the same algebra.  A fully consistent
  alternative (``dHe/dH = 1 + alpha dM/dH`` in both terms) reads
  ``dM/dH = A / (1 - alpha A)`` with ``A = (1 - c) delta_M g / (k delta) + c chi_e``; it differs
  substantially from the implemented form near the pinning barrier ``alpha |g| -> k`` and is a
  known variant, not what this module computes.  Some authors also drop the ``alpha dM/dH``
  feedback in the reversible term altogether (``dM_an/dH ~ chi_e``).
* The denominator ``1 - alpha c chi_e`` must stay positive; since ``chi_e <= Ms/(3a)`` this is the
  parameter condition ``alpha c Ms / (3a) < 1`` enforced at construction.
* With ``delta_M = 1`` one has ``g delta > 0`` and the irreversible denominator equals
  ``delta (k - alpha |g|)``.  Along a continuous trajectory ``|g|`` stays below ``k / alpha``
  (``|M_an - M| < (1 - c) k / alpha``, the *admissible gap*) because the irreversible
  susceptibility diverges as it is approached; if a trial step of the adaptive solver nevertheless
  reaches ``k - alpha |g| <= 0`` the right-hand side returns ``nan`` so that the step is rejected
  (counted in the solver diagnostics) instead of producing a nonphysical negative susceptibility.

Initial susceptibility (used as an analytical reference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
At the demagnetised origin (``H = 0``, ``M = 0``, ``M_an = 0``) the irreversible term vanishes with
``g`` and ``chi_e = Ms/(3a)``, hence::

    chi_0 = dM/dH |_(0,0) = c * Ms/(3a) / (1 - alpha * c * Ms/(3a))

The anhysteretic curve itself has initial susceptibility ``(Ms/3a) / (1 - alpha Ms/3a)
= Ms / (3a - alpha Ms)``, finite and positive only for ``alpha Ms / (3a) < 1``.  For
``alpha Ms / (3a) >= 1`` the anhysteretic ``M_an(H)`` is multivalued near the origin, the
demagnetised state is not a usable starting point and the integration runs into the pinning
barrier; such parameter sets are rejected at construction (:class:`InvalidInputError`).

Initial state and pre-magnetisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The state must be *admissible* at the start of the drive: with ``delta_0 = sign(dH/dt(t0))`` and
``g_0 = (M_an - M)(t0)``, either ``g_0 delta_0 <= 0`` (reversible start) or
``alpha |g_0| < (1 - c) k`` (inside the admissible gap).  The demagnetised state ``M = 0`` is
admissible only at ``H(t0) = 0``; for any excitation with ``H(t0) != 0`` (DC offset, non-zero phase,
triangular waves, measured records) and no explicit ``initial_state``, :meth:`JilesAthertonModel.simulate`
*pre-magnetises* the demagnetised sample along the initial magnetisation curve with a monotone ramp
``H: 0 -> H(t0)`` integrated before ``t0`` (the model is rate independent, so the ramp duration is
immaterial; the same solver settings are used).  The ramp end state is the initial state of the
requested window, is reported in the diagnostics (``premagnetisation``) and stated as an assumption.
A constant field therefore means "ramp from the demagnetised state to ``H0`` and hold": ``M`` is the
initial-curve magnetisation at ``H0`` (not the vacuum response ``M = 0``).  A user-supplied
``initial_state`` is used as is but validated for admissibility at ``(H(t0), dH/dt(t0))``; an
inadmissible pair raises :class:`InvalidInputError` with the admissible gap in the message.

Periodic steady state
~~~~~~~~~~~~~~~~~~~~~
Periodic excitations are integrated period by period with
:func:`aipe_magnetics.models.dynamic.ode.periodic_steady_state` (contraction-aware criterion, boundary
change relative to the peak-to-peak excursion of ``M``).  Minor loops of the standard model do not
close, so the loop drifts towards its steady position with a contraction ratio per period that
approaches 1 for small field amplitudes (0.99 at a few A/m for the illustrative set: hundreds of
periods by plain iteration).  ``simulate`` therefore enables the Steffensen extrapolation of the
period map by default (``accelerate=True``), which reaches the same steady loop in a handful of
periods; the result is always verified by plain iterations.

Conventions and outputs
~~~~~~~~~~~~~~~~~~~~~~~
``B`` and ``H`` are positive along the same reference direction, so a dissipative loop is traversed
counter-clockwise in the ``(H, B)`` plane and ``w = oint H dB >= 0`` (J/m^3 per cycle).  Remanence is
``|B|`` at the ``H = 0`` crossings and coercivity ``|H|`` at the ``B = 0`` crossings of the last period
(linear interpolation between samples, strict sign changes only, the loop treated cyclically).  The
scalar ``remanence`` / ``coercivity`` values are the mean magnitude over the crossings: for a
symmetric loop both crossings have the same magnitude; for an asymmetric (DC-biased) loop the two
differ, the individual crossings are reported in the diagnostics and a ``ja.asymmetric_loop``
warning is attached.  When ``H`` (or ``B``) never crosses zero the corresponding scalar is omitted.

Known limitations (declared in the validity domain): scalar (one-dimensional) model; rate
independent, i.e. no eddy-current or excess loss and no frequency dependence of the loop; minor
loops of the standard model do not close exactly; parameters must be identified per material -
the shipped parameter set is illustrative only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aipe_magnetics import __version__
from aipe_magnetics.common.errors import ConvergenceError, InvalidInputError, NumericalError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import Assumption, Limitation, ModelIdentity, SourceReference, Warning_
from aipe_magnetics.common.units import MU_0
from aipe_magnetics.models.contract import (
    DomainPolicy,
    HysteresisModel,
    ModelOutput,
    ModelParameter,
    ValidityDomain,
)
from aipe_magnetics.models.dynamic.excitation import Excitation
from aipe_magnetics.models.dynamic.ode import ODEModel, SolverConfig, periodic_steady_state, solve
from aipe_magnetics.models.dynamic.solution import (
    ODESolution,
    cycle_energy_density,
    interpolated_crossings,
    loop_closure_error,
)
from aipe_magnetics.models.physics.anhysteretic import (
    langevin,
    langevin_derivative,
    langevin_derivative_scalar,
    langevin_scalar,
)
from aipe_magnetics.models.physics.reference_parameters import (
    JA_1986,
    JILES_1992_MINOR_LOOPS,
    JTD_1992,
    JA_ILLUSTRATIVE,
    JilesAthertonParameterSet,
)

__all__ = ["JA_MODEL_ID", "JilesAthertonODE", "JilesAthertonModel", "JATrace", "loop_metrics", "SIMULATE_KWARGS"]

JA_MODEL_ID = "aipe.magnetics.model.jiles_atherton"
_KIND_MAP = {"constant": "dc", "sinusoidal": "sinusoidal", "triangular": "triangular",
             "piecewise_linear": "piecewise_linear", "sampled": "sampled", "custom": "custom"}
#: keyword arguments accepted by :meth:`JilesAthertonModel.simulate` (and forwarded by ``trajectory``)
SIMULATE_KWARGS = frozenset({"config", "max_periods", "steady_state_tol", "points_per_period", "initial_state",
                             "t_span", "steady_state", "accelerate"})
#: relative difference between the two crossing magnitudes above which an *unbiased* loop is flagged
#: asymmetric with severity 'warning' (a biased drive is always flagged with severity 'info').  Below
#: it the asymmetry is the residual steady-state offset of M amplified by ptp(M) / M_r, i.e. the
#: steady-state tolerance expressed in remanence units, not a physical asymmetry.
LOOP_ASYMMETRY_TOLERANCE = 1e-2


def _real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise InvalidInputError(f"{name} must be a real number, got {type(value).__name__}", details={"name": name})
    value = float(value)
    if not math.isfinite(value):
        raise InvalidInputError(f"{name} must be finite, got {value}", details={"name": name})
    return value


def _span(t_span: Any) -> tuple[float, float]:
    try:
        t0, t1 = (float(v) for v in t_span)
    except (TypeError, ValueError):
        raise InvalidInputError("t_span must be a pair (t0, t1) of numbers") from None
    if not (math.isfinite(t0) and math.isfinite(t1)) or t1 <= t0:
        raise InvalidInputError(f"t_span must be finite with t1 > t0, got ({t0}, {t1})")
    return t0, t1


class JilesAthertonODE(ODEModel):
    """``dM/dt = chi(H, M, sign(dH/dt)) * dH/dt`` with the closed-form ``chi`` of the module docstring."""

    state_names = ("M",)
    default_initial_state = {"M": 0.0}  # demagnetised state; pass M=... for a remanent start
    requires_du_dt = True
    name = "jiles_atherton"

    def __init__(self, Ms: float, a: float, alpha: float, k: float, c: float):
        self.Ms = _real("Ms", Ms)
        self.a = _real("a", a)
        self.alpha = _real("alpha", alpha)
        self.k = _real("k", k)
        self.c = _real("c", c)
        if self.Ms <= 0.0:
            raise InvalidInputError(f"Ms must be positive, got {self.Ms}", details={"name": "Ms"})
        if self.a <= 0.0:
            raise InvalidInputError(f"a must be positive, got {self.a}", details={"name": "a"})
        if self.k <= 0.0:
            raise InvalidInputError(f"k must be positive, got {self.k}", details={"name": "k"})
        if self.alpha < 0.0:
            raise InvalidInputError(f"alpha must be non-negative, got {self.alpha}", details={"name": "alpha"})
        if not 0.0 <= self.c < 1.0:
            raise InvalidInputError(f"c must satisfy 0 <= c < 1, got {self.c}", details={"name": "c"})
        self.chi_e0 = self.Ms / (3.0 * self.a)
        self.anhysteretic_feedback = self.alpha * self.chi_e0
        if self.anhysteretic_feedback >= 1.0:
            raise InvalidInputError(
                f"alpha*Ms/(3a) = {self.anhysteretic_feedback:.4g} >= 1: the anhysteretic curve M_an(H) is multivalued "
                "near the origin (its initial susceptibility Ms/(3a - alpha Ms) is not finite and positive); the "
                "demagnetised state is not a usable starting point and the integration runs into the pinning "
                "barrier. Identify the parameters with a > alpha Ms / 3",
                details={"alpha_Ms_over_3a": self.anhysteretic_feedback})
        self.reversible_feedback = self.alpha * self.c * self.chi_e0
        if self.reversible_feedback >= 1.0:  # implied by the previous test; kept as an explicit, named condition
            raise InvalidInputError(
                f"alpha*c*Ms/(3a) = {self.reversible_feedback:.4g} >= 1: the reversible feedback 1 - alpha c chi_e "
                "becomes singular; the parameter set is outside the model's admissible range",
                details={"alpha_c_Ms_over_3a": self.reversible_feedback})
        self.parameters = {
            "Ms": ModelParameter(name="Ms", value=self.Ms, unit="A/m", description="saturation magnetisation"),
            "a": ModelParameter(name="a", value=self.a, unit="A/m", description="anhysteretic shape parameter"),
            "alpha": ModelParameter(name="alpha", value=self.alpha, unit="1", description="inter-domain coupling"),
            "k": ModelParameter(name="k", value=self.k, unit="A/m", description="pinning parameter"),
            "c": ModelParameter(name="c", value=self.c, unit="1", description="reversibility coefficient"),
        }
        super().__init__()

    # -- analytic properties --------------------------------------------------------------------
    def initial_susceptibility(self) -> float:
        """``chi_0 = c Ms/(3a) / (1 - alpha c Ms/(3a))`` at the demagnetised origin."""
        return self.c * self.chi_e0 / (1.0 - self.reversible_feedback)

    def anhysteretic_initial_susceptibility(self) -> float:
        """``(Ms/3a) / (1 - alpha Ms/3a) = Ms / (3a - alpha Ms)`` (finite: enforced at construction)."""
        return self.chi_e0 / (1.0 - self.anhysteretic_feedback)

    @property
    def admissible_gap(self) -> float:
        """``(1 - c) k / alpha``: the bound on ``|M_an - M|`` along the irreversible branch (inf for alpha = 0)."""
        if self.alpha == 0.0:
            return math.inf
        return (1.0 - self.c) * self.k / self.alpha

    # -- constitutive pieces --------------------------------------------------------------------
    def effective_field(self, H: Any, M: Any) -> Any:
        return np.asarray(H, dtype=float) + self.alpha * np.asarray(M, dtype=float)

    def anhysteretic(self, He: Any) -> Any:
        return self.Ms * langevin(np.asarray(He, dtype=float) / self.a)

    def anhysteretic_susceptibility(self, He: Any) -> Any:
        return (self.Ms / self.a) * langevin_derivative(np.asarray(He, dtype=float) / self.a)

    def susceptibility(self, H: float, M: float, delta: int) -> float:
        """Closed-form ``dM/dH`` for drive direction ``delta`` (+1/-1); ``nan`` beyond the admissible gap."""
        He = H + self.alpha * M
        x = He / self.a
        M_an = self.Ms * langevin_scalar(x)
        chi_e = (self.Ms / self.a) * langevin_derivative_scalar(x)
        diff = M_an - M  # = (1 - c) * g
        irreversible = 0.0
        if diff * delta > 0.0:
            denominator = (1.0 - self.c) * delta * self.k - self.alpha * diff
            if denominator * delta <= 0.0:
                return math.nan  # k - alpha |g| <= 0: outside the admissible region (trial step rejected)
            irreversible = (1.0 - self.c) * diff / denominator
        return (irreversible + self.c * chi_e) / (1.0 - self.alpha * self.c * chi_e)

    def check_initial_state(self, H: float, M: float, dH_dt: float) -> None:
        """Raise :class:`InvalidInputError` when ``(H, M)`` is not admissible for the drive direction ``dH_dt``."""
        H, M, dH_dt = _real("H", H), _real("M", M), _real("dH_dt", dH_dt)
        if dH_dt == 0.0:
            return  # static start: dM/dt = 0 whatever the state
        delta = 1 if dH_dt > 0.0 else -1
        if math.isfinite(self.susceptibility(H, M, delta)):
            return
        gap = float(self.anhysteretic(H + self.alpha * M)) - M
        raise InvalidInputError(
            f"initial magnetisation M(t0) = {M:g} A/m is not admissible at H(t0) = {H:g} A/m with dH/dt "
            f"{'>' if delta > 0 else '<'} 0: |M_an - M| = {abs(gap):g} A/m reaches the admissible gap "
            f"(1 - c) k / alpha = {self.admissible_gap:g} A/m where the irreversible susceptibility diverges; "
            "choose M(t0) closer to the anhysteretic curve or omit initial_state so that the demagnetised sample is "
            "ramped to H(t0) along its initial curve",
            details={"H": H, "M": M, "gap": gap, "admissible_gap": self.admissible_gap, "delta": delta})

    # -- ODE contract ---------------------------------------------------------------------------
    def rhs(self, t: float, y: np.ndarray, u: float, du_dt: float | None) -> list[float]:  # noqa: ARG002
        if du_dt is None:
            raise InvalidInputError("Jiles-Atherton needs dH/dt; the excitation must provide a derivative")
        if du_dt == 0.0:
            return [0.0]
        delta = 1 if du_dt > 0.0 else -1
        return [self.susceptibility(float(u), float(y[0]), delta) * du_dt]

    def outputs(self, t: np.ndarray, y: np.ndarray, u: np.ndarray) -> dict[str, np.ndarray]:  # noqa: ARG002
        H = np.asarray(u, dtype=float)
        M = y[0]
        He = H + self.alpha * M
        M_an = self.anhysteretic(He)
        return {"H": H, "B": MU_0 * (H + M), "He": He, "M_an": M_an, "M_irr": (M - self.c * M_an) / (1.0 - self.c),
                "B_an": MU_0 * (H + M_an)}

    def constraints(self, t: np.ndarray, y: np.ndarray) -> list[str]:  # noqa: ARG002
        peak = float(np.max(np.abs(y[0])))
        if peak > self.Ms:
            return [f"|M| = {peak:.6g} A/m exceeds Ms = {self.Ms:.6g} A/m"]
        return []


@dataclass
class JATrace:
    """Raw simulation result of :meth:`JilesAthertonModel.simulate` (numpy arrays)."""

    t: np.ndarray
    H: np.ndarray
    B: np.ndarray
    M: np.ndarray
    He: np.ndarray
    M_an: np.ndarray
    M_irr: np.ndarray
    solution: ODESolution
    periodic: bool
    periods: int | None
    steady_state: bool
    boundary_change: float | None
    excitation: dict[str, Any]
    history: list[float]
    initial_state: dict[str, Any] = field(default_factory=dict)
    premagnetisation: dict[str, Any] | None = None
    contraction_ratio: float | None = None
    estimated_remaining_change: float | None = None
    accelerated: bool = False
    extrapolations: int = 0


def loop_metrics(H: Any, B: Any) -> dict[str, Any]:
    """Peak values, remanence, coercivity (linear interpolation) and ``oint H dB`` of one closed loop.

    ``H`` and ``B`` sample one period; when the last sample repeats the first (closure within 1e-6 of the
    peaks) it is dropped and the loop is treated cyclically, starting at the ``|H|`` peak so that no zero
    crossing falls on the record boundary.  ``remanence`` is the mean ``|B|`` over the ``H = 0`` crossings
    and ``coercivity`` the mean ``|H|`` over the ``B = 0`` crossings (``None`` without crossings); the
    individual crossings and the relative asymmetry between the two crossing magnitudes
    (``None`` unless exactly two crossings exist) are reported alongside.
    """
    h = np.asarray(H, dtype=float)
    b = np.asarray(B, dtype=float)
    if h.ndim != 1 or h.shape != b.shape or h.size < 3:
        raise InvalidInputError("H and B must be 1-D arrays of equal length >= 3")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(b))):
        raise InvalidInputError("H and B must be finite")
    h_scale = max(float(np.max(np.abs(h))), float(np.finfo(float).tiny))
    b_scale = max(float(np.max(np.abs(b))), float(np.finfo(float).tiny))
    hc, bc = h, b
    if h.size > 3 and abs(float(h[-1] - h[0])) <= 1e-6 * h_scale and abs(float(b[-1] - b[0])) <= 1e-6 * b_scale:
        hc, bc = h[:-1], b[:-1]  # closing sample duplicates the first: cyclic record
    shift = int(np.argmax(np.abs(hc)))
    hc, bc = np.roll(hc, -shift), np.roll(bc, -shift)
    rem = interpolated_crossings(hc, bc, 0.0)
    coe = interpolated_crossings(bc, hc, 0.0)

    def asymmetry(crossings: list[dict[str, float]]) -> float | None:
        if len(crossings) != 2:
            return None
        m1, m2 = abs(crossings[0]["y"]), abs(crossings[1]["y"])
        top = max(m1, m2)
        return 0.0 if top == 0.0 else abs(m1 - m2) / top

    out: dict[str, Any] = {
        "field_peak": float(np.max(np.abs(h))),
        "flux_density_peak": float(np.max(np.abs(b))),
        "loop_energy_density": cycle_energy_density(h, b, close=True),
        "closure_error_B": loop_closure_error(b),
        "closure_error_H": loop_closure_error(h),
        "remanence_crossings": [{"B": float(c["y"]), "direction_dH": float(c["direction"])} for c in rem],
        "coercivity_crossings": [{"H": float(c["y"]), "direction_dB": float(c["direction"])} for c in coe],
        "n_remanence_crossings": len(rem),
        "n_coercivity_crossings": len(coe),
        "remanence_asymmetry": asymmetry(rem),
        "coercivity_asymmetry": asymmetry(coe),
    }
    out["remanence"] = float(np.mean([abs(c["y"]) for c in rem])) if rem else None
    out["coercivity"] = float(np.mean([abs(c["y"]) for c in coe])) if coe else None
    return out


class JilesAthertonModel(HysteresisModel):
    """Contract wrapper: ``trajectory(field)`` runs the JA ODE through the dynamic framework."""

    identity = ModelIdentity(id=JA_MODEL_ID, name="Jiles-Atherton scalar hysteresis", version=__version__,
                             family="physics", method_version="ode/solve_ivp/2")

    def __init__(self, Ms: float, a: float, alpha: float, k: float, c: float, *,
                 parameter_set: JilesAthertonParameterSet | None = None, source: SourceReference | None = None,
                 material: str | None = None):
        self.ode = JilesAthertonODE(Ms, a, alpha, k, c)
        self.parameter_set = parameter_set
        self.material = material
        param_source = source if source is not None else (parameter_set.source if parameter_set else None)
        ref = param_source.id if param_source is not None else None
        self.parameters = [p.model_copy(update={"source_ref": ref}) for p in self.ode.parameters.values()]
        self.sources = [JA_1986, JTD_1992, JILES_1992_MINOR_LOOPS] + ([param_source] if param_source else [])
        self.domain = ValidityDomain(
            materials=[material] if material else None,
            excitation=["sinusoidal", "triangular", "piecewise_linear", "sampled", "dc"],
            frequency=None,  # rate independent: the loop does not depend on frequency
            temperature=None,
            dc_bias="supported",
            requires_history=True,
            numerical_constraints=[
                "0 <= c < 1, alpha >= 0, Ms, a, k > 0",
                "alpha*Ms/(3a) < 1 (single-valued anhysteretic, Ms/(3a - alpha Ms) finite and positive) and hence "
                "alpha*c*Ms/(3a) < 1 (regular reversible feedback)",
                "excitation must provide dH/dt (analytic derivative)",
                "initial state admissible at (H(t0), dH/dt(t0)): the demagnetised state is ramped to H(t0) along the "
                "initial curve when H(t0) != 0; a supplied initial_state must satisfy |M_an - M| < (1-c)k/alpha on the "
                "irreversible branch",
                "adaptive RK45 by default with rtol 1e-8, atol 1e-9 Ms, max_step = period/200; tighten rtol for "
                "loop-energy accuracy",
                "periodic steady state: boundary change of M relative to its peak-to-peak excursion below "
                "steady_state_tol with contraction-aware extrapolation; Steffensen acceleration on by default "
                "(accelerate=True) because small-amplitude minor loops contract by less than 1 % per period",
                "steady_state_tol must stay above the solver's period-to-period noise floor (about 50 rtol relative "
                "to the excursion of M for DC-biased minor loops, far less for symmetric loops); below it the "
                "iteration cannot be verified and raises ConvergenceError",
            ],
            known_limitations=[
                "scalar (one-dimensional) model; no vector or rotational hysteresis",
                "rate independent: no eddy-current or excess loss, loop shape independent of frequency",
                "minor loops of the standard Jiles-Atherton model do not close exactly",
                "small-amplitude minor loops drift towards the steady loop with a contraction ratio per period close "
                "to 1 (hundreds of periods by plain iteration; accelerate=False needs a large max_periods)",
                "parameters must be identified per material; the shipped set is illustrative, not a material",
            ],
        )

    # -- constructors ---------------------------------------------------------------------------
    @classmethod
    def from_parameter_set(cls, parameter_set: JilesAthertonParameterSet, *, material: str | None = None
                           ) -> "JilesAthertonModel":
        return cls(**parameter_set.as_kwargs(), parameter_set=parameter_set, material=material)

    @classmethod
    def illustrative(cls) -> "JilesAthertonModel":
        """Model with the illustrative demonstration parameters (not a material)."""
        return cls.from_parameter_set(JA_ILLUSTRATIVE)

    # -- helpers --------------------------------------------------------------------------------
    def initial_susceptibility(self) -> float:
        return self.ode.initial_susceptibility()

    def default_config(self, period: float | None) -> SolverConfig:
        """RK45, ``rtol = 1e-8``, ``atol = 1e-9 Ms``, ``max_step = period / 200``.

        ``rtol`` acts on ``|M|``, not on the excursion of ``M``: for a DC-biased minor loop (``|M|`` near
        saturation, excursion a few per cent of it) the period-to-period noise of the boundary state relative
        to the excursion is about ``50 rtol`` (measured 2e-5 at rtol 1e-7, 3e-6 at 1e-8 for a 200 A/m loop at
        -1 kA/m with the illustrative set); 1e-8 keeps that floor well below the default ``steady_state_tol``
        of 1e-4 at a cost of 5 % (minor loops) to 40 % (10 kA/m major loop) more steps per period.
        """
        max_step = None if period is None else period / 200.0
        return SolverConfig(method="RK45", rtol=1e-8, atol=1e-9 * self.ode.Ms, max_step=max_step)

    @staticmethod
    def _as_excitation(field: Any) -> tuple[Excitation, np.ndarray | None]:
        """Return ``(excitation, sample_times or None)``."""
        if isinstance(field, Excitation):
            return field, None
        if isinstance(field, dict):
            times = field.get("t", field.get("times"))
            values = field.get("H", field.get("values"))
            if times is None or values is None:
                raise InvalidInputError("field dict needs keys 't'/'times' and 'H'/'values'")
        elif isinstance(field, (tuple, list)) and len(field) == 2:
            times, values = field
        elif isinstance(field, np.ndarray) and field.ndim == 2 and 2 in field.shape:
            times, values = (field[0], field[1]) if field.shape[0] == 2 else (field[:, 0], field[:, 1])
        else:
            raise InvalidInputError(
                "field must be an Excitation, a (times, H) pair, a {'t':..., 'H':...} dict or a 2 x n array")
        exc = Excitation.sampled(times, values, interpolation="linear", periodic=False)
        return exc, np.asarray(times, dtype=float)

    def _premagnetise(self, h_end: float, config: Any, duration: float, t_end: float) -> tuple[float, dict[str, Any]]:
        """Integrate the initial magnetisation curve along a monotone ramp ``H: 0 -> h_end`` ending at ``t_end``."""
        t_start = t_end - duration
        ramp = Excitation.piecewise_linear([t_start, t_end], [0.0, h_end], periodic=False)
        sol = solve(self.ode, ramp, (t_start, t_end), config, y0=[0.0], t_eval=[t_start, t_end])
        m_end = float(sol.final_state[0])
        gap = float(self.ode.anhysteretic(h_end + self.ode.alpha * m_end)) - m_end
        return m_end, {
            "method": "monotone ramp of H from 0 to H(t0) integrated from the demagnetised state before t0 "
                      "(rate-independent model: the ramp duration is immaterial)",
            "H_end_A_per_m": h_end, "M_end_A_per_m": m_end, "duration_s": duration,
            "n_steps": int(sol.diagnostics["n_steps"]), "gap_M_an_minus_M_A_per_m": gap,
            "admissible_gap_A_per_m": self.ode.admissible_gap,
        }

    # -- simulation -----------------------------------------------------------------------------
    def simulate(self, field: Any, *, config: SolverConfig | dict[str, Any] | None = None, max_periods: int = 50,
                 steady_state_tol: float = 1e-4, points_per_period: int = 1000, initial_state: float | None = None,
                 t_span: tuple[float, float] | None = None, steady_state: bool = True,
                 accelerate: bool = True) -> JATrace:
        """Run the ODE; periodic excitations are integrated to periodic steady state (last period returned).

        See the module docstring for the initial state (pre-magnetisation ramp when ``H(t0) != 0`` and no
        ``initial_state`` is given) and the steady-state criterion / acceleration.
        """
        exc, sample_times = self._as_excitation(field)
        if isinstance(points_per_period, bool) or int(points_per_period) != points_per_period or points_per_period < 16:
            raise InvalidInputError("points_per_period must be an integer >= 16")
        if isinstance(max_periods, bool) or int(max_periods) != max_periods or max_periods < 1:
            raise InvalidInputError("max_periods must be a positive integer")
        periodic = exc.period is not None and bool(steady_state)
        if periodic:
            if t_span is not None:
                raise InvalidInputError("t_span is not used for periodic excitations (steady state is integrated); "
                                        "pass steady_state=False to integrate a fixed span")
            t_begin, horizon = 0.0, float(exc.period)
        else:
            if t_span is None:
                if exc.domain is not None:
                    t_span = exc.domain
                elif exc.period is not None:
                    t_span = (0.0, exc.period)
                elif exc.kind == "constant":
                    t_span = (0.0, 1.0)  # dH/dt = 0 and rate independent: the window length is immaterial
                else:
                    raise InvalidInputError("t_span is required for a non-periodic excitation")
            t_span = _span(t_span)
            t_begin, horizon = t_span[0], t_span[1] - t_span[0]
        cfg = self.default_config(horizon) if config is None else config
        h_begin = float(exc.value(t_begin))
        dh_begin = float(exc.derivative(t_begin))
        premag: dict[str, Any] | None = None
        if initial_state is None:
            if h_begin != 0.0:
                m_begin, premag = self._premagnetise(h_begin, cfg, horizon, t_begin)
                y0: np.ndarray | None = np.array([m_begin])
                source = "premagnetisation_ramp"
            else:
                y0, m_begin, source = None, 0.0, "default"
        else:
            y0 = self.ode.initial_state(M=initial_state)
            m_begin = float(y0[0])
            self.ode.check_initial_state(h_begin, m_begin, dh_begin)
            source = "user_input"
        start = {"t0_s": t_begin, "H_A_per_m": h_begin, "dH_dt_A_per_m_s": dh_begin, "M_A_per_m": m_begin,
                 "source": source}

        if periodic:
            try:
                pss = periodic_steady_state(self.ode, exc, exc.period, cfg, max_periods, steady_state_tol, y0=y0,
                                            points_per_period=int(points_per_period), accelerate=accelerate)
            except ConvergenceError as error:
                raise ConvergenceError(
                    f"Jiles-Atherton periodic steady state not reached: {error.message}. Minor loops of the "
                    "Jiles-Atherton model do not close, so the loop drifts towards its steady position with a "
                    "contraction ratio per period close to 1 at small field amplitudes; keep accelerate=True "
                    f"(now {bool(accelerate)}), raise max_periods (now {int(max_periods)}) or loosen "
                    f"steady_state_tol (now {float(steady_state_tol):g})",
                    details={**error.details, "max_periods": int(max_periods), "steady_state_tol": float(steady_state_tol),
                             "accelerate": bool(accelerate), "cause": error.message}) from error
            sol = pss.solution
            return JATrace(t=sol.t, H=sol.outputs["H"], B=sol.outputs["B"], M=sol.y[0], He=sol.outputs["He"],
                           M_an=sol.outputs["M_an"], M_irr=sol.outputs["M_irr"], solution=sol, periodic=True,
                           periods=pss.periods, steady_state=pss.converged, boundary_change=pss.boundary_change,
                           excitation=exc.describe(), history=pss.history, initial_state=start,
                           premagnetisation=premag, contraction_ratio=pss.contraction_ratio,
                           estimated_remaining_change=pss.estimated_remaining_change, accelerated=pss.accelerated,
                           extrapolations=pss.extrapolations)
        if sample_times is not None:
            grid = sample_times
        else:
            grid = np.linspace(float(t_span[0]), float(t_span[1]), int(points_per_period) + 1)
        sol = solve(self.ode, exc, t_span, cfg, y0=y0, t_eval=grid)
        return JATrace(t=sol.t, H=sol.outputs["H"], B=sol.outputs["B"], M=sol.y[0], He=sol.outputs["He"],
                       M_an=sol.outputs["M_an"], M_irr=sol.outputs["M_irr"], solution=sol, periodic=False,
                       periods=None, steady_state=False, boundary_change=None, excitation=exc.describe(),
                       history=[], initial_state=start, premagnetisation=premag)

    # -- contract -------------------------------------------------------------------------------
    def trajectory(self, field: Any, *, policy: DomainPolicy = "reject", **kwargs: Any) -> ModelOutput:
        """Evaluate the model for ``H(t)``; see :meth:`simulate` for keyword arguments."""
        if policy not in ("reject", "warn"):
            raise InvalidInputError(f"policy must be 'reject' or 'warn', got {policy!r}")
        unknown = sorted(set(kwargs) - SIMULATE_KWARGS)
        if unknown:
            raise InvalidInputError(f"unexpected keyword argument(s) {unknown}; trajectory() forwards only "
                                    f"{sorted(SIMULATE_KWARGS)} to simulate()", details={"unknown": unknown})
        exc, _ = self._as_excitation(field)
        kind = _KIND_MAP.get(exc.kind, exc.kind)
        lo, hi = exc.extrema(0.0, exc.period) if exc.period is not None else (
            exc.extrema(*exc.domain) if exc.domain is not None else (None, None))
        has_bias = None
        if lo is not None and hi is not None:
            has_bias = bool(abs(lo + hi) > 1e-12 * max(abs(lo), abs(hi), 1.0))
        check = self.check_domain(material=self.material, excitation=kind, dc_bias=has_bias)
        if policy == "reject":
            check.raise_if_outside(self.identity.id)
        trace = self.simulate(field, **kwargs)  # the original field keeps sample times for array inputs
        warnings: list[Warning_] = []
        if not check.inside:
            warnings.append(Warning_(code="out_of_domain", message="; ".join(check.violations)))
        for violation in trace.solution.constraint_violations:
            warnings.append(Warning_(code="constraint_violation", message=violation, severity="error"))

        values: dict[str, Quantity] = {
            "field_peak": Quantity(value=float(np.max(np.abs(trace.H))), unit="A/m"),
            "flux_density_peak": Quantity(value=float(np.max(np.abs(trace.B))), unit="T"),
            "magnetisation_peak": Quantity(value=float(np.max(np.abs(trace.M))), unit="A/m"),
        }
        diagnostics: dict[str, Any] = {
            "trajectory": {"t": trace.t.tolist(), "H": trace.H.tolist(), "B": trace.B.tolist(), "M": trace.M.tolist()},
            "solver": dict(trace.solution.diagnostics),
            "periodic": trace.periodic, "periods": trace.periods, "steady_state": trace.steady_state,
            "boundary_change": trace.boundary_change, "steady_state_history": list(trace.history),
            "contraction_ratio": trace.contraction_ratio,
            "estimated_remaining_change": trace.estimated_remaining_change,
            "accelerated": trace.accelerated, "extrapolations": trace.extrapolations,
            "initial_state": dict(trace.initial_state), "premagnetisation": trace.premagnetisation,
            "excitation": trace.excitation, "constraint_violations": list(trace.solution.constraint_violations),
            "initial_susceptibility": self.ode.initial_susceptibility(),
            "anhysteretic_initial_susceptibility": self.ode.anhysteretic_initial_susceptibility(),
            "admissible_gap": self.ode.admissible_gap,
            "parameters": {k: p.value for k, p in self.ode.parameters.items()},
        }
        if trace.periodic:
            metrics = loop_metrics(trace.H, trace.B)
            if metrics["loop_energy_density"] < 0.0:
                raise NumericalError(
                    f"loop energy density {metrics['loop_energy_density']:.6g} J/m^3 is negative: nonphysical "
                    "(energy released by the material over a closed cycle)",
                    details={"loop_energy_density": metrics["loop_energy_density"]})
            values["loop_energy_density"] = Quantity(value=metrics["loop_energy_density"], unit="J/m^3")
            if metrics["remanence"] is not None:
                values["remanence"] = Quantity(value=metrics["remanence"], unit="T")
            else:
                warnings.append(Warning_(code="ja.no_remanence", message="H never crosses zero: remanence undefined",
                                         severity="info"))
            if metrics["coercivity"] is not None:
                values["coercivity"] = Quantity(value=metrics["coercivity"], unit="A/m")
            else:
                warnings.append(Warning_(code="ja.no_coercivity", message="B never crosses zero: coercivity undefined",
                                         severity="info"))
            asym = [metrics["remanence_asymmetry"], metrics["coercivity_asymmetry"]]
            odd_counts = (metrics["n_remanence_crossings"] not in (0, 2)) or (metrics["n_coercivity_crossings"] not in (0, 2))
            biased_with_crossings = bool(has_bias) and (metrics["n_remanence_crossings"] + metrics["n_coercivity_crossings"] > 0)
            large = any(a is not None and a > LOOP_ASYMMETRY_TOLERANCE for a in asym)
            if biased_with_crossings or odd_counts or large:
                rem_txt = ", ".join(f"{c['B']:.6g}" for c in metrics["remanence_crossings"]) or "none"
                coe_txt = ", ".join(f"{c['H']:.6g}" for c in metrics["coercivity_crossings"]) or "none"
                if has_bias:
                    reason = "DC-biased drive: the two crossings of an asymmetric loop differ"
                elif odd_counts:
                    reason = "unexpected number of zero crossings"
                else:
                    reason = (f"relative asymmetry above {LOOP_ASYMMETRY_TOLERANCE:g} for a symmetric drive (steady state "
                              "not reached to the accuracy implied by the remanence: tighten steady_state_tol)")
                warnings.append(Warning_(
                    code="ja.asymmetric_loop",
                    message=(f"{reason}; H = 0 crossings at B = [{rem_txt}] T, B = 0 crossings at H = [{coe_txt}] A/m; "
                             "the scalar remanence and coercivity are the mean magnitudes over the crossings, use "
                             "diagnostics['loop'] for the individual values"),
                    severity="info" if has_bias else "warning"))
            diagnostics["loop"] = metrics
        else:
            # open-path integral of H dB (trapezoidal), not closed: sign carries no dissipation meaning
            path_energy = float(np.sum(0.5 * (trace.H[1:] + trace.H[:-1]) * np.diff(trace.B)))
            values["path_energy_density"] = Quantity(value=path_energy, unit="J/m^3")
            warnings.append(Warning_(code="ja.not_periodic",
                                     message="no periodic steady state established: remanence, coercivity and the "
                                             "closed-loop energy are not evaluated; path_energy_density is the open "
                                             "integral of H dB along the trajectory", severity="info"))
            if exc.kind == "constant":
                held = {"premagnetisation_ramp": "the initial-curve magnetisation reached by the pre-magnetisation "
                                                 "ramp from the demagnetised state",
                        "user_input": "the supplied initial state",
                        "default": "the demagnetised state (H = 0)"}[trace.initial_state["source"]]
                warnings.append(Warning_(
                    code="ja.static_field",
                    message=(f"constant field H = {trace.initial_state['H_A_per_m']:g} A/m: dH/dt = 0, so M is held at "
                             f"M(t0) = {trace.initial_state['M_A_per_m']:g} A/m ({held}); B = mu0 (H + M) = "
                             f"{float(trace.B[0]):.6g} T; no loop is traversed"),
                    severity="info"))

        start = trace.initial_state
        if start["source"] == "user_input":
            initial_assumption = Assumption(
                id="ja.initial_state", basis="user_input", affects=list(values),
                statement=f"initial magnetisation M(t0) = {start['M_A_per_m']:g} A/m supplied by the caller "
                          f"(admissible at H(t0) = {start['H_A_per_m']:g} A/m)")
        elif start["source"] == "premagnetisation_ramp":
            initial_assumption = Assumption(
                id="ja.initial_state", basis="model_assumption", affects=list(values),
                statement=f"demagnetised state (M = 0 at H = 0) magnetised along the initial curve by a monotone ramp "
                          f"of H to H(t0) = {start['H_A_per_m']:g} A/m before t0 (rate-independent model: ramp "
                          f"duration immaterial), giving M(t0) = {start['M_A_per_m']:g} A/m")
        else:
            initial_assumption = Assumption(id="ja.initial_state", basis="default", affects=list(values),
                                            statement="demagnetised initial state M(t0) = 0 at H(t0) = 0")
        assumptions = [
            initial_assumption,
            Assumption(id="ja.sign_convention", statement="B and H positive along the same reference direction; "
                       "oint H dB > 0 is dissipated energy", basis="model_assumption", affects=["loop_energy_density"]),
            Assumption(id="ja.rate_independent", statement="rate-independent constitutive law; the trajectory depends "
                       "only on the ordered sequence of H values", basis="model_assumption", affects=list(values)),
        ]
        if self.parameter_set is not None and self.parameter_set.basis == "illustrative":
            assumptions.append(Assumption(id="ja.illustrative_parameters",
                                          statement=f"parameter set {self.parameter_set.id!r} is illustrative and "
                                                    "does not represent a specific material",
                                          basis="model_assumption", affects=list(values)))
        limitations = [Limitation(code=f"ja.limit.{i}", statement=s)
                       for i, s in enumerate(self.domain.known_limitations)]
        return ModelOutput(model=self.identity, values=values, domain=check, assumptions=assumptions,
                           limitations=limitations, warnings=warnings, sources=list(self.sources),
                           diagnostics=diagnostics)
