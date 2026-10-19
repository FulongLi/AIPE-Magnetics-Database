"""Verification checks for the dynamic / ODE framework (``aipe_magnetics.models.dynamic``).

Capability: ``aipe.magnetics.model.ode_framework``.  All checks are deterministic, offline and run
in well under two seconds each.  They use the reference models of ``models.dynamic.examples``
(linear R-L circuit with closed-form solutions, saturating inductor with analytic stored energy,
finite-time blow-up).  The evidence is range independent (no magnetic operating range applies).
"""

from __future__ import annotations

import math

import numpy as np

from aipe_magnetics.common.errors import ConvergenceError, InvalidInputError
from aipe_magnetics.models.dynamic import (
    Excitation,
    FiniteTimeBlowUp,
    LinearRL,
    NonlinearInductor,
    SolverConfig,
    convergence_study,
    cumulative_trapezoid,
    periodic_steady_state,
    solve,
    trapezoid,
)
from aipe_magnetics.verification.suite import CheckResult, check

CAPABILITY = "aipe.magnetics.model.ode_framework"


@check(
    id="dynamic.rl-step-matches-closed-form",
    capability=CAPABILITY, level="L2", kind="analytical_reference",
    description="LinearRL step and sinusoidal responses from the ODE framework match the closed-form solutions.",
    method="Integrate di/dt = (v - R i)/L (R = 1 Ohm, L = 1 mH) with DOP853, rtol 1e-12, atol 1e-14 on a 501-point "
           "grid over 5 tau for a 10 V step and a 10 V, 1 kHz sinusoid with 2 V offset and 0.5 A initial current; "
           "compare with i(t) = V/R + (i0 - V/R) exp(-t/tau) and the phasor particular + transient solution.",
    tags=("dynamic", "analytical"),
)
def rl_step_matches_closed_form() -> CheckResult:
    rl = LinearRL(R=1.0, L=1e-3)
    cfg = SolverConfig(method="DOP853", rtol=1e-12, atol=1e-14)
    t = np.linspace(0.0, 5.0 * rl.tau, 501)
    step = Excitation.constant(10.0)
    sol = solve(rl, step, (0.0, t[-1]), cfg, t_eval=t)
    exact = rl.closed_form(step, t)
    err_step = float(np.max(np.abs(sol.y[0] - exact)) / np.max(np.abs(exact)))
    sine = Excitation.sinusoidal(10.0, 1000.0, phase=0.3, offset=2.0)
    sol_s = solve(rl, sine, (0.0, t[-1]), cfg, y0=[0.5], t_eval=t)
    exact_s = rl.closed_form(sine, t, i0=0.5)
    err_sine = float(np.max(np.abs(sol_s.y[0] - exact_s)) / np.max(np.abs(exact_s)))
    tol = 1e-6
    return CheckResult(
        passed=bool(err_step < tol and err_sine < tol and sol.diagnostics["success"] and sol_s.diagnostics["success"]),
        metrics={"max_rel_error_step": err_step, "max_rel_error_sinusoid": err_sine, "tolerance": tol,
                 "n_steps_step": sol.diagnostics["n_steps"], "n_steps_sinusoid": sol_s.diagnostics["n_steps"]},
        notes=["closed forms: i = V/R + (i0 - V/R) e^{-t/tau}; sinusoid via |Z| = sqrt(R^2 + (wL)^2), "
               "theta = atan2(wL, R) (models/dynamic/examples.py docstring)"],
    )


@check(
    id="dynamic.timestep-convergence-order",
    capability=CAPABILITY, level="L2", kind="convergence",
    description="With a fixed max_step the RK45 (Dormand-Prince 5(4)) global error on the LinearRL step response "
                "decreases monotonically with observed order close to 5 and the finest error is below 1e-7.",
    method="convergence_study() on the 10 V step response of LinearRL (tau = 1 ms) over 5 tau with max_step = "
           "first_step = tau * (0.4, 0.2, 0.1, 0.05, 0.025), rtol 1e-3 / atol 1e-6 so that the error controller never "
           "rejects a step; errors are the max-norm against the analytic solution on a 201-point grid; the observed "
           "order is the median of the orders estimated from successive differences. Expected 3.5 <= p <= 5.5 "
           "(DP5 propagates its 5th-order solution; the pre-asymptotic sinusoidal case shows apparent orders of 7-8 "
           "through error cancellation and is therefore not used for the order estimate).",
    tags=("dynamic", "convergence"),
)
def timestep_convergence_order() -> CheckResult:
    rl = LinearRL(R=1.0, L=1e-3)
    tau = rl.tau
    step = Excitation.constant(10.0)
    cfg = SolverConfig(method="RK45", rtol=1e-3, atol=1e-6)
    study = convergence_study(rl, step, (0.0, 5.0 * tau), cfg,
                              max_steps=tuple(tau * f for f in (0.4, 0.2, 0.1, 0.05, 0.025)), tolerance=1e-7,
                              reference=lambda tt: rl.closed_form(step, tt))
    order = study["observed_order"]
    order_ok = order is not None and 3.5 <= order <= 5.5
    passed = bool(order_ok and study["errors_monotone_decreasing"] and study["finest_error"] < 1e-7
                  and study["converged"])
    return CheckResult(
        passed=passed,
        metrics={"observed_order": order, "orders_successive": study["orders_successive"],
                 "orders_vs_analytic": study["orders_vs_reference"], "errors_vs_analytic": study["errors"],
                 "max_steps": study["max_steps"], "n_steps": study["n_steps"], "finest_error": study["finest_error"],
                 "errors_monotone_decreasing": study["errors_monotone_decreasing"]},
        notes=["max_step caps the step but does not disable the error controller; tolerances were chosen so that "
               "every step equals max_step (n_steps = span / max_step + 1 for the automatic first step)."],
    )


@check(
    id="dynamic.energy-consistency-nonlinear-inductor",
    capability=CAPABILITY, level="L2", kind="energy_consistency",
    description="Power balance of the saturating inductor: int v i dt = int R i^2 dt + W(lambda_end) - W(lambda_0) "
                "with the analytic stored energy W(lambda) = -L0 I_sat^2 ln cos(lambda/(L0 I_sat)).",
    method="NonlinearInductor(L0 = 1 mH, I_sat = 2 A, R = 0.5 Ohm) driven by a 3 V, 1 kHz sinusoid for 3 ms (DOP853, "
           "rtol 1e-10, atol 1e-13, 6001-point grid); trapezoidal integration of the output powers; residual relative "
           "to the absorbed energy must be below 1e-4. The analytic W(lambda) is independently checked against "
           "cumulative trapezoidal quadrature of i(lambda) dlambda.",
    tags=("dynamic", "energy"),
)
def energy_consistency_nonlinear_inductor() -> CheckResult:
    model = NonlinearInductor(L0=1e-3, I_sat=2.0, R=0.5)
    v = Excitation.sinusoidal(3.0, 1000.0)
    t = np.linspace(0.0, 3e-3, 6001)
    sol = solve(model, v, (0.0, 3e-3), SolverConfig(method="DOP853", rtol=1e-10, atol=1e-13), t_eval=t)
    e_in = trapezoid(sol.outputs["p_in"], t)
    e_r = trapezoid(sol.outputs["p_R"], t)
    d_w = float(sol.outputs["w"][-1] - sol.outputs["w"][0])
    residual = abs(e_in - e_r - d_w) / max(abs(e_in), 1e-300)
    # independent check of the analytic stored energy by quadrature of i(lambda)
    lam = np.linspace(0.0, 0.9 * model.lambda_max, 20001)
    w_quad = cumulative_trapezoid(model.current(lam), lam)
    w_err = float(np.max(np.abs(w_quad - model.stored_energy(lam))) / np.max(model.stored_energy(lam)))
    peak_i = float(np.max(np.abs(sol.outputs["i"])))
    passed = bool(residual < 1e-4 and w_err < 1e-6 and not sol.constraint_violations and peak_i > 0.3 * model.I_sat)
    return CheckResult(
        passed=passed,
        metrics={"energy_in_J": e_in, "energy_resistive_J": e_r, "stored_energy_change_J": d_w,
                 "power_balance_residual_rel": residual, "stored_energy_quadrature_rel_error": w_err,
                 "peak_current_A": peak_i, "I_sat_A": model.I_sat, "n_steps": sol.diagnostics["n_steps"]},
        notes=["the excitation drives the inductor well into saturation (peak current > 0.3 I_sat) so the nonlinear "
               "term matters in the balance"],
    )


@check(
    id="dynamic.solver-failure-is-explicit",
    capability=CAPABILITY, level="L1", kind="invalid_input_rejection",
    description="Solver failures raise ConvergenceError (never a partial result), invalid inputs raise "
                "InvalidInputError, and failures inside user callables surface as typed package errors, never as "
                "bare TypeError / ValueError / ZeroDivisionError.",
    method="Integrate dy/dt = y^2 (blow-up at t = 1) to t = 2 with RK45 and BDF; exceed max_steps and "
           "max_rhs_evaluations on LinearRL; pass NaN excitation value, NaN initial state, NaN t_span, wrong-length y0 "
           "and an excitation without derivative to a model that needs it; drive LinearRL with custom excitation "
           "callables that return a 2-vector, a string, or divide by zero (InvalidInputError), and with model rhs "
           "functions that raise ZeroDivisionError (ConvergenceError) or TypeError (InvalidInputError).",
    tags=("dynamic", "errors"),
)
def solver_failure_is_explicit() -> CheckResult:
    outcomes: dict[str, bool] = {}
    rl = LinearRL(R=1.0, L=1e-3)
    sine = Excitation.sinusoidal(10.0, 1000.0)

    def expect(name: str, func, exc_type) -> None:
        try:
            func()
        except exc_type:
            outcomes[name] = True
        except Exception:  # noqa: BLE001 - wrong error type is a failure of this check
            outcomes[name] = False
        else:
            outcomes[name] = False

    expect("blowup_rk45", lambda: solve(FiniteTimeBlowUp(), Excitation.constant(0.0), (0.0, 2.0), SolverConfig()),
           ConvergenceError)
    expect("blowup_bdf", lambda: solve(FiniteTimeBlowUp(), Excitation.constant(0.0), (0.0, 2.0),
                                       SolverConfig(method="BDF")), ConvergenceError)
    expect("max_steps", lambda: solve(rl, sine, (0.0, 5e-3), SolverConfig(max_steps=10)), ConvergenceError)
    expect("max_rhs_evaluations", lambda: solve(rl, sine, (0.0, 5e-3), SolverConfig(max_rhs_evaluations=25)),
           ConvergenceError)
    expect("nan_excitation", lambda: Excitation.constant(float("nan")), InvalidInputError)
    expect("nan_y0", lambda: solve(rl, sine, (0.0, 1e-3), SolverConfig(), y0=[float("nan")]), InvalidInputError)
    expect("nan_t_span", lambda: solve(rl, sine, (0.0, float("nan")), SolverConfig()), InvalidInputError)
    expect("wrong_y0_length", lambda: solve(rl, sine, (0.0, 1e-3), SolverConfig(), y0=[0.0, 1.0]), InvalidInputError)
    expect("reversed_span", lambda: solve(rl, sine, (1e-3, 0.0), SolverConfig()), InvalidInputError)

    class NeedsDerivative(LinearRL):
        requires_du_dt = True

    expect("missing_derivative", lambda: solve(NeedsDerivative(1.0, 1e-3), Excitation.from_callable(lambda t: 0 * t),
                                               (0.0, 1e-3), SolverConfig()), InvalidInputError)
    # untyped failures of user callables must surface as typed package errors
    expect("callable_returns_vector", lambda: solve(rl, lambda t: np.array([1.0, 2.0]), (0.0, 1e-3)), InvalidInputError)
    expect("callable_returns_string", lambda: solve(rl, lambda t: "x", (0.0, 1e-3)), InvalidInputError)
    expect("callable_divides_by_zero", lambda: solve(rl, lambda t: 1.0 / 0.0, (0.0, 1e-3)), InvalidInputError)

    def dropout(t):  # valid at the span ends, raises a programming error in the middle of the span
        tt = np.asarray(t, dtype=float)
        if np.any((tt > 0.4e-3) & (tt < 0.6e-3)):
            raise KeyError("sensor dropout")
        return 1.0 + 0.0 * tt

    expect("callable_raises_midway", lambda: solve(rl, dropout, (0.0, 1e-3)), InvalidInputError)

    class ArithmeticRhs(LinearRL):
        def rhs(self, t, y, u, du_dt):  # noqa: ARG002
            return [1.0 / 0.0]

    class ProgrammingErrorRhs(LinearRL):
        def rhs(self, t, y, u, du_dt):  # noqa: ARG002
            return [u + "volts"]  # type: ignore[operator]

    class WrongLengthRhs(LinearRL):
        def rhs(self, t, y, u, du_dt):  # noqa: ARG002
            return [0.0, 0.0]

    expect("rhs_arithmetic_error", lambda: solve(ArithmeticRhs(1.0, 1e-3), sine, (0.0, 1e-3)), ConvergenceError)
    expect("rhs_type_error", lambda: solve(ProgrammingErrorRhs(1.0, 1e-3), sine, (0.0, 1e-3)), InvalidInputError)
    expect("rhs_wrong_length", lambda: solve(WrongLengthRhs(1.0, 1e-3), sine, (0.0, 1e-3)), InvalidInputError)
    return CheckResult(passed=all(outcomes.values()), metrics={"cases": outcomes, "n_cases": len(outcomes)})


@check(
    id="dynamic.constraints-reported-not-clipped",
    capability=CAPABILITY, level="L2", kind="physical_consistency",
    description="A violated physical constraint is listed on the solution while the state itself is left untouched.",
    method="LinearRL with a reported limit i_max = 5 A and a 10 V step (steady state 10 A): the violation message is "
           "present, max|i| is 10 (1 - e^-5) A to 1e-9 relative and the trajectory still equals the closed form.",
    tags=("dynamic", "constraints"),
)
def constraints_reported_not_clipped() -> CheckResult:
    rl = LinearRL(R=1.0, L=1e-3, i_max=5.0)
    step = Excitation.constant(10.0)
    t = np.linspace(0.0, 5e-3, 251)
    sol = solve(rl, step, (0.0, 5e-3), SolverConfig(method="DOP853", rtol=1e-12, atol=1e-14), t_eval=t)
    exact = rl.closed_form(step, t)
    peak = float(np.max(np.abs(sol.y[0])))
    expected_peak = 10.0 * (1.0 - math.exp(-5.0))
    err = float(np.max(np.abs(sol.y[0] - exact)) / np.max(np.abs(exact)))
    reported = any("i_max" in v for v in sol.constraint_violations)
    passed = bool(reported and abs(peak - expected_peak) / expected_peak < 1e-9 and err < 1e-9 and peak > 5.0)
    return CheckResult(passed=passed, metrics={"violations": list(sol.constraint_violations), "peak_current_A": peak,
                                               "expected_peak_A": expected_peak, "max_rel_error_vs_closed_form": err})


def _rl_particular(rl: LinearRL, exc: Excitation, t: np.ndarray) -> np.ndarray:
    p = exc.parameters
    w = 2.0 * math.pi * p["frequency"]
    z = math.hypot(rl.R, w * rl.L)
    theta = math.atan2(w * rl.L, rl.R)
    return p["offset"] / rl.R + (p["amplitude"] / z) * np.sin(w * t + p["phase"] - theta)


@check(
    id="dynamic.periodic-steady-state-detection",
    capability=CAPABILITY, level="L2", kind="analytical_reference",
    description="periodic_steady_state() stops when the boundary state has settled to the requested tolerance of the "
                "periodic orbit - also for a DC-biased, slowly contracting transient - the returned period equals the "
                "analytic steady-state response, Steffensen acceleration reaches the same orbit in a few periods, "
                "and insufficient max_periods raises ConvergenceError.",
    method="(a) LinearRL (tau = 1 ms) with a 10 V, 1 kHz sinusoid (period = tau): the transient decays by e^-1 per "
           "period; compare the last period with the analytic particular solution (rel. error < 1e-5 for tol = 1e-6), "
           "check the per-period decay ratio and the predicted period count. (b) LinearRL (tau = 5 ms) with a 1 V, "
           "1 kHz sinusoid on a 5 V DC offset from i0 = 0: the ripple (0.31 A peak-to-peak) is small against the "
           "4.8 A transient, the contraction ratio is e^-0.2 = 0.819 per period; require the measured ratio within "
           "1e-3 of e^-0.2, the period count equal to the analytic prediction (first k with A r^(k+1) / ptp <= tol) "
           "within one period, the residual transient at the end of the returned period below tol and at its start "
           "below change / (1 - r) relative to the peak-to-peak ripple, and the same accuracy with accelerate=True in "
           "at most 6 periods. (c) max_periods = 3 on case (a) must raise ConvergenceError.",
    tags=("dynamic", "steady-state"),
)
def periodic_steady_state_detection() -> CheckResult:
    rl = LinearRL(R=1.0, L=1e-3)
    sine = Excitation.sinusoidal(10.0, 1000.0)
    cfg = SolverConfig(method="DOP853", rtol=1e-10, atol=1e-13)
    pss = periodic_steady_state(rl, sine, 1e-3, cfg, max_periods=40, tol=1e-6, points_per_period=200)
    particular = _rl_particular(rl, sine, pss.solution.t)
    err = float(np.max(np.abs(pss.solution.y[0] - particular)) / np.max(np.abs(particular)))
    # absolute boundary-state differences are pure transient terms: they decay by exactly e^(-T/tau) per period
    diffs = [abs(float(pss.boundary_states[i + 1][0] - pss.boundary_states[i][0]))
             for i in range(len(pss.boundary_states) - 1)]
    ratios = [diffs[i + 1] / diffs[i] for i in range(len(diffs) - 1)]
    ratio_ok = all(abs(r - math.exp(-1.0)) < 1e-3 for r in ratios)
    # r/(1-r) < 1 for r = e^-1, so the criterion reduces to change_k <= tol with change_k = history[0] r^(k-1)
    expected_periods = math.ceil(math.log(pss.history[0] / 1e-6)) + 1
    try:
        periodic_steady_state(rl, sine, 1e-3, cfg, max_periods=3, tol=1e-6)
        raises = False
    except ConvergenceError:
        raises = True

    # (b) DC-biased, slowly contracting transient
    slow = LinearRL(R=1.0, L=5e-3)
    biased = Excitation.sinusoidal(1.0, 1000.0, offset=5.0)
    tol_b = 1e-6
    r_exact = math.exp(-1e-3 / slow.tau)
    pss_b = periodic_steady_state(slow, biased, 1e-3, cfg, max_periods=400, tol=tol_b, points_per_period=100)
    ip = _rl_particular(slow, biased, pss_b.solution.t)
    ptp = float(np.max(ip) - np.min(ip))
    transient = np.abs(pss_b.solution.y[0] - ip) / ptp
    residual_end = float(transient[-1])
    residual_start = float(transient[0])
    amplitude = abs(0.0 - float(_rl_particular(slow, biased, np.array([0.0]))[0]))  # |i0 - i_p(0)|
    # converged after period k (1-based) when A r^k / ptp <= tol, i.e. the end state is within tol of the orbit
    expected_b = math.ceil(math.log(tol_b * ptp / amplitude) / math.log(r_exact))
    ratio_b = pss_b.contraction_ratio
    start_bound = pss_b.boundary_change / (1.0 - r_exact)
    pss_acc = periodic_steady_state(slow, biased, 1e-3, cfg, max_periods=400, tol=tol_b, points_per_period=100,
                                    accelerate=True)
    ip_acc = _rl_particular(slow, biased, pss_acc.solution.t)
    residual_acc = float(np.max(np.abs(pss_acc.solution.y[0] - ip_acc)) / ptp)
    slow_ok = bool(pss_b.converged and ratio_b is not None and abs(ratio_b - r_exact) < 1e-3
                   and abs(pss_b.periods - expected_b) <= 1 and residual_end <= tol_b
                   and residual_start <= start_bound * (1.0 + 1e-6) + 1e-12
                   and pss_b.estimated_remaining_change is not None and pss_b.estimated_remaining_change <= tol_b)
    acc_ok = bool(pss_acc.converged and pss_acc.periods <= 6 and residual_acc <= tol_b and pss_acc.extrapolations >= 1)

    passed = bool(pss.converged and err < 1e-5 and ratio_ok and raises and abs(pss.periods - expected_periods) <= 1
                  and slow_ok and acc_ok)
    return CheckResult(passed=passed, metrics={
        "periods": pss.periods, "expected_periods": expected_periods, "boundary_change": pss.boundary_change,
        "history": pss.history, "decay_ratio_min": min(ratios), "decay_ratio_max": max(ratios),
        "max_rel_error_vs_particular": err, "raises_when_not_converged": raises,
        "biased_periods": pss_b.periods, "biased_expected_periods": expected_b, "biased_contraction_ratio": ratio_b,
        "biased_ratio_exact": r_exact, "biased_residual_end_rel_ptp": residual_end,
        "biased_residual_start_rel_ptp": residual_start, "biased_start_bound": start_bound,
        "biased_estimated_remaining": pss_b.estimated_remaining_change, "biased_ripple_ptp_A": ptp,
        "biased_transient_amplitude_A": amplitude, "accelerated_periods": pss_acc.periods,
        "accelerated_extrapolations": pss_acc.extrapolations, "accelerated_residual_rel_ptp": residual_acc,
        "tolerance": tol_b})
