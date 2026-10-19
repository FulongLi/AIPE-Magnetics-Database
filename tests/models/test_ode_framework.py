"""Tests for the dynamic / ODE framework (models.dynamic) and its verification checks."""

from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest

from aipe_magnetics.common.errors import ConvergenceError, InvalidInputError, UnsupportedOperationError
from aipe_magnetics.models.dynamic import (
    Excitation,
    FiniteTimeBlowUp,
    LinearRL,
    NonlinearInductor,
    ODEModel,
    SolverConfig,
    convergence_study,
    cumulative_trapezoid,
    cycle_energy_density,
    interpolated_crossings,
    loop_closure_error,
    periodic_steady_state,
    solve,
    trapezoid,
)
from aipe_magnetics.verification.checks import check_dynamic


# ---------------------------------------------------------------------------------------------
# Excitation
# ---------------------------------------------------------------------------------------------
def _fd(exc: Excitation, t: np.ndarray, h: float = 1e-7) -> np.ndarray:
    return (exc(t + h) - exc(t - h)) / (2 * h)


class TestExcitation:
    def test_constant(self):
        exc = Excitation.constant(2.5)
        assert exc(0.3) == 2.5 and exc.derivative(0.3) == 0.0
        assert exc(np.array([0.0, 1.0])).tolist() == [2.5, 2.5]
        assert exc.period is None and exc.kind == "constant"

    def test_sinusoidal_values_and_derivative(self):
        exc = Excitation.sinusoidal(3.0, 50.0, phase=0.2, offset=1.0)
        t = np.linspace(0.001, 0.019, 7)
        assert np.allclose(exc(t), 1.0 + 3.0 * np.sin(2 * np.pi * 50 * t + 0.2))
        assert np.allclose(exc.derivative(t), _fd(exc, t), rtol=1e-6, atol=1e-6)
        assert exc.period == pytest.approx(0.02)
        assert exc.extrema(0.0, 0.02) == (-2.0, 4.0)

    def test_triangular_shape_and_derivative(self):
        exc = Excitation.triangular(2.0, 100.0, duty=0.25, offset=0.5)
        T = 0.01
        assert exc(0.0) == pytest.approx(-1.5)
        assert exc(0.25 * T) == pytest.approx(2.5)
        assert exc(T) == pytest.approx(-1.5)
        assert exc(0.125 * T) == pytest.approx(0.5)
        assert exc.derivative(0.1 * T) == pytest.approx(4.0 / (0.25 * T))
        assert exc.derivative(0.5 * T) == pytest.approx(-4.0 / (0.75 * T))
        assert np.allclose(exc.breakpoints(0.0, 2.5 * T), [0.25 * T, T, 1.25 * T, 2 * T, 2.25 * T])
        assert exc.split_hint is True

    def test_piecewise_linear_non_periodic_rejects_extrapolation(self):
        exc = Excitation.piecewise_linear([0.0, 1.0, 3.0], [0.0, 2.0, 0.0], periodic=False)
        assert exc(0.5) == pytest.approx(1.0) and exc(2.0) == pytest.approx(1.0)
        assert exc.derivative(0.5) == pytest.approx(2.0) and exc.derivative(2.0) == pytest.approx(-1.0)
        assert exc.domain == (0.0, 3.0)
        with pytest.raises(InvalidInputError):
            exc(3.5)
        assert np.allclose(exc.breakpoints(0.0, 3.0), [1.0])

    def test_piecewise_linear_periodic_wraps(self):
        exc = Excitation.piecewise_linear([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], periodic=True)
        assert exc.period == 2.0
        assert exc(2.5) == pytest.approx(0.5) and exc(-0.5) == pytest.approx(0.5)
        assert np.allclose(exc.breakpoints(0.0, 4.0), [1.0, 2.0, 3.0])

    def test_sampled_cubic_derivative(self):
        t = np.linspace(0.0, 1.0, 41)
        v = np.sin(2 * np.pi * t)
        exc = Excitation.sampled(t, v, interpolation="cubic", periodic=True)
        tt = np.linspace(0.05, 0.95, 11)
        assert np.allclose(exc(tt), np.sin(2 * np.pi * tt), atol=1e-3)
        assert np.allclose(exc.derivative(tt), _fd(exc, tt), rtol=1e-4, atol=1e-4)
        assert exc.split_hint is False
        with pytest.raises(InvalidInputError):
            Excitation.sampled(t, v + t, interpolation="cubic", periodic=True)  # endpoints differ

    @pytest.mark.parametrize("factory", [
        lambda: Excitation.constant(float("nan")),
        lambda: Excitation.sinusoidal(1.0, 0.0),
        lambda: Excitation.sinusoidal(-1.0, 10.0),
        lambda: Excitation.sinusoidal(1.0, float("inf")),
        lambda: Excitation.triangular(1.0, 10.0, duty=0.0),
        lambda: Excitation.triangular(1.0, 10.0, duty=1.0),
        lambda: Excitation.piecewise_linear([0.0, 1.0, 1.0], [0.0, 1.0, 2.0], periodic=False),
        lambda: Excitation.piecewise_linear([0.0, 1.0], [0.0], periodic=False),
        lambda: Excitation.piecewise_linear([0.0, 1.0], [0.0, float("nan")], periodic=False),
        lambda: Excitation.sampled([0.0, 1.0], [0.0, 1.0], interpolation="spline"),
        lambda: Excitation.from_callable("not callable"),
    ])
    def test_invalid_inputs_rejected(self, factory):
        with pytest.raises(InvalidInputError):
            factory()

    def test_from_callable_without_derivative(self):
        exc = Excitation.from_callable(lambda t: 2.0 * t, period=None)
        assert exc(1.5) == 3.0 and exc.has_derivative is False
        with pytest.raises(UnsupportedOperationError):
            exc.derivative(1.0)

    def test_from_waveform_without_module_is_explicit(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "aipe_magnetics.models.waveforms", None)  # force ImportError
        with pytest.raises(UnsupportedOperationError):
            Excitation.from_waveform(object())

    def test_from_waveform_matches_waveform_samples(self):
        # integration with the real models.waveforms module (a missing module is a failure, never a skip)
        from aipe_magnetics.models.waveforms import Waveform

        cases = [
            Waveform.sinusoidal(0.2, 1e3, dc=0.05),
            Waveform.triangular(0.3, 2e3, duty=0.3, dc=-0.1),
            Waveform.trapezoidal(0.25, 1e3, duty_rise=0.2, duty_flat=0.3),
            Waveform.piecewise_linear(1e3, [[0.0, 0.0], [0.25, 0.2], [0.6, -0.1], [1.0, 0.0]]),
            Waveform.sampled(1e3, np.sin(np.linspace(0.0, 2 * np.pi, 64))),
        ]
        for wave in cases:
            exc = Excitation.from_waveform(wave)
            t, b = wave.samples(257)
            assert np.allclose(exc(t), b, atol=1e-12), wave.kind
            assert np.allclose(exc(t + 3 * wave.period), b, atol=1e-12), wave.kind  # periodic continuation
            assert exc.period == pytest.approx(wave.period) and exc.has_derivative
            assert exc.parameters["waveform_kind"] == wave.kind
            assert exc.extrema(0.0, wave.period) == pytest.approx((wave.minimum, wave.maximum), abs=1e-12)
        with pytest.raises(UnsupportedOperationError):
            Excitation.from_waveform(types.SimpleNamespace(kind="sinusoidal"))

    def test_from_waveform_module_stub_without_class_is_unsupported(self, monkeypatch):
        module = types.ModuleType("aipe_magnetics.models.waveforms")
        monkeypatch.setitem(sys.modules, "aipe_magnetics.models.waveforms", module)
        with pytest.raises(UnsupportedOperationError):
            Excitation.from_waveform(object())


# ---------------------------------------------------------------------------------------------
# SolverConfig and solve
# ---------------------------------------------------------------------------------------------
class TestSolverConfig:
    def test_defaults(self):
        cfg = SolverConfig()
        assert cfg.method == "RK45" and cfg.max_step is None and cfg.max_steps > 0

    @pytest.mark.parametrize("kwargs", [
        {"rtol": 1e-20}, {"rtol": 1.0}, {"rtol": float("nan")}, {"atol": -1.0}, {"max_step": 0.0},
        {"first_step": -1.0}, {"first_step": 2.0, "max_step": 1.0}, {"method": "Euler"}, {"max_steps": 0},
        {"unknown": 1},
    ])
    def test_invalid_rejected(self, kwargs):
        with pytest.raises(ValueError):
            SolverConfig(**kwargs)

    def test_dict_config_is_validated_in_solve(self):
        with pytest.raises(InvalidInputError):
            solve(LinearRL(1.0, 1e-3), Excitation.constant(1.0), (0.0, 1e-3), {"rtol": -1.0})


class TestSolve:
    def setup_method(self):
        self.rl = LinearRL(R=2.0, L=1e-3)
        self.v = Excitation.constant(4.0)

    def test_outputs_and_diagnostics(self):
        t = np.linspace(0.0, 2e-3, 21)
        sol = solve(self.rl, self.v, (0.0, 2e-3), SolverConfig(rtol=1e-9, atol=1e-12), t_eval=t)
        assert sol.y.shape == (1, 21) and sol.t.shape == (21,) and sol.u.shape == (21,)
        assert set(sol.outputs) == {"v", "p_in", "p_R", "w_L"}
        for key in ("nfev", "njev", "nlu", "status", "message", "success", "n_steps", "n_segments"):
            assert key in sol.diagnostics
        assert sol.diagnostics["success"] is True and sol.diagnostics["status"] == 0
        assert np.allclose(sol.state("i"), self.rl.closed_form(self.v, t), rtol=1e-7, atol=1e-9)
        assert sol.satisfies_constraints and sol.constraint_violations == []
        summary = sol.summary()
        assert summary["n_points"] == 21 and summary["state_names"] == ["i"]

    def test_natural_steps_and_dense_interpolation(self):
        sol = solve(self.rl, self.v, (0.0, 2e-3), SolverConfig(rtol=1e-9, atol=1e-12, dense_output=True))
        assert sol.n_points == sol.diagnostics["n_steps"] + 1
        t = np.linspace(0.0, 2e-3, 9)
        assert np.allclose(sol.interpolate(t)[0], self.rl.closed_form(self.v, t), rtol=1e-6, atol=1e-8)
        with pytest.raises(InvalidInputError):
            sol.interpolate([3e-3])
        no_dense = solve(self.rl, self.v, (0.0, 2e-3), SolverConfig())
        with pytest.raises(UnsupportedOperationError):
            no_dense.interpolate([1e-3])

    def test_callable_excitation_and_y0_override(self):
        sol = solve(self.rl, lambda t: 4.0 + 0.0 * t, (0.0, 1e-3), SolverConfig(), y0=[1.0])
        assert sol.initial_state[0] == 1.0 and sol.diagnostics["excitation"]["kind"] == "custom"

    def test_segmentation_at_triangular_corners(self):
        tri = Excitation.triangular(1.0, 1000.0)
        sol = solve(self.rl, tri, (0.0, 2.5e-3), SolverConfig())
        assert sol.diagnostics["n_segments"] == 5
        unsplit = solve(self.rl, tri, (0.0, 2.5e-3), SolverConfig(split_at_breakpoints=False))
        assert unsplit.diagnostics["n_segments"] == 1
        with pytest.raises(InvalidInputError):
            solve(self.rl, tri, (0.0, 2.5e-3), SolverConfig(max_segments=2))

    @pytest.mark.parametrize("bad", [
        lambda rl, v: solve("model", v, (0.0, 1e-3)),
        lambda rl, v: solve(rl, 3.0, (0.0, 1e-3)),
        lambda rl, v: solve(rl, v, (0.0, 0.0)),
        lambda rl, v: solve(rl, v, (0.0, 1e-3), y0=[1.0, 2.0]),
        lambda rl, v: solve(rl, v, (0.0, 1e-3), y0=[float("inf")]),
        lambda rl, v: solve(rl, v, (0.0, 1e-3), t_eval=[0.0, 2e-3]),
        lambda rl, v: solve(rl, v, (0.0, 1e-3), t_eval=[5e-4, 1e-4]),
        lambda rl, v: solve(rl, Excitation.piecewise_linear([0.0, 1e-4], [0.0, 1.0], periodic=False), (0.0, 1e-3)),
    ])
    def test_invalid_inputs(self, bad):
        with pytest.raises(InvalidInputError):
            bad(self.rl, self.v)

    def test_failure_is_explicit(self):
        with pytest.raises(ConvergenceError):
            solve(FiniteTimeBlowUp(), Excitation.constant(0.0), (0.0, 2.0), SolverConfig())
        with pytest.raises(ConvergenceError):
            solve(self.rl, Excitation.sinusoidal(1.0, 1000.0), (0.0, 5e-3), SolverConfig(max_steps=5))

    def test_required_initial_state_has_no_silent_default(self):
        class Needs(ODEModel):
            state_names = ("x",)
            default_initial_state = {"x": None}

            def __init__(self):
                self.parameters = {}
                super().__init__()

            def rhs(self, t, y, u, du_dt):
                return [-y[0]]

        with pytest.raises(InvalidInputError):
            Needs().initial_state()
        assert Needs().initial_state(x=2.0).tolist() == [2.0]
        with pytest.raises(InvalidInputError):
            Needs().initial_state(z=1.0)
        with pytest.raises(InvalidInputError):
            solve(Needs(), Excitation.constant(0.0), (0.0, 1.0))


class TestConvergenceAndSteadyState:
    def test_convergence_study_contract(self):
        rl = LinearRL(1.0, 1e-3)
        v = Excitation.constant(1.0)
        study = convergence_study(rl, v, (0.0, 5e-3), SolverConfig(rtol=1e-3, atol=1e-6),
                                  max_steps=(4e-4, 2e-4, 1e-4, 5e-5), tolerance=1e-6)
        for key in ("max_steps", "errors", "successive_differences", "observed_order", "converged", "finest_error"):
            assert key in study
        assert study["max_steps"] == [4e-4, 2e-4, 1e-4, 5e-5]
        assert len(study["errors"]) == 3 and study["converged"] is True
        assert 3.5 <= study["observed_order"] <= 5.5
        with pytest.raises(InvalidInputError):
            convergence_study(rl, v, (0.0, 5e-3), SolverConfig(), max_steps=(1e-4,))
        with pytest.raises(InvalidInputError):
            convergence_study(rl, v, (0.0, 5e-3), SolverConfig(), max_steps=(1.0, 0.5))

    def test_periodic_steady_state(self):
        rl = LinearRL(1.0, 1e-3)
        sine = Excitation.sinusoidal(1.0, 1000.0)
        pss = periodic_steady_state(rl, sine, 1e-3, SolverConfig(rtol=1e-9, atol=1e-12), max_periods=40, tol=1e-4,
                                    points_per_period=50)
        assert pss.converged and pss.periods > 1 and pss.solution.n_points == 51
        assert pss.solution.t_span == pytest.approx(((pss.periods - 1) * 1e-3, pss.periods * 1e-3))
        with pytest.raises(ConvergenceError):
            periodic_steady_state(rl, sine, 1e-3, SolverConfig(), max_periods=2, tol=1e-8)
        with pytest.raises(InvalidInputError):
            periodic_steady_state(rl, sine, 1.5e-3, SolverConfig())  # not a multiple of the excitation period
        with pytest.raises(InvalidInputError):
            periodic_steady_state(rl, sine, -1.0, SolverConfig())


# ---------------------------------------------------------------------------------------------
# solution utilities and example models
# ---------------------------------------------------------------------------------------------
class TestUtilities:
    def test_trapezoid_and_cumulative(self):
        x = np.linspace(0.0, 1.0, 1001)
        assert trapezoid(x ** 2, x) == pytest.approx(1.0 / 3.0, rel=1e-5)
        cum = cumulative_trapezoid(2 * x, x)
        assert cum[0] == 0.0 and cum[-1] == pytest.approx(1.0, rel=1e-12)
        with pytest.raises(InvalidInputError):
            trapezoid([1.0, 2.0], [0.0, 0.0])
        with pytest.raises(InvalidInputError):
            trapezoid([1.0, 2.0, 3.0], [0.0, 1.0])

    def test_cycle_energy_sign_convention(self):
        phi = np.linspace(0.0, 2 * np.pi, 4001)[:-1]
        ccw = cycle_energy_density(np.cos(phi), np.sin(phi))  # H = cos, B = sin: counter-clockwise
        cw = cycle_energy_density(np.cos(-phi), np.sin(-phi))
        assert ccw == pytest.approx(math.pi, rel=1e-6) and cw == pytest.approx(-math.pi, rel=1e-6)
        with pytest.raises(InvalidInputError):
            cycle_energy_density([0.0, 1.0], [0.0, 1.0])

    def test_crossings_and_closure(self):
        res = interpolated_crossings(np.array([-1.0, 1.0, 1.0, -3.0]), np.array([0.0, 2.0, 4.0, 6.0]))
        assert [c["direction"] for c in res] == [1.0, -1.0]
        assert res[0]["y"] == pytest.approx(1.0) and res[1]["y"] == pytest.approx(4.5)
        assert loop_closure_error(np.array([1.0, 2.0, 1.5])) == pytest.approx(0.25)
        assert loop_closure_error(np.zeros(4)) == 0.0


class TestExampleModels:
    def test_linear_rl_validation(self):
        for bad in ({"R": 0.0, "L": 1e-3}, {"R": 1.0, "L": -1e-3}, {"R": float("nan"), "L": 1e-3},
                    {"R": 1.0, "L": 1e-3, "i_max": 0.0}):
            with pytest.raises(InvalidInputError):
                LinearRL(**bad)
        with pytest.raises(UnsupportedOperationError):
            LinearRL(1.0, 1e-3).closed_form(Excitation.triangular(1.0, 10.0), [0.0])
        assert "R" in LinearRL(1.0, 1e-3).describe()["parameters"]

    def test_nonlinear_inductor_relations(self):
        ni = NonlinearInductor(L0=2e-3, I_sat=1.5, R=0.1)
        i = np.array([0.0, 0.5, 3.0])
        assert np.allclose(ni.current(ni.flux_linkage(i)), i)
        lam = 1e-6  # x = lam/(L0 I_sat) = 3.3e-4: relative series correction x^2/6 ~ 2e-8
        assert ni.stored_energy(lam) == pytest.approx(lam ** 2 / (2 * ni.L0), rel=1e-6)
        lam_big = 0.6 * ni.lambda_max
        assert ni.stored_energy(lam_big) > lam_big ** 2 / (2 * ni.L0)  # saturation stores more energy per flux
        assert ni.constraints(np.array([0.0]), np.array([[ni.lambda_max]])) != []
        assert math.isnan(ni.rhs(0.0, np.array([ni.lambda_max]), 0.0, None)[0])
        with pytest.raises(InvalidInputError):
            NonlinearInductor(L0=1e-3, I_sat=0.0, R=0.1)


# ---------------------------------------------------------------------------------------------
# verification checks of this module must pass
# ---------------------------------------------------------------------------------------------
CHECKS = [
    check_dynamic.rl_step_matches_closed_form,
    check_dynamic.timestep_convergence_order,
    check_dynamic.energy_consistency_nonlinear_inductor,
    check_dynamic.solver_failure_is_explicit,
    check_dynamic.constraints_reported_not_clipped,
    check_dynamic.periodic_steady_state_detection,
]


@pytest.mark.parametrize("func", CHECKS, ids=[f.check_id for f in CHECKS])
def test_dynamic_checks_pass(func):
    result = func()
    assert result.passed, f"{func.check_id} failed: {result.metrics}"
