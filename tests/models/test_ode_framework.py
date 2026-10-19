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
# regressions from the independent review (breakpoints, closure, callables, steady state, crossings)
# ---------------------------------------------------------------------------------------------
def _particular(rl: LinearRL, exc: Excitation, t: np.ndarray) -> np.ndarray:
    p = exc.parameters
    w = 2.0 * math.pi * p["frequency"]
    z = math.hypot(rl.R, w * rl.L)
    theta = math.atan2(w * rl.L, rl.R)
    return p["offset"] / rl.R + (p["amplitude"] / z) * np.sin(w * t + p["phase"] - theta)


class TestExcitationRegressions:
    @pytest.mark.parametrize("t0", [0.3, 100.0, -2.5])
    def test_periodic_table_breakpoints_follow_the_table_origin(self, t0):
        times = np.array([0.0, 1.0, 2.0]) + t0  # triangle with corners at t0 + k, period 2
        exc = Excitation.piecewise_linear(times, [0.0, 1.0, 0.0], periodic=True)
        q0, q1 = t0 - 2.5, t0 + 5.3  # window ends between corners: corners t0 - 2 ... t0 + 5 lie inside
        pts = exc.breakpoints(q0, q1)
        t = np.linspace(q0, q1, 78001)
        jumps = t[1:][np.abs(np.diff(exc.derivative(t))) > 0.5]  # numerically located derivative jumps
        assert pts.size == jumps.size == 8
        assert np.allclose(pts, t0 + np.arange(-2.0, 6.0))
        assert np.allclose(pts, jumps, atol=2e-4)
        sampled = Excitation.sampled(times, [0.0, 1.0, 0.0], periodic=True)
        assert np.allclose(sampled.breakpoints(q0, q1), pts)

    def test_shifted_periodic_table_solves_like_the_unshifted_one(self):
        T = 1e-3
        a = Excitation.piecewise_linear([0.0, 0.5 * T, T], [-1.0, 1.0, -1.0], periodic=True)
        b = Excitation.piecewise_linear([0.3 * T, 0.5 * T, T, 1.3 * T], [0.2, 1.0, -1.0, 0.2], periodic=True)
        probe = np.linspace(0.3 * T, 3.3 * T, 13)
        assert np.allclose(a(probe), b(probe))
        rl = LinearRL(1.0, 1e-3)
        grid = np.linspace(0.3 * T, 3.3 * T, 301)
        sa = solve(rl, a, (0.3 * T, 3.3 * T), SolverConfig(), t_eval=grid)
        sb = solve(rl, b, (0.3 * T, 3.3 * T), SolverConfig(), t_eval=grid)
        ref = solve(rl, a, (0.3 * T, 3.3 * T), SolverConfig(method="DOP853", rtol=1e-12, atol=1e-14), t_eval=grid)
        assert sa.diagnostics["n_segments"] == 7  # corners 0.5T, T, ..., 3T
        assert sb.diagnostics["n_segments"] == 9  # the same corners plus the (slope-continuous) wrap knots 1.3T, 2.3T
        assert np.allclose(b.breakpoints(0.3 * T, 3.3 * T), T * np.array([0.5, 1.0, 1.3, 1.5, 2.0, 2.3, 2.5, 3.0]))
        scale = float(np.max(np.abs(ref.y)))
        assert np.max(np.abs(sb.y - ref.y)) / scale < 1e-6
        assert np.max(np.abs(sa.y - sb.y)) / scale < 1e-6

    def test_periodic_tables_must_close(self):
        with pytest.raises(InvalidInputError):
            Excitation.piecewise_linear([0.0, 1.0], [0.0, 1.0], periodic=True)
        t = np.linspace(0.0, 0.9, 10)
        with pytest.raises(InvalidInputError):
            Excitation.sampled(t, np.sin(2 * np.pi * t), periodic=True)  # [0, T) data without the closing sample
        exc = Excitation.piecewise_linear([0.0, 1.0, 2.0], [0.0, 1.0, 1e-8], periodic=True)
        assert exc(2.0) == 0.0 and exc.parameters["closure_mismatch"] == pytest.approx(1e-8)
        assert abs(exc(1.999999) - exc(2.000001)) < 1e-5  # continuous continuation, no jump at the wrap
        loose = Excitation.piecewise_linear([0.0, 1.0, 2.0], [0.0, 1.0, 1e-3], periodic=True, closure_tolerance=1e-2)
        assert loose(2.0) == 0.0
        record = Excitation.sampled([0.0, 1.0], [0.0, 1.0])  # a record of samples is finite unless stated periodic
        assert record.period is None and record.domain == (0.0, 1.0)
        with pytest.raises(InvalidInputError):
            record(1.5)

    def test_cubic_periodic_accepts_round_off_closure(self):
        t = np.linspace(0.0, 0.02, 41)
        v = 1e3 * np.sin(2 * np.pi * 50 * t)  # v[-1] = -2.4e-13, not bit-identical to v[0]
        exc = Excitation.sampled(t, v, interpolation="cubic", periodic=True)
        assert exc(0.02) == exc(0.0) == 0.0
        assert abs(exc(0.0301) - exc(0.0101)) < 1e-9 * 1e3
        with pytest.raises(InvalidInputError):
            Excitation.sampled(t, v + 1.0 * t, interpolation="cubic", periodic=True)

    def test_extrema_honour_the_interval(self):
        exc = Excitation.piecewise_linear([0.0, 1.0, 2.0], [0.0, 10.0, 0.0], periodic=False)
        assert exc.extrema(0.0, 0.1) == (0.0, pytest.approx(1.0))
        assert exc.extrema(0.0, 2.0) == (0.0, 10.0)
        per = Excitation.piecewise_linear([0.0, 1.0, 2.0], [0.0, 10.0, 0.0], periodic=True)
        assert per.extrema(0.0, 0.5) == (0.0, pytest.approx(5.0))
        assert per.extrema(0.0, 2.0) == (0.0, 10.0) and per.extrema(1.5, 4.0) == (0.0, 10.0)
        sine = Excitation.sinusoidal(1.0, 1.0)
        lo, hi = sine.extrema(0.0, 0.25)
        assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(1.0, abs=1e-6)
        with pytest.raises(InvalidInputError):
            sine.extrema(1.0, 0.0)

    @pytest.mark.parametrize("func", [lambda t: np.array([1.0, 2.0]), lambda t: "x", lambda t: 1.0 / 0.0,
                                      lambda t: {"a": 1}])
    def test_custom_callable_failures_are_invalid_input(self, func):
        with pytest.raises(InvalidInputError):
            solve(LinearRL(1.0, 1e-3), func, (0.0, 1e-3))
        with pytest.raises(InvalidInputError):
            Excitation.from_callable(func)(0.5)

    def test_custom_callable_scalar_result_is_broadcast(self):
        exc = Excitation.from_callable(lambda t: 5.0)
        assert exc(0.1) == 5.0 and exc(np.array([0.0, 1.0])).tolist() == [5.0, 5.0]
        sol = solve(LinearRL(1.0, 1e-3), lambda t: 5.0, (0.0, 1e-3), SolverConfig(rtol=1e-9, atol=1e-12), t_eval=[0.0, 1e-3])
        assert sol.y[0, -1] == pytest.approx(5.0 * (1 - math.exp(-1.0)), rel=1e-6)

    def test_rhs_failures_are_typed(self):
        class Arithmetic(LinearRL):
            def rhs(self, t, y, u, du_dt):  # noqa: ARG002
                return [1.0 / 0.0]

        class Programming(LinearRL):
            def rhs(self, t, y, u, du_dt):  # noqa: ARG002
                return [u + "volts"]

        class NonNumeric(LinearRL):
            def rhs(self, t, y, u, du_dt):  # noqa: ARG002
                return ["volts"]

        with pytest.raises(ConvergenceError):
            solve(Arithmetic(1.0, 1e-3), Excitation.constant(1.0), (0.0, 1e-3))
        with pytest.raises(InvalidInputError):
            solve(Programming(1.0, 1e-3), Excitation.constant(1.0), (0.0, 1e-3))
        with pytest.raises(InvalidInputError):
            solve(NonNumeric(1.0, 1e-3), Excitation.constant(1.0), (0.0, 1e-3))


class TestSteadyStateRegressions:
    def setup_method(self):
        self.rl = LinearRL(1.0, 5e-3)  # tau = 5 ms, contraction e^-0.2 per 1 ms period
        self.exc = Excitation.sinusoidal(1.0, 1000.0, offset=5.0)  # 5 V DC: ripple small against the transient
        self.cfg = SolverConfig(method="DOP853", rtol=1e-10, atol=1e-13)

    def test_biased_slow_transient_is_resolved_to_tolerance(self):
        tol = 1e-6
        pss = periodic_steady_state(self.rl, self.exc, 1e-3, self.cfg, max_periods=400, tol=tol, points_per_period=100)
        ip = _particular(self.rl, self.exc, pss.solution.t)
        ptp = float(np.max(ip) - np.min(ip))
        assert ptp < 0.4 and np.max(np.abs(ip)) > 4.5  # offset-dominated state
        assert abs(pss.solution.y[0, -1] - ip[-1]) / ptp <= tol  # end state within tol of the orbit (ptp-relative)
        assert pss.contraction_ratio == pytest.approx(math.exp(-0.2), abs=1e-3)
        assert pss.estimated_remaining_change is not None and pss.estimated_remaining_change <= tol
        assert pss.scale[0] == pytest.approx(ptp, rel=1e-2)
        assert pss.periods > 50 and not pss.accelerated and pss.extrapolations == 0

    def test_acceleration_reaches_the_same_orbit_in_few_periods(self):
        tol = 1e-6
        pss = periodic_steady_state(self.rl, self.exc, 1e-3, self.cfg, max_periods=400, tol=tol, points_per_period=100,
                                    accelerate=True)
        ip = _particular(self.rl, self.exc, pss.solution.t)
        ptp = float(np.max(ip) - np.min(ip))
        assert pss.converged and pss.accelerated and pss.extrapolations >= 1 and pss.periods <= 6
        assert np.max(np.abs(pss.solution.y[0] - ip)) / ptp <= tol

    def test_non_convergence_error_reports_contraction(self):
        # early phase: the state still drifts monotonically (boundary change == excursion), the hint says so
        with pytest.raises(ConvergenceError) as early:
            periodic_steady_state(self.rl, self.exc, 1e-3, self.cfg, max_periods=5, tol=1e-6)
        assert "drifts monotonically" in str(early.value) and "accelerate=True" in str(early.value)
        assert early.value.details["history"][-1] == pytest.approx(1.0)
        # later phase: the ripple dominates the excursion and the contraction ratio is measured
        with pytest.raises(ConvergenceError) as info:
            periodic_steady_state(self.rl, self.exc, 1e-3, self.cfg, max_periods=30, tol=1e-6)
        message = str(info.value)
        assert "contraction ratio" in message and "accelerate=True" in message and "more periods" in message
        assert info.value.details["contraction_ratio"] == pytest.approx(math.exp(-0.2), abs=0.03)
        assert info.value.details["estimated_additional_periods"] > 10
        assert info.value.details["estimated_remaining_change"] > 1e-6

    def test_first_period_acceptance_needs_contraction_evidence(self):
        tol = 1e-6
        r = math.exp(-0.2)
        i_fixed = float(_particular(self.rl, self.exc, np.array([0.0]))[0])
        exact = periodic_steady_state(self.rl, self.exc, 1e-3, self.cfg, max_periods=5, tol=tol, y0=[i_fixed])
        assert exact.periods == 1 and exact.boundary_change <= 1e-3 * tol  # exactly periodic: accepted at once
        ip = _particular(self.rl, self.exc, np.linspace(0.0, 1e-3, 201))
        ptp = float(np.max(ip) - np.min(ip))
        eps = 0.5 * tol * ptp / (1.0 - r)  # first change ~ 0.5 tol: below tol, but 0.5 tol * r/(1-r) = 2.3 tol remain
        near = periodic_steady_state(self.rl, self.exc, 1e-3, self.cfg, max_periods=20, tol=tol, y0=[i_fixed + eps])
        assert near.history[0] <= tol and near.periods >= 2  # a sub-tolerance first change is not accepted blindly
        assert near.estimated_remaining_change is not None and near.estimated_remaining_change <= tol
        assert abs(near.solution.y[0, -1] - ip[-1]) / ptp <= tol
        expected = 1 + math.ceil(math.log(tol * (1.0 - r) / (r * near.history[0])) / math.log(r))
        assert abs(near.periods - expected) <= 1
        with pytest.raises(ConvergenceError):  # same start, too few periods: never declared converged early
            periodic_steady_state(self.rl, self.exc, 1e-3, self.cfg, max_periods=2, tol=tol, y0=[i_fixed + eps])
        tiny = periodic_steady_state(self.rl, self.exc, 1e-3, self.cfg, max_periods=5, tol=tol, y0=[i_fixed + 1e-4 * eps])
        assert tiny.periods == 1 and tiny.boundary_change <= 1e-3 * tol  # below tol/1000: remaining <= tol for r <= 0.999


class TestCrossingRegressions:
    def test_crossings_require_strict_sign_change(self):
        assert interpolated_crossings(np.zeros(5), np.arange(5.0)) == []
        assert interpolated_crossings([1.0, 0.0, 1.0], [0.0, 1.0, 2.0]) == []  # tangency
        assert interpolated_crossings([0.0, 0.0, 1.0], [0.0, 1.0, 2.0]) == []  # level run touching the start
        res = interpolated_crossings([1.0, 0.0, -1.0], [0.0, 1.0, 2.0])
        assert len(res) == 1 and res[0]["index"] == 1.0 and res[0]["direction"] == -1.0 and res[0]["y"] == 1.0
        res = interpolated_crossings([-1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 2.0, 3.0])
        assert len(res) == 1 and res[0]["index"] == 1.5 and res[0]["direction"] == 1.0 and res[0]["y"] == 1.5
        res = interpolated_crossings([1.0, -1.0, 0.0], [0.0, 1.0, 2.0])
        assert len(res) == 1 and res[0]["index"] == 0.5  # the trailing zero run is not a second crossing


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
