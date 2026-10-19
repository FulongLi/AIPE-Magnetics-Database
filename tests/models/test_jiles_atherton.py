"""Tests for the Langevin anhysteretic function, the Jiles-Atherton model and its verification checks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aipe_magnetics.common.errors import ConvergenceError, DataNotFoundError, InvalidInputError, OutOfDomainError
from aipe_magnetics.common.units import MU_0
from aipe_magnetics.models.contract import ModelOutput
from aipe_magnetics.models.dynamic import Excitation, SolverConfig, solve
from aipe_magnetics.models.physics import (
    JA_ILLUSTRATIVE,
    JA_MODEL_ID,
    MODELS,
    JilesAthertonModel,
    JilesAthertonODE,
    anhysteretic_magnetisation,
    anhysteretic_susceptibility,
    get_parameter_set,
    langevin,
    langevin_derivative,
    loop_metrics,
)
from aipe_magnetics.models.physics.anhysteretic import (
    LANGEVIN_SERIES_THRESHOLD,
    langevin_derivative_scalar,
    langevin_scalar,
)
from aipe_magnetics.verification.checks import check_physics


# ---------------------------------------------------------------------------------------------
# Langevin
# ---------------------------------------------------------------------------------------------
class TestLangevin:
    def test_limits_at_zero(self):
        assert langevin(0.0) == 0.0 and langevin_derivative(0.0) == pytest.approx(1.0 / 3.0)
        x = 1e-6
        assert langevin(x) == pytest.approx(x / 3.0, rel=1e-12)
        assert langevin(-x) == -langevin(x)

    def test_series_matches_direct_formula_at_threshold(self):
        for x in (LANGEVIN_SERIES_THRESHOLD * (1 - 1e-9), LANGEVIN_SERIES_THRESHOLD * (1 + 1e-9), 0.049, 0.2, 1.0):
            direct = 1.0 / math.tanh(x) - 1.0 / x
            assert langevin_scalar(x) == pytest.approx(direct, rel=1e-10)
            d_direct = 1.0 / x ** 2 - 1.0 / math.sinh(x) ** 2
            assert langevin_derivative_scalar(x) == pytest.approx(d_direct, rel=1e-9)

    def test_large_arguments_saturate_without_overflow(self):
        for x in (25.0, 800.0, 1e6):
            assert langevin(x) == pytest.approx(1.0 - 1.0 / x, rel=1e-14)
            assert langevin(-x) == pytest.approx(-1.0 + 1.0 / x, rel=1e-14)
            assert langevin_derivative(x) == pytest.approx(1.0 / x ** 2, rel=1e-12)
        assert np.all(np.abs(langevin(np.array([1e3, 1e9]))) < 1.0)

    def test_derivative_matches_finite_difference(self):
        xs = np.array([-3.0, -0.3, -0.04, 0.001, 0.02, 0.5, 4.0, 15.0])
        h = 1e-5
        fd = (langevin(xs + h) - langevin(xs - h)) / (2 * h)
        assert np.allclose(langevin_derivative(xs), fd, rtol=1e-7, atol=1e-9)

    def test_vectorised_and_validated(self):
        out = langevin(np.array([0.0, 0.01, 2.0]))
        assert out.shape == (3,) and isinstance(langevin(0.5), float)
        with pytest.raises(InvalidInputError):
            langevin(float("nan"))
        with pytest.raises(InvalidInputError):
            anhysteretic_magnetisation(100.0, Ms=-1.0, a=10.0)
        assert anhysteretic_magnetisation(0.0, 1.6e6, 1100.0) == 0.0
        assert anhysteretic_susceptibility(0.0, 1.6e6, 1100.0) == pytest.approx(1.6e6 / 3300.0)


# ---------------------------------------------------------------------------------------------
# ODE model
# ---------------------------------------------------------------------------------------------
class TestJilesAthertonODE:
    @pytest.mark.parametrize("kwargs", [
        {"Ms": 0.0}, {"Ms": -1.0}, {"a": 0.0}, {"k": 0.0}, {"alpha": -1e-3}, {"c": 1.0}, {"c": -0.1},
        {"c": float("nan")}, {"Ms": "1.6e6"}, {"Ms": True},
        {"alpha": 0.05, "c": 0.9},  # alpha*c*Ms/(3a) >= 1: singular reversible feedback
        {"alpha": 2.1e-3},  # alpha*Ms/(3a) >= 1: multivalued anhysteretic
    ])
    def test_invalid_parameters(self, kwargs):
        params = {**JA_ILLUSTRATIVE.as_kwargs(), **kwargs}
        with pytest.raises(InvalidInputError):
            JilesAthertonODE(**params)

    def test_initial_state_and_rhs(self):
        ode = JilesAthertonODE(**JA_ILLUSTRATIVE.as_kwargs())
        assert ode.initial_state().tolist() == [0.0]
        assert ode.initial_state(M=1e5).tolist() == [1e5]
        with pytest.raises(InvalidInputError):
            ode.initial_state(B=0.1)
        assert ode.rhs(0.0, np.array([0.0]), 0.0, 0.0) == [0.0]  # dH/dt = 0 -> no change
        assert ode.rhs(0.0, np.array([0.0]), 0.0, 1.0)[0] == pytest.approx(ode.initial_susceptibility())
        with pytest.raises(InvalidInputError):
            ode.rhs(0.0, np.array([0.0]), 0.0, None)

    def test_reversal_is_purely_reversible(self):
        ode = JilesAthertonODE(**JA_ILLUSTRATIVE.as_kwargs())
        H, M = 500.0, 2e5  # on the ascending branch M < M_an(He): decreasing H gives delta_M = 0
        He = H + ode.alpha * M
        assert float(ode.anhysteretic(He)) > M
        chi_e = float(ode.anhysteretic_susceptibility(He))
        expected = ode.c * chi_e / (1.0 - ode.alpha * ode.c * chi_e)
        assert ode.susceptibility(H, M, -1) == pytest.approx(expected, rel=1e-12)
        assert ode.susceptibility(H, M, +1) > expected  # irreversible part adds along the drive

    def test_outputs_and_constraints(self):
        ode = JilesAthertonODE(**JA_ILLUSTRATIVE.as_kwargs())
        t = np.array([0.0, 1.0])
        y = np.array([[0.0, 1e5]])
        u = np.array([0.0, 100.0])
        out = ode.outputs(t, y, u)
        assert set(out) == {"H", "B", "He", "M_an", "M_irr", "B_an"}
        assert out["B"][1] == pytest.approx(MU_0 * (100.0 + 1e5))
        assert ode.constraints(t, y) == []
        assert ode.constraints(t, np.array([[0.0, 1.01 * ode.Ms]])) != []


# ---------------------------------------------------------------------------------------------
# contract model
# ---------------------------------------------------------------------------------------------
class TestJilesAthertonModel:
    def test_metadata_and_provenance(self):
        model = JilesAthertonModel.illustrative()
        assert model.identity.id == JA_MODEL_ID and JilesAthertonModel in MODELS
        assert model.parameter_set.basis == "illustrative"
        assert model.parameter_set.source.license == "NOASSERTION"
        assert all(p.source_ref == JA_ILLUSTRATIVE.source.id for p in model.parameters)
        assert {s.id for s in model.sources} >= {"pub.jiles-atherton-1986", "pub.jiles-thoelke-devine-1992"}
        assert model.domain.requires_history is True and model.domain.dc_bias == "supported"
        assert "rate independent" in " ".join(model.domain.known_limitations)
        described = model.describe()
        assert described["identity"]["family"] == "physics" and len(described["parameters"]) == 5
        assert get_parameter_set("ja.illustrative-demo") is JA_ILLUSTRATIVE
        with pytest.raises(DataNotFoundError):
            get_parameter_set("ja.not-there")

    def test_trajectory_periodic_output(self):
        model = JilesAthertonModel.illustrative()
        out = model.trajectory(Excitation.sinusoidal(3000.0, 50.0))
        assert isinstance(out, ModelOutput) and out.domain.inside
        for key, unit in (("flux_density_peak", "T"), ("field_peak", "A/m"), ("remanence", "T"),
                          ("coercivity", "A/m"), ("loop_energy_density", "J/m^3"), ("magnetisation_peak", "A/m")):
            assert out.values[key].unit == unit and math.isfinite(out.values[key].value)
        assert out.values["field_peak"].value == pytest.approx(3000.0)
        assert out.values["loop_energy_density"].value > 0.0
        assert out.values["magnetisation_peak"].value <= model.ode.Ms
        traj = out.diagnostics["trajectory"]
        assert len(traj["t"]) == len(traj["H"]) == len(traj["B"]) == len(traj["M"]) == 1001
        assert out.diagnostics["steady_state"] is True and out.diagnostics["periods"] >= 2
        assert out.diagnostics["solver"]["success"] is True
        out.model_dump(mode="json")  # JSON-serialisable (no numpy objects inside)

    def test_trajectory_from_sampled_arrays_and_dict(self):
        model = JilesAthertonModel.illustrative()
        t = np.linspace(0.0, 0.02, 201)
        H = 2000.0 * np.sin(2 * np.pi * 50 * t)
        out = model.trajectory((t, H))
        assert "path_energy_density" in out.values and "loop_energy_density" not in out.values
        assert any(w.code == "ja.not_periodic" for w in out.warnings)
        assert len(out.diagnostics["trajectory"]["t"]) == 201
        out2 = model.trajectory({"t": t, "H": H})
        assert out2.values["flux_density_peak"].value == pytest.approx(out.values["flux_density_peak"].value)
        with pytest.raises(InvalidInputError):
            model.trajectory("H(t)")
        with pytest.raises(InvalidInputError):
            model.trajectory((t, H[:-1]))

    def test_policy_for_unsupported_excitation(self):
        model = JilesAthertonModel.illustrative()
        custom = Excitation.from_callable(lambda t: 1000.0 * np.sin(2 * np.pi * 50 * t),
                                          lambda t: 1000.0 * 2 * np.pi * 50 * np.cos(2 * np.pi * 50 * t), period=0.02)
        with pytest.raises(OutOfDomainError):
            model.trajectory(custom)
        out = model.trajectory(custom, policy="warn")
        assert not out.domain.inside and any(w.code == "out_of_domain" for w in out.warnings)
        with pytest.raises(InvalidInputError):
            model.trajectory(custom, policy="maybe")

    def test_initial_state_and_misuse(self):
        model = JilesAthertonModel.illustrative()
        out = model.trajectory(Excitation.sinusoidal(1000.0, 50.0), initial_state=5e5)
        assert any(a.id == "ja.initial_state" and a.basis == "user_input" for a in out.assumptions)
        with pytest.raises(InvalidInputError):
            model.trajectory(Excitation.sinusoidal(1000.0, 50.0), t_span=(0.0, 0.1))
        with pytest.raises(InvalidInputError):
            model.simulate(Excitation.from_callable(lambda t: 0 * t, lambda t: 0 * t))  # no period, no t_span
        with pytest.raises(OutOfDomainError):  # the domain check (custom kind) fires before simulation
            model.trajectory(Excitation.from_callable(lambda t: 0 * t, lambda t: 0 * t))
        with pytest.raises(InvalidInputError):
            model.trajectory(Excitation.sinusoidal(1000.0, 50.0), points_per_period=4)

    def test_dc_bias_and_triangular(self):
        model = JilesAthertonModel.illustrative()
        out = model.trajectory(Excitation.triangular(2000.0, 50.0, offset=500.0))
        assert out.domain.inside and out.domain.checked["dc_bias"] == "supported"
        assert out.values["field_peak"].value == pytest.approx(2500.0)
        assert out.values["loop_energy_density"].value > 0.0

    def test_solver_config_passthrough(self):
        model = JilesAthertonModel.illustrative()
        cfg = SolverConfig(method="DOP853", rtol=1e-8, atol=1e-3, max_step=1e-4)
        out = model.trajectory(Excitation.sinusoidal(2000.0, 50.0), config=cfg)
        assert out.diagnostics["solver"]["method"] == "DOP853"

    def test_ode_driven_directly_through_framework(self):
        ode = JilesAthertonModel.illustrative().ode
        sol = solve(ode, Excitation.sinusoidal(1000.0, 50.0), (0.0, 0.02), SolverConfig(rtol=1e-8, atol=1e-3))
        assert sol.state_names == ("M",) and "B" in sol.outputs and sol.constraint_violations == []
        with pytest.raises(InvalidInputError):
            solve(ode, Excitation.from_callable(lambda t: 0 * t), (0.0, 0.02))  # needs dH/dt


# ---------------------------------------------------------------------------------------------
# regressions from the independent review (initial state, DC bias, small signal, asymmetry, errors)
# ---------------------------------------------------------------------------------------------
class TestInitialStateAndBias:
    @pytest.mark.parametrize("exc", [
        Excitation.sinusoidal(2000.0, 50.0, offset=500.0),
        Excitation.sinusoidal(2000.0, 50.0, offset=1000.0),
        Excitation.sinusoidal(2000.0, 50.0, phase=0.3),
        Excitation.sinusoidal(5000.0, 50.0, phase=math.pi / 2),
        Excitation.triangular(200.0, 50.0, offset=-1000.0),
        Excitation.triangular(50.0, 50.0, offset=-2000.0),
    ], ids=["offset500", "offset1000", "phase0.3", "phase_pi_2", "tri_minor_-1000", "tri_minor_-2000"])
    def test_nonzero_start_is_premagnetised(self, exc):
        model = JilesAthertonModel.illustrative()
        out = model.trajectory(exc)
        assert out.diagnostics["steady_state"] and out.values["loop_energy_density"].value > 0.0
        assert out.values["magnetisation_peak"].value <= model.ode.Ms and not out.diagnostics["constraint_violations"]
        premag = out.diagnostics["premagnetisation"]
        assert premag["H_end_A_per_m"] == pytest.approx(exc(0.0))
        assert abs(premag["gap_M_an_minus_M_A_per_m"]) < model.ode.admissible_gap
        assert out.diagnostics["initial_state"]["source"] == "premagnetisation_ramp"
        assumption = next(a for a in out.assumptions if a.id == "ja.initial_state")
        assert "ramp" in assumption.statement and assumption.basis == "model_assumption"

    def test_sampled_field_with_nonzero_start_is_premagnetised(self):
        model = JilesAthertonModel.illustrative()
        t = np.linspace(0.0, 0.02, 201)
        out = model.trajectory((t, 500.0 + 1000.0 * np.sin(2 * np.pi * 50 * t)))
        assert out.diagnostics["premagnetisation"] is not None
        assert out.diagnostics["premagnetisation"]["H_end_A_per_m"] == pytest.approx(500.0)
        assert out.diagnostics["trajectory"]["M"][0] == pytest.approx(out.diagnostics["premagnetisation"]["M_end_A_per_m"])

    def test_premagnetised_state_equals_the_initial_curve(self):
        model = JilesAthertonModel.illustrative()
        cfg = model.default_config(1.0)
        ramp = Excitation.piecewise_linear([0.0, 1.0], [0.0, 1000.0], periodic=False)
        sol = solve(model.ode, ramp, (0.0, 1.0), cfg, t_eval=[0.0, 1.0])
        out = model.trajectory(Excitation.constant(1000.0))
        assert out.diagnostics["premagnetisation"]["M_end_A_per_m"] == pytest.approx(float(sol.y[0, -1]), rel=1e-9)

    def test_constant_field_holds_the_initial_curve_magnetisation(self):
        model = JilesAthertonModel.illustrative()
        out = model.trajectory(Excitation.constant(1000.0))
        m_peak = out.values["magnetisation_peak"].value
        b_peak = out.values["flux_density_peak"].value
        assert m_peak > 1e5 and b_peak > 100.0 * MU_0 * 1000.0  # not the vacuum response
        assert b_peak == pytest.approx(MU_0 * (1000.0 + m_peak))
        static = next(w for w in out.warnings if w.code == "ja.static_field")
        assert "pre-magnetisation ramp" in static.message and static.severity == "info"
        held = model.trajectory(Excitation.constant(1000.0), initial_state=2e5)
        assert held.values["magnetisation_peak"].value == 2e5
        assert "supplied initial state" in next(w for w in held.warnings if w.code == "ja.static_field").message
        zero = model.trajectory(Excitation.constant(0.0))
        assert zero.values["magnetisation_peak"].value == 0.0 and zero.diagnostics["premagnetisation"] is None

    def test_inadmissible_initial_state_is_rejected(self):
        model = JilesAthertonModel.illustrative()
        with pytest.raises(InvalidInputError) as info:
            model.trajectory(Excitation.sinusoidal(1000.0, 50.0), initial_state=-1.5e6)  # far below M_an, ascending
        assert "admissible gap" in str(info.value)
        with pytest.raises(InvalidInputError):
            model.ode.check_initial_state(0.0, -1.5e6, 1.0)
        model.ode.check_initial_state(0.0, -1.5e6, -1.0)  # descending start is reversible: admissible at t0
        model.ode.check_initial_state(0.0, -1.5e6, 0.0)  # static start: no change at all
        remanent = model.trajectory(Excitation.sinusoidal(1000.0, 50.0), initial_state=4e5)  # reachable remanent state
        assert remanent.diagnostics["initial_state"]["source"] == "user_input" and remanent.diagnostics["steady_state"]

    def test_small_signal_loops_converge_with_defaults(self):
        model = JilesAthertonModel.illustrative()
        for amp in (10.0, 50.0, 100.0):
            out = model.trajectory(Excitation.sinusoidal(amp, 50.0))
            assert out.diagnostics["steady_state"] and out.diagnostics["periods"] <= 12
            assert out.diagnostics["accelerated"] and out.values["loop_energy_density"].value > 0.0
            assert not any(w.severity != "info" for w in out.warnings)
        with pytest.raises(ConvergenceError) as info:
            model.trajectory(Excitation.sinusoidal(50.0, 50.0), accelerate=False, max_periods=10)
        message = str(info.value)
        assert "contraction ratio" in message and "max_periods" in message and "accelerate" in message
        assert info.value.details["contraction_ratio"] is not None and info.value.details["max_periods"] == 10

    def test_biased_loop_reports_asymmetry(self):
        model = JilesAthertonModel.illustrative()
        out = model.trajectory(Excitation.sinusoidal(2000.0, 50.0, offset=-1000.0))
        warning = next(w for w in out.warnings if w.code == "ja.asymmetric_loop")
        assert warning.severity == "info" and "DC-biased" in warning.message
        loop = out.diagnostics["loop"]
        assert loop["n_remanence_crossings"] == 2 and loop["remanence_asymmetry"] > 1e-3
        magnitudes = [abs(c["B"]) for c in loop["remanence_crossings"]]
        assert out.values["remanence"].value == pytest.approx(float(np.mean(magnitudes)))
        symmetric = model.trajectory(Excitation.sinusoidal(3000.0, 50.0))
        assert not any(w.code == "ja.asymmetric_loop" for w in symmetric.warnings)
        assert symmetric.diagnostics["loop"]["remanence_asymmetry"] < 1e-4
        no_crossing = model.trajectory(Excitation.triangular(200.0, 50.0, offset=-1000.0))
        assert "remanence" not in no_crossing.values and any(w.code == "ja.no_remanence" for w in no_crossing.warnings)

    def test_unknown_kwargs_and_bad_counts_rejected(self):
        model = JilesAthertonModel.illustrative()
        with pytest.raises(InvalidInputError):
            model.trajectory(Excitation.sinusoidal(1e3, 50.0), bogus=1)
        with pytest.raises(InvalidInputError):
            model.trajectory(Excitation.sinusoidal(1e3, 50.0), max_periods=0)

    def test_multivalued_anhysteretic_set_is_rejected(self):
        with pytest.raises(InvalidInputError) as info:
            JilesAthertonModel(1.6e6, 1100.0, 0.01, 400.0, 0.1)  # alpha*c*Ms/(3a) < 1 but alpha*Ms/(3a) = 4.8
        assert "multivalued" in str(info.value)
        ode = JilesAthertonODE(**JA_ILLUSTRATIVE.as_kwargs())
        assert ode.anhysteretic_initial_susceptibility() == pytest.approx(ode.chi_e0 / (1.0 - ode.anhysteretic_feedback))
        assert ode.admissible_gap == pytest.approx((1 - ode.c) * ode.k / ode.alpha)

    def test_loop_metrics_are_robust_to_round_off_at_the_record_ends(self):
        phi = np.linspace(0.0, 2 * np.pi, 1001)
        for eps in (-2e-12, 2e-12):
            H = 1e4 * np.sin(phi)
            B = 0.6 * np.sin(phi - 0.3)
            H[0], H[-1] = eps, -eps  # round-off of opposite signs at the two ends of the period
            metrics = loop_metrics(H, B)
            assert metrics["n_remanence_crossings"] == 2 and metrics["n_coercivity_crossings"] == 2
            assert metrics["remanence"] == pytest.approx(0.6 * math.sin(0.3), rel=1e-4)
            assert metrics["coercivity"] == pytest.approx(1e4 * math.sin(0.3), rel=1e-4)
            assert metrics["remanence_asymmetry"] < 1e-6
        assert loop_metrics(np.zeros(8), np.zeros(8))["remanence"] is None


# ---------------------------------------------------------------------------------------------
# verification checks of this module must pass
# ---------------------------------------------------------------------------------------------
JA_CHECKS = [
    check_physics.ja_zero_excitation,
    check_physics.ja_small_signal_initial_susceptibility,
    check_physics.ja_major_loop_closes_and_is_dissipative,
    check_physics.ja_descending_branch_above_ascending,
    check_physics.ja_solver_tolerance_convergence,
    check_physics.ja_dc_biased_minor_loops_admissible,
    check_physics.ja_small_signal_steady_state,
]


@pytest.mark.parametrize("func", JA_CHECKS, ids=[f.check_id for f in JA_CHECKS])
def test_jiles_atherton_checks_pass(func):
    result = func()
    assert result.passed, f"{func.check_id} failed: {result.metrics}"
