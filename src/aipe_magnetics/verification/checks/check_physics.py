"""Verification checks for the physics-based models (Jiles-Atherton, Bertotti).

Capabilities: ``aipe.magnetics.model.jiles_atherton`` and ``aipe.magnetics.model.bertotti``.

The Jiles-Atherton checks use the *illustrative* parameter set ``ja.illustrative-demo`` (Ms = 1.6e6
A/m, a = 1100 A/m, alpha = 1.6e-3, k = 400 A/m, c = 0.2; not a material).  They establish numerical
and physical consistency of the implementation (L2), not agreement with any measured material.
Every evidence record declares exactly the excitation kinds that the check actually drives.
The Bertotti checks use synthetic coefficients to exercise scaling laws and limiting cases, and
example lamination values (sigma = 2e6 S/m, d = 0.35 mm, labelled as examples) for the classical
eddy-current coefficient.
"""

from __future__ import annotations

import math

import numpy as np

from aipe_magnetics.common.errors import InvalidInputError, NumericalError, OutOfDomainError
from aipe_magnetics.models.contract import ValidityRange
from aipe_magnetics.models.dynamic import Excitation, SolverConfig, solve
from aipe_magnetics.models.physics import (
    BERTOTTI_MODEL_ID,
    JA_ILLUSTRATIVE,
    JA_MODEL_ID,
    LAMINATION_EXAMPLE,
    BertottiLossModel,
    JilesAthertonModel,
)
from aipe_magnetics.verification.evidence import OperatingRange
from aipe_magnetics.verification.suite import CheckResult, check

_JA_NOTE = ("illustrative parameter set ja.illustrative-demo (not a material); H peak up to 10 kA/m; rate independent, "
            "frequency immaterial")
JA_RANGE_SINUSOIDAL = OperatingRange(excitation=["sinusoidal"], note=_JA_NOTE)
JA_RANGE_SIN_TRI = OperatingRange(excitation=["sinusoidal", "triangular"], note=_JA_NOTE)
JA_RANGE_ZERO = OperatingRange(excitation=["sinusoidal", "triangular", "dc"], note=_JA_NOTE + "; zero amplitude")
JA_RANGE_RAMP = OperatingRange(excitation=["piecewise_linear"], note=_JA_NOTE + "; monotone ramp 0 -> 0.1 A/m")
JA_RANGE_SMALL = OperatingRange(excitation=["sinusoidal"], note=_JA_NOTE + "; small-signal minor loops 10-100 A/m")
BERTOTTI_RANGE = OperatingRange(excitation=["sinusoidal"], frequency=ValidityRange(min=10.0, max=1e5, unit="Hz"),
                                flux_density_peak=ValidityRange(min=0.01, max=2.0, unit="T"),
                                note="synthetic coefficients; scaling laws only")


def _ja() -> JilesAthertonModel:
    return JilesAthertonModel.from_parameter_set(JA_ILLUSTRATIVE)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# ---------------------------------------------------------------------------------------------
# Jiles-Atherton
# ---------------------------------------------------------------------------------------------
@check(
    id="ja.zero-excitation-gives-zero-magnetisation",
    capability=JA_MODEL_ID, level="L2", kind="limiting_case",
    description="With H identically zero (constant 0, a zero-amplitude sinusoid and a zero-amplitude triangle) the "
                "Jiles-Atherton ODE keeps M = 0 exactly, B = 0 and the loop energy is 0.",
    method="trajectory() from the demagnetised state with Excitation.constant(0), Excitation.sinusoidal(0, 50 Hz) and "
           "Excitation.triangular(0, 50 Hz); require max|M| == 0, max|B| == 0 and zero loop energy (the rhs returns "
           "exactly 0 when dH/dt = 0 and the Langevin series gives M_an(0) = 0); remanence and coercivity must be "
           "reported as undefined (no zero crossing of an identically zero signal).",
    range=JA_RANGE_ZERO, tags=("ja", "limiting"),
)
def ja_zero_excitation() -> CheckResult:
    model = _ja()
    out_dc = model.trajectory(Excitation.constant(0.0))
    out_ac = model.trajectory(Excitation.sinusoidal(0.0, 50.0))
    out_tri = model.trajectory(Excitation.triangular(0.0, 50.0))
    m_dc = max(abs(v) for v in out_dc.diagnostics["trajectory"]["M"])
    m_ac = max(abs(v) for v in out_ac.diagnostics["trajectory"]["M"])
    m_tri = max(abs(v) for v in out_tri.diagnostics["trajectory"]["M"])
    b_ac = out_ac.values["flux_density_peak"].value
    w_ac = out_ac.values["loop_energy_density"].value
    w_tri = out_tri.values["loop_energy_density"].value
    undefined = all("remanence" not in o.values and "coercivity" not in o.values for o in (out_ac, out_tri))
    passed = bool(m_dc == 0.0 and m_ac == 0.0 and m_tri == 0.0 and b_ac == 0.0 and w_ac == 0.0 and w_tri == 0.0
                  and undefined and not out_ac.diagnostics["constraint_violations"])
    return CheckResult(passed=passed, metrics={"max_abs_M_constant": m_dc, "max_abs_M_zero_sinusoid": m_ac,
                                               "max_abs_M_zero_triangle": m_tri, "flux_density_peak_T": b_ac,
                                               "loop_energy_density_sinusoid": w_ac, "loop_energy_density_triangle": w_tri,
                                               "remanence_coercivity_undefined": undefined})


@check(
    id="ja.small-signal-initial-susceptibility",
    capability=JA_MODEL_ID, level="L2", kind="analytical_reference",
    description="The initial slope dM/dH of the ODE at the demagnetised origin equals the analytic "
                "chi_0 = c Ms/(3a) / (1 - alpha c Ms/(3a)).",
    method="Derivation: at H = M = 0 the gap M_an - M vanishes so the irreversible term is zero and the closed form "
           "reduces to c chi_e / (1 - alpha c chi_e) with chi_e = Ms/(3a). Check (a) rhs(0, M=0, H=0, dH/dt=1) equals "
           "chi_0 to 1e-12 and (b) the secant M(H)/H after integrating a linear ramp 0 -> 0.1 A/m (rtol 1e-10) is "
           "within 2 % of chi_0 (first-order correction ~ (dM_an/dH - chi_0) H / k is about 0.06 % here).",
    range=JA_RANGE_RAMP, tags=("ja", "analytical"),
)
def ja_small_signal_initial_susceptibility() -> CheckResult:
    model = _ja()
    ode = model.ode
    chi0 = ode.initial_susceptibility()
    rhs_slope = ode.rhs(0.0, np.array([0.0]), 0.0, 1.0)[0]
    rel_rhs = abs(rhs_slope - chi0) / chi0
    h_small = 0.1
    ramp = Excitation.piecewise_linear([0.0, 1.0], [0.0, h_small], periodic=False)
    sol = solve(ode, ramp, (0.0, 1.0), SolverConfig(rtol=1e-10, atol=1e-12), t_eval=[0.0, 1.0])
    secant = float(sol.y[0, -1]) / h_small
    rel_secant = abs(secant - chi0) / chi0
    passed = bool(rel_rhs < 1e-12 and rel_secant < 0.02)
    return CheckResult(passed=passed,
                       metrics={"chi_0_analytic": chi0, "rhs_slope": rhs_slope, "rel_error_rhs": rel_rhs,
                                "secant_slope": secant, "rel_error_secant": rel_secant, "H_small_A_per_m": h_small,
                                "tolerance_secant": 0.02,
                                "anhysteretic_initial_susceptibility": ode.anhysteretic_initial_susceptibility()})


def _branches(trace, h_fraction: float = 0.8, n: int = 41):
    """Interpolate B on the ascending (dH/dt > 0) and descending branches at common interior H values."""
    dH = np.gradient(trace.H, trace.t)
    out = []
    for mask in (dH > 0.0, dH < 0.0):
        h, b = trace.H[mask], trace.B[mask]
        order = np.argsort(h)
        out.append((h[order], b[order]))
    h_peak = float(np.max(np.abs(trace.H)))
    grid = np.linspace(-h_fraction * h_peak, h_fraction * h_peak, n)
    b_asc = np.interp(grid, out[0][0], out[0][1])
    b_des = np.interp(grid, out[1][0], out[1][1])
    return grid, b_asc, b_des


@check(
    id="ja.major-loop-closes-and-is-dissipative",
    capability=JA_MODEL_ID, level="L2", kind="physical_consistency",
    description="After periodic steady state the 10 kA/m major loop closes, dissipates energy (oint H dB > 0), keeps "
                "|M| <= Ms, has positive, symmetric coercivity and remanence, and - the model being rate independent - "
                "the loop driven by a 10 kA/m triangular wave is the same loop as the sinusoidal one.",
    method="trajectory() with Excitation.sinusoidal(1e4 A/m, 50 Hz) and Excitation.triangular(1e4 A/m, 50 Hz), "
           "steady-state tolerance 1e-4; require convergence, closure error of B below 1e-6, loop energy > 0, no "
           "|M| > Ms constraint violation, coercivity > 0, remanence > 0, the two H = 0 / B = 0 crossings symmetric to "
           "1e-4 relative, and triangular vs sinusoidal agreement of the loop energy (1e-4), remanence (1e-4), "
           "coercivity (1e-3) and peak flux density (1e-5); the maximum B difference between the two loops at common "
           "H values is reported (interpolation-limited, bound 1 % of B peak).",
    range=JA_RANGE_SIN_TRI, tags=("ja", "loop"),
)
def ja_major_loop_closes_and_is_dissipative() -> CheckResult:
    model = _ja()
    out = model.trajectory(Excitation.sinusoidal(1e4, 50.0))
    loop = out.diagnostics["loop"]
    w = out.values["loop_energy_density"].value
    hc = out.values["coercivity"].value
    br = out.values["remanence"].value
    m_peak = out.values["magnetisation_peak"].value
    rem = [c["B"] for c in loop["remanence_crossings"]]
    coe = [c["H"] for c in loop["coercivity_crossings"]]
    sym_rem = abs(abs(rem[0]) - abs(rem[1])) / br if len(rem) == 2 and br > 0 else math.inf
    sym_coe = abs(abs(coe[0]) - abs(coe[1])) / hc if len(coe) == 2 and hc > 0 else math.inf
    sinus_ok = bool(out.diagnostics["steady_state"] and loop["closure_error_B"] < 1e-6 and w > 0.0 and hc > 0.0
                    and br > 0.0 and m_peak <= model.ode.Ms and not out.diagnostics["constraint_violations"]
                    and sym_rem < 1e-4 and sym_coe < 1e-4)
    # rate independence: the triangular drive must produce the same steady loop
    tri = model.trajectory(Excitation.triangular(1e4, 50.0))
    tri_loop = tri.diagnostics["loop"]
    d_w = _rel(tri.values["loop_energy_density"].value, w)
    d_br = _rel(tri.values["remanence"].value, br)
    d_hc = _rel(tri.values["coercivity"].value, hc)
    d_bpk = _rel(tri.values["flux_density_peak"].value, out.values["flux_density_peak"].value)
    trace_sin = model.simulate(Excitation.sinusoidal(1e4, 50.0))
    trace_tri = model.simulate(Excitation.triangular(1e4, 50.0))
    grid, asc_s, des_s = _branches(trace_sin)
    _, asc_t, des_t = _branches(trace_tri)
    branch_diff = float(max(np.max(np.abs(asc_s - asc_t)), np.max(np.abs(des_s - des_t))))
    bpk = out.values["flux_density_peak"].value
    tri_ok = bool(tri.diagnostics["steady_state"] and tri_loop["closure_error_B"] < 1e-6 and d_w < 1e-4 and d_br < 1e-4
                  and d_hc < 1e-3 and d_bpk < 1e-5 and branch_diff < 1e-2 * bpk)
    return CheckResult(passed=bool(sinus_ok and tri_ok),
                       metrics={"periods": out.diagnostics["periods"], "closure_error_B": loop["closure_error_B"],
                                "loop_energy_density_J_per_m3": w, "coercivity_A_per_m": hc, "remanence_T": br,
                                "flux_density_peak_T": bpk, "magnetisation_peak_over_Ms": m_peak / model.ode.Ms,
                                "remanence_symmetry": sym_rem, "coercivity_symmetry": sym_coe,
                                "n_steps": out.diagnostics["solver"]["n_steps"],
                                "triangular_periods": tri.diagnostics["periods"],
                                "triangular_closure_error_B": tri_loop["closure_error_B"],
                                "triangular_vs_sinusoidal_rel": {"loop_energy": d_w, "remanence": d_br,
                                                                 "coercivity": d_hc, "flux_density_peak": d_bpk},
                                "triangular_vs_sinusoidal_max_dB_T": branch_diff},
                       notes=["illustrative parameters: the numbers characterise the model implementation, not a "
                              "material",
                              "the branch comparison interpolates B(H) on a 41-point grid from 1000 samples per period; "
                              "its residual (1e-3 T) is interpolation error near the steep parts of the loop, the "
                              "integral and crossing quantities agree to 1e-4 or better"])


@check(
    id="ja.descending-branch-above-ascending",
    capability=JA_MODEL_ID, level="L2", kind="physical_consistency",
    description="Branch ordering / sign convention: at the same interior H the descending branch (dH/dt < 0) carries a "
                "higher B than the ascending branch, i.e. the loop is traversed counter-clockwise in the (H, B) plane, "
                "consistent with oint H dB > 0.",
    method="Simulate the 10 kA/m, 50 Hz steady-state loop, split it by the sign of dH/dt, sort each branch by H and "
           "interpolate B at 41 common values with |H| <= 0.8 H_peak; require B_desc - B_asc > 0 everywhere and equal "
           "to 2 B_r at H = 0 within 1e-3. The task brief phrased this as 'descending below ascending'; with B and H "
           "positive along the same reference direction the physically correct ordering, verified here, is "
           "descending >= ascending.",
    range=JA_RANGE_SINUSOIDAL, tags=("ja", "loop", "sign-convention"),
)
def ja_descending_branch_above_ascending() -> CheckResult:
    model = _ja()
    trace = model.simulate(Excitation.sinusoidal(1e4, 50.0))
    grid, b_asc, b_des = _branches(trace)
    gap = b_des - b_asc
    out = model.trajectory(Excitation.sinusoidal(1e4, 50.0))
    br = out.values["remanence"].value
    gap_at_zero = float(np.interp(0.0, grid, gap))
    rel_zero = abs(gap_at_zero - 2.0 * br) / (2.0 * br)
    passed = bool(np.all(gap > 0.0) and rel_zero < 1e-3)
    return CheckResult(passed=passed, metrics={"min_gap_T": float(np.min(gap)), "max_gap_T": float(np.max(gap)),
                                               "gap_at_H0_T": gap_at_zero, "two_Br_T": 2.0 * br,
                                               "rel_error_gap_vs_2Br": rel_zero, "n_grid": int(grid.size)})


@check(
    id="ja.solver-tolerance-convergence",
    capability=JA_MODEL_ID, level="L2", kind="convergence",
    description="The steady-state loop energy is insensitive to the solver tolerance (rtol 1e-6 vs 1e-9 changes it by "
                "less than 0.5 %) and agrees between RK45 and DOP853.",
    method="trajectory() with a 5 kA/m, 50 Hz sinusoid, max_step = period/200, RK45 at rtol 1e-6 and 1e-9 and DOP853 at "
           "rtol 1e-9; compare loop energy densities and coercivities.",
    range=JA_RANGE_SINUSOIDAL, tags=("ja", "convergence"),
)
def ja_solver_tolerance_convergence() -> CheckResult:
    model = _ja()
    exc = Excitation.sinusoidal(5e3, 50.0)
    max_step = exc.period / 200.0
    atol = 1e-9 * model.ode.Ms
    results = {}
    for label, method, rtol in (("rk45_1e-6", "RK45", 1e-6), ("rk45_1e-9", "RK45", 1e-9), ("dop853_1e-9", "DOP853", 1e-9)):
        out = model.trajectory(exc, config=SolverConfig(method=method, rtol=rtol, atol=atol * (rtol / 1e-6),
                                                        max_step=max_step))
        results[label] = {"w": out.values["loop_energy_density"].value, "hc": out.values["coercivity"].value,
                          "n_steps": out.diagnostics["solver"]["n_steps"]}
    w_ref = results["rk45_1e-9"]["w"]
    change_tol = abs(results["rk45_1e-6"]["w"] - w_ref) / w_ref
    change_method = abs(results["dop853_1e-9"]["w"] - w_ref) / w_ref
    change_hc = abs(results["rk45_1e-6"]["hc"] - results["rk45_1e-9"]["hc"]) / results["rk45_1e-9"]["hc"]
    passed = bool(change_tol < 5e-3 and change_method < 1e-3 and change_hc < 5e-3)
    return CheckResult(passed=passed, metrics={"loop_energy_rel_change_rtol": change_tol,
                                               "loop_energy_rel_change_method": change_method,
                                               "coercivity_rel_change_rtol": change_hc, "runs": results,
                                               "tolerance": 5e-3})


@check(
    id="ja.dc-biased-minor-loops-admissible",
    capability=JA_MODEL_ID, level="L2", kind="physical_consistency",
    description="Excitations that do not start at H = 0 (DC offset, non-zero phase, biased triangular minor loops) are "
                "integrated from an admissible pre-magnetised state, converge to dissipative steady loops, obey the "
                "H -> -H point symmetry of the model, and the Steffensen-accelerated steady state coincides with "
                "plain period iteration.",
    method="Illustrative set. Cases: sinusoidal 2 kA/m at +1 kA/m and -1 kA/m offset, sinusoidal 5 kA/m with phase "
           "pi/2 (starts at the peak, dH/dt(0) ~ 0), triangular 200 A/m at -1 kA/m offset (minor loop without zero "
           "crossing of H). Require: no ConvergenceError, steady state reached, oint H dB > 0, |M| <= Ms, the "
           "pre-magnetisation ramp recorded with |M_an - M| below the admissible gap (1-c)k/alpha; the +1 kA/m and "
           "-1 kA/m loops mirror each other (B_-(t) = -B_+(t + T/2) within 1e-4 of B peak, loop energies within "
           "1e-5); the biased triangular loop computed with accelerate=True and accelerate=False (both at "
           "steady_state_tol 1e-5 with the default solver settings, whose period-to-period noise floor for this "
           "loop is about 3e-6 of the excursion of M) agrees in loop energy to 5e-4 and in M(t) to 2e-4 of its "
           "peak-to-peak excursion (both bounds are at least eight times the tolerance-limited differences "
           "observed, 3.4e-5 and 2.5e-5).",
    range=JA_RANGE_SIN_TRI, tags=("ja", "loop", "dc-bias"),
)
def ja_dc_biased_minor_loops_admissible() -> CheckResult:
    model = _ja()
    gap_limit = model.ode.admissible_gap
    cases = {
        "sin_2000_offset_+1000": Excitation.sinusoidal(2000.0, 50.0, offset=1000.0),
        "sin_2000_offset_-1000": Excitation.sinusoidal(2000.0, 50.0, offset=-1000.0),
        "sin_5000_phase_pi_2": Excitation.sinusoidal(5000.0, 50.0, phase=math.pi / 2),
        "tri_200_offset_-1000": Excitation.triangular(200.0, 50.0, offset=-1000.0),
    }
    outputs = {}
    per_case: dict[str, dict] = {}
    all_ok = True
    for name, exc in cases.items():
        out = model.trajectory(exc)
        outputs[name] = out
        premag = out.diagnostics["premagnetisation"]
        gap_ok = premag is not None and abs(premag["gap_M_an_minus_M_A_per_m"]) < gap_limit
        ok = bool(out.diagnostics["steady_state"] and out.values["loop_energy_density"].value > 0.0
                  and out.values["magnetisation_peak"].value <= model.ode.Ms
                  and not out.diagnostics["constraint_violations"] and gap_ok)
        per_case[name] = {"ok": ok, "periods": out.diagnostics["periods"],
                          "loop_energy_density": out.values["loop_energy_density"].value,
                          "flux_density_peak_T": out.values["flux_density_peak"].value,
                          "premagnetisation_M_A_per_m": None if premag is None else premag["M_end_A_per_m"],
                          "premagnetisation_gap_A_per_m": None if premag is None else premag["gap_M_an_minus_M_A_per_m"],
                          "warnings": [w.code for w in out.warnings]}
        all_ok = all_ok and ok
    # point symmetry H -> -H: the -1 kA/m loop is the +1 kA/m loop shifted by half a period and negated
    plus, minus = outputs["sin_2000_offset_+1000"], outputs["sin_2000_offset_-1000"]
    b_plus = np.array(plus.diagnostics["trajectory"]["B"])
    b_minus = np.array(minus.diagnostics["trajectory"]["B"])
    n = b_plus.size - 1
    half = n // 2
    mirror = float(np.max(np.abs(b_minus[:half] + b_plus[half:half + half])) / np.max(np.abs(b_plus)))
    mirror_w = _rel(plus.values["loop_energy_density"].value, minus.values["loop_energy_density"].value)
    # accelerated vs plain period iteration on the biased minor loop (same solver settings for both)
    tri = cases["tri_200_offset_-1000"]
    acc = model.trajectory(tri, steady_state_tol=1e-5)
    plain = model.trajectory(tri, steady_state_tol=1e-5, accelerate=False, max_periods=400)
    d_w = _rel(acc.values["loop_energy_density"].value, plain.values["loop_energy_density"].value)
    m_acc = np.array(acc.diagnostics["trajectory"]["M"])
    m_plain = np.array(plain.diagnostics["trajectory"]["M"])
    d_m = float(np.max(np.abs(m_acc - m_plain)) / (np.max(m_plain) - np.min(m_plain)))
    passed = bool(all_ok and mirror < 1e-4 and mirror_w < 1e-5 and d_w < 5e-4 and d_m < 2e-4
                  and acc.diagnostics["steady_state"] and plain.diagnostics["steady_state"])
    return CheckResult(passed=passed, metrics={
        "cases": per_case, "admissible_gap_A_per_m": gap_limit, "mirror_max_rel_B_error": mirror,
        "mirror_loop_energy_rel": mirror_w, "accelerated_periods": acc.diagnostics["periods"],
        "accelerated_extrapolations": acc.diagnostics["extrapolations"], "plain_periods": plain.diagnostics["periods"],
        "plain_contraction_ratio": plain.diagnostics["contraction_ratio"],
        "accelerated_vs_plain_loop_energy_rel": d_w, "accelerated_vs_plain_max_dM_rel_ptp": d_m},
        notes=["illustrative parameters; the pre-magnetisation ramp (0 -> H(t0) along the initial curve) is the "
               "documented default initial state for excitations with H(t0) != 0"])


@check(
    id="ja.small-signal-steady-state",
    capability=JA_MODEL_ID, level="L2", kind="convergence",
    description="Small-amplitude symmetric minor loops (10, 50, 100 A/m), whose period map contracts by only a few "
                "per cent per period, reach the periodic steady state with the default settings in a handful of "
                "periods, are symmetric, dissipative and ordered in energy, and the accelerated steady loop equals "
                "the plain-iteration loop.",
    method="Illustrative set, 50 Hz sinusoids of 10, 50 and 100 A/m with trajectory() defaults (accelerate=True, "
           "steady_state_tol 1e-4): require convergence in at most 12 integrated periods each, oint H dB > 0 "
           "increasing with amplitude, |mean M| of the last period below 10 x tol of the peak-to-peak M (symmetry), "
           "and at 100 A/m agreement of the loop energy between the accelerated and the plain period iteration "
           "(accelerate=False, up to 150 periods; both RK45 rtol 1e-7, max_step = period/100) to 1e-4 while the plain "
           "iteration needs more than 20 periods (contraction ratio reported). The exponent of the "
           "energy-vs-amplitude power law between 50 and 100 A/m is reported as a metric only.",
    range=JA_RANGE_SMALL, tags=("ja", "convergence", "small-signal"),
)
def ja_small_signal_steady_state() -> CheckResult:
    model = _ja()
    tol = 1e-4
    runs: dict[str, dict] = {}
    energies = []
    ok = True
    for amp in (10.0, 50.0, 100.0):
        out = model.trajectory(Excitation.sinusoidal(amp, 50.0))
        m = np.array(out.diagnostics["trajectory"]["M"])
        ptp = float(np.max(m) - np.min(m))
        offset = abs(float(np.mean(m[:-1]))) / ptp
        w = out.values["loop_energy_density"].value
        energies.append(w)
        case_ok = bool(out.diagnostics["steady_state"] and out.diagnostics["periods"] <= 12 and w > 0.0
                       and offset < 10.0 * tol and not out.diagnostics["constraint_violations"])
        runs[f"{amp:g}_A_per_m"] = {"ok": case_ok, "periods": out.diagnostics["periods"],
                                    "extrapolations": out.diagnostics["extrapolations"],
                                    "boundary_change": out.diagnostics["boundary_change"],
                                    "loop_energy_density": w, "mean_M_over_ptp": offset,
                                    "remanence_T": out.values["remanence"].value,
                                    "coercivity_A_per_m": out.values["coercivity"].value,
                                    "warnings": [wn.code for wn in out.warnings]}
        ok = ok and case_ok
    monotone = energies[0] < energies[1] < energies[2]
    exponent = math.log(energies[2] / energies[1]) / math.log(2.0)
    exc100 = Excitation.sinusoidal(100.0, 50.0)
    cfg = SolverConfig(method="RK45", rtol=1e-7, atol=1e-9 * model.ode.Ms, max_step=exc100.period / 100.0)
    fast = model.trajectory(exc100, config=cfg)
    plain = model.trajectory(exc100, accelerate=False, max_periods=150, config=cfg)
    d_w = _rel(plain.values["loop_energy_density"].value, fast.values["loop_energy_density"].value)
    passed = bool(ok and monotone and d_w < 1e-4 and plain.diagnostics["periods"] > 20 and plain.diagnostics["steady_state"])
    return CheckResult(passed=passed, metrics={
        "runs": runs, "energies_monotone": monotone, "energy_exponent_50_to_100": exponent,
        "plain_periods_100_A_per_m": plain.diagnostics["periods"],
        "accelerated_periods_100_A_per_m_same_config": fast.diagnostics["periods"],
        "plain_contraction_ratio_100_A_per_m": plain.diagnostics["contraction_ratio"],
        "accelerated_vs_plain_loop_energy_rel_100_A_per_m": d_w, "steady_state_tol": tol},
        notes=["illustrative parameters; the energy exponent is reported, not asserted (a cubic, Rayleigh-like "
               "dependence is a plausibility observation, not a reference comparison)"])


# ---------------------------------------------------------------------------------------------
# Bertotti
# ---------------------------------------------------------------------------------------------
def _synthetic_bertotti(**kwargs) -> BertottiLossModel:
    # synthetic coefficients chosen only to exercise the scaling laws; not a material fit
    return BertottiLossModel(k_h=20.0, beta_h=1.8, k_c=0.05, k_e=0.5, **kwargs)


@check(
    id="bertotti.scaling-and-limiting-cases",
    capability=BERTOTTI_MODEL_ID, level="L2", kind="limiting_case",
    description="Each Bertotti term scales as its power law (f, f^2, f^1.5; Bpk^beta_h, Bpk^2, Bpk^1.5), the total is "
                "the sum of the components, zero excitation gives zero loss, the loss is monotone in f and Bpk and "
                "overflow is reported as NumericalError.",
    method="Synthetic coefficients (k_h = 20, beta_h = 1.8, k_c = 0.05, k_e = 0.5); doubling ratios on a grid of "
           "f in [10, 1e5] Hz and Bpk in [0.01, 2] T compared with the exact exponents to 1e-12; energy per cycle "
           "= Pv / f; Bpk = 1e200 T must raise NumericalError.",
    range=BERTOTTI_RANGE, tags=("bertotti", "scaling"),
)
def bertotti_scaling_and_limiting_cases() -> CheckResult:
    model = _synthetic_bertotti()
    worst = 0.0
    exponents = {"hysteresis": (1.0, model.beta_h), "classical": (2.0, 2.0), "excess": (1.5, 1.5)}
    for f in (10.0, 1e3, 5e4):
        for b in (0.01, 0.3, 1.0):
            base = model.components(f, b)
            f2 = model.components(2.0 * f, b)
            b2 = model.components(f, 2.0 * b)
            for name, (pf, pb) in exponents.items():
                worst = max(worst, abs(f2[name] / base[name] - 2.0 ** pf), abs(b2[name] / base[name] - 2.0 ** pb))
            worst = max(worst, abs(base["total"] - (base["hysteresis"] + base["classical"] + base["excess"]))
                        / base["total"])
    zero_f = model.loss_density(frequency=0.0, flux_density_peak=1.0).values["loss_density"].value
    zero_b = model.loss_density(frequency=1e3, flux_density_peak=0.0).values["loss_density"].value
    out = model.loss_density(frequency=1e3, flux_density_peak=1.0)
    per_cycle = out.values["energy_per_cycle_density"].value
    per_cycle_ok = abs(per_cycle - out.values["loss_density"].value / 1e3) < 1e-12 * per_cycle
    fs = np.logspace(1, 5, 9)
    bs = np.linspace(0.01, 2.0, 9)
    mono_f = all(model.components(fs[i + 1], 1.0)["total"] > model.components(fs[i], 1.0)["total"]
                 for i in range(fs.size - 1))
    mono_b = all(model.components(1e3, bs[i + 1])["total"] > model.components(1e3, bs[i])["total"]
                 for i in range(bs.size - 1))
    try:
        model.loss_density(frequency=1e3, flux_density_peak=1e200)
        overflow_reported = False
    except NumericalError:
        overflow_reported = True
    passed = bool(worst < 1e-12 and zero_f == 0.0 and zero_b == 0.0 and per_cycle_ok and mono_f and mono_b
                  and overflow_reported)
    return CheckResult(passed=passed, metrics={"max_scaling_deviation": worst, "zero_frequency_loss": zero_f,
                                               "zero_flux_loss": zero_b, "energy_per_cycle_consistent": per_cycle_ok,
                                               "monotone_in_f": mono_f, "monotone_in_B": mono_b,
                                               "overflow_reported": overflow_reported},
                       notes=["synthetic coefficients; the check verifies the implemented functional form, not a fit"])


@check(
    id="bertotti.invalid-parameters-rejected",
    capability=BERTOTTI_MODEL_ID, level="L1", kind="invalid_input_rejection",
    description="Invalid coefficients and inputs are rejected explicitly (InvalidInputError), out-of-domain inputs "
                "raise OutOfDomainError under policy 'reject' and are flagged under policy 'warn', while numpy real "
                "scalars are accepted like Python numbers.",
    method="Negative/NaN/inf/boolean/string coefficients, beta_h <= 0, negative or NaN frequency and flux density, "
           "non-positive temperature, unexpected keyword arguments, invalid lamination data (missing/unknown keys, "
           "k_c inconsistent with pi^2 sigma d^2/6) and inputs outside the declared frequency/flux ranges; "
           "np.float32 / np.int64 coefficients and inputs must be accepted.",
    tags=("bertotti", "errors"),
)
def bertotti_invalid_parameters_rejected() -> CheckResult:
    outcomes: dict[str, bool] = {}

    def expect(name: str, func, exc_type) -> None:
        try:
            func()
        except exc_type:
            outcomes[name] = True
        except Exception:  # noqa: BLE001
            outcomes[name] = False
        else:
            outcomes[name] = False

    expect("negative_k_h", lambda: BertottiLossModel(-1.0, 1.8, 0.0, 0.0), InvalidInputError)
    expect("negative_k_c", lambda: BertottiLossModel(1.0, 1.8, -1e-3, 0.0), InvalidInputError)
    expect("negative_k_e", lambda: BertottiLossModel(1.0, 1.8, 0.0, -1.0), InvalidInputError)
    expect("zero_beta", lambda: BertottiLossModel(1.0, 0.0, 0.0, 0.0), InvalidInputError)
    expect("nan_k_c", lambda: BertottiLossModel(1.0, 1.8, float("nan"), 0.0), InvalidInputError)
    expect("inf_k_h", lambda: BertottiLossModel(float("inf"), 1.8, 0.0, 0.0), InvalidInputError)
    expect("bool_k_h", lambda: BertottiLossModel(True, 1.8, 0.0, 0.0), InvalidInputError)
    expect("numpy_bool_k_h", lambda: BertottiLossModel(np.bool_(True), 1.8, 0.0, 0.0), InvalidInputError)
    expect("string_beta", lambda: BertottiLossModel(1.0, "1.8", 0.0, 0.0), InvalidInputError)
    expect("bad_range", lambda: BertottiLossModel(1.0, 1.8, 0.0, 0.0, frequency_range=(1e3, 10.0)), InvalidInputError)
    expect("bad_lamination", lambda: BertottiLossModel.classical_eddy_coefficient(-2e6, 0.35e-3), InvalidInputError)
    expect("lamination_missing_key", lambda: BertottiLossModel(20.0, 1.8, 0.05, 0.5, lamination={"sigma": 2e6}),
           InvalidInputError)
    expect("lamination_unknown_key",
           lambda: BertottiLossModel(20.0, 1.8, 0.05, 0.5, lamination={"conductivity": 2e6, "thickness": 0.35e-3,
                                                                       "width": 0.1}), InvalidInputError)
    expect("lamination_inconsistent_k_c",
           lambda: BertottiLossModel(20.0, 1.8, 0.05, 0.5, lamination={"conductivity": 2e6, "thickness": 0.35e-3}),
           InvalidInputError)
    model = _synthetic_bertotti(frequency_range=(10.0, 1e5), flux_density_range=(0.0, 2.0))
    expect("negative_frequency", lambda: model.loss_density(frequency=-1.0, flux_density_peak=1.0), InvalidInputError)
    expect("nan_flux", lambda: model.loss_density(frequency=1e3, flux_density_peak=float("nan")), InvalidInputError)
    expect("zero_temperature", lambda: model.loss_density(frequency=1e3, flux_density_peak=1.0, temperature=0.0),
           InvalidInputError)
    expect("unexpected_kwarg", lambda: model.loss_density(frequency=1e3, flux_density_peak=1.0, duty=0.5),
           InvalidInputError)
    expect("bad_policy", lambda: model.loss_density(frequency=1e3, flux_density_peak=1.0, policy="ignore"),
           InvalidInputError)
    expect("out_of_domain_reject", lambda: model.loss_density(frequency=1e6, flux_density_peak=1.0), OutOfDomainError)
    warned = model.loss_density(frequency=1e6, flux_density_peak=1.0, policy="warn")
    outcomes["out_of_domain_warn_flagged"] = bool((not warned.domain.inside)
                                                  and any(w.code == "out_of_domain" for w in warned.warnings))
    try:
        numpy_model = BertottiLossModel(np.float32(20.0), np.float64(1.8), np.int64(0), 0.5,
                                        frequency_range=(np.float32(10.0), 1e5))
        ref = _synthetic_bertotti(frequency_range=(10.0, 1e5))
        a = numpy_model.loss_density(frequency=np.int64(1000), flux_density_peak=np.float32(1.0)).values["loss_density"].value
        b = ref.loss_density(frequency=1000.0, flux_density_peak=1.0).values["loss_density"].value
        # k_c differs (0 vs 0.05) by construction; compare the hysteresis + excess parts only
        a_parts = numpy_model.components(1000.0, 1.0)
        b_parts = ref.components(1000.0, 1.0)
        outcomes["numpy_scalars_accepted"] = bool(math.isfinite(a) and math.isfinite(b)
                                                  and abs(a_parts["hysteresis"] - b_parts["hysteresis"]) <= 1e-6 * b_parts["hysteresis"]
                                                  and abs(a_parts["excess"] - b_parts["excess"]) <= 1e-12 * b_parts["excess"])
    except Exception:  # noqa: BLE001
        outcomes["numpy_scalars_accepted"] = False
    return CheckResult(passed=all(outcomes.values()), metrics={"cases": outcomes, "n_cases": len(outcomes)})


@check(
    id="bertotti.classical-eddy-coefficient",
    capability=BERTOTTI_MODEL_ID, level="L2", kind="independent_numerical",
    description="k_c = pi^2 sigma d^2 / 6 agrees with a direct numerical evaluation of the classical eddy-current loss "
                "of a thin lamination under uniform sinusoidal flux density.",
    method="Example values sigma = 2e6 S/m, d = 0.35 mm (labelled example, not a product). Direct computation: "
           "E(x, t) = -x dB/dt with B = Bpk sin(2 pi f t); P = (1/(d T)) int_0^T int_{-d/2}^{d/2} sigma E^2 dx dt by "
           "Gauss-Legendre quadrature in x (exact for the quadratic integrand) and periodic trapezoidal rule in t; "
           "compare with k_c f^2 Bpk^2 at f = 50 Hz and 1 kHz, Bpk = 1.0 and 1.5 T (relative error < 1e-10).",
    range=OperatingRange(excitation=["sinusoidal"], frequency=ValidityRange(min=50.0, max=1e3, unit="Hz"),
                         flux_density_peak=ValidityRange(min=1.0, max=1.5, unit="T"),
                         note="thin-lamination assumption (skin depth >> d) assumed, not verified here"),
    reference=LAMINATION_EXAMPLE, tags=("bertotti", "eddy"),
)
def bertotti_classical_eddy_coefficient() -> CheckResult:
    sigma, d = 2.0e6, 0.35e-3
    k_c = BertottiLossModel.classical_eddy_coefficient(sigma, d)
    expected_closed = math.pi ** 2 * sigma * d * d / 6.0
    xs, ws = np.polynomial.legendre.leggauss(8)
    x = 0.5 * d * xs
    wx = 0.5 * d * ws
    worst = 0.0
    for f in (50.0, 1e3):
        for bpk in (1.0, 1.5):
            period = 1.0 / f
            t = np.linspace(0.0, period, 2001)[:-1]
            dbdt = bpk * 2.0 * math.pi * f * np.cos(2.0 * math.pi * f * t)
            e2 = (x[:, None] * dbdt[None, :]) ** 2  # E^2 over thickness and time
            p_direct = float(np.sum(wx[:, None] * sigma * e2) / d / t.size)
            p_formula = k_c * f * f * bpk * bpk
            worst = max(worst, abs(p_direct - p_formula) / p_formula)
    via_model = BertottiLossModel.from_lamination(20.0, 1.8, 0.5, conductivity=sigma, thickness=d)
    same = abs(via_model.k_c - k_c) <= 1e-15 * k_c
    passed = bool(worst < 1e-10 and abs(k_c - expected_closed) <= 1e-15 * k_c and same)
    return CheckResult(passed=passed, metrics={"k_c_W_per_m3_Hz2_T2": k_c, "max_rel_error_direct_vs_formula": worst,
                                               "sigma_S_per_m": sigma, "thickness_m": d,
                                               "from_lamination_consistent": same},
                       notes=["sigma and d are example values (see reference); the derivation assumes uniform flux "
                              "across the thickness, i.e. skin depth much larger than d"])
