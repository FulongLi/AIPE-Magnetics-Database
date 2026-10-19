"""Verification checks for the empirical core-loss model family (classical Steinmetz, iGSE).

All checks are deterministic, offline and fast. They use the clearly labelled **synthetic** parameter
triple below (``k = 1 W/m^3``, ``alpha = 1.5``, ``beta = 2.5``, chosen so that ``Pv(100 kHz, 0.1 T) =
100 kW/m^3``); it is not a fit to any material and carries no engineering meaning. Where a check
sweeps ``alpha``/``beta`` it says so in its metrics.

Levels: L1 software correctness (invalid-input rejection, descriptor consistency), L2 numerical / physical
consistency (analytical references, convergence, limiting cases, scaling laws, published approximation,
minor-loop bound).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from aipe_magnetics.common.errors import InvalidInputError, NumericalError, OutOfDomainError
from aipe_magnetics.common.results import SourceReference
from aipe_magnetics.models.contract import ValidityRange
from aipe_magnetics.models.empirical.igse import (
    IGSE_SOURCE,
    IGSEModel,
    abs_cos_power_integral,
    abs_cos_power_integral_closed_form,
    igse_coefficient,
    igse_coefficient_venkatachalam_approximation,
    igse_trapezoidal_closed_form,
    igse_triangular_closed_form,
)
from aipe_magnetics.models.empirical.steinmetz import SteinmetzModel
from aipe_magnetics.models.waveforms import Waveform
from aipe_magnetics.verification.evidence import OperatingRange
from aipe_magnetics.verification.suite import CheckResult, check

STEINMETZ_CAPABILITY = "aipe.magnetics.model.steinmetz"
IGSE_CAPABILITY = "aipe.magnetics.model.igse"
WAVEFORM_CAPABILITY = "aipe.magnetics.model.waveforms"

SYNTHETIC_K, SYNTHETIC_ALPHA, SYNTHETIC_BETA = 1.0, 1.5, 2.5
SYNTHETIC_SOURCE = SourceReference(
    id="internal.synthetic-steinmetz-parameters",
    kind="internal",
    reference=("Synthetic Steinmetz parameter triple k=1 W/m^3, alpha=1.5, beta=2.5 (f in Hz, B in T) used only for "
               "verification of the model implementation"),
    note=("SYNTHETIC / illustrative: not a fit to any material, datasheet or measurement. Chosen so that "
          "Pv(100 kHz, 0.1 T) = 100 kW/m^3 exactly."),
)
ALPHA_SWEEP = (1.0, 1.2, 1.5, 1.8, 2.0, 2.3, 2.5)
BETA_SWEEP = (2.0, 2.5, 3.0)
FREQUENCIES = (1e3, 1e4, 1e5, 1e6)
PEAKS = (1e-3, 1e-2, 1e-1, 0.3)
TRAPEZOID_RISES = (0.1, 0.2, 0.35)  # duty_flat = 0.5 - duty_rise: genuine trapezoids with flats of 0.4, 0.3, 0.15
SAMPLED_POINTS = ((1e3, 1e-3), (1e5, 0.1), (1e6, 0.3))  # (f, Bpk) corners and centre of the declared range

SINUSOIDAL_RANGE = OperatingRange(
    excitation=["sinusoidal"],
    frequency=ValidityRange(min=1e3, max=1e6, unit="Hz"),
    flux_density_peak=ValidityRange(min=1e-3, max=0.3, unit="T"),
    note="synthetic parameters, material-agnostic; alpha in [1.0, 2.5], beta in [2.0, 3.0]",
)
TRIANGULAR_RANGE = OperatingRange(
    excitation=["triangular", "trapezoidal"],
    frequency=ValidityRange(min=1e3, max=1e6, unit="Hz"),
    flux_density_peak=ValidityRange(min=1e-3, max=0.3, unit="T"),
    note=("triangular duty 0.1-0.9 and symmetric trapezoids with duty_rise 0.1-0.35 (duty_flat = 0.5 - duty_rise); "
          "synthetic parameters, alpha in [1.0, 2.5], beta in [2.0, 3.0]"),
)
SAMPLED_RANGE = OperatingRange(
    excitation=["sampled"],
    frequency=ValidityRange(min=1e3, max=1e6, unit="Hz"),
    flux_density_peak=ValidityRange(min=1e-3, max=0.3, unit="T"),
    note=("sinusoid sampled with 64-4096 points per period at (1 kHz, 1 mT), (100 kHz, 0.1 T) and (1 MHz, 0.3 T); the "
          "relative error of the sampled path is scale invariant in f and Bpk (the check verifies this), so the "
          "convergence evidence applies across the declared range; synthetic parameters"),
)


def _models(alpha: float = SYNTHETIC_ALPHA, beta: float = SYNTHETIC_BETA,
            **kwargs: Any) -> tuple[SteinmetzModel, IGSEModel]:
    steinmetz = SteinmetzModel(SYNTHETIC_K, alpha, beta, sources=[SYNTHETIC_SOURCE], synthetic=True, **kwargs)
    return steinmetz, IGSEModel.from_steinmetz(steinmetz)


def _expect(exception: type[BaseException], action: Callable[[], Any]) -> str | None:
    """Return None when ``action`` raises exactly ``exception``; otherwise a description of what happened."""
    try:
        result = action()
    except exception:
        return None
    except Exception as error:  # noqa: BLE001 - report the wrong exception type explicitly
        return f"raised {type(error).__name__} instead of {exception.__name__}: {error}"
    return f"returned {result!r} instead of raising {exception.__name__}"


# --------------------------------------------------------------------------------------------------
# iGSE
# --------------------------------------------------------------------------------------------------
@check(
    id="igse.sinusoid-equals-steinmetz",
    capability=IGSE_CAPABILITY,
    level="L2", kind="limiting_case",
    description="For sinusoidal excitation the iGSE reduces exactly to the classical Steinmetz equation.",
    method=("Evaluate IGSEModel (default sinusoidal waveform and an explicit Waveform.sinusoidal) and SteinmetzModel "
            "on a grid of f x Bpk for several (alpha, beta) with the synthetic parameters; compare relative error."),
    range=SINUSOIDAL_RANGE,
    reference=IGSE_SOURCE,
)
def igse_sinusoid_equals_steinmetz() -> CheckResult:
    worst = 0.0
    points = 0
    for alpha in ALPHA_SWEEP:
        for beta in BETA_SWEEP:
            steinmetz, igse = _models(alpha, beta)
            for f in FREQUENCIES:
                for b in PEAKS:
                    reference = steinmetz.loss_density(frequency=f, flux_density_peak=b).value("loss_density").value
                    implicit = igse.loss_density(frequency=f, flux_density_peak=b).value("loss_density").value
                    explicit = igse.loss_density(waveform=Waveform.sinusoidal(peak=b, frequency=f)).value(
                        "loss_density").value
                    worst = max(worst, abs(implicit - reference) / reference, abs(explicit - reference) / reference)
                    points += 1
    return CheckResult(passed=worst < 1e-9, metrics={"max_rel_error": worst, "tolerance": 1e-9, "n_points": points,
                                                    "alpha_sweep": list(ALPHA_SWEEP), "beta_sweep": list(BETA_SWEEP)},
                       notes=["synthetic parameters k=1 W/m^3; alpha/beta swept"])


@check(
    id="igse.triangular-closed-form",
    capability=IGSE_CAPABILITY,
    level="L2", kind="analytical_reference",
    description=("The piecewise-linear iGSE path reproduces the analytic triangular result "
                 "Pv = k_i (2 Bpk)^beta f^alpha [D^(1-alpha) + (1-D)^(1-alpha)] and the analytic symmetric-trapezoid "
                 "result Pv = 2 k_i (2 Bpk)^beta f^alpha D_rise^(1-alpha)."),
    method=("Evaluate IGSEModel on Waveform.triangular for duty 0.1-0.9 and on Waveform.trapezoidal for duty_rise "
            "0.1, 0.2, 0.35 (duty_flat = 0.5 - duty_rise) over a grid of f x Bpk x (alpha, beta); compare with "
            "igse_triangular_closed_form / igse_trapezoidal_closed_form; also confirm that a trapezoid with "
            "duty_flat=0 equals the D=0.5 triangle and equation (5) at D_rise=0.5."),
    range=TRIANGULAR_RANGE,
    reference=IGSE_SOURCE,
)
def igse_triangular_closed_form_check() -> CheckResult:
    worst_triangle = 0.0
    worst_trapezoid = 0.0
    triangles = 0
    trapezoids = 0
    for alpha in ALPHA_SWEEP:
        for beta in BETA_SWEEP:
            _, igse = _models(alpha, beta)
            for f in (1e3, 1e5, 1e6):
                for b in (1e-3, 0.1, 0.3):
                    for duty in (0.1, 0.25, 0.5, 0.75, 0.9):
                        numeric = igse.loss_density(waveform=Waveform.triangular(peak=b, frequency=f, duty=duty)).value(
                            "loss_density").value
                        analytic = igse_triangular_closed_form(SYNTHETIC_K, alpha, beta, peak=b, frequency=f, duty=duty)
                        worst_triangle = max(worst_triangle, abs(numeric - analytic) / analytic)
                        triangles += 1
                    for rise in TRAPEZOID_RISES:
                        waveform = Waveform.trapezoidal(peak=b, frequency=f, duty_rise=rise, duty_flat=0.5 - rise)
                        numeric = igse.loss_density(waveform=waveform).value("loss_density").value
                        analytic = igse_trapezoidal_closed_form(SYNTHETIC_K, alpha, beta, peak=b, frequency=f,
                                                                duty_rise=rise)
                        worst_trapezoid = max(worst_trapezoid, abs(numeric - analytic) / analytic)
                        trapezoids += 1
    _, igse = _models()
    trapezoid = igse.loss_density(waveform=Waveform.trapezoidal(peak=0.1, frequency=1e5, duty_rise=0.5, duty_flat=0.0))
    triangle = igse.loss_density(waveform=Waveform.triangular(peak=0.1, frequency=1e5, duty=0.5))
    degenerate = abs(trapezoid.value("loss_density").value - triangle.value("loss_density").value) / triangle.value(
        "loss_density").value
    equation_5 = igse_trapezoidal_closed_form(SYNTHETIC_K, SYNTHETIC_ALPHA, SYNTHETIC_BETA, peak=0.1, frequency=1e5,
                                              duty_rise=0.5)
    equation_4 = igse_triangular_closed_form(SYNTHETIC_K, SYNTHETIC_ALPHA, SYNTHETIC_BETA, peak=0.1, frequency=1e5,
                                             duty=0.5)
    closed_forms_agree = abs(equation_5 - equation_4) / equation_4
    passed = worst_triangle < 1e-12 and worst_trapezoid < 1e-12 and degenerate < 1e-12 and closed_forms_agree < 1e-12
    return CheckResult(passed=passed,
                       metrics={"max_rel_error_triangular": worst_triangle, "max_rel_error_trapezoidal": worst_trapezoid,
                                "degenerate_trapezoid_rel_error": degenerate,
                                "closed_forms_rel_difference_at_half_duty": closed_forms_agree, "tolerance": 1e-12,
                                "n_triangular_points": triangles, "n_trapezoidal_points": trapezoids,
                                "trapezoid_duty_rise": list(TRAPEZOID_RISES)})


@check(
    id="igse.sampled-converges",
    capability=IGSE_CAPABILITY,
    level="L2", kind="convergence",
    description=("The sampled-waveform iGSE path converges to the analytic sinusoidal result as the number of samples "
                 "per period grows (N = 64, 256, 1024, 4096), with the same relative error at every (f, Bpk)."),
    method=("Sample Waveform.sinusoidal with N points spanning one period at (1 kHz, 1 mT), (100 kHz, 0.1 T) and "
            "(1 MHz, 0.3 T), wrap as Waveform.sampled, evaluate IGSEModel and compare with the Steinmetz closed form; "
            "require monotonically decreasing error, < 1e-4 at N=4096 and a relative error that is independent of "
            "(f, Bpk) to 1e-9 (scale invariance: the sampled path scales exactly as f^alpha Bpk^beta)."),
    range=SAMPLED_RANGE,
)
def igse_sampled_converges() -> CheckResult:
    steinmetz, igse = _models()
    counts = (64, 256, 1024, 4096)
    errors: dict[tuple[float, float], dict[int, float]] = {}
    warned: dict[int, bool] = {}
    for f, b in SAMPLED_POINTS:
        reference = steinmetz.closed_form(f, b)
        errors[(f, b)] = {}
        for n in counts:
            _, samples = Waveform.sinusoidal(peak=b, frequency=f).samples(n)
            output = igse.loss_density(waveform=Waveform.sampled(f, samples))
            errors[(f, b)][n] = abs(output.value("loss_density").value - reference) / reference
            warned[n] = any(w.code == "sampling_sensitivity" for w in output.warnings)
            if output.diagnostics.get("n_samples") != n:
                return CheckResult(passed=False, metrics={"n_samples_diagnostic": output.diagnostics.get("n_samples")},
                                   notes=[f"diagnostics n_samples mismatch for N={n} at f={f:g} Hz, Bpk={b:g} T"])
    centre = errors[(1e5, 0.1)]
    ordered = [centre[n] for n in counts]
    monotone = all(later < earlier for earlier, later in zip(ordered, ordered[1:]))
    orders = [math.log(earlier / later, 4.0) for earlier, later in zip(ordered, ordered[1:])]
    # scale invariance: the relative errors at the three points agree to rounding (the relative error is a ratio of
    # order-one quantities, so its spread is bounded by a few eps in absolute terms, independently of its size)
    spread = max(abs(errors[point][n] - centre[n]) for point in errors for n in counts)
    passed = monotone and ordered[-1] < 1e-4 and warned[64] and not warned[256] and spread < 1e-13
    return CheckResult(passed=passed, metrics={
        "rel_error_n64": centre[64], "rel_error_n256": centre[256], "rel_error_n1024": centre[1024],
        "rel_error_n4096": centre[4096], "monotone_decrease": monotone, "tolerance_n4096": 1e-4,
        "observed_convergence_orders": orders, "sampling_sensitivity_warning_at_n64": warned[64],
        "scale_invariance_max_abs_spread_of_rel_error": spread, "scale_invariance_tolerance": 1e-13,
        "points": [list(point) for point in SAMPLED_POINTS]},
        notes=["error measured against the Steinmetz closed form; the relative error is independent of f and Bpk "
               "because the sampled path scales exactly as f^alpha Bpk^beta, so one convergence curve covers the range",
               "second-order convergence holds for alpha >= 1; for alpha < 1 the integrand |dB/dt|^alpha has a cusp at "
               "dB/dt = 0 and the order is irregular (not claimed here)"])


@check(
    id="igse.ki-approximation-agreement",
    capability=IGSE_CAPABILITY,
    level="L2", kind="published_reference",
    description=("The quadrature-based k_i agrees with Venkatachalam's published closed-form approximation "
                 "k_i ~= k / (2^(beta+1) pi^(alpha-1) (0.2761 + 1.7061/(alpha+1.354))) within 1 %."),
    method=("Sweep alpha in [1.0, 2.5] (16 values) and beta in [2, 3] (5 values); compare igse_coefficient (quad) with "
            "igse_coefficient_venkatachalam_approximation; also compare the quadrature integral with the exact "
            "Gamma-function expression 2 sqrt(pi) Gamma((alpha+1)/2) / Gamma(alpha/2+1)."),
    reference=IGSE_SOURCE,
)
def igse_ki_approximation_agreement() -> CheckResult:
    worst_approx = 0.0
    worst_exact = 0.0
    for alpha in np.linspace(1.0, 2.5, 16):
        a = float(alpha)
        exact = abs_cos_power_integral_closed_form(a)
        worst_exact = max(worst_exact, abs(abs_cos_power_integral(a) - exact) / exact)
        for beta in (2.0, 2.25, 2.5, 2.75, 3.0):
            k_i = igse_coefficient(SYNTHETIC_K, a, beta)
            approx = igse_coefficient_venkatachalam_approximation(SYNTHETIC_K, a, beta)
            worst_approx = max(worst_approx, abs(k_i - approx) / k_i)
    return CheckResult(passed=worst_approx < 0.01 and worst_exact < 1e-12,
                       metrics={"max_rel_deviation_from_approximation": worst_approx, "approximation_tolerance": 0.01,
                                "max_rel_error_vs_gamma_closed_form": worst_exact, "exact_tolerance": 1e-12,
                                "alpha_range": [1.0, 2.5], "beta_range": [2.0, 3.0]})


def _loop_decomposed_reference(alpha: float, beta: float, frequency: float) -> float:
    """Loop-decomposed iGSE (Venkatachalam et al. 2002) for the minor-loop waveform of the check below.

    Waveform: B = -0.1 -> 0.1 (x 0..0.3) -> 0.0 (0.3..0.5) -> 0.08 (0.5..0.7) -> -0.1 (0.7..1). The fall from 0.1 to
    0.0 passes 0.08 T after a fraction 0.04 of the period, so the minor loop is 0.08 -> 0.0 -> 0.08 (DeltaB_minor =
    0.08 T, fractions 0.16 and 0.2) and the major loop is -0.1 -> 0.1 -> 0.08 -> -0.1 (DeltaB_major = 0.2 T,
    fractions 0.3, 0.04, 0.3). Per segment (1/T) Integral |dB/dt|^alpha dt = |dB|^alpha x^(1-alpha) f^alpha.
    """
    k_i = igse_coefficient(SYNTHETIC_K, alpha, beta)

    def segment(db: float, x: float) -> float:
        return abs(db) ** alpha * x ** (1.0 - alpha) * frequency ** alpha

    major = segment(0.2, 0.3) + segment(0.02, 0.04) + segment(0.18, 0.3)
    minor = segment(0.08, 0.16) + segment(0.08, 0.2)
    return k_i * (0.2 ** (beta - alpha) * major + 0.08 ** (beta - alpha) * minor)


@check(
    id="igse.minor-loop-detection",
    capability=IGSE_CAPABILITY,
    level="L2", kind="physical_consistency",
    description=("Waveforms with minor loops are detected (reversal count > 2) and rejected under policy 'reject' or "
                 "flagged with 'minor_loops_ignored' under policy 'warn'; the flagged value lies above the "
                 "loop-decomposed iGSE result for beta > alpha and below it for beta < alpha, as the warning states; "
                 "single-loop waveforms are not flagged."),
    method=("Build a piecewise-linear waveform with an intermediate dip (4 reversals) and its 513-sample version; check "
            "count_reversals/has_minor_loops, OutOfDomainError under reject and the warning under warn; compare the "
            "warn value with a hand-derived loop-decomposed reference for (alpha, beta) = (1.5, 2.5) and (2.0, 1.5) "
            "and require the sign stated in the warning; confirm sinusoidal, triangular, flat-topped trapezoidal and "
            "flat-topped PWL waveforms report 2 reversals."),
    range=OperatingRange(excitation=["piecewise_linear", "sampled", "triangular", "trapezoidal", "sinusoidal"]),
    reference=IGSE_SOURCE,
)
def igse_minor_loop_detection() -> CheckResult:
    _, igse = _models()
    f = 1e5
    minor = Waveform.piecewise_linear(f, [(0.0, -0.1), (0.3, 0.1), (0.5, 0.0), (0.7, 0.08), (1.0, -0.1)])
    _, samples = minor.samples(513)
    minor_sampled = Waveform.sampled(f, samples)
    notes: list[str] = []
    metrics: dict[str, Any] = {
        "pwl_reversals": minor.count_reversals(), "sampled_reversals": minor_sampled.count_reversals(),
    }
    ok = minor.count_reversals() == 4 and minor.has_minor_loops
    ok = ok and minor_sampled.count_reversals() == 4 and minor_sampled.has_minor_loops
    for name, waveform in (("pwl", minor), ("sampled", minor_sampled)):
        problem = _expect(OutOfDomainError, lambda w=waveform: igse.loss_density(waveform=w))
        if problem:
            ok = False
            notes.append(f"{name} reject policy: {problem}")
        output = igse.loss_density(waveform=waveform, policy="warn")
        codes = [w.code for w in output.warnings]
        metrics[f"{name}_warn_codes"] = codes
        if "minor_loops_ignored" not in codes or not output.diagnostics.get("has_minor_loops") \
                or not output.diagnostics.get("out_of_domain"):
            ok = False
            notes.append(f"{name} warn policy did not flag minor loops: {codes}")
    # direction of the undecomposed value relative to the loop-decomposed reference
    for alpha, beta, expected_word in ((SYNTHETIC_ALPHA, SYNTHETIC_BETA, "overestimates"), (2.0, 1.5, "underestimates")):
        _, model = _models(alpha, beta)
        output = model.loss_density(waveform=minor, policy="warn")
        undecomposed = output.value("loss_density").value
        decomposed = _loop_decomposed_reference(alpha, beta, f)
        ratio = undecomposed / decomposed
        metrics[f"undecomposed_over_decomposed_alpha{alpha:g}_beta{beta:g}"] = ratio
        message = next((w.message for w in output.warnings if w.code == "minor_loops_ignored"), "")
        direction_ok = ratio > 1.0 if beta > alpha else ratio < 1.0
        if not direction_ok or expected_word not in message:
            ok = False
            notes.append(f"alpha={alpha:g}, beta={beta:g}: ratio {ratio:.6f}, warning says: {message[:160]}")
    single = {
        "sinusoidal": Waveform.sinusoidal(0.1, f),
        "triangular": Waveform.triangular(0.1, f, duty=0.3),
        "trapezoidal": Waveform.trapezoidal(0.1, f, duty_rise=0.2, duty_flat=0.3),
        "pwl_flat_top": Waveform.piecewise_linear(f, [(0.0, -0.1), (0.4, 0.1), (0.6, 0.1), (1.0, -0.1)]),
    }
    for name, waveform in single.items():
        count = waveform.count_reversals()
        metrics[f"{name}_reversals"] = count
        if count != 2 or waveform.has_minor_loops:
            ok = False
            notes.append(f"{name} reported {count} reversals")
        output = igse.loss_density(waveform=waveform)
        if any(w.code == "minor_loops_ignored" for w in output.warnings):
            ok = False
            notes.append(f"{name} wrongly flagged minor loops")
    notes.append("the loop-decomposed reference is a hand derivation for this one waveform (documented in the check "
                 "source); the model itself never decomposes loops")
    return CheckResult(passed=ok, metrics=metrics, notes=notes)


# --------------------------------------------------------------------------------------------------
# Steinmetz
# --------------------------------------------------------------------------------------------------
@check(
    id="steinmetz.scaling-laws",
    capability=STEINMETZ_CAPABILITY,
    level="L2", kind="physical_consistency",
    description="Doubling the frequency scales the loss density by 2^alpha and doubling Bpk by 2^beta.",
    method=("For the synthetic parameters and swept (alpha, beta), evaluate SteinmetzModel at (f, B), (2f, B) and "
            "(f, 2B) on a grid and compare the ratios with 2^alpha and 2^beta (relative tolerance 1e-12); also check "
            "the vectorised helper loss_density_array against the scalar path."),
    range=SINUSOIDAL_RANGE,
)
def steinmetz_scaling_laws() -> CheckResult:
    worst = 0.0
    worst_vector = 0.0
    points = 0
    for alpha in ALPHA_SWEEP:
        for beta in BETA_SWEEP:
            steinmetz, _ = _models(alpha, beta)
            for f in (1e3, 1e4, 1e5):
                for b in (1e-3, 1e-2, 0.1):
                    base = steinmetz.loss_density(frequency=f, flux_density_peak=b).value("loss_density").value
                    double_f = steinmetz.loss_density(frequency=2 * f, flux_density_peak=b).value("loss_density").value
                    double_b = steinmetz.loss_density(frequency=f, flux_density_peak=2 * b).value("loss_density").value
                    worst = max(worst, abs(double_f / base - 2.0 ** alpha) / 2.0 ** alpha,
                                abs(double_b / base - 2.0 ** beta) / 2.0 ** beta)
                    points += 1
            grid = steinmetz.loss_density_array(np.array(FREQUENCIES)[:, None], np.array(PEAKS)[None, :])
            for i, f in enumerate(FREQUENCIES):
                for j, b in enumerate(PEAKS):
                    scalar = steinmetz.loss_density(frequency=f, flux_density_peak=b).value("loss_density").value
                    worst_vector = max(worst_vector, abs(grid[i, j] - scalar) / scalar)
    return CheckResult(passed=worst < 1e-12 and worst_vector < 1e-12,
                       metrics={"max_rel_error_scaling": worst, "max_rel_error_vectorised": worst_vector,
                                "tolerance": 1e-12, "n_points": points})


@check(
    id="steinmetz.zero-excitation",
    capability=STEINMETZ_CAPABILITY,
    level="L2", kind="limiting_case",
    description="Zero peak flux density yields exactly 0 W/m^3 with an informational 'zero_excitation' warning.",
    method=("Evaluate SteinmetzModel and IGSEModel with flux_density_peak=0 at several frequencies and the vectorised "
            "helper with zero entries; require exact zeros, the info warning and no exception."),
)
def steinmetz_zero_excitation() -> CheckResult:
    steinmetz, igse = _models(frequency_range=(1e3, 1e6), flux_density_range=(1e-3, 0.3))
    ok = True
    notes: list[str] = []
    for model in (steinmetz, igse):
        for f in FREQUENCIES:
            output = model.loss_density(frequency=f, flux_density_peak=0.0)
            value = output.value("loss_density")
            info = [w for w in output.warnings if w.code == "zero_excitation" and w.severity == "info"]
            if value.value != 0.0 or value.unit != "W/m^3" or not info or not output.diagnostics.get("zero_excitation"):
                ok = False
                notes.append(f"{model.identity.id} at {f:g} Hz: value={value.value!r} warnings="
                             f"{[w.code for w in output.warnings]}")
    grid = steinmetz.loss_density_array([1e3, 1e5], [0.0, 0.0])
    if not np.all(grid == 0.0):
        ok = False
        notes.append(f"vectorised helper returned {grid.tolist()} for zero peaks")
    return CheckResult(passed=ok, metrics={"n_cases": 2 * len(FREQUENCIES) + 1}, notes=notes)


@check(
    id="steinmetz.invalid-inputs-rejected",
    capability=STEINMETZ_CAPABILITY,
    level="L1", kind="invalid_input_rejection",
    description=("Negative, zero-frequency, non-finite, non-real, wrongly typed, physically inadmissible, overflowing and "
                 "inconsistent inputs raise InvalidInputError (never clipped, coerced or evaluated) for the models and "
                 "the waveform constructors."),
    method="Call constructors and loss_density with a table of invalid inputs and require InvalidInputError for each.",
)
def steinmetz_invalid_inputs_rejected() -> CheckResult:
    steinmetz, igse = _models()
    nan, inf = float("nan"), float("inf")
    sine = (0.1 * np.sin(2 * np.pi * np.linspace(0.0, 1.0, 64))).tolist()
    clashing_source = SourceReference(id="publication.steinmetz-1892", kind="publication", reference="SOMETHING ELSE")
    cases: dict[str, Callable[[], Any]] = {
        "k=0": lambda: SteinmetzModel(0.0, 1.5, 2.5),
        "k<0": lambda: SteinmetzModel(-1.0, 1.5, 2.5),
        "alpha=0": lambda: SteinmetzModel(1.0, 0.0, 2.5),
        "beta=nan": lambda: SteinmetzModel(1.0, 1.5, nan),
        "k=str": lambda: SteinmetzModel("1.0", 1.5, 2.5),
        "alpha=bool": lambda: SteinmetzModel(1.0, True, 2.5),
        "fit_temperature<=0": lambda: SteinmetzModel(1.0, 1.5, 2.5, fit_temperature=0.0),
        "range min>max": lambda: SteinmetzModel(1.0, 1.5, 2.5, frequency_range=(1e6, 1e3)),
        "range wrong unit": lambda: SteinmetzModel(1.0, 1.5, 2.5, frequency_range=ValidityRange(min=1, max=2, unit="T")),
        "frequency range min 0 Hz": lambda: SteinmetzModel(1.0, 1.5, 2.5, frequency_range=(0.0, 1e6)),
        "flux range negative": lambda: SteinmetzModel(1.0, 1.5, 2.5, flux_density_range=(-1.0, 0.3)),
        "temperature range <= 0 K": lambda: SteinmetzModel(1.0, 1.5, 2.5, temperature_range=(-10.0, 400.0)),
        "temperature range degC below 0 K": lambda: SteinmetzModel(
            1.0, 1.5, 2.5, temperature_range=ValidityRange(min=-300.0, max=100.0, unit="degC")),
        "synthetic=str": lambda: SteinmetzModel(1.0, 1.5, 2.5, synthetic="False"),
        "source id clash": lambda: SteinmetzModel(1.0, 1.5, 2.5, sources=[clashing_source]),
        "f<0": lambda: steinmetz.loss_density(frequency=-1e5, flux_density_peak=0.1),
        "f=0": lambda: steinmetz.loss_density(frequency=0.0, flux_density_peak=0.1),
        "f=nan": lambda: steinmetz.loss_density(frequency=nan, flux_density_peak=0.1),
        "f=inf": lambda: steinmetz.loss_density(frequency=inf, flux_density_peak=0.1),
        "f=bool": lambda: steinmetz.loss_density(frequency=True, flux_density_peak=0.1),
        "f=str": lambda: steinmetz.loss_density(frequency="1e5", flux_density_peak=0.1),
        "B<0": lambda: steinmetz.loss_density(frequency=1e5, flux_density_peak=-0.1),
        "B=inf": lambda: steinmetz.loss_density(frequency=1e5, flux_density_peak=inf),
        "B=nan": lambda: steinmetz.loss_density(frequency=1e5, flux_density_peak=nan),
        "T<=0": lambda: steinmetz.loss_density(frequency=1e5, flux_density_peak=0.1, temperature=-5.0),
        "policy=clip": lambda: steinmetz.loss_density(frequency=1e5, flux_density_peak=0.1, policy="clip"),
        "unexpected kwarg": lambda: steinmetz.loss_density(frequency=1e5, flux_density_peak=0.1, duty=0.5),
        "array f<0": lambda: steinmetz.loss_density_array([-1e5, 1e5], [0.1, 0.1]),
        "array B nan": lambda: steinmetz.loss_density_array([1e5], [nan]),
        "array str": lambda: steinmetz.loss_density_array(["1e5"], [0.1]),
        "igse missing f/B": lambda: igse.loss_density(flux_density_peak=0.1),
        "igse waveform type": lambda: igse.loss_density(waveform="triangular"),
        "igse f mismatch": lambda: igse.loss_density(frequency=2e5, waveform=Waveform.triangular(0.1, 1e5)),
        "igse B mismatch": lambda: igse.loss_density(flux_density_peak=0.2, waveform=Waveform.triangular(0.1, 1e5)),
        "igse reversal_tolerance<0": lambda: igse.loss_density(waveform=Waveform.triangular(0.1, 1e5),
                                                               reversal_tolerance=-1.0),
        "igse reversal_tolerance suppresses everything": lambda: igse.loss_density(
            waveform=Waveform.triangular(0.1, 1e5), reversal_tolerance=2.0),
        "waveform peak=0": lambda: Waveform.sinusoidal(0.0, 1e5),
        "waveform f=0": lambda: Waveform.sinusoidal(0.1, 0.0),
        "waveform duty=1": lambda: Waveform.triangular(0.1, 1e5, duty=1.0),
        "waveform duty=nan": lambda: Waveform.triangular(0.1, 1e5, duty=nan),
        "trapezoid fractions": lambda: Waveform.trapezoidal(0.1, 1e5, duty_rise=0.3, duty_flat=0.3),
        "pwl not closed": lambda: Waveform.piecewise_linear(1e5, [(0.0, -0.1), (0.5, 0.1), (1.0, 0.0)]),
        "pwl not from 0": lambda: Waveform.piecewise_linear(1e5, [(0.1, -0.1), (0.5, 0.1), (1.0, -0.1)]),
        "pwl not increasing": lambda: Waveform.piecewise_linear(1e5, [(0.0, -0.1), (0.5, 0.1), (0.5, 0.0), (1.0, -0.1)]),
        "pwl constant": lambda: Waveform.piecewise_linear(1e5, [(0.0, 0.1), (1.0, 0.1)]),
        "pwl excursion overflow": lambda: Waveform.piecewise_linear(1e5, [(0.0, -1e308), (0.5, 1e308), (1.0, -1e308)]),
        "pwl zero-duration segment": lambda: Waveform.piecewise_linear(1e5, [(0.0, 0.0), (5e-324, 0.1), (1.0, 0.0)]),
        "sampled too short": lambda: Waveform.sampled(1e5, [0, 1, 0, -1, 0, 1, 0]),
        "sampled nan": lambda: Waveform.sampled(1e5, [0, 1, 0, -1, 0, 1, 0, nan, 0]),
        "sampled not closed": lambda: Waveform.sampled(1e5, [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        "sampled 2-D": lambda: Waveform.sampled(1e5, [[0, 1], [0, -1], [0, 1], [0, -1], [0, 1]]),
        "sampled excursion overflow": lambda: Waveform.sampled(1e5, [-1e308, 1e308] * 4 + [-1e308]),
        "sampled endpoint flag not bool": lambda: Waveform.sampled(1e5, sine, endpoint_included="no"),
        "direct ctor f<0": lambda: Waveform("sinusoidal", -1e5, peak=0.1, dc_component=0.0, maximum=0.1, minimum=-0.1),
        "direct ctor peak inconsistent": lambda: Waveform("sinusoidal", 1e5, peak=0.2, dc_component=0.0, maximum=0.1,
                                                          minimum=-0.1),
        "direct ctor dc outside": lambda: Waveform("sinusoidal", 1e5, peak=0.1, dc_component=5.0, maximum=0.1,
                                                   minimum=-0.1),
    }
    failures = {name: problem for name, action in cases.items() if (problem := _expect(InvalidInputError, action))}
    return CheckResult(passed=not failures, metrics={"n_cases": len(cases), "n_failures": len(failures)},
                       notes=[f"{name}: {problem}" for name, problem in failures.items()])


@check(
    id="steinmetz.overflow-detected",
    capability=STEINMETZ_CAPABILITY,
    level="L2", kind="limiting_case",
    description="Inputs that overflow double precision raise NumericalError instead of returning inf or nan.",
    method=("Evaluate the scalar Steinmetz path, the vectorised helper, the iGSE sinusoidal, piecewise-linear (including "
            "a resolvable but near-zero segment fraction) and sampled paths and the closed-form references with extreme "
            "inputs such as 1e300 and require NumericalError with no inf/nan escaping."),
)
def steinmetz_overflow_detected() -> CheckResult:
    steinmetz = SteinmetzModel(SYNTHETIC_K, SYNTHETIC_ALPHA, SYNTHETIC_BETA, sources=[SYNTHETIC_SOURCE], synthetic=True)
    igse = IGSEModel.from_steinmetz(steinmetz)
    steep = IGSEModel(SYNTHETIC_K, 2.5, 3.0, sources=[SYNTHETIC_SOURCE], synthetic=True)  # x^(1-alpha) = x^-1.5
    _, huge_sine = Waveform.sinusoidal(peak=1e200, frequency=1e150).samples(64)
    near_zero_segment = Waveform.piecewise_linear(1e5, [(0.0, 0.0), (1e-300, 0.1), (1.0, 0.0)])
    cases: dict[str, Callable[[], Any]] = {
        "steinmetz 1e300/1e300": lambda: steinmetz.loss_density(frequency=1e300, flux_density_peak=1e300),
        "steinmetz 1e200/1e100": lambda: steinmetz.loss_density(frequency=1e200, flux_density_peak=1e100),
        "steinmetz closed_form": lambda: steinmetz.closed_form(1e300, 1e300),
        "steinmetz array": lambda: steinmetz.loss_density_array([1e5, 1e300], [0.1, 1e300]),
        "igse sinusoid": lambda: igse.loss_density(frequency=1e300, flux_density_peak=1e300),
        "igse triangular": lambda: igse.loss_density(waveform=Waveform.triangular(peak=1e200, frequency=1e150)),
        "igse near-zero segment fraction": lambda: steep.loss_density(waveform=near_zero_segment),
        "igse sampled": lambda: igse.loss_density(waveform=Waveform.sampled(1e150, huge_sine)),
        "igse k_i overflow": lambda: igse_coefficient(1.0, 1.5, 5000.0),
        "triangular closed form": lambda: igse_triangular_closed_form(1.0, 1.5, 2.5, peak=1e200, frequency=1e150,
                                                                      duty=0.3),
        "trapezoidal closed form": lambda: igse_trapezoidal_closed_form(1.0, 1.5, 2.5, peak=1e200, frequency=1e150,
                                                                        duty_rise=0.2),
    }
    failures = {name: problem for name, action in cases.items() if (problem := _expect(NumericalError, action))}
    return CheckResult(passed=not failures, metrics={"n_cases": len(cases), "n_failures": len(failures)},
                       notes=[f"{name}: {problem}" for name, problem in failures.items()])


# --------------------------------------------------------------------------------------------------
# waveform descriptors
# --------------------------------------------------------------------------------------------------
@check(
    id="waveform.descriptors-consistent",
    capability=WAVEFORM_CAPABILITY,
    level="L1", kind="software",
    description=("Waveform peak, peak-to-peak, DC component, segments and samples are mutually consistent across the "
                 "analytic, piecewise-linear and sampled representations, including the [0, T) sampling convention."),
    method=("Compare dc_component/peak of sampled and PWL versions of analytic waveforms with their constructor values, "
            "check that segments sum to one period and to zero net flux change, that samples close, and that the same "
            "waveform given as [0, T) samples with endpoint_included=False reproduces the inclusive representation."),
)
def waveform_descriptors_consistent() -> CheckResult:
    f = 1e5
    notes: list[str] = []
    ok = True
    analytic = {
        "sinusoidal": Waveform.sinusoidal(0.2, f, dc=0.05),
        "triangular": Waveform.triangular(0.2, f, duty=0.3, dc=-0.02),
        "trapezoidal": Waveform.trapezoidal(0.2, f, duty_rise=0.15, duty_flat=0.35, dc=0.01),
    }
    worst_dc = 0.0
    n_samples = 4001  # 4000 intervals: every vertex / extremum of the waveforms above lies on the sampling grid
    for name, waveform in analytic.items():
        t, b = waveform.samples(n_samples)
        sampled = Waveform.sampled(f, b)
        half_open = Waveform.sampled(f, b[:-1], endpoint_included=False)
        worst_dc = max(worst_dc, abs(sampled.dc_component - waveform.dc_component))
        if abs(sampled.peak - waveform.peak) > 1e-12 * waveform.peak:
            ok = False
            notes.append(f"{name}: sampled peak {sampled.peak} != {waveform.peak}")
        if abs(t[-1] - waveform.period) > 1e-15 or abs(b[0] - b[-1]) > 1e-12 * waveform.peak_to_peak:
            ok = False
            notes.append(f"{name}: samples do not span/close one period")
        # the inclusive form ends on B(T) as evaluated (e.g. sin(2 pi) ~ -2.4e-16), the half-open form on B(0) exactly
        if (half_open.n_samples != sampled.n_samples
                or abs(half_open.dc_component - sampled.dc_component) > 1e-12 * waveform.peak_to_peak
                or not np.allclose(half_open.samples()[1], sampled.samples()[1], rtol=0.0,
                                   atol=1e-12 * waveform.peak_to_peak)):
            ok = False
            notes.append(f"{name}: [0, T) samples with endpoint_included=False do not reproduce the inclusive form")
        if waveform.is_piecewise_linear:
            segments = waveform.segments()
            total_dt = sum(dt for dt, _ in segments)
            total_db = sum(db for _, db in segments)
            if abs(total_dt - waveform.period) > 1e-12 * waveform.period or abs(total_db) > 1e-12:
                ok = False
                notes.append(f"{name}: segments sum dt={total_dt} dB={total_db}")
    if worst_dc > 1e-12:
        ok = False
        notes.append(f"sampled dc component deviates by {worst_dc} T")
    pwl = Waveform.piecewise_linear(f, [(0.0, -0.1), (0.25, 0.3), (0.5, 0.3), (1.0, -0.1)])
    expected_dc = 0.25 * 0.1 + 0.25 * 0.3 + 0.5 * 0.1  # exact trapezoid areas of the three segments
    if abs(pwl.dc_component - expected_dc) > 1e-15 or pwl.peak != 0.2 or pwl.peak_to_peak != 0.4:
        ok = False
        notes.append(f"pwl descriptors dc={pwl.dc_component} peak={pwl.peak} pp={pwl.peak_to_peak}")
    return CheckResult(passed=ok, metrics={"max_sampled_dc_deviation_T": worst_dc, "n_samples": n_samples,
                                           "dc_tolerance_T": 1e-12, "peak_rel_tolerance": 1e-12}, notes=notes)


ALL_CHECKS = [
    igse_sinusoid_equals_steinmetz,
    igse_triangular_closed_form_check,
    igse_sampled_converges,
    igse_ki_approximation_agreement,
    igse_minor_loop_detection,
    steinmetz_scaling_laws,
    steinmetz_zero_excitation,
    steinmetz_invalid_inputs_rejected,
    steinmetz_overflow_detected,
    waveform_descriptors_consistent,
]
