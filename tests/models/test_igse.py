"""Tests for aipe_magnetics.models.empirical.igse and the empirical verification checks."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from aipe_magnetics.common.errors import InvalidInputError, NumericalError, OutOfDomainError
from aipe_magnetics.common.results import SourceReference
from aipe_magnetics.models.contract import CoreLossModel, ValidityRange
from aipe_magnetics.models.empirical import MODELS, MODEL_IDS
from aipe_magnetics.models.empirical.igse import (
    DC_BIAS_RELATIVE_THRESHOLD,
    IGSE_MODEL_ID,
    IGSE_SOURCE,
    SAMPLING_WARNING_THRESHOLD,
    IGSEModel,
    _abs_cos_power_integral_cached,
    _igse_coefficient_cached,
    abs_cos_power_integral,
    abs_cos_power_integral_closed_form,
    igse_coefficient,
    igse_coefficient_venkatachalam_approximation,
    igse_trapezoidal_closed_form,
    igse_triangular_closed_form,
)
from aipe_magnetics.models.empirical.steinmetz import STEINMETZ_SOURCE, SteinmetzModel
from aipe_magnetics.models.waveforms import DEFAULT_REVERSAL_TOLERANCE, Waveform
from aipe_magnetics.verification import suite
from aipe_magnetics.verification.checks import check_empirical

# SYNTHETIC parameters: k=1 W/m^3 (f in Hz, B in T), alpha=1.5, beta=2.5 -> Pv(100 kHz, 0.1 T) = 100 kW/m^3.
K, ALPHA, BETA = 1.0, 1.5, 2.5
F = 1e5
NAN, INF = float("nan"), float("inf")
SYNTHETIC_SOURCE = SourceReference(id="internal.test-synthetic", kind="internal",
                                   reference="synthetic parameters for unit tests (not a material fit)")
MINOR_LOOP_POINTS = [(0.0, -0.1), (0.3, 0.1), (0.5, 0.0), (0.7, 0.08), (1.0, -0.1)]


def make_models(**kwargs) -> tuple[SteinmetzModel, IGSEModel]:
    defaults = dict(material_id="synthetic.test-material", fit_temperature=373.15, frequency_range=(1e3, 1e6),
                    flux_density_range=(1e-3, 0.3), sources=[SYNTHETIC_SOURCE], synthetic=True)
    defaults.update(kwargs)
    steinmetz = SteinmetzModel(K, ALPHA, BETA, **defaults)
    return steinmetz, IGSEModel.from_steinmetz(steinmetz)


# --------------------------------------------------------------------------------------------------
# k_i
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.3, 2.5, 3.0])
def test_quadrature_integral_matches_gamma_closed_form(alpha):
    exact = abs_cos_power_integral_closed_form(alpha)
    assert abs_cos_power_integral(alpha) == pytest.approx(exact, rel=1e-12)
    if alpha == 1.0:
        assert exact == pytest.approx(4.0)
    if alpha == 2.0:
        assert exact == pytest.approx(math.pi)


def test_gamma_closed_form_uses_lgamma_and_does_not_overflow():
    for alpha in (0.7, 1.5, 3.0, 50.0, 300.0):
        direct = 2.0 * math.sqrt(math.pi) * math.gamma((alpha + 1.0) / 2.0) / math.gamma(alpha / 2.0 + 1.0)
        assert abs_cos_power_integral_closed_form(alpha) == pytest.approx(direct, rel=1e-13)
    large = abs_cos_power_integral_closed_form(1000.0)  # math.gamma overflows here
    assert math.isfinite(large) and 0.0 < large < 1.0
    assert abs_cos_power_integral(1000.0) == pytest.approx(large, rel=1e-11)
    with pytest.raises(InvalidInputError):
        abs_cos_power_integral_closed_form(0.0)


def test_ki_definition_and_published_approximation():
    k_i = igse_coefficient(K, ALPHA, BETA)
    expected = K / ((2 * math.pi) ** (ALPHA - 1) * 2 ** (BETA - ALPHA) * abs_cos_power_integral_closed_form(ALPHA))
    assert k_i == pytest.approx(expected, rel=1e-12)
    for alpha in np.linspace(1.0, 2.5, 7):
        for beta in (2.0, 2.5, 3.0):
            quad_value = igse_coefficient(K, float(alpha), beta)
            approx = igse_coefficient_venkatachalam_approximation(K, float(alpha), beta)
            assert abs(quad_value - approx) / quad_value < 0.01
    with pytest.raises(InvalidInputError):
        igse_coefficient(0.0, ALPHA, BETA)
    with pytest.raises(InvalidInputError):
        igse_coefficient(K, -1.0, BETA)
    with pytest.raises(InvalidInputError):
        igse_coefficient_venkatachalam_approximation(K, NAN, BETA)
    with pytest.raises(NumericalError):
        igse_coefficient(K, ALPHA, 5000.0)


def test_ki_is_cached_per_parameter_triple():
    _igse_coefficient_cached.cache_clear()
    _abs_cos_power_integral_cached.cache_clear()
    first = igse_coefficient(2.0, 1.4, 2.6)
    second = igse_coefficient(2.0, 1.4, 2.6)
    assert first == second
    assert _igse_coefficient_cached.cache_info().hits >= 1
    assert IGSEModel(2.0, 1.4, 2.6).k_i == first


# --------------------------------------------------------------------------------------------------
# identity / domain
# --------------------------------------------------------------------------------------------------
def test_identity_domain_and_sources():
    steinmetz, igse = make_models()
    assert isinstance(igse, CoreLossModel) and not isinstance(igse, SteinmetzModel)
    assert igse.identity.id == IGSE_MODEL_ID == "aipe.magnetics.model.igse"
    assert igse.identity.family == "empirical" and igse.identity.version == "0.2.0"
    assert IGSEModel in MODELS and MODEL_IDS[IGSE_MODEL_ID] is IGSEModel
    assert igse.domain.excitation == ["sinusoidal", "triangular", "trapezoidal", "piecewise_linear", "sampled"]
    assert igse.domain.dc_bias == "ignored"
    assert "iGSE ignores DC bias" in " ".join(igse.domain.known_limitations)
    assert any("overestimates" in item and "beta > alpha" in item for item in igse.domain.known_limitations)
    assert any("exact integral of the piecewise-linear interpolant" in item for item in igse.domain.numerical_constraints)
    assert igse.domain.frequency == steinmetz.domain.frequency
    assert igse.domain.temperature == steinmetz.domain.temperature and igse.temperature_range_implied
    assert (igse.k, igse.alpha, igse.beta, igse.material_id, igse.synthetic) == (K, ALPHA, BETA,
                                                                                 "synthetic.test-material", True)
    assert [s.id for s in igse.sources] == [IGSE_SOURCE.id, STEINMETZ_SOURCE.id, SYNTHETIC_SOURCE.id]
    with pytest.raises(InvalidInputError):
        IGSEModel.from_steinmetz("not a model")
    json.dumps(igse.describe())


# --------------------------------------------------------------------------------------------------
# sinusoidal
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("alpha,beta", [(1.0, 2.0), (1.5, 2.5), (2.0, 3.0), (2.5, 2.2)])
@pytest.mark.parametrize("frequency,peak", [(1e3, 1e-3), (1e5, 0.1), (1e6, 0.3)])
def test_sinusoid_equals_steinmetz(alpha, beta, frequency, peak):
    steinmetz = SteinmetzModel(K, alpha, beta)
    igse = IGSEModel(K, alpha, beta)
    reference = steinmetz.loss_density(frequency=frequency, flux_density_peak=peak).value("loss_density").value
    implicit = igse.loss_density(frequency=frequency, flux_density_peak=peak)
    explicit = igse.loss_density(waveform=Waveform.sinusoidal(peak=peak, frequency=frequency))
    assert implicit.value("loss_density").value == pytest.approx(reference, rel=1e-9)
    assert explicit.value("loss_density").value == pytest.approx(reference, rel=1e-9)
    assert implicit.value("loss_density").unit == "W/m^3"
    assert implicit.diagnostics["waveform_kind"] == "sinusoidal" and implicit.diagnostics["delta_b"] == 2 * peak
    assert implicit.diagnostics["k_i"] == pytest.approx(igse_coefficient(K, alpha, beta))


def test_sinusoid_envelope_content():
    _, igse = make_models()
    output = igse.loss_density(frequency=F, flux_density_peak=0.1, temperature=373.15)
    assert output.model.id == IGSE_MODEL_ID and output.warnings == [] and output.domain.inside
    assert output.domain.checked["minor_loops"].startswith("unsupported")
    assert {a.id for a in output.assumptions} == {"steinmetz_parameters", "temperature_handling", "major_loop_delta_b"}
    codes = {item.code for item in output.limitations}
    assert {"dc_bias_ignored", "no_minor_loops", "no_relaxation", "sinusoidal_parameter_fit",
            "temperature_specific_parameters", "fit_range_only", "noisy_sampled_data", "synthetic_parameters"} <= codes
    assert any("iGSE ignores DC bias" in item.statement for item in output.limitations)
    assert output.diagnostics["has_minor_loops"] is False and output.diagnostics["n_reversals"] == 2
    assert output.diagnostics["reversal_tolerance"] == DEFAULT_REVERSAL_TOLERANCE
    assert output.diagnostics["dc_bias_threshold"] == DC_BIAS_RELATIVE_THRESHOLD
    json.dumps(output.model_dump(mode="json"))


# --------------------------------------------------------------------------------------------------
# piecewise-linear kinds
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("duty", [0.1, 0.25, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("alpha,beta", [(1.0, 2.0), (1.5, 2.5), (2.3, 3.0)])
def test_triangular_matches_closed_form(duty, alpha, beta):
    igse = IGSEModel(K, alpha, beta)
    waveform = Waveform.triangular(peak=0.1, frequency=F, duty=duty)
    output = igse.loss_density(waveform=waveform)
    expected = igse_triangular_closed_form(K, alpha, beta, peak=0.1, frequency=F, duty=duty)
    assert output.value("loss_density").value == pytest.approx(expected, rel=1e-12)
    assert output.diagnostics["n_segments"] == 2 and output.diagnostics["delta_b"] == pytest.approx(0.2)
    assert output.diagnostics["integration_method"].startswith("analytic per piecewise-linear")
    # duty D and 1-D give the same loss (equation (4) is symmetric)
    mirrored = igse.loss_density(waveform=Waveform.triangular(peak=0.1, frequency=F, duty=1 - duty))
    assert mirrored.value("loss_density").value == pytest.approx(expected, rel=1e-12)


def test_triangular_closed_form_formula_and_validation():
    k_i = igse_coefficient(K, ALPHA, BETA)
    duty = 0.3
    expected = k_i * (2 * 0.1) ** BETA * F ** ALPHA * (duty ** (1 - ALPHA) + (1 - duty) ** (1 - ALPHA))
    assert igse_triangular_closed_form(K, ALPHA, BETA, peak=0.1, frequency=F, duty=duty) == pytest.approx(expected)
    # symmetric triangle at alpha=1.5: factor D^(1-a)+(1-D)^(1-a) = 2 * 0.5^-0.5 = 2 sqrt(2)
    symmetric = igse_triangular_closed_form(K, ALPHA, BETA, peak=0.1, frequency=F, duty=0.5)
    assert symmetric == pytest.approx(k_i * 0.2 ** BETA * F ** ALPHA * 2 * math.sqrt(2.0))
    with pytest.raises(InvalidInputError):
        igse_triangular_closed_form(K, ALPHA, BETA, peak=0.1, frequency=F, duty=1.0)
    with pytest.raises(InvalidInputError):
        igse_triangular_closed_form(K, ALPHA, BETA, peak=0.0, frequency=F, duty=0.5)


def test_trapezoidal_matches_hand_derivation():
    _, igse = make_models()
    rise, flat = 0.2, 0.3
    waveform = Waveform.trapezoidal(peak=0.1, frequency=F, duty_rise=rise, duty_flat=flat)
    output = igse.loss_density(waveform=waveform)
    # two sloped segments with |dB| = 2 Bpk over the fraction `rise` each; flats contribute nothing
    expected = 2 * igse.k_i * (2 * 0.1) ** BETA * F ** ALPHA * rise ** (1 - ALPHA)
    assert output.value("loss_density").value == pytest.approx(expected, rel=1e-12)
    assert output.diagnostics["n_segments"] == 4
    # the trapezoid with duty_flat = 0 is the symmetric triangle
    degenerate = igse.loss_density(waveform=Waveform.trapezoidal(peak=0.1, frequency=F, duty_rise=0.5, duty_flat=0.0))
    assert degenerate.value("loss_density").value == pytest.approx(
        igse_triangular_closed_form(K, ALPHA, BETA, peak=0.1, frequency=F, duty=0.5), rel=1e-12)


def test_trapezoidal_closed_form_reference():
    _, igse = make_models()
    for rise in (0.1, 0.2, 0.35, 0.5):
        waveform = Waveform.trapezoidal(peak=0.1, frequency=F, duty_rise=rise, duty_flat=0.5 - rise)
        expected = igse_trapezoidal_closed_form(K, ALPHA, BETA, peak=0.1, frequency=F, duty_rise=rise)
        assert igse.loss_density(waveform=waveform).value("loss_density").value == pytest.approx(expected, rel=1e-12)
    assert igse_trapezoidal_closed_form(K, ALPHA, BETA, peak=0.1, frequency=F, duty_rise=0.5) == pytest.approx(
        igse_triangular_closed_form(K, ALPHA, BETA, peak=0.1, frequency=F, duty=0.5), rel=1e-14)
    for bad in (0.0, 0.6, -0.1, NAN):
        with pytest.raises(InvalidInputError):
            igse_trapezoidal_closed_form(K, ALPHA, BETA, peak=0.1, frequency=F, duty_rise=bad)
    with pytest.raises(NumericalError):
        igse_trapezoidal_closed_form(K, ALPHA, BETA, peak=1e200, frequency=1e150, duty_rise=0.2)


def test_piecewise_linear_general_waveform():
    _, igse = make_models()
    # rise over 0.4 (dB=+0.2), flat top over 0.2, fall over 0.4 (dB=-0.2); the flat top gives a +0.02 T DC component
    waveform = Waveform.piecewise_linear(F, [(0.0, -0.1), (0.4, 0.1), (0.6, 0.1), (1.0, -0.1)])
    assert waveform.dc_component == pytest.approx(0.02)
    output = igse.loss_density(waveform=waveform)
    expected = igse.k_i * 0.2 ** (BETA - ALPHA) * F ** ALPHA * (0.2 ** ALPHA * 0.4 ** (1 - ALPHA) * 2)
    assert output.value("loss_density").value == pytest.approx(expected, rel=1e-12)
    assert output.diagnostics["n_segments"] == 3
    assert [w.code for w in output.warnings] == ["dc_bias_ignored"]  # the offset is reported, never used
    # the same slopes with the flat time split evenly between top and bottom have no DC component
    centred = Waveform.piecewise_linear(F, [(0.0, -0.1), (0.4, 0.1), (0.5, 0.1), (0.9, -0.1), (1.0, -0.1)])
    assert centred.dc_component == pytest.approx(0.0, abs=1e-15)
    centred_output = igse.loss_density(waveform=centred)
    assert centred_output.warnings == []
    assert centred_output.value("loss_density").value == pytest.approx(expected, rel=1e-12)


def test_near_zero_segment_fraction_raises_typed_numerical_error():
    steep = IGSEModel(K, 2.5, 3.0)  # x^(1-alpha) = x^-1.5 overflows for x = 1e-300
    waveform = Waveform.piecewise_linear(F, [(0.0, 0.0), (1e-300, 0.1), (1.0, 0.0)])  # representable in time
    with pytest.raises(NumericalError):
        steep.loss_density(waveform=waveform)
    with pytest.raises(InvalidInputError):  # a segment whose duration underflows to 0 s never reaches the model
        Waveform.piecewise_linear(F, [(0.0, 0.0), (5e-324, 0.1), (1.0, 0.0)])


# --------------------------------------------------------------------------------------------------
# sampled
# --------------------------------------------------------------------------------------------------
def test_sampled_sinusoid_converges_and_warns_on_coarse_sampling():
    steinmetz, igse = make_models()
    reference = steinmetz.closed_form(F, 0.1)
    errors = []
    for n in (64, 256, 1024, 4096):
        _, samples = Waveform.sinusoidal(peak=0.1, frequency=F).samples(n)
        output = igse.loss_density(waveform=Waveform.sampled(F, samples))
        errors.append(abs(output.value("loss_density").value - reference) / reference)
        assert output.diagnostics["n_samples"] == n
        assert output.diagnostics["integration_method"].startswith("exact integral of the piecewise-linear interpolant")
        warned = any(w.code == "sampling_sensitivity" for w in output.warnings)
        assert warned == (n < SAMPLING_WARNING_THRESHOLD)
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 1e-4
    # second-order convergence: each 4x refinement reduces the error by ~16x
    for coarse, fine in zip(errors, errors[1:]):
        assert 12.0 < coarse / fine < 20.0


def test_sampled_path_scale_invariance_and_half_open_convention():
    igse = IGSEModel(K, ALPHA, BETA)  # no declared range: the sampled 1 mT sinusoid peaks at 0.99998 mT
    steinmetz = SteinmetzModel(K, ALPHA, BETA)
    errors = []
    for f, b in ((1e3, 1e-3), (1e5, 0.1), (1e6, 0.3)):
        _, samples = Waveform.sinusoidal(peak=b, frequency=f).samples(256)
        value = igse.loss_density(waveform=Waveform.sampled(f, samples)).value("loss_density").value
        errors.append(abs(value - steinmetz.closed_form(f, b)) / steinmetz.closed_form(f, b))
    assert max(errors) - min(errors) < 1e-13  # the relative error does not depend on f or Bpk (rounding only)
    # a declared range is applied to the *sampled* peak, which lies below the analytic crest between samples
    with pytest.raises(OutOfDomainError):
        make_models()[1].loss_density(waveform=Waveform.sampled(1e3, Waveform.sinusoidal(1e-3, 1e3).samples(256)[1]))
    # [0, T) data: endpoint_included=False reproduces the exact PWL result when the corners lie on the grid
    trapezoid = Waveform.trapezoidal(peak=0.1, frequency=F, duty_rise=0.2, duty_flat=0.3)
    exact = igse.loss_density(waveform=trapezoid).value("loss_density").value
    half_open = np.interp(np.arange(1000) / 1000, trapezoid.breakpoints[:, 0], trapezoid.breakpoints[:, 1])
    correct = igse.loss_density(waveform=Waveform.sampled(F, half_open, endpoint_included=False))
    assert correct.value("loss_density").value == pytest.approx(exact, rel=1e-12)
    assert correct.diagnostics["n_samples"] == 1001 and correct.diagnostics["waveform"]["parameters"]["endpoint_included"] is False
    # the same data mis-declared as [0, T] inclusive is biased by (n/(n-1))^(1-alpha) (flat ends: nothing else is lost)
    biased = igse.loss_density(waveform=Waveform.sampled(F, half_open)).value("loss_density").value
    assert biased / exact == pytest.approx((1000 / 999) ** (1 - ALPHA), rel=1e-9)


def test_sampled_path_converges_for_alpha_below_one_without_order_claim():
    igse = IGSEModel(K, 0.7, 2.5)
    reference = SteinmetzModel(K, 0.7, 2.5).closed_form(F, 0.1)
    errors = []
    for n in (64, 256, 1024, 4096):
        _, samples = Waveform.sinusoidal(peak=0.1, frequency=F).samples(n)
        value = igse.loss_density(waveform=Waveform.sampled(F, samples)).value("loss_density").value
        errors.append(abs(value - reference) / reference)
    assert errors == sorted(errors, reverse=True) and errors[-1] < 1e-6  # converges, but not at a fixed order


def test_sampled_triangle_with_corners_on_samples_is_exact():
    _, igse = make_models()
    _, samples = Waveform.triangular(peak=0.1, frequency=F, duty=0.5).samples(65)
    output = igse.loss_density(waveform=Waveform.sampled(F, samples))
    expected = igse_triangular_closed_form(K, ALPHA, BETA, peak=0.1, frequency=F, duty=0.5)
    assert output.value("loss_density").value == pytest.approx(expected, rel=1e-12)
    assert any(w.code == "sampling_sensitivity" for w in output.warnings)


# --------------------------------------------------------------------------------------------------
# minor loops, DC bias, domain, consistency of arguments
# --------------------------------------------------------------------------------------------------
def test_minor_loops_rejected_or_warned():
    _, igse = make_models()
    minor = Waveform.piecewise_linear(F, MINOR_LOOP_POINTS)
    assert minor.has_minor_loops
    with pytest.raises(OutOfDomainError) as excinfo:
        igse.loss_density(waveform=minor)
    assert "minor loops" in str(excinfo.value)
    output = igse.loss_density(waveform=minor, policy="warn")
    codes = [w.code for w in output.warnings]
    assert "minor_loops_ignored" in codes and "out_of_domain" in codes
    assert output.diagnostics["has_minor_loops"] is True and output.diagnostics["n_reversals"] == 4
    assert output.diagnostics["delta_b"] == pytest.approx(0.2)
    assert any(item.code == "no_minor_loops" for item in output.limitations)
    _, samples = minor.samples(1025)
    with pytest.raises(OutOfDomainError):
        igse.loss_density(waveform=Waveform.sampled(F, samples))


def _loop_decomposed_reference(alpha: float, beta: float) -> float:
    """Loop-decomposed iGSE (Venkatachalam et al. 2002) for MINOR_LOOP_POINTS at frequency F.

    The fall 0.1 -> 0.0 over x = 0.3..0.5 passes 0.08 T after a fraction 0.04; the minor loop is 0.08 -> 0.0 -> 0.08
    (DeltaB_minor = 0.08 T, fractions 0.16 and 0.2) and the major loop is -0.1 -> 0.1 -> 0.08 -> -0.1
    (DeltaB_major = 0.2 T, fractions 0.3, 0.04, 0.3). Per segment (1/T) Integral |dB/dt|^alpha dt = |dB|^alpha x^(1-alpha) f^alpha.
    """
    k_i = igse_coefficient(K, alpha, beta)

    def segment(db: float, x: float) -> float:
        return abs(db) ** alpha * x ** (1 - alpha) * F ** alpha

    major = segment(0.2, 0.3) + segment(0.02, 0.04) + segment(0.18, 0.3)
    minor = segment(0.08, 0.16) + segment(0.08, 0.2)
    return k_i * (0.2 ** (beta - alpha) * major + 0.08 ** (beta - alpha) * minor)


@pytest.mark.parametrize("alpha,beta,sign,word", [
    (1.5, 2.5, 1, "overestimates"), (2.0, 1.5, -1, "underestimates"), (1.5, 1.5, 0, "equals"),
])
def test_minor_loop_warning_states_the_correct_direction(alpha, beta, sign, word):
    igse = IGSEModel(K, alpha, beta)
    output = igse.loss_density(waveform=Waveform.piecewise_linear(F, MINOR_LOOP_POINTS), policy="warn")
    undecomposed = output.value("loss_density").value
    decomposed = _loop_decomposed_reference(alpha, beta)
    if sign == 0:
        assert undecomposed == pytest.approx(decomposed, rel=1e-12)  # beta == alpha: DeltaB does not enter
    else:
        assert math.copysign(1.0, undecomposed - decomposed) == sign
        assert abs(undecomposed / decomposed - 1.0) > 0.05  # 1.179 and 0.885 for this waveform
    message = next(w.message for w in output.warnings if w.code == "minor_loops_ignored")
    assert word in message
    if sign > 0:
        assert "underestimat" not in message
    assert any(item.code == "no_minor_loops" and "overestimates" in item.statement and "beta > alpha" in item.statement
               for item in output.limitations)


def test_dc_bias_is_ignored_with_warning():
    _, igse = make_models()
    biased = igse.loss_density(waveform=Waveform.sinusoidal(peak=0.1, frequency=F, dc=0.05))
    unbiased = igse.loss_density(waveform=Waveform.sinusoidal(peak=0.1, frequency=F))
    assert biased.value("loss_density") == unbiased.value("loss_density")
    assert [w.code for w in biased.warnings] == ["dc_bias_ignored"]
    assert biased.domain.inside and biased.domain.checked["dc_bias"] == "ignored"
    assert biased.diagnostics["dc_bias_present"] is True and biased.diagnostics["dc_component"] == 0.05
    assert unbiased.diagnostics["dc_bias_present"] is False
    assert any(item.code == "dc_bias_ignored" for item in biased.limitations)


def test_dc_bias_threshold_ignores_discretisation_residuals_but_not_real_offsets():
    _, igse = make_models()
    assert DC_BIAS_RELATIVE_THRESHOLD == 1e-3
    for n in (65, 1024, 4096):
        _, samples = Waveform.triangular(peak=0.1, frequency=F, duty=0.3).samples(n)  # corners off the grid
        output = igse.loss_density(waveform=Waveform.sampled(F, samples))
        assert output.diagnostics["dc_component"] != 0.0  # the residual is reported ...
        assert output.diagnostics["dc_bias_present"] is False  # ... but is not a DC bias
        assert not any(w.code == "dc_bias_ignored" for w in output.warnings)
    peak_to_peak = 0.2
    below = igse.loss_density(waveform=Waveform.sinusoidal(peak=0.1, frequency=F,
                                                           dc=0.9 * DC_BIAS_RELATIVE_THRESHOLD * peak_to_peak))
    above = igse.loss_density(waveform=Waveform.sinusoidal(peak=0.1, frequency=F,
                                                           dc=1.1 * DC_BIAS_RELATIVE_THRESHOLD * peak_to_peak))
    assert below.warnings == [] and below.diagnostics["dc_bias_present"] is False
    assert [w.code for w in above.warnings] == ["dc_bias_ignored"] and above.diagnostics["dc_bias_present"] is True
    assert above.value("loss_density") == below.value("loss_density")  # the offset never enters the value


def test_reversal_tolerance_is_explicit_recorded_and_flagged():
    _, igse = make_models()
    base = Waveform.trapezoidal(peak=0.1, frequency=F, duty_rise=0.2, duty_flat=0.3)
    x = np.linspace(0.0, 1.0, 1001)
    _, b = base.samples(1001)
    rippled = b + 1e-7 * np.sin(997.0 * x)  # 5e-7 of the excursion: above the default tolerance, physically negligible
    rippled[-1] = rippled[0]
    waveform = Waveform.sampled(F, rippled)
    assert waveform.count_reversals() > 2
    with pytest.raises(OutOfDomainError):
        igse.loss_density(waveform=waveform)
    output = igse.loss_density(waveform=waveform, reversal_tolerance=1e-5)
    assert output.domain.inside and output.diagnostics["n_reversals"] == 2 and output.diagnostics["has_minor_loops"] is False
    assert output.diagnostics["reversal_tolerance"] == 1e-5 and output.diagnostics["waveform"]["n_reversals"] == 2
    assert [w.code for w in output.warnings] == ["reversal_tolerance_raised"]
    assert "loss integral still includes" in output.warnings[0].message
    clean = igse.loss_density(waveform=base).value("loss_density").value
    assert output.value("loss_density").value == pytest.approx(clean, rel=1e-3)
    default = igse.loss_density(waveform=base)
    assert default.diagnostics["reversal_tolerance"] == DEFAULT_REVERSAL_TOLERANCE and default.warnings == []
    stricter = igse.loss_density(waveform=base, reversal_tolerance=0.0)  # below the default: no flag
    assert stricter.warnings == [] and stricter.diagnostics["reversal_tolerance"] == 0.0
    with pytest.raises(InvalidInputError):
        igse.loss_density(waveform=base, reversal_tolerance=-1.0)
    with pytest.raises(InvalidInputError):
        igse.loss_density(waveform=base, reversal_tolerance=NAN)
    with pytest.raises(InvalidInputError):  # suppresses every change of the waveform: classification is meaningless
        igse.loss_density(waveform=base, reversal_tolerance=2.0)


def test_noisy_measured_data_is_not_rescued_by_the_tolerance():
    _, igse = make_models()
    rng = np.random.default_rng(0)
    _, clean = Waveform.sinusoidal(peak=0.1, frequency=F).samples(1024)
    noisy = clean + rng.normal(0.0, 1e-4, clean.size)  # 1e-3 relative Gaussian noise
    noisy[-1] = noisy[0]
    waveform = Waveform.sampled(F, noisy)
    assert waveform.count_reversals() > 2
    with pytest.raises(OutOfDomainError):
        igse.loss_density(waveform=waveform)
    output = igse.loss_density(waveform=waveform, policy="warn")
    reference = SteinmetzModel(K, ALPHA, BETA).closed_form(F, 0.1)
    assert 1.02 < output.value("loss_density").value / reference < 1.2  # noise inflates |dB/dt|^alpha (~6 % here)
    assert any(item.code == "noisy_sampled_data" for item in output.limitations)
    # a tolerance large enough to hide the noise also hides the waveform at this sample count
    with pytest.raises(InvalidInputError):
        igse.loss_density(waveform=waveform, reversal_tolerance=1e-2)


def test_domain_policy_with_waveforms():
    _, igse = make_models()
    with pytest.raises(OutOfDomainError):
        igse.loss_density(waveform=Waveform.triangular(peak=0.1, frequency=1e7))
    output = igse.loss_density(waveform=Waveform.triangular(peak=0.1, frequency=1e7), policy="warn")
    assert output.diagnostics["out_of_domain"] is True and [w.code for w in output.warnings] == ["out_of_domain"]
    with pytest.raises(OutOfDomainError):
        igse.loss_density(waveform=Waveform.triangular(peak=0.5, frequency=F))
    with pytest.raises(OutOfDomainError):
        igse.loss_density(waveform=Waveform.triangular(peak=0.1, frequency=F), temperature=298.15)
    assert igse.loss_density(waveform=Waveform.triangular(peak=0.1, frequency=F), temperature=375.0).domain.inside


def test_waveform_argument_consistency():
    _, igse = make_models()
    waveform = Waveform.triangular(peak=0.1, frequency=F, duty=0.4)
    reference = igse.loss_density(waveform=waveform).value("loss_density")
    assert igse.loss_density(frequency=F, flux_density_peak=0.1, waveform=waveform).value("loss_density") == reference
    assert igse.loss_density(frequency=F * (1 + 1e-12), waveform=waveform).value("loss_density") == reference
    with pytest.raises(InvalidInputError):
        igse.loss_density(frequency=2 * F, waveform=waveform)
    with pytest.raises(InvalidInputError):
        igse.loss_density(flux_density_peak=0.2, waveform=waveform)
    with pytest.raises(InvalidInputError):
        igse.loss_density(frequency=F)
    with pytest.raises(InvalidInputError):
        igse.loss_density(flux_density_peak=0.1)
    with pytest.raises(InvalidInputError):
        igse.loss_density(waveform="triangular")
    with pytest.raises(InvalidInputError):
        igse.loss_density(waveform=waveform, policy="clip")
    with pytest.raises(InvalidInputError):
        igse.loss_density(waveform=waveform, duty=0.4)
    with pytest.raises(InvalidInputError):
        igse.loss_density(waveform=waveform, temperature=-1.0)


@pytest.mark.parametrize("kwargs", [
    dict(frequency=-F, flux_density_peak=0.1), dict(frequency=0.0, flux_density_peak=0.1),
    dict(frequency=NAN, flux_density_peak=0.1), dict(frequency=F, flux_density_peak=-0.1),
    dict(frequency=F, flux_density_peak=INF), dict(frequency=True, flux_density_peak=0.1),
    dict(frequency="1e5", flux_density_peak=0.1),
])
def test_invalid_scalar_inputs(kwargs):
    with pytest.raises(InvalidInputError):
        make_models()[1].loss_density(**kwargs)


def test_zero_excitation():
    _, igse = make_models()
    output = igse.loss_density(frequency=F, flux_density_peak=0.0)
    assert output.value("loss_density").value == 0.0
    assert [(w.code, w.severity) for w in output.warnings] == [("zero_excitation", "info")]
    assert output.diagnostics["zero_excitation"] is True and output.diagnostics["waveform"] is None
    assert output.domain.inside


def test_overflow_paths_raise_numerical_error():
    igse = IGSEModel(K, ALPHA, BETA)
    with pytest.raises(NumericalError):
        igse.loss_density(frequency=1e300, flux_density_peak=1e300)
    with pytest.raises(NumericalError):
        igse.loss_density(waveform=Waveform.triangular(peak=1e200, frequency=1e150))
    _, samples = Waveform.sinusoidal(peak=1e200, frequency=1e150).samples(64)
    with pytest.raises(NumericalError):
        igse.loss_density(waveform=Waveform.sampled(1e150, samples))


def test_evaluate_passthrough_and_vectorised_helper():
    _, igse = make_models()
    via_contract = igse.evaluate(frequency=F, flux_density_peak=0.1)
    assert via_contract.value("loss_density").value == pytest.approx(1e5)
    grid = igse.loss_density_array([1e4, F], [0.1, 0.1])
    assert grid[1] == pytest.approx(1e5) and grid[0] == pytest.approx(1e5 / 10 ** ALPHA)


# --------------------------------------------------------------------------------------------------
# verification checks of this module
# --------------------------------------------------------------------------------------------------
EXPECTED_CHECK_IDS = {
    "igse.sinusoid-equals-steinmetz", "igse.triangular-closed-form", "igse.sampled-converges",
    "igse.ki-approximation-agreement", "igse.minor-loop-detection", "steinmetz.scaling-laws",
    "steinmetz.zero-excitation", "steinmetz.invalid-inputs-rejected", "steinmetz.overflow-detected",
    "waveform.descriptors-consistent",
}


def test_all_expected_checks_are_registered():
    registered = {spec.id for spec in suite.registered_checks().values() if spec.module == check_empirical.__name__}
    assert registered == EXPECTED_CHECK_IDS
    assert {func.check_id for func in check_empirical.ALL_CHECKS} == EXPECTED_CHECK_IDS
    for spec in suite.registered_checks().values():
        if spec.module == check_empirical.__name__:
            assert spec.capability in {"aipe.magnetics.model.steinmetz", "aipe.magnetics.model.igse",
                                       "aipe.magnetics.model.waveforms"}
            assert spec.level == ("L1" if spec.id in ("steinmetz.invalid-inputs-rejected",
                                                      "waveform.descriptors-consistent") else "L2")


@pytest.mark.parametrize("check_function", check_empirical.ALL_CHECKS, ids=lambda f: f.check_id)
def test_check_passes(check_function):
    result = check_function()
    assert isinstance(result, suite.CheckResult)
    assert result.passed, f"{check_function.check_id} failed: metrics={result.metrics} notes={result.notes}"
    json.dumps(result.metrics)
    assert result == check_function()  # deterministic


def test_registered_evidence_covers_what_it_declares():
    specs = {spec.id: spec for spec in suite.registered_checks().values() if spec.module == check_empirical.__name__}
    triangular = specs["igse.triangular-closed-form"]
    assert triangular.range is not None and set(triangular.range.excitation) == {"triangular", "trapezoidal"}
    metrics = check_empirical.igse_triangular_closed_form_check().metrics
    assert metrics["n_trapezoidal_points"] >= 3 * 9 * len(check_empirical.ALPHA_SWEEP) * len(check_empirical.BETA_SWEEP)
    assert metrics["max_rel_error_trapezoidal"] < 1e-12
    sampled = check_empirical.igse_sampled_converges()
    assert sampled.metrics["scale_invariance_max_abs_spread_of_rel_error"] < 1e-13
    assert any("scales exactly" in note for note in sampled.notes)
    minor = check_empirical.igse_minor_loop_detection().metrics
    assert minor["undecomposed_over_decomposed_alpha1.5_beta2.5"] > 1.0
    assert minor["undecomposed_over_decomposed_alpha2_beta1.5"] < 1.0


def test_checks_run_through_suite_runner_and_build_evidence():
    specs = [spec for spec in suite.registered_checks().values() if spec.module == check_empirical.__name__]
    outcomes = [suite.run_check(spec) for spec in sorted(specs, key=lambda s: s.id)]
    assert all(outcome.status == "pass" for outcome in outcomes), [
        (o.spec.id, o.error) for o in outcomes if o.status != "pass"]
    report = suite.build_report(outcomes)
    assert report.passed()
    assert report.summary["aipe.magnetics.model.igse"]["highest_passing_level"] == "L2"
    assert report.summary["aipe.magnetics.model.steinmetz"]["highest_passing_level"] == "L2"
    registry = suite.build_registry(outcomes)
    status = registry.status_for("aipe.magnetics.model.igse", frequency=F, flux_density_peak=0.1,
                                 excitation="sinusoidal")
    assert status.state == "verified" and status.level == "L2" and status.in_verified_range is True
    trapezoidal = registry.status_for("aipe.magnetics.model.igse", frequency=F, flux_density_peak=0.1,
                                      excitation="trapezoidal")
    assert trapezoidal.state == "verified" and "check:igse.triangular-closed-form" in trapezoidal.evidence_refs
    outside = registry.status_for("aipe.magnetics.model.igse", frequency=1e8, flux_density_peak=0.1,
                                  excitation="sinusoidal")
    assert outside.state in ("verified", "out_of_range")
    assert "check:igse.sampled-converges" in {record.id for record in registry.records}
    for record in registry.records:
        if record.id == "check:igse.ki-approximation-agreement":
            assert record.reference is not None and record.reference.id == IGSE_SOURCE.id
    assert ValidityRange(min=1e3, max=1e6, unit="Hz") == next(
        r.range.frequency for r in registry.records if r.id == "check:igse.sinusoid-equals-steinmetz")
