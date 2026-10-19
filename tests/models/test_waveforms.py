"""Tests for aipe_magnetics.models.waveforms."""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

from aipe_magnetics.common.errors import InvalidInputError, UnsupportedOperationError
from aipe_magnetics.models.waveforms import (
    DEFAULT_REVERSAL_TOLERANCE,
    PIECEWISE_LINEAR_KINDS,
    Waveform,
    validate_bool,
    validate_real,
)

F = 1e5
NAN, INF = float("nan"), float("inf")


# --------------------------------------------------------------------------------------------------
# validate_real / validate_bool
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("value", [1, 1.5, np.float64(2.0), np.int64(3), np.float32(0.5)])
def test_validate_real_accepts_python_and_numpy_scalars(value):
    assert validate_real("x", value) == float(value)


@pytest.mark.parametrize("value", [True, np.bool_(True), "1.0", None, 1 + 1j, [1.0], NAN, INF, -INF])
def test_validate_real_rejects_non_real_or_non_finite(value):
    with pytest.raises(InvalidInputError):
        validate_real("x", value)


def test_validate_real_sign_constraints():
    with pytest.raises(InvalidInputError):
        validate_real("x", 0.0, positive=True)
    with pytest.raises(InvalidInputError):
        validate_real("x", -1e-300, nonnegative=True)
    assert validate_real("x", 0.0, nonnegative=True) == 0.0


def test_validate_bool_rejects_truthiness_coercion():
    assert validate_bool("flag", True) is True and validate_bool("flag", np.bool_(False)) is False
    for value in ("False", 1, 0, None, 1.0, [True]):
        with pytest.raises(InvalidInputError):
            validate_bool("flag", value)


# --------------------------------------------------------------------------------------------------
# sinusoidal
# --------------------------------------------------------------------------------------------------
def test_sinusoidal_descriptors_and_samples():
    wf = Waveform.sinusoidal(peak=0.1, frequency=F, dc=0.02)
    assert wf.kind == "sinusoidal"
    assert wf.frequency == F and wf.period == pytest.approx(1e-5)
    assert wf.peak == 0.1 and wf.peak_to_peak == pytest.approx(0.2)
    assert wf.dc_component == 0.02 and wf.maximum == pytest.approx(0.12) and wf.minimum == pytest.approx(-0.08)
    assert wf.count_reversals() == 2 and not wf.has_minor_loops
    assert wf.n_samples is None and not wf.is_piecewise_linear
    t, b = wf.samples(9)
    assert t[0] == 0.0 and t[-1] == pytest.approx(wf.period)
    assert b[0] == pytest.approx(0.02) and b[2] == pytest.approx(0.12) and b[6] == pytest.approx(-0.08)
    assert b[-1] == pytest.approx(b[0])
    t_default, _ = wf.samples()
    assert t_default.size == 1024


def test_sinusoidal_rejects_invalid_inputs():
    with pytest.raises(InvalidInputError):
        Waveform.sinusoidal(peak=0.0, frequency=F)
    with pytest.raises(InvalidInputError):
        Waveform.sinusoidal(peak=-0.1, frequency=F)
    with pytest.raises(InvalidInputError):
        Waveform.sinusoidal(peak=0.1, frequency=0.0)
    with pytest.raises(InvalidInputError):
        Waveform.sinusoidal(peak=0.1, frequency=F, dc=NAN)
    with pytest.raises(InvalidInputError):
        Waveform.sinusoidal(peak="0.1", frequency=F)


def test_segments_unsupported_for_sinusoidal_and_sampled():
    with pytest.raises(UnsupportedOperationError):
        Waveform.sinusoidal(0.1, F).segments()
    with pytest.raises(UnsupportedOperationError):
        Waveform.sinusoidal(0.1, F).breakpoints
    _, b = Waveform.sinusoidal(0.1, F).samples(64)
    with pytest.raises(UnsupportedOperationError):
        Waveform.sampled(F, b).segments()


# --------------------------------------------------------------------------------------------------
# triangular / trapezoidal
# --------------------------------------------------------------------------------------------------
def test_triangular_segments_are_analytic():
    wf = Waveform.triangular(peak=0.1, frequency=F, duty=0.3, dc=0.05)
    assert wf.kind == "triangular" and wf.is_piecewise_linear
    assert wf.peak == 0.1 and wf.peak_to_peak == pytest.approx(0.2) and wf.dc_component == 0.05
    (dt1, db1), (dt2, db2) = wf.segments()
    assert dt1 == pytest.approx(0.3e-5) and db1 == pytest.approx(0.2)
    assert dt2 == pytest.approx(0.7e-5) and db2 == pytest.approx(-0.2)
    assert wf.count_reversals() == 2 and not wf.has_minor_loops
    assert wf.parameters == {"duty": 0.3, "dc": 0.05}
    _, b = wf.samples(11)
    assert b[0] == pytest.approx(-0.05) and b[3] == pytest.approx(0.15) and b[-1] == pytest.approx(-0.05)


@pytest.mark.parametrize("duty", [0.0, 1.0, -0.1, 1.5, NAN])
def test_triangular_rejects_invalid_duty(duty):
    with pytest.raises(InvalidInputError):
        Waveform.triangular(peak=0.1, frequency=F, duty=duty)


def test_trapezoidal_segments_and_validation():
    wf = Waveform.trapezoidal(peak=0.1, frequency=F, duty_rise=0.2, duty_flat=0.3)
    assert wf.kind == "trapezoidal"
    segments = wf.segments()
    assert len(segments) == 4
    assert [round(dt * F, 12) for dt, _ in segments] == [0.2, 0.3, 0.2, 0.3]
    assert [db for _, db in segments] == pytest.approx([0.2, 0.0, -0.2, 0.0])
    assert sum(dt for dt, _ in segments) == pytest.approx(wf.period)
    assert wf.count_reversals() == 2 and wf.dc_component == 0.0
    with pytest.raises(InvalidInputError):
        Waveform.trapezoidal(peak=0.1, frequency=F, duty_rise=0.3, duty_flat=0.3)  # fractions sum to 1.2
    with pytest.raises(InvalidInputError):
        Waveform.trapezoidal(peak=0.1, frequency=F, duty_rise=0.0, duty_flat=0.5)
    with pytest.raises(InvalidInputError):
        Waveform.trapezoidal(peak=0.1, frequency=F, duty_rise=0.6, duty_flat=-0.1)


def test_trapezoidal_without_flat_degenerates_to_triangle():
    wf = Waveform.trapezoidal(peak=0.1, frequency=F, duty_rise=0.5, duty_flat=0.0)
    assert wf.breakpoints.tolist() == [[0.0, -0.1], [0.5, 0.1], [1.0, -0.1]]
    assert wf.segments() == pytest.approx([(0.5e-5, 0.2), (0.5e-5, -0.2)])


# --------------------------------------------------------------------------------------------------
# piecewise linear
# --------------------------------------------------------------------------------------------------
def test_piecewise_linear_descriptors():
    wf = Waveform.piecewise_linear(F, [(0.0, -0.1), (0.25, 0.3), (0.5, 0.3), (1.0, -0.1)])
    assert wf.kind == "piecewise_linear"
    assert wf.peak == pytest.approx(0.2) and wf.peak_to_peak == pytest.approx(0.4)
    assert wf.maximum == 0.3 and wf.minimum == -0.1
    # exact time average of the linear interpolant: segment trapezoid areas
    assert wf.dc_component == pytest.approx(0.25 * 0.1 + 0.25 * 0.3 + 0.5 * 0.1, abs=1e-15)
    assert wf.count_reversals() == 2  # flat top is not a reversal
    _, b = wf.samples(5)
    assert b.tolist() == pytest.approx([-0.1, 0.3, 0.3, 0.1, -0.1])
    assert wf.parameters == {"n_breakpoints": 4}


def test_piecewise_linear_accepts_numpy_array_breakpoints():
    wf = Waveform.piecewise_linear(F, np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]]))
    assert wf.segments() == pytest.approx([(0.5e-5, 1.0), (0.5e-5, -1.0)])


@pytest.mark.parametrize("breakpoints", [
    [(0.0, -0.1), (0.5, 0.1), (1.0, 0.0)],             # not closed
    [(0.1, -0.1), (0.5, 0.1), (1.0, -0.1)],            # does not start at 0
    [(0.0, -0.1), (0.5, 0.1), (0.9, -0.1)],            # does not end at 1
    [(0.0, -0.1), (0.5, 0.1), (0.5, 0.0), (1.0, -0.1)],  # not strictly increasing
    [(0.0, 0.1), (1.0, 0.1)],                          # constant
    [(0.0, 0.1)],                                      # too short
    [(0.0, -0.1), (0.5, NAN), (1.0, -0.1)],            # non-finite
    [(0.0, -0.1, 1.0), (0.5, 0.1, 1.0), (1.0, -0.1, 1.0)],  # wrong shape
    [("0", -0.1), ("0.5", 0.1), ("1", -0.1)],          # non-numeric
    [(0.0, 0.0), (5e-324, 0.1), (1.0, 0.0)],           # segment duration underflows to 0 s (infinite dB/dt)
])
def test_piecewise_linear_rejects_invalid_breakpoints(breakpoints):
    with pytest.raises(InvalidInputError):
        Waveform.piecewise_linear(F, breakpoints)


def test_piecewise_linear_closure_tolerance_is_explicit():
    points = [(0.0, -0.1), (0.5, 0.1), (1.0, -0.1 + 1e-6)]
    with pytest.raises(InvalidInputError):
        Waveform.piecewise_linear(F, points)
    assert Waveform.piecewise_linear(F, points, closure_tolerance=1e-4).kind == "piecewise_linear"
    with pytest.raises(InvalidInputError):
        Waveform.piecewise_linear(F, points, closure_tolerance=-1.0)


def test_piecewise_linear_segment_duration_must_be_resolvable():
    with pytest.raises(InvalidInputError, match="not resolvable in time"):
        Waveform.piecewise_linear(F, [(0.0, 0.0), (5e-324, 0.1), (1.0, 0.0)])
    tiny = Waveform.piecewise_linear(F, [(0.0, 0.0), (1e-300, 0.1), (1.0, 0.0)])  # tiny but representable
    assert tiny.segments()[0][0] > 0.0 and np.diff(tiny.breakpoints[:, 0])[0] == 1e-300


def test_overflowing_excursions_are_rejected_without_numpy_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a bare numpy RuntimeWarning would surface as an exception here
        with pytest.raises(InvalidInputError, match="overflows double precision"):
            Waveform.piecewise_linear(F, [(0.0, -1e308), (0.5, 1e308), (1.0, -1e308)])
        with pytest.raises(InvalidInputError, match="overflows double precision"):
            Waveform.piecewise_linear(F, [(0.0, 1e308), (0.5, 1e308), (0.75, 1e307), (1.0, 1e308)])  # finite excursion
        with pytest.raises(InvalidInputError, match="overflows double precision"):
            Waveform.sampled(F, np.array([-1e308, 1e308] * 4 + [-1e308]))
        with pytest.raises(InvalidInputError, match="overflows double precision"):
            Waveform.sampled(F, np.array([1e308] * 8 + [1e307, 1e308]))  # finite excursion, sum overflows
    # large but representable values stay JSON-serialisable
    big = Waveform.piecewise_linear(F, [(0.0, -1e153), (0.5, 1e153), (1.0, -1e153)])
    assert math.isfinite(big.peak) and math.isfinite(big.dc_component)
    json.dumps(big.describe(), allow_nan=False)


# --------------------------------------------------------------------------------------------------
# sampled
# --------------------------------------------------------------------------------------------------
def test_sampled_descriptors_and_resampling():
    _, b = Waveform.triangular(peak=0.1, frequency=F, duty=0.5, dc=0.01).samples(65)
    wf = Waveform.sampled(F, b)
    assert wf.kind == "sampled" and wf.n_samples == 65 and not wf.is_piecewise_linear
    assert wf.peak == pytest.approx(0.1) and wf.peak_to_peak == pytest.approx(0.2)
    assert wf.dc_component == pytest.approx(0.01, abs=1e-15)
    assert wf.count_reversals() == 2 and not wf.has_minor_loops
    assert wf.parameters == {"n_samples": 65, "n_input_samples": 65, "endpoint_included": True}
    t, native = wf.samples()
    assert native.shape == (65,) and t[-1] == pytest.approx(1e-5)
    native[0] = 99.0  # copy, not a view
    assert wf.samples()[1][0] == pytest.approx(-0.09)
    _, resampled = wf.samples(129)
    assert resampled.shape == (129,) and resampled[64] == pytest.approx(0.11)
    with pytest.raises(InvalidInputError):
        wf.samples(1)
    with pytest.raises(InvalidInputError):
        wf.samples(True)
    with pytest.raises(InvalidInputError):
        wf.samples(10.0)


@pytest.mark.parametrize("samples", [
    [0, 1, 0, -1, 0, 1, 0],                      # fewer than 8
    [0, 1, 0, -1, 0, 1, 0, NAN, 0],              # non-finite
    [0, 1, 2, 3, 4, 5, 6, 7, 8],                 # not closed
    [0.1] * 16,                                  # constant
    [[0, 1], [0, -1], [0, 1], [0, -1], [0, 1]],  # 2-D
    [True, False] * 8,                           # bool dtype
    [1 + 1j] * 8,                                # complex
    ["0", "1", "0", "-1", "0", "1", "0", "0"],   # strings
])
def test_sampled_rejects_invalid_samples(samples):
    with pytest.raises(InvalidInputError):
        Waveform.sampled(F, samples)


def test_sampled_closure_tolerance_is_explicit():
    b = np.sin(2 * np.pi * np.linspace(0.0, 1.0, 64))
    b[-1] = 1e-3
    with pytest.raises(InvalidInputError):
        Waveform.sampled(F, b)
    assert Waveform.sampled(F, b, closure_tolerance=1e-3).n_samples == 64


def test_sampled_half_open_convention():
    trapezoid = Waveform.trapezoidal(1.0, F, duty_rise=0.2, duty_flat=0.3)
    _, inclusive = trapezoid.samples(1001)  # [0, T] with 1000 intervals; the corners lie on the grid
    half_open = inclusive[:-1]              # the same waveform in the [0, T) convention
    wf_inclusive = Waveform.sampled(F, inclusive)
    wf_half_open = Waveform.sampled(F, half_open, endpoint_included=False)
    assert wf_half_open.n_samples == 1001
    assert wf_half_open.parameters == {"n_samples": 1001, "n_input_samples": 1000, "endpoint_included": False}
    assert wf_inclusive.parameters == {"n_samples": 1001, "n_input_samples": 1001, "endpoint_included": True}
    np.testing.assert_array_equal(wf_half_open.samples()[1], wf_inclusive.samples()[1])
    assert wf_half_open.dc_component == wf_inclusive.dc_component and wf_half_open.peak == wf_inclusive.peak
    assert wf_half_open.count_reversals() == 2
    # the same [0, T) data passed as inclusive is accepted (flat at both ends) but describes a stretched waveform
    stretched = Waveform.sampled(F, half_open)
    assert stretched.n_samples == 1000 and stretched.samples()[0][1] == pytest.approx(wf_half_open.period / 999)
    with pytest.raises(InvalidInputError):
        Waveform.sampled(F, half_open, endpoint_included="no")
    # a [0, T) sinusoid is rejected under the inclusive convention and accepted under the half-open one
    sine = 0.1 * np.sin(2 * np.pi * np.arange(64) / 64)
    with pytest.raises(InvalidInputError, match="endpoint_included=False"):
        Waveform.sampled(F, sine)
    assert Waveform.sampled(F, sine, endpoint_included=False).count_reversals() == 2


# --------------------------------------------------------------------------------------------------
# reversals / minor loops
# --------------------------------------------------------------------------------------------------
def test_minor_loops_detected_in_pwl_and_sampled_waveforms():
    minor = Waveform.piecewise_linear(F, [(0.0, -0.1), (0.3, 0.1), (0.5, 0.0), (0.7, 0.08), (1.0, -0.1)])
    assert minor.count_reversals() == 4 and minor.has_minor_loops
    _, b = minor.samples(257)
    assert Waveform.sampled(F, b).count_reversals() == 4
    two_dips = Waveform.piecewise_linear(F, [(0.0, 0.0), (0.2, 1.0), (0.3, 0.5), (0.4, 1.0), (0.6, 0.0), (0.7, 0.4),
                                             (0.8, 0.0), (1.0, 0.0)])
    assert two_dips.count_reversals() == 6


def test_reversal_tolerance_ignores_rounding_noise():
    base = Waveform.trapezoidal(peak=0.1, frequency=F, duty_rise=0.2, duty_flat=0.3)
    x = np.linspace(0.0, 1.0, 1001)  # 1000 intervals: the vertices at x = 0.2, 0.5, 0.7 lie on the grid
    _, b = base.samples(1001)
    noisy = b + 1e-12 * np.sin(997.0 * x)  # deterministic ripple far below 1e-9 * peak_to_peak on the flat tops
    noisy[-1] = noisy[0]
    wf = Waveform.sampled(F, noisy)
    assert wf.count_reversals() == 2 and not wf.has_minor_loops  # sub-tolerance ripple is ignored
    assert wf.count_reversals(tolerance=0.0) > 2  # ... but is visible without tolerance
    with pytest.raises(InvalidInputError):
        wf.count_reversals(tolerance=-1.0)
    assert DEFAULT_REVERSAL_TOLERANCE == 1e-9


def test_describe_forwards_the_reversal_tolerance():
    base = Waveform.trapezoidal(peak=0.1, frequency=F, duty_rise=0.2, duty_flat=0.3)
    x = np.linspace(0.0, 1.0, 1001)
    _, b = base.samples(1001)
    rippled = b + 1e-7 * np.sin(997.0 * x)  # 5e-7 of the excursion: above the default tolerance
    rippled[-1] = rippled[0]
    wf = Waveform.sampled(F, rippled)
    default = wf.describe()
    assert default["n_reversals"] > 2 and default["has_minor_loops"] is True
    assert default["reversal_tolerance"] == DEFAULT_REVERSAL_TOLERANCE
    relaxed = wf.describe(reversal_tolerance=1e-5)
    assert relaxed["n_reversals"] == 2 and relaxed["has_minor_loops"] is False and relaxed["reversal_tolerance"] == 1e-5
    json.dumps(relaxed)


# --------------------------------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------------------------------
def test_describe_is_json_serialisable():
    for wf in (Waveform.sinusoidal(0.1, F), Waveform.triangular(0.1, F, 0.4), Waveform.trapezoidal(0.1, F, 0.2, 0.3),
               Waveform.piecewise_linear(F, [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]),
               Waveform.sampled(F, np.sin(2 * np.pi * np.linspace(0, 1, 32)))):
        text = json.dumps(wf.describe(), allow_nan=False)
        data = json.loads(text)
        assert data["kind"] == wf.kind and data["units"] == {"flux_density": "T", "frequency": "Hz", "time": "s"}
        assert data["has_minor_loops"] is False


def test_breakpoints_are_read_only():
    wf = Waveform.triangular(0.1, F)
    with pytest.raises(ValueError):
        wf.breakpoints[0, 1] = 5.0
    assert wf.kind in PIECEWISE_LINEAR_KINDS


def test_internal_constructor_rejects_unknown_kind():
    with pytest.raises(InvalidInputError):
        Waveform("square", F, peak=0.1, dc_component=0.0, maximum=0.1, minimum=-0.1)


@pytest.mark.parametrize("kwargs", [
    dict(frequency=-5.0, peak=1.0, dc_component=0.0, maximum=1.0, minimum=-1.0),   # negative frequency
    dict(frequency=0.0, peak=1.0, dc_component=0.0, maximum=1.0, minimum=-1.0),    # zero frequency
    dict(frequency=F, peak=-1.0, dc_component=0.0, maximum=1.0, minimum=-1.0),     # negative peak
    dict(frequency=F, peak=0.5, dc_component=0.0, maximum=1.0, minimum=-1.0),      # peak != (max - min) / 2
    dict(frequency=F, peak=1.0, dc_component=5.0, maximum=1.0, minimum=-1.0),      # dc outside [min, max]
    dict(frequency=F, peak=1.0, dc_component=0.0, maximum=-1.0, minimum=1.0),      # max < min
    dict(frequency=F, peak=1.0, dc_component=0.0, maximum=NAN, minimum=-1.0),      # non-finite
    dict(frequency=F, peak=1e308, dc_component=0.0, maximum=1e308, minimum=-1e308),  # excursion overflows
])
def test_direct_construction_revalidates_descriptors(kwargs):
    with pytest.raises(InvalidInputError):
        Waveform("sinusoidal", kwargs.pop("frequency"), **kwargs)


def test_direct_construction_requires_payload_matching_the_kind():
    with pytest.raises(InvalidInputError):
        Waveform("triangular", F, peak=1.0, dc_component=0.0, maximum=1.0, minimum=-1.0)  # breakpoints missing
    with pytest.raises(InvalidInputError):
        Waveform("sinusoidal", F, peak=1.0, dc_component=0.0, maximum=1.0, minimum=-1.0, samples=np.zeros(8))
    direct = Waveform("sinusoidal", F, peak=0.3, dc_component=1.0, maximum=1.3, minimum=0.7, parameters={"dc": 1.0})
    assert direct.describe() == Waveform.sinusoidal(0.3, F, dc=1.0).describe()


def test_large_dc_offset_descriptors_tolerate_cancellation():
    wf = Waveform.sinusoidal(1e-6, F, dc=1e6)  # (maximum - minimum) / 2 suffers cancellation but must construct
    assert wf.peak == 1e-6 and wf.dc_component == 1e6 and wf.maximum > wf.minimum


def test_peak_is_half_of_peak_to_peak_for_all_kinds():
    waveforms = [Waveform.sinusoidal(0.3, F, dc=1.0), Waveform.triangular(0.3, F, 0.2, dc=-1.0),
                 Waveform.piecewise_linear(F, [(0.0, 0.0), (0.1, 0.6), (0.4, 0.2), (1.0, 0.0)])]
    for wf in waveforms:
        assert math.isclose(wf.peak, 0.5 * wf.peak_to_peak)
