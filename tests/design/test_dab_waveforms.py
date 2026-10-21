"""Single-phase-shift DAB waveform model (design/transformers/dab.py): power relation, inductance, exact piecewise-linear currents."""

from __future__ import annotations

import math

import numpy as np
import pytest

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

from aipe_magnetics.common.errors import InvalidInputError
from aipe_magnetics.design.transformers import dab

P, V1, V2, N, F = 10e3, 800.0, 400.0, 2.0, 100e3
PHI = math.radians(22.5)


def test_power_inductance_and_phase_shift_are_mutually_consistent():
    inductance = dab.inductance_for_phase_shift(P, V1, V2, N, F, PHI)
    assert inductance == pytest.approx(35e-6, rel=1e-6)  # the brief's 35 uH at 22.5 deg
    assert dab.sps_power(V1, V2, N, F, inductance, PHI) == pytest.approx(P, rel=1e-12)
    assert dab.phase_shift_for_power(P, V1, V2, N, F, inductance) == pytest.approx(PHI, rel=1e-12)
    with pytest.raises(InvalidInputError):
        dab.phase_shift_for_power(1e6, V1, V2, N, F, inductance)  # above the SPS maximum at pi/2
    with pytest.raises(InvalidInputError):
        dab.inductance_for_phase_shift(P, V1, V2, N, F, 0.0)


def test_inductor_current_is_exact_piecewise_linear_and_matches_the_simulated_rms():
    current = dab.inductor_current(V1, V2, N, F, 35e-6, PHI)
    x = np.linspace(0.0, 1.0, 200001)
    values = current.value(x)
    # exact closed-form RMS / mean agree with a fine numerical integration of the same waveform
    assert current.rms() == pytest.approx(math.sqrt(_trapezoid(values ** 2, x)), rel=1e-6)
    assert abs(current.mean()) < 1e-9 and abs(_trapezoid(values, x)) < 1e-6
    # half-wave antisymmetry i(t + T/2) = -i(t) and the switching-instant currents
    assert np.allclose(current.value(x[:100000] + 0.5), -values[:100000], atol=1e-9)
    assert current.value(0.0) < 0 < current.value(PHI / (2 * math.pi))  # ZVS indications of both bridges at the nominal point
    # the ngspice level-1 run of the same converter (brief): i_L rms 13.68 A, peak 14.3 A (trapezoidal, not square)
    assert current.rms() == pytest.approx(13.68, rel=2e-3) and current.peak() == pytest.approx(14.3, rel=5e-3)
    assert current.rms() > P / V1  # the square-current approximation P/V underestimates the RMS
    # a lossless Fourier series: 41 harmonics carry (almost) all of the AC energy and only odd harmonics are present
    amplitudes = current.harmonic_amplitudes(10)
    assert current.harmonic_energy_fraction(41) > 0.9999
    assert all(a < 1e-9 for n, a, _ in amplitudes if n % 2 == 0)


def test_operating_points_cover_the_secondary_voltage_range():
    ops = dab.operating_points(power=P, v1=V1, v2_nominal=V2, v2_range=(360.0, 440.0), n=N, frequency=F, inductance=35e-6)
    assert [op.label for op in ops] == ["nominal", "V2 min", "V2 max"]
    nominal = ops[0]
    assert nominal.phi == pytest.approx(PHI, rel=1e-12)
    summary = nominal.summary(F, N)
    assert summary["primary_current_rms_A"] == pytest.approx(nominal.current.rms()) and summary["secondary_current_rms_A"] == pytest.approx(N * nominal.current.rms())
    assert summary["voltage_conversion_ratio"] == pytest.approx(1.0) and summary["zvs_primary_indicated"] and summary["zvs_secondary_indicated"]
    low = next(op for op in ops if op.label == "V2 min")
    assert low.phi > nominal.phi  # a lower secondary voltage needs a larger phase shift for the same power
    assert all(op.current.points[0][1] == op.current.points[-1][1] for op in ops)  # periodic closure of every waveform


def test_phase_shift_zero_degenerates_to_a_two_breakpoint_triangle():
    current = dab.inductor_current(V1, 400.0, 2.0, F, 35e-6, 0.0)
    xs = [p[0] for p in current.points]
    assert xs[0] == 0.0 and xs[-1] == 1.0 and all(b > a for a, b in zip(xs, xs[1:]))
    assert current.peak() == pytest.approx(0.0, abs=1e-12)  # matched voltages, no phase shift: no current
