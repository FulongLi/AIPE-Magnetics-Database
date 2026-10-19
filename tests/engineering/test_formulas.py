"""Closed-form design relations: independent re-derivations, limiting cases and invalid inputs."""

from __future__ import annotations

import math

import pytest

from aipe_magnetics.common.errors import InvalidInputError
from aipe_magnetics.common.units import MU_0
from aipe_magnetics.engineering import formulas as F


def test_faraday_square_and_sine_against_hand_calculation():
    # 800 V square wave, 100 kHz, 16 turns, Ae = 535 mm^2 -> B_pk = V/(4 f N Ae)
    assert F.flux_density_peak_square(800, 1e5, 16, 535e-6) == pytest.approx(800 / (4 * 1e5 * 16 * 535e-6), rel=1e-12)
    # duty 0.25 halves the volt-seconds
    assert F.flux_density_peak_square(800, 1e5, 16, 535e-6, duty=0.25) == pytest.approx(0.5 * 800 / (4 * 1e5 * 16 * 535e-6))
    # sinusoidal: V_rms = 4.4429 f N Ae B_pk
    b = F.flux_density_peak_sine(565.7, 1e5, 16, 535e-6)
    assert 565.7 == pytest.approx(2 * math.pi / math.sqrt(2) * 1e5 * 16 * 535e-6 * b, rel=1e-12)


def test_turns_round_up_and_realised_flux_never_exceeds_limit():
    for v, f, ae, blim in [(800, 1e5, 535e-6, 0.15), (400, 2e5, 368e-6, 0.1), (48, 5e5, 1e-5, 0.05)]:
        n = F.turns_for_flux_density_square(v, f, ae, blim)
        assert isinstance(n, int) and n >= 1
        assert F.flux_density_peak_square(v, f, n, ae) <= blim * (1 + 1e-12)
        if n > 1:
            assert F.flux_density_peak_square(v, f, n - 1, ae) > blim


def test_gapped_inductance_round_trip_and_ungapped_limit():
    L = F.gapped_inductance(20, 535e-6, 1e-3, 147e-3, 2200)
    assert F.air_gap_for_inductance(L, 20, 535e-6, 147e-3, 2200) == pytest.approx(1e-3, rel=1e-9)
    ungapped = MU_0 * 20 ** 2 * 535e-6 * 2200 / 147e-3
    assert F.gapped_inductance(20, 535e-6, 0.0, 147e-3, 2200) == pytest.approx(ungapped)
    with pytest.raises(InvalidInputError):
        F.air_gap_for_inductance(ungapped * 1.01, 20, 535e-6, 147e-3, 2200)


def test_dowell_factor_against_independent_evaluation_and_asymptotes():
    def dowell(m, delta):
        nu3 = (math.sinh(2 * delta) + math.sin(2 * delta)) / (math.cosh(2 * delta) - math.cos(2 * delta))
        nu2 = (math.sinh(delta) - math.sin(delta)) / (math.cosh(delta) + math.cos(delta))
        return delta * (nu3 + 2 * (m * m - 1) / 3 * nu2)

    skin = 0.2365e-3
    for m, h in [(1, skin), (2, 0.5 * skin), (4, 2 * skin), (6, 3 * skin)]:
        assert F.dowell_ac_resistance_factor(m, h, skin) == pytest.approx(dowell(m, h / skin), rel=1e-12)
    # thin conductor limit -> 1 ; thick limit -> Delta (1 + 2(m^2-1)/3)
    assert F.dowell_ac_resistance_factor(3, 0.01 * skin, skin) == pytest.approx(1.0, abs=1e-3)
    assert F.dowell_ac_resistance_factor(3, 10 * skin, skin) == pytest.approx(10 * (1 + 2 * 8 / 3), rel=1e-3)
    # porosity reduces the effective penetration ratio
    assert F.dowell_ac_resistance_factor(3, skin, skin, porosity=0.5) < F.dowell_ac_resistance_factor(3, skin, skin, porosity=1.0)


def test_copper_resistivity_and_skin_depth():
    assert F.copper_resistivity(293.15) == pytest.approx(1.68e-8)
    assert F.copper_resistivity(373.15) == pytest.approx(1.68e-8 * (1 + 0.00393 * 80))
    d = F.skin_depth(1e5, 1.68e-8)
    assert d == pytest.approx(math.sqrt(1.68e-8 / (math.pi * 1e5 * MU_0)), rel=1e-12)


def test_area_products_scale_as_expected():
    ap = F.transformer_area_product(20000, 1e5, 0.15, 4e6, 0.4, 4.0)
    assert F.transformer_area_product(40000, 1e5, 0.15, 4e6, 0.4, 4.0) == pytest.approx(2 * ap)
    assert F.transformer_area_product(20000, 2e5, 0.15, 4e6, 0.4, 4.0) == pytest.approx(ap / 2)
    ind = F.inductor_area_product(100e-6, 20, 14, 0.3, 4e6, 0.4)
    assert ind == pytest.approx(100e-6 * 20 * 14 / (0.3 * 4e6 * 0.4))


@pytest.mark.parametrize("call", [
    lambda: F.flux_density_peak_square(-800, 1e5, 16, 535e-6),
    lambda: F.flux_density_peak_square(800, 0, 16, 535e-6),
    lambda: F.flux_density_peak_square(800, 1e5, 0, 535e-6),
    lambda: F.flux_density_peak_square(800, 1e5, 16.0, 535e-6),  # non-integer turns
    lambda: F.flux_density_peak_square(800, 1e5, 16, 535e-6, duty=0.7),
    lambda: F.flux_density_peak_square(float("nan"), 1e5, 16, 535e-6),
    lambda: F.dowell_ac_resistance_factor(0, 1e-4, 1e-4),
    lambda: F.dowell_ac_resistance_factor(2, 1e-4, 1e-4, porosity=1.5),
    lambda: F.copper_resistivity(-5),
    lambda: F.window_fill(10, 1e-6, 0),
    lambda: F.inductor_area_product(100e-6, 20, 14, 0.3, 4e6, 1.5),
    lambda: F.rms_of_square_current(10, duty=0.9),
])
def test_invalid_inputs_are_rejected_not_clipped(call):
    with pytest.raises(InvalidInputError):
        call()


def test_overflow_in_dowell_is_reported_not_saturated():
    with pytest.raises(InvalidInputError):
        F.dowell_ac_resistance_factor(1, 1.0, 1e-6)
