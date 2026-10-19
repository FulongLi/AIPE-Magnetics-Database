"""Winding sizing / copper loss and saturation screening."""

from __future__ import annotations

import pytest

from aipe_magnetics.api.types import FluxLimitSpec, WindingSpec
from aipe_magnetics.common.errors import InvalidInputError, OutOfDomainError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.data.records import MaterialProperties
from aipe_magnetics.engineering.saturation import screen_saturation
from aipe_magnetics.engineering.winding import design_winding


@pytest.fixture
def material():
    return MaterialProperties.model_validate({
        "saturation_flux_density": [
            {"value": {"value": 0.49, "unit": "T"}, "temperature": {"value": 298.15, "unit": "K"}, "field": {"value": 1200, "unit": "A/m"}},
            {"value": {"value": 0.39, "unit": "T"}, "temperature": {"value": 373.15, "unit": "K"}, "field": {"value": 1200, "unit": "A/m"}},
        ]})


@pytest.mark.parametrize("conductor", ["round", "litz", "foil"])
def test_winding_design_is_consistent(conductor):
    d = design_winding(turns=12, current_rms=12.5, frequency=1e5, mean_turn_length=0.15, spec=WindingSpec(conductor_type=conductor),
                       temperature=373.15, window_area=394e-6)
    # DC resistance = rho * N * MLT / A_cu
    assert d.resistance_dc == pytest.approx(d.resistivity * 12 * 0.15 / d.conductor_area, rel=1e-9)
    assert d.resistance_ac == pytest.approx(d.resistance_dc * d.ac_resistance_factor)
    assert d.copper_loss == pytest.approx(12.5 ** 2 * d.resistance_ac)
    assert d.ac_resistance_factor >= 1.0
    assert d.window_fill == pytest.approx(12 * d.conductor_area / 394e-6)
    assert any(a.id.startswith("current_density") for a in d.assumptions)
    assert any(lim.code == "dowell_1d" for lim in d.limitations)


def test_round_wire_at_100khz_is_flagged_and_litz_is_better():
    r = design_winding(turns=12, current_rms=12.5, frequency=1e5, mean_turn_length=0.15, spec=WindingSpec(conductor_type="round"), temperature=373.15)
    l = design_winding(turns=12, current_rms=12.5, frequency=1e5, mean_turn_length=0.15, spec=WindingSpec(conductor_type="litz"), temperature=373.15)
    assert r.ac_resistance_factor > l.ac_resistance_factor
    assert any(w.code == "high_ac_resistance_factor" for w in r.warnings)


def test_user_current_density_and_layers_are_honoured():
    d = design_winding(turns=10, current_rms=10, frequency=5e4, mean_turn_length=0.1,
                       spec=WindingSpec(conductor_type="round", current_density=Quantity(value=2, unit="A/mm^2"), layers=2), temperature=313.15)
    assert d.conductor_area == pytest.approx(10 / 2e6)
    assert d.layers == 2
    assert any(a.id == "current_density_user" for a in d.assumptions)


@pytest.mark.parametrize("kwargs", [dict(turns=0), dict(current_rms=-1), dict(frequency=0), dict(mean_turn_length=0)])
def test_winding_rejects_invalid_inputs(kwargs):
    base = dict(turns=10, current_rms=10, frequency=1e5, mean_turn_length=0.1, spec=WindingSpec(), temperature=300.0)
    base.update(kwargs)
    with pytest.raises(InvalidInputError):
        design_winding(**base)


def test_saturation_screening_lookup_interpolation_and_refused_extrapolation(material):
    s = screen_saturation(flux_density_peak=0.2, temperature=373.15, material=material)
    assert s.saturation_flux_density == pytest.approx(0.39) and s.saturation_status == "looked_up"
    assert s.flux_limit == pytest.approx(0.7 * 0.39) and s.within_limit and not s.saturated
    mid = screen_saturation(flux_density_peak=0.2, temperature=335.65, material=material)  # 62.5 degC, half way
    assert mid.saturation_flux_density == pytest.approx(0.44) and mid.saturation_status == "estimated"
    assert any(a.id == "bsat_interpolated" for a in mid.assumptions)
    with pytest.raises(OutOfDomainError):
        screen_saturation(flux_density_peak=0.2, temperature=423.15, material=material)


def test_saturation_policy_and_flags(material):
    hot = screen_saturation(flux_density_peak=0.3, temperature=373.15, material=material)
    assert not hot.within_limit and not hot.saturated and any(w.code == "flux_limit_exceeded" for w in hot.warnings)
    sat = screen_saturation(flux_density_peak=0.4, temperature=373.15, material=material)
    assert sat.saturated and any(w.code == "saturated" and w.severity == "error" for w in sat.warnings)
    absolute = screen_saturation(flux_density_peak=0.3, temperature=373.15, material=material,
                                 policy=FluxLimitSpec(mode="absolute", absolute=Quantity(value=320, unit="mT")))
    assert absolute.flux_limit == pytest.approx(0.32) and absolute.within_limit
