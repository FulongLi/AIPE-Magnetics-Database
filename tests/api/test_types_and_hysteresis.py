"""API input validation (units, rejections) and the hysteresis capability through the public API."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import aipe_magnetics.api as api
from aipe_magnetics.api.types import CoreLossRequest, TransformerDesignRequest, WaveformSpec


def test_core_loss_request_normalises_units_and_keeps_originals():
    r = CoreLossRequest(material="N87", operating_point={"frequency": {"value": 100, "unit": "kHz"}, "flux_density_peak": {"value": 200, "unit": "mT"},
                                                        "temperature": {"value": 100, "unit": "degC"}, "waveform": {"kind": "triangular"}})
    op = r.operating_point
    assert (op.frequency.value, op.frequency.unit) == (1e5, "Hz") and op.frequency.original.unit == "kHz"
    assert op.flux_density_peak.value == pytest.approx(0.2) and op.temperature.value == pytest.approx(373.15)
    assert op.waveform.duty == 0.5


@pytest.mark.parametrize("op", [
    {"frequency": 100000, "flux_density_peak": {"value": 0.2, "unit": "T"}},                       # bare number
    {"frequency": {"value": -1, "unit": "Hz"}, "flux_density_peak": {"value": 0.2, "unit": "T"}},   # negative
    {"frequency": {"value": 1e5, "unit": "T"}, "flux_density_peak": {"value": 0.2, "unit": "T"}},   # wrong dimension
    {"frequency": {"value": 1e5, "unit": "Hz"}, "flux_density_peak": {"value": 0.2, "unit": "furlong"}},
    {"frequency": {"value": 1e5, "unit": "Hz"}, "flux_density_peak": {"value": 0.2, "unit": "T"}, "temperature": {"value": -300, "unit": "degC"}},
    {"frequency": {"value": 1e5, "unit": "Hz"}, "flux_density_peak": {"value": 0.2, "unit": "T"}, "waveform": {"kind": "trapezoidal"}},
])
def test_invalid_operating_points_are_rejected(op):
    with pytest.raises((ValidationError, ValueError)):
        CoreLossRequest(material="N87", operating_point=op)


def test_waveform_spec_consistency():
    assert WaveformSpec(kind="trapezoidal", duty_rise=0.3, duty_flat=0.2).duty_flat == 0.2
    with pytest.raises(ValueError):
        WaveformSpec(kind="trapezoidal", duty_rise=0.4, duty_flat=0.2)
    with pytest.raises(ValueError):
        WaveformSpec(kind="sampled", samples=[0.0, 0.1])


def test_transformer_request_defaults():
    t = TransformerDesignRequest(power={"value": 10, "unit": "kW"}, frequency={"value": 100, "unit": "kHz"},
                                 primary={"name": "p", "voltage": {"value": 800, "unit": "V"}},
                                 secondaries=[{"name": "s", "voltage": {"value": 400, "unit": "V"}}], material="N87")
    assert t.power.value == 10000 and t.temperature.value == pytest.approx(373.15) and t.flux_limit.fraction == 0.7


def test_api_rejections_are_results_not_exceptions():
    r = api.estimate_core_loss(material="N87", frequency={"value": 1e5, "unit": "Hz"}, flux_density_peak=5)
    assert r.status == "rejected" and r.errors and r.errors[0].code == "invalid_input"
    json.dumps(r.to_json_dict())


def test_hysteresis_capability_returns_physical_loop():
    r = api.evaluate_hysteresis(field_peak={"value": 1000, "unit": "A/m"}, periods=6, samples_per_period=400)
    assert r.status == "ok", r.errors
    assert r.model is not None and r.model.id.endswith("jiles_atherton")
    assert r.values["loop_energy_density"].quantity.value > 0
    assert 0 < r.values["coercivity"].quantity.value < r.values["field_peak"].quantity.value
    assert 0 < r.values["remanence"].quantity.value < r.values["flux_density_peak"].quantity.value
    assert any(w.code == "illustrative_parameters" for w in r.warnings)
    assert any(a.id == "ja_illustrative_parameters" for a in r.assumptions)
    json.dumps(r.to_json_dict())


def test_hysteresis_rejects_bad_parameters_and_negative_field():
    assert api.evaluate_hysteresis(field_peak={"value": -5, "unit": "A/m"}).status == "rejected"
    r = api.evaluate_hysteresis(field_peak={"value": 100, "unit": "A/m"}, parameters={"Ms": 1e6})
    assert r.status == "rejected" and "parameters" in r.errors[0].message


def test_list_capabilities_and_verification_status_shapes():
    caps = api.list_capabilities()
    assert set(api.CAPABILITIES) <= set(caps.data["capabilities"])
    vs = api.verification_status("aipe.magnetics.model.steinmetz")
    assert vs.capability.endswith("verification_status") and "records" in vs.data
