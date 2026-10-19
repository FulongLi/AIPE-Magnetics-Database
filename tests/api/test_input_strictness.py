"""Typed-input contract: no lax coercion, engine constraints enforced at validation, rejections are results."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

import aipe_magnetics.api as api
from aipe_magnetics.api.types import DesignValidationRequest, HysteresisRequest, InductorDesignRequest, WaveformSpec
from aipe_magnetics.common.errors import InvalidInputError
from aipe_magnetics.common.quantity import Quantity, Uncertainty

KHZ100 = {"value": 100, "unit": "kHz"}
C100 = {"value": 100, "unit": "degC"}


@pytest.mark.parametrize("value", ["100", True, False, "1e5", [1], None])
def test_quantity_rejects_non_numeric_values(value):
    with pytest.raises((InvalidInputError, ValidationError, ValueError)):
        Quantity.parse({"value": value, "unit": "T"})


def test_quantity_tuple_parse_errors_are_invalid_input():
    with pytest.raises(InvalidInputError):
        Quantity.parse(("abc", "kHz"))
    with pytest.raises(InvalidInputError):
        Quantity.parse((True, "kHz"))
    assert Quantity.parse((100, "kHz")).value == pytest.approx(1e5)


def test_unknown_uncertainty_carries_no_numbers():
    with pytest.raises(ValidationError):
        Uncertainty(kind="unknown", value=3)
    assert Uncertainty().kind == "unknown"
    assert Uncertainty(kind="standard", value=0.1).value == 0.1


def test_integer_and_float_fields_are_strict():
    base = dict(material="N87", core="core.tdk-e65-32-27", frequency=KHZ100, voltage_peak={"value": 800, "unit": "V"})
    with pytest.raises(ValidationError):
        DesignValidationRequest(turns="14", **base)
    with pytest.raises(ValidationError):
        DesignValidationRequest(turns=14.0, **base)
    assert DesignValidationRequest(turns=14, **base).turns == 14
    with pytest.raises(ValidationError):
        HysteresisRequest(field_peak={"value": 500, "unit": "A/m"}, periods=4.0)
    with pytest.raises(ValidationError):
        HysteresisRequest(field_peak={"value": 500, "unit": "A/m"}, periods="4")
    with pytest.raises(ValidationError):
        HysteresisRequest(field_peak={"value": 500, "unit": "A/m"}, parameters={"Ms": "1e6", "a": 1, "alpha": 1, "k": 1, "c": 1})
    with pytest.raises(ValidationError):
        InductorDesignRequest(inductance={"value": 1, "unit": "mH"}, current_peak={"value": 2, "unit": "A"}, current_rms={"value": 1, "unit": "A"},
                              frequency=KHZ100, material="N87", max_candidates="5")
    # an integer for a float field is still fine (0 for duty is caught by the bound, not by the type)
    assert WaveformSpec(kind="triangular", duty=1 / 2).duty == 0.5


def test_trapezoid_follows_the_engine_constraint():
    derived = WaveformSpec(kind="trapezoidal", duty_rise=0.25)
    assert derived.duty_flat == pytest.approx(0.25)
    assert WaveformSpec(kind="trapezoidal", duty_rise=0.3, duty_flat=0.2).duty_flat == 0.2
    with pytest.raises(ValidationError, match="2\\*\\(duty_rise \\+ duty_flat\\) == 1"):
        WaveformSpec(kind="trapezoidal", duty_rise=0.25, duty_flat=0.0)
    with pytest.raises(ValidationError):
        WaveformSpec(kind="trapezoidal", duty_flat=0.2)
    r = api.estimate_core_loss(material="N87", frequency=KHZ100, flux_density_peak={"value": 150, "unit": "mT"}, temperature=C100,
                               waveform={"kind": "trapezoidal", "duty_rise": 0.25})
    assert r.status == "ok", r.errors


def test_sampled_waveform_conventions_are_explicit():
    half_open = [0.15 * math.sin(2 * math.pi * i / 64) for i in range(64)]
    rejected = api.estimate_core_loss(material="N87", frequency=KHZ100, flux_density_peak={"value": 150, "unit": "mT"}, temperature=C100,
                                      waveform={"kind": "sampled", "samples": half_open}, model="igse")
    assert rejected.status == "rejected" and "does not close" in rejected.errors[0].message
    accepted = api.estimate_core_loss(material="N87", frequency=KHZ100, flux_density_peak={"value": 150, "unit": "mT"}, temperature=C100,
                                      waveform={"kind": "sampled", "samples": half_open, "endpoint_included": False}, model="igse")
    assert accepted.status == "ok", accepted.errors
    sine = api.estimate_core_loss(material="N87", frequency=KHZ100, flux_density_peak={"value": 150, "unit": "mT"}, temperature=C100, model="igse")
    assert accepted.values["loss_density"].quantity.value == pytest.approx(sine.values["loss_density"].quantity.value, rel=2e-2)


def test_zero_flux_density_is_rejected_at_validation():
    r = api.estimate_core_loss(material="N87", frequency=KHZ100, flux_density_peak={"value": 0, "unit": "T"}, temperature=C100,
                               waveform={"kind": "triangular"})
    assert r.status == "rejected" and r.errors[0].code == "invalid_input" and "flux_density_peak must be positive" in r.errors[0].message


def test_api_type_errors_become_rejected_results():
    assert api.verification_status("aipe.magnetics.model.igse", frequency=5).status == "rejected"
    wrong = api.verification_status("aipe.magnetics.model.igse", frequency={"value": 1, "unit": "T"})
    assert wrong.status == "rejected" and "dimension" in wrong.errors[0].message
    assert api.compare_materials(["material.tdk-n87"], ("abc", "kHz")).status == "rejected"
    assert api.compare_materials(123).status == "rejected"
    assert api.list_materials(123).status == "rejected"
    assert api.list_materials(material_class=["x"]).status == "rejected"


def test_verification_status_without_capability_reports_the_registry_state():
    vs = api.verification_status()
    assert vs.status == "ok" and "evidence_file_exists" in vs.data
    if not vs.data["evidence_file_exists"]:
        assert vs.verification.state == "unverified" and vs.verification.level == "L0"
        assert any(w.code == "bundled_evidence_missing" for w in vs.warnings)
    else:
        assert vs.data["record_count"] > 0 and vs.verification.state == "verified"
