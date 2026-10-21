"""Measurement requests: coverage assessment per category, priorities, test conditions, strict inputs."""

from __future__ import annotations

import pytest

import aipe_magnetics.api as api
from aipe_magnetics.engineering.measurement_requests import CATEGORIES, REQUEST_SCHEMA

KHZ100 = {"value": 100, "unit": "kHz"}
MT178 = {"value": 178, "unit": "mT"}
C100 = {"value": 100, "unit": "degC"}


def _by_material(result):
    return {m["material"]: m for m in result.data["requests"]}


def test_every_category_is_assessed_and_prioritised():
    r = api.measurement_requests(materials=["N87", "3C94", "N49"], frequency=KHZ100, flux_density_peak=MT178, temperature=C100)
    assert r.status == "ok" and r.data["schema"] == REQUEST_SCHEMA and r.data["summary"]["materials"] == 3
    rows = _by_material(r)
    for row in rows.values():
        assert set(row["coverage"]) == set(CATEGORIES)
        assert {q["category"] for q in row["requests"]} == set(CATEGORIES)
        priorities = [q["priority"] for q in row["requests"]]
        assert priorities == sorted(priorities, key=["high", "medium", "low"].index)
    n49 = {q["category"]: q for q in rows["material.tdk-n49"]["requests"]}
    assert n49["saturation_vs_temperature"]["priority"] == "high" and n49["saturation_vs_temperature"]["status_before"] == "missing"
    assert n49["core_loss_vs_f_b_t"]["priority"] == "high"  # fits end at 90 degC: the 100 degC design extrapolates
    c94 = {q["category"]: q for q in rows["material.ferroxcube-3c94"]["requests"]}
    assert c94["saturation_vs_temperature"]["status_before"] == "available" and c94["core_loss_vs_f_b_t"]["priority"] == "high"
    n87 = {q["category"]: q for q in rows["material.tdk-n87"]["requests"]}
    assert n87["core_loss_vs_f_b_t"]["priority"] == "medium"  # the 100 degC datasheet fit covers the design point; waveform validation remains
    assert n87["thermal_conductivity"]["status_before"] == "generic_class_value" and n87["thermal_conductivity"]["available"]["bounds_W_per_mK"] == [1.0, 5.0]


def test_test_conditions_follow_the_excitation():
    r = api.measurement_requests(materials=["N87"], frequency=KHZ100, flux_density_peak=MT178, temperature=C100,
                                 temperature_range=[{"value": 25, "unit": "degC"}, {"value": 120, "unit": "degC"}],
                                 dc_bias_field_max={"value": 100, "unit": "A/m"}, harmonics=7)
    requests = {q["category"]: q for q in _by_material(r)["material.tdk-n87"]["requests"]}
    loss = requests["core_loss_vs_f_b_t"]["test_conditions"]
    assert loss["frequencies_Hz"] == [50e3, 100e3, 200e3] and 100.0 in loss["temperatures_degC"] and 25.0 in loss["temperatures_degC"]
    assert pytest.approx(0.178, rel=1e-6) in loss["flux_density_peaks_T"] and max(loss["flux_density_peaks_T"]) <= 0.9 * 0.39
    assert "triangular" in loss["waveforms"] and "sinusoidal" in loss["waveforms"]
    assert requests["dc_bias"]["test_conditions"]["dc_bias_fields_A_per_m"][-1] == 100.0
    assert requests["winding_ac_resistance"]["test_conditions"]["frequencies_Hz"] == [100e3, 300e3, 500e3, 700e3]
    assert all(q["deliverable"] and q["rationale"] and q["record_kind"] for q in requests.values())


def test_available_categories_can_be_suppressed_and_inputs_are_strict():
    r = api.measurement_requests(materials=["N87"], frequency=KHZ100, include_available=False)
    categories = {q["category"] for q in _by_material(r)["material.tdk-n87"]["requests"]}
    assert "saturation_vs_temperature" not in categories and "core_loss_vs_f_b_t" in categories
    assert api.measurement_requests(frequency=KHZ100, temperature_range=[C100, {"value": 50, "unit": "degC"}]).status == "rejected"
    assert api.measurement_requests(frequency=KHZ100, materials=["unobtainium"]).status == "rejected"
    assert api.measurement_requests(frequency={"value": 100, "unit": "mT"}).status == "rejected"
    everything = api.measurement_requests(frequency=KHZ100)
    assert everything.status == "ok" and everything.data["summary"]["materials"] >= 10 and not any(m["synthetic"] for m in everything.data["requests"])
