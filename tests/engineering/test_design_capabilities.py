"""Engineering capabilities on the bundled catalogue (N87 + TDK cores): core loss, transformer, inductor, validation, report."""

from __future__ import annotations

import json
import math

import pytest

import aipe_magnetics.api as api
from aipe_magnetics.engineering import formulas as F

N87 = "N87"
E65 = "core.tdk-e65-32-27"
KHZ100 = {"value": 100, "unit": "kHz"}
C100 = {"value": 100, "unit": "degC"}
C90 = {"value": 90, "unit": "degC"}


def test_catalogue_contains_the_demonstration_material_and_cores():
    listing = api.list_materials()
    ids = {m["id"] for m in listing.data["materials"]}
    assert "material.tdk-n87" in ids and len(ids) >= 10
    n87 = api.get_material(N87)
    assert n87.status == "ok"
    summary = n87.data["material"]
    assert summary["grade"] == "N87" and summary["manufacturer_ref"] == "manufacturer.tdk"
    assert any(s["basis"] == "datasheet_fit" for s in summary["steinmetz_parameter_sets"])
    assert any(s["basis"] == "measurement_fit" for s in summary["steinmetz_parameter_sets"])
    assert all(e["review_status"] == "unreviewed" for e in summary["evidence"])
    assert any(e["kind"] == "source" and "tdk" in e["id"] for e in summary["evidence"])


def test_steinmetz_estimate_reproduces_tdk_datasheet_points_within_fit_error():
    # TDK N87 datasheet typical values at 100 degC (sinusoidal); the 4-point fit residuals are <= 4.7 %
    points = [(25e3, 0.2, 57e3), (100e3, 0.2, 375e3), (300e3, 0.1, 390e3), (500e3, 0.05, 215e3)]
    for f, b, pv in points:
        r = api.estimate_core_loss(material=N87, frequency={"value": f, "unit": "Hz"}, flux_density_peak={"value": b, "unit": "T"},
                                   temperature=C100, model="steinmetz")
        assert r.status == "ok", r.errors
        assert r.model.id == "aipe.magnetics.model.steinmetz"
        assert r.values["loss_density"].quantity.value == pytest.approx(pv, rel=0.06)
        assert r.data["parameter_set"]["id"] == "set.tdk-datasheet-fit-n87-100c"
        assert any(a.id == "parameter_set_selection" for a in r.assumptions)
        assert any(s.id == "evidence.tdk-n87-datasheet-2025-06" for s in r.sources)


def test_igse_triangular_matches_closed_form_and_sinusoid_matches_steinmetz():
    sine = api.estimate_core_loss(material=N87, frequency=KHZ100, flux_density_peak={"value": 150, "unit": "mT"}, temperature=C100, model="steinmetz")
    igse_sine = api.estimate_core_loss(material=N87, frequency=KHZ100, flux_density_peak={"value": 150, "unit": "mT"}, temperature=C100, model="igse")
    assert igse_sine.values["loss_density"].quantity.value == pytest.approx(sine.values["loss_density"].quantity.value, rel=1e-6)
    tri = api.estimate_core_loss(material=N87, frequency=KHZ100, flux_density_peak={"value": 150, "unit": "mT"}, temperature=C100,
                                 waveform={"kind": "triangular", "duty": 0.5}, model="auto")
    assert tri.status == "ok" and tri.model.id == "aipe.magnetics.model.igse"
    ps = tri.data["parameter_set"]
    k, alpha, beta = ps["k"]["value"], ps["alpha"], ps["beta"]
    from scipy.integrate import quad
    integral = quad(lambda th: abs(math.cos(th)) ** alpha, 0, 2 * math.pi, points=[math.pi / 2, math.pi, 3 * math.pi / 2])[0]
    k_i = k / ((2 * math.pi) ** (alpha - 1) * integral * 2 ** (beta - alpha))
    expected = k_i * 2 ** (alpha + beta) * 0.15 ** beta * 1e5 ** alpha  # D = 0.5 closed form with B_pk
    assert tri.values["loss_density"].quantity.value == pytest.approx(expected, rel=1e-6)


def test_out_of_range_temperature_is_rejected_not_extrapolated():
    r = api.estimate_core_loss(material=N87, frequency=KHZ100, flux_density_peak={"value": 150, "unit": "mT"}, temperature={"value": 150, "unit": "degC"})
    assert r.status == "rejected" and r.errors[0].code in {"data_not_found", "out_of_domain"}
    r2 = api.estimate_core_loss(material=N87, frequency=KHZ100, flux_density_peak={"value": 150, "unit": "mT"})
    assert r2.status == "rejected" and "temperature" in r2.errors[0].message


def test_core_loss_with_core_volume_and_verification_block():
    r = api.estimate_core_loss(material=N87, frequency=KHZ100, flux_density_peak={"value": 100, "unit": "mT"}, temperature=C100, core=E65)
    assert r.status == "ok"
    assert r.values["core_loss"].quantity.value == pytest.approx(r.values["loss_density"].quantity.value * 78650e-9, rel=1e-9)
    assert r.verification.level in {"L0", "L1", "L2"} and r.verification.state in {"verified", "partially_verified", "out_of_range", "unverified"}
    json.dumps(r.to_json_dict())


def test_transformer_design_dab_10kw_is_consistent():
    r = api.design_transformer(power={"value": 10, "unit": "kW"}, frequency=KHZ100, primary={"name": "primary", "voltage": {"value": 800, "unit": "V"},
                               "winding": {"conductor_type": "litz"}}, secondaries=[{"name": "secondary", "voltage": {"value": 400, "unit": "V"},
                               "winding": {"conductor_type": "litz"}}], material=N87, temperature=C100,
                               loss_density_target={"value": 300, "unit": "kW/m^3"})
    assert r.status == "ok", r.errors
    n1 = r.values["turns_primary"].quantity.value
    n2 = r.values["turns_secondary"].quantity.value
    assert float(n1).is_integer() and float(n2).is_integer() and n1 >= n2
    core = next(c for c in r.data["candidates"] if c["core"] == r.data["selected"])
    b_pk = F.flux_density_peak_square(800, 1e5, int(n1), core["effective_area_m2"])
    assert r.values["flux_density_peak"].quantity.value == pytest.approx(b_pk, rel=1e-12)
    assert b_pk <= r.values["flux_density_allowed"].quantity.value * (1 + 1e-9)
    assert r.values["loss_density"].quantity.value <= 300e3 * 1.001
    assert r.values["saturation_margin"].quantity.value > 0.3
    assert r.values["total_loss"].quantity.value == pytest.approx(r.values["core_loss"].quantity.value + r.values["copper_loss"].quantity.value)
    assert {"flux_limit_fraction", "loss_density_target", "winding_currents"} <= {a.id for a in r.assumptions}
    assert {"analytical_design", "no_hardware_validation"} <= {lim.code for lim in r.limitations}
    assert r.values["current_rms_primary"].status == "assumed"
    assert all("feasible" in c for c in r.data["candidates"])


def test_transformer_design_rejects_impossible_specification():
    # 1 MW through a single ETD59 at 20 kHz cannot satisfy window and loss constraints
    r = api.design_transformer(power={"value": 1, "unit": "MW"}, frequency={"value": 20, "unit": "kHz"}, primary={"name": "p", "voltage": {"value": 800, "unit": "V"}},
                               secondaries=[{"name": "s", "voltage": {"value": 400, "unit": "V"}}], material=N87, cores=["core.tdk-etd59-31-22"], temperature=C100)
    assert r.status == "rejected" and r.errors[0].code == "no_feasible_candidate"
    assert r.data["candidates"][0]["feasible"] is False and r.data["candidates"][0]["reasons"]


def test_inductor_design_is_physically_consistent():
    r = api.design_inductor(inductance={"value": 100, "unit": "uH"}, current_peak={"value": 25, "unit": "A"}, current_rms={"value": 18, "unit": "A"},
                            current_ripple_peak_to_peak={"value": 5, "unit": "A"}, frequency=KHZ100, material=N87, core=E65, temperature=C100,
                            winding={"conductor_type": "litz"})
    assert r.status == "ok", r.errors
    n = int(r.values["turns"].quantity.value)
    gap = r.values["air_gap"].quantity.value
    assert gap >= 0
    assert F.gapped_inductance(n, 535e-6, gap, 147e-3, 2200) == pytest.approx(100e-6, rel=1e-9)
    assert r.values["flux_density_peak"].quantity.value == pytest.approx(100e-6 * 25 / (n * 535e-6), rel=1e-12)
    assert r.values["flux_density_peak"].quantity.value <= r.values["flux_density_allowed"].quantity.value * (1 + 1e-9)
    assert {"ungapped_permeability", "no_fringing"} <= {a.id for a in r.assumptions}
    assert any(lim.code == "dc_bias_ignored_in_loss" for lim in r.limitations)


def test_validate_design_flags_saturation_and_passes_sane_design():
    bad = api.validate_design(material=N87, core=E65, turns=4, frequency=KHZ100, temperature=C100, voltage_peak={"value": 800, "unit": "V"})
    assert bad.status == "rejected" and "saturation" in bad.data["failed_checks"]
    good = api.validate_design(material=N87, core=E65, turns=20, frequency=KHZ100, temperature=C100, voltage_peak={"value": 800, "unit": "V"})
    assert good.status == "ok" and good.data["passed"]
    assert {c["id"] for c in good.data["checks"]} >= {"curie_temperature", "frequency_range", "saturation", "loss_parameters"}


def test_compare_materials_ranks_and_reports_missing_data():
    r = api.compare_materials(["N87", "3C94", "N49"], frequency=KHZ100, flux_density_peak={"value": 100, "unit": "mT"}, temperature=C90)
    assert r.status == "ok"
    rows = {row["id"]: row for row in r.data["comparison"]}
    assert rows["material.tdk-n87"]["loss_density_W_per_m3"] > 0 and rows["material.ferroxcube-3c94"]["loss_density_W_per_m3"] > 0
    assert rows["material.tdk-n87"]["saturation_margin"] is not None
    assert rows["material.ferroxcube-3c94"]["saturation_flux_density_T"] is None  # no datasheet Bsat transcribed yet -> reported, not guessed
    assert any("saturation" in note for note in rows["material.ferroxcube-3c94"]["notes"])


def test_report_renders_markdown_with_status_separation():
    loss = api.estimate_core_loss(material=N87, frequency=KHZ100, flux_density_peak={"value": 100, "unit": "mT"}, temperature=C100, core=E65)
    rep = api.generate_report([loss], title="Test report")
    text = rep.data["content"]
    assert text.startswith("# Test report") and "estimated" in text and "Assumptions" in text and "Sources" in text and "Limitations" in text
