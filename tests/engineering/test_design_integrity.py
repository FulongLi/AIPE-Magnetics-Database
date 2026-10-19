"""Design-flow integrity: feasibility semantics, warning propagation, power sharing, assumptions, validation verdicts, reports.

Regression tests for the v0.2 review findings on the engineering layer. Everything runs on the bundled catalogue.
"""

from __future__ import annotations

import math

import pytest

import aipe_magnetics.api as api
from aipe_magnetics.common.quantity import Quantity, Uncertainty
from aipe_magnetics.common.results import EngineeringResult
from aipe_magnetics.verification.evidence import EvidenceRecord, OperatingRange, VerificationRegistry

N87 = "N87"
E65 = "core.tdk-e65-32-27"
UU93 = "core.tdk-uu93-152-30"
KHZ100 = {"value": 100, "unit": "kHz"}
C100 = {"value": 100, "unit": "degC"}
LITZ = {"conductor_type": "litz", "strand_diameter": {"value": 0.1, "unit": "mm"}, "current_density": {"value": 4, "unit": "A/mm^2"},
        "fill_factor": 0.4}


def _spec(**over):
    spec = {"power": {"value": 10, "unit": "kW"}, "frequency": KHZ100,
            "primary": {"name": "primary", "voltage": {"value": 800, "unit": "V"}, "winding": LITZ},
            "secondaries": [{"name": "secondary", "voltage": {"value": 400, "unit": "V"}, "winding": LITZ}],
            "material": N87, "temperature": C100, "loss_density_target": {"value": 300, "unit": "kW/m^3"}, "max_candidates": 10}
    spec.update(over)
    return spec


# -------------------------------------------------------------------------------------------------
# transformer: incomplete evaluations are never feasible or ranked
# -------------------------------------------------------------------------------------------------
def test_core_without_mean_turn_length_is_not_evaluable_and_never_selected():
    r = api.design_transformer(_spec())
    assert r.status == "ok"
    rows = {c["core"]: c for c in r.data["candidates"]}
    uu = rows[UU93]
    assert uu["feasible"] is False and uu["evaluation"] == "incomplete" and uu["total_loss_W"] is None and uu["copper_loss_W"] is None
    assert any("not evaluable" in reason and "mean turn length" in reason for reason in uu["reasons"])
    assert r.data["selected"] != UU93
    for c in r.data["candidates"]:
        if c["feasible"]:
            assert c["evaluation"] == "complete" and c["copper_loss_W"] is not None and c["window_fill"] is not None
    # a complete total is never compared with a core-loss-only total: at 30 kW the UU core used to win on core loss alone
    big = api.design_transformer(_spec(power={"value": 30, "unit": "kW"}))
    assert big.status == "ok" and big.data["selected"] == "core.tdk-pm87-70"
    assert big.values["total_loss"].quantity.value == pytest.approx(big.values["core_loss"].quantity.value + big.values["copper_loss"].quantity.value)
    assert {w.code for w in big.warnings} >= {"high_ac_resistance_factor"}  # selected-candidate warnings reach the top level


def test_only_the_unevaluable_core_requested_gives_an_explicit_rejection():
    r = api.design_transformer(_spec(cores=[UU93]))
    assert r.status == "rejected" and r.errors[0].code == "no_feasible_candidate"
    assert "mean turn length" in r.errors[0].message
    assert r.errors[0].details["reasons"][UU93]


def test_winding_model_failure_makes_the_candidate_infeasible():
    r = api.design_transformer(_spec(frequency={"value": 1e12, "unit": "Hz"}))
    assert r.status == "rejected" and r.errors[0].code == "no_feasible_candidate"
    assert "Dowell penetration ratio" in r.errors[0].message
    for c in r.data["candidates"]:
        assert c["feasible"] is False and c["copper_loss_W"] is None and c["total_loss_W"] is None
    assert {w.code for w in r.warnings} >= {"frequency_range", "loss_parameters"}


def test_selected_candidate_warnings_are_promoted_to_the_result():
    # 800 V : 250 V on E65 -> N2 = round(21 * 0.3125) = 7, realised ratio deviates +6.7 % from the voltage ratio
    r = api.design_transformer(_spec(secondaries=[{"name": "secondary", "voltage": {"value": 250, "unit": "V"}, "winding": LITZ}], cores=[E65]))
    assert r.status == "ok"
    assert any(w.code == "turns_ratio_deviation" for w in r.warnings), [w.code for w in r.warnings]
    hot = dict(LITZ, conductor_temperature={"value": 1000, "unit": "degC"})
    r = api.design_transformer(_spec(primary={"name": "primary", "voltage": {"value": 800, "unit": "V"}, "winding": hot}, cores=[E65]))
    assert r.status == "ok" and any(w.code == "resistivity_model_extrapolated" for w in r.warnings)
    assert any("validity -50..200 degC" in a.statement for a in r.assumptions if a.id == "copper_resistivity_model")


def test_foil_wider_than_the_window_is_infeasible():
    foil = {"conductor_type": "foil"}
    r = api.design_transformer(_spec(primary={"name": "primary", "voltage": {"value": 800, "unit": "V"}, "winding": foil},
                                     secondaries=[{"name": "secondary", "voltage": {"value": 400, "unit": "V"}, "winding": foil}], cores=[E65]))
    assert r.status == "rejected" and "foil width" in r.errors[0].message


def test_two_secondaries_share_the_throughput_power():
    secondaries = [{"name": "s1", "voltage": {"value": 400, "unit": "V"}, "winding": LITZ},
                   {"name": "s2", "voltage": {"value": 400, "unit": "V"}, "winding": LITZ}]
    r = api.design_transformer(_spec(secondaries=secondaries, cores=[E65]))
    assert r.status == "ok", r.errors
    currents = r.data["winding_currents"]
    assert currents["primary"]["current_rms_A"] == pytest.approx(12.5)
    assert currents["s1"]["current_rms_A"] == pytest.approx(12.5) and currents["s2"]["current_rms_A"] == pytest.approx(12.5)
    assert currents["s1"]["power_share"] == pytest.approx(0.5)
    # sum of secondary VA equals the throughput power
    assert sum(currents[n]["current_rms_A"] * 400 for n in ("s1", "s2")) == pytest.approx(10e3)
    assert any(a.id == "winding_currents" and "sharing" in a.statement for a in r.assumptions)
    # explicit, unequal shares
    secondaries[0]["power_share"], secondaries[1]["power_share"] = 0.75, 0.25
    r2 = api.design_transformer(_spec(secondaries=secondaries, cores=[E65]))
    assert r2.status == "ok" and r2.data["winding_currents"]["s1"]["current_rms_A"] == pytest.approx(18.75)
    # inconsistent shares are rejected before any calculation
    secondaries[1]["power_share"] = 0.5
    bad = api.design_transformer(_spec(secondaries=secondaries, cores=[E65]))
    assert bad.status == "rejected" and "sum to 1" in bad.errors[0].message


def test_design_assumptions_and_labels_are_recorded_honestly():
    r = api.design_transformer(_spec(cores=[E65]))
    ids = {a.id for a in r.assumptions}
    assert {"zero_dc_bias", "winding_currents", "flux_limit_fraction", "loss_density_target"} <= ids
    assert r.values["current_rms_primary"].status == "assumed"
    spec = _spec(cores=[E65])
    spec["primary"] = dict(spec["primary"], current_rms={"value": 14, "unit": "A"})
    r2 = api.design_transformer(spec)
    assert r2.values["current_rms_primary"].status == "specified"
    trap = api.design_transformer(_spec(voltage_waveform="trapezoidal", cores=[E65]))
    assert any(a.id == "trapezoidal_voltage_as_square" for a in trap.assumptions)
    # the verification block names both ingredients (sizing relations and the loss model) and is the weaker of the two
    assert "Sizing relations" in r.verification.statement and "Core-loss model" in r.verification.statement
    assert "weaker of the two" in r.verification.statement
    assert any(lim.code == "candidate_enumeration" for lim in r.limitations)
    assert r.model is not None and r.model.id == "aipe.magnetics.model.igse"  # identity of the model behind the estimated losses


def test_no_feasible_candidate_names_the_distinct_reasons():
    r = api.design_transformer(_spec(temperature={"value": 60, "unit": "degC"}))
    assert r.status == "rejected"
    assert "no Steinmetz parameter set is valid at 333.15 K" in r.errors[0].message
    assert set(r.errors[0].details["reasons"]) == {c["core"] for c in r.data["candidates"]}


def test_operating_temperature_at_or_above_curie_is_rejected():
    r = api.design_transformer(_spec(temperature={"value": 215, "unit": "degC"}, cores=[E65]))
    assert r.status == "rejected" and "Curie" in r.errors[0].message


# -------------------------------------------------------------------------------------------------
# inductor
# -------------------------------------------------------------------------------------------------
def _inductor(**over):
    spec = dict(inductance={"value": 100, "unit": "uH"}, current_peak={"value": 25, "unit": "A"}, current_rms={"value": 18, "unit": "A"},
                current_ripple_peak_to_peak={"value": 5, "unit": "A"}, frequency=KHZ100, material=N87, core=E65, temperature=C100,
                winding={"conductor_type": "litz"})
    spec.update(over)
    return api.design_inductor(**spec)


def test_inductor_ripple_waveform_is_an_explicit_controllable_assumption():
    sym = _inductor()
    assert sym.status == "ok", sym.errors
    assert any(a.id == "ripple_waveform_triangular" and a.basis == "default" for a in sym.assumptions)
    assert sym.data["flux_waveform"] == {"kind": "triangular", "duty": 0.5}
    asym = _inductor(ripple_duty=0.2)
    assert asym.status == "ok"
    assert asym.data["flux_waveform"]["duty"] == 0.2
    assert any(a.id == "ripple_waveform_triangular" and a.basis == "user_input" for a in asym.assumptions)
    # iGSE triangular factor D^(1-a) + (1-D)^(1-a) grows for D != 0.5
    ps = asym.data["candidates"][0]
    pv_sym, pv_asym = sym.values["loss_density"].quantity.value, asym.values["loss_density"].quantity.value
    assert pv_asym > pv_sym * 1.05
    assert ps["flux_density_ac_amplitude_T"] == pytest.approx(sym.data["candidates"][0]["flux_density_ac_amplitude_T"])


def test_inductor_ripple_larger_than_twice_the_peak_is_rejected():
    r = _inductor(current_peak={"value": 20, "unit": "A"}, current_rms={"value": 15, "unit": "A"}, current_ripple_peak_to_peak={"value": 100, "unit": "A"})
    assert r.status == "rejected" and "2 x current_peak" in r.errors[0].message


def test_inductor_without_mean_turn_length_is_not_feasible():
    r = _inductor(core=UU93)
    assert r.status == "rejected" and r.errors[0].code == "no_feasible_candidate"
    assert r.data["candidates"][0]["evaluation"] == "incomplete"
    assert "mean turn length" in r.errors[0].message


# -------------------------------------------------------------------------------------------------
# Dowell porosity convention (round wire)
# -------------------------------------------------------------------------------------------------
def test_round_wire_porosity_follows_dowell_layer_transformation():
    from aipe_magnetics.api.types import WindingSpec
    from aipe_magnetics.engineering.winding import INSULATION_BUILD_FACTOR, design_winding

    d = design_winding(turns=10, current_rms=10, frequency=1e5, mean_turn_length=0.1, spec=WindingSpec(conductor_type="round"), temperature=373.15)
    diameter = d.conductor_diameter
    pitch = diameter * INSULATION_BUILD_FACTOR
    expected = (math.pi / 4) ** 0.75 * (diameter / d.skin_depth) * math.sqrt(diameter / pitch)
    assert d.dowell_penetration_ratio == pytest.approx(expected, rel=1e-12)


# -------------------------------------------------------------------------------------------------
# validation verdicts
# -------------------------------------------------------------------------------------------------
def test_validate_design_is_inconclusive_when_a_check_cannot_be_evaluated():
    r = api.validate_design(material=N87, core=E65, turns=21, frequency=KHZ100, temperature={"value": 150, "unit": "degC"},
                            voltage_peak={"value": 800, "unit": "V"})
    assert r.status == "ok" and r.data["passed"] is None
    assert "saturation" in r.data["inconclusive_checks"]
    assert any(w.code == "validation_inconclusive" for w in r.warnings)
    good = api.validate_design(material=N87, core=E65, turns=20, frequency=KHZ100, temperature=C100, voltage_peak={"value": 800, "unit": "V"})
    assert good.data["passed"] is True and good.data["inconclusive_checks"] == []
    specified = api.validate_design(material=N87, core=E65, turns=20, frequency=KHZ100, temperature=C100, flux_density_peak={"value": 100, "unit": "mT"})
    assert specified.values["flux_density_peak"].status == "specified"


def test_curie_check_repeats_the_record_note_about_the_datasheet_bound():
    r = api.validate_design(material=N87, core=E65, turns=20, frequency=KHZ100, temperature=C100, voltage_peak={"value": 800, "unit": "V"})
    curie = next(c for c in r.data["checks"] if c["id"] == "curie_temperature")
    assert "lower bound" in curie["message"]
    material = api.get_material(N87)
    assert material.values["curie_temperature"].note and "lower bound" in material.values["curie_temperature"].note


# -------------------------------------------------------------------------------------------------
# compare_materials verification comes from the registry, Bertotti works
# -------------------------------------------------------------------------------------------------
def _registry_with(level: str, status: str = "pass") -> VerificationRegistry:
    from aipe_magnetics.models.contract import ValidityRange
    rec = EvidenceRecord(id="check:test-steinmetz", capability="aipe.magnetics.model.steinmetz", level=level, kind="analytical_reference",
                         status=status, description="test", method="test", check_id="test-steinmetz", tool_version="0.2.0",
                         range=OperatingRange(frequency=ValidityRange(min=1e3, max=1e6, unit="Hz")))
    return VerificationRegistry([rec])


def test_compare_materials_verification_is_derived_from_evidence_not_asserted():
    from aipe_magnetics.engineering.materials import compare_materials

    f, b, t = Quantity(value=1e5, unit="Hz"), Quantity(value=0.1, unit="T"), Quantity(value=373.15, unit="K")
    verified = compare_materials(["N87", "3C94"], f, b, t, registry=_registry_with("L2"))
    assert verified.verification.level == "L2" and verified.verification.state == "verified"
    rows = {row["id"]: row for row in verified.data["comparison"]}
    assert rows["material.tdk-n87"]["loss_verification"]["state"] == "verified"
    failed = compare_materials(["N87"], f, b, t, registry=_registry_with("L2", status="fail"))
    assert failed.verification.state == "failed"
    empty = compare_materials(["N87"], f, b, t, registry=VerificationRegistry())
    assert empty.verification.level == "L0" and empty.verification.state == "unverified"
    public = api.compare_materials(["N87"], frequency=KHZ100, flux_density_peak={"value": 100, "unit": "mT"}, temperature=C100)
    loss = api.estimate_core_loss(material=N87, frequency=KHZ100, flux_density_peak={"value": 100, "unit": "mT"}, temperature=C100)
    assert (public.verification.level, public.verification.state) == (loss.verification.level, loss.verification.state)
    no_point = api.compare_materials(["N87", "3C94"])
    assert no_point.verification.level == "L1"


def test_bertotti_with_user_coefficients_returns_a_result():
    r = api.estimate_core_loss(material=N87, frequency=KHZ100, flux_density_peak={"value": 150, "unit": "mT"}, temperature=C100,
                               model="bertotti", model_parameters={"k_h": 10, "beta_h": 2, "k_c": 1e-4, "k_e": 1e-6})
    assert r.status == "ok", r.errors
    f, b = 1e5, 0.15
    assert r.values["loss_density"].quantity.value == pytest.approx(10 * f * b ** 2 + 1e-4 * f ** 2 * b ** 2 + 1e-6 * f ** 1.5 * b ** 1.5, rel=1e-12)
    assert r.model.id == "aipe.magnetics.model.bertotti"
    bad = api.estimate_core_loss(material=N87, frequency=KHZ100, flux_density_peak={"value": 150, "unit": "mT"}, temperature=C100,
                                 model="bertotti", model_parameters={"k_h": 10, "beta_h": 2, "k_c": 1e-4})
    assert bad.status == "rejected" and "k_e" in bad.errors[0].message
    none = api.estimate_core_loss(material=N87, frequency=KHZ100, flux_density_peak={"value": 150, "unit": "mT"}, temperature=C100, model="bertotti")
    assert none.status == "rejected" and none.errors[0].code == "unsupported_operation"


# -------------------------------------------------------------------------------------------------
# report rendering
# -------------------------------------------------------------------------------------------------
def test_report_renders_comparison_rows_check_verdicts_uncertainty_and_escapes_pipes():
    compare = api.compare_materials(["N87"], frequency=KHZ100, flux_density_peak={"value": 100, "unit": "mT"}, temperature=C100)
    validation = api.validate_design(material=N87, core=E65, turns=14, frequency=KHZ100, temperature=C100, voltage_peak={"value": 800, "unit": "V"})
    synthetic = EngineeringResult(capability="aipe.magnetics.test")
    synthetic.add_value("pv", "Loss | density", Quantity(value=0.123456789, unit="T", uncertainty=Uncertainty(kind="standard", value=0.001)))
    synthetic.add_value("n", "Turns", Quantity(value=14, unit="1"))
    synthetic.data["candidates"] = [{"core": "x", "feasible": True, "reasons": ["r" * 100], "warnings": ["w" * 100]}]
    text = api.generate_report([compare, validation, synthetic]).data["content"]
    assert "### Material comparison" in text and "material.tdk-n87" in text
    pv = compare.data["comparison"][0]["loss_density_W_per_m3"]
    assert format(pv / 1e3, ".4g") in text
    assert "### Validation checks" in text and "Verdict: **PASSED**" in text and "| curie_temperature | PASS |" in text
    assert "+/- 0.001 T (standard uncertainty)" in text
    assert "| Loss \\| density (`pv`) |" in text
    assert "| Turns (`n`) | 14 |" in text
    assert "r" * 100 + "; " + "w" * 100 in text  # notes are not truncated
    assert "* **specified**" in text


def test_report_destination_exists_has_its_own_error_code(tmp_path):
    loss = api.estimate_core_loss(material=N87, frequency=KHZ100, flux_density_peak={"value": 100, "unit": "mT"}, temperature=C100)
    target = tmp_path / "r.md"
    first = api.generate_report([loss], output_path=str(target), allow_write=True)
    assert first.status == "ok" and target.exists()
    second = api.generate_report([loss], output_path=str(target), allow_write=True)
    assert second.status == "rejected" and second.errors[0].code == "destination_exists"
    refused = api.generate_report([loss], output_path=str(tmp_path / "other.md"))
    assert refused.status == "rejected" and refused.errors[0].code == "authorization_required"
