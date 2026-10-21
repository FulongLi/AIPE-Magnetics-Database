"""Waveform-informed DAB transformer design: SPS-synthesised and imported waveforms, candidate ranking, comparisons, Core export."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import aipe_magnetics.api as api
from aipe_magnetics.api.types import WaveformInformedTransformerRequest
from aipe_magnetics.common.jsonio import sha256_file
from aipe_magnetics.common.units import MU_0
from aipe_magnetics.database import core_data
from aipe_magnetics.design.transformers import dab
from aipe_magnetics.design.transformers import waveform_informed as wd
from aipe_magnetics.engineering.periodic import PiecewiseLinear
from aipe_magnetics.geometry.conductors import ConductorSpec, litz_for_area
from aipe_magnetics.integrations.aipe_core.state import validate_engineering_state

REAL_FILE = Path("D:/Cursor Workspace/AIPE-Full-Ecosystem-Integration/runs/dab-l1-nominal-ic/waveforms.h5")
KHZ100 = {"value": 100, "unit": "kHz"}
CREATED = "2026-10-21T12:00:00Z"
SPS = {"kind": "sps", "sps": {"power": {"value": 10, "unit": "kW"}, "primary_voltage": {"value": 800, "unit": "V"},
                              "secondary_voltage": {"value": 400, "unit": "V"}, "series_inductance": {"value": 35, "unit": "uH"}}}


def _request(**over):
    base = {"frequency": KHZ100, "turns_ratio_target": 2.0, "waveforms": SPS, "materials": ["N87"], "loss_density_target": {"value": 300, "unit": "kW/m^3"},
            "max_candidates": 20}
    base.update(over)
    return base


@pytest.fixture(scope="module")
def sps_result():
    r = api.design_transformer_from_waveforms(_request(), created_at=CREATED, export_state=True)
    assert r.status == "ok", [e.message for e in r.errors]
    return r


def _candidate(result, core: str, turns: tuple[int, int], arrangement: str):
    for c in result.data["candidates"]:
        if c["core"] == core and (c["turns"]["primary"], c["turns"]["secondary"]) == turns and c["arrangement"] == arrangement:
            return c
    raise AssertionError(f"{core} {turns} {arrangement} not among the feasible candidates: {[c['id'] for c in result.data['candidates']]}")


# -------------------------------------------------------------------------------------------------
# SPS-synthesised excitation (fast path)
# -------------------------------------------------------------------------------------------------
def test_sps_excitation_reproduces_the_dab_operating_point(sps_result):
    ex = sps_result.data["excitation"]
    assert ex["basis"] == "synthesised_sps" and ex["flux_basis"] == "ideal_square_voltage"
    assert ex["primary_current"]["rms"] == pytest.approx(13.68, rel=2e-3) and ex["secondary_current"]["rms"] == pytest.approx(2 * 13.68, rel=2e-3)
    assert ex["flux_linkage_peak_to_peak_Vs"] == pytest.approx(800 / (2 * 100e3))
    assert ex["throughput_power_W"] == pytest.approx(10e3, rel=0.01) and ex["series_inductance_total_H"] == pytest.approx(35e-6)
    assert ex["primary_current"]["harmonic_energy_fraction"] > 0.999
    assert all(h["amplitude"] < 1e-9 for h in ex["primary_current"]["harmonics"] if h["n"] % 2 == 0)


def test_known_candidate_matches_the_public_demo_core_loss_and_reports_every_quantity(sps_result):
    c = _candidate(sps_result, "core.tdk-e65-32-27", (21, 10), "P-S-P")
    assert c["flux_density_peak_T"] == pytest.approx(0.178, abs=1e-3)              # 4 mV*s / (2 * 21 * 535 mm^2)
    assert c["core_loss_W"] == pytest.approx(21.5, rel=0.02)                        # identical flux -> identical iGSE core loss as the v0.2 demo
    assert c["turns_ratio"] == 2.1 and c["turns_ratio_error"] == pytest.approx(0.05)
    assert 0 < c["leakage_inductance_H"] < 5e-6
    assert c["external_inductance_required_H"] == pytest.approx(35e-6 - c["leakage_inductance_H"])
    lm = c["magnetising_inductance"]
    core = core_data("core.tdk-e65-32-27")
    expected = 21 ** 2 * MU_0 * core.effective_area / (core.effective_length / 2200.0 + 2 * wd.RESIDUAL_GAP_PER_LEG)
    assert lm["reluctance_model_H"] == pytest.approx(expected) and lm["datasheet_AL_H"] == pytest.approx(7900e-9 * 21 ** 2)
    assert c["thermal"]["method"] == "thermal_network_electrothermal" and c["thermal"]["converged"]
    assert c["temperature_rise_K"] > 0 and c["windings"]["primary"]["method"].startswith("dowell_per_layer_harmonic")
    assert c["windings"]["primary"]["effective_ac_factor"] > 1.0 and "litz_1d_reference" in c["windings"]["primary"]
    assert 0 < c["window_fill"] < 0.45 and c["layout"]["fidelity"] == "parametric_e_core"
    assert c["saturation_margin"] == pytest.approx(1 - c["flux_density_max_T"] / 0.39)


def test_ranking_values_and_verification_are_truthful(sps_result):
    r = sps_result
    totals = [c["total_loss_W"] for c in r.data["candidates"]]
    assert totals == sorted(totals) and r.data["candidate_counts"]["feasible"] >= len(totals) > 0
    assert all(not c["violations"] for c in r.data["candidates"])
    for key in ("turns_primary", "turns_secondary", "turns_ratio", "flux_density_peak", "saturation_margin", "core_loss", "copper_loss", "total_loss",
                "leakage_inductance", "external_inductance_required", "magnetising_inductance", "window_fill", "temperature_rise", "current_rms_primary"):
        assert key in r.values, key
    assert r.values["total_loss"].quantity.value == pytest.approx(r.values["core_loss"].quantity.value + r.values["copper_loss"].quantity.value)
    assert r.values["current_rms_primary"].status == "assumed"  # synthesised, not imported
    assert r.data["selected"] == r.data["selected_core"] and r.data["selected"].startswith("core.")
    assert r.model is not None and r.model.id == "aipe.magnetics.model.igse"
    # no evidence is recorded for the new capability: the result says so instead of inheriting the loss model's L2
    assert r.verification.state in ("unverified", "partially_verified") and "No verification evidence recorded" in r.verification.statement
    assert r.data["loss_model_verification"]["level"] in ("L0", "L1", "L2")
    codes = {lim.code for lim in r.limitations}
    assert {"analytical_models", "igse_scope", "dowell_litz", "thermal_estimate", "waveform_scope", "no_hardware_validation"} <= codes
    assert any(a.id == "sps_waveforms" for a in r.assumptions) and any(a.id == "cooling" for a in r.assumptions)
    summary = r.data["infeasible_summary"]
    assert all(set(item) == {"reason", "candidates"} for item in summary)


def test_square_current_comparison_isolates_rms_and_harmonic_effects(sps_result):
    sq = sps_result.data["square_current_comparison"]
    assert sq["rms_ratio_actual_over_square"]["primary"] == pytest.approx(13.68 / 12.5, rel=3e-3)
    losses = sq["copper_loss_total_W"]
    v02, actual_rms, harmonic = (losses["square_rms_dowell_at_fundamental_v02_method"], losses["actual_rms_dowell_at_fundamental"],
                                 losses["actual_waveform_per_layer_harmonic_dowell"])
    assert actual_rms == pytest.approx(v02 * (13.68 / 12.5) ** 2, rel=3e-3)   # same geometry, only the RMS changes
    assert harmonic > v02                                                      # harmonics and the MMF profile add loss on top of the RMS effect
    public = sq["public_design_transformer"]
    assert public["status"] == "ok" and public["turns"]["primary"] >= 1 and public["total_loss_W"] > 0


def test_material_comparison_and_measurement_requests_name_missing_data(sps_result):
    cmp = sps_result.data["material_comparison"]
    rows = {r["material"]: r for r in cmp["rows"]}
    assert cmp["reference_core"] == sps_result.data["selected_core"] and len(rows) >= 10
    assert rows["material.tdk-n87"]["evaluable_for_design"] and rows["material.tdk-n87"]["loss_notes"] == []
    assert not rows["material.tdk-n49"]["evaluable_for_design"] and "saturation flux density vs temperature" in rows["material.tdk-n49"]["missing_data"]
    assert any("temperature_extrapolated" in n for n in rows["material.ferroxcube-3c94"]["loss_notes"])
    assert rows["material.ferroxcube-3c94"]["saturation_margin"] is not None and rows["material.tdk-n49"]["saturation_margin"] is None
    considered = {m["material"]: m for m in sps_result.data["materials_considered"]}
    assert set(considered) == {"material.tdk-n87"} and considered["material.tdk-n87"]["evaluable"]  # the request restricted the design to N87
    assert rows["material.tdk-n27"]["reasons_not_evaluable"] == ["no saturation flux density data"]  # the comparison still covers the catalogue
    requests = sps_result.data["measurement_requests"]
    assert requests["schema"] == "aipe.magnetics.measurement-request/0.3" and requests["conditions"]["winding_context"]["turns"]["primary"] >= 1
    n49 = next(m for m in requests["requests"] if m["material"] == "material.tdk-n49")
    assert any(q["category"] == "saturation_vs_temperature" and q["priority"] == "high" for q in n49["requests"])


def test_engineering_state_export_is_contract_valid_and_links_the_excitation(sps_result):
    state = sps_result.data["engineering_state"]
    assert validate_engineering_state(state) == [] and state["validation"]["status"] == "analytically_checked"
    kinds = {e["kind"] for e in state["evidence"]}
    assert kinds <= {"assumption", "analytical", "source"}  # never simulation / measurement evidence
    analytical = next(e for e in state["evidence"] if e["kind"] == "analytical")
    wave = next(e for e in state["evidence"] if e["id"].endswith("waveform-sps"))
    assert wave["id"] in analytical["provenance"]["inputs"] and wave["kind"] == "assumption"
    entry = state["magnetics"][0]
    assert entry["kind"] == "transformer" and entry["database_ref"] == sps_result.data["selected_core"]
    assert entry["leakage_inductance"]["value"] == pytest.approx(sps_result.values["leakage_inductance"].quantity.value)
    assert entry["inductance"]["value"] == pytest.approx(sps_result.values["magnetising_inductance"].quantity.value)
    assert state["requirements"]["switching_frequency"] == {"value": 100e3, "unit": "Hz"}


def test_export_state_requires_a_caller_timestamp_and_rejections_are_explicit():
    with pytest.raises(Exception):
        wd.design_transformer_from_waveforms(WaveformInformedTransformerRequest.model_validate(_request()), export_state=True)
    assert api.design_transformer_from_waveforms(_request(materials=["unobtainium"])).status == "rejected"
    no_voltage = api.design_transformer_from_waveforms({"frequency": KHZ100, "turns_ratio_target": 2.0,
                                                        "waveforms": {"kind": "json", "document": {"frequency": KHZ100, "channels": {"i": {"unit": "A", "values": [0, 1] * 8}}},
                                                                      "channels": {"primary_current": "i"}}})
    assert no_voltage.status == "rejected" and "primary_voltage" in no_voltage.errors[0].message
    n49 = api.design_transformer_from_waveforms(_request(materials=["N49"]))
    assert n49.status == "rejected" and n49.errors[0].code == "no_evaluable_material"


def test_json_document_source_runs_the_import_path():
    current = dab.inductor_current(800.0, 400.0, 2.0, 100e3, 35e-6, math.radians(22.5))
    x = np.linspace(0.0, 1.0, 2001)[:-1]
    doc = {"frequency": KHZ100, "channels": {"i_p": {"unit": "A", "values": current.value(x).tolist()},
                                             "v_p": {"unit": "V", "values": np.where(x < 0.5, 800.0, -800.0).tolist()}}}
    r = api.design_transformer_from_waveforms(_request(waveforms={"kind": "json", "document": doc, "channels": {"primary_current": "i_p", "primary_voltage": "v_p"}},
                                                       series_inductance_total={"value": 35, "unit": "uH"}, turn_steps=1, max_candidates=5))
    assert r.status == "ok", [e.message for e in r.errors]
    ex = r.data["excitation"]
    assert ex["basis"] == "imported" and ex["secondary_basis"] == "derived_from_primary_and_turns_ratio_target"
    assert ex["primary_current"]["rms"] == pytest.approx(13.68, rel=3e-3) and ex["flux_linkage_peak_to_peak_Vs"] == pytest.approx(4e-3, rel=2e-3)
    assert ex["period_detection"]["method"] == "frequency_hint_unchecked"  # a single period cannot be checked for periodicity
    assert any(a.id == "secondary_current_derived" for a in r.assumptions)


# -------------------------------------------------------------------------------------------------
# building blocks
# -------------------------------------------------------------------------------------------------
def test_harmonic_dowell_tends_to_the_dc_loss_at_low_frequency():
    # the same trapezoidal shape (22.5 deg phase shift) at 50 Hz: every strand is far thinner than the skin depth of the 41st harmonic
    shape_sps = {"kind": "sps", "sps": {**SPS["sps"], "series_inductance": None, "phase_shift": {"value": 22.5, "unit": "deg"}}}
    shape_sps["sps"] = {k: v for k, v in shape_sps["sps"].items() if v is not None}
    excitation = wd.build_excitation(WaveformInformedTransformerRequest.model_validate(_request(frequency={"value": 50, "unit": "Hz"}, waveforms=shape_sps)))
    core = core_data("core.tdk-e65-32-27")
    layout = wd.build_layout(core, "material.tdk-n87", 21, 10, litz_for_area(13.68 / 4e6, 0.1e-3), litz_for_area(27.4 / 4e6, 0.1e-3), "P-S", 0.0, 0.2e-3)
    losses = wd.winding_losses(layout, excitation, 21, 10, 373.15, 41)
    for w in ("primary", "secondary"):
        assert losses[w]["loss_W"] == pytest.approx(losses[w]["loss_dc_resistance_W"], rel=1e-3)   # F_R -> 1 when the strands are far thinner than delta
        assert losses[w]["effective_ac_factor"] == pytest.approx(1.0, abs=1e-3)


def test_leakage_inductance_matches_the_classical_two_winding_formula():
    a1, a2, delta, h, mlt = 2e-3, 3e-3, 0.5e-3, 40e-3, 150e-3
    foil = ConductorSpec(kind="foil", thickness=a1, width=h)
    foil2 = ConductorSpec(kind="foil", thickness=a2, width=h)
    layers = [wd.LayerInfo("primary", 0, 0.0, a1, 10, mlt, foil), wd.LayerInfo("secondary", 1, a1 + delta, a1 + delta + a2, 10, mlt, foil2)]
    layout = wd.Layout(fidelity="test", layers=layers, winding_height=h, checks=[], derived={})
    expected = MU_0 * 10 ** 2 * mlt / h * (a1 / 3 + a2 / 3 + delta)
    assert wd.leakage_inductance(layout, 10, 10) == pytest.approx(expected, rel=1e-12)


def test_generic_layout_is_used_for_cores_without_the_e_outline_and_uu_core_is_not_evaluable():
    etd = core_data("core.tdk-etd59-31-22")
    layout = wd.build_layout(etd, "material.tdk-n87", 24, 12, litz_for_area(13.68 / 4e6, 0.1e-3), litz_for_area(27.4 / 4e6, 0.1e-3), "P-S", 0.0, 0.2e-3)
    assert layout.fidelity == "generic_layer_stack" and all(l.mean_turn_length == etd.mean_turn_length for l in layout.layers)
    assert layout.winding_height == pytest.approx(2 * 22.0e-3 - 2 * wd.BOBBIN_FLANGE) and any(c["code"] == "radial_fit" for c in layout.checks)
    with pytest.raises(Exception):
        wd.build_layout(core_data("core.tdk-uu93-152-30"), "material.tdk-n87", 10, 5, litz_for_area(1e-6, 0.1e-3), litz_for_area(2e-6, 0.1e-3), "P-S", 0.0, 0.2e-3)


def test_flux_shape_normalisation_and_excitation_description_are_consistent():
    excitation = wd.build_excitation(WaveformInformedTransformerRequest.model_validate(_request()))
    shape = excitation.flux_shape()
    assert shape.kind == "piecewise_linear" and shape.igse_shape_factor(1.0) == pytest.approx(2.0)  # triangle: sum |dB/dBpp| x^0 = 1 + 1
    described = excitation.describe()
    assert described["primary_current"]["unit"] == "A" and described["flux_linkage_peak_to_peak_Vs"] == pytest.approx(4e-3)


# -------------------------------------------------------------------------------------------------
# the executed ngspice waveform file (skipped when absent)
# -------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not REAL_FILE.exists(), reason=f"executed DAB waveform file not present: {REAL_FILE}")
def test_design_from_the_executed_dab_waveform_file():
    pytest.importorskip("h5py", reason="the HDF5 waveform loader needs h5py (aipe-magnetics[hdf5])")
    r = api.design_transformer_from_waveforms(
        frequency=KHZ100, turns_ratio_target=2.0, loss_density_target={"value": 300, "unit": "kW/m^3"}, max_candidates=12,
        waveforms={"kind": "hdf5", "path": str(REAL_FILE), "channels": {"primary_current": "i_L", "secondary_current": "i_s", "primary_voltage": "v_tp",
                                                                        "magnetising_current": "i_m"},
                   "window": {"t0": {"value": 200, "unit": "us"}, "t1": {"value": 240, "unit": "us"}}},
        created_at=CREATED, export_state=True)
    assert r.status == "ok", [e.message for e in r.errors]
    ex = r.data["excitation"]
    assert ex["basis"] == "imported" and ex["flux_basis"] == "integral_of_voltage_channel" and ex["secondary_basis"] == "imported"
    assert ex["primary_current"]["rms"] == pytest.approx(13.68, rel=2e-3) and ex["secondary_current"]["rms"] == pytest.approx(27.9, rel=3e-3)
    assert ex["flux_linkage_peak_to_peak_Vs"] == pytest.approx(4.0e-3, rel=1e-3) and ex["series_inductance_total_H"] == pytest.approx(35e-6)
    assert ex["period_detection"]["periodicity"] > 0.999 and ex["steady_state"]["rms_relative_spread"] < 1e-6
    assert r.values["current_rms_primary"].status == "specified"
    c = _candidate(r, "core.tdk-e65-32-27", (21, 10), "P-S-P")
    assert c["flux_density_peak_T"] == pytest.approx(0.178, abs=1e-3) and c["core_loss_W"] == pytest.approx(21.5, rel=0.02)
    assert c["external_inductance_required_H"] == pytest.approx(35e-6 - c["leakage_inductance_H"])
    assert c["magnetising_current_peak_A"] < ex["magnetising_current"]["peak"]  # the design's L_m exceeds the 1 mH used by the simulation
    state = r.data["engineering_state"]
    assert validate_engineering_state(state) == []
    wave = next(e for e in state["evidence"] if e["id"].endswith("waveform-source"))
    assert wave["kind"] == "source" and wave["artifacts"][0]["sha256"] == sha256_file(REAL_FILE) == ex["source"]["sha256"]
    assert wave["artifacts"][0]["uri"].startswith("file:///") and wave["id"] in next(e for e in state["evidence"] if e["kind"] == "analytical")["provenance"]["inputs"]
    json.dumps(r.to_json_dict())  # the whole envelope is JSON serialisable
