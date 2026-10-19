"""Design result -> Core Engineering State bridge: exported state is contract-valid and round-trips."""

from __future__ import annotations

from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import EngineeringResult, ModelIdentity, SourceReference
from aipe_magnetics.integrations.aipe_core.bridge import component_summary_from_design
from aipe_magnetics.integrations.aipe_core.export import from_engineering_state, to_engineering_state
from aipe_magnetics.integrations.aipe_core.state import validate_engineering_state

CREATED = "2026-10-19T00:00:00Z"


def _fake_design() -> EngineeringResult:
    r = EngineeringResult(capability="aipe.magnetics.design_transformer",
                          model=ModelIdentity(id="aipe.magnetics.model.igse", name="iGSE", version="0.2.0", family="empirical"))
    r.add_value("turns_primary", "Turns primary", Quantity(value=16, unit="1"))
    r.add_value("turns_secondary", "Turns secondary", Quantity(value=8, unit="1"))
    r.add_value("flux_density_peak", "Bpk", Quantity(value=0.117, unit="T"))
    r.add_value("core_loss", "Core loss", Quantity(value=12.3, unit="W"), status="estimated")
    r.assume("flux_limit_fraction", "Peak flux density limited to 70 % of Bsat.", "default", ["turns_primary"])
    r.assume("winding_currents", "Currents estimated as P/V.", "engineering_judgement", ["copper_loss"])
    r.cite(SourceReference(id="evidence.tdk-n87-datasheet-2025-06", kind="datasheet", reference="TDK N87 datasheet", locator="PV table"))
    r.limit("analytical_design", "Analytical sizing only.")
    r.data.update(selected="core.tdk-e65-32-27", material="material.tdk-n87", candidates=[])
    return r


def test_bridge_exports_contract_valid_state_and_round_trips():
    summary = component_summary_from_design(_fake_design(), role="DAB isolation transformer", created_at=CREATED)
    assert summary.kind == "transformer" and summary.turns_ratio.value == 2.0
    kinds = [e.kind for e in summary.evidence]
    assert kinds.count("analytical") == 1 and "assumption" in kinds and "source" in kinds
    analytical = next(e for e in summary.evidence if e.kind == "analytical")
    assert set(analytical.provenance.inputs) == {e.id for e in summary.evidence if e.kind != "analytical"}
    state = to_engineering_state(summary, created_at=CREATED, project_id="project.bridge-test",
                                 requirements={"nominal_power": Quantity(value=10000, unit="W"), "switching_frequency": Quantity(value=1e5, unit="Hz"),
                                               "evidence": [{"id": "evidence.brief", "kind": "assumption", "description": "test brief",
                                                             "source": {"type": "user_requirement", "reference": "unit test"},
                                                             "provenance": {"created_at": CREATED, "activity": "record", "inputs": []}}]})
    assert validate_engineering_state(state) == []
    assert state["validation"]["status"] == "not_run"
    detail = state["magnetics"][0]["extensions"]["aipe.magnetics-detail"]
    assert detail["detail"]["selected_core"] == "core.tdk-e65-32-27"
    assert detail["quantities"]["turns_ratio"] == {"value": 2.0, "unit": "1"}
    back = from_engineering_state(state)
    assert len(back) == 1 and back[0].id == summary.id and back[0].turns_ratio.value == 2.0
    assert back[0].limitations == summary.limitations


def test_bridge_refuses_rejected_results():
    import pytest
    from aipe_magnetics.common.errors import InvalidInputError
    bad = EngineeringResult.rejected("aipe.magnetics.design_transformer", InvalidInputError("nope"))
    with pytest.raises(InvalidInputError):
        component_summary_from_design(bad, role="x", created_at=CREATED)
