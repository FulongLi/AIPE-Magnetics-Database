"""Evidence registry semantics: undecidable range dimensions, missing bundle, reproducible bundle payload."""

from __future__ import annotations

from pathlib import Path

from aipe_magnetics.models.contract import ValidityRange
from aipe_magnetics.verification.evidence import EvidenceRecord, OperatingRange, VerificationRegistry, VerificationReport
from aipe_magnetics.verification.release import bundle_payload
from aipe_magnetics.verification.suite import build_report, load_bundled_registry


def _record(id: str, level: str = "L2", status: str = "pass", **range_kwargs) -> EvidenceRecord:
    rng = OperatingRange(**range_kwargs) if range_kwargs else None
    return EvidenceRecord(id=id, capability="cap.x", level=level, kind="analytical_reference", status=status, description="d", method="m",
                          range=rng, check_id=id, tool_version="0.2.0")


def test_missing_condition_is_undecidable_not_outside():
    ranged = _record("f", frequency=ValidityRange(min=1e3, max=1e6, unit="Hz"))
    reg = VerificationRegistry([ranged])
    none = reg.status_for("cap.x")
    assert none.state == "verified" and none.level == "L2" and none.in_verified_range is None
    assert "not checked" in none.statement and "frequency" in none.statement
    inside = reg.status_for("cap.x", frequency=5e5)
    assert inside.in_verified_range is True and inside.state == "verified"
    outside = reg.status_for("cap.x", frequency=5e9)
    assert outside.state == "out_of_range" and outside.in_verified_range is False and outside.level == "L0"
    # two dimensions, one supplied and inside: still undecidable, never 'out of range'
    both = VerificationRegistry([_record("fb", frequency=ValidityRange(min=1e3, max=1e6, unit="Hz"), flux_density_peak=ValidityRange(min=0.01, max=0.3, unit="T"))])
    partial = both.status_for("cap.x", frequency=5e5)
    assert partial.state == "verified" and partial.in_verified_range is None and "flux_density_peak" in partial.statement
    assert both.status_for("cap.x", frequency=5e5, flux_density_peak=1.0).state == "out_of_range"
    # materials-only range queried without a material
    mat = VerificationRegistry([_record("m", materials=["material.a"])])
    assert mat.status_for("cap.x").in_verified_range is None
    assert mat.status_for("cap.x", material="material.b").state == "out_of_range"
    assert mat.status_for("cap.x", material="material.a").in_verified_range is True


def test_unranged_and_ranged_records_combine():
    reg = VerificationRegistry([_record("l1", level="L1"), _record("f", frequency=ValidityRange(min=1e3, max=1e6, unit="Hz"))])
    assert reg.status_for("cap.x").level == "L2"
    out = reg.status_for("cap.x", frequency=5e9)
    assert out.state == "out_of_range" and out.level == "L1" and out.evidence_refs == ["l1"]
    assert reg.status_for("cap.x", frequency=5e5).evidence_refs == ["l1", "f"]


def test_operating_range_decide_reports_undecided_dimensions():
    rng = OperatingRange(frequency=ValidityRange(min=1e3, max=1e6, unit="Hz"), excitation=["sinusoidal"])
    assert rng.decide(frequency=5e5) == (None, ["excitation"])
    assert rng.decide(frequency=5e5, excitation="sinusoidal") == (True, [])
    assert rng.decide(frequency=5e9, excitation="sinusoidal")[0] is False
    assert rng.contains() is None and OperatingRange().decide() == (None, [])


def test_missing_bundle_is_named_in_every_status():
    reg = VerificationRegistry(missing_source=Path("/nowhere/v0.2.0.json"))
    status = reg.status_for("aipe.magnetics.model.igse")
    assert status.state == "unverified" and "nowhere" in status.statement and "--update-bundled" in status.statement
    assert reg.evidence_available is False
    bundled = load_bundled_registry()
    assert bundled.evidence_available == (bundled.missing_source is None and bool(bundled.records))


def test_bundle_payload_has_no_wall_clock_and_is_reproducible():
    report = VerificationReport(tool_version="0.2.0", records=[_record("a")], summary={"_totals": {"checks": 1, "pass": 1, "fail": 0, "duration_s": 1.23}})
    payload = bundle_payload(report)
    assert "duration_s" not in payload["summary"]["_totals"]
    assert payload == bundle_payload(report)
    built = build_report([])
    assert "duration_s" in built.summary["_totals"] and "duration_s" not in bundle_payload(built)["summary"]["_totals"]
