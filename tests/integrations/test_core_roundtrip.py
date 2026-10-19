"""AIPE-Core compatibility: vendored example, Core negative cases, executable round trip, CLI, reference validator."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from referencing.exceptions import NoSuchResource

from aipe_magnetics.common.errors import DataIntegrityError, InvalidInputError, UnsupportedOperationError
from aipe_magnetics.common.jsonio import dump
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.integrations.aipe_core import (
    CORE_COMMIT,
    CORE_SCHEMA_BASE,
    EXTENSION_KEY,
    SCHEMA_DIR,
    VENDORED_SHA256,
    CoreEvidence,
    CoreValidationCheck,
    DesignRequirements,
    MagneticComponentSummary,
    core_schema_ids,
    extract_requirements,
    from_engineering_state,
    load_core_example,
    load_core_schemas,
    load_state,
    to_engineering_state,
    validate_engineering_state,
    verify_vendored_integrity,
)
from aipe_magnetics.integrations.aipe_core.__main__ import main as core_cli
from aipe_magnetics.verification.checks.check_integrations import (
    FIXTURE_TIMESTAMP,
    core_example_validates,
    core_roundtrip,
    export_fixture_state,
    illustrative_dab_transformer,
)

T = FIXTURE_TIMESTAMP
SKIP_REASON = "AIPE_ECOSYSTEM_CHECKOUTS not set"


# --------------------------------------------------------------------------------------------------
# Vendored schemas and example
# --------------------------------------------------------------------------------------------------
def test_vendored_files_match_pins_and_notice_exists():
    digests = verify_vendored_integrity()
    assert len(digests) == 15 and digests == VENDORED_SHA256
    notice = (SCHEMA_DIR / "NOTICE.md").read_text(encoding="utf-8")
    assert CORE_COMMIT in notice and "MIT License" in notice and "https://github.com/FulongLi/AIPE-Core" in notice
    for name, digest in VENDORED_SHA256.items():
        assert digest in notice, name
    assert len(core_schema_ids()) == 14
    assert all(identifier.startswith(CORE_SCHEMA_BASE) for identifier in core_schema_ids())


def test_schema_registry_is_offline():
    registry = load_core_schemas()
    assert registry.get_or_retrieve(CORE_SCHEMA_BASE + "common.schema.json").value.contents["title"] == "AIPE Common"
    with pytest.raises(NoSuchResource):
        registry.get_or_retrieve(CORE_SCHEMA_BASE + "missing.schema.json")
    with pytest.raises(NoSuchResource):
        registry.get_or_retrieve("https://example.com/remote.schema.json")


def test_vendored_example_validates_without_errors():
    state = load_core_example()
    assert state["id"] == "aipe.example.dab-10kw"
    assert validate_engineering_state(state) == []
    # a fresh copy each time: mutating one copy never affects the next
    state["magnetics"].append({"bogus": True})
    assert load_core_example()["magnetics"] == []


def test_state_loader_rejects_duplicate_keys_and_nonstandard_numbers(tmp_path):
    for content in ['{"id": 1, "id": 2}', '{"value": NaN}', '{"value": Infinity}']:
        path = tmp_path / "invalid.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            load_state(path)


# --------------------------------------------------------------------------------------------------
# Negative cases reproduced from AIPE-Core tests/test_contract.py
# --------------------------------------------------------------------------------------------------
def _missing_domain(s):
    del s["thermal"]


def _unknown_field(s):
    s["requirements"]["typo"] = 1


def _bare_number(s):
    s["requirements"]["nominal_power"] = 10000


def _prefixed_unit(s):
    s["requirements"]["nominal_power"] = {"value": 10, "unit": "kW"}


def _wrong_dimension(s):
    s["requirements"]["input_voltage"]["unit"] = "W"


def _negative_frequency(s):
    s["requirements"]["switching_frequency"]["value"] = -1


def _non_finite(s):
    s["requirements"]["nominal_power"]["value"] = float("inf")


def _bad_date(s):
    s["project"]["created_at"] = "2026-99-99"


def _bad_version(s):
    s["schema_version"] = "1.0.0"


def _unscoped_extension(s):
    s["extensions"] = {"typo": {}}


def _missing_evidence_reference(s):
    s["requirements"]["evidence_refs"] = ["evidence.missing"]


def _quantities_without_evidence(s):
    s["requirements"]["evidence_refs"] = []


def _duplicate_ids(s):
    s["evidence"].append(copy.deepcopy(s["evidence"][0]))


def _provenance_cycle(s):
    s["evidence"][0]["provenance"]["inputs"] = ["evidence.design-brief"]


def _rejected_evidence(s):
    s["evidence"][0]["review_status"] = "rejected"


def _measurement_without_artifact(s):
    s["evidence"][0]["kind"] = "measurement"


def _analytical_without_inputs(s):
    s["evidence"][0]["kind"] = "analytical"


def _assumption_promoted_to_measured(s):
    s["validation"]["status"] = "measured"
    s["validation"]["checks"] = [{"id": "check.measurement", "kind": "measurement", "status": "pass",
                                  "description": "Unsupported measurement claim", "evidence_refs": ["evidence.design-brief"]}]


def _status_without_check(s):
    s["validation"]["status"] = "schema_validated"


def _failed_check_with_ok_status(s):
    s["validation"]["checks"] = [{"id": "check.failure", "kind": "schema", "status": "fail",
                                  "description": "Failure", "evidence_refs": []}]


def _completed_simulation_without_run(s):
    s["simulation"]["status"] = "completed"


def _completed_run_without_evidence(s):
    s["simulation"] = {"status": "completed", "evidence_refs": [], "runs": [{
        "id": "run.example", "status": "completed", "tool": {"id": "tool.example", "version": "1"},
        "model": {"uri": "example.json", "media_type": "application/json"},
        "outputs": [], "operating_point_ref": "operating-point.missing", "evidence_refs": []}]}


CORE_MUTATIONS = [
    (_missing_domain, "required property"),
    (_unknown_field, "Additional properties"),
    (_bare_number, "object"),
    (_prefixed_unit, "W"),
    (_wrong_dimension, "V"),
    (_negative_frequency, ""),
    (_non_finite, "finite"),
    (_bad_date, "date-time"),
    (_bad_version, "0.1.0"),
    (_unscoped_extension, ""),
    (_missing_evidence_reference, "unknown evidence"),
    (_quantities_without_evidence, "require evidence_refs"),
    (_duplicate_ids, "Duplicate entity id"),
    (_provenance_cycle, "Cyclic"),
    (_rejected_evidence, "rejected evidence"),
    (_measurement_without_artifact, ""),
    (_analytical_without_inputs, ""),
    (_assumption_promoted_to_measured, "matching evidence"),
    (_status_without_check, "passing schema check"),
    (_failed_check_with_ok_status, "Failed checks"),
    (_completed_simulation_without_run, "matching run"),
    (_completed_run_without_evidence, "unknown operating point"),
]


@pytest.mark.parametrize("mutate,expected", CORE_MUTATIONS, ids=[m.__name__.strip("_") for m, _ in CORE_MUTATIONS])
def test_core_contract_violations_are_rejected(mutate, expected):
    state = load_core_example()
    mutate(state)
    errors = validate_engineering_state(state)
    assert errors, mutate.__name__
    assert any(expected in error for error in errors), errors


def test_completed_run_also_needs_outputs_and_simulation_evidence():
    state = load_core_example()
    _completed_run_without_evidence(state)
    errors = validate_engineering_state(state)
    assert any("completed simulation requires" in error for error in errors), errors


def test_namespaced_extensions_are_allowed():
    state = load_core_example()
    state["extensions"] = {"example.custom": {"independent_domain": 3}}
    assert validate_engineering_state(state) == []


# --------------------------------------------------------------------------------------------------
# Export round trip
# --------------------------------------------------------------------------------------------------
def test_transformer_summary_round_trip():
    fixture = illustrative_dab_transformer()
    summary = fixture["summary"]
    assert {record.kind for record in summary.evidence} == {"assumption", "source", "analytical"}
    state = export_fixture_state(fixture)
    assert validate_engineering_state(state) == []
    assert state["schema_version"] == "0.1.0" and state["project"]["purpose"] == "example"
    entry = state["magnetics"][0]
    assert entry["inductance"] == {"value": pytest.approx(850e-6), "unit": "H"}
    assert entry["leakage_inductance"]["unit"] == "H" and entry["turns_ratio"] == {"value": 2.0, "unit": "1"}
    assert entry["evidence_refs"] == [record.id for record in summary.evidence]
    assert set(entry["extensions"]) == {EXTENSION_KEY}
    detail = entry["extensions"][EXTENSION_KEY]
    assert detail["detail"] == summary.detail and detail["limitations"] == summary.limitations
    assert detail["quantities"]["inductance"]["original"] == {"value": 850.0, "unit": "uH"}
    imported = from_engineering_state(state)
    assert imported == [summary]
    assert imported[0].inductance.original is not None and imported[0].inductance.original.unit == "uH"
    # JSON serialisation round trip through disk keeps equality too
    reloaded = json.loads(json.dumps(state, sort_keys=True, allow_nan=False))
    assert from_engineering_state(reloaded) == [summary]


def test_requirements_and_validation_status():
    fixture = illustrative_dab_transformer()
    state = export_fixture_state(fixture)
    requirements = extract_requirements(state)
    assert set(requirements) == {"nominal_power", "input_voltage", "output_voltage", "switching_frequency"}
    assert requirements["nominal_power"] == Quantity(value=10000.0, unit="W")
    assert requirements["switching_frequency"] == Quantity(value=100000.0, unit="Hz")
    assert state["requirements"]["evidence_refs"] == ["evidence.design-inputs"]
    assert state["validation"]["status"] == "analytically_checked"
    assert state["validation"]["checks"][0]["kind"] == "analytical"
    assert any("no simulation, measurement" in item for item in state["validation"]["limitations"])
    # the DAB example has requirements but no magnetics
    assert extract_requirements(load_core_example())["input_voltage"] == Quantity(value=800.0, unit="V")
    assert from_engineering_state(load_core_example()) == []


def test_status_not_run_without_passing_analytical_check():
    fixture = illustrative_dab_transformer()
    kwargs = dict(created_at=T, project_id=fixture["project_id"], requirements=fixture["requirements"])
    plain = to_engineering_state(fixture["summary"], **kwargs)
    assert plain["validation"]["status"] == "not_run" and plain["validation"]["checks"] == []
    not_run_check = CoreValidationCheck(id="check.pending", status="not_run", description="Not executed yet.",
                                        evidence_refs=[])
    pending = to_engineering_state(fixture["summary"], checks=[not_run_check], **kwargs)
    assert pending["validation"]["status"] == "not_run"
    failing = CoreValidationCheck(id="check.failed", status="fail", description="Did not hold.",
                                  evidence_refs=["evidence.turns-ratio-analytical"])
    failed = to_engineering_state(fixture["summary"], checks=[fixture["checks"][0], failing], **kwargs)
    assert failed["validation"]["status"] == "failed"
    assert validate_engineering_state(failed) == []


def test_passing_check_requires_analytical_evidence():
    fixture = illustrative_dab_transformer()
    bad = CoreValidationCheck(id="check.unsupported", status="pass", description="Claims a pass from an assumption.",
                              evidence_refs=["evidence.magnetics-design-brief"])
    with pytest.raises(InvalidInputError, match="must reference analytical evidence"):
        to_engineering_state(fixture["summary"], created_at=T, project_id=fixture["project_id"], checks=[bad])
    unknown = CoreValidationCheck(id="check.unknown", status="pass", description="x", evidence_refs=["evidence.nope"])
    with pytest.raises(InvalidInputError, match="unknown evidence"):
        to_engineering_state(fixture["summary"], created_at=T, project_id=fixture["project_id"], checks=[unknown])


def test_extending_base_state_preserves_extensions_and_is_idempotent():
    fixture = illustrative_dab_transformer()
    summary = fixture["summary"]
    base = load_core_example()
    base["extensions"] = {"example.custom": {"independent_domain": 3}}
    frozen = copy.deepcopy(base)
    state = to_engineering_state(summary, created_at=T, base_state=base)
    assert base == frozen, "base_state must not be mutated"
    assert validate_engineering_state(state) == []
    assert state["extensions"] == {"example.custom": {"independent_domain": 3}}
    assert state["id"] == base["id"] and state["project"]["updated_at"] == T
    assert state["project"]["created_at"] == base["project"]["created_at"]
    assert [item["id"] for item in state["evidence"]][0] == "evidence.design-brief"
    assert state["requirements"] == base["requirements"]  # untouched without new requirements
    assert state["validation"]["status"] == "not_run"
    assert from_engineering_state(state) == [summary]
    again = to_engineering_state(summary, created_at=T, base_state=state, replace_existing=True)
    assert again["magnetics"] == state["magnetics"] and again["evidence"] == state["evidence"]
    with pytest.raises(InvalidInputError, match="already contains magnetics entry"):
        to_engineering_state(summary, created_at=T, base_state=state)
    with pytest.raises(InvalidInputError, match="does not match base_state project"):
        to_engineering_state(summary, created_at=T, base_state=base, project_id="project.other")


def test_conflicting_evidence_and_invalid_base_are_rejected():
    fixture = illustrative_dab_transformer()
    summary = fixture["summary"]
    base = export_fixture_state(fixture)
    changed = summary.evidence[0].model_copy(update={"description": "Different wording, same id."})
    conflicting = summary.model_copy(update={"evidence": [changed, *summary.evidence[1:]]})
    with pytest.raises(InvalidInputError, match="already exists in base_state with different content"):
        to_engineering_state(conflicting, created_at=T, base_state=base, replace_existing=True)
    broken = load_core_example()
    broken["requirements"]["evidence_refs"] = []
    with pytest.raises(InvalidInputError, match="base_state is not a valid"):
        to_engineering_state(summary, created_at=T, base_state=broken)
    with pytest.raises(InvalidInputError, match="is not a valid Core Engineering State"):
        from_engineering_state(broken)
    with pytest.raises(InvalidInputError):
        extract_requirements(broken)


@pytest.mark.parametrize("kwargs,match", [
    (dict(created_at="2026-10-19", project_id="project.x"), "RFC 3339"),
    (dict(created_at=T), "project_id is required"),
    (dict(created_at=T, project_id="Project X"), "Core id pattern"),
    (dict(created_at=T, project_id="project.x", project_purpose="demo"), "project_purpose"),
    (dict(created_at=T, project_id="project.x", converter_topology=""), "converter_topology"),
])
def test_invalid_export_arguments(kwargs, match):
    summary = illustrative_dab_transformer()["summary"]
    with pytest.raises(InvalidInputError, match=match):
        to_engineering_state(summary, **kwargs)


def test_summary_model_rejects_invalid_inputs():
    base = illustrative_dab_transformer()["summary"].model_dump()
    with pytest.raises(ValidationError, match="dimension"):
        MagneticComponentSummary.model_validate({**base, "inductance": {"value": 1, "unit": "F"}})
    with pytest.raises(ValidationError, match="non-negative"):
        MagneticComponentSummary.model_validate({**base, "inductance": {"value": -1e-6, "unit": "H"}})
    with pytest.raises(ValidationError, match="positive"):
        MagneticComponentSummary.model_validate({**base, "turns_ratio": {"value": 0, "unit": "1"}})
    with pytest.raises(ValidationError, match="URI reference"):
        MagneticComponentSummary.model_validate({**base, "core_material_ref": "has space"})
    with pytest.raises(ValidationError, match="Core id pattern"):
        MagneticComponentSummary.model_validate({**base, "id": "Magnetics"})
    with pytest.raises(ValidationError, match="requires evidence_refs"):
        MagneticComponentSummary.model_validate({**base, "evidence": []})
    with pytest.raises(ValidationError, match="duplicate"):
        MagneticComponentSummary.model_validate({**base, "evidence": [base["evidence"][0], base["evidence"][0]]})
    with pytest.raises(ValidationError, match="JSON data"):
        MagneticComponentSummary.model_validate({**base, "detail": {"nan": float("nan")}})
    with pytest.raises(InvalidInputError, match="summary is invalid"):
        to_engineering_state({**base, "kind": "capacitor"}, created_at=T, project_id="project.x")
    # a summary without quantities needs no evidence
    bare = MagneticComponentSummary(id="magnetics.unspecified", role="placeholder", kind="other")
    state = to_engineering_state(bare, created_at=T, project_id="project.x")
    assert validate_engineering_state(state) == [] and state["magnetics"][0]["evidence_refs"] == []


def test_evidence_model_rules():
    common = dict(description="x", source={"type": "document", "reference": "ref"},
                  provenance={"created_at": T, "activity": "a", "inputs": []})
    with pytest.raises(ValidationError, match="input evidence ids"):
        CoreEvidence(id="evidence.a", kind="analytical", **common)
    with pytest.raises(ValidationError):
        CoreEvidence(id="evidence.a", kind="measurement", **common)  # kind not emitted by this tool
    with pytest.raises(ValidationError, match="RFC 3339"):
        CoreEvidence(id="evidence.a", kind="assumption", description="x", source={"type": "document", "reference": "r"},
                     provenance={"created_at": "2026-10-19T00:00:00", "activity": "a", "inputs": []})
    with pytest.raises(ValidationError, match="namespaced"):
        CoreEvidence(id="evidence.a", kind="assumption", extensions={"typo": {}}, **common)
    rejected = CoreEvidence(id="evidence.a", kind="assumption", review_status="rejected", **common)
    summary = MagneticComponentSummary(id="magnetics.x", role="r", kind="inductor",
                                       inductance=Quantity(value=1e-3, unit="H"), evidence=[rejected])
    with pytest.raises(InvalidInputError, match="rejected"):
        to_engineering_state(summary, created_at=T, project_id="project.x")
    dangling = CoreEvidence(id="evidence.b", kind="analytical", description="x", source={"type": "document", "reference": "r"},
                            provenance={"created_at": T, "activity": "a", "inputs": ["evidence.absent"]})
    summary = MagneticComponentSummary(id="magnetics.x", role="r", kind="inductor", evidence=[dangling])
    with pytest.raises(InvalidInputError, match="unknown input evidence"):
        to_engineering_state(summary, created_at=T, project_id="project.x")


def test_requirements_model_rules():
    with pytest.raises(ValidationError, match="requires evidence_refs"):
        DesignRequirements(nominal_power=Quantity(value=1, unit="W"))
    evidence = [CoreEvidence(id="evidence.req", kind="assumption", description="x",
                             source={"type": "user_requirement", "reference": "r"},
                             provenance={"created_at": T, "activity": "a", "inputs": []})]
    with pytest.raises(ValidationError, match="dimension"):
        DesignRequirements(nominal_power=Quantity(value=1, unit="V"), evidence=evidence)
    with pytest.raises(ValidationError, match="positive"):
        DesignRequirements(switching_frequency=Quantity(value=0, unit="Hz"), evidence=evidence)
    with pytest.raises(ValidationError, match="non-negative"):
        DesignRequirements(nominal_power=Quantity(value=-1, unit="W"), evidence=evidence)
    partial = DesignRequirements(switching_frequency=Quantity.si(100, "kHz"), evidence=evidence)
    summary = MagneticComponentSummary(id="magnetics.x", role="r", kind="inductor")
    state = to_engineering_state(summary, created_at=T, project_id="project.x", requirements=partial)
    assert extract_requirements(state) == {"switching_frequency": Quantity(value=100000.0, unit="Hz")}


def test_import_rejects_unsupported_evidence_kind_and_inconsistent_extension():
    state = export_fixture_state()
    measured = copy.deepcopy(state)
    measured["evidence"].append({
        "id": "evidence.measured", "kind": "measurement", "description": "Hypothetical measurement record.",
        "source": {"type": "measurement_record", "reference": "lab record"},
        "provenance": {"created_at": T, "activity": "measure", "inputs": [], "tool": {"id": "tool.bench", "version": "1"}},
        "artifacts": [{"uri": "bench.csv", "media_type": "text/csv"}], "review_status": "unreviewed"})
    measured["magnetics"][0]["evidence_refs"].append("evidence.measured")
    assert validate_engineering_state(measured) == []
    with pytest.raises(UnsupportedOperationError, match="measurement"):
        from_engineering_state(measured)
    inconsistent = copy.deepcopy(state)
    inconsistent["magnetics"][0]["extensions"][EXTENSION_KEY]["quantities"]["inductance"]["value"] = 1.0
    with pytest.raises(DataIntegrityError, match="disagrees"):
        from_engineering_state(inconsistent)
    foreign = copy.deepcopy(state)
    foreign["magnetics"][0]["extensions"][EXTENSION_KEY]["schema"] = "other.tool/9"
    with pytest.raises(UnsupportedOperationError, match="unsupported"):
        from_engineering_state(foreign)
    # an entry without our extension imports from Core fields alone
    plain = copy.deepcopy(state)
    del plain["magnetics"][0]["extensions"]
    [imported] = from_engineering_state(plain)
    assert imported.inductance == Quantity(value=state["magnetics"][0]["inductance"]["value"], unit="H")
    assert imported.limitations == [] and imported.detail == {}


# --------------------------------------------------------------------------------------------------
# CLI and verification checks
# --------------------------------------------------------------------------------------------------
def test_cli_validate_reports_pass_and_fail(tmp_path, capsys):
    good = tmp_path / "good.json"
    dump(export_fixture_state(), good)
    bad = tmp_path / "bad.json"
    state = load_core_example()
    state["requirements"]["evidence_refs"] = []
    dump(state, bad)
    assert core_cli(["validate", str(good)]) == 0
    assert "PASS" in capsys.readouterr().out
    assert core_cli(["validate", str(good), str(bad)]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "require evidence_refs" in out
    assert core_cli(["validate", "--json", str(bad)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["passed"] is False and payload["results"][0]["errors"]
    assert core_cli(["info"]) == 0
    assert CORE_COMMIT in capsys.readouterr().out
    result = subprocess.run([sys.executable, "-m", "aipe_magnetics.integrations.aipe_core", "validate", str(good)],
                            capture_output=True, text=True)
    assert result.returncode == 0 and "PASS" in result.stdout, result.stderr


def test_verification_checks_pass():
    for function in (core_example_validates, core_roundtrip):
        result = function()
        assert result.passed, (function.check_id, result.metrics, result.notes)
    assert core_roundtrip().metrics["roundtrip_equal"] is True


# --------------------------------------------------------------------------------------------------
# External: AIPE-Core's own validator on our exported state
# --------------------------------------------------------------------------------------------------
def _checkouts() -> Path:
    value = os.environ.get("AIPE_ECOSYSTEM_CHECKOUTS")
    if not value:
        print(SKIP_REASON)
        pytest.skip(SKIP_REASON)
    return Path(value)


@pytest.mark.external
def test_core_reference_validator_accepts_exported_state(tmp_path):
    core = _checkouts() / "AIPE-Core"
    script = core / "scripts" / "validate.py"
    if not script.is_file():
        pytest.fail(f"AIPE-Core checkout incomplete: {script} not found")
    exported = tmp_path / "exported-state.json"
    dump(export_fixture_state(), exported)
    extended = tmp_path / "extended-example.json"
    base = load_core_example()
    base["extensions"] = {"example.custom": {"independent_domain": 3}}
    dump(to_engineering_state(illustrative_dab_transformer()["summary"], created_at=T, base_state=base), extended)
    result = subprocess.run([sys.executable, str(script), str(exported), str(extended)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS") == 2, result.stdout
    broken = tmp_path / "broken.json"
    state = load_core_example()
    state["evidence"][0]["provenance"]["inputs"] = ["evidence.design-brief"]
    dump(state, broken)
    negative = subprocess.run([sys.executable, str(script), str(broken)], capture_output=True, text=True)
    assert negative.returncode == 1 and "Cyclic" in negative.stdout, negative.stdout + negative.stderr


@pytest.mark.external
def test_vendored_core_files_match_checkout():
    core = _checkouts() / "AIPE-Core"
    mapping = {name: ("examples/dab-10kw/engineering-state.json" if name.startswith("examples/") else f"schemas/{name}")
               for name in VENDORED_SHA256}
    for name, upstream in mapping.items():
        path = core / upstream
        if not path.is_file():
            pytest.fail(f"AIPE-Core checkout incomplete: {path} not found")
        normalised = path.read_bytes().replace(b"\r\n", b"\n")  # checkouts may use CRLF on Windows
        assert hashlib.sha256(normalised).hexdigest() == VENDORED_SHA256[name], name
