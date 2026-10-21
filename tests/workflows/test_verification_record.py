"""The DAB workflow records only what it executed: caller-supplied timestamps and verification levels that follow from the checks."""

from __future__ import annotations

import pytest

from aipe_magnetics.common.errors import InvalidInputError
from aipe_magnetics.workflows.dab_transformer import WorkflowOptions, build_verification_record, run_dab_workflow

PASSING = [{"kind": "detailed_constraints", "id": "detailed-constraints", "status": "pass", "violations": []},
           {"kind": "energy_balance", "id": "thermal-energy-balance", "status": "pass", "value": 1e-9},
           {"kind": "reference_comparison", "id": "fea-vs-analytical", "status": "recorded", "comparisons": {}}]


def _level(record, name: str):
    return next(lvl for lvl in record.levels if lvl.level == name)


def test_software_schema_level_follows_the_executed_validation():
    pending = build_verification_record(PASSING, schema_validated=None)
    assert _level(pending, "software_schema").status == "partial"
    passed = build_verification_record(PASSING, schema_validated=True)
    assert _level(passed, "software_schema").status == "passed" and "in this run" in _level(passed, "software_schema").statement
    failed = build_verification_record(PASSING, schema_validated=False)
    assert _level(failed, "software_schema").status == "failed"
    for record in (pending, passed, failed):
        assert "release-verification-v0.3.0" not in " ".join(lvl.statement for lvl in record.levels)


def test_numerical_level_is_never_claimed_passed_from_self_checks():
    record = build_verification_record(PASSING, schema_validated=True)
    numerical = _level(record, "numerical_physical")
    assert numerical.status == "partial" and "own checks" in numerical.statement
    failing = build_verification_record(PASSING + [{"kind": "electrothermal", "id": "x", "status": "fail"}], schema_validated=True)
    assert _level(failing, "numerical_physical").status == "failed"
    assert _level(record, "experimental").status == "not_performed" and record.manufacturing_readiness == "preliminary_design"


def test_workflow_requires_a_caller_supplied_timestamp(tmp_path):
    assert WorkflowOptions().created_at is None
    with pytest.raises(InvalidInputError, match="created_at"):
        run_dab_workflow(tmp_path, options=WorkflowOptions())
