"""The packaged MES 0.3.0 JSON Schema exists, is in sync with the model and validates a minimal state."""

from __future__ import annotations

import json
from pathlib import Path

from aipe_magnetics.state import PACKAGED_SCHEMA, STATE_SCHEMA, check_state_schema, state_json_schema, validate_state

ROOT = Path(__file__).resolve().parents[2]
MIRROR = ROOT / "schemas" / "state" / "v0.3.0" / "magnetic-engineering-state.schema.json"


def test_packaged_schema_exists_and_matches_a_fresh_export():
    assert PACKAGED_SCHEMA.is_file(), f"missing packaged schema {PACKAGED_SCHEMA} (export with aipe_magnetics.state.export_state_schema())"
    assert check_state_schema() == []
    assert MIRROR.is_file() and json.loads(MIRROR.read_text(encoding="utf-8")) == state_json_schema()
    schema = state_json_schema()
    assert schema["$id"].endswith("schemas/state/v0.3.0/magnetic-engineering-state.schema.json") and schema["title"].endswith("0.3.0")


def test_minimal_state_validates_and_broken_references_are_reported():
    state = {"state_schema": STATE_SCHEMA, "state_version": "0.3.0", "id": "mes.test", "title": "test", "component": {"type": "transformer"},
             "provenance": {"tool": {"version": "0.2.0"}}}
    assert validate_state(state) == []
    broken = dict(state, solvers=[], results={"electromagnetic": [{"id": "r", "solver_ref": "solver.missing", "quantities": {}}]})
    assert any("unknown solver" in e for e in validate_state(broken))
