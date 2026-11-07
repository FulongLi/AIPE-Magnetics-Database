"""Quality checks: real data are clean of errors; synthetic defects are detected and never altered."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aipe_magnetics_database.ingest.quality import run_checks, summary_markdown

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def report():
    return run_checks(ROOT, validate_schema=False)


def test_committed_data_has_no_errors(report):
    assert report["summary"]["errors"] == 0, [c for c in report["checks"] if c["status"] == "fail"]
    assert report["summary"]["rows_checked"] > 2_000_000
    assert "Schema validity is not proof of experimental accuracy" in report["disclaimer"]


def test_known_upstream_unit_problem_is_flagged(report):
    # MAS advanced_core_materials: half of the Ferroxcube 3C94 B-H cycle is ~-508 "T" (apparently mT); kept, flagged
    rng = next(c for c in report["checks"] if c["id"] == "physical_range")
    assert any(t["record"] == "measurement.mas-ferroxcube-3c94-0036-bh-cycle" for t in rng["top_records"])


def test_markdown_summary(report):
    text = summary_markdown(report)
    assert text.startswith("# Data quality summary") and "| `schema` |" in text


def _write_source(root: Path, rows, columns, quantity="core_loss", conditions=None):
    base = root / "data/normalized/syn"
    (base / "measurements").mkdir(parents=True)
    table = {"layout": "aipe.magnetics.table/0.1", "quantity": quantity, "columns": columns, "row_count": len(rows), "rows": rows}
    if conditions:
        table["conditions"] = conditions
    rec = {"id": "measurement.syn", "kind": "measurement", "extensions": {"aipe.table": table}}
    (base / "measurements/m.jsonl").write_text(json.dumps(rec) + "\n")
    (base / "index.json").write_text(json.dumps({"source_id": "syn", "record_counts": {"measurement": 1}, "shards": {"measurements": ["measurements/m.jsonl"]},
                                                 "datasets": [], "tables_local": []}))


def test_synthetic_defects_are_detected(tmp_path: Path):
    cols = [{"name": "frequency", "unit": "kHz"}, {"name": "flux_density_peak", "unit": "T"}, {"name": "loss_density", "unit": "W/m^3"},
            {"name": "note"}]
    rows = [[1e5, 0.1, 1e4, "a"], [1e5, 0.1, 1e4, "a"], [1e5, 9.0, -5.0, "b"]]
    _write_source(tmp_path, rows, cols)
    before = (tmp_path / "data/normalized/syn/measurements/m.jsonl").read_bytes()
    r = run_checks(tmp_path, validate_schema=False, previous={"record_counts": {"syn": {"measurement": 3}, "gone": {"measurement": 1}}})
    by = {c["id"]: c for c in r["checks"]}
    assert by["units"]["count"] == 2  # frequency in kHz, 'note' without unit
    assert by["physical_range"]["count"] == 2  # B = 9 T, negative loss density
    assert by["duplicate_rows"]["count"] == 1
    assert by["operating_conditions"]["count"] == 1 and by["operating_conditions"]["examples"][0]["missing"] == ["temperature"]
    assert by["record_counts"]["count"] == 2
    assert r["summary"]["errors"] == 2 and "units" in r["summary"]["checks_failed"]
    assert (tmp_path / "data/normalized/syn/measurements/m.jsonl").read_bytes() == before  # nothing altered


def test_non_finite_and_missing_sidecar(tmp_path: Path):
    cols = [{"name": "frequency", "unit": "Hz"}, {"name": "temperature", "unit": "K"}, {"name": "flux_density_peak", "unit": "T"},
            {"name": "loss_density", "unit": "W/m^3"}]
    _write_source(tmp_path, [[1e5, 300.0, 0.1, float("nan")]], cols)
    r = run_checks(tmp_path, validate_schema=False)
    assert {c["id"]: c["count"] for c in r["checks"]}["non_finite"] == 1
