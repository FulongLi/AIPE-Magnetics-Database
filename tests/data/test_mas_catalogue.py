"""The MAS catalogue: every committed record validates, and the first import (v0.2 batch) is preserved."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from aipe_magnetics_database.data.records import MagneticRecord, validate_catalogue

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "normalized" / "mas"
FIRST_IMPORT_COMMIT = "1bdf6ce"  # main after PR #5/#6: 1,137 materials + 2,294 shapes


def _records(group: str) -> list[dict]:
    index = json.loads((DATA / "index.json").read_text(encoding="utf-8"))
    return [json.loads(line) for rel in index["shards"][group] for line in (DATA / rel).read_text(encoding="utf-8").splitlines()]


def test_mas_records_validate_and_resolve():
    index = json.loads((DATA / "index.json").read_text(encoding="utf-8"))
    records, counts = [], {}
    for group, files in index["shards"].items():
        for rel in files:
            for line in (DATA / rel).read_text(encoding="utf-8").splitlines():
                record = MagneticRecord.model_validate_json(line)
                assert record.license == "Apache-2.0" and record.evidence[0].review_status == "unreviewed"
                assert "third_party_unreviewed" in record.quality.flags
                counts[record.kind] = counts.get(record.kind, 0) + 1
                records.append(record)
    records += [MagneticRecord.model_validate_json((DATA / rel).read_text(encoding="utf-8")) for rel in index["datasets"]]
    counts["dataset"] = len(index["datasets"])
    assert counts == index["record_counts"]
    assert counts["magnetic_material"] == 1137 and counts["core_geometry"] == 2294 + 18943
    assert not validate_catalogue(records)


def test_shape_dimensions_retain_tolerance_not_fabricated_nominals():
    for record in _records("shapes"):
        assert not {"effective_area", "effective_length", "effective_volume"} & set(record["properties"])
        mas = record["extensions"]["aipe.mas"]
        for dims in mas["dimensions_range_m"].values():
            assert set(dims) <= {"minimum_m", "maximum_m"}
            if len(dims) == 2:
                assert dims["minimum_m"] <= dims["maximum_m"]
        inconsistent = mas.get("inconsistent_dimension_ranges_m", {})
        assert bool(inconsistent) == ("upstream_dimension_range_inverted" in record["quality"]["flags"])


def test_core_parts_carry_no_effective_parameters():
    for record in _records("cores")[:2000]:
        assert "no_effective_parameters" in record["quality"]["flags"]
        assert not {"effective_area", "effective_length", "effective_volume"} & set(record["properties"])


def _git_show(path: str) -> str | None:
    proc = subprocess.run(["git", "show", f"{FIRST_IMPORT_COMMIT}:{path}"], capture_output=True, text=True, cwd=ROOT)
    return proc.stdout if proc.returncode == 0 else None


def _subset(old, new) -> bool:
    if isinstance(old, dict):
        return isinstance(new, dict) and all(k in new and _subset(v, new[k]) for k, v in old.items())
    if isinstance(old, list):
        return isinstance(new, list) and len(old) == len(new) and all(_subset(a, b) for a, b in zip(old, new))
    return old == new


@pytest.mark.parametrize("group, prefix, shards", [("materials", "materials", 12), ("shapes", "shapes", 5)])
def test_first_import_ids_and_property_values_are_preserved(group, prefix, shards):
    if _git_show(f"data/normalized/mas/{group}/{prefix}-001.jsonl") is None:
        pytest.skip(f"git history with commit {FIRST_IMPORT_COMMIT} not available (shallow clone)")
    new = {r["id"]: r for r in _records(group)}
    for n in range(1, shards + 1):
        for line in _git_show(f"data/normalized/mas/{group}/{prefix}-{n:03d}.jsonl").splitlines():
            old = json.loads(line)
            assert old["id"] in new, old["id"]
            assert _subset(old["properties"], new[old["id"]]["properties"]), old["id"]
