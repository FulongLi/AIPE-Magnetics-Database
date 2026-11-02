"""Validation of the imported, third-party MAS catalogue (offline, data-only)."""
from __future__ import annotations

import json
from pathlib import Path

from aipe_magnetics.data.records import MagneticRecord, validate_catalogue

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "normalized" / "mas"


def test_mas_catalogue_records_validate_and_resolve():
    index = json.loads((DATA / "index.json").read_text(encoding="utf-8"))
    records = []
    counts = {"magnetic_material": 0, "core_geometry": 0, "dataset": 0}
    ids = set()
    paths = index["material_shards"] + index["shape_shards"] + index["datasets"]
    for relative_path in paths:
        path = DATA / relative_path
        assert path.is_file(), relative_path
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines, relative_path
        for line in lines:
            record = MagneticRecord.model_validate(json.loads(line))
            assert record.id not in ids, record.id
            ids.add(record.id)
            assert record.license == "Apache-2.0"
            assert record.evidence[0].review_status == "unreviewed"
            assert record.quality is not None
            assert "third_party_unreviewed" in record.quality.flags
            counts[record.kind] += 1
            records.append(record)
    assert counts == index["record_counts"]
    assert counts == {"magnetic_material": 1137, "core_geometry": 2294, "dataset": 2}
    assert not validate_catalogue(records)
    assert all(record.dataset_ref in ids for record in records if record.dataset_ref)


def test_mas_shape_dimensions_retain_tolerance_not_fabricated_nominals():
    for filename in sorted((DATA / "shapes").glob("*.jsonl")):
        for line in filename.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            assert "effective_area" not in record["properties"]
            assert "effective_length" not in record["properties"]
            assert "effective_volume" not in record["properties"]
            ranges = record["extensions"]["aipe.mas"]["dimensions_range_m"]
            for dimensions in ranges.values():
                for key in dimensions:
                    assert key in {"minimum_m", "maximum_m"}
                if "minimum_m" in dimensions and "maximum_m" in dimensions:
                    assert dimensions["minimum_m"] <= dimensions["maximum_m"]
