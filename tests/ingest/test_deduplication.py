"""Source-aware duplicate detection (unit level with small synthetic sources, and on the committed catalogue)."""
from __future__ import annotations

import json
from pathlib import Path

from aipe_magnetics_database.ingest import canonical
from aipe_magnetics_database.ingest.common import column, evidence, finish_source, measurement_record, ImportSummary, SourceWriter, validated

ROOT = Path(__file__).resolve().parents[2]
T = "2026-01-01T00:00:00Z"


def _material(mid: str, name: str, grade: str, manufacturer: str, source: str) -> dict:
    ev = [evidence("evidence.s", "source", "test", "test", T, "MIT")]
    return validated({"schema_version": "0.2.0", "id": mid, "kind": "magnetic_material", "name": name, "synthetic": False, "license": "MIT",
                      "properties": {"grade": grade}, "evidence_refs": ["evidence.s"], "evidence": ev,
                      "extensions": {"aipe.mas": {"manufacturer": manufacturer}, "aipe.data": {"data_class": "identity", "source_id": source}}})


def _loss(rid: str, material: str, rows: list, source: str, underlying: str | None, data_class: str = "experimental_measurement") -> dict:
    ev = [evidence("evidence.s", "source", "t", "t", T, "MIT"), evidence("evidence.m", "measurement", "t", "t", T, "MIT")]
    return measurement_record(rid=rid, name=rid, material_ref=material, license="MIT", evidence_items=ev, data_class=data_class, source_id=source,
                              quantity="core_loss", columns=[column("frequency", "f"), column("flux_density_peak", "B"), column("temperature", "T"),
                                                             column("loss_density", "P")],
                              rows=rows, writer=None, provenance={"activity": "test"}, underlying_source=underlying)


def _source(root: Path, directory: str, source: str, records: list[dict]) -> None:
    summary = ImportSummary(source, "v", "MIT", "permitted")
    writer = SourceWriter(root / directory, summary)
    mats = [r for r in records if r["kind"] == "magnetic_material"]
    meas = [r for r in records if r["kind"] == "measurement"]
    finish_source(writer, summary, {"materials": writer.shards(mats, "materials", "materials", 100),
                                    "measurements": writer.shards(meas, "measurements", "measurements", 100)}, [])


def test_republished_experiment_identical_rows_and_cross_material_copies(tmp_path: Path):
    rows = [[100e3, 0.1, 298.15, 1.0e4], [200e3, 0.1, 298.15, 3.0e4]]
    _source(tmp_path, "magnet", "princeton_magnet", [_material("material.princeton-magnet-n87", "TDK N87", "N87", "TDK", "princeton_magnet"),
                                                      _loss("measurement.p-n87", "material.princeton-magnet-n87", rows, "princeton_magnet",
                                                            "princeton-magnet:n87")])
    rounded = [[r[0] + 7, r[1], r[2], r[3]] for r in rows]  # re-publication with re-estimated frequency (< 0.1 %)
    _source(tmp_path, "mas", "openmagnetics_mas", [
        _material("material.mas-n87", "TDK N87", "N87", "TDK", "openmagnetics_mas"),
        _material("material.mas-n97", "TDK N97", "N97", "TDK", "openmagnetics_mas"),
        _loss("measurement.mas-n87-magnet", "material.mas-n87", rounded, "openmagnetics_mas", "princeton-magnet:n87"),
        _loss("measurement.mas-n87-ds", "material.mas-n87", [[1e5, 0.2, 373.15, 5e4], [2e5, 0.2, 373.15, 9e4]], "openmagnetics_mas", None,
              "manufacturer_datasheet"),
        _loss("measurement.mas-n87-ds-copy", "material.mas-n87", [[1e5, 0.2, 373.15, 5e4], [2e5, 0.2, 373.15, 9e4]], "openmagnetics_mas", None,
              "manufacturer_datasheet"),
        _loss("measurement.mas-n97-ds", "material.mas-n97", [[1e5, 0.2, 373.15, 5e4], [2e5, 0.2, 373.15, 9e4]], "openmagnetics_mas", None,
              "manufacturer_datasheet")])
    index = canonical.build(tmp_path, tmp_path / "canonical", {"princeton_magnet": {"directory": "magnet"}, "openmagnetics_mas": {"directory": "mas"}}, [])
    entries = {json.loads(line)["id"]: json.loads(line) for line in (tmp_path / "canonical" / "measurements.jsonl").read_text().splitlines()}
    assert entries["measurement.mas-n87-magnet"]["status"] == "duplicate"
    assert entries["measurement.mas-n87-magnet"]["duplicate_of"] == "measurement.p-n87"
    assert entries["measurement.mas-n87-magnet"]["duplicate_match"]["match_fraction"] == 1.0
    assert entries["measurement.p-n87"]["status"] == "primary"
    assert entries["measurement.mas-n87-ds-copy"]["status"] == "duplicate"           # identical rows, same material
    assert entries["measurement.mas-n97-ds"]["status"] == "primary"                  # identical rows, other material: flagged, kept
    assert "identical_table_in_other_material" in entries["measurement.mas-n97-ds"]["flags"]
    mats = [json.loads(line) for line in (tmp_path / "canonical" / "materials.jsonl").read_text().splitlines()]
    n87 = next(m for m in mats if m["id"] == "material.canonical-tdk-n87")
    assert {m["source_id"] for m in n87["extensions"]["aipe.canonical"]["members"]} == {"princeton_magnet", "openmagnetics_mas"}
    assert index["counts"]["canonical_materials"] == 2 and index["counts"]["independent_measurement_records"] == 3


def test_committed_catalogue_detects_magnet_republications():
    duplicates = json.loads((ROOT / "data" / "canonical" / "duplicates.json").read_text(encoding="utf-8"))
    magnet = {d["record"]: d for d in duplicates if d.get("rule") == "same underlying experiment and quantity"}
    assert set(magnet) == {f"measurement.mas-ferroxcube-{g}-core-loss-advanced-magnet" for g in ("3c90-0032", "3c94-0036", "3f4-0044")}
    assert all(d["match_fraction"] == 1.0 for d in magnet.values())  # every MAS 'origin = MagNet' row is in the Princeton tables
    derived = [d for d in duplicates if d.get("rule", "").startswith("derived quantity")]
    assert len(derived) == 11 and all(not d["counted_as_independent"] for d in derived)


def test_identity_keys_normalise_manufacturer_spellings():
    assert canonical.manufacturer_key("fair-rite") == canonical.manufacturer_key("Fair-Rite") == "fair-rite"
    assert canonical.manufacturer_key("EPCOS") == "tdk"
    assert canonical.grade_key("Kool Mu 26") == canonical.grade_key("KoolMu-26")
