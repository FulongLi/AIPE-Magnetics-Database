"""Importers of the 2026-11 open laboratory sources: Nabih (Access export), Guillod (MAT), INRIM (text blocks), Zenodo 7368936 (CSV)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from aipe_magnetics_database.common.errors import DataIntegrityError
from aipe_magnetics_database.ingest.sources import openlab

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
NORM = ROOT / "data" / "normalized"


def _records(directory: Path, group: str = "measurements") -> dict[str, dict]:
    out = {}
    for path in sorted((directory / group).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def _rows(directory: Path, rec: dict) -> list[list]:
    table = rec["extensions"]["aipe.table"]
    if "rows" in table:
        return table["rows"]
    return [json.loads(line) for line in (directory / table["rows_file"]).read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("source_id, raw_dir", [("nabih_magneto", "nabih-magneto"), ("guillod_n87_dielectric", "guillod-n87-dielectric"),
                                                ("zenodo_10725890_inrim", "zenodo-10725890-inrim")])
def test_offline_reimport_is_byte_identical(source_id: str, raw_dir: str, tmp_path: Path):
    pytest.importorskip("scipy")
    fn, directory = openlab.IMPORTERS[source_id]
    fn(tmp_path, local_dir=RAW / raw_dir / "files")
    committed = NORM / directory
    for path in (tmp_path / directory).rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (committed / path.relative_to(tmp_path / directory)).read_bytes(), path
    first = {p: p.read_bytes() for p in (tmp_path / directory).rglob("*") if p.is_file()}
    fn(tmp_path, local_dir=RAW / raw_dir / "files")
    assert first == {p: p.read_bytes() for p in (tmp_path / directory).rglob("*") if p.is_file()}  # idempotent


def test_decimal_unit_scaling_is_exact():
    assert openlab.dec("4.35", "100") == 435.0 and 4.35 * 100 != 435.0  # binary scaling would add 1e-14
    assert openlab.dec("20.015", "0.001") == 0.020015 and openlab.dec("25.0", "1", "273.15") == 298.15


def test_nabih_mapping_and_exclusions():
    recs = _records(NORM / "nabih-magneto")
    loss = recs["measurement.nabih-ml91s-core-loss"]
    cols = [c["name"] for c in loss["extensions"]["aipe.table"]["columns"]]
    row = _rows(NORM / "nabih-magneto", loss)[0]
    # first point of test 1: 1.0 MHz, 20.015 mT, 10.675237 kW/m^3, 25 degC
    assert dict(zip(cols, row))["frequency"] == 1e6 and row[cols.index("flux_density_peak")] == 0.020015
    assert row[cols.index("loss_density")] == 10675.237 and row[cols.index("temperature")] == 298.15
    assert recs["measurement.nabih-dmr96a-complex-permeability"]["extensions"]["aipe.data"]["data_class"] == "manufacturer_datasheet"
    assert recs["measurement.nabih-ml95s-complex-permittivity"]["extensions"]["aipe.data"]["data_class"] == "experimental_measurement"
    report = json.loads((ROOT / "data/reports/import-nabih_magneto.json").read_text(encoding="utf-8"))
    excluded = {e["what"]: 0 for e in report["exclusions"]}
    for e in report["exclusions"]:
        excluded[e["what"]] += e["count"]
    assert excluded == {"Bm vs Pv table": 41, "permeability": 402, "permittivity": 607}
    points = sum(r["extensions"]["aipe.table"]["row_count"] for k, r in recs.items() if k.endswith("core-loss"))
    assert points == 429 - 41
    mats = _records(NORM / "nabih-magneto", "materials")
    assert {m["name"] for m in mats.values()} >= {"Proterial ML91S", "ACME P63", "DMEGC DMR51W", "Ferroxcube 3F36"}


def test_nabih_export_must_match_the_original(tmp_path: Path):
    shutil.copytree(RAW / "nabih-magneto", tmp_path / "nabih-magneto")
    tsv = tmp_path / "nabih-magneto/derived/Core_loss_database.tsv"
    tsv.write_text(tsv.read_text(encoding="utf-8").replace("ML91S", "ML91X", 1), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="EXPORT.json"):
        openlab.run_nabih(tmp_path / "out", local_dir=tmp_path / "nabih-magneto/files")


def test_guillod_reads_full_precision_mat_and_skips_permeability():
    pytest.importorskip("scipy")
    import numpy as np
    recs = _records(NORM / "guillod-n87-dielectric")
    assert set(recs) == {"measurement.guillod-n87-small-signal-permittivity", "measurement.guillod-n87-permittivity-vs-temperature",
                         "measurement.guillod-n87-large-signal-permittivity"}
    small = recs["measurement.guillod-n87-small-signal-permittivity"]
    rows = np.array([r[2] for r in _rows(NORM / "guillod-n87-dielectric", small)])
    csv = np.loadtxt(RAW / "guillod-n87-dielectric/files/dataset/N87_eps_small/var_eps_vec.csv", delimiter=",")
    assert np.allclose(rows, csv, rtol=1e-12, atol=0) and not np.array_equal(rows, csv)  # MAT keeps digits the CSV rounds away
    large = recs["measurement.guillod-n87-large-signal-permittivity"]
    assert [c["name"] for c in large["extensions"]["aipe.table"]["columns"]][2] == "electric_field_peak"
    assert large["extensions"]["aipe.table"]["row_count"] == 60
    assert not (RAW / "guillod-n87-dielectric/files/dataset/N87_mu_small").exists()


def test_inrim_blocks_units_and_column_order():
    text = (RAW / "zenodo-10725890-inrim/files" / openlab.INRIM_FILE).read_text(encoding="utf-8", errors="replace")
    blocks = openlab.parse_inrim(text)
    assert len(blocks) == 22 and sum(len(b[2]) for b in blocks) == 527
    assert {b[0] for b in blocks} >= {"430 F", "430FR", "FeSiAl", "SMC", "CAST IRON"}
    recs = _records(NORM / "zenodo-10725890-inrim")
    c10 = recs["measurement.inrim-c10-first-magnetization"]
    assert [c["name"] for c in c10["extensions"]["aipe.table"]["columns"]] == ["field", "flux_density"]  # published as 'B H'
    assert c10["extensions"]["aipe.table"]["rows"][0] == [14.051, 0.00711]
    assert "units_not_stated" in c10["quality"]["flags"]


def test_cwh_peak_definition_and_duplicate_link():
    directory = NORM / "zenodo-7368936-cwh-n87"
    recs = _records(directory)
    amb = recs["measurement.zenodo-7368936-n87-ambient-core-loss"]
    cols = [c["name"] for c in amb["extensions"]["aipe.table"]["columns"]]
    rows = _rows(directory, amb)
    assert len(rows) == 12341
    ip, ipp, b = cols.index("flux_density_peak"), cols.index("flux_density_peak_to_peak"), [cols.index(f"waveform_b{i}") for i in range(1, 6)]
    for r in rows:
        assert r[ip] == r[ipp] / 2 and abs((max(r[i] for i in b) - min(r[i] for i in b)) - r[ipp]) < 1e-12
    var = recs["measurement.zenodo-7368936-n87-variable-temperature-core-loss"]
    assert var["extensions"]["aipe.table"]["rows_file"].startswith("tables-local/")  # MagNet repeat kept out of Git
    dup = {d.get("record"): d for d in json.loads((ROOT / "data/canonical/duplicates.json").read_text(encoding="utf-8"))}
    assert dup[var["id"]]["duplicate_of"] == "measurement.princeton-magnet-n87-core-loss"
    assert amb["id"] not in dup


def test_new_materials_consolidate_with_mas():
    canon = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data/canonical/materials.jsonl").read_text(encoding="utf-8").splitlines()}
    members = {m["source_id"] for m in canon["material.canonical-proterial-ml91s"]["extensions"]["aipe.canonical"]["members"]}
    assert members == {"openmagnetics_mas", "nabih_magneto"}
    n87 = {m["source_id"] for m in canon["material.canonical-tdk-n87"]["extensions"]["aipe.canonical"]["members"]}
    assert {"guillod_n87_dielectric", "zenodo_7368936_cwh_n87", "princeton_magnet"} <= n87
