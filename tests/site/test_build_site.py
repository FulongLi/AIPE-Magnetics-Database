"""The GitHub Pages explorer is built from the committed data store without changing it, and publishes values unchanged."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

from aipe_magnetics_database import api
from aipe_magnetics_database.store import MagneticDatabase

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import build_site  # noqa: E402


def _digest(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    out = tmp_path_factory.mktemp("pages") / "AIPE-Magnetics-Database"
    watched = [p for p in (ROOT / "data" / "canonical").glob("*")] + [ROOT / "data" / "sources.json"] + \
              list((ROOT / "data" / "normalized" / "princeton-magnet").rglob("*.jsonl"))
    before = _digest(watched)
    result = build_site.SiteBuilder(out, commit="0123456789abcdef").run()
    assert _digest(watched) == before, "the site build must not modify canonical or normalised records"
    return out, result


def _json(out, rel):
    return json.loads((out / "data" / rel).read_text(encoding="utf-8"))


def test_static_shell_is_subpath_safe(site):
    out, _ = site
    assert (out / ".nojekyll").is_file()
    html = (out / "index.html").read_text(encoding="utf-8")
    # every local reference is relative, so the site works under /AIPE-Magnetics-Database/
    assert not re.search(r'(src|href)="/(?!/)', html)
    assert not re.search(r"<script[^>]+src=\"https?://", html), "no third-party scripts"
    js = (out / "assets" / "app.js").read_text(encoding="utf-8")
    assert 'load("data/' in js and not re.search(r'load\("/', js)


def test_counts_match_the_database(site):
    out, result = site
    db = MagneticDatabase()
    summary = _json(out, "summary.json")
    assert summary["counts"]["canonical_materials"] == db.summary()["canonical"]["canonical_materials"] == len(_json(out, "materials.json"))
    assert summary["counts"]["measurement_records"] == len(_json(out, "measurements.json")) == len(db.measurements(include_duplicates=True))
    assert summary["counts"]["core_shapes"] + summary["counts"]["core_parts"] == len(db.cores())
    assert summary["counts"]["fitted_loss_models"] == len(db.loss_models())
    assert result["tables_full"] + result["tables_sample"] + result["tables_none"] == result["measurements"]


def test_material_detail_uses_real_records(site):
    out, _ = site
    n87 = _json(out, "materials/material.canonical-tdk-n87.json")
    assert n87["record"] == api.get_material("TDK N87")
    assert {m["source_id"] for m in n87["members"]} >= {"openmagnetics_mas", "princeton_magnet", "paderborn_lea_materialdatabase"}
    assert "measurement.princeton-magnet-n87-core-loss" in {m["id"] for m in n87["measurements"]}
    assert all(m["data_class"] == "fitted_parameters" for m in n87["loss_models"])
    row = next(m for m in _json(out, "materials.json") if m["id"] == "material.canonical-tdk-n87")
    assert row["manufacturer"] == "TDK" and row["bsat"]["u"] == "T" and row["bsat"]["T"] == pytest.approx(298.15)


def test_missing_values_are_absent_not_zero(site):
    out, _ = site
    canonical = {r["id"]: r for r in MagneticDatabase().materials()}
    for m in _json(out, "materials.json"):
        props = canonical[m["id"]]["properties"]
        for key, prop in build_site.HEADLINE.items():
            assert (key in m) == (prop in props and build_site.headline(props[prop]) is not None)


def test_full_tables_are_published_unchanged(site):
    out, _ = site
    rid = "measurement.paderborn-lea-n27-complex-permeability-magnet"
    page = _json(out, f"measurements/{rid}.json")
    table = api.get_table(rid)
    assert 0 < table["total_rows"] <= build_site.PREVIEW_ROWS
    assert page["sampling"]["sampled"] is False and page["rows"] == table["rows"] and page["columns"] == table["columns"]
    assert page["entry"]["license"] == "GPL-3.0-only" and page["dataset"]["redistribution_permitted"] is True


def test_large_tables_are_labelled_samples_of_the_real_rows(site):
    out, _ = site
    rid = "measurement.princeton-magnet-n87-core-loss"
    page = _json(out, f"measurements/{rid}.json")
    full = MagneticDatabase().table(rid).rows
    s = page["sampling"]
    assert s["sampled"] is True and s["total_rows"] == len(full) == 142871 and s["published_rows"] == len(page["rows"]) <= build_site.PREVIEW_ROWS
    assert "Systematic sample" in s["method"]
    # every published row is an upstream row, in upstream order
    positions = {json.dumps(r): i for i, r in enumerate(full)}
    idx = [positions[json.dumps(r)] for r in page["rows"]]
    assert idx == sorted(idx)
    # every (waveform, temperature) group of the full table is represented
    names = [c["name"] for c in page["columns"]]
    key = lambda r: (r[names.index("waveform")], r[names.index("temperature")])  # noqa: E731
    assert {key(r) for r in page["rows"]} == {key(r) for r in full}
    assert page["links"]["rows_file"].endswith("/data/normalized/princeton-magnet/tables/princeton-magnet-n87-core-loss.jsonl")
    assert page["links"]["rows_file"].startswith("https://raw.githubusercontent.com/FulongLi/AIPE-Magnetics-Database/0123456789abcdef/")


def test_uncommitted_sidecars_are_listed_without_rows(site):
    out, _ = site
    db = MagneticDatabase()
    for entry in _json(out, "measurements.json"):
        rec = db.measurement(entry["id"])
        if (rec["extensions"]["aipe.table"].get("rows_file") or "").startswith("tables-local/"):
            page = _json(out, f"measurements/{entry['id']}.json")
            assert entry["published"] == "none" and page["rows"] == [] and "refresh" in page["unavailable"]


def test_sample_rows_is_deterministic_and_keeps_groups():
    columns = [{"name": "waveform"}, {"name": "temperature"}, {"name": "x"}]
    rows = [["sine" if i % 10 else "triangle", 298.15 + 25 * (i % 3), float(i)] for i in range(1000)]
    a, meta = build_site.sample_rows(columns, rows, limit=100)
    assert a == build_site.sample_rows(columns, rows, limit=100)[0]
    assert meta["sampled"] and meta["total_rows"] == 1000 and len(a) <= 100
    assert {(r[0], r[1]) for r in a} == {(r[0], r[1]) for r in rows}
    small, meta = build_site.sample_rows(columns, rows[:50], limit=100)
    assert small == rows[:50] and meta["sampled"] is False
