"""Source-aware deduplication and the canonical catalogue (``data/canonical/``).

Materials
    Records of all sources are grouped by a normalised ``(manufacturer, grade)`` identity key. Each group becomes one
    canonical ``magnetic_material`` record ``material.canonical-<manufacturer>-<grade>`` whose typed properties are
    copied from a single *primary* member (precedence: AIPE curated catalogue > MAS > Paderborn LEA > other sources), so
    values are never mixed across sources. ``extensions["aipe.canonical"]`` lists every member record and the
    canonical measurement entries of the material. Laboratory specimens without a manufacturer grade stay separate.

Measurements
    Every ``measurement`` record becomes one entry of ``measurements.jsonl`` with a condition summary (ranges of
    frequency, peak flux density, temperature, DC bias; waveform labels) used by the query API. Duplicate detection
    is source aware and conservative:

    1. *same underlying experiment, same quantity* (``aipe.data.underlying_source``, e.g. ``princeton-magnet:3c90``):
       the record of the originating source is ``primary``; re-publications (MAS ``origin = MagNet`` points) are
       ``duplicate`` with ``duplicate_of`` and a row-level match statistic;
    2. *same underlying experiment, different quantity* (Paderborn mu', mu'' recomputed from MagNet): ``derived_view``
       - kept, but not counted as an independent measurement;
    3. *identical rows* (SHA-256 of the canonicalised rows) in different records: ``duplicate``.

    Nothing is deleted: duplicates stay in ``data/normalized`` and remain queryable with ``include_duplicates=True``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from aipe_magnetics_database import __version__
from aipe_magnetics_database.ingest.common import dumps_line, dumps_pretty, slug, validated, write_if_changed

SOURCE_PRECEDENCE = ["aipe_curated", "openmagnetics_mas", "paderborn_lea_materialdatabase", "princeton_magnet",
                     "zenodo_21626116_smc_ferrite_triangular", "zenodo_8292861_hefmag", "nabih_magneto", "guillod_n87_dielectric",
                     "zenodo_7368936_cwh_n87", "zenodo_10725890_inrim", "princeton_magnet_scope"]
ORIGINATING = {"princeton-magnet": "princeton_magnet", "hefmag": "zenodo_8292861_hefmag", "zenodo-21626116": "zenodo_21626116_smc_ferrite_triangular",
               "nabih": "nabih_magneto", "guillod": "guillod_n87_dielectric", "zenodo-7368936": "zenodo_7368936_cwh_n87",
               "inrim": "zenodo_10725890_inrim", "princeton-magnet-scope": "princeton_magnet_scope"}
MANUFACTURER_ALIASES = {"epcos": "tdk", "tdk electronics": "tdk", "tdk-epcos": "tdk", "fair rite": "fair-rite", "fairrite": "fair-rite",
                        "magnetics inc": "magnetics", "magnetics inc.": "magnetics", "ferroxcube": "ferroxcube", "proterial": "proterial",
                        "hitachi metals": "proterial", "vacuumschmelze": "vac"}
CONDITION_COLUMNS = {"frequency": "frequency", "flux_density_peak": "flux_density_peak", "temperature": "temperature",
                     "dc_bias_field": "dc_bias_field", "duty": "duty", "duty_p": "duty"}


def manufacturer_key(name: str | None) -> str | None:
    if not name:
        return None
    key = re.sub(r"\s+", " ", name.strip().casefold())
    key = MANUFACTURER_ALIASES.get(key, key)
    return slug(key)


def grade_key(grade: str | None) -> str | None:
    return re.sub(r"[^a-z0-9]", "", grade.casefold()) if grade else None


def record_manufacturer(record: dict[str, Any]) -> str | None:
    ext = record.get("extensions") or {}
    for ns in ("aipe.mas", "aipe.princeton", "aipe.paderborn"):
        if isinstance(ext.get(ns), dict) and ext[ns].get("manufacturer"):
            return ext[ns]["manufacturer"]
    ref = (record.get("properties") or {}).get("manufacturer_ref")
    if ref:
        return ref.split(".", 1)[-1]
    name = record.get("name") or ""
    grade = (record.get("properties") or {}).get("grade") or ""
    if grade and name.endswith(grade) and name[: -len(grade)].strip():
        return name[: -len(grade)].strip()
    return None


def identity_key(record: dict[str, Any]) -> str:
    m, g = manufacturer_key(record_manufacturer(record)), grade_key((record.get("properties") or {}).get("grade"))
    if m and g and not record.get("synthetic"):
        return f"{m}-{g}"
    return f"record-{record['id']}"


# -------------------------------------------------------------------------------------------------
# condition summaries
# -------------------------------------------------------------------------------------------------
def table_rows(record: dict[str, Any], source_root: Path) -> Iterable[list[Any]]:
    table = record["extensions"]["aipe.table"]
    if "rows" in table:
        yield from table["rows"]
        return
    path = source_root / table["rows_file"]
    if not path.is_file():
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def summarise(record: dict[str, Any], source_root: Path) -> tuple[dict[str, Any], str | None, bool]:
    """``(conditions, rows_sha256, rows_available)`` of a measurement table."""
    table = record["extensions"]["aipe.table"]
    names = [c["name"] for c in table["columns"]]
    index = {CONDITION_COLUMNS[n]: i for i, n in enumerate(names) if n in CONDITION_COLUMNS}
    wf = names.index("waveform") if "waveform" in names else None
    lo: dict[str, float] = {}
    hi: dict[str, float] = {}
    waveforms: set[str] = set()
    digest = hashlib.sha256()
    n = 0
    for row in table_rows(record, source_root):
        n += 1
        digest.update(json.dumps(row, separators=(",", ":")).encode())
        for key, i in index.items():
            v = row[i]
            if isinstance(v, (int, float)) and math.isfinite(v):
                lo[key] = min(lo.get(key, v), v)
                hi[key] = max(hi.get(key, v), v)
        if wf is not None and row[wf]:
            waveforms.add(row[wf])
    available = n == table["row_count"]
    conditions: dict[str, Any] = {k: {"min": lo[k], "max": hi[k], "unit": {"frequency": "Hz", "flux_density_peak": "T", "temperature": "K",
                                                                            "dc_bias_field": "A/m", "duty": "1"}[k]} for k in sorted(lo)}
    if wf is not None:
        conditions["waveforms"] = sorted(waveforms)
    elif "waveform" in (table.get("conditions") or {}):
        conditions["waveforms"] = [table["conditions"]["waveform"]]
    return conditions, (digest.hexdigest() if available else table.get("rows_sha256")), available


def _row_key(row: list[Any], cols: list[str]) -> tuple | None:
    try:
        f = row[cols.index("frequency")]
        b = row[cols.index("flux_density_peak")]
        t = row[cols.index("temperature")]
        p = row[cols.index("loss_density")]
    except (ValueError, IndexError):
        return None
    # MagNet frequency is a 10 Hz Welch estimate; re-publications round differently -> compare at 1e-3 relative in f, B, P_v
    return (round(t, 1), round(math.log10(f), 3), round(math.log10(b), 3), round(math.log10(p), 3)) if min(f, b, p) > 0 else None


# -------------------------------------------------------------------------------------------------
# build
# -------------------------------------------------------------------------------------------------
def build(normalized_root: Path, canonical_root: Path, sources: dict[str, dict[str, Any]], curated: list[dict[str, Any]]) -> dict[str, Any]:
    """Write ``materials.jsonl``, ``measurements.jsonl``, ``duplicates.json`` and ``index.json``; returns the index."""
    # synthetic (illustrative) curated records are not part of the canonical catalogue of real materials
    materials: list[tuple[str, dict[str, Any]]] = [("aipe_curated", r) for r in curated if r["kind"] == "magnetic_material" and not r.get("synthetic")]
    measurements: list[tuple[str, Path, dict[str, Any]]] = []
    for source_id, info in sorted(sources.items()):
        root = normalized_root / info["directory"]
        index = json.loads((root / "index.json").read_text(encoding="utf-8"))
        for group, files in index["shards"].items():
            for rel in files:
                for line in (root / rel).read_text(encoding="utf-8").splitlines():
                    rec = json.loads(line)
                    if rec["kind"] == "magnetic_material":
                        materials.append((source_id, rec))
                    elif rec["kind"] == "measurement":
                        measurements.append((source_id, root, rec))
    # --- materials -----------------------------------------------------------------------------------------
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    member_key: dict[str, str] = {}
    for source_id, rec in materials:
        key = identity_key(rec)
        groups[key].append((source_id, rec))
        member_key[rec["id"]] = key
    rank = {s: i for i, s in enumerate(SOURCE_PRECEDENCE)}

    # --- measurements ----------------------------------------------------------------------------------------
    entries = []
    for source_id, root, rec in measurements:
        conditions, rows_sha, available = summarise(rec, root)
        data = rec["extensions"]["aipe.data"]
        table = rec["extensions"]["aipe.table"]
        entries.append({"id": rec["id"], "source_id": source_id, "source_directory": root.name, "name": rec["name"],
                        "quantity": table["quantity"], "data_class": data["data_class"], "material_record": rec["properties"]["material_ref"],
                        "material": None, "row_count": table["row_count"], "rows_sha256": rows_sha, "rows_available": available,
                        "rows_file": table.get("rows_file"), "conditions": conditions, "underlying_source": data.get("underlying_source"),
                        "license": rec["license"], "flags": rec.get("quality", {}).get("flags", []), "status": "primary"})
    duplicates = []
    by_underlying: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        if e["underlying_source"]:
            by_underlying[e["underlying_source"]].append(e)
    rec_by_id = {rec["id"]: (root, rec) for _, root, rec in measurements}
    for underlying, group in sorted(by_underlying.items()):
        origin = ORIGINATING.get(underlying.split(":", 1)[0])
        originals = [e for e in group if e["source_id"] == origin]
        for e in group:
            if e["source_id"] == origin:
                continue
            same_quantity = [o for o in originals if o["quantity"] == e["quantity"]]
            if same_quantity:
                o = same_quantity[0]
                e["status"], e["duplicate_of"] = "duplicate", o["id"]
                match = _match_rows(rec_by_id[e["id"]], rec_by_id[o["id"]])
                e["duplicate_match"] = match
                duplicates.append({"record": e["id"], "duplicate_of": o["id"], "rule": "same underlying experiment and quantity",
                                   "underlying_source": underlying, **match})
            else:
                e["status"] = "derived_view"
                e["derived_from"] = [o["id"] for o in originals] or [underlying]
                duplicates.append({"record": e["id"], "derived_from": e["derived_from"], "rule": "derived quantity of the same underlying experiment",
                                   "underlying_source": underlying, "counted_as_independent": False})
    # --- canonical material records ------------------------------------------------------------------------------
    canonical_records = []
    canon_id_of_key: dict[str, str] = {}
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda m: (rank.get(m[0], 99), m[1]["id"]))
        primary_source, primary = members[0]
        cid = f"material.canonical-{key}" if not key.startswith("record-") else f"material.canonical-{slug(primary['id'].split('.', 1)[1])}"
        canon_id_of_key[key] = cid
        typed = next((m for m in members if len(m[1]["properties"]) > 2), members[0])
        props = dict(typed[1]["properties"])
        props.pop("manufacturer_ref", None)
        props["description"] = (f"Canonical identity of {primary['name']} ({key}) across {len(members)} source record(s); typed values are copied "
                                f"unchanged from {typed[1]['id']} ({typed[0]}).")
        evidence = [dict(e) for e in typed[1]["evidence"]]
        grade = (primary["properties"].get("grade") or "").casefold()
        # display name "<Manufacturer> <grade>" when any member spells it that way (curated records use the bare grade)
        name = next((m[1]["name"] for m in members if grade and m[1]["name"].casefold().endswith(grade) and len(m[1]["name"]) > len(grade)),
                    primary["name"])
        rec = {"schema_version": "0.2.0", "id": cid, "kind": "magnetic_material", "name": name, "synthetic": False,
               "license": typed[1]["license"], "properties": props, "evidence_refs": [e["id"] for e in evidence], "evidence": evidence,
               "quality": {"status": "curated", "flags": sorted(set(typed[1].get("quality", {}).get("flags", [])) | {"canonical_consolidation"})},
               "provenance": {"activity": "canonical_consolidation", "inputs": [m[1]["id"] for m in members],
                              "tool": {"id": "aipe.magnetics.database.ingest.canonical", "version": __version__},
                              "normalization": "identity grouping by normalised (manufacturer, grade); typed properties copied from one member"},
               "extensions": {"aipe.canonical": {"identity_key": key, "properties_from": typed[1]["id"],
                                                 "members": [{"record": m[1]["id"], "source_id": m[0],
                                                              "data_class": (m[1].get("extensions") or {}).get("aipe.data", {}).get("data_class", "manufacturer_datasheet"),
                                                              "license": m[1]["license"]} for m in members],
                                                 "measurements": []}}}
        canonical_records.append(rec)
    by_cid = {r["id"]: r for r in canonical_records}
    for e in entries:
        key = member_key.get(e["material_record"])
        e["material"] = canon_id_of_key.get(key) if key else None
    # identical rows: a duplicate when both tables belong to the same canonical material; across *different* materials an
    # identical table is an upstream copy (one of the two is wrong) - both stay primary and are flagged for review
    by_hash: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        if e["rows_sha256"] and e["status"] == "primary" and e["row_count"] > 1:
            by_hash[(e["quantity"], e["rows_sha256"])].append(e)
    for (_, digest), group in sorted(by_hash.items()):
        if len(group) < 2:
            continue
        group.sort(key=lambda e: (rank.get(e["source_id"], 99), e["id"]))
        by_material: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
        for e in group:
            by_material[e["material"]].append(e)
        for same in by_material.values():
            for e in same[1:]:
                e["status"], e["duplicate_of"] = "duplicate", same[0]["id"]
                duplicates.append({"record": e["id"], "duplicate_of": same[0]["id"], "rule": "identical table rows (sha256), same material",
                                   "rows_sha256": digest})
        if len(by_material) > 1:
            heads = [same[0] for same in by_material.values()]
            for e in heads:
                e["identical_rows_in_other_materials"] = sorted(h["id"] for h in heads if h is not e)
                e["flags"] = sorted(set(e["flags"]) | {"identical_table_in_other_material"})
            duplicates.append({"records": sorted(h["id"] for h in heads), "rule": "identical table rows (sha256) in different materials - "
                               "suspected upstream copy, kept as primary and flagged", "rows_sha256": digest, "counted_as_independent": True})

    for e in entries:
        if e["material"]:
            by_cid[e["material"]]["extensions"]["aipe.canonical"]["measurements"].append({"id": e["id"], "status": e["status"]})
    canonical_records = [validated(r) for r in canonical_records]
    entries.sort(key=lambda e: e["id"])
    canonical_root.mkdir(parents=True, exist_ok=True)
    write_if_changed(canonical_root / "materials.jsonl", "".join(dumps_line(r) + "\n" for r in canonical_records).encode("utf-8"))
    write_if_changed(canonical_root / "measurements.jsonl", "".join(dumps_line(e) + "\n" for e in entries).encode("utf-8"))
    write_if_changed(canonical_root / "duplicates.json", dumps_pretty(sorted(duplicates, key=lambda d: d.get("record") or d["records"][0])).encode("utf-8"))
    independent = [e for e in entries if e["status"] == "primary"]
    index = {"schema_version": "0.2.0", "builder_version": __version__,
             "counts": {"source_material_records": len(materials), "canonical_materials": len(canonical_records),
                        "canonical_materials_with_multiple_sources": sum(1 for r in canonical_records
                                                                         if len({m["source_id"] for m in r["extensions"]["aipe.canonical"]["members"]}) > 1),
                        "measurement_records": len(entries), "independent_measurement_records": len(independent),
                        "duplicate_measurement_records": sum(1 for e in entries if e["status"] == "duplicate"),
                        "derived_view_records": sum(1 for e in entries if e["status"] == "derived_view"),
                        "table_rows_total": sum(e["row_count"] for e in entries),
                        "table_rows_independent": sum(e["row_count"] for e in independent),
                        "experimental_rows_independent": sum(e["row_count"] for e in independent if e["data_class"] == "experimental_measurement")},
             "files": ["materials.jsonl", "measurements.jsonl", "duplicates.json"]}
    write_if_changed(canonical_root / "index.json", dumps_pretty(index).encode("utf-8"))
    return index


def _match_rows(candidate: tuple[Path, dict[str, Any]], original: tuple[Path, dict[str, Any]]) -> dict[str, Any]:
    """Fraction of ``candidate`` loss rows that occur in ``original`` (temperature 0.1 K, f / B / P_v within ~0.1 %)."""
    def keys(item: tuple[Path, dict[str, Any]]) -> list[tuple]:
        root, rec = item
        cols = [c["name"] for c in rec["extensions"]["aipe.table"]["columns"]]
        return [k for k in (_row_key(r, cols) for r in table_rows(rec, root)) if k is not None]

    cand, orig = keys(candidate), set(keys(original))
    if not cand or not orig:
        return {"rows_compared": len(cand), "rows_matched": 0, "match_fraction": None}
    matched = sum(1 for k in cand if k in orig)
    return {"rows_compared": len(cand), "rows_matched": matched, "match_fraction": round(matched / len(cand), 4)}


__all__ = ["SOURCE_PRECEDENCE", "build", "grade_key", "identity_key", "manufacturer_key", "summarise"]
