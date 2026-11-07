"""Automated scientific quality checks of the data store (``aipe-magnetics-db quality``).

The checks *flag* questionable records; they never change an observation. A clean report means the files are
internally consistent (schema, units, checksums, references) and no value falls outside broad physical plausibility
ranges - it is not proof of experimental accuracy.

Checks (``id`` in the report):

``schema``                 every record validates against schema v0.2 and its references resolve
``units``                  table columns carry their canonical SI unit; numeric columns have a unit
``non_finite``             NaN / Infinity in table rows or scalar properties
``physical_range``         values outside broad plausibility bounds (per column / property), kept and counted
``duplicate_rows``         identical rows repeated inside one table
``duplicate_tables``       identical tables across records / sources (from the canonical duplicate detection)
``table_files``            missing sidecars, sidecars whose SHA-256 differs from ``rows_sha256``, row-count mismatches
``raw_checksums``          committed raw files whose SHA-256 differs from ``data/raw/<dir>/SOURCES.json``
``material_identity``      one manufacturer + grade mapped to several canonical materials or material classes
``operating_conditions``   loss / permeability tables without frequency, temperature or flux-density information
``record_counts``          record counts that dropped (or more than doubled) since the previous report
``hub_consistency``        GitHub files that differ from the published Hugging Face release (data/huggingface.json)
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from aipe_magnetics_database.ingest.common import COLUMN_UNITS, sha256_path

REPORT_VERSION = "aipe.magnetics.quality/1.0"
#: broad plausibility bounds (SI); values outside are flagged, never altered
COLUMN_RANGES: dict[str, tuple[float, float]] = {
    "frequency": (0.0, 1e11), "temperature": (4.0, 1200.0), "flux_density_peak": (0.0, 3.0), "flux_density": (-3.0, 3.0),
    "field": (-1e7, 1e7), "dc_bias_field": (-1e6, 1e6), "loss_density": (0.0, 1e10), "mass_loss_density": (0.0, 1e7),
    "energy_density_per_cycle": (0.0, 1e6), "duty": (0.0, 1.0), "duty_p": (0.0, 1.0), "duty_n": (0.0, 1.0),
    "mu_imag": (0.0, 1e7), "mu_real": (-1e6, 1e7), "relative_permeability": (0.0, 1e7), "amplitude_permeability": (0.0, 1e7),
    "loss_factor": (0.0, 1.0), "eps_real": (-1e9, 1e10), "eps_imag": (-1e9, 1e10),
}
#: strictly-positive columns: a zero is flagged too
STRICTLY_POSITIVE = {"frequency", "loss_density", "mass_loss_density", "energy_density_per_cycle"}
PROPERTY_RANGES: dict[str, tuple[float, float, str]] = {
    "saturation_flux_density": (0.0, 3.0, "T"), "remanence": (0.0, 3.0, "T"), "coercivity": (0.0, 1e6, "A/m"),
    "curie_temperature": (250.0, 1500.0, "K"), "density": (500.0, 20000.0, "kg/m^3"), "resistivity": (1e-9, 1e12, "Ohm*m"),
    "initial_permeability": (1.0, 1e7, "1"), "permeability_vs_temperature": (0.0, 1e7, "1"),
}
#: manufacturer names that denote the same company (a grade under two of them is probably one material)
MANUFACTURER_ALIASES = ({"tdk", "epcos", "tdk-epcos"}, {"ferroxcube", "philips"}, {"magnetics", "mag-inc", "magnetics-inc"},
                        {"vac", "vacuumschmelze"}, {"proterial", "hitachi", "hitachi-metals", "metglas"}, {"fair-rite", "fairrite"},
                        {"wurth", "wurth-elektronik", "we"}, {"tdg", "tdg-holding"})
LOSS_QUANTITIES = ("core_loss", "mass_core_loss", "energy_loss", "energy_loss_and_permeability")
MAX_EXAMPLES = 10


class Check:
    def __init__(self, cid: str, severity: str, description: str):
        self.id, self.severity, self.description = cid, severity, description
        self.count = 0
        self.examples: list[Any] = []
        self.by_record: Counter[str] = Counter()

    def hit(self, example: Any, n: int = 1, record: str | None = None) -> None:
        self.count += n
        if record is not None:
            self.by_record[record] += n
        if len(self.examples) < MAX_EXAMPLES:
            self.examples.append(example)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "severity": self.severity, "description": self.description, "count": self.count,
                               "status": "pass" if self.count == 0 else ("fail" if self.severity == "error" else "flag"),
                               "examples": self.examples}
        if self.by_record:
            out["records_affected"] = len(self.by_record)
            out["top_records"] = [{"record": r, "count": n} for r, n in self.by_record.most_common(MAX_EXAMPLES)]
        return out


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _iter_values(prop: Any):
    """Yield (value, unit) of a typed property (single item, list of items, nested quantities)."""
    items = prop if isinstance(prop, list) else [prop]
    for item in items:
        if not isinstance(item, dict):
            continue
        val = item.get("value")
        if isinstance(val, dict) and "value" in val:
            yield val.get("value"), val.get("unit")
        elif _num(val):
            yield val, item.get("unit")


def _records(base: Path, index: dict[str, Any], group: str):
    for rel in index["shards"].get(group, []):
        with open(base / rel, encoding="utf-8") as handle:
            for line in handle:
                yield rel, json.loads(line)


def run_checks(root: Path, *, raw_checksums: dict[str, str] | None = None, previous: dict[str, Any] | None = None,
               validate_schema: bool = True) -> dict[str, Any]:
    """Run every check over ``root/data``; returns the machine-readable report (also see :func:`summary_markdown`)."""
    data = root / "data"
    normalized = data / "normalized"
    checks = {c.id: c for c in [
        Check("schema", "error", "schema v0.2 validation and reference resolution"),
        Check("units", "error", "table columns in their canonical SI unit; numeric columns carry a unit"),
        Check("non_finite", "error", "NaN or infinite numbers in table rows or properties"),
        Check("physical_range", "warning", "values outside broad physical plausibility bounds (flagged, kept verbatim)"),
        Check("duplicate_rows", "info", "identical rows repeated inside one table (may be genuine repeated measurements)"),
        Check("duplicate_tables", "info", "identical or overlapping tables across records (canonical duplicate detection)"),
        Check("table_files", "error", "table sidecars missing, with a different SHA-256 or row count"),
        Check("local_only_tables", "info", "large table sidecars not kept in Git (published to Hugging Face only)"),
        Check("raw_checksums", "error", "original files whose SHA-256 differs from the pinned value"),
        Check("material_identity", "warning", "one manufacturer + grade mapped to several canonical materials or classes"),
        Check("operating_conditions", "warning", "loss / permeability tables lacking frequency, temperature or flux-density information"),
        Check("record_counts", "warning", "record counts that dropped, or more than doubled, since the previous report"),
        Check("hub_consistency", "info", "normalised files that differ from the published Hugging Face release"),
    ]}
    indexes = {}
    for path in sorted(normalized.glob("*/index.json")):
        idx = json.loads(path.read_text(encoding="utf-8"))
        indexes[idx["source_id"]] = (path.parent, idx)

    if validate_schema:
        from aipe_magnetics_database.ingest.pipeline import validate_tree
        for problem in validate_tree(normalized, [base.name for base, _ in indexes.values()]):
            checks["schema"].hit(problem[:400])

    rows_checked = records_checked = 0
    counts: dict[str, dict[str, int]] = {}
    material_class: dict[str, str] = {}
    for sid, (base, idx) in indexes.items():
        counts[sid] = dict(idx["record_counts"])
        local_only = set(idx.get("tables_local", []))
        for _, rec in _records(base, idx, "materials"):
            records_checked += 1
            if (rec.get("properties") or {}).get("material_class"):
                material_class[rec["id"]] = rec["properties"]["material_class"]
            for name, prop in (rec.get("properties") or {}).items():
                bounds = PROPERTY_RANGES.get(name)
                for value, unit in _iter_values(prop):
                    if not _num(value):
                        continue
                    if not math.isfinite(value):
                        checks["non_finite"].hit({"record": rec["id"], "property": name}, record=rec["id"])
                    elif bounds and unit == bounds[2] and not bounds[0] <= value <= bounds[1]:
                        checks["physical_range"].hit({"record": rec["id"], "property": name, "value": value, "unit": unit,
                                                      "bounds": bounds[:2]}, record=rec["id"])
        for _, rec in _records(base, idx, "measurements"):
            records_checked += 1
            table = (rec.get("extensions") or {}).get("aipe.table")
            if not table:
                continue
            rid = rec["id"]
            cols = table["columns"]
            names = [c["name"] for c in cols]
            for c in cols:
                expected = COLUMN_UNITS.get(c["name"])
                if expected is not None and c.get("unit") != expected:
                    checks["units"].hit({"record": rid, "column": c["name"], "unit": c.get("unit"), "expected": expected}, record=rid)
                elif "unit" not in c:
                    checks["units"].hit({"record": rid, "column": c["name"], "unit": None}, record=rid)
            conditions = table.get("conditions") or {}
            if table["quantity"] in LOSS_QUANTITIES or "permeability" in table["quantity"]:
                missing = [k for k in ("frequency", "temperature") if k not in names and k not in conditions]
                flags = set((rec.get("quality") or {}).get("flags", []))
                if table["quantity"] in LOSS_QUANTITIES and "flux_density_peak" not in names and "flux_density_peak" not in conditions:
                    missing.append("flux_density_peak")
                if missing:
                    checks["operating_conditions"].hit({"record": rid, "quantity": table["quantity"], "missing": missing,
                                                        "declared_by_flags": sorted(f for f in flags if f.endswith("_not_stated"))}, record=rid)
            if "rows" in table:
                rows_iter = table["rows"]
            else:
                path = base / table["rows_file"]
                if not path.is_file():
                    if table["rows_file"] in local_only:
                        checks["local_only_tables"].hit({"record": rid, "rows_file": table["rows_file"]}, record=rid)
                    else:
                        checks["table_files"].hit({"record": rid, "problem": "missing sidecar", "rows_file": table["rows_file"]}, record=rid)
                    continue
                if table.get("rows_sha256") and sha256_path(path) != table["rows_sha256"]:
                    checks["table_files"].hit({"record": rid, "problem": "sha256 mismatch", "rows_file": table["rows_file"]}, record=rid)
                with open(path, encoding="utf-8") as handle:
                    rows_iter = [json.loads(line) for line in handle]
            if len(rows_iter) != table["row_count"]:
                checks["table_files"].hit({"record": rid, "problem": "row_count mismatch", "rows": len(rows_iter), "row_count": table["row_count"]},
                                          record=rid)
            seen: Counter[str] = Counter()
            bounds = [(i, COLUMN_RANGES.get(n), n) for i, n in enumerate(names)]
            for row in rows_iter:
                rows_checked += 1
                seen[json.dumps(row)] += 1
                for i, b, n in bounds:
                    v = row[i] if i < len(row) else None
                    if not _num(v):
                        continue
                    if not math.isfinite(v):
                        checks["non_finite"].hit({"record": rid, "column": n}, record=rid)
                    elif b is not None and (not b[0] <= v <= b[1] or (n in STRICTLY_POSITIVE and v <= 0)):
                        checks["physical_range"].hit({"record": rid, "column": n, "value": v, "bounds": list(b)}, record=rid)
            repeated = sum(k - 1 for k in seen.values() if k > 1)
            if repeated:
                checks["duplicate_rows"].hit({"record": rid, "repeated_rows": repeated}, repeated, record=rid)

    duplicates_path = data / "canonical" / "duplicates.json"
    if duplicates_path.is_file():
        for group in json.loads(duplicates_path.read_text(encoding="utf-8")):
            checks["duplicate_tables"].hit({"records": group.get("records"), "rule": group.get("rule")})

    canonical_materials = data / "canonical" / "materials.jsonl"
    if canonical_materials.is_file():
        by_grade: dict[str, set[str]] = defaultdict(set)
        with open(canonical_materials, encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                canon = (rec.get("extensions") or {}).get("aipe.canonical") or {}
                key = canon.get("identity_key") or ""
                grade = str((rec.get("properties") or {}).get("grade") or "").casefold().replace(" ", "")
                if key and grade:
                    by_grade[grade].add(key)
                member_classes = {material_class[m["record"]] for m in canon.get("members", []) if m.get("record") in material_class}
                if len(member_classes) > 1:
                    checks["material_identity"].hit({"canonical_id": rec["id"], "problem": "source records disagree on material_class",
                                                     "material_classes": sorted(member_classes)}, record=rec["id"])
        for grade, keys in sorted(by_grade.items()):
            prefixes = {k[: -len(grade)].strip("-") if k.endswith(grade) else k for k in keys}
            for group in MANUFACTURER_ALIASES:
                hits = sorted(p for p in prefixes if p in group)
                if len(hits) > 1:
                    checks["material_identity"].hit({"grade": grade, "problem": "same grade under aliased manufacturer names",
                                                     "identity_keys": sorted(k for k in keys if any(k.startswith(h) for h in hits))})

    for sub in sorted((data / "raw").glob("*/SOURCES.json")):
        manifest = json.loads(sub.read_text(encoding="utf-8"))
        for item in manifest.get("files", []):
            path = sub.parent / "files" / item["path"]
            if path.is_file() and sha256_path(path) != item["sha256"]:
                checks["raw_checksums"].hit({"source": manifest.get("source_id"), "path": item["path"]})
    for rel, digest in (raw_checksums or {}).items():
        if not isinstance(digest, str) or len(digest) != 64:
            checks["raw_checksums"].hit({"path": rel, "problem": "no sha256"})

    if previous is None:
        prev_path = data / "reports" / "quality-report.json"
        previous = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.is_file() else None
    if previous:
        for sid, old in (previous.get("record_counts") or {}).items():
            new = counts.get(sid)
            if new is None:
                checks["record_counts"].hit({"source": sid, "problem": "source disappeared", "previous": old})
                continue
            for kind, n in old.items():
                m = new.get(kind, 0)
                if m < n or (n and m > 2 * n):
                    checks["record_counts"].hit({"source": sid, "kind": kind, "previous": n, "current": m})

    hub_path = data / "huggingface.json"
    if hub_path.is_file():
        from aipe_magnetics_database.hub.layout import normalized_path
        hub = json.loads(hub_path.read_text(encoding="utf-8"))
        published = hub.get("files") or {}
        if published:
            for sid, (base, idx) in indexes.items():
                for path in sorted(base.rglob("*")):
                    if not path.is_file() or "tables-local" in path.parts:
                        continue
                    rel = normalized_path(sid, path.relative_to(base).as_posix())
                    meta = published.get(rel)
                    if meta is None:
                        checks["hub_consistency"].hit({"path": rel, "problem": "not in the published release"})
                    elif meta["size"] != path.stat().st_size or meta["sha256"] != sha256_path(path):
                        checks["hub_consistency"].hit({"path": rel, "problem": "differs from the published release"})

    results = [c.to_json() for c in checks.values()]
    summary = {"errors": sum(c["count"] for c in results if c["severity"] == "error"),
               "warnings": sum(c["count"] for c in results if c["severity"] == "warning"),
               "info": sum(c["count"] for c in results if c["severity"] == "info"),
               "checks_failed": [c["id"] for c in results if c["status"] == "fail"],
               "checks_flagged": [c["id"] for c in results if c["status"] == "flag"],
               "records_checked": records_checked, "rows_checked": rows_checked}
    return {"report": REPORT_VERSION, "summary": summary, "record_counts": counts, "checks": results,
            "disclaimer": "Automated consistency and plausibility checks only. Flags identify values to review; no observation was "
                          "changed. Schema validity is not proof of experimental accuracy."}


def summary_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = ["# Data quality summary", "", "Generated by `aipe-magnetics-db quality` (`src/aipe_magnetics_database/ingest/quality.py`); do not edit.",
             "", f"- Records checked: {s['records_checked']:,}; table rows checked: {s['rows_checked']:,}",
             f"- Errors: {s['errors']:,}; warnings: {s['warnings']:,}; informational: {s['info']:,}", "",
             "| Check | Severity | Status | Count | Records affected | Description |", "| --- | --- | --- | --- | --- | --- |"]
    for c in report["checks"]:
        lines.append(f"| `{c['id']}` | {c['severity']} | {c['status']} | {c['count']:,} | {c.get('records_affected', '-')} | {c['description']} |")
    lines += ["", report["disclaimer"], ""]
    flagged = [c for c in report["checks"] if c["count"] and c["severity"] != "info"]
    for c in flagged:
        lines += [f"## `{c['id']}`", ""]
        for ex in c["examples"][:5]:
            lines.append(f"- `{json.dumps(ex, ensure_ascii=False)[:300]}`")
        lines.append("")
    return "\n".join(lines)


__all__ = ["COLUMN_RANGES", "PROPERTY_RANGES", "REPORT_VERSION", "run_checks", "summary_markdown"]
