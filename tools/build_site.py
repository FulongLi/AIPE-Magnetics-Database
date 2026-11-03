"""Build the AIPE Magnetics Data Explorer: a static, read-only GitHub Pages site over the committed data store.

    python tools/build_site.py --out _site

The builder reads the data only through ``aipe_magnetics_database.store.MagneticDatabase`` (the same store behind the
Python API, CLI and MCP server) and writes only to ``--out``: lightweight JSON indexes for client-side search, one detail
file per material and per measurement table (loaded on demand), and the static frontend from ``site/``. It never
modifies records under ``data/``, never downloads anything and performs no fitting, interpolation or derivation.

Tables with more than ``PREVIEW_ROWS`` rows are published as a labelled, deterministic sample (every k-th row within each
group of categorical conditions, upstream order kept); the full table is linked at the pinned commit. Rows are only
published when the source dataset permits redistribution, and tables whose sidecar is not committed (regenerated MAS
B-H loops) are listed without rows.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aipe_magnetics_database import __version__  # noqa: E402
from aipe_magnetics_database.common.errors import DataNotFoundError  # noqa: E402
from aipe_magnetics_database.ingest.canonical import record_manufacturer  # noqa: E402
from aipe_magnetics_database.ingest.pipeline import curated_records  # noqa: E402
from aipe_magnetics_database.store import MagneticDatabase  # noqa: E402

REPOSITORY = "FulongLi/AIPE-Magnetics-Database"
PREVIEW_ROWS = 5000
GROUP_COLUMNS = ("waveform", "temperature", "specimen", "branch")
HEADLINE = {"mu_i": "initial_permeability", "bsat": "saturation_flux_density", "rho": "resistivity", "density": "density",
            "tc": "curie_temperature"}
ROOM_TEMPERATURE_K = 298.15


# -- helpers ---------------------------------------------------------------------------------------------------------
def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout.strip()
        return out or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _write(path: Path, payload: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def _entry_value(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Compact ``{v, u, T, note}`` of one property entry (values copied unchanged; ``T`` in K when stated)."""
    value = entry.get("value")
    if not isinstance(value, dict) or "value" not in value:
        return None
    out = {"v": value["value"], "u": value.get("unit", "")}
    temperature = entry.get("temperature")
    if isinstance(temperature, dict) and "value" in temperature:
        out["T"] = temperature["value"]
    if entry.get("note"):
        out["note"] = entry["note"]
    return out


def headline(prop: Any) -> dict[str, Any] | None:
    """Representative value of a property for the material list: the entry closest to 25 degC (or the only one)."""
    entries = prop if isinstance(prop, list) else [prop] if isinstance(prop, dict) else []
    values = [v for v in (_entry_value(e) for e in entries if isinstance(e, dict)) if v is not None]
    if not values:
        return None
    best = min(values, key=lambda v: abs(v["T"] - ROOM_TEMPERATURE_K) if "T" in v else 1e9)
    if len(values) > 1:
        best["n"] = len(values)
    return best


def sample_rows(columns: list[dict[str, Any]], rows: list[list[Any]], limit: int = PREVIEW_ROWS) -> tuple[list[list[Any]], dict[str, Any]]:
    """Deterministic preview: all rows when ``len(rows) <= limit``, otherwise every k-th row within each group of the
    categorical condition columns (waveform, temperature, specimen, branch), keeping the upstream row order."""
    total = len(rows)
    if total <= limit:
        return rows, {"sampled": False, "total_rows": total, "published_rows": total}
    names = [c["name"] for c in columns]
    keys = [names.index(c) for c in GROUP_COLUMNS if c in names]
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[tuple(row[k] for k in keys)].append(i)
    keep: list[int] = []
    for members in groups.values():
        quota = min(len(members), max(1, math.floor(len(members) * limit / total)))
        step = len(members) / quota
        keep.extend(members[int(j * step)] for j in range(quota))
    keep.sort()
    return [rows[i] for i in keep], {
        "sampled": True, "total_rows": total, "published_rows": len(keep),
        "method": "Systematic sample: every k-th row within each group of " + (", ".join(c for c in GROUP_COLUMNS if c in names) or "the table")
                  + f" (upstream order kept, about {limit} rows in total); values are unchanged, rows are omitted",
    }


# -- builder ---------------------------------------------------------------------------------------------------------
class SiteBuilder:
    def __init__(self, out: Path, *, root: str | Path | None = None, commit: str | None = None, preview_rows: int = PREVIEW_ROWS):
        self.db = MagneticDatabase(root)
        self.out = Path(out)
        self.data = self.out / "data"
        self.preview_rows = preview_rows
        self.commit = commit or os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD") or "main"
        self.built_from = _git("show", "-s", "--format=%cI", self.commit) or _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
        self.bytes = 0
        self.stats: Counter = Counter()

    # links at the pinned commit (records, sidecars, licence files)
    def blob(self, rel: str) -> str:
        return f"https://github.com/{REPOSITORY}/blob/{self.commit}/{rel}"

    def raw(self, rel: str) -> str:
        return f"https://raw.githubusercontent.com/{REPOSITORY}/{self.commit}/{rel}"

    def run(self) -> dict[str, Any]:
        if self.out.exists():
            shutil.rmtree(self.out)
        shutil.copytree(ROOT / "site", self.out, ignore=shutil.ignore_patterns("README.md", ".DS_Store"))
        (self.out / ".nojekyll").write_text("", encoding="utf-8")
        self._load()
        self._sources()
        self._measurements()
        self._materials()
        self._cores()
        self._summary()
        return {"out": str(self.out), "bytes": self.bytes, **self.stats}

    def _load(self) -> None:
        db = self.db
        self.indexes = db.source_indexes
        self.source_dir = {sid: idx["_directory"] for sid, idx in self.indexes.items()}
        self.canonical = db.materials()
        self.entries = db.measurements(include_duplicates=True)  # canonical index: primary, duplicate and derived-view tables
        members = {r["id"]: r for r in db.materials(canonical=False)}
        members.update({r["id"]: r | {"_source": "aipe_curated"} for r in curated_records()})
        self.member_records = members
        self.canonical_of: dict[str, str] = {}
        for c in self.canonical:
            for m in c["extensions"]["aipe.canonical"]["members"]:
                self.canonical_of[m["record"]] = c["id"]
        self.material_name = {c["id"]: c["name"] for c in self.canonical}
        # dataset records: redistribution permission, citation, title
        self.datasets: dict[str, dict[str, Any]] = {}
        self.redistributable: dict[str, bool] = {}
        for sid, idx in self.indexes.items():
            flags = []
            for rel in idx["datasets"]:
                d = json.loads((db.normalized / idx["_directory"] / rel).read_text(encoding="utf-8"))
                self.datasets[d["id"]] = {"id": d["id"], "license": d["license"], **{k: d["properties"].get(k) for k in (
                    "title", "url", "version", "publisher", "citation", "description", "redistribution_permitted")}}
                flags.append(d["properties"].get("redistribution_permitted") is True)
            self.redistributable[sid] = bool(flags) and all(flags)
        # record id -> committed shard file, for links to the original record
        self.shard_of: dict[str, str] = {}
        id_re = re.compile(r'"id"\s*:\s*"([^"]+)"')
        for sid, idx in self.indexes.items():
            for rel_list in idx["shards"].values():
                for rel in rel_list:
                    path = db.normalized / idx["_directory"] / rel
                    with open(path, encoding="utf-8") as handle:
                        for line in handle:
                            m = id_re.search(line)
                            if m:
                                self.shard_of.setdefault(m.group(1), f"data/normalized/{idx['_directory']}/{rel}")

    # -- sources ------------------------------------------------------------------------------------------------------
    def _sources(self) -> None:
        out = []
        for s in self.db.sources:
            sid = s["id"]
            idx = self.indexes.get(sid)
            item = {k: s.get(k) for k in ("id", "title", "priority", "status", "url", "version", "license", "redistribution", "notes")}
            if idx:
                d = self.db.normalized / idx["_directory"]
                item["records"] = idx["record_counts"]
                item["table_rows"] = idx["table_rows"]
                item["directory"] = self.blob(f"data/normalized/{idx['_directory']}")
                item["licence_files"] = [self.blob(f"data/normalized/{idx['_directory']}/{n}") for n in ("LICENSE", "NOTICE.md") if (d / n).is_file()]
                item["rows_published"] = self.redistributable[sid]
            out.append(item)
        self.sources = out
        self.bytes += _write(self.data / "sources.json", out)

    # -- materials ----------------------------------------------------------------------------------------------------
    def _materials(self) -> None:
        by_material: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in self.entries:
            if e.get("material"):
                by_material[e["material"]].append(e)
        models: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in self.db.loss_models():
            for ref in r["properties"].get("material_refs", []):
                if ref in self.canonical_of:
                    models[self.canonical_of[ref]].append(r)
        parts = Counter()
        for r in self.db.cores(kind="parts"):
            for ref in set(self.canonical_of.get(x) for x in r["properties"].get("material_refs", [])) - {None}:
                parts[ref] += 1
        self.parts_per_material = parts
        index = []
        for c in self.canonical:
            canon = c["extensions"]["aipe.canonical"]
            mem = [self.member_records[m["record"]] for m in canon["members"] if m["record"] in self.member_records]
            props = c["properties"]
            grade = props.get("grade") or ""
            # display spelling from the canonical name ("TDK N87" -> "TDK"), else the members' manufacturer field
            manufacturer = c["name"][: -len(grade)].strip() if grade and c["name"].endswith(grade) and c["name"] != grade else None
            manufacturer = manufacturer or next((record_manufacturer(r) for r in mem if record_manufacturer(r)), None)
            meas = by_material.get(c["id"], [])
            item = {"id": c["id"], "name": c["name"], "grade": props.get("grade"), "manufacturer": manufacturer,
                    "class": props.get("material_class"), "sources": sorted({m["source_id"] for m in canon["members"]}),
                    "data_classes": sorted({m["data_class"] for m in canon["members"]}),
                    "licenses": sorted({m["license"] for m in canon["members"]}),
                    "measurements": sum(1 for e in meas if e["status"] == "primary"),
                    "experimental": sum(1 for e in meas if e["status"] == "primary" and e["data_class"] == "experimental_measurement"),
                    "quantities": sorted({e["quantity"] for e in meas}), "loss_models": len(models.get(c["id"], [])),
                    "parts": parts.get(c["id"], 0), "flags": c["quality"].get("flags", [])}
            for key, prop in HEADLINE.items():
                h = headline(props.get(prop))
                if h is not None:
                    item[key] = h
            if isinstance(props.get("recommended_frequency_range"), dict):
                item["f_range"] = props["recommended_frequency_range"]
            index.append(item)
            detail = {"summary": item, "record": c,
                      "members": [{"source_id": r.get("_source"), "shard": self.blob(self.shard_of[r["id"]]) if r["id"] in self.shard_of else None,
                                   "record": {k: v for k, v in r.items() if k != "_source"}} for r in mem],
                      "measurements": [self._entry_summary(e) for e in meas],
                      "loss_models": [self._loss_model(r) for r in models.get(c["id"], [])]}
            self.bytes += _write(self.data / "materials" / f"{c['id']}.json", detail)
        index.sort(key=lambda r: r["name"].casefold())
        self.material_index = index
        self.bytes += _write(self.data / "materials.json", index)
        self.stats["materials"] = len(index)

    def _loss_model(self, r: dict[str, Any]) -> dict[str, Any]:
        p = r["properties"]
        return {"id": r["id"], "name": r["name"], "license": r["license"], "source_id": r.get("_source"),
                "data_class": (r.get("extensions") or {}).get("aipe.data", {}).get("data_class", "fitted_parameters"),
                "model_id": p.get("model_id"), "architecture": p.get("architecture"), "description": p.get("description"),
                "validity": p.get("validity"), "inference_config": p.get("inference_config"), "evidence": r.get("evidence", []),
                "flags": r.get("quality", {}).get("flags", []), "shard": self.blob(self.shard_of[r["id"]]) if r["id"] in self.shard_of else None}

    # -- measurements -------------------------------------------------------------------------------------------------
    def _entry_summary(self, e: dict[str, Any]) -> dict[str, Any]:
        keys = ("id", "name", "quantity", "data_class", "source_id", "material", "row_count", "conditions", "status", "license",
                "flags", "duplicate_of", "derived_from", "duplicate_match", "underlying_source")
        out = {k: e[k] for k in keys if e.get(k) is not None}
        out["material_name"] = self.material_name.get(e.get("material"))
        out["published"] = self.published[e["id"]]
        return out

    def _measurements(self) -> None:
        self.published: dict[str, str] = {}
        details = {}
        for e in self.entries:
            rec = self.db.measurement(e["id"])
            table = rec["extensions"]["aipe.table"]
            meta = json.loads(json.dumps(rec))
            meta["extensions"]["aipe.table"].pop("rows", None)
            sid = e["source_id"]
            links = {"record": self.blob(self.shard_of[e["id"]]) if e["id"] in self.shard_of else None}
            rows_file = table.get("rows_file")
            local_only = bool(rows_file) and rows_file.startswith("tables-local/")
            if rows_file and not local_only:
                links["rows_file"] = self.raw(f"data/normalized/{self.source_dir[sid]}/{rows_file}")
            unavailable = None
            rows: list[list[Any]] = []
            sampling = {"sampled": False, "total_rows": table["row_count"], "published_rows": 0}
            regenerate = ("Large regenerated table: its rows are not committed to the repository, so they are not published here. "
                          f"Regenerate them locally with 'aipe-magnetics-db refresh --source {sid} --allow-download'.")
            if not self.redistributable.get(sid):
                unavailable = "The source dataset does not grant redistribution; rows are not published on this site."
            elif local_only:  # git-ignored sidecars: published builds must not depend on what happens to exist locally
                unavailable = regenerate
            else:
                try:
                    all_rows = self.db.table(e["id"]).rows
                    rows, sampling = sample_rows(table["columns"], all_rows, self.preview_rows)
                except DataNotFoundError:
                    unavailable = regenerate
            self.published[e["id"]] = "none" if unavailable else "sample" if sampling["sampled"] else "full"
            self.stats[f"tables_{self.published[e['id']]}"] += 1
            details[e["id"]] = {"record": meta, "columns": table["columns"], "rows": rows, "sampling": sampling, "unavailable": unavailable,
                                "dataset": self.datasets.get(rec.get("dataset_ref")), "links": links}
        index = []
        for e in self.entries:
            detail = details[e["id"]]
            detail["entry"] = self._entry_summary(e)
            self.bytes += _write(self.data / "measurements" / f"{e['id']}.json", detail)
            index.append(detail["entry"] | {"columns": [c["name"] for c in detail["columns"]]})
        self.bytes += _write(self.data / "measurements.json", index)
        self.stats["measurements"] = len(index)

    # -- cores --------------------------------------------------------------------------------------------------------
    def _cores(self) -> None:
        shapes, parts = [], []
        for r in self.db.cores():
            mas = r["extensions"].get("aipe.mas", {})
            ev = (r.get("evidence") or [{}])[0]
            if r["id"].startswith("core.mas-part-"):
                refs = r["properties"].get("material_refs", [])
                canon = next((self.canonical_of[x] for x in refs if x in self.canonical_of), None)
                gaps = [[g.get("type"), g.get("length")] for g in mas.get("gapping") or []]
                parts.append([r["id"], r["name"], r["properties"].get("part_number"), mas.get("manufacturer"), mas.get("material_name"), canon,
                              mas.get("shape_name"), mas.get("shape_ref"), mas.get("status"), mas.get("datasheet_url"), gaps,
                              mas.get("coating"), mas.get("number_stacks"), r["properties"].get("set_description"), mas.get("source_line"),
                              r["license"]])
            else:
                shapes.append({"id": r["id"], "name": r["name"], "family": mas.get("family"), "aliases": mas.get("aliases", []),
                               "type": mas.get("shape_type"), "circuit": mas.get("magnetic_circuit"),
                               "ranges": mas.get("dimensions_range_m", {}), "nominal": mas.get("dimensions_nominal_m"),
                               "inconsistent": mas.get("inconsistent_dimension_ranges_m"), "flags": r["quality"].get("flags", []),
                               "license": r["license"], "evidence": ev.get("url"), "description": r["properties"].get("description")})
        part_columns = ["id", "name", "part_number", "manufacturer", "material_name", "material", "shape_name", "shape_ref", "status",
                        "datasheet_url", "gapping", "coating", "number_stacks", "set_description", "source_line", "license"]
        mas_url = next((d["url"] for d in self.datasets.values() if d["id"] == "dataset.mas-cores"), None)
        meta = {"source_id": "openmagnetics_mas", "evidence_base": "https://github.com/Power-Supply-Manufacturers-Association/MAS/blob/"
                f"{self.indexes['openmagnetics_mas']['version']}/data/cores.ndjson", "dataset_url": mas_url,
                "note": "Part identity, manufacturer, material and shape references and gapping as published by MAS; effective parameters "
                        "(Ae, le, Ve) are not published by the source and are not computed here."}
        self.bytes += _write(self.data / "cores" / "shapes.json", {"shapes": shapes, "note": "Dimension tolerance ranges in m as published by MAS; "
                             "nominal values only where MAS publishes them. No midpoints or effective parameters are derived."})
        self.bytes += _write(self.data / "cores" / "parts.json", {"columns": part_columns, "rows": parts, **meta})
        self.stats["shapes"], self.stats["parts"] = len(shapes), len(parts)

    # -- summary ------------------------------------------------------------------------------------------------------
    def _summary(self) -> None:
        canonical = self.db.summary()["canonical"]
        primary = [e for e in self.entries if e["status"] == "primary"]
        summary = {
            "title": "AIPE Magnetics Data Explorer", "package": "aipe-magnetics-database", "version": __version__, "schema_version": "0.2.0",
            "repository": f"https://github.com/{REPOSITORY}", "commit": self.commit, "data_date": self.built_from,
            "preview_rows": self.preview_rows,
            "counts": {**canonical, "core_shapes": self.stats["shapes"], "core_parts": self.stats["parts"],
                       "fitted_loss_models": len(self.db.loss_models()),
                       "sources_investigated": len(self.sources),
                       "sources_integrated": sum(1 for s in self.sources if s.get("status") == "integrated"),
                       "tables_published_full": self.stats["tables_full"], "tables_published_sample": self.stats["tables_sample"],
                       "tables_not_published": self.stats["tables_none"]},
            "quantities": dict(Counter(e["quantity"] for e in primary).most_common()),
            "data_classes": dict(Counter(e["data_class"] for e in primary).most_common()),
            "material_classes": dict(Counter(m["class"] or "unspecified" for m in self.material_index).most_common()),
        }
        self.bytes += _write(self.data / "summary.json", summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--out", default=str(ROOT / "_site"), help="output directory (replaced), default: _site/")
    parser.add_argument("--data", default=None, help="data store root (default: the repository data/ directory)")
    parser.add_argument("--commit", default=None, help="commit used for links to records (default: GITHUB_SHA or git HEAD)")
    parser.add_argument("--preview-rows", type=int, default=PREVIEW_ROWS, help=f"rows per published table preview (default {PREVIEW_ROWS})")
    args = parser.parse_args(argv)
    result = SiteBuilder(Path(args.out), root=args.data, commit=args.commit, preview_rows=args.preview_rows).run()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
