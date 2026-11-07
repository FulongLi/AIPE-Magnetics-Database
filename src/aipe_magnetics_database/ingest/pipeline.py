"""One reproducible, idempotent refresh of all approved sources.

``aipe-magnetics-db refresh [--source ID ...] [--local ID=DIR ...] [--allow-download] [--cache DIR]``

1. obtain every pinned input (local copy or cache; download only with ``--allow-download``; SHA-256 verified;
   interrupted downloads resume because the cache only admits verified files),
2. run the importers (licence status from :mod:`registry`; only ``integrated`` sources are written),
3. validate every written record against schema v0.2 and resolve references (``validate_tree``),
4. build the canonical layer with duplicate detection (:mod:`canonical`),
5. write ``data/reports/import-<source>.json``, ``data/reports/coverage.json``, ``data/sources.json`` and
   ``docs/source-catalogue.md``.

Re-running with the same inputs rewrites no file (byte-identical output).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from aipe_magnetics_database.ingest import canonical, registry
from aipe_magnetics_database.ingest.common import ImportSummary, dumps_pretty, write_if_changed

ImporterFn = Callable[..., ImportSummary]


def importers() -> dict[str, tuple[ImporterFn, str]]:
    from aipe_magnetics_database.ingest.sources import mas, openlab, paderborn, princeton, zenodo
    return {
        "openmagnetics_mas": (mas.run, "mas"),
        "princeton_magnet": (princeton.run, "princeton-magnet"),
        "paderborn_lea_materialdatabase": (paderborn.run, "paderborn-lea"),
        "zenodo_8292861_hefmag": (zenodo.run_hefmag, "hefmag"),
        "zenodo_21626116_smc_ferrite_triangular": (zenodo.run_triangular, "zenodo-smc-ferrite-triangular"),
        **openlab.IMPORTERS,
    }


def _archived(committed_raw: Path, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Committed original files the importer does not read (documentation, licence, alternative formats): kept and pinned too."""
    from aipe_magnetics_database.ingest.common import sha256_path
    if not committed_raw.is_dir():
        return []
    read = {i["path"] for i in inputs} | {Path(i["path"]).name for i in inputs}
    return [{"path": p.relative_to(committed_raw).as_posix(), "sha256": sha256_path(p), "role": "archived (not read by the importer)"}
            for p in sorted(committed_raw.rglob("*")) if p.is_file() and p.relative_to(committed_raw).as_posix() not in read
            and p.name not in read]


def repository_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "data" / "normalized").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("repository root with data/normalized not found; run refresh from a checkout of AIPE-Magnetics-Database")


def curated_records() -> list[dict[str, Any]]:
    from aipe_magnetics_database.data.catalog import Catalog
    return [r.to_json_dict() for r in Catalog.default().records()]


def validate_tree(normalized: Path, directories: list[str]) -> list[str]:
    """Schema-validate every record of the given source directories and check cross-record references."""
    from aipe_magnetics_database.data.records import MagneticRecord, validate_catalogue
    problems: list[str] = []
    records = []
    for directory in directories:
        root = normalized / directory
        index = json.loads((root / "index.json").read_text(encoding="utf-8"))
        files = [f for group in index["shards"].values() for f in group] + index["datasets"]
        for rel in files:
            text = (root / rel).read_text(encoding="utf-8")
            objects = [json.loads(text)] if rel.endswith(".json") else [json.loads(line) for line in text.splitlines()]
            for obj in objects:
                try:
                    records.append(MagneticRecord.model_validate(obj))
                except Exception as error:  # noqa: BLE001 - collect every problem
                    problems.append(f"{directory}/{rel}: {obj.get('id')}: {error}")
    problems += validate_catalogue(records)
    return problems


def coverage(normalized: Path, canonical_index: dict[str, Any], summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_source = []
    for source in registry.SOURCES:
        entry: dict[str, Any] = {k: source.get(k) for k in ("id", "title", "priority", "status", "version", "license", "redistribution")}
        directory = source.get("directory")
        if directory and (normalized / directory / "index.json").is_file():
            index = json.loads((normalized / directory / "index.json").read_text(encoding="utf-8"))
            entry["record_counts"] = index["record_counts"]
            entry["table_rows"] = index["table_rows"]
            entry["tables_local_only"] = len(index.get("tables_local", []))
        if source["id"] in summaries:
            entry["exclusions"] = sum(e.get("count", 1) for e in summaries[source["id"]]["exclusions"])
        if source.get("reason"):
            entry["reason"] = source["reason"]
        per_source.append(entry)
    totals: dict[str, int] = {}
    rows = 0
    for entry in per_source:
        for kind, n in (entry.get("record_counts") or {}).items():
            totals[kind] = totals.get(kind, 0) + n
        rows += sum((entry.get("table_rows") or {}).values())
    return {"schema_version": "0.2.0", "sources": per_source, "totals_source_records": dict(sorted(totals.items())),
            "table_rows_source": rows, "canonical": canonical_index["counts"]}


def refresh(sources: list[str] | None = None, *, root: Path | None = None, local: dict[str, str] | None = None, allow_download: bool = False,
            cache_dir: str | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    from aipe_magnetics_database.data.sources.cache import LocalCache

    root = root or repository_root()
    normalized, reports = root / "data" / "normalized", root / "data" / "reports"
    cache = LocalCache(cache_dir) if cache_dir else LocalCache()
    table = importers()
    chosen = sources or list(table)
    unknown = sorted(set(chosen) - set(table))
    if unknown:
        raise ValueError(f"unknown or non-integrated source(s) {unknown}; integrated: {sorted(table)}")
    summaries: dict[str, dict[str, Any]] = {}
    for path in reports.glob("import-*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        summaries[data["source_id"]] = data
    for source_id in chosen:
        fn, directory = table[source_id]
        local_dir = (local or {}).get(source_id)
        committed_raw = root / "data" / "raw" / directory / "files"
        if local_dir is None and committed_raw.is_dir():
            local_dir = str(committed_raw)  # redistributable original files committed with the repository (SHA-256 verified)
        log(f"[{source_id}] importing ...")
        summary = fn(normalized, local_dir=local_dir, allow_download=allow_download, cache=cache)
        source = registry.by_id(source_id)
        manifest = {"source_id": source_id, "title": source["title"], "upstream": source["url"], "version": summary.version,
                    "license": summary.license, "redistribution": summary.redistribution,
                    "raw_policy": ("original files committed under files/ (redistribution permitted)" if committed_raw.is_dir()
                                   else "retrieved on demand at the pinned version and verified by SHA-256; not committed"),
                    "files": summary.inputs + _archived(committed_raw, summary.inputs), "importer": source.get("importer")}
        write_if_changed(root / "data" / "raw" / directory / "SOURCES.json", dumps_pretty(manifest).encode("utf-8"))
        summaries[source_id] = summary.to_json()
        write_if_changed(reports / f"import-{source_id}.json", dumps_pretty(summaries[source_id]).encode("utf-8"))
        log(f"[{source_id}] {summary.counts} rows {sum(summary.table_rows.values())} written {summary.files_written} unchanged {summary.files_unchanged}")
    directories = [d for _, d in table.values() if (normalized / d / "index.json").is_file()]
    problems = validate_tree(normalized, directories)
    if problems:
        raise ValueError(f"{len(problems)} validation problem(s), first: {problems[:5]}")
    log("validation: all records schema-valid, references resolve")
    index = canonical.build(normalized, root / "data" / "canonical",
                            {sid: {"directory": d} for sid, (_, d) in table.items() if (normalized / d / "index.json").is_file()},
                            curated_records())
    log(f"canonical: {index['counts']}")
    report = coverage(normalized, index, summaries)
    write_if_changed(reports / "coverage.json", dumps_pretty(report).encode("utf-8"))
    write_if_changed(root / "data" / "sources.json", dumps_pretty(registry.SOURCES).encode("utf-8"))
    write_if_changed(root / "docs" / "source-catalogue.md", source_catalogue_markdown(report).encode("utf-8"))
    return report


def source_catalogue_markdown(report: dict[str, Any]) -> str:
    """docs/source-catalogue.md (generated; edit ingest/registry.py instead)."""
    lines = ["# Source catalogue", "",
             "Generated by `aipe-magnetics-db refresh` from `src/aipe_magnetics_database/ingest/registry.py` and the import reports;",
             "do not edit by hand. Counts are records written to `data/normalized/` (source records, before deduplication).", "",
             "| Source | Priority | Status | Version | Licence | Records | Table rows | Notes |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in report["sources"]:
        records = ", ".join(f"{k} {v:,}" for k, v in (s.get("record_counts") or {}).items()) or "-"
        rows = f"{sum((s.get('table_rows') or {}).values()):,}" if s.get("table_rows") else "-"
        version = (s.get("version") or "-")
        version = version[:12] if len(version) == 40 else version
        note = s.get("reason") or s.get("redistribution") or ""
        lines.append(f"| {s['title']} (`{s['id']}`) | {s['priority']} | {s['status']} | {version} | {s['license']} | {records} | {rows} | {note} |")
    c = report["canonical"]
    lines += ["", "## Totals", "",
              "- Source records: " + ", ".join(f"{k} {v:,}" for k, v in report["totals_source_records"].items()),
              f"- Table rows (all source records): {report['table_rows_source']:,}",
              f"- Canonical materials: {c['canonical_materials']:,} (from {c['source_material_records']:,} source material records; "
              f"{c['canonical_materials_with_multiple_sources']:,} combine several sources)",
              f"- Measurement tables: {c['measurement_records']:,} source records, {c['independent_measurement_records']:,} independent, "
              f"{c['duplicate_measurement_records']:,} duplicates, {c['derived_view_records']:,} derived views",
              f"- Table rows in independent records: {c['table_rows_independent']:,}, of which experimental measurements: "
              f"{c['experimental_rows_independent']:,}", ""]
    return "\n".join(lines)


__all__ = ["coverage", "curated_records", "importers", "refresh", "repository_root", "source_catalogue_markdown", "validate_tree"]
