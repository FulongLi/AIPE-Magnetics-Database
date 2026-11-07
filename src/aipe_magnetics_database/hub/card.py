"""Dataset card (``README.md``) of the Hugging Face release, generated from the release manifests."""

from __future__ import annotations

from typing import Any

HF_LICENSE_IDS = {"MIT": "mit", "Apache-2.0": "apache-2.0", "CC-BY-4.0": "cc-by-4.0", "GPL-3.0-only": "gpl-3.0", "CC0-1.0": "cc0-1.0",
                  "CC-BY-SA-4.0": "cc-by-sa-4.0", "BSD-3-Clause": "bsd-3-clause", "ODbL-1.0": "odbl"}


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return str(n)


def dataset_card(index: dict[str, Any], sources: list[dict[str, Any]]) -> str:
    published = index["sources"]
    licences = sorted({s["license"] for s in published})
    totals = index["totals"]
    rows = sum(totals["table_rows"].values())
    q = index["quality_summary"]
    lines = [
        "---",
        "license: other",
        "license_name: per-source-licences",
        "license_link: LICENSE.md",
        "pretty_name: AIPE Magnetics Dataset",
        "language:",
        "- en",
        "tags:",
        "- magnetics", "- power-electronics", "- core-loss", "- ferrite", "- magnetic-materials", "- physics",
        "size_categories:",
        "- 1M<n<10M",
        "viewer: false",
        "---",
        "",
        "# AIPE Magnetics Dataset",
        "",
        "Original files and standardised records of open soft-magnetic material and core data for power electronics: material",
        "properties, core shapes and parts, and measured core loss, complex permeability, permittivity and B-H tables. This",
        "repository is the storage layer; it is built, validated and published by the",
        "[AIPE Magnetics Database](https://github.com/FulongLi/AIPE-Magnetics-Database) (Python API, CLI, MCP tools,",
        "[Data Explorer](https://fulongli.github.io/AIPE-Magnetics-Database/)).",
        "",
        f"This release: {len(published)} sources, " + ", ".join(f"{v:,} {k.replace('_', ' ')}" for k, v in totals["record_counts"].items())
        + f" records, {rows:,} table rows; raw {_fmt_bytes(totals['raw_bytes'])}, normalised {_fmt_bytes(totals['normalized_bytes'])}.",
        f"Built by `aipe-magnetics-database` {index['builder']['version']} from commit `{(index['builder'].get('commit') or 'unknown')[:12]}`.",
        "",
        "## Layout",
        "",
        "| Path | Content |",
        "| --- | --- |",
        "| `raw/<source-id>/files/` | original files, byte-for-byte, as published upstream (immutable) |",
        "| `raw/<source-id>/SOURCES.json` | pinned URL, version and SHA-256 of every original file; upstream LICENSE / NOTICE alongside |",
        "| `normalized/v0.2/<source-id>/` | AIPE Magnetic Data Schema v0.2 records: `index.json`, JSONL shards, `datasets/`, table sidecars (`tables/`, `tables-local/`) |",
        "| `manifests/sources.json` | every investigated source, its licence, status, and what is (not) published and why |",
        "| `manifests/dataset-index.json` | per-source record counts, table rows, sizes, formats, licence and citation |",
        "| `manifests/checksums.json` | SHA-256 and size of every file in the release |",
        "| `manifests/quality-report.json` | automated consistency and plausibility checks |",
        "",
        "## Records",
        "",
        "One JSON record per line, SI units with the published unit kept as `original`; a missing value is omitted, never",
        "zero. `extensions[\"aipe.data\"][\"data_class\"]` keeps experimental measurements, manufacturer datasheet curves,",
        "manufacturer-tool exports, derived views and fitted parameters apart. Tables (`extensions[\"aipe.table\"]`) list",
        "their columns (SI unit, original unit, definition); rows are JSON arrays in a sidecar named by `rows_file` and",
        "verified by `rows_sha256`. Schema: [docs/data-standard-v0.2.md](https://github.com/FulongLi/AIPE-Magnetics-Database/blob/main/docs/data-standard-v0.2.md).",
        "",
        "```python",
        "import aipe_magnetics_database.api as db   # pip install aipe-magnetics-database",
        "rows = db.get_table(\"measurement.princeton-magnet-n87-core-loss\", frequency=\"100 kHz\", temperature=\"90 degC\")",
        "```",
        "",
        "or directly: `huggingface_hub.hf_hub_download(\"FulongLi/AIPE-Magnetics-Dataset\", \"normalized/v0.2/princeton_magnet/index.json\", repo_type=\"dataset\")`.",
        "",
        "## Sources and licences",
        "",
        "Each source keeps its own licence; nothing is relicensed. Cite the original work.",
        "",
        "| Source | Licence | Version | Records | Table rows | Raw | Citation |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in published:
        version = str(s.get("version") or "-")
        version = version[:12] if len(version) == 40 else version
        rec = ", ".join(f"{k} {v:,}" for k, v in s["record_counts"].items())
        lines.append(f"| [{s['title']}]({s['url']}) (`{s['id']}`) | {s['license']} | {version} | {rec} | {sum(s['table_rows'].values()):,} | "
                     f"{s['raw_files']} files, {_fmt_bytes(s['raw_bytes'])} | {(s.get('citation') or '-').replace('|', '/')} |")
    excluded = [s for s in sources if not s.get("huggingface", {}).get("published")]
    lines += ["", f"{len(excluded)} further investigated sources are listed in `manifests/sources.json` as metadata only (licence does not",
              "permit redistribution, not established, or no pinnable download). Public availability is not redistribution permission.",
              "", "## Quality", "",
              f"Automated checks over {q['records_checked']:,} records and {q['rows_checked']:,} table rows: {q['errors']:,} errors, "
              f"{q['warnings']:,} warnings ({', '.join(q['checks_flagged']) or 'none flagged'}). Flags mark values to review; no",
              "observation is changed. Schema validity is not proof of experimental accuracy.", "",
              "## Licence summary", "", "Licences present: " + ", ".join(f"`{x}`" for x in licences) + ". See `LICENSE.md`.", ""]
    return "\n".join(lines)


def licence_file(index: dict[str, Any]) -> str:
    lines = ["# Licences", "", "This dataset aggregates files under different licences. Each `raw/<source-id>/` and",
             "`normalized/v0.2/<source-id>/` subtree is licensed by its source as listed below; the upstream licence text",
             "is kept next to the original files where the source provides one.", "", "| Source | Licence | Redistribution |", "| --- | --- | --- |"]
    for s in index["sources"]:
        lines.append(f"| `{s['id']}` | {s['license']} | {s.get('redistribution') or 'permitted'} |")
    lines += ["", "GPL-3.0 subtrees: modified (normalised) forms of the upstream files; the original files are included under",
              "`raw/<source-id>/files/` as the corresponding source.", ""]
    return "\n".join(lines)


__all__ = ["dataset_card", "licence_file"]
