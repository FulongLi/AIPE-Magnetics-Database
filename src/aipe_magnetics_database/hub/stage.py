"""Build a Hugging Face release tree (``raw/``, ``normalized/v0.2/``, ``manifests/``, ``README.md``) from a checkout.

``aipe-magnetics-db hub stage --out DIR [--source ID ...] [--allow-download] [--cache DIR]``

* Only sources whose registry status is ``integrated`` **and** whose licence permits redistribution are staged
  (:func:`publishable`); everything else appears in ``manifests/sources.json`` as metadata only, with the reason.
* Raw files are taken byte-for-byte from ``data/raw/<dir>/files`` (committed, redistributable) or from the SHA-256
  verifying download cache at the pinned URL (``--allow-download``). Every file is hashed again before staging.
* Normalised records are copied unchanged from ``data/normalized/<dir>`` - including the large ``tables-local/``
  sidecars that are not kept in Git - and every table sidecar is checked against the ``rows_sha256`` of its record.
* Files are hard-linked into the stage when possible (no extra disk space), copied otherwise.

The stage never modifies ``data/``; the manifests are deterministic except for ``generated_at``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from aipe_magnetics_database import __version__
from aipe_magnetics_database.common.errors import DataIntegrityError, DataNotFoundError
from aipe_magnetics_database.hub.layout import (
    DEFAULT_REPO_ID,
    LAYOUT_VERSION,
    MANIFEST_PREFIX,
    NORMALIZED_PREFIX,
    RAW_PREFIX,
    SCHEMA_DIR,
    normalized_path,
    raw_path,
)
from aipe_magnetics_database.ingest import registry
from aipe_magnetics_database.ingest.common import dumps_pretty, sha256_path

#: licences under which AIPE may republish the original and normalised files (each keeps its own licence)
REDISTRIBUTABLE_LICENSES = frozenset({"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0",
                                      "ODbL-1.0", "PDDL-1.0", "GPL-2.0-only", "GPL-3.0-only", "GPL-3.0-or-later", "LGPL-3.0-only"})
FORMATS = {".csv": "CSV", ".tsv": "TSV", ".txt": "text", ".json": "JSON", ".ndjson": "NDJSON", ".jsonl": "JSONL", ".xml": "XML",
           ".xlsx": "Excel XLSX", ".xls": "Excel XLS", ".mat": "MATLAB MAT", ".h5": "HDF5", ".hdf5": "HDF5", ".parquet": "Parquet",
           ".zip": "ZIP", ".md": "Markdown", ".dat": "text"}
LICENSE_NAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENSE.upstream.md", "NOTICE", "NOTICE.md", "CITATION.cff")


def publishable(source: dict[str, Any]) -> tuple[bool, str]:
    """``(True, "")`` when a registry entry may be published to the public dataset, else ``(False, reason)``."""
    if source.get("status") != "integrated":
        return False, source.get("reason") or f"status '{source.get('status')}': metadata reference only"
    if not source.get("directory"):
        return False, "bundled with the package (no normalised source directory)"
    if source.get("license") not in REDISTRIBUTABLE_LICENSES:
        return False, f"licence {source.get('license')!r} does not establish redistribution permission"
    if source.get("hf_publish") is False:
        return False, source.get("hf_exclusion_reason", "excluded from the public dataset by the registry")
    return True, ""


def file_format(name: str) -> str:
    return FORMATS.get(Path(name).suffix.lower(), Path(name).suffix.lstrip(".").upper() or "unknown")


def _place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _git_commit(root: Path) -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


class Stage:
    """Accumulates the staged files and their checksums."""

    def __init__(self, out: Path):
        self.out = out
        self.files: dict[str, dict[str, Any]] = {}

    def add(self, rel: str, src: Path, *, sha256: str | None = None) -> dict[str, Any]:
        digest = sha256_path(src)
        if sha256 is not None and digest != sha256:
            raise DataIntegrityError(f"{src}: sha256 {digest} differs from the pinned {sha256}", details={"path": rel})
        _place(src, self.out / rel)
        entry = {"sha256": digest, "size": src.stat().st_size}
        self.files[rel] = entry
        return entry

    def write(self, rel: str, data: bytes) -> None:
        path = self.out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.files[rel] = {"sha256": sha256_path(path), "size": len(data)}


def _raw_inputs(root: Path, source: dict[str, Any], stage: Stage, cache: Any, allow_download: bool) -> list[dict[str, Any]]:
    sid, directory = source["id"], source["directory"]
    raw_dir = root / "data" / "raw" / directory
    manifest_path = raw_dir / "SOURCES.json"
    if not manifest_path.is_file():
        raise DataNotFoundError(f"{sid}: {manifest_path} is missing; run 'aipe-magnetics-db refresh --source {sid}' first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    staged = []
    unavailable = source.get("raw_unavailable") or {}
    for item in manifest["files"]:
        committed = raw_dir / "files" / item["path"]
        if item["path"].startswith("derived/") and (raw_dir / item["path"]).is_file():
            continue  # declared intermediate, staged with derived/ below
        if item["path"] in unavailable and not committed.is_file() and cache.lookup(item["url"]) is None:
            staged.append({"path": None, "original_filename": Path(item["path"]).name, "upstream_path": item["path"], "url": item.get("url"),
                           "format": file_format(item["path"]), "sha256": item["sha256"], "size": item.get("size"), "available": False,
                           "reason": unavailable[item["path"]]})
            continue
        if committed.is_file():
            src = committed
        elif item.get("url"):
            src = cache.fetch(item["url"], item["sha256"], allow_download=allow_download, source_id=sid,
                              filename=Path(item["path"]).name, license=source.get("license", "NOASSERTION"),
                              commit=source.get("version") if len(str(source.get("version"))) == 40 else None)
        else:
            raise DataNotFoundError(f"{sid}: raw file {item['path']} is neither committed nor downloadable (no URL)")
        entry = stage.add(raw_path(sid, f"files/{item['path']}"), Path(src), sha256=item["sha256"])
        staged.append({"path": raw_path(sid, f"files/{item['path']}"), "original_filename": Path(item["path"]).name,
                       "upstream_path": item["path"], "url": item.get("url"), "format": file_format(item["path"]), **entry})
    derived = raw_dir / "derived"  # declared intermediates (e.g. a TSV export of an MS Access file), never originals
    if derived.is_dir():
        for path in sorted(derived.rglob("*")):
            if path.is_file():
                stage.add(raw_path(sid, path.relative_to(raw_dir).as_posix()), path)
    for name in ["SOURCES.json", *LICENSE_NAMES]:
        for candidate in (raw_dir / name, raw_dir / "files" / name):
            if candidate.is_file():
                rel = raw_path(sid, candidate.relative_to(raw_dir).as_posix())
                if rel not in stage.files:
                    stage.add(rel, candidate)
    return staged


def _normalized(root: Path, source: dict[str, Any], stage: Stage, *, require_local_tables: bool) -> dict[str, Any]:
    sid, directory = source["id"], source["directory"]
    base = root / "data" / "normalized" / directory
    index = json.loads((base / "index.json").read_text(encoding="utf-8"))
    rels = ["index.json", *[f for g in index["shards"].values() for f in g], *index["datasets"], *index.get("tables", [])]
    rels += [n for n in LICENSE_NAMES if (base / n).is_file()]
    expected = _table_checksums(base, index)
    expected.update(_waveform_checksums(base, index))
    missing_local: list[str] = []
    for rel in [*index.get("tables_local", []), *index.get("waveforms_local", [])]:
        if (base / rel).is_file():
            rels.append(rel)
        else:
            missing_local.append(rel)
    if missing_local and require_local_tables:
        raise DataNotFoundError(f"{sid}: {len(missing_local)} regenerated table sidecar(s) are missing (e.g. {missing_local[0]}); run "
                                f"'aipe-magnetics-db refresh --source {sid} --allow-download' first")
    for rel in rels:
        stage.add(normalized_path(sid, rel), base / rel, sha256=expected.get(rel))
    return {"index": index, "files": len(rels), "missing_local_tables": missing_local}


def _dataset_citation(root: Path, source: dict[str, Any], index: dict[str, Any]) -> str | None:
    base = root / "data" / "normalized" / source["directory"]
    for rel in index["datasets"]:
        props = json.loads((base / rel).read_text(encoding="utf-8")).get("properties") or {}
        if props.get("citation"):
            return props["citation"]
    return None


def _waveform_checksums(base: Path, index: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in index["shards"].get("waveforms", []):
        for line in (base / rel).read_text(encoding="utf-8").splitlines():
            for art in json.loads(line).get("artifacts", []):
                if art.get("role") == "waveform_data" and art["uri"].startswith("../"):
                    out[art["uri"][3:]] = art["sha256"]
    return out


def _table_checksums(base: Path, index: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in index["shards"].get("measurements", []):
        for line in (base / rel).read_text(encoding="utf-8").splitlines():
            table = json.loads(line).get("extensions", {}).get("aipe.table") or {}
            if table.get("rows_file") and table.get("rows_sha256"):
                out[table["rows_file"]] = table["rows_sha256"]
    return out


def build_stage(out: str | Path, *, root: Path | None = None, sources: list[str] | None = None, allow_download: bool = False,
                cache_dir: str | None = None, require_local_tables: bool = True, repo_id: str = DEFAULT_REPO_ID,
                log: Callable[[str], None] = print) -> dict[str, Any]:
    """Write the release tree to ``out`` (which must be empty or absent); returns the dataset index."""
    from aipe_magnetics_database.data.sources.cache import LocalCache
    from aipe_magnetics_database.ingest.pipeline import repository_root
    from aipe_magnetics_database.ingest.quality import run_checks

    root = root or repository_root()
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"stage directory {out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    stage = Stage(out)
    cache = LocalCache(cache_dir) if cache_dir else LocalCache()
    chosen = set(sources or [])
    unknown = chosen - {s["id"] for s in registry.SOURCES}
    if unknown:
        raise ValueError(f"unknown source id(s): {sorted(unknown)}")

    entries, per_source = [], []
    for source in registry.SOURCES:
        ok, reason = publishable(source)
        selected = ok and (not chosen or source["id"] in chosen)
        entry = {k: source.get(k) for k in ("id", "title", "priority", "status", "url", "version", "license", "redistribution",
                                            "retrieved_at", "citation", "importer", "adapter", "reason", "notes") if source.get(k) is not None}
        if not selected:
            entry["huggingface"] = {"published": False, "reason": reason or "not selected for this release"}
            entries.append(entry)
            continue
        log(f"[{source['id']}] staging raw files ...")
        raw = _raw_inputs(root, source, stage, cache, allow_download)
        log(f"[{source['id']}] staging normalised records ...")
        norm = _normalized(root, source, stage, require_local_tables=require_local_tables)
        index = norm["index"]
        archived = [f for f in raw if f.get("available", True)]
        raw_bytes = sum(f["size"] for f in archived)
        citation = source.get("citation") or _dataset_citation(root, source, index)
        norm_bytes = sum(v["size"] for k, v in stage.files.items() if k.startswith(f"{NORMALIZED_PREFIX}/{source['id']}/"))
        entry["huggingface"] = {"published": True, "raw_prefix": f"{RAW_PREFIX}/{source['id']}/",
                                "normalized_prefix": f"{NORMALIZED_PREFIX}/{source['id']}/", "raw_files": raw,
                                "normalized_files": norm["files"], "missing_local_tables": norm["missing_local_tables"]}
        entries.append(entry)
        per_source.append({"id": source["id"], "title": source["title"], "license": source["license"], "version": source.get("version"),
                           "url": source["url"], "citation": citation, "redistribution": source.get("redistribution"),
                           "record_counts": index["record_counts"],
                           "table_rows": index["table_rows"], "raw_files": len(archived), "raw_bytes": raw_bytes,
                           "raw_unavailable": [f["upstream_path"] for f in raw if not f.get("available", True)],
                           "retrieved_at": source.get("retrieved_at"),
                           "normalized_files": norm["files"], "normalized_bytes": norm_bytes,
                           "raw_formats": sorted({f["format"] for f in archived}),
                           "normalized_prefix": f"{NORMALIZED_PREFIX}/{source['id']}/", "raw_prefix": f"{RAW_PREFIX}/{source['id']}/"})

    log("running quality checks ...")
    quality = run_checks(root, raw_checksums={k: v["sha256"] for k, v in stage.files.items() if k.startswith(f"{RAW_PREFIX}/")})
    now = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    totals: dict[str, int] = {}
    rows: dict[str, int] = {}
    for s in per_source:
        for k, v in s["record_counts"].items():
            totals[k] = totals.get(k, 0) + v
        for k, v in s["table_rows"].items():
            rows[k] = rows.get(k, 0) + v
    dataset_index = {
        "layout": LAYOUT_VERSION, "schema_version": "0.2.0", "schema_dir": SCHEMA_DIR, "repo_id": repo_id, "generated_at": now,
        "builder": {"package": "aipe-magnetics-database", "version": __version__, "commit": _git_commit(root),
                    "repository": "https://github.com/FulongLi/AIPE-Magnetics-Database"},
        "sources": per_source, "totals": {"record_counts": dict(sorted(totals.items())), "table_rows": dict(sorted(rows.items())),
                                          "raw_bytes": sum(s["raw_bytes"] for s in per_source),
                                          "normalized_bytes": sum(s["normalized_bytes"] for s in per_source)},
        "quality_summary": quality["summary"],
        "note": "Each source keeps its own licence. Schema validity is not proof of experimental accuracy.",
    }
    stage.write(f"{MANIFEST_PREFIX}/sources.json", dumps_pretty(entries).encode("utf-8"))
    stage.write(f"{MANIFEST_PREFIX}/dataset-index.json", dumps_pretty(dataset_index).encode("utf-8"))
    stage.write(f"{MANIFEST_PREFIX}/quality-report.json", dumps_pretty(quality).encode("utf-8"))
    from aipe_magnetics_database.hub.card import dataset_card, licence_file
    stage.write("README.md", dataset_card(dataset_index, entries).encode("utf-8"))
    stage.write("LICENSE.md", licence_file(dataset_index).encode("utf-8"))
    checksums = {"layout": LAYOUT_VERSION, "algorithm": "sha256",
                 "files": {k: stage.files[k] for k in sorted(stage.files)}}
    (out / MANIFEST_PREFIX / "checksums.json").write_bytes(dumps_pretty(checksums).encode("utf-8"))
    log(f"staged {len(stage.files) + 1} file(s): raw {dataset_index['totals']['raw_bytes']:,} B, normalised "
        f"{dataset_index['totals']['normalized_bytes']:,} B in {out}")
    return dataset_index


def load_checksums(stage_dir: str | Path) -> dict[str, dict[str, Any]]:
    """``{path: {sha256, size}}`` of a stage, including ``manifests/checksums.json`` itself."""
    stage_dir = Path(stage_dir)
    data = json.loads((stage_dir / MANIFEST_PREFIX / "checksums.json").read_text(encoding="utf-8"))["files"]
    path = stage_dir / MANIFEST_PREFIX / "checksums.json"
    data[f"{MANIFEST_PREFIX}/checksums.json"] = {"sha256": sha256_path(path), "size": path.stat().st_size}
    return data


def verify_stage(stage_dir: str | Path) -> list[str]:
    """Re-hash every staged file against ``manifests/checksums.json``; returns the problems (empty when intact)."""
    stage_dir = Path(stage_dir)
    problems = []
    listed = load_checksums(stage_dir)
    for rel, meta in listed.items():
        path = stage_dir / rel
        if not path.is_file():
            problems.append(f"missing: {rel}")
        elif path.stat().st_size != meta["size"] or sha256_path(path) != meta["sha256"]:
            problems.append(f"checksum mismatch: {rel}")
    on_disk = {p.relative_to(stage_dir).as_posix() for p in stage_dir.rglob("*") if p.is_file() and ".cache" not in p.parts}
    problems += [f"unlisted file: {rel}" for rel in sorted(on_disk - set(listed))]
    return problems


__all__ = ["REDISTRIBUTABLE_LICENSES", "build_stage", "file_format", "load_checksums", "publishable", "verify_stage"]
