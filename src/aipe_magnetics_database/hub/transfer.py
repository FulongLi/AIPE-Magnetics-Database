"""Restore regenerated table sidecars whose upstream input is no longer retrievable (``hub restore-tables``).

Some ``tables-local/`` sidecars are too large for Git and can only be regenerated from an upstream file. When that
upstream file disappears (MAS ``advanced_core_materials.ndjson``: its Git LFS object is no longer served), the last
verified copy travels to the publishing workflow as a pinned transfer asset (``pins/transfer-assets.json``: URL and
SHA-256 of a tar.gz). Restoring downloads it through the SHA-256 verifying cache, extracts only regular files into the
listed directory (no absolute paths, no ``..``, no links) and checks every extracted sidecar against the
``rows_sha256`` of the committed record that references it - so the asset cannot introduce anything the committed
records do not already pin. Once a release is on Hugging Face, the store reads these tables from there instead.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from aipe_magnetics_database.common.errors import DataIntegrityError

PINS = Path(__file__).resolve().parent.parent / "ingest" / "pins" / "transfer-assets.json"


def pins() -> list[dict[str, Any]]:
    return json.loads(PINS.read_text(encoding="utf-8"))["assets"] if PINS.is_file() else []


def _expected(directory: Path) -> dict[str, str]:
    from aipe_magnetics_database.hub.stage import _table_checksums
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    local = set(index.get("tables_local", []))
    return {k: v for k, v in _table_checksums(directory, index).items() if k in local}


def safe_extract(archive: Path, target: Path, allowed: set[str]) -> list[str]:
    """Extract the regular files of ``archive`` whose path is in ``allowed`` into ``target``; returns their paths."""
    written = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = PurePosixPath(member.name)
            if member.isdir():
                continue
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                raise DataIntegrityError(f"{archive.name}: refusing archive member {member.name!r}")
            rel = name.as_posix()
            if rel not in allowed:
                raise DataIntegrityError(f"{archive.name}: unexpected member {rel!r} (not referenced by a committed record)")
            source = tar.extractfile(member)
            assert source is not None
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(source.read())
            written.append(rel)
    return written


def restore(*, root: Path | None = None, cache_dir: str | None = None, allow_download: bool = False,
            log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    from aipe_magnetics_database.data.sources.cache import LocalCache
    from aipe_magnetics_database.ingest.common import sha256_path
    from aipe_magnetics_database.ingest.pipeline import repository_root

    root = root or repository_root()
    cache = LocalCache(cache_dir) if cache_dir else LocalCache()
    results = []
    for asset in pins():
        directory = root / asset["extract_to"]
        expected = _expected(directory)
        missing = [rel for rel in expected if not (directory / rel).is_file()]
        if not missing:
            log(f"[{asset['id']}] all {len(expected)} sidecar(s) already present")
            results.append({"id": asset["id"], "restored": 0, "present": len(expected)})
            continue
        archive = cache.fetch(asset["url"], asset["sha256"], allow_download=allow_download, source_id=asset.get("source_id"),
                              filename=Path(asset["url"]).name, license=asset.get("license", "NOASSERTION"))
        written = safe_extract(Path(archive), directory, set(expected))
        bad = [rel for rel in written if sha256_path(directory / rel) != expected[rel]]
        if bad:
            for rel in bad:
                (directory / rel).unlink()
            raise DataIntegrityError(f"[{asset['id']}] {len(bad)} sidecar(s) differ from their record's rows_sha256: {bad[:3]}")
        still = [rel for rel in expected if not (directory / rel).is_file()]
        if still:
            raise DataIntegrityError(f"[{asset['id']}] {len(still)} sidecar(s) are not in the transfer asset: {still[:3]}")
        log(f"[{asset['id']}] restored {len(written)} sidecar(s), each verified against its record's rows_sha256")
        results.append({"id": asset["id"], "restored": len(written), "present": len(expected)})
    return results


__all__ = ["PINS", "pins", "restore", "safe_extract"]
