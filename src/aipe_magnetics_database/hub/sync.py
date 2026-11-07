"""Record a published Hugging Face release in the GitHub Database Manager (``data/huggingface.json``).

``aipe-magnetics-db hub sync [--revision main|<commit>|refs/pr/N]`` reads the public release manifests at that
revision over HTTPS (no account needed), resolves the revision to its exact commit, checks that every file listed
in ``manifests/checksums.json`` exists on the Hub with the same size and SHA-256, and writes ``data/huggingface.json``
(repository, commit, per-source prefixes and the checksum of every published file). The data store then reads large
tables from exactly that commit (:mod:`.remote`). Nothing is written when verification fails.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
from pathlib import Path
from typing import Any, Callable

from aipe_magnetics_database.common.errors import DataIntegrityError
from aipe_magnetics_database.data.sources.cache import urllib_transport
from aipe_magnetics_database.hub.layout import DEFAULT_REPO_ID, ENDPOINT, LAYOUT_VERSION, MANIFEST_PREFIX, resolve_url
from aipe_magnetics_database.hub.remote import RELEASE_FILE
from aipe_magnetics_database.ingest.common import dumps_pretty, write_if_changed

Getter = Callable[[str], bytes]


def _get(url: str) -> bytes:
    with urllib_transport(url, 60.0) as response:
        return response.read()


def _api(repo_id: str, suffix: str, get: Getter) -> Any:
    return json.loads(get(f"{ENDPOINT}/api/datasets/{repo_id}{suffix}"))


def remote_tree(repo_id: str, commit: str, get: Getter = _get) -> dict[str, dict[str, Any]]:
    """``{path: {size, sha256|None, blob_id}}`` of every file at ``commit`` (public tree API, paginated)."""
    out: dict[str, dict[str, Any]] = {}
    url = f"{ENDPOINT}/api/datasets/{repo_id}/tree/{urllib.parse.quote(commit, safe='')}?recursive=true&expand=false"
    while url:
        if get is _get:
            with urllib_transport(url, 60.0) as response:
                body = response.read()
                link = response.headers.get("Link") or ""
        else:
            body, link = get(url), ""
        for item in json.loads(body):
            if item.get("type") == "file":
                out[item["path"]] = {"size": item["size"], "sha256": (item.get("lfs") or {}).get("oid"), "blob_id": item.get("oid")}
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip().strip("<>")
    return out


def sync_release(*, repo_id: str = DEFAULT_REPO_ID, revision: str = "main", root: Path | None = None, get: Getter = _get,
                 verify: bool = True, log: Callable[[str], None] = print) -> dict[str, Any]:
    from aipe_magnetics_database.ingest.pipeline import repository_root

    root = root or repository_root()
    info = _api(repo_id, f"/revision/{urllib.parse.quote(revision, safe='')}", get)
    commit = info["sha"]
    raw_checksums = get(resolve_url(repo_id, commit, f"{MANIFEST_PREFIX}/checksums.json"))
    index = json.loads(get(resolve_url(repo_id, commit, f"{MANIFEST_PREFIX}/dataset-index.json")))
    files = dict(json.loads(raw_checksums)["files"])
    files[f"{MANIFEST_PREFIX}/checksums.json"] = {"sha256": hashlib.sha256(raw_checksums).hexdigest(), "size": len(raw_checksums)}
    if verify:
        tree = remote_tree(repo_id, commit, get)
        problems = []
        for path, meta in files.items():
            remote = tree.get(path)
            if remote is None:
                problems.append(f"missing: {path}")
            elif remote["size"] != meta["size"] or (remote["sha256"] is not None and remote["sha256"] != meta["sha256"]):
                problems.append(f"differs: {path}")
        if problems:
            raise DataIntegrityError(f"release at {repo_id}@{commit[:12]} does not match its checksums.json: {problems[:5]}",
                                     details={"problems": problems[:50]})
        log(f"verified {len(files)} file(s) of {repo_id}@{commit}")
    release = {
        "layout": LAYOUT_VERSION, "repo_id": repo_id, "repo_type": "dataset", "revision": revision, "commit": commit,
        "published_at": info.get("lastModified"), "url": f"https://huggingface.co/datasets/{repo_id}/tree/{commit}",
        "builder": index.get("builder"),
        "sources": {s["id"]: {"raw_prefix": s["raw_prefix"], "normalized_prefix": s["normalized_prefix"], "license": s["license"],
                              "raw_files": s["raw_files"], "raw_bytes": s["raw_bytes"], "normalized_files": s["normalized_files"],
                              "normalized_bytes": s["normalized_bytes"]} for s in index["sources"]},
        "totals": index.get("totals"),
        "files": {k: {"sha256": v["sha256"], "size": v["size"]} for k, v in sorted(files.items())},
    }
    write_if_changed(root / "data" / RELEASE_FILE, dumps_pretty(release).encode("utf-8"))
    return release


__all__ = ["remote_tree", "sync_release"]
