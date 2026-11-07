"""Version-pinned, SHA-256 verified reads from the published Hugging Face release.

``data/huggingface.json`` (committed, bundled into the wheel) records the dataset repository, the exact commit of
the published release and the SHA-256 / size of every published file. :func:`fetch` downloads one of those files
from ``https://huggingface.co/datasets/<repo>/resolve/<commit>/<path>`` into the local cache
(``AIPE_MAGNETICS_CACHE`` or ``~/.cache/aipe-magnetics``) through :class:`LocalCache`, which streams to a temporary
file, verifies the digest and only then admits the file - so an interrupted download is simply retried and a cached
file is never downloaded twice. No Hugging Face account or ``huggingface_hub`` installation is needed.

Downloads happen only when allowed: ``allow_download=True`` or the environment variable
``AIPE_MAGNETICS_ALLOW_DOWNLOAD=1``. Offline use of the data bundled with the package is unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from aipe_magnetics_database.common.errors import DataNotFoundError
from aipe_magnetics_database.hub.layout import DEFAULT_REPO_ID, normalized_path, resolve_url

RELEASE_FILE = "huggingface.json"
ALLOW_ENV = "AIPE_MAGNETICS_ALLOW_DOWNLOAD"


def downloads_allowed(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return explicit
    return os.environ.get(ALLOW_ENV, "").strip().lower() in ("1", "true", "yes")


def default_cache_dir() -> Path:
    env = os.environ.get("AIPE_MAGNETICS_CACHE")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "aipe-magnetics"


def load_release(data_root: str | Path) -> dict[str, Any] | None:
    """The published-release record of a data store, or ``None`` when nothing has been published yet."""
    path = Path(data_root) / RELEASE_FILE
    if not path.is_file():
        return None
    release = json.loads(path.read_text(encoding="utf-8"))
    return release if release.get("commit") else None


class HubRelease:
    """One published release (repository + exact commit + file checksums)."""

    def __init__(self, release: dict[str, Any], *, cache_dir: str | Path | None = None):
        self.release = release
        self.repo_id = release.get("repo_id", DEFAULT_REPO_ID)
        self.commit = release["commit"]
        self.files: dict[str, dict[str, Any]] = release.get("files", {})
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()

    @classmethod
    def from_data_root(cls, data_root: str | Path, **kwargs: Any) -> "HubRelease | None":
        release = load_release(data_root)
        return cls(release, **kwargs) if release else None

    def url(self, path: str) -> str:
        return resolve_url(self.repo_id, self.commit, path)

    def has(self, path: str) -> bool:
        return path in self.files

    def table_path(self, source_id: str, rows_file: str) -> str:
        return normalized_path(source_id, rows_file)

    def fetch(self, path: str, *, allow_download: bool | None = None) -> Path:
        """Local path of a published file (cached; SHA-256 verified)."""
        from aipe_magnetics_database.data.sources.cache import LocalCache
        meta = self.files.get(path)
        if meta is None:
            raise DataNotFoundError(f"{path} is not part of the published release {self.repo_id}@{self.commit[:12]}")
        cache = LocalCache(self.cache_dir)
        return cache.fetch(self.url(path), meta["sha256"], allow_download=downloads_allowed(allow_download), commit=self.commit,
                           license=meta.get("license", "NOASSERTION"), source_id=path.split("/")[2] if path.startswith("normalized/") else None,
                           filename=Path(path).name)

    def status(self) -> dict[str, Any]:
        return {k: self.release.get(k) for k in ("repo_id", "revision", "commit", "published_at", "release_id", "pull_request", "layout")} | {
            "files": len(self.files), "bytes": sum(m["size"] for m in self.files.values()),
            "url": f"https://huggingface.co/datasets/{self.repo_id}/tree/{self.commit}", "cache_dir": str(self.cache_dir)}


__all__ = ["ALLOW_ENV", "HubRelease", "RELEASE_FILE", "default_cache_dir", "downloads_allowed", "load_release"]
