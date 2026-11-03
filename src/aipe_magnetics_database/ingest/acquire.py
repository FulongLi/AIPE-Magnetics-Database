"""Pinned, checksum-verified acquisition of original source files (download only on request).

A :class:`RawFile` names one original file by its pinned URL and SHA-256. :func:`obtain` resolves it, in this
order, from a user-supplied local directory (``--local <source>=<dir>``; the file must hash to the pinned value),
then from the :class:`~aipe_magnetics_database.data.sources.cache.LocalCache` (``AIPE_MAGNETICS_CACHE`` or
``.aipe-cache/``), downloading into the cache only with ``allow_download=True``. Partial downloads never enter
the cache (the cache streams to a temporary file and verifies the digest), so an interrupted refresh simply
resumes on the next run; files already cached are not downloaded again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aipe_magnetics_database.common.errors import DataIntegrityError, DataNotFoundError
from aipe_magnetics_database.data.sources.cache import LocalCache
from aipe_magnetics_database.ingest.common import sha256_path


@dataclass(frozen=True)
class RawFile:
    name: str                 # path relative to the source root, e.g. data/core_materials.ndjson
    url: str                  # pinned URL
    sha256: str
    size: int | None = None
    license: str = "NOASSERTION"


def obtain(source_id: str, files: list[RawFile], *, local_dir: str | Path | None = None, cache: LocalCache | None = None,
           allow_download: bool = False, commit: str | None = None) -> dict[str, Path]:
    """Local paths of ``files`` keyed by :attr:`RawFile.name`; every file is SHA-256 verified."""
    out: dict[str, Path] = {}
    for item in files:
        if local_dir is not None:
            for candidate in (Path(local_dir) / item.name, Path(local_dir) / Path(item.name).name):
                if candidate.is_file():
                    digest = sha256_path(candidate)
                    if digest != item.sha256:
                        raise DataIntegrityError(f"{candidate}: sha256 {digest} differs from the pinned {item.sha256} of {item.url}",
                                                 details={"expected": item.sha256, "actual": digest, "source_id": source_id})
                    out[item.name] = candidate
                    break
            if item.name in out:
                continue
        cache = cache or LocalCache()
        try:
            out[item.name] = cache.fetch(item.url, item.sha256, allow_download=allow_download, commit=commit, license=item.license,
                                         source_id=source_id, filename=Path(item.name).name)
        except DataNotFoundError:
            raise
    return out


__all__ = ["RawFile", "obtain"]
