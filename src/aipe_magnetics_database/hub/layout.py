"""Path conventions of the Hugging Face dataset repository (see :mod:`aipe_magnetics_database.hub`)."""

from __future__ import annotations

import posixpath

DEFAULT_REPO_ID = "FulongLi/AIPE-Magnetics-Dataset"
REPO_TYPE = "dataset"
OIDC_RESOURCE = f"datasets/{DEFAULT_REPO_ID}"
ENDPOINT = "https://huggingface.co"
LAYOUT_VERSION = "aipe.magnetics.hf-layout/1.0"
SCHEMA_DIR = "v0.2"
RAW_PREFIX = "raw"
NORMALIZED_PREFIX = f"normalized/{SCHEMA_DIR}"
MANIFEST_PREFIX = "manifests"
MANIFEST_FILES = ("sources.json", "dataset-index.json", "checksums.json", "quality-report.json")
#: raw files are immutable: a published path under raw/ may never be replaced by different bytes
IMMUTABLE_PREFIXES = (f"{RAW_PREFIX}/",)


def _clean(rel: str) -> str:
    rel = rel.replace("\\", "/")
    norm = posixpath.normpath(rel)
    if norm.startswith("../") or norm == ".." or norm.startswith("/"):
        raise ValueError(f"path escapes its prefix: {rel!r}")
    return norm


def raw_path(source_id: str, rel: str) -> str:
    """``raw/<source-id>/<rel>`` (``rel`` relative to the source's raw directory, e.g. ``files/x.csv``)."""
    return f"{RAW_PREFIX}/{source_id}/{_clean(rel)}"


def normalized_path(source_id: str, rel: str) -> str:
    """``normalized/v0.2/<source-id>/<rel>`` (``rel`` relative to ``data/normalized/<directory>``)."""
    return f"{NORMALIZED_PREFIX}/{source_id}/{_clean(rel)}"


def resolve_url(repo_id: str, revision: str, path: str, endpoint: str = ENDPOINT) -> str:
    """Direct download URL of one file at an exact revision (``/resolve/<revision>/<path>``)."""
    from urllib.parse import quote
    return f"{endpoint}/datasets/{repo_id}/resolve/{quote(revision, safe='')}/{quote(path)}"


__all__ = ["DEFAULT_REPO_ID", "ENDPOINT", "IMMUTABLE_PREFIXES", "LAYOUT_VERSION", "MANIFEST_FILES", "MANIFEST_PREFIX", "NORMALIZED_PREFIX",
           "OIDC_RESOURCE", "RAW_PREFIX", "REPO_TYPE", "SCHEMA_DIR", "normalized_path", "raw_path", "resolve_url"]
