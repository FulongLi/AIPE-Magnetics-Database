"""Hugging Face storage layer of the AIPE Magnetics Database.

The Hugging Face dataset ``FulongLi/AIPE-Magnetics-Dataset`` (Layer 1) stores the original source files, the
normalised v0.2 records, large table sidecars and the release manifests. This repository (Layer 2) builds the
release (:mod:`.stage`), publishes it (:mod:`.publish`, GitHub Actions Trusted Publisher, never a stored token) and
reads it back version-pinned (:mod:`.remote`; plain HTTPS through the SHA-256 verifying cache, so the optional
``huggingface_hub`` dependency is only needed to publish).

Layout of the dataset repository::

    raw/<source-id>/                 original files byte-for-byte + SOURCES.json, LICENSE / NOTICE files
    normalized/v0.2/<source-id>/     index.json, record shards, datasets/, tables/ (every sidecar, incl. large ones)
    manifests/sources.json           every investigated source, its status and what is published for it
    manifests/dataset-index.json     per-source record counts, table rows, sizes, licence, citation
    manifests/checksums.json         SHA-256 and size of every published file
    manifests/quality-report.json    automated quality checks of the release
    README.md                        dataset card
"""

from aipe_magnetics_database.hub.layout import (
    DEFAULT_REPO_ID,
    LAYOUT_VERSION,
    NORMALIZED_PREFIX,
    RAW_PREFIX,
    REPO_TYPE,
    normalized_path,
    raw_path,
)

__all__ = ["DEFAULT_REPO_ID", "LAYOUT_VERSION", "NORMALIZED_PREFIX", "RAW_PREFIX", "REPO_TYPE", "normalized_path", "raw_path"]
