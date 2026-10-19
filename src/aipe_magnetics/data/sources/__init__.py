"""External data-source adapters (discovery, inspection, extraction, normalisation, attribution).

No third-party dataset is bundled. Each adapter reads a user-supplied checkout or an on-demand,
pinned, checksum-verified download in the local cache, subject to the source licence:

* :class:`PrincetonMagnetAdapter` (MIT) - packaged scalar core-loss databases, website cycle zips;
* :class:`MagnetChallenge2Adapter` (NOASSERTION) - reference + user-supplied import only, never downloads;
* :class:`PaderbornMaterialDatabaseAdapter` (GPL-3.0-only) - user checkout / explicit pinned clone; never bundled;
* :class:`MagnetHubAdapter` (MIT) - model/fixture discovery, fixture import, delegated inference.

See ``docs/sources-adapters.md`` for usage, unit mappings, download/caching and licence behaviour.
"""

from aipe_magnetics.data.sources.base import (
    LICENSE_ACTIONS,
    SOURCES_EXTENSION,
    TABLE_LAYOUT_VERSION,
    UNPINNED_FLAG,
    ExtractedData,
    PinStatus,
    SourceAdapter,
    SourceDescriptor,
    SourceItem,
    TableData,
    TableDescription,
    TableMeta,
    TableStore,
)
from aipe_magnetics.data.sources.cache import CACHE_ENV_VAR, DEFAULT_CACHE_DIRNAME, DownloadNotAllowedError, LocalCache
from aipe_magnetics.data.sources.magnet_challenge2 import MagnetChallenge2Adapter
from aipe_magnetics.data.sources.paderborn_magnethub import MagnetHubAdapter
from aipe_magnetics.data.sources.paderborn_materialdatabase import PaderbornMaterialDatabaseAdapter
from aipe_magnetics.data.sources.princeton_magnet import PrincetonMagnetAdapter
from aipe_magnetics.data.sources.princeton_magnet_db import (
    MagnetDatabase,
    SteinmetzFit,
    find_database_files,
    fit_steinmetz,
    fit_steinmetz_by_temperature,
    read_magnet_database,
)

ADAPTERS: dict[str, type[SourceAdapter]] = {
    PrincetonMagnetAdapter.descriptor.source_id: PrincetonMagnetAdapter,
    MagnetChallenge2Adapter.descriptor.source_id: MagnetChallenge2Adapter,
    PaderbornMaterialDatabaseAdapter.descriptor.source_id: PaderbornMaterialDatabaseAdapter,
    MagnetHubAdapter.descriptor.source_id: MagnetHubAdapter,
}


def descriptors() -> list[SourceDescriptor]:
    """The descriptors of every registered adapter (deterministic order)."""
    return [cls.descriptor for cls in ADAPTERS.values()]


def get_adapter(source_id: str, **kwargs: object) -> SourceAdapter:
    """Instantiate the adapter registered under ``source_id``."""
    from aipe_magnetics.common.errors import DataNotFoundError

    try:
        cls = ADAPTERS[source_id]
    except KeyError:
        raise DataNotFoundError(f"unknown source adapter {source_id!r}; registered: {sorted(ADAPTERS)}") from None
    return cls(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "ADAPTERS", "CACHE_ENV_VAR", "DEFAULT_CACHE_DIRNAME", "LICENSE_ACTIONS", "SOURCES_EXTENSION", "TABLE_LAYOUT_VERSION", "UNPINNED_FLAG",
    "DownloadNotAllowedError", "ExtractedData", "LocalCache", "MagnetChallenge2Adapter", "MagnetDatabase", "MagnetHubAdapter",
    "PaderbornMaterialDatabaseAdapter", "PinStatus", "PrincetonMagnetAdapter", "SourceAdapter", "SourceDescriptor", "SourceItem", "SteinmetzFit",
    "TableData", "TableDescription", "TableMeta", "TableStore", "descriptors", "find_database_files", "fit_steinmetz",
    "fit_steinmetz_by_temperature", "get_adapter", "read_magnet_database",
]
