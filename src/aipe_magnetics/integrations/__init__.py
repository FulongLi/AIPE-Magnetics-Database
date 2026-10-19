"""AIPE ecosystem compatibility adapters.

* :mod:`aipe_magnetics.integrations.aipe_core` - Core Engineering State 0.1.0: vendored schemas,
  offline contract validation and the executable summary <-> state round trip.
* :mod:`aipe_magnetics.integrations.aipe_registry` - Registry capability manifest (``aipe.yaml``):
  vendored schema, Registry loader rules, diagnostics and the ``aipe.magnetics`` extension model.

Logical schema compatibility (these adapters) is separate from runtime installation: neither
AIPE-Core nor AIPE-Registry needs to be installed or running; the pinned copies are MIT-licensed
snapshots used for offline compatibility testing only (see the NOTICE.md files).
"""

__all__ = ["aipe_core", "aipe_registry"]
