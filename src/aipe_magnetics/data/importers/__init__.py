"""Explicit, non-overwriting importers for user-supplied data (no manufacturer data is bundled).

* :func:`import_json_record` - v0.2 (and, with ``migrate_v01=True``, v0.1) record files;
  :func:`import_record` is its path-oriented variant for command-line consumers.
* :func:`import_csv_loss_table` - CSV core-loss tables with an explicit column -> (quantity, unit) mapping.
* :func:`import_waveform_sequences` - HDF5 / npz / CSV waveform sets into the AIPE HDF5 layout.

All importers record the SHA-256 of the original bytes in ``extensions["aipe.import"]``, never
overwrite destinations, never infer units and validate every record with every layer (JSON Schema,
normative semantics, physical plausibility, catalogue cross-references) before anything is written.

The names below are re-exported lazily (PEP 562) so that ``python -m aipe_magnetics.data.importers.json_record``
executes that module exactly once, without a ``RuntimeWarning``.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS_BY_MODULE: dict[str, tuple[str, ...]] = {
    "base": ("IMPORT_EXTENSION", "NO_NORMALIZATION", "ImportResult", "check_against_catalogue", "coerce_evidence",
             "explicit_null_paths", "import_extension", "write_record"),
    "csv_loss_table": ("LOSS_TABLE_QUANTITIES", "REQUIRED_QUANTITIES", "import_csv_loss_table", "parse_loss_table"),
    "hdf5_waveforms": ("import_waveform_sequences",),
    "json_record": ("import_json_record", "import_record"),
}
_EXPORTS: dict[str, str] = {name: module for module, names in _EXPORTS_BY_MODULE.items() for name in names}
_SUBMODULES: frozenset[str] = frozenset(_EXPORTS_BY_MODULE)

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:  # ``importers.base`` as an attribute, as with an eager package
        return importlib.import_module(f"{__name__}.{name}")
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)
