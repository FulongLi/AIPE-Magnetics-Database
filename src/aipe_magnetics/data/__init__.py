"""AIPE Magnetic Data Standard v0.2: records, JSON Schema, migration, HDF5 waveforms, catalogue, importers.

* :mod:`aipe_magnetics.data.records`   - normative pydantic record models (``MagneticRecord`` + property models)
* :mod:`aipe_magnetics.data.schema`    - JSON Schema 2020-12 generation / export / validation (version dispatch)
* :mod:`aipe_magnetics.data.migrate`   - mechanical v0.1 -> v0.2 migration
* :mod:`aipe_magnetics.data.hdf5`      - ``WaveformStore`` for the ``aipe.magnetics.waveforms/0.2`` HDF5 layout
* :mod:`aipe_magnetics.data.catalog`   - bundled + user catalogues with typed look-ups
* :mod:`aipe_magnetics.data.importers` - explicit, non-overwriting importers (JSON, CSV loss tables, waveform sets)
* :mod:`aipe_magnetics.data.fixtures`  - packaged synthetic fixtures (v0.1 records per kind)

The names in ``__all__`` are re-exported **lazily** (PEP 562 module ``__getattr__``): importing the
package does not import its submodules, so ``python -m aipe_magnetics.data.schema`` (and the other
``-m`` entry points) execute their module exactly once and without the ``RuntimeWarning`` that an
eager package import would cause. ``from aipe_magnetics.data import Catalog`` keeps working.

``validate_catalogue`` exported here is the version-dispatching wrapper of
:func:`aipe_magnetics.data.schema.validate_catalogue`; the pure cross-record check on
``MagneticRecord`` lists remains available as :func:`aipe_magnetics.data.records.validate_catalogue`.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS_BY_MODULE: dict[str, tuple[str, ...]] = {
    "records": (
        "ID_PATTERN", "PROPERTY_MODELS", "REFERENCE_KINDS", "SCHEMA_VERSION", "Artifact", "ArrayLayout", "Attribution",
        "BHLoopProperties", "ComponentProperties", "ConditionedValue", "CoreDimensions", "CoreFamilyProperties",
        "CoreGeometryProperties", "CoreLossMeasurementProperties", "CoreMaterialProperties", "DatasetProperties", "Evidence",
        "ExcitationWaveformProperties", "LossModelProperties", "LossReferencePoint", "MagneticRecord", "ManufacturerProperties",
        "MaterialProperties", "MeasurementProperties", "ModelMetadataProperties", "OperatingConditionProperties", "Provenance",
        "Quality", "RangeSI", "RecordKind", "SteinmetzParameterSet", "TemperaturePoint", "WaveformDatasetProperties",
        "WindingProperties",
    ),
    "schema": (
        "LEGACY_SCHEMA_VERSION", "SCHEMA_ID_V02", "SUPPORTED_SCHEMA_VERSIONS", "check_exported", "check_exported_schemas",
        "export_schemas", "generate_schema_v02", "generator_versions", "legacy_record_errors", "load_schema", "mirror_schema_dir",
        "package_schema_dir", "physical_errors", "repository_root", "schema_errors", "semantic_errors", "validate_catalogue",
        "validate_files", "validate_record",
    ),
    "migrate": ("MIGRATION_EXTENSION", "migrate_file", "migrate_record"),
    "hdf5": (
        "CONDITION_QUANTITIES", "LAYOUT_VERSION", "SEQUENCE_QUANTITIES", "PreparedWaveforms", "WaveformData", "WaveformMeta",
        "WaveformStore", "array_sha256", "convert_array", "h5py_available", "missing_mask",
    ),
    "catalog": ("BUNDLED_CATALOG_DIR", "Catalog", "load_record_file"),
    "importers": (
        "IMPORT_EXTENSION", "ImportResult", "import_csv_loss_table", "import_json_record", "import_record",
        "import_waveform_sequences",
    ),
}
_EXPORTS: dict[str, str] = {name: module for module, names in _EXPORTS_BY_MODULE.items() for name in names}
_SUBMODULES: frozenset[str] = frozenset({"records", "schema", "migrate", "hdf5", "catalog", "importers", "fixtures", "sources"})

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:  # ``aipe_magnetics.data.schema`` as an attribute, as with an eager package
        return importlib.import_module(f"{__name__}.{name}")
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value  # cache for subsequent look-ups
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)
