"""Query layer of the AIPE Magnetics Database over the curated catalogue (``Catalog``).

Read-only, compact, paginated result envelopes for the bundled curated catalogue (MCP / CLI / AI use) and
typed snapshots consumed by the separate AIPE-Magnetics-Design package. The repository-wide store of
normalised source records and canonical consolidated records is :mod:`aipe_magnetics_database.store`.

Dependency rule: this package imports only ``common`` and ``data``. Loss prediction, transformer sizing and
optimisation are deliberately absent; they live in ``aipe_magnetics_design``.

Interfaces
----------
* Query functions: :func:`search_materials`, :func:`get_material_properties`, :func:`search_cores`,
  :func:`get_core`, :func:`query_saturation_data`, :func:`query_core_loss_data`,
  :func:`get_physical_properties`, :func:`validate_dataset`, :func:`import_dataset`, :func:`info`.
* Typed snapshots: :func:`material_data`, :func:`core_data`, :func:`list_core_data`
  (:class:`MaterialData`, :class:`CoreData`) and :func:`physical_property` (thermal and bulk properties with bounds).
"""

from __future__ import annotations

from aipe_magnetics_database.database.accessors import CoreData, MaterialData, core_data, list_core_data, material_data
from aipe_magnetics_database.database.properties import describe_entry, physical_property, property_materials, property_source
from aipe_magnetics_database.database.query import (
    get_core,
    get_material_properties,
    get_physical_properties,
    import_dataset,
    info,
    query_core_loss_data,
    query_saturation_data,
    search_cores,
    search_materials,
    validate_dataset,
)

CAPABILITIES: dict[str, dict[str, str]] = {
    "search_materials": {"id": "aipe.magnetics.database.search_materials", "kind": "read_only",
                         "description": "Search materials by text, class, manufacturer, data availability and loss-model coverage (paginated)."},
    "get_material_properties": {"id": "aipe.magnetics.database.get_material_properties", "kind": "read_only",
                                "description": "Material properties, valid operating ranges, parameter sets, thermal properties, provenance."},
    "search_cores": {"id": "aipe.magnetics.database.search_cores", "kind": "read_only",
                     "description": "Search core geometries by material, shape, size and area product (paginated)."},
    "get_core": {"id": "aipe.magnetics.database.get_core", "kind": "read_only", "description": "Core geometry record with dimensions."},
    "query_saturation_data": {"id": "aipe.magnetics.database.query_saturation_data", "kind": "read_only",
                              "description": "Saturation, remanence, coercivity, permeability points and B-H loop records (decimated)."},
    "query_core_loss_data": {"id": "aipe.magnetics.database.query_core_loss_data", "kind": "read_only",
                             "description": "Core-loss reference points and Steinmetz parameter sets (decimated)."},
    "get_physical_properties": {"id": "aipe.magnetics.database.get_physical_properties", "kind": "read_only",
                                "description": "Thermal and bulk properties (ferrite, copper, winding packs, insulation, air) with bounds."},
    "validate_dataset": {"id": "aipe.magnetics.database.validate_dataset", "kind": "read_only",
                         "description": "Validate record files against the v0.1 / v0.2 schemas."},
    "import_dataset": {"id": "aipe.magnetics.database.import_dataset", "kind": "write_optional",
                       "description": "Import a JSON record file into a validated review candidate (writing requires authorisation)."},
}

__all__ = [
    "CAPABILITIES", "CoreData", "MaterialData", "core_data", "describe_entry",
    "get_core", "get_material_properties", "get_physical_properties", "import_dataset", "info",
    "list_core_data", "material_data", "physical_property", "property_materials", "property_source", "query_core_loss_data",
    "query_saturation_data", "search_cores", "search_materials", "validate_dataset",
]
