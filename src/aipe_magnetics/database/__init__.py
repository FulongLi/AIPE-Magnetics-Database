"""AIPE Magnetics **Database** capability: the stable, documented data interface.

This package is the Database half of the v0.3 architecture. It is usable on its own (``pip install
aipe-magnetics``, no CAD / FEA / plotting dependencies) and it is the *only* way the Design half
(:mod:`aipe_magnetics.design`, ``geometry``, ``simulation``, ``optimisation``, ``workflows``) reads
material and core data.

Dependency rule (enforced by ``tests/architecture/test_boundaries.py``): ``database`` imports only
``common``, ``data``, ``models``, ``verification`` and the data-oriented ``engineering`` modules
(``materials``, ``core_loss``, ``saturation``, ``formulas``); it never imports any Design package.

Interfaces
----------
* Query functions returning compact, paginated result envelopes (MCP / CLI / AI use):
  :func:`search_materials`, :func:`get_material_properties`, :func:`search_cores`, :func:`get_core`,
  :func:`query_saturation_data`, :func:`query_core_loss_data`, :func:`estimate_material_loss`,
  :func:`get_physical_properties`, :func:`validate_dataset`, :func:`import_dataset`, :func:`info`.
* Typed snapshots for programmatic use by the Design layer: :func:`material_data`, :func:`core_data`,
  :func:`list_core_data` (:class:`MaterialData`, :class:`CoreData`), :class:`LossEvaluator` /
  :class:`FluxShape` (vectorised Steinmetz / iGSE with temperature interpolation) and
  :func:`physical_property` (thermal and bulk properties with bounds).
"""

from __future__ import annotations

from aipe_magnetics.database.accessors import CoreData, MaterialData, core_data, list_core_data, material_data
from aipe_magnetics.database.loss import FluxShape, LossEvaluation, LossEvaluator
from aipe_magnetics.database.properties import describe_entry, physical_property, property_materials, property_source
from aipe_magnetics.database.query import (
    estimate_material_loss,
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
    "compare_materials": {"id": "aipe.magnetics.compare_materials", "kind": "deterministic",
                          "description": "Compare candidate materials at an operating point (Steinmetz loss, saturation margin)."},
    "search_cores": {"id": "aipe.magnetics.database.search_cores", "kind": "read_only",
                     "description": "Search core geometries by material, shape, size and area product (paginated)."},
    "get_core": {"id": "aipe.magnetics.database.get_core", "kind": "read_only", "description": "Core geometry record with dimensions."},
    "query_saturation_data": {"id": "aipe.magnetics.database.query_saturation_data", "kind": "read_only",
                              "description": "Saturation, remanence, coercivity, permeability points and B-H loop records (decimated)."},
    "query_core_loss_data": {"id": "aipe.magnetics.database.query_core_loss_data", "kind": "read_only",
                             "description": "Core-loss reference points and Steinmetz parameter sets (decimated)."},
    "estimate_material_loss": {"id": "aipe.magnetics.database.estimate_material_loss", "kind": "deterministic",
                               "description": "Steinmetz / iGSE loss density with optional temperature interpolation between sets."},
    "estimate_core_loss": {"id": "aipe.magnetics.estimate_core_loss", "kind": "deterministic",
                           "description": "Full core-loss capability (Steinmetz, iGSE, Bertotti) with verification status."},
    "get_physical_properties": {"id": "aipe.magnetics.database.get_physical_properties", "kind": "read_only",
                                "description": "Thermal and bulk properties (ferrite, copper, winding packs, insulation, air) with bounds."},
    "validate_dataset": {"id": "aipe.magnetics.database.validate_dataset", "kind": "read_only",
                         "description": "Validate record files against the v0.1 / v0.2 schemas."},
    "import_dataset": {"id": "aipe.magnetics.database.import_dataset", "kind": "write_optional",
                       "description": "Import a JSON record file into a validated review candidate (writing requires authorisation)."},
}

__all__ = [
    "CAPABILITIES", "CoreData", "FluxShape", "LossEvaluation", "LossEvaluator", "MaterialData", "core_data", "describe_entry",
    "estimate_material_loss", "get_core", "get_material_properties", "get_physical_properties", "import_dataset", "info",
    "list_core_data", "material_data", "physical_property", "property_materials", "property_source", "query_core_loss_data",
    "query_saturation_data", "search_cores", "search_materials", "validate_dataset",
]
