"""Typed, read-only accessors used by the Design layer (the stable Database -> Design interface).

The Design packages (:mod:`aipe_magnetics.design`, ``geometry``, ``simulation``, ``optimisation``,
``workflows``) obtain material and core data **only** through :func:`material_data`, :func:`core_data`
and :func:`list_core_data` and the loss evaluator in :mod:`aipe_magnetics.database.loss`. Each accessor
returns an immutable snapshot with SI floats, the record id it came from and the source references, so a
design can record exactly which database records (and versions) it used.

Missing data stays missing (``None``); nothing is defaulted here. Callers decide (and record as an
assumption) what to do about a missing value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aipe_magnetics.common.errors import DataNotFoundError, InvalidInputError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import SourceReference
from aipe_magnetics.data.records import CoreFamilyProperties, CoreGeometryProperties, MagneticRecord, MaterialProperties, SteinmetzParameterSet


def _catalog(catalog: Any | None):
    if catalog is not None:
        return catalog
    from aipe_magnetics.data.catalog import Catalog
    return Catalog.default()


def _sources(record: MagneticRecord) -> list[SourceReference]:
    from aipe_magnetics.engineering.materials import evidence_sources
    return evidence_sources(record)


@dataclass(frozen=True)
class MaterialData:
    """Snapshot of one magnetic material record (SI floats; ``None`` = not recorded)."""

    id: str
    name: str
    grade: str | None
    material_class: str | None
    synthetic: bool
    initial_permeability: float | None
    density: float | None
    resistivity: float | None
    curie_temperature: float | None
    steinmetz_sets: tuple[SteinmetzParameterSet, ...]
    saturation_points: tuple[tuple[float, float, float | None], ...]  # (temperature K, Bsat T, field A/m)
    sources: tuple[SourceReference, ...] = field(default_factory=tuple)
    record_schema_version: str | None = None
    _props: Any = field(default=None, repr=False, compare=False)

    def saturation_flux_density(self, temperature: float) -> tuple[float, str]:
        """``(Bsat, status)`` at ``temperature`` (K) by the catalogue rule (exact point or linear interpolation, no extrapolation)."""
        from aipe_magnetics.engineering.materials import saturation_flux_density_at
        q, status, _ = saturation_flux_density_at(self._props, Quantity(value=float(temperature), unit="K"))
        return q.to("T").value, status

    def has_saturation_data(self) -> bool:
        return bool(self.saturation_points)

    def loss_temperatures(self) -> list[float]:
        return sorted(s.temperature.magnitude("K") for s in self.steinmetz_sets if s.temperature is not None)

    def properties(self) -> MaterialProperties:
        return self._props


@dataclass(frozen=True)
class CoreData:
    """Snapshot of one core-geometry record (SI floats; ``None`` = not recorded)."""

    id: str
    name: str
    shape: str | None
    part_number: str | None
    effective_area: float | None
    effective_length: float | None
    effective_volume: float | None
    minimum_area: float | None
    window_area: float | None
    bobbin_window_area: float | None
    mean_turn_length: float | None
    mass: float | None
    dimensions: dict[str, float]
    material_refs: tuple[str, ...]
    air_gap_options: tuple[float, ...]
    sources: tuple[SourceReference, ...] = field(default_factory=tuple)

    @property
    def is_e_type(self) -> bool:
        return (self.shape or "").upper() in {"E", "EE", "EI", "ETD", "EFD", "ER", "EQ", "PQ"}

    def has_e_geometry(self) -> bool:
        """True when the outline dimensions needed for the parametric E-core model are recorded."""
        return all(k in self.dimensions for k in ("length_a", "height_b_half", "depth_c", "window_height_half",
                                                   "window_width_total", "centre_leg_width"))


def material_data(query: str, catalog: Any | None = None) -> MaterialData:
    """Material snapshot by id, name or grade (``N87``, ``material.tdk-n87``)."""
    from aipe_magnetics.engineering.materials import lookup_material
    record, props = lookup_material(query, catalog)
    sat = tuple((p.temperature.magnitude("K") if p.temperature else float("nan"), p.value.to("T").value,
                 p.field.magnitude("A/m") if p.field else None) for p in props.saturation_flux_density)
    return MaterialData(
        id=record.id, name=record.name, grade=props.grade, material_class=props.material_class, synthetic=record.synthetic,
        initial_permeability=props.initial_permeability.value.value if props.initial_permeability else None,
        density=props.density.value.to("kg/m^3").value if props.density else None,
        resistivity=props.resistivity.value.to("Ohm*m").value if props.resistivity else None,
        curie_temperature=props.curie_temperature.value.to("K").value if props.curie_temperature else None,
        steinmetz_sets=tuple(props.steinmetz_parameter_sets), saturation_points=sat, sources=tuple(_sources(record)),
        record_schema_version=record.schema_version, _props=props)


def _shape_of(core: CoreGeometryProperties, catalog: Any) -> str | None:
    if core.family_ref and core.family_ref in catalog:
        family = CoreFamilyProperties.model_validate(catalog.get(core.family_ref).properties)
        return family.shape
    return None


def core_data(core_id: str, catalog: Any | None = None) -> CoreData:
    """Core snapshot by record id (``core.tdk-e65-32-27``)."""
    if not isinstance(core_id, str) or not core_id:
        raise InvalidInputError("core id must be a non-empty string")
    cat = _catalog(catalog)
    record = cat.get(core_id)
    if record.kind != "core_geometry":
        raise DataNotFoundError(f"{core_id} is a {record.kind} record, not a core geometry")
    core = CoreGeometryProperties.model_validate(record.properties)

    def si(q: Quantity | None) -> float | None:
        return None if q is None else q.to_si().value

    return CoreData(
        id=record.id, name=record.name, shape=_shape_of(core, cat), part_number=core.part_number,
        effective_area=si(core.effective_area), effective_length=si(core.effective_length), effective_volume=si(core.effective_volume),
        minimum_area=si(core.minimum_area), window_area=si(core.window_area), bobbin_window_area=si(core.bobbin_window_area),
        mean_turn_length=si(core.mean_turn_length), mass=si(core.mass),
        dimensions={k: q.to_si().value for k, q in (core.dimensions.values.items() if core.dimensions else [])},
        material_refs=tuple(core.material_refs), air_gap_options=tuple(q.to_si().value for q in core.air_gap_options),
        sources=tuple(_sources(record)))


def list_core_data(material: str | None = None, shape: str | None = None, catalog: Any | None = None,
                   include_synthetic: bool = False) -> list[CoreData]:
    """Every catalogue core (optionally offered in ``material`` and of family ``shape``)."""
    cat = _catalog(catalog)
    material_id = None
    if material is not None:
        material_id = material_data(material, cat).id
    out = []
    for record in cat.cores(material_id):
        if record.synthetic and not include_synthetic:
            continue
        data = core_data(record.id, cat)
        if shape is not None and (data.shape or "").upper() != shape.upper():
            continue
        out.append(data)
    return out


__all__ = ["CoreData", "MaterialData", "core_data", "list_core_data", "material_data"]
