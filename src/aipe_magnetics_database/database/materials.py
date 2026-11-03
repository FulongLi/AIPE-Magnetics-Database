"""Material lookup and inspection over catalogue records (data layer).

All information comes from catalogue records (:class:`~aipe_magnetics_database.data.records.MagneticRecord`
of kind ``magnetic_material``). Nothing is guessed: when a parameter is missing the result says so.
The only derived quantity is the saturation flux density *between* two datasheet temperatures,
which is a linear interpolation explicitly labelled ``estimated`` with an assumption record;
extrapolation outside the datasheet temperature points is refused.
"""

from __future__ import annotations

from typing import Any

from aipe_magnetics_database.common.errors import DataNotFoundError, InvalidInputError, OutOfDomainError
from aipe_magnetics_database.common.quantity import Quantity
from aipe_magnetics_database.common.results import EngineeringResult, SourceReference
from aipe_magnetics_database.data.records import ConditionedValue, MagneticRecord, MaterialProperties

CAP_LIST = "aipe.magnetics.list_materials"
CAP_GET = "aipe.magnetics.get_material"
CAP_COMPARE = "aipe.magnetics.compare_materials"


# -------------------------------------------------------------------------------------------------
# catalogue access
# -------------------------------------------------------------------------------------------------
def _catalog(catalog: Any | None):
    if catalog is not None:
        return catalog
    from aipe_magnetics_database.data.catalog import Catalog
    return Catalog.default()


def lookup_material(query: str, catalog: Any | None = None) -> tuple[MagneticRecord, MaterialProperties]:
    """Resolve ``query`` (id, name or grade) to a material record and its typed properties."""
    if not isinstance(query, str) or not query.strip():
        raise InvalidInputError("material query must be a non-empty string")
    record = _catalog(catalog).find_material(query.strip())
    if record.kind != "magnetic_material":
        raise DataNotFoundError(f"{record.id} is a {record.kind} record, not a magnetic material")
    return record, MaterialProperties.model_validate(record.properties)


def evidence_sources(record: MagneticRecord) -> list[SourceReference]:
    out: list[SourceReference] = []
    kind_map = {"source": "datasheet", "measurement": "measurement_record", "analytical": "publication",
                "assumption": "assumption", "simulation": "internal"}
    for item in record.evidence:
        out.append(SourceReference(
            id=item.id, kind=kind_map.get(item.kind, "internal"), reference=item.reference, locator=item.locator,
            url=item.url, version=item.version, license=item.license, sha256=item.sha256,
            note=f"review_status={item.review_status}; {item.description}"))
    return out


# -------------------------------------------------------------------------------------------------
# derived material quantities
# -------------------------------------------------------------------------------------------------
def saturation_flux_density_at(props: MaterialProperties, temperature: Quantity) -> tuple[Quantity, str, list[str]]:
    """Bsat at ``temperature`` from the datasheet points.

    Returns ``(value, status, source_refs)`` where status is ``looked_up`` (exact datasheet point) or
    ``estimated`` (linear interpolation between the two bracketing datasheet temperatures). Points
    measured at different field strengths are not mixed: the set with the most points at a common
    field is used, and that field is reported in the note.
    """
    t = temperature.to("K").value
    points = [p for p in props.saturation_flux_density if p.temperature is not None]
    if not points:
        if props.saturation_flux_density:
            p = props.saturation_flux_density[0]
            return p.value, "looked_up", [p.source_ref] if p.source_ref else []
        raise DataNotFoundError("material has no saturation flux density data")
    # group by field strength
    groups: dict[float | None, list[ConditionedValue]] = {}
    for p in points:
        key = None if p.field is None else round(p.field.value, 6)
        groups.setdefault(key, []).append(p)
    group = max(groups.values(), key=len)
    group = sorted(group, key=lambda p: p.temperature.value)  # type: ignore[union-attr]
    for p in group:
        if abs(p.temperature.value - t) <= 0.5:  # type: ignore[union-attr]
            return p.value, "looked_up", [p.source_ref] if p.source_ref else []
    temps = [p.temperature.value for p in group]  # type: ignore[union-attr]
    if t < temps[0] or t > temps[-1]:
        raise OutOfDomainError(
            f"temperature {t - 273.15:.1f} degC is outside the datasheet Bsat points "
            f"({temps[0] - 273.15:.0f}..{temps[-1] - 273.15:.0f} degC); extrapolation is not performed",
            details={"temperature_K": t, "available_K": temps})
    for lo, hi in zip(group, group[1:]):
        t0, t1 = lo.temperature.value, hi.temperature.value  # type: ignore[union-attr]
        if t0 <= t <= t1:
            w = (t - t0) / (t1 - t0)
            value = lo.value.value + w * (hi.value.value - lo.value.value)
            refs = [r for r in (lo.source_ref, hi.source_ref) if r]
            return Quantity(value=value, unit=lo.value.unit), "estimated", refs
    raise DataNotFoundError("could not bracket temperature in saturation data")  # pragma: no cover


def material_summary(record: MagneticRecord, props: MaterialProperties) -> dict[str, Any]:
    """Compact JSON-serialisable description used in listings and comparisons."""
    def q(x: Quantity | None) -> dict[str, Any] | None:
        return None if x is None else x.model_dump(mode="json", exclude_none=True)

    return {
        "id": record.id,
        "name": record.name,
        "grade": props.grade,
        "manufacturer_ref": props.manufacturer_ref,
        "material_class": props.material_class,
        "synthetic": record.synthetic,
        "license": record.license,
        "initial_permeability": q(props.initial_permeability.value) if props.initial_permeability else None,
        "saturation_flux_density": [
            {"value": q(p.value), "temperature": q(p.temperature), "field": q(p.field), "source_ref": p.source_ref}
            for p in props.saturation_flux_density],
        "curie_temperature": q(props.curie_temperature.value) if props.curie_temperature else None,
        "density": q(props.density.value) if props.density else None,
        "resistivity": q(props.resistivity.value) if props.resistivity else None,
        "recommended_frequency_range": (props.recommended_frequency_range.model_dump(mode="json", exclude_none=True)
                                        if props.recommended_frequency_range else None),
        "steinmetz_parameter_sets": [
            {"id": s.id, "k": q(s.k), "alpha": s.alpha, "beta": s.beta, "temperature": q(s.temperature),
             "frequency_range": s.frequency_range.model_dump(mode="json", exclude_none=True) if s.frequency_range else None,
             "flux_density_range": s.flux_density_range.model_dump(mode="json", exclude_none=True) if s.flux_density_range else None,
             "temperature_range": s.temperature_range.model_dump(mode="json", exclude_none=True) if s.temperature_range else None,
             "basis": s.basis, "source_ref": s.source_ref, "fit_metrics": s.fit_metrics}
            for s in props.steinmetz_parameter_sets],
        "loss_reference_points": len(props.loss_reference_points),
        "evidence": [{"id": e.id, "kind": e.kind, "reference": e.reference, "locator": e.locator,
                      "review_status": e.review_status, "license": e.license} for e in record.evidence],
    }


# -------------------------------------------------------------------------------------------------
# results
# -------------------------------------------------------------------------------------------------
def list_materials(query: str | None = None, material_class: str | None = None, catalog: Any | None = None) -> EngineeringResult:
    for name, value in (("query", query), ("material_class", material_class)):
        if value is not None and not isinstance(value, str):
            raise InvalidInputError(f"{name} must be a string or None, got {type(value).__name__}")
    cat = _catalog(catalog)
    result = EngineeringResult(capability=CAP_LIST, inputs={"query": query, "material_class": material_class})
    items: list[dict[str, Any]] = []
    needle = (query or "").strip().lower()
    for record in cat.materials():
        props = MaterialProperties.model_validate(record.properties)
        if material_class and props.material_class != material_class:
            continue
        haystack = " ".join(filter(None, [record.id, record.name, props.grade or ""])).lower()
        if needle and needle not in haystack:
            continue
        items.append({"id": record.id, "name": record.name, "grade": props.grade, "material_class": props.material_class,
                      "manufacturer_ref": props.manufacturer_ref, "synthetic": record.synthetic,
                      "steinmetz_sets": len(props.steinmetz_parameter_sets),
                      "saturation_points": len(props.saturation_flux_density), "license": record.license})
    result.data["materials"] = items
    result.data["count"] = len(items)
    if not items:
        result.warn("no_match", "No catalogue material matches the query; import your own records or check the spelling.", "info")
    result.limit("catalogue_scope", "Only materials present in the loaded catalogue(s) are listed; absence is not evidence of non-existence.")
    result.verification.state = "verified"
    result.verification.level = "L1"
    result.verification.statement = "Listing is a deterministic read of validated catalogue records (software correctness, L1)."
    return result


def get_material(query: str, catalog: Any | None = None) -> EngineeringResult:
    record, props = lookup_material(query, catalog)
    result = EngineeringResult(capability=CAP_GET, inputs={"material": query})
    result.data["material"] = material_summary(record, props)
    result.data["record"] = record.to_json_dict()
    if props.initial_permeability is not None:
        result.add_value("initial_permeability", "Initial permeability", props.initial_permeability.value, status="looked_up")
    for index, p in enumerate(props.saturation_flux_density):
        label = "Saturation flux density"
        if p.temperature is not None:
            label += f" at {p.temperature.value - 273.15:.0f} degC"
        if p.field is not None:
            label += f", H = {p.field.value:g} A/m"
        result.add_value(f"saturation_flux_density_{index}", label, p.value, status="looked_up")
    if props.curie_temperature is not None:
        # the record note states e.g. that the datasheet gives only a lower bound (TC > 210 degC); it travels with the value
        result.add_value("curie_temperature", "Curie temperature", props.curie_temperature.value, status="looked_up",
                         note=props.curie_temperature.note)
    if props.density is not None:
        result.add_value("density", "Density", props.density.value, status="looked_up")
    if props.resistivity is not None:
        result.add_value("resistivity", "Resistivity", props.resistivity.value, status="looked_up")
    for source in evidence_sources(record):
        result.cite(source)
    if record.synthetic:
        result.warn("synthetic_material", "This material record is synthetic (illustrative); it does not describe a real product.", "warning")
    unreviewed = [e.id for e in record.evidence if e.review_status == "unreviewed"]
    if unreviewed:
        result.limit("unreviewed_evidence", f"Evidence {', '.join(unreviewed)} is unreviewed; values are datasheet transcriptions awaiting owner review.")
    if not props.steinmetz_parameter_sets:
        result.limit("no_loss_parameters", "No Steinmetz parameter set is recorded; core-loss estimation is not possible for this material.")
    result.verification.state = "verified"
    result.verification.level = "L1"
    result.verification.statement = "Values are transcribed catalogue data validated against the v0.2 schema (L1); physical correctness rests on the cited sources."
    return result

