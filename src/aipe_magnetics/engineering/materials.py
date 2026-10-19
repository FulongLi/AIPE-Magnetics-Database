"""Material lookup, inspection and comparison (engineering layer).

All information comes from catalogue records (:class:`~aipe_magnetics.data.records.MagneticRecord`
of kind ``magnetic_material``). Nothing is guessed: when a parameter is missing the result says so.
The only derived quantity is the saturation flux density *between* two datasheet temperatures,
which is a linear interpolation explicitly labelled ``estimated`` with an assumption record;
extrapolation outside the datasheet temperature points is refused.
"""

from __future__ import annotations

from typing import Any

from aipe_magnetics.common.errors import DataNotFoundError, InvalidInputError, OutOfDomainError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import EngineeringResult, SourceReference
from aipe_magnetics.data.records import ConditionedValue, MagneticRecord, MaterialProperties

CAP_LIST = "aipe.magnetics.list_materials"
CAP_GET = "aipe.magnetics.get_material"
CAP_COMPARE = "aipe.magnetics.compare_materials"


# -------------------------------------------------------------------------------------------------
# catalogue access
# -------------------------------------------------------------------------------------------------
def _catalog(catalog: Any | None):
    if catalog is not None:
        return catalog
    from aipe_magnetics.data.catalog import Catalog
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


def compare_materials(materials: list[str], frequency: Quantity | None = None, flux_density_peak: Quantity | None = None,
                      temperature: Quantity | None = None, catalog: Any | None = None, registry: Any | None = None) -> EngineeringResult:
    """Compare materials; when an operating point is given, add Steinmetz loss density and saturation margin.

    The verification block is derived from the evidence registry for the Steinmetz model under the requested
    conditions (per material in ``data.comparison[].loss_verification``); it is never asserted.
    """
    if not isinstance(materials, (list, tuple)) or not materials or not all(isinstance(m, str) and m.strip() for m in materials):
        raise InvalidInputError("compare_materials needs a non-empty list of material ids, names or grades (strings)")
    if registry is None:
        from aipe_magnetics.verification.suite import load_bundled_registry
        registry = load_bundled_registry()
    result = EngineeringResult(capability=CAP_COMPARE, inputs={
        "materials": list(materials),
        "frequency": None if frequency is None else frequency.model_dump(mode="json", exclude_none=True),
        "flux_density_peak": None if flux_density_peak is None else flux_density_peak.model_dump(mode="json", exclude_none=True),
        "temperature": None if temperature is None else temperature.model_dump(mode="json", exclude_none=True)})
    rows: list[dict[str, Any]] = []
    for query in materials:
        record, props = lookup_material(query, catalog)
        row: dict[str, Any] = {"query": query, "id": record.id, "name": record.name, "grade": props.grade,
                               "material_class": props.material_class, "synthetic": record.synthetic,
                               "initial_permeability": props.initial_permeability.value.value if props.initial_permeability else None,
                               "curie_temperature_K": props.curie_temperature.value.value if props.curie_temperature else None,
                               "curie_temperature_note": props.curie_temperature.note if props.curie_temperature else None,
                               "loss_density_W_per_m3": None, "loss_model": None, "parameter_set": None,
                               "saturation_flux_density_T": None, "saturation_margin": None, "notes": []}
        for source in evidence_sources(record):
            result.cite(source)
        if temperature is not None:
            try:
                bsat, status, _refs = saturation_flux_density_at(props, temperature)
                row["saturation_flux_density_T"] = bsat.value
                row["saturation_status"] = status
                if flux_density_peak is not None:
                    row["saturation_margin"] = 1.0 - flux_density_peak.to("T").value / bsat.value
            except (DataNotFoundError, OutOfDomainError) as error:
                row["notes"].append(f"saturation: {error.message}")
        if frequency is not None and flux_density_peak is not None:
            try:
                from aipe_magnetics.models.empirical.steinmetz import SteinmetzModel, select_parameter_set
                t_k = None if temperature is None else temperature.to("K").value
                pset = select_parameter_set(props.steinmetz_parameter_sets, t_k)
                model = SteinmetzModel.from_parameter_set(pset, material_id=record.id, sources=evidence_sources(record))
                out = model.loss_density(frequency=frequency.to("Hz").value, flux_density_peak=flux_density_peak.to("T").value,
                                         temperature=t_k, policy="warn")
                row["loss_density_W_per_m3"] = out.values["loss_density"].value
                row["loss_model"] = f"{out.model.id}@{out.model.version}"
                row["parameter_set"] = pset.id
                row["loss_verification"] = registry.status_for(out.model.id, frequency=frequency.to("Hz").value, temperature=t_k,
                                                               flux_density_peak=flux_density_peak.to("T").value, material=record.id,
                                                               excitation="sinusoidal").model_dump(mode="json")
                if not out.domain.inside:
                    row["notes"].append("loss: outside parameter-set validity: " + "; ".join(out.domain.violations))
            except (DataNotFoundError, OutOfDomainError, InvalidInputError) as error:
                row["notes"].append(f"loss: {error.message}")
        rows.append(row)
    ranked = sorted(rows, key=lambda r: (r["loss_density_W_per_m3"] is None, r["loss_density_W_per_m3"] or 0.0))
    result.data["comparison"] = ranked
    result.assume("sinusoidal_comparison", "Loss densities in the comparison use the classical Steinmetz equation "
                  "(sinusoidal flux) with the parameter set covering the requested temperature.", "model_assumption",
                  ["comparison"])
    result.limit("comparison_scope", "The comparison ranks datasheet-derived Steinmetz losses and saturation margin only; "
                 "it is not a thermal, cost or availability assessment.")
    missing = [r["id"] for r in rows if r["loss_density_W_per_m3"] is None]
    if missing and frequency is not None and flux_density_peak is not None:
        result.warn("loss_not_available", f"No loss estimate for {', '.join(missing)} at this operating point (see notes).")
    evaluated = [r for r in rows if r.get("loss_verification")]
    if evaluated and frequency is not None and flux_density_peak is not None:
        model_id = evaluated[0]["loss_model"].split("@")[0]
        status = registry.status_for(model_id, frequency=frequency.to("Hz").value,
                                     temperature=None if temperature is None else temperature.to("K").value,
                                     flux_density_peak=flux_density_peak.to("T").value, excitation="sinusoidal")
        result.verification = status
        result.verification.statement = (f"Steinmetz model (evidence registry): {status.statement} Per-material status in "
                                         "data.comparison[].loss_verification; the datasheet parameters themselves are transcriptions "
                                         "and remain unreviewed.")
    else:
        result.verification.state = "verified"
        result.verification.level = "L1"
        result.verification.statement = ("No loss model was evaluated (no complete operating point or no covering parameter set); "
                                         "the comparison is a deterministic read of validated catalogue records (L1).")
    return result
