"""Token-efficient database queries: filtering, pagination, result limits and configurable detail.

Every function returns an :class:`~aipe_magnetics.common.results.EngineeringResult` whose ``data`` holds a
compact, JSON-serialisable payload. Lists are paginated (``limit`` / ``offset`` with ``total`` and
``next_offset``); ``detail`` selects ``"summary"`` (identifiers and headline values), ``"standard"``
(adds validity ranges, parameter sets and sources) or ``"full"`` (adds the complete normalised record).
Large numerical arrays are never inlined: series longer than ``max_points`` are decimated for the reply and,
when the caller supplies an ``export_path``, written to a CSV file whose path is returned instead.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Literal

from aipe_magnetics.common.errors import AIPEMagneticsError, DataNotFoundError, InvalidInputError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import EngineeringResult
from aipe_magnetics.data.records import CoreGeometryProperties, MaterialProperties

Detail = Literal["summary", "standard", "full"]
DETAILS = ("summary", "standard", "full")
MAX_LIMIT = 200

CAP_SEARCH_MATERIALS = "aipe.magnetics.database.search_materials"
CAP_MATERIAL_PROPERTIES = "aipe.magnetics.database.get_material_properties"
CAP_SEARCH_CORES = "aipe.magnetics.database.search_cores"
CAP_GET_CORE = "aipe.magnetics.database.get_core"
CAP_SATURATION = "aipe.magnetics.database.query_saturation_data"
CAP_LOSS_DATA = "aipe.magnetics.database.query_core_loss_data"
CAP_MATERIAL_LOSS = "aipe.magnetics.database.estimate_material_loss"
CAP_PROPERTIES = "aipe.magnetics.database.get_physical_properties"
CAP_VALIDATE = "aipe.magnetics.database.validate_dataset"
CAP_IMPORT = "aipe.magnetics.database.import_dataset"
CAP_INFO = "aipe.magnetics.database.info"

HAS_FILTERS = ("saturation_flux_density", "steinmetz", "initial_permeability", "density", "curie_temperature", "loss_reference_points")


def _catalog(catalog: Any | None):
    if catalog is not None:
        return catalog
    from aipe_magnetics.data.catalog import Catalog
    return Catalog.default()


def _check_detail(detail: str) -> str:
    if detail not in DETAILS:
        raise InvalidInputError(f"detail must be one of {DETAILS}, got {detail!r}")
    return detail


def _page(items: list[Any], limit: int, offset: int) -> tuple[list[Any], dict[str, Any]]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= MAX_LIMIT):
        raise InvalidInputError(f"limit must be an integer in 1..{MAX_LIMIT}, got {limit!r}")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise InvalidInputError(f"offset must be a non-negative integer, got {offset!r}")
    total = len(items)
    page = items[offset: offset + limit]
    nxt = offset + limit if offset + limit < total else None
    return page, {"total": total, "offset": offset, "limit": limit, "returned": len(page), "next_offset": nxt}


def _l1(result: EngineeringResult, statement: str) -> EngineeringResult:
    result.verification.level = "L1"
    result.verification.state = "verified"
    result.verification.statement = statement
    return result


def _q(value: Any, unit: str, name: str) -> float | None:
    if value is None:
        return None
    q = Quantity.parse(value, name=name)
    from aipe_magnetics.common.units import units
    if not units.compatible(q.unit, unit):
        raise InvalidInputError(f"{name} must have the dimension of {unit!r}, got {q.unit!r}")
    return q.magnitude(unit)


def _rng(r: Any) -> dict[str, Any] | None:
    return None if r is None else r.model_dump(mode="json", exclude_none=True)


def _provenance(record: Any) -> dict[str, Any]:
    return {"record_id": record.id, "schema_version": record.schema_version, "license": record.license,
            "evidence": [e.id for e in record.evidence], "review_status": sorted({e.review_status for e in record.evidence}),
            "synthetic": record.synthetic}


# -------------------------------------------------------------------------------------------------
# materials
# -------------------------------------------------------------------------------------------------
def _operating_ranges(props: MaterialProperties) -> dict[str, Any]:
    temps = sorted(s.temperature.magnitude("K") for s in props.steinmetz_parameter_sets if s.temperature is not None)
    fmins = [s.frequency_range.min for s in props.steinmetz_parameter_sets if s.frequency_range and s.frequency_range.min is not None]
    fmaxs = [s.frequency_range.max for s in props.steinmetz_parameter_sets if s.frequency_range and s.frequency_range.max is not None]
    bmaxs = [s.flux_density_range.max for s in props.steinmetz_parameter_sets if s.flux_density_range and s.flux_density_range.max is not None]
    return {
        "loss_model_temperatures_K": temps,
        "loss_model_frequency_Hz": {"min": min(fmins), "max": max(fmaxs)} if fmins and fmaxs else None,
        "loss_model_flux_density_max_T": max(bmaxs) if bmaxs else None,
        "recommended_frequency_Hz": _rng(props.recommended_frequency_range),
        "saturation_data_temperatures_K": sorted(p.temperature.magnitude("K") for p in props.saturation_flux_density if p.temperature),
        "curie_temperature_K": props.curie_temperature.value.magnitude("K") if props.curie_temperature else None,
    }


def _material_item(record: Any, props: MaterialProperties, detail: str) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": record.id, "name": record.name, "grade": props.grade, "material_class": props.material_class,
        "manufacturer_ref": props.manufacturer_ref, "synthetic": record.synthetic,
        "initial_permeability": props.initial_permeability.value.value if props.initial_permeability else None,
        "has": {"saturation_flux_density": bool(props.saturation_flux_density), "steinmetz": len(props.steinmetz_parameter_sets),
                "initial_permeability": props.initial_permeability is not None, "density": props.density is not None,
                "curie_temperature": props.curie_temperature is not None, "loss_reference_points": len(props.loss_reference_points)},
    }
    if detail in ("standard", "full"):
        from aipe_magnetics.engineering.materials import material_summary
        item["valid_operating_ranges"] = _operating_ranges(props)
        summary = material_summary(record, props)
        item["saturation_flux_density"] = summary["saturation_flux_density"]
        item["steinmetz_parameter_sets"] = summary["steinmetz_parameter_sets"]
        item["density"] = summary["density"]
        item["resistivity"] = summary["resistivity"]
        item["curie_temperature"] = summary["curie_temperature"]
        item["provenance"] = _provenance(record)
        item["evidence"] = summary["evidence"]
    if detail == "full":
        item["record"] = record.to_json_dict()
    return item


def search_materials(query: str | None = None, *, material_class: str | None = None, manufacturer: str | None = None,
                     has: list[str] | None = None, frequency: Any = None, temperature: Any = None, include_synthetic: bool = False,
                     limit: int = 20, offset: int = 0, detail: Detail = "summary", catalog: Any | None = None) -> EngineeringResult:
    """Search catalogue materials by text, class, manufacturer, data availability and loss-model coverage."""
    inputs = {"query": query, "material_class": material_class, "manufacturer": manufacturer, "has": has, "limit": limit,
              "offset": offset, "detail": detail, "include_synthetic": include_synthetic}
    result = EngineeringResult(capability=CAP_SEARCH_MATERIALS, inputs=inputs)
    try:
        _check_detail(detail)
        unknown = sorted(set(has or []) - set(HAS_FILTERS))
        if unknown:
            raise InvalidInputError(f"unknown 'has' filter(s) {unknown}; allowed: {list(HAS_FILTERS)}")
        f = _q(frequency, "Hz", "frequency")
        t = _q(temperature, "K", "temperature")
        cat = _catalog(catalog)
        needle = (query or "").strip().casefold()
        rows: list[tuple[Any, MaterialProperties]] = []
        for record in cat.materials():
            props = MaterialProperties.model_validate(record.properties)
            if record.synthetic and not include_synthetic:
                continue
            if material_class and props.material_class != material_class:
                continue
            if manufacturer and manufacturer.casefold() not in (props.manufacturer_ref or "").casefold():
                continue
            if needle and needle not in " ".join(filter(None, [record.id, record.name, props.grade or "", props.description or ""])).casefold():
                continue
            flags = _material_item(record, props, "summary")["has"]
            if any(not flags[h] for h in (has or [])):
                continue
            if f is not None or t is not None:
                from aipe_magnetics.database.loss import LossEvaluator
                if not props.steinmetz_parameter_sets:
                    continue
                try:
                    LossEvaluator(record.id, props.steinmetz_parameter_sets).select(f if f is not None else 1e5, t if t is not None else 373.15,
                                                                                     policy="reject")
                except AIPEMagneticsError:
                    continue
            rows.append((record, props))
        page, meta = _page(rows, limit, offset)
        result.data["items"] = [_material_item(r, p, detail) for r, p in page]
        result.data["page"] = meta
        if not meta["total"]:
            result.warn("no_match", "No catalogue material matches the filters; relax the filters or import records.", "info")
        if f is not None or t is not None:
            result.assume("coverage_filter", "Frequency/temperature filters keep materials whose Steinmetz parameter sets cover the "
                          "conditions (fit temperature within 5 K or bracketed for interpolation, frequency inside the fitted range).",
                          "model_assumption")
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(CAP_SEARCH_MATERIALS, error, inputs=inputs)
    result.limit("catalogue_scope", "Only materials present in the loaded catalogue(s) are searched; absence is not evidence of non-existence.")
    return _l1(result, "Deterministic read of schema-validated catalogue records (L1).")


def get_material_properties(material: str, *, detail: Detail = "standard", fields: list[str] | None = None,
                            catalog: Any | None = None) -> EngineeringResult:
    """Material properties, valid operating ranges, parameter sets, thermal properties and provenance."""
    inputs = {"material": material, "detail": detail, "fields": fields}
    try:
        _check_detail(detail)
        from aipe_magnetics.engineering.materials import evidence_sources, lookup_material
        record, props = lookup_material(material, catalog)
        result = EngineeringResult(capability=CAP_MATERIAL_PROPERTIES, inputs=inputs)
        item = _material_item(record, props, "standard" if detail == "summary" else detail)
        if props.material_class:
            from aipe_magnetics.database.properties import describe_entry
            try:
                item["thermal_properties"] = describe_entry(props.material_class)["properties"]
            except DataNotFoundError:
                item["thermal_properties"] = None
        if fields:
            missing = [k for k in fields if k not in item]
            if missing:
                raise InvalidInputError(f"unknown field(s) {missing}; available: {sorted(item)}")
            item = {k: item[k] for k in ["id", *fields] if k in item}
        result.data["material"] = item
        for source in evidence_sources(record):
            result.cite(source)
        if record.synthetic:
            result.warn("synthetic_material", "This material record is synthetic (illustrative).")
        if not props.saturation_flux_density:
            result.limit("no_saturation_data", "No saturation flux density is recorded: design capabilities that need Bsat reject this material.")
        result.limit("thermal_properties_generic", "Thermal properties are generic values for the material class with stated bounds, not grade data.")
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(CAP_MATERIAL_PROPERTIES, error, inputs=inputs)
    return _l1(result, "Transcribed catalogue data validated against the v0.2 schema (L1); physical correctness rests on the cited sources.")


# -------------------------------------------------------------------------------------------------
# cores
# -------------------------------------------------------------------------------------------------
def _core_item(record: Any, core: CoreGeometryProperties, detail: str, catalog: Any) -> dict[str, Any]:
    from aipe_magnetics.database.accessors import core_data
    data = core_data(record.id, catalog)

    def mm(v: float | None, p: int = 1) -> float | None:
        return None if v is None else round(v * 10 ** (3 * p), 6)

    item: dict[str, Any] = {"id": record.id, "name": record.name, "shape": data.shape, "part_number": data.part_number,
                            "effective_area_mm2": mm(data.effective_area, 2), "effective_length_mm": mm(data.effective_length),
                            "effective_volume_mm3": mm(data.effective_volume, 3), "mass_g": None if data.mass is None else data.mass * 1e3,
                            "material_refs": list(data.material_refs), "synthetic": record.synthetic,
                            "parametric_geometry": data.has_e_geometry()}
    if detail in ("standard", "full"):
        item.update({"window_area_mm2": mm(data.window_area, 2), "bobbin_window_area_mm2": mm(data.bobbin_window_area, 2),
                     "mean_turn_length_mm": mm(data.mean_turn_length), "minimum_area_mm2": mm(data.minimum_area, 2),
                     "dimensions_mm": {k: round(v * 1e3, 6) for k, v in data.dimensions.items()},
                     "air_gap_options_mm": [round(g * 1e3, 6) for g in data.air_gap_options],
                     "area_product_mm4": (mm(data.effective_area, 2) * mm(data.bobbin_window_area or data.window_area, 2)
                                          if data.effective_area and (data.bobbin_window_area or data.window_area) else None),
                     "provenance": _provenance(record)})
    if detail == "full":
        item["record"] = record.to_json_dict()
    return item


def search_cores(query: str | None = None, *, material: str | None = None, shape: str | None = None,
                 min_effective_area: Any = None, max_effective_area: Any = None, min_window_area: Any = None,
                 min_area_product: Any = None, parametric_only: bool = False, include_synthetic: bool = False,
                 sort: Literal["id", "effective_volume", "area_product"] = "effective_volume",
                 limit: int = 20, offset: int = 0, detail: Detail = "summary", catalog: Any | None = None) -> EngineeringResult:
    """Search core geometries by text, material offering, shape, size and area product."""
    inputs = {"query": query, "material": material, "shape": shape, "limit": limit, "offset": offset, "detail": detail, "sort": sort}
    result = EngineeringResult(capability=CAP_SEARCH_CORES, inputs=inputs)
    try:
        _check_detail(detail)
        if sort not in ("id", "effective_volume", "area_product"):
            raise InvalidInputError(f"sort must be id, effective_volume or area_product, got {sort!r}")
        from aipe_magnetics.database.accessors import core_data, material_data
        cat = _catalog(catalog)
        material_id = material_data(material, cat).id if material else None
        amin, amax = _q(min_effective_area, "m^2", "min_effective_area"), _q(max_effective_area, "m^2", "max_effective_area")
        wmin, apmin = _q(min_window_area, "m^2", "min_window_area"), _q(min_area_product, "m^4", "min_area_product")
        needle = (query or "").strip().casefold()
        rows = []
        for record in cat.cores(material_id):
            if record.synthetic and not include_synthetic:
                continue
            d = core_data(record.id, cat)
            if needle and needle not in f"{record.id} {record.name} {d.part_number or ''}".casefold():
                continue
            if shape and (d.shape or "").upper() != shape.upper():
                continue
            if amin is not None and (d.effective_area is None or d.effective_area < amin):
                continue
            if amax is not None and (d.effective_area is None or d.effective_area > amax):
                continue
            window = d.bobbin_window_area or d.window_area
            if wmin is not None and (window is None or window < wmin):
                continue
            ap = d.effective_area * window if d.effective_area and window else None
            if apmin is not None and (ap is None or ap < apmin):
                continue
            if parametric_only and not d.has_e_geometry():
                continue
            rows.append((record, d, ap))
        key = {"id": lambda r: r[0].id, "effective_volume": lambda r: (r[1].effective_volume is None, r[1].effective_volume or 0.0),
               "area_product": lambda r: (r[2] is None, r[2] or 0.0)}[sort]
        rows.sort(key=key)
        page, meta = _page(rows, limit, offset)
        result.data["items"] = [_core_item(r, CoreGeometryProperties.model_validate(r.properties), detail, cat) for r, _, _ in page]
        result.data["page"] = meta
        if not meta["total"]:
            result.warn("no_match", "No catalogue core matches the filters.", "info")
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(CAP_SEARCH_CORES, error, inputs=inputs)
    result.limit("catalogue_scope", "Only cores present in the loaded catalogue(s) are searched.")
    return _l1(result, "Deterministic read of schema-validated catalogue records (L1).")


def get_core(core: str, *, detail: Detail = "standard", catalog: Any | None = None) -> EngineeringResult:
    inputs = {"core": core, "detail": detail}
    try:
        _check_detail(detail)
        cat = _catalog(catalog)
        record = cat.get(core)
        if record.kind != "core_geometry":
            raise DataNotFoundError(f"{core} is a {record.kind} record, not a core geometry")
        result = EngineeringResult(capability=CAP_GET_CORE, inputs=inputs)
        result.data["core"] = _core_item(record, CoreGeometryProperties.model_validate(record.properties), detail, cat)
        from aipe_magnetics.engineering.materials import evidence_sources
        for s in evidence_sources(record):
            result.cite(s)
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(CAP_GET_CORE, error, inputs=inputs)
    return _l1(result, "Transcribed datasheet values validated against the v0.2 schema (L1).")


# -------------------------------------------------------------------------------------------------
# measured / datasheet data series
# -------------------------------------------------------------------------------------------------
def _decimate(rows: list[dict[str, Any]], max_points: int) -> tuple[list[dict[str, Any]], bool]:
    if len(rows) <= max_points:
        return rows, False
    step = math.ceil(len(rows) / max_points)
    return rows[::step], True


def _export_csv(rows: list[dict[str, Any]], export_path: str | Path | None) -> str | None:
    if export_path is None or not rows:
        return None
    path = Path(export_path)
    if path.exists():
        raise InvalidInputError(f"export_path {path} exists; choose a new file (never overwritten)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def query_saturation_data(material: str, *, max_points: int = 50, export_path: str | Path | None = None,
                          catalog: Any | None = None) -> EngineeringResult:
    """Datasheet saturation / remanence / coercivity points, amplitude-permeability points and B-H loop records of a material."""
    inputs = {"material": material, "max_points": max_points}
    try:
        from aipe_magnetics.engineering.materials import evidence_sources, lookup_material
        record, props = lookup_material(material, catalog)
        cat = _catalog(catalog)

        def rows_of(items: list[Any], name: str, unit: str) -> list[dict[str, Any]]:
            return [{"quantity": name, "value": p.value.magnitude(unit), "unit": unit,
                     "temperature_K": p.temperature.magnitude("K") if p.temperature else None,
                     "field_A_per_m": p.field.magnitude("A/m") if p.field else None,
                     "frequency_Hz": p.frequency.magnitude("Hz") if p.frequency else None, "source_ref": p.source_ref} for p in items]

        rows = (rows_of(props.saturation_flux_density, "saturation_flux_density", "T") + rows_of(props.remanence, "remanence", "T")
                + rows_of(props.coercivity, "coercivity", "A/m") + rows_of(props.amplitude_permeability, "amplitude_permeability", "1"))
        loops = [{"id": r.id, "name": r.name, **{k: v for k, v in r.properties.items() if k in ("frequency", "temperature", "loop_type", "dataset_ref")}}
                 for r in cat.records("bh_loop") if r.properties.get("material_ref") == record.id]
        shown, decimated = _decimate(rows, max_points)
        result = EngineeringResult(capability=CAP_SATURATION, inputs=inputs)
        result.data.update({"material": record.id, "points": shown, "point_count": len(rows), "decimated": decimated,
                            "bh_loop_records": loops, "exported_to": _export_csv(rows, export_path)})
        if not loops:
            result.limit("no_bh_loops", "No measured B-H loop record of this material is in the catalogue; only datasheet points are available. "
                         "Import loop data with the HDF5 waveform importer to query loops.")
        for s in evidence_sources(record):
            result.cite(s)
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(CAP_SATURATION, error, inputs=inputs)
    return _l1(result, "Datasheet transcriptions validated against the v0.2 schema (L1).")


def query_core_loss_data(material: str, *, temperature: Any = None, max_points: int = 50, export_path: str | Path | None = None,
                         catalog: Any | None = None) -> EngineeringResult:
    """Loss reference points (datasheet / measurement) and Steinmetz parameter sets of a material, optionally near a temperature."""
    inputs = {"material": material, "temperature": temperature, "max_points": max_points}
    try:
        from aipe_magnetics.engineering.materials import evidence_sources, lookup_material
        record, props = lookup_material(material, catalog)
        t = _q(temperature, "K", "temperature")
        points = []
        for p in props.loss_reference_points + props.loss_density_vs_temperature:
            tk = p.temperature.magnitude("K") if p.temperature else None
            if t is not None and tk is not None and abs(tk - t) > 5.0:
                continue
            points.append({"frequency_Hz": p.frequency.magnitude("Hz"), "flux_density_peak_T": p.flux_density_peak.magnitude("T"),
                           "temperature_K": tk, "loss_density_W_per_m3": p.loss_density.magnitude("W/m^3"), "excitation": p.excitation,
                           "basis": p.basis, "source_ref": p.source_ref})
        sets = [{"id": s.id, "k_W_per_m3": s.k.magnitude("W/m^3"), "alpha": s.alpha, "beta": s.beta,
                 "temperature_K": s.temperature.magnitude("K") if s.temperature else None, "frequency_range": _rng(s.frequency_range),
                 "flux_density_range": _rng(s.flux_density_range), "basis": s.basis, "fit_metrics": s.fit_metrics, "source_ref": s.source_ref}
                for s in props.steinmetz_parameter_sets
                if t is None or s.temperature is None or abs(s.temperature.magnitude("K") - t) <= 25.0]
        shown, decimated = _decimate(points, max_points)
        result = EngineeringResult(capability=CAP_LOSS_DATA, inputs=inputs)
        result.data.update({"material": record.id, "reference_points": shown, "point_count": len(points), "decimated": decimated,
                            "parameter_sets": sets, "exported_to": _export_csv(points, export_path)})
        for s in evidence_sources(record):
            result.cite(s)
        result.limit("bundled_scope", "Bundled data are datasheet points and fitted parameter sets; raw MagNet waveforms are available through "
                     "the source adapters (downloaded on demand, never bundled).")
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(CAP_LOSS_DATA, error, inputs=inputs)
    return _l1(result, "Datasheet / fitted parameters validated against the v0.2 schema (L1).")


def estimate_material_loss(material: str, frequency: Any, flux_density_peak: Any, temperature: Any, *, waveform: str = "sinusoidal",
                           duty: float = 0.5, interpolate_temperature: bool = True, policy: Literal["reject", "warn"] = "reject",
                           volume: Any = None, catalog: Any | None = None) -> EngineeringResult:
    """Loss density (and loss for a given volume) with optional temperature interpolation between parameter sets.

    Without interpolation this is numerically identical to :func:`aipe_magnetics.api.estimate_core_loss`; the
    interpolated path is reported as such (sets, weights, warnings).
    """
    inputs = {"material": material, "frequency": frequency, "flux_density_peak": flux_density_peak, "temperature": temperature,
              "waveform": waveform, "duty": duty, "interpolate_temperature": interpolate_temperature, "policy": policy}
    try:
        from aipe_magnetics.database.loss import FluxShape, LossEvaluator
        from aipe_magnetics.engineering.materials import evidence_sources, lookup_material
        record, props = lookup_material(material, catalog)
        f, b, t = _q(frequency, "Hz", "frequency"), _q(flux_density_peak, "T", "flux_density_peak"), _q(temperature, "K", "temperature")
        if f is None or b is None or t is None:
            raise InvalidInputError("frequency, flux_density_peak and temperature are required")
        shape = {"sinusoidal": lambda: FluxShape.sinusoid(), "triangular": lambda: FluxShape.triangular(duty),
                 "trapezoidal": lambda: FluxShape.trapezoidal(duty)}.get(waveform)
        if shape is None:
            raise InvalidInputError("waveform must be sinusoidal, triangular or trapezoidal (duty = rise fraction)")
        ev = LossEvaluator(record.id, props.steinmetz_parameter_sets).loss_density(f, b, t, shape(), interpolate_temperature=interpolate_temperature,
                                                                                  policy=policy)
        result = EngineeringResult(capability=CAP_MATERIAL_LOSS, inputs=inputs)
        result.add_value("loss_density", "Core loss density", Quantity(value=float(ev.loss_density), unit="W/m^3"), status="estimated")
        vol = _q(volume, "m^3", "volume")
        if vol is not None:
            result.add_value("core_loss", "Core loss for the given volume", Quantity(value=float(ev.loss_density) * vol, unit="W"), status="estimated")
        result.data.update({"model": ev.model, "parameter_sets": ev.sets, "weights": ev.weights, "interpolated": ev.interpolated})
        for w in ev.warnings:
            result.warn("loss_model_domain", w)
        if ev.interpolated:
            result.assume("temperature_interpolation", f"Loss density interpolated linearly in temperature between {ev.sets[0]} and {ev.sets[1]} "
                          f"(weights {ev.weights[0]:.2f}/{ev.weights[1]:.2f}).", "model_assumption", ["loss_density"])
        for s in evidence_sources(record):
            result.cite(s)
        result.limit("dc_bias_ignored", "Steinmetz/iGSE ignore DC bias and minor loops.")
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(CAP_MATERIAL_LOSS, error, inputs=inputs)
    from aipe_magnetics.verification.suite import load_bundled_registry
    model_id = "aipe.magnetics.model.steinmetz" if waveform == "sinusoidal" else "aipe.magnetics.model.igse"
    result.verification = load_bundled_registry().status_for(model_id, frequency=f, temperature=t, flux_density_peak=b, material=record.id,
                                                             excitation=waveform)
    return result


# -------------------------------------------------------------------------------------------------
# physical properties, datasets, info
# -------------------------------------------------------------------------------------------------
def get_physical_properties(material: str | None = None) -> EngineeringResult:
    """Thermal / bulk properties from the property library (``None`` lists the entries)."""
    from aipe_magnetics.database.properties import describe_entry, property_materials
    inputs = {"material": material}
    try:
        result = EngineeringResult(capability=CAP_PROPERTIES, inputs=inputs)
        if material is None:
            result.data["entries"] = [{"id": m, "name": describe_entry(m)["name"], "role": describe_entry(m)["role"]} for m in property_materials()]
        else:
            entry = describe_entry(material)
            result.data["entry"] = entry
            for prop in entry["properties"].values():
                from aipe_magnetics.common.results import SourceReference
                result.cite(SourceReference.model_validate(prop["source"]))
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(CAP_PROPERTIES, error, inputs=inputs)
    result.limit("generic_values", "Library values are generic class values with stated bounds; use measured data for qualification.")
    return _l1(result, "Static library values with sources (L1).")


def validate_dataset(path: str | Path) -> EngineeringResult:
    """Validate a record file (JSON, v0.1 or v0.2) or a directory of records against the schemas and catalogue rules."""
    inputs = {"path": str(path)}
    try:
        from aipe_magnetics.common.jsonio import load
        from aipe_magnetics.data import schema as sc
        p = Path(path)
        if not p.exists():
            raise DataNotFoundError(f"{p} does not exist")
        files = sorted(p.rglob("*.json")) if p.is_dir() else [p]
        report = []
        for file in files:
            errors = sc.validate_record(load(file))
            report.append({"file": str(file), "valid": not errors, "errors": errors[:20], "error_count": len(errors)})
        result = EngineeringResult(capability=CAP_VALIDATE, inputs=inputs)
        result.data["files"] = report
        result.data["valid"] = all(r["valid"] for r in report)
        if not result.data["valid"]:
            result.status = "rejected"
            from aipe_magnetics.common.results import ErrorRecord
            result.errors.append(ErrorRecord(code="schema_violation", message=f"{sum(not r['valid'] for r in report)} of {len(report)} file(s) failed validation"))
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(CAP_VALIDATE, error, inputs=inputs)
    return _l1(result, "JSON-Schema validation (v0.1 / v0.2 record schemas) (L1).")


def import_dataset(source: str | Path, destination: str | Path, *, migrate_v01: bool = False) -> EngineeringResult:
    """Import (validate, normalise, checksum) a JSON record file into a review candidate file; never overwrites."""
    inputs = {"source": str(source), "destination": str(destination), "migrate_v01": migrate_v01}
    try:
        from aipe_magnetics.data.importers.json_record import import_json_record
        imported = import_json_record(source, destination, migrate_v01=migrate_v01)
        result = EngineeringResult(capability=CAP_IMPORT, inputs=inputs)
        result.data.update({"destination": str(destination), "records": len(imported.records), "source_sha256": imported.source_sha256,
                            "warnings": list(imported.warnings)})
    except FileExistsError as error:
        return EngineeringResult.rejected(CAP_IMPORT, InvalidInputError(f"destination exists: {error}"), inputs=inputs)
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(CAP_IMPORT, error, inputs=inputs)
    return _l1(result, "Importer validated the records against the schema and recorded the source checksum (L1).")


def info(catalog: Any | None = None) -> EngineeringResult:
    from aipe_magnetics import SCHEMA_VERSION, __version__
    from aipe_magnetics.database.properties import property_materials
    cat = _catalog(catalog)
    result = EngineeringResult(capability=CAP_INFO)
    result.data.update({"package_version": __version__, "record_schema_version": SCHEMA_VERSION, "catalogue": cat.describe() | {"files": None},
                        "materials": len(cat.materials()), "cores": len(cat.cores()), "physical_property_entries": property_materials()})
    return _l1(result, "Static catalogue summary (L1).")


__all__ = [
    "DETAILS", "HAS_FILTERS", "MAX_LIMIT", "estimate_material_loss", "get_core", "get_material_properties", "get_physical_properties",
    "import_dataset", "info", "query_core_loss_data", "query_saturation_data", "search_cores", "search_materials", "validate_dataset",
]
