"""Magnetic-material measurement requests derived from the catalogue coverage of an excitation (structured JSON).

For every catalogue material the coverage of seven data categories is assessed against the requested operating
conditions - B-H curve, core loss versus frequency / flux density / temperature, DC bias, saturation versus
temperature, complex permeability, winding AC resistance, thermal conductivity - and a measurement request with
explicit **test conditions** and a **priority** is generated for each category that is missing, partial or only
covered by a generic class value. Nothing is invented: the request says what the catalogue holds, what the design
needs and why the gap matters.

Priorities
----------
* ``high``   - the gap blocks or invalidates the design (no saturation data at the design temperature, no loss
  parameter set covering the design temperature / flux density, extrapolated loss sets);
* ``medium`` - the gap limits the accuracy of a quantity the design reports (non-sinusoidal loss under the actual
  waveform, DC bias, winding AC resistance, grade-specific thermal conductivity, magnetising inductance);
* ``low``    - data the current models do not consume (full B-H loops, complex permeability spectra) but that a
  hysteresis or small-signal model would need.

Output schema ``aipe.magnetics.measurement-request/0.3`` (``data.requests[]``): ``material``, ``grade``,
``manufacturer_ref``, ``coverage`` (category -> status and what is available) and ``requests[]`` (``id``,
``category``, ``priority``, ``status_before``, ``rationale``, ``test_conditions``, ``deliverable``, ``record_kind``).
"""

from __future__ import annotations

from typing import Any

from aipe_magnetics.api.types import MeasurementRequestSpec
from aipe_magnetics.common.errors import AIPEMagneticsError, DataNotFoundError, OutOfDomainError
from aipe_magnetics.common.results import EngineeringResult
from aipe_magnetics.data.records import MaterialProperties

CAP_MEASUREMENT_REQUESTS = "aipe.magnetics.measurement_requests"
REQUEST_SCHEMA = "aipe.magnetics.measurement-request/0.3"
CATEGORIES = ("bh_curve", "core_loss_vs_f_b_t", "dc_bias", "saturation_vs_temperature", "complex_permeability",
              "winding_ac_resistance", "thermal_conductivity")
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_DEFAULT_TEMPERATURES_C = (25.0, 50.0, 70.0, 90.0, 100.0, 120.0)
_DEFAULT_B_GRID_T = (0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
_DEFAULT_H_DC_GRID = (0.0, 25.0, 50.0, 100.0, 200.0)
_FIT_TEMPERATURE_TOLERANCE_K = 5.0


def _c(kelvin: float) -> float:
    return round(kelvin - 273.15, 1)


def _temperature_grid(spec: MeasurementRequestSpec) -> list[float]:
    temps = {round(t, 1) for t in _DEFAULT_TEMPERATURES_C}
    temps.add(_c(spec.temperature.value))
    if spec.temperature_range:
        temps.add(_c(spec.temperature_range[0].value))
        temps.add(_c(spec.temperature_range[1].value))
    return sorted(temps)


def _flux_grid(spec: MeasurementRequestSpec, bsat: float | None) -> list[float]:
    if spec.flux_density_peak is None:
        grid = list(_DEFAULT_B_GRID_T)
    else:
        b = spec.flux_density_peak.value
        grid = [round(b * k, 4) for k in (0.5, 0.75, 1.0, 1.25, 1.5)]
    if bsat:
        grid = [b for b in grid if b <= 0.9 * bsat] or [round(0.5 * bsat, 4)]
    return grid


def _frequency_grid(spec: MeasurementRequestSpec) -> list[float]:
    f = spec.frequency.value
    return [0.5 * f, f, 2.0 * f]


def _loss_set_coverage(props: MaterialProperties, frequency: float, temperature: float, b_pk: float | None) -> dict[str, Any]:
    sets = props.steinmetz_parameter_sets
    if not sets:
        return {"status": "missing", "available": {"parameter_sets": 0, "loss_reference_points": len(props.loss_reference_points)},
                "gaps": ["no Steinmetz / iGSE parameter set"]}
    temps = sorted({s.temperature.magnitude("K") for s in sets if s.temperature is not None})
    covering_f = []
    for s in sets:
        rng = s.frequency_range
        if rng is None or ((rng.min is None or frequency >= rng.min) and (rng.max is None or frequency <= rng.max)):
            covering_f.append(s)
    gaps: list[str] = []
    if not covering_f:
        gaps.append(f"no parameter set covers {frequency / 1e3:.0f} kHz")
    t_cover = [s for s in covering_f if s.temperature is not None and (
        (s.temperature_range is not None and (s.temperature_range.min is None or temperature >= s.temperature_range.min)
         and (s.temperature_range.max is None or temperature <= s.temperature_range.max))
        or abs(s.temperature.magnitude("K") - temperature) <= _FIT_TEMPERATURE_TOLERANCE_K)]
    span = (min(temps), max(temps)) if temps else None
    interpolable = span is not None and span[0] <= temperature <= span[1]
    if not t_cover and not interpolable:
        gaps.append(f"no parameter set fitted at {_c(temperature):g} degC (fits span {_c(span[0]) if span else '-'}..{_c(span[1]) if span else '-'} degC); "
                    "the design extrapolates")
    elif not t_cover:
        gaps.append(f"{_c(temperature):g} degC is covered only by interpolation between fits at {[_c(t) for t in temps]} degC")
    b_gap = False
    if b_pk is not None:
        in_b = [s for s in (t_cover or covering_f) if s.flux_density_range is None or (
            (s.flux_density_range.min is None or b_pk >= s.flux_density_range.min) and (s.flux_density_range.max is None or b_pk <= s.flux_density_range.max))]
        if (t_cover or covering_f) and not in_b:
            b_gap = True
            gaps.append(f"peak flux density {b_pk * 1e3:.0f} mT lies outside the fitted flux-density range of the usable set(s)")
    excitations = sorted({s.excitation for s in sets})
    nonsinusoidal = any(p.excitation != "sinusoidal" for p in props.loss_reference_points)
    if not nonsinusoidal:
        gaps.append("all loss data are sinusoidal fits / points; no measurement under the actual (non-sinusoidal) waveform")
    status = "missing" if not covering_f else ("partial" if (gaps and (not t_cover or b_gap or not interpolable)) else ("available" if not gaps else "partial"))
    if status == "available" and gaps:
        status = "partial"
    return {"status": status, "available": {"parameter_sets": [s.id for s in sets], "fit_temperatures_degC": [_c(t) for t in temps],
                                            "excitations": excitations, "loss_reference_points": len(props.loss_reference_points),
                                            "covering_frequency": [s.id for s in covering_f], "covering_temperature": [s.id for s in t_cover]},
            "gaps": gaps, "blocking": not covering_f or (not t_cover and not interpolable) or b_gap}


def assess_coverage(record: Any, props: MaterialProperties, spec: MeasurementRequestSpec, catalog: Any) -> dict[str, dict[str, Any]]:
    """Coverage status of every category for one material under the requested conditions."""
    from aipe_magnetics.engineering.materials import saturation_flux_density_at
    f, t = spec.frequency.value, spec.temperature.value
    b_pk = None if spec.flux_density_peak is None else spec.flux_density_peak.value
    out: dict[str, dict[str, Any]] = {}
    # B-H curve
    loops = [r.id for r in catalog.records("bh_loop") if r.properties.get("material_ref") == record.id]
    if loops:
        out["bh_curve"] = {"status": "available", "available": {"bh_loop_records": loops}, "gaps": []}
    elif props.coercivity or props.remanence:
        out["bh_curve"] = {"status": "partial", "available": {"coercivity_points": len(props.coercivity), "remanence_points": len(props.remanence)},
                           "gaps": ["datasheet coercivity / remanence points only; no measured B-H loop"]}
    else:
        out["bh_curve"] = {"status": "missing", "available": {}, "gaps": ["no B-H loop, coercivity or remanence data"]}
    # core loss
    out["core_loss_vs_f_b_t"] = _loss_set_coverage(props, f, t, b_pk)
    # DC bias
    biased = [p for p in props.loss_reference_points if p.dc_bias_field is not None]
    if props.dc_bias_points:
        out["dc_bias"] = {"status": "available", "available": {"dc_bias_points": len(props.dc_bias_points)}, "gaps": []}
    elif biased:
        out["dc_bias"] = {"status": "partial", "available": {"loss_points_with_dc_bias": len(biased)}, "gaps": ["no permeability / B versus H_dc data"]}
    else:
        out["dc_bias"] = {"status": "missing", "available": {}, "gaps": ["no DC-bias (premagnetisation) data"]}
    # saturation vs temperature
    sat_temps = sorted({p.temperature.magnitude("K") for p in props.saturation_flux_density if p.temperature is not None})
    bsat = None
    if props.saturation_flux_density:
        try:
            from aipe_magnetics.common.quantity import Quantity
            bsat = saturation_flux_density_at(props, Quantity(value=t, unit="K"))[0].to("T").value
            status = "available" if len(sat_temps) >= 2 else "partial"
            gaps = [] if len(sat_temps) >= 2 else ["single datasheet point; temperature dependence unknown"]
        except (OutOfDomainError, DataNotFoundError) as error:
            status, gaps = "partial", [error.message]
        out["saturation_vs_temperature"] = {"status": status, "available": {"temperatures_degC": [_c(x) for x in sat_temps], "bsat_at_design_T": bsat},
                                            "gaps": gaps, "blocking": bsat is None}
    else:
        out["saturation_vs_temperature"] = {"status": "missing", "available": {}, "gaps": ["no saturation flux density data"], "blocking": True}
    # complex permeability
    if props.amplitude_permeability or props.permeability_vs_temperature:
        out["complex_permeability"] = {"status": "partial", "available": {"amplitude_permeability_points": len(props.amplitude_permeability),
                                                                           "permeability_vs_temperature_points": len(props.permeability_vs_temperature)},
                                       "gaps": ["no frequency-resolved complex permeability (mu', mu'')"]}
    elif props.initial_permeability is not None:
        out["complex_permeability"] = {"status": "partial", "available": {"initial_permeability": props.initial_permeability.value.value},
                                       "gaps": ["initial permeability only (single small-signal value); no mu'(f), mu''(f), no temperature dependence"]}
    else:
        out["complex_permeability"] = {"status": "missing", "available": {}, "gaps": ["no permeability data"]}
    # winding AC resistance (component level, never in a material record)
    out["winding_ac_resistance"] = {"status": "missing", "available": {}, "gaps": ["no measured winding AC resistance; Dowell / strand models only"]}
    # thermal conductivity (property library holds class values with bounds)
    from aipe_magnetics.database.properties import property_entry
    try:
        entry = property_entry(props.material_class or "ferrite_mnzn")
        k = entry.get("thermal_conductivity", {})
        out["thermal_conductivity"] = {"status": "generic_class_value", "available": {"class": props.material_class, "value_W_per_mK": k.get("value"),
                                                                                       "bounds_W_per_mK": [k.get("lower"), k.get("upper")], "basis": k.get("basis")},
                                       "gaps": ["generic class value with a wide range; no grade-specific measurement"]}
    except DataNotFoundError:
        out["thermal_conductivity"] = {"status": "missing", "available": {}, "gaps": ["no thermal conductivity data"]}
    out["_bsat"] = {"value": bsat}
    return out


def build_requests(record: Any, props: MaterialProperties, coverage: dict[str, dict[str, Any]], spec: MeasurementRequestSpec,
                   winding_context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    bsat = coverage.get("_bsat", {}).get("value")
    temps = _temperature_grid(spec)
    bgrid = _flux_grid(spec, bsat)
    f = spec.frequency.value
    t_design = _c(spec.temperature.value)
    slug = record.id.replace("material.", "")
    out: list[dict[str, Any]] = []

    def add(category: str, priority: str, rationale: str, conditions: dict[str, Any], deliverable: str, record_kind: str) -> None:
        cov = coverage[category]
        if cov["status"] == "available" and not spec.include_available:
            return
        out.append({"id": f"request.{slug}.{category}", "material": record.id, "category": category, "priority": priority if cov["status"] != "available" else "low",
                    "status_before": cov["status"], "available": cov.get("available", {}), "gaps": cov.get("gaps", []), "rationale": rationale,
                    "test_conditions": conditions, "deliverable": deliverable, "record_kind": record_kind})

    sat = coverage["saturation_vs_temperature"]
    add("saturation_vs_temperature", "high" if sat.get("blocking") else ("medium" if sat["status"] == "partial" else "low"),
        "Bsat(T) sets the flux limit and the minimum primary turns; without it saturation cannot be screened at the design temperature.",
        {"quantity": "B at H = 1200 A/m (datasheet convention) and B-H knee", "field_strength_A_per_m": [250, 500, 1200],
         "temperatures_degC": temps, "frequency_Hz": 10e3, "waveform": "sinusoidal H, major loop"},
        "table B(H, T) at the stated fields and temperatures with uncertainty; one ConditionedValue per (T, H)", "magnetic_material.saturation_flux_density")
    loss = coverage["core_loss_vs_f_b_t"]
    add("core_loss_vs_f_b_t", "high" if loss.get("blocking") else "medium",
        ("Core loss is the dominant model uncertainty: the design evaluates iGSE on sinusoidal-fit parameters; " +
         ("the catalogue sets do not cover the design temperature / flux density (extrapolation)." if loss.get("blocking")
          else "measurements under the actual waveform at the design temperature validate the iGSE hypothesis for this excitation.")),
        {"frequencies_Hz": _frequency_grid(spec), "flux_density_peaks_T": bgrid, "temperatures_degC": temps,
         "waveforms": sorted({spec.excitation, "sinusoidal"}), "dc_bias_field_A_per_m": 0.0, "duty": 0.5 if spec.excitation == "triangular" else None,
         "method": "calorimetric or two-winding wattmetric loss measurement on a toroid (IEC 62044-3), >= 3 samples"},
        "loss density table Pv(f, B_pk, T, waveform) with uncertainty -> core_loss_measurement record; refitted Steinmetz / iGSE sets per temperature",
        "core_loss_measurement")
    h_grid = list(_DEFAULT_H_DC_GRID) if spec.dc_bias_field_max is None else sorted({0.0, 0.25 * spec.dc_bias_field_max.value, 0.5 * spec.dc_bias_field_max.value,
                                                                                      spec.dc_bias_field_max.value})
    add("dc_bias", "medium" if (spec.dc_bias_field_max is None or spec.dc_bias_field_max.value > 0) else "low",
        "Steinmetz / iGSE ignore premagnetisation; volt-second imbalance or asymmetric modulation produces a DC flux whose effect on loss and "
        "permeability is unknown without data.",
        {"dc_bias_fields_A_per_m": h_grid, "frequency_Hz": f, "flux_density_peaks_T": bgrid[:3], "temperatures_degC": [25.0, t_design],
         "waveform": spec.excitation, "quantities": ["loss density", "amplitude permeability"]},
        "Pv and mu_a versus H_dc at the stated conditions -> loss reference points with dc_bias_field and dc_bias_points", "magnetic_material.dc_bias_points")
    add("bh_curve", "low", "The design models (Steinmetz / iGSE, reluctance) do not consume B-H loops; a hysteresis model (Jiles-Atherton) fit would.",
        {"frequencies_Hz": [10e3, f], "field_peaks_A_per_m": [100, 300, 1200], "temperatures_degC": [25.0, t_design], "waveform": "sinusoidal H",
         "sampling": ">= 1024 samples per period"},
        "measured major and minor B-H loops -> bh_loop records with HDF5 waveform artifacts", "bh_loop")
    add("complex_permeability", "medium" if winding_context else "low",
        "The magnetising inductance uses the 25 degC initial permeability with an assumed residual gap; mu'(f, T) and mu''(f) are needed for the "
        "magnetising current and the small-signal model at the switching frequency.",
        {"frequencies_Hz": [10e3, 100e3, 300e3, 1e6], "flux_density_T": 0.00025, "temperatures_degC": temps, "quantities": ["mu_prime", "mu_double_prime"]},
        "complex permeability spectra per temperature -> amplitude_permeability / permeability_vs_temperature points", "magnetic_material.permeability")
    harmonics = [k * f for k in range(1, spec.harmonics + 1) if k % 2 == 1] or [f]
    wc = winding_context or {}
    add("winding_ac_resistance", "medium",
        "Copper loss comes from Dowell's 1-D layer model (litz as strand sub-layers); a measured R_ac(f) of the actual winding bounds the model error.",
        {"frequencies_Hz": harmonics, "temperatures_degC": [25.0, wc.get("winding_temperature_degC", t_design)], "excitation": "small-signal sinusoidal, "
         "secondary short-circuited (leakage path) and open (magnetising path)", "winding": wc.get("conductor"), "turns": wc.get("turns"), "core": wc.get("core"),
         "arrangement": wc.get("arrangement"), "quantities": ["R_ac / R_dc", "leakage inductance", "magnetising inductance"]},
        "R_ac(f) table and inductances of the prototype winding -> measurement record referencing the winding / transformer record", "measurement")
    add("thermal_conductivity", "medium",
        "The thermal estimate uses a class value (1-5 W/m/K range) for the ferrite; the hot-spot estimate scales with it.",
        {"temperatures_degC": [25.0, 60.0, 100.0, 120.0], "method": "laser flash or guarded hot plate on a sintered sample of the grade",
         "quantities": ["thermal conductivity", "specific heat"]},
        "grade-specific k(T) and c_p(T) with uncertainty -> property library entry", "physical_property")
    out.sort(key=lambda r: PRIORITY_ORDER[r["priority"]])
    return out


def measurement_requests(spec: MeasurementRequestSpec, catalog: Any | None = None, *, winding_context: dict[str, Any] | None = None) -> EngineeringResult:
    inputs = spec.model_dump(mode="json", exclude_none=True)
    from aipe_magnetics.data.catalog import Catalog
    from aipe_magnetics.engineering.materials import evidence_sources, lookup_material
    cat = catalog if catalog is not None else Catalog.default()
    result = EngineeringResult(capability=CAP_MEASUREMENT_REQUESTS, inputs=inputs)
    try:
        if spec.materials:
            records = [lookup_material(q, cat)[0] for q in spec.materials]
        else:
            records = [m for m in cat.materials() if not m.synthetic]
        materials: list[dict[str, Any]] = []
        totals = {"high": 0, "medium": 0, "low": 0}
        for record in records:
            props = MaterialProperties.model_validate(record.properties)
            coverage = assess_coverage(record, props, spec, cat)
            requests = build_requests(record, props, coverage, spec, winding_context)
            for r in requests:
                totals[r["priority"]] += 1
            materials.append({"material": record.id, "grade": props.grade, "manufacturer_ref": props.manufacturer_ref, "synthetic": record.synthetic,
                              "coverage": {k: v for k, v in coverage.items() if not k.startswith("_")}, "requests": requests})
            for s in evidence_sources(record):
                result.cite(s)
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(CAP_MEASUREMENT_REQUESTS, error, inputs=inputs)
    result.data.update({"schema": REQUEST_SCHEMA, "conditions": {"frequency_Hz": spec.frequency.value, "temperature_degC": _c(spec.temperature.value),
                                                                   "flux_density_peak_T": None if spec.flux_density_peak is None else spec.flux_density_peak.value,
                                                                   "excitation": spec.excitation, "winding_context": winding_context},
                        "requests": materials, "summary": {"materials": len(materials), "requests_by_priority": totals}})
    result.assume("priority_rules", "Priority high = the gap blocks or invalidates the design (saturation or loss data missing at the design conditions); "
                  "medium = limits the accuracy of a reported quantity; low = data the current models do not consume.", "engineering_judgement")
    result.limit("catalogue_scope", "Coverage is assessed against the loaded catalogue only; data the manufacturer publishes but that was not transcribed "
                 "appears as missing.")
    result.limit("not_a_test_plan", "Test conditions are proposals derived from the excitation; sample sizes, fixtures and acceptance criteria must be "
                 "defined by the measuring laboratory.")
    result.verification.level = "L1"
    result.verification.state = "verified"
    result.verification.statement = "Deterministic coverage assessment of schema-validated catalogue records (L1); no physical claim."
    return result


__all__ = ["CAP_MEASUREMENT_REQUESTS", "CATEGORIES", "REQUEST_SCHEMA", "assess_coverage", "build_requests", "measurement_requests"]
