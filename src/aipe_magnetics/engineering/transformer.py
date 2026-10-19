"""Basic high-frequency transformer design (analytical sizing).

Flow for each candidate core (user-supplied list or catalogue screening by area product):

1. allowed peak flux density  ``B_allow = min(policy limit from Bsat(T), B at the loss-density target)``
2. primary turns              ``N1 = ceil(V1 * D / (2 f A_e B_allow))`` (square/trapezoidal) or the
                              sinusoidal EMF equation; realised ``B_pk`` recomputed with the integer N1
3. secondary turns            ``N2 = round(N1 * V2 / V1)`` (>= 1), realised ratio and voltage error
4. saturation screening       margin to Bsat(T) (interpolated datasheet points)
5. core loss                  :func:`aipe_magnetics.engineering.core_loss.estimate_core_loss` with the flux
                              waveform implied by the voltage waveform (square -> triangular flux, duty D)
6. winding losses             :func:`aipe_magnetics.engineering.winding.design_winding` per winding
                              (RMS currents from the specification or the ``P / V`` square-wave estimate)
7. window fill                sum of bare copper areas versus the bobbin window area and fill factor
8. ranking                    feasible candidates ordered by total estimated loss

Everything not modelled (leakage inductance, fringing, thermal, insulation, manufacturing) is listed
as a limitation in the result. Values are tagged ``calculated`` (turns, flux density), ``estimated``
(losses) or ``assumed`` (currents from P/V, fill factor, current density, flux limit).
"""

from __future__ import annotations

import math
from typing import Any

from aipe_magnetics.api.types import CoreLossRequest, OperatingPointSpec, TransformerDesignRequest, WaveformSpec
from aipe_magnetics.common.errors import AIPEMagneticsError, DataNotFoundError, InvalidInputError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import EngineeringResult
from aipe_magnetics.data.records import CoreGeometryProperties, MagneticRecord, MaterialProperties
from aipe_magnetics.engineering import formulas as F
from aipe_magnetics.engineering.core_loss import estimate_core_loss
from aipe_magnetics.engineering.materials import evidence_sources, lookup_material
from aipe_magnetics.engineering.saturation import flux_limit_from_policy, saturation_flux_density_at, screen_saturation
from aipe_magnetics.engineering.winding import design_winding

CAP_TRANSFORMER = "aipe.magnetics.design_transformer"
WAVEFORM_FACTOR = {"square": 4.0, "trapezoidal": 4.0, "sinusoidal": 4.44}


def _catalog(catalog: Any | None):
    if catalog is not None:
        return catalog
    from aipe_magnetics.data.catalog import Catalog
    return Catalog.default()


def candidate_cores(material_record: MagneticRecord, requested: list[str] | None, catalog: Any) -> list[MagneticRecord]:
    """Resolve requested core ids or screen catalogue cores offered in the material (fallback: all cores)."""
    if requested:
        cores = []
        for core_id in requested:
            record = catalog.get(core_id)
            if record.kind != "core_geometry":
                raise DataNotFoundError(f"{core_id} is a {record.kind} record, not a core geometry")
            cores.append(record)
        return cores
    offered = [c for c in catalog.cores() if material_record.id in CoreGeometryProperties.model_validate(c.properties).material_refs]
    return offered or list(catalog.cores())


def flux_waveform_for(voltage_waveform: str, duty: float) -> WaveformSpec:
    if voltage_waveform == "sinusoidal":
        return WaveformSpec(kind="sinusoidal")
    if voltage_waveform == "square":
        if abs(duty - 0.5) < 1e-12:
            return WaveformSpec(kind="triangular", duty=0.5)
        # square voltage with dead time: flux rises during D*T, flat during (0.5-D)*T, falls, flat
        return WaveformSpec(kind="trapezoidal", duty_rise=duty, duty_flat=0.5 - duty)
    if voltage_waveform == "trapezoidal":
        return WaveformSpec(kind="trapezoidal", duty_rise=duty, duty_flat=0.5 - duty)
    raise InvalidInputError(f"unsupported voltage waveform {voltage_waveform}")  # pragma: no cover


def flux_density_for_loss_target(*, request: TransformerDesignRequest, material_id: str, b_upper: float, waveform: WaveformSpec,
                                 target: float, catalog: Any) -> tuple[float | None, str | None]:
    """Largest B_pk <= b_upper whose loss density stays below ``target`` (bisection on the loss model)."""
    def pv(b: float) -> float | None:
        req = CoreLossRequest(material=material_id, model=request.loss_model, policy="warn",
                              operating_point=OperatingPointSpec(frequency=request.frequency, flux_density_peak=Quantity(value=b, unit="T"),
                                                                 temperature=request.temperature, waveform=waveform))
        res = estimate_core_loss(req, catalog=catalog)
        if res.status != "ok":
            return None
        return res.values["loss_density"].quantity.value

    top = pv(b_upper)
    if top is None:
        return None, "loss model unavailable for the loss-density target search"
    if top <= target:
        return b_upper, None
    lo, hi = 0.0, b_upper
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        value = pv(mid)
        if value is None:
            return None, "loss model failed during the loss-density target search"
        if value > target:
            hi = mid
        else:
            lo = mid
    return lo, None


def design_transformer(request: TransformerDesignRequest, catalog: Any | None = None) -> EngineeringResult:
    inputs = request.model_dump(mode="json", exclude_none=True)
    cat = _catalog(catalog)
    try:
        material_record, material = lookup_material(request.material, cat)
        cores = candidate_cores(material_record, request.cores, cat)
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(CAP_TRANSFORMER, error, inputs=inputs)

    result = EngineeringResult(capability=CAP_TRANSFORMER, inputs=inputs)
    for s in evidence_sources(material_record):
        result.cite(s)
    f = request.frequency.value
    t_k = request.temperature.value
    power = request.power.value
    v1 = request.primary.voltage.value
    duty = request.duty
    waveform = flux_waveform_for(request.voltage_waveform, duty)

    # material-level quantities
    try:
        bsat_q, bsat_status, _ = saturation_flux_density_at(material, request.temperature)
        bsat = bsat_q.to("T").value
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(CAP_TRANSFORMER, error, inputs=inputs)
    b_policy, flux_assumption = flux_limit_from_policy(request.flux_limit, bsat)
    result.assume(flux_assumption.id, flux_assumption.statement, flux_assumption.basis, ["turns", "flux_density_peak"])
    if bsat_status == "estimated":
        result.assume("bsat_interpolated", f"Bsat at {t_k - 273.15:.0f} degC linearly interpolated between datasheet points.",
                      "engineering_judgement", ["flux_density_peak"])
    b_target, target_note = b_policy, None
    if request.loss_density_target is not None:
        b_target, target_note = flux_density_for_loss_target(request=request, material_id=material_record.id, b_upper=b_policy,
                                                             waveform=waveform, target=request.loss_density_target.value, catalog=cat)
        if b_target is None:
            result.warn("loss_target_unavailable", target_note or "loss-density target could not be applied")
            b_target = b_policy
        else:
            result.assume("loss_density_target", f"Peak flux density additionally limited to {b_target:.3f} T so that the core loss "
                          f"density stays below {request.loss_density_target.value / 1e3:.0f} kW/m^3.", "user_input", ["turns", "flux_density_peak"])
    result.add_value("flux_density_allowed", "Allowed peak flux density (policy and loss target)", Quantity(value=b_target, unit="T"), status="assumed")
    result.add_value("saturation_flux_density", f"Bsat at {t_k - 273.15:.0f} degC", Quantity(value=bsat, unit="T"), status=bsat_status)

    # winding currents
    windings = [request.primary, *request.secondaries]
    currents: dict[str, tuple[float, str]] = {}
    for w in windings:
        if w.current_rms is not None:
            currents[w.name] = (w.current_rms.value, "user_input")
        else:
            currents[w.name] = (F.dab_primary_rms_current_square_approximation(power, w.voltage.value), "estimated")
    if any(basis == "estimated" for _, basis in currents.values()):
        result.assume("winding_currents", "Winding RMS currents estimated as P / V (square-wave current approximation); the real "
                      "converter current shape (e.g. trapezoidal DAB current) changes copper loss.", "engineering_judgement",
                      ["copper_loss"])

    candidates: list[dict[str, Any]] = []
    for core_record in cores:
        core = CoreGeometryProperties.model_validate(core_record.properties)
        row: dict[str, Any] = {"core": core_record.id, "core_name": core_record.name, "feasible": True, "reasons": [], "warnings": []}
        if core.effective_area is None or core.effective_volume is None:
            row.update(feasible=False, reasons=["core record lacks effective area/volume"])
            candidates.append(row)
            continue
        ae, ve = core.effective_area.value, core.effective_volume.value
        row["effective_area_m2"] = ae
        row["effective_volume_m3"] = ve
        row["mass_kg"] = core.mass.value if core.mass else None
        try:
            if request.voltage_waveform == "sinusoidal":
                n1 = F.turns_for_flux_density_sine(v1, f, ae, b_target)
                b_pk = F.flux_density_peak_sine(v1, f, n1, ae)
            else:
                n1 = F.turns_for_flux_density_square(v1, f, ae, b_target, duty)
                b_pk = F.flux_density_peak_square(v1, f, n1, ae, duty)
        except AIPEMagneticsError as error:
            row.update(feasible=False, reasons=[error.message])
            candidates.append(row)
            continue
        row["turns"] = {request.primary.name: n1}
        row["flux_density_peak_T"] = b_pk
        turns_error = {}
        for w in request.secondaries:
            n = max(1, int(round(n1 * w.voltage.value / v1)))
            row["turns"][w.name] = n
            realised = n / n1
            ideal = w.voltage.value / v1
            turns_error[w.name] = realised / ideal - 1.0
            if abs(turns_error[w.name]) > 0.05:
                row["warnings"].append(f"{w.name}: realised turns ratio deviates {turns_error[w.name]:+.1%} from the voltage ratio")
        row["turns_ratio_error"] = turns_error
        # saturation
        screening = screen_saturation(flux_density_peak=b_pk, temperature=t_k, material=material, policy=request.flux_limit)
        row["saturation_margin"] = screening.margin_to_saturation
        if screening.saturated:
            row.update(feasible=False)
            row["reasons"].append("core saturates")
        # core loss
        loss_req = CoreLossRequest(material=material_record.id, core=core_record.id, model=request.loss_model, policy=request.policy,
                                   operating_point=OperatingPointSpec(frequency=request.frequency, flux_density_peak=Quantity(value=b_pk, unit="T"),
                                                                      temperature=request.temperature, waveform=waveform))
        loss = estimate_core_loss(loss_req, catalog=cat)
        if loss.status != "ok":
            row.update(feasible=False)
            row["reasons"].append("core loss: " + "; ".join(e.message for e in loss.errors))
            row["core_loss_W"] = None
        else:
            row["loss_density_W_per_m3"] = loss.values["loss_density"].quantity.value
            row["core_loss_W"] = loss.values["core_loss"].quantity.value if "core_loss" in loss.values else None
            row["loss_model"] = f"{loss.model.id}@{loss.model.version}" if loss.model else None
            row["loss_verification"] = loss.verification.model_dump(mode="json")
            row["warnings"].extend(f"core loss: {w.message}" for w in loss.warnings if w.severity != "info")
            for lim in loss.limitations:
                result.limit(lim.code, lim.statement)
            for a in loss.assumptions:
                result.assume(a.id, a.statement, a.basis, a.affects)
            for s in loss.sources:
                result.cite(s)
        # windings
        mlt = core.mean_turn_length.value if core.mean_turn_length else None
        window = (core.bobbin_window_area or core.window_area)
        window_area = window.value if window else None
        copper_total = 0.0
        copper_area_total = 0.0
        row["windings"] = {}
        if mlt is None:
            row["warnings"].append("core record has no mean turn length; winding losses not evaluated")
            row["copper_loss_W"] = None
        else:
            dims = {k: q.value for k, q in core.dimensions.values.items()} if core.dimensions else None
            for w in windings:
                i_rms, basis = currents[w.name]
                try:
                    wd = design_winding(turns=row["turns"][w.name], current_rms=i_rms, frequency=f, mean_turn_length=mlt,
                                        spec=w.winding, temperature=t_k, window_area=window_area, dimensions=dims)
                except AIPEMagneticsError as error:
                    row["warnings"].append(f"{w.name}: winding not evaluated: {error.message}")
                    continue
                copper_total += wd.copper_loss
                copper_area_total += wd.turns * wd.conductor_area
                row["windings"][w.name] = {"turns": wd.turns, "current_rms_A": i_rms, "current_basis": basis,
                                           "conductor_type": wd.conductor_type, "conductor_area_m2": wd.conductor_area,
                                           "layers": wd.layers, "resistance_dc_Ohm": wd.resistance_dc,
                                           "ac_resistance_factor": wd.ac_resistance_factor, "copper_loss_W": wd.copper_loss}
                row["warnings"].extend(f"{w.name}: {x.message}" for x in wd.warnings)
                for a in wd.assumptions:
                    result.assume(a.id, a.statement, a.basis, ["copper_loss"])
                for lim in wd.limitations:
                    result.limit(lim.code, lim.statement)
                for s in wd.sources:
                    result.cite(s)
            row["copper_loss_W"] = copper_total
        if window_area is not None and copper_area_total > 0:
            fill = copper_area_total / window_area
            row["window_fill"] = fill
            limit = min(w.winding.fill_factor for w in windings)
            if fill > limit:
                row.update(feasible=False)
                row["reasons"].append(f"window fill {fill:.2f} exceeds the fill factor {limit:.2f}")
        else:
            row["window_fill"] = None
            if window_area is None:
                row["warnings"].append("core record has no window area; fill not evaluated")
        total = (row.get("core_loss_W") or 0.0) + (row.get("copper_loss_W") or 0.0)
        row["total_loss_W"] = total if row.get("core_loss_W") is not None else None
        candidates.append(row)

    feasible = [c for c in candidates if c["feasible"] and c.get("total_loss_W") is not None]
    feasible.sort(key=lambda c: c["total_loss_W"])
    ranked = feasible + [c for c in candidates if c not in feasible]
    result.data["candidates"] = ranked[: max(request.max_candidates, len(feasible))] if request.cores is None else ranked
    result.data["material"] = material_record.id
    result.data["flux_waveform"] = waveform.model_dump(mode="json", exclude_none=True)
    result.data["winding_currents"] = {k: {"current_rms_A": v[0], "basis": v[1]} for k, v in currents.items()}
    if not feasible:
        result.status = "rejected"
        result.errors.append(__import__("aipe_magnetics.common.results", fromlist=["ErrorRecord"]).ErrorRecord(
            code="no_feasible_candidate", message="No candidate core satisfied saturation, window and loss-model constraints; see data.candidates",
            details={"candidates": [c["core"] for c in candidates]}))
    else:
        best = feasible[0]
        result.data["selected"] = best["core"]
        for name, n in best["turns"].items():
            result.add_value(f"turns_{name}", f"Turns of winding {name}", Quantity(value=float(n), unit="1"), status="calculated",
                             derived_from=["voltage", "frequency", "effective_area", "flux_density_allowed"])
        result.add_value("flux_density_peak", "Peak flux density (selected core)", Quantity(value=best["flux_density_peak_T"], unit="T"),
                         status="calculated", derived_from=["turns_" + request.primary.name])
        result.add_value("saturation_margin", "Margin to saturation", Quantity(value=best["saturation_margin"], unit="1"), status="calculated")
        if best.get("loss_density_W_per_m3") is not None:
            result.add_value("loss_density", "Core loss density", Quantity(value=best["loss_density_W_per_m3"], unit="W/m^3"), status="estimated")
        if best.get("core_loss_W") is not None:
            result.add_value("core_loss", "Core loss", Quantity(value=best["core_loss_W"], unit="W"), status="estimated")
        if best.get("copper_loss_W") is not None:
            result.add_value("copper_loss", "Copper loss (all windings)", Quantity(value=best["copper_loss_W"], unit="W"), status="estimated")
        if best.get("total_loss_W") is not None:
            result.add_value("total_loss", "Total estimated transformer loss", Quantity(value=best["total_loss_W"], unit="W"), status="estimated")
        if best.get("window_fill") is not None:
            result.add_value("window_fill", "Window fill (bare copper / window)", Quantity(value=best["window_fill"], unit="1"), status="estimated")
        for name, (i_rms, basis) in currents.items():
            result.add_value(f"current_rms_{name}", f"RMS current of winding {name}", Quantity(value=i_rms, unit="A"),
                             status="assumed" if basis == "estimated" else "looked_up")
        verifications = [c.get("loss_verification") for c in feasible if c.get("loss_verification")]
        if verifications:
            from aipe_magnetics.common.results import VerificationStatus
            result.verification = VerificationStatus.model_validate(best["loss_verification"])
            result.verification.statement = ("Turns and flux density follow from Faraday's law (L1 software checks); core loss "
                                             "verification status: " + result.verification.statement)
    result.limit("analytical_design", "Analytical sizing only: no finite-element, leakage-inductance, fringing, thermal, insulation or "
                 "manufacturing analysis was performed.")
    result.limit("no_hardware_validation", "No hardware was built or measured; results are design estimates, not qualification data.")
    result.limit("effective_parameters", "Core effective parameters (Ae, le, Ve, window, mean turn length) are datasheet values; "
                 "gapping, stacking and tolerances are not considered.")
    return result
