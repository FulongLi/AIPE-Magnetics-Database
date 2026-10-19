"""Basic high-frequency transformer design (analytical sizing).

Candidate cores are the user-supplied list or, with ``cores=None``, **every** catalogue core offered in
the material (fallback: every catalogue core). No area-product pre-screening is performed; the window
fill check below is the exact form of that constraint. Flow for each candidate:

1. allowed peak flux density  ``B_allow = min(policy limit from Bsat(T), B at the loss-density target)``
2. primary turns              ``N1 = ceil(V1 * D / (2 f A_e B_allow))`` (square/trapezoidal) or the
                              sinusoidal EMF equation; realised ``B_pk`` recomputed with the integer N1
3. secondary turns            ``N2 = round(N1 * V2 / V1)`` (>= 1), realised ratio and voltage error
4. saturation screening       margin to Bsat(T) (interpolated datasheet points)
5. core loss                  :func:`aipe_magnetics.engineering.core_loss.estimate_core_loss` with the flux
                              waveform implied by the voltage waveform (square -> triangular flux, duty D)
6. winding losses             :func:`aipe_magnetics.engineering.winding.design_winding` per winding
                              (RMS currents from the specification or the ``P_w / V`` square-wave estimate,
                              where the secondaries share the throughput power)
7. window fill                sum of bare copper areas versus the bobbin window area and fill factor
8. ranking                    feasible, *completely evaluated* candidates ordered by total estimated loss

A candidate is **feasible** only when saturation, window fill and the loss model are satisfied *and*
every part of the evaluation could be carried out. A core record without mean turn length or window
area, or a winding the Dowell model cannot evaluate, leaves copper loss and fill unknown: such a
candidate is reported as not feasible with the reason (``evaluation: incomplete``) and is never ranked
against complete totals. The warnings of the selected candidate (turns-ratio deviation, loss-model
extrapolation, high AC resistance factor, ...) are repeated at the top level of the result.

Everything not modelled (leakage inductance, fringing, thermal, insulation, manufacturing) is listed
as a limitation in the result. Values are tagged ``calculated`` (turns, flux density), ``estimated``
(losses), ``assumed`` (currents from P/V, fill factor, current density, flux limit) or ``specified``
(currents supplied by the caller).
"""

from __future__ import annotations

from typing import Any

from aipe_magnetics.api.types import CoreLossRequest, OperatingPointSpec, TransformerDesignRequest, WaveformSpec
from aipe_magnetics.common.errors import AIPEMagneticsError, DataNotFoundError, InvalidInputError, OutOfDomainError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import EngineeringResult, ErrorRecord, ModelIdentity, VerificationStatus
from aipe_magnetics.data.records import CoreGeometryProperties, MagneticRecord
from aipe_magnetics.engineering import formulas as F
from aipe_magnetics.engineering.core_loss import estimate_core_loss
from aipe_magnetics.engineering.materials import evidence_sources, lookup_material
from aipe_magnetics.engineering.operating_point import validate_operating_point
from aipe_magnetics.engineering.saturation import flux_limit_from_policy, saturation_flux_density_at, screen_saturation
from aipe_magnetics.engineering.winding import design_winding

CAP_TRANSFORMER = "aipe.magnetics.design_transformer"
TURNS_RATIO_TOLERANCE = 0.05
INFEASIBLE_WINDING_WARNINGS = {"foil_width_exceeds_window"}


def _catalog(catalog: Any | None):
    if catalog is not None:
        return catalog
    from aipe_magnetics.data.catalog import Catalog
    return Catalog.default()


def candidate_cores(material_record: MagneticRecord, requested: list[str] | None, catalog: Any) -> list[MagneticRecord]:
    """Resolve requested core ids, or every catalogue core offered in the material (fallback: every catalogue core).

    This is an enumeration, not a screening: every returned core is evaluated and ranked by the design flow.
    """
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
    """Flux waveform implied by the voltage waveform: ``duty`` is the fraction of the full period with +V applied."""
    if voltage_waveform == "sinusoidal":
        return WaveformSpec(kind="sinusoidal")
    if voltage_waveform in ("square", "trapezoidal"):
        if abs(duty - 0.5) < 1e-12:
            return WaveformSpec(kind="triangular", duty=0.5)
        # voltage applied for D*T, zero for (0.5-D)*T: flux rises during D*T, is flat, falls, is flat
        return WaveformSpec(kind="trapezoidal", duty_rise=duty, duty_flat=0.5 - duty)
    raise InvalidInputError(f"unsupported voltage waveform {voltage_waveform}")  # pragma: no cover


def power_shares(request: TransformerDesignRequest) -> dict[str, float]:
    """Share of the throughput power carried by each secondary (explicit ``power_share`` or equal split)."""
    given = {w.name: w.power_share for w in request.secondaries if w.power_share is not None}
    if given:
        return {w.name: float(w.power_share) for w in request.secondaries}  # validated complete by the request model
    n = len(request.secondaries)
    return {w.name: 1.0 / n for w in request.secondaries}


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

    # material-level operating-point checks (Curie margin, recommended frequency range, loss-parameter coverage)
    op_checks = validate_operating_point(material=material, frequency=f, flux_density_peak=None, temperature=t_k, flux_limit=request.flux_limit)
    for check in op_checks.checks:
        if check.id == "saturation":
            continue  # screened per candidate below with the realised peak flux density
        if check.status == "fail":
            if check.id == "curie_temperature":
                return EngineeringResult.rejected(CAP_TRANSFORMER, OutOfDomainError(check.message, details={"check": check.id}), inputs=inputs)
            result.warn(check.id, check.message, "error")
        elif check.status == "warning":
            result.warn(check.id, check.message, "warning")
    result.data["operating_point_checks"] = [c.model_dump(mode="json") for c in op_checks.checks if c.id != "saturation"]

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
    result.assume("zero_dc_bias", "The flux waveform has no DC component (volt-second balanced bipolar voltage); the Steinmetz/iGSE "
                  "loss models ignore DC bias, so any residual bias in the converter would raise the core loss above this estimate.",
                  "model_assumption", ["loss_density", "core_loss"])
    if request.voltage_waveform == "trapezoidal":
        result.assume("trapezoidal_voltage_as_square", f"The trapezoidal voltage is treated as a square wave of the same amplitude applied "
                      f"for the duty fraction {duty:g} of the period (flux rises linearly for {duty:g} T); the finite voltage slew - rounded "
                      "flux corners and a lower peak dB/dt - is not modelled.", "model_assumption", ["flux_density_peak", "loss_density"])

    # winding currents: user-specified, or P_w / V with the secondaries sharing the throughput power
    windings = [request.primary, *request.secondaries]
    shares = power_shares(request)
    currents: dict[str, tuple[float, str]] = {}
    for w in windings:
        if w.current_rms is not None:
            currents[w.name] = (w.current_rms.value, "user_input")
        elif w is request.primary:
            currents[w.name] = (F.dab_primary_rms_current_square_approximation(power, v1), "estimated")
        else:
            currents[w.name] = (F.dab_primary_rms_current_square_approximation(power * shares[w.name], w.voltage.value), "estimated")
    if any(basis == "estimated" for _, basis in currents.values()):
        share_text = ", ".join(f"{name} {share:.0%}" for name, share in shares.items())
        result.assume("winding_currents", "Winding RMS currents estimated as P_w / V (square-wave current approximation) with the full "
                      f"throughput power in the primary and the secondaries sharing it ({share_text}); the real converter current "
                      "shape (e.g. trapezoidal DAB current) changes copper loss.", "engineering_judgement", ["copper_loss"])

    candidates: list[dict[str, Any]] = []
    coded_warnings: dict[str, list[tuple[str, str]]] = {}

    def cwarn(row: dict[str, Any], code: str, message: str) -> None:
        row["warnings"].append(message)
        coded_warnings.setdefault(row["core"], []).append((code, message))

    for core_record in cores:
        core = CoreGeometryProperties.model_validate(core_record.properties)
        row: dict[str, Any] = {"core": core_record.id, "core_name": core_record.name, "feasible": True, "evaluation": "complete",
                               "reasons": [], "warnings": []}
        if core.effective_area is None or core.effective_volume is None:
            row.update(feasible=False, evaluation="incomplete", reasons=["core record lacks effective area/volume"])
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
            row.update(feasible=False, evaluation="incomplete", reasons=[error.message])
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
            if abs(turns_error[w.name]) > TURNS_RATIO_TOLERANCE:
                cwarn(row, "turns_ratio_deviation", f"{w.name}: realised turns ratio {n}:{n1} deviates {turns_error[w.name]:+.1%} from the "
                      f"voltage ratio (integer turns)")
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
            row["loss_model_identity"] = loss.model.model_dump(mode="json", exclude_none=True) if loss.model else None
            row["loss_verification"] = loss.verification.model_dump(mode="json")
            for w_ in loss.warnings:
                if w_.severity != "info":
                    cwarn(row, f"core_loss_{w_.code}", f"core loss: {w_.message}")
            for lim in loss.limitations:
                result.limit(lim.code, lim.statement)
            for a in loss.assumptions:
                result.assume(a.id, a.statement, a.basis, a.affects)
            for s in loss.sources:
                result.cite(s)
        # windings and window fill (both the mean turn length and the window area are required for a complete evaluation)
        mlt = core.mean_turn_length.value if core.mean_turn_length else None
        window = (core.bobbin_window_area or core.window_area)
        window_area = window.value if window else None
        row["windings"] = {}
        row["copper_loss_W"] = None
        row["window_fill"] = None
        missing = [name for name, value in (("mean turn length", mlt), ("window area", window_area)) if value is None]
        if missing:
            row.update(feasible=False, evaluation="incomplete")
            row["reasons"].append(f"not evaluable: core record has no {' and no '.join(missing)}; copper loss and window fill cannot be assessed")
        else:
            dims = {k: q.value for k, q in core.dimensions.values.items()} if core.dimensions else None
            copper_total = 0.0
            copper_area_total = 0.0
            complete = True
            for w in windings:
                i_rms, basis = currents[w.name]
                try:
                    wd = design_winding(turns=row["turns"][w.name], current_rms=i_rms, frequency=f, mean_turn_length=mlt,
                                        spec=w.winding, temperature=t_k, window_area=window_area, dimensions=dims)
                except AIPEMagneticsError as error:
                    complete = False
                    row.update(feasible=False, evaluation="incomplete")
                    row["reasons"].append(f"{w.name}: winding not evaluable: {error.message}")
                    continue
                copper_total += wd.copper_loss
                copper_area_total += wd.turns * wd.conductor_area
                row["windings"][w.name] = {"turns": wd.turns, "current_rms_A": i_rms, "current_basis": basis,
                                           "conductor_type": wd.conductor_type, "conductor_area_m2": wd.conductor_area,
                                           "layers": wd.layers, "resistance_dc_Ohm": wd.resistance_dc,
                                           "ac_resistance_factor": wd.ac_resistance_factor, "copper_loss_W": wd.copper_loss}
                for x in wd.warnings:
                    cwarn(row, x.code, f"{w.name}: {x.message}")
                    if x.code in INFEASIBLE_WINDING_WARNINGS:
                        row.update(feasible=False)
                        row["reasons"].append(f"{w.name}: {x.message}")
                for a in wd.assumptions:
                    result.assume(a.id, a.statement, a.basis, ["copper_loss"])
                for lim in wd.limitations:
                    result.limit(lim.code, lim.statement)
                for s in wd.sources:
                    result.cite(s)
            if complete:
                row["copper_loss_W"] = copper_total
                fill = copper_area_total / window_area
                row["window_fill"] = fill
                limit = min(w.winding.fill_factor for w in windings)
                if fill > limit:
                    row.update(feasible=False)
                    row["reasons"].append(f"window fill {fill:.2f} exceeds the fill factor {limit:.2f}")
        core_w, copper_w = row.get("core_loss_W"), row.get("copper_loss_W")
        row["total_loss_W"] = core_w + copper_w if core_w is not None and copper_w is not None else None
        if row["feasible"] and row["total_loss_W"] is None:  # defensive: an incomplete evaluation is never feasible
            row.update(feasible=False, evaluation="incomplete")
            row["reasons"].append("total loss could not be evaluated")
        candidates.append(row)

    feasible = [c for c in candidates if c["feasible"] and c.get("total_loss_W") is not None]
    feasible.sort(key=lambda c: c["total_loss_W"])
    ranked = feasible + [c for c in candidates if c not in feasible]
    result.data["candidates"] = ranked[: max(request.max_candidates, len(feasible))] if request.cores is None else ranked
    result.data["material"] = material_record.id
    result.data["flux_waveform"] = waveform.model_dump(mode="json", exclude_none=True)
    result.data["winding_currents"] = {k: {"current_rms_A": v[0], "basis": v[1], **({"power_share": shares[k]} if k in shares else {})}
                                       for k, v in currents.items()}
    if not feasible:
        reasons = sorted({reason for c in candidates for reason in c["reasons"]})
        shown = "; ".join(reasons[:6]) + (f"; ... ({len(reasons) - 6} more in data.candidates)" if len(reasons) > 6 else "")
        result.status = "rejected"
        result.errors.append(ErrorRecord(
            code="no_feasible_candidate",
            message="No candidate core satisfied the saturation, window-fill and loss-model constraints with a complete evaluation. "
                    f"Reasons: {shown}",
            details={"candidates": [c["core"] for c in candidates], "reasons": {c["core"]: c["reasons"] for c in candidates}}))
    else:
        best = feasible[0]
        result.data["selected"] = best["core"]
        for code, message in coded_warnings.get(best["core"], []):
            result.warn(code, f"{best['core_name']}: {message}")
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
        result.add_value("copper_loss", "Copper loss (all windings)", Quantity(value=best["copper_loss_W"], unit="W"), status="estimated")
        result.add_value("total_loss", "Total estimated transformer loss (core + copper)", Quantity(value=best["total_loss_W"], unit="W"),
                         status="estimated", derived_from=["core_loss", "copper_loss"])
        result.add_value("window_fill", "Window fill (bare copper / window)", Quantity(value=best["window_fill"], unit="1"), status="estimated")
        for name, (i_rms, basis) in currents.items():
            result.add_value(f"current_rms_{name}", f"RMS current of winding {name}", Quantity(value=i_rms, unit="A"),
                             status="assumed" if basis == "estimated" else "specified")
        if best.get("loss_model_identity"):
            result.model = ModelIdentity.model_validate(best["loss_model_identity"])  # the model behind the estimated loss values
        from aipe_magnetics.engineering._verification import capability_status, combine
        design_status = capability_status(None, CAP_TRANSFORMER, frequency=f, temperature=t_k,
                                          flux_density_peak=best["flux_density_peak_T"], material=material_record.id)
        loss_status = VerificationStatus.model_validate(best["loss_verification"]) if best.get("loss_verification") else None
        result.verification = combine(design_status, loss_status,
                                      design_label="Sizing relations (Faraday's law, turns ratio, window fill, Dowell factor)",
                                      model_label="Core-loss model of the selected core")
    result.limit("analytical_design", "Analytical sizing only: no finite-element, leakage-inductance, fringing, thermal, insulation or "
                 "manufacturing analysis was performed.")
    result.limit("no_hardware_validation", "No hardware was built or measured; results are design estimates, not qualification data.")
    result.limit("effective_parameters", "Core effective parameters (Ae, le, Ve, window, mean turn length) are datasheet values; "
                 "gapping, stacking and tolerances are not considered.")
    result.limit("candidate_enumeration", "Candidate cores are enumerated from the catalogue (every core offered in the material) and "
                 "ranked by total estimated loss; no area-product or thermal pre-selection is applied.")
    return result
