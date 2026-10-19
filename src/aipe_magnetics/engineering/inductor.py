"""Basic gapped-core inductor design (analytical sizing).

Candidate cores are the requested core or, with ``core=None``, **every** catalogue core offered in the
material (no area-product pre-screening; the window-fill check is the exact form of that constraint).
For each candidate:

1. energy-method turns      ``N = ceil(L I_pk / (B_allow A_e))`` (B_allow from the flux-limit policy)
2. ungapped feasibility     ``N >= ceil(sqrt(L l_e / (mu_0 mu_i A_e)))`` so a non-negative gap exists
3. air gap                  ``g = mu_0 N^2 A_e / L - l_e / mu_i`` (series reluctance, fringing neglected)
4. peak flux density        ``B_pk = L I_pk / (N A_e)``; saturation screening against Bsat(T)
5. AC flux swing            ``dB = L dI / (N A_e)`` from the ripple current; core loss with a triangular
                            flux waveform of amplitude dB/2 and the requested ``ripple_duty`` (default 0.5,
                            symmetric); DC bias is *ignored* by SE/iGSE - flagged
6. winding                  DC/AC resistance and copper loss at the RMS current; window fill
7. ranking                  feasible, completely evaluated candidates by total estimated loss

A candidate whose winding or window fill cannot be evaluated (core record without mean turn length or
window area, Dowell model failure) is reported as not feasible with the reason and is never ranked.
Assumptions and limitations are explicit in the result (initial permeability for the ungapped
reluctance, no fringing correction, no thermal model, DC-bias effect on loss ignored, ripple waveform).
"""

from __future__ import annotations

import math
from typing import Any

from aipe_magnetics.api.types import CoreLossRequest, InductorDesignRequest, OperatingPointSpec, WaveformSpec
from aipe_magnetics.common.errors import AIPEMagneticsError, DataNotFoundError, OutOfDomainError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import EngineeringResult, ErrorRecord, ModelIdentity, VerificationStatus
from aipe_magnetics.common.units import MU_0
from aipe_magnetics.data.records import CoreGeometryProperties
from aipe_magnetics.engineering import formulas as F
from aipe_magnetics.engineering.core_loss import estimate_core_loss
from aipe_magnetics.engineering.materials import evidence_sources, lookup_material
from aipe_magnetics.engineering.operating_point import validate_operating_point
from aipe_magnetics.engineering.saturation import flux_limit_from_policy, saturation_flux_density_at, screen_saturation
from aipe_magnetics.engineering.transformer import INFEASIBLE_WINDING_WARNINGS, candidate_cores
from aipe_magnetics.engineering.winding import design_winding

CAP_INDUCTOR = "aipe.magnetics.design_inductor"


def _catalog(catalog: Any | None):
    if catalog is not None:
        return catalog
    from aipe_magnetics.data.catalog import Catalog
    return Catalog.default()


def design_inductor(request: InductorDesignRequest, catalog: Any | None = None) -> EngineeringResult:
    inputs = request.model_dump(mode="json", exclude_none=True)
    cat = _catalog(catalog)
    try:
        material_record, material = lookup_material(request.material, cat)
        cores = candidate_cores(material_record, [request.core] if request.core else None, cat)
        bsat_q, bsat_status, _ = saturation_flux_density_at(material, request.temperature)
        bsat = bsat_q.to("T").value
        if material.initial_permeability is None:
            raise DataNotFoundError(f"{material_record.id} has no initial permeability; the ungapped reluctance cannot be evaluated")
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(CAP_INDUCTOR, error, inputs=inputs)

    result = EngineeringResult(capability=CAP_INDUCTOR, inputs=inputs)
    for s in evidence_sources(material_record):
        result.cite(s)
    L, i_pk, i_rms, f = request.inductance.value, request.current_peak.value, request.current_rms.value, request.frequency.value
    t_k = request.temperature.value
    mu_i = material.initial_permeability.value.value

    # material-level operating-point checks (Curie margin, recommended frequency range, loss-parameter coverage)
    op_checks = validate_operating_point(material=material, frequency=f, flux_density_peak=None, temperature=t_k, flux_limit=request.flux_limit)
    for check in op_checks.checks:
        if check.id == "saturation":
            continue  # screened per candidate with the realised peak flux density
        if check.status == "fail":
            if check.id == "curie_temperature":
                return EngineeringResult.rejected(CAP_INDUCTOR, OutOfDomainError(check.message, details={"check": check.id}), inputs=inputs)
            result.warn(check.id, check.message, "error")
        elif check.status == "warning":
            result.warn(check.id, check.message, "warning")
    result.data["operating_point_checks"] = [c.model_dump(mode="json") for c in op_checks.checks if c.id != "saturation"]

    b_allow, flux_assumption = flux_limit_from_policy(request.flux_limit, bsat)
    result.assume(flux_assumption.id, flux_assumption.statement, flux_assumption.basis, ["turns", "flux_density_peak"])
    result.assume("ungapped_permeability", f"Ungapped core reluctance uses the initial permeability mu_i = {mu_i:g}; the gap dominates "
                  "the reluctance so the error from amplitude/bias dependence of mu is small.", "model_assumption", ["air_gap"])
    result.assume("no_fringing", "Gap fringing flux is neglected (reluctance model); real inductance is somewhat higher and the gap "
                  "slightly longer for the same L.", "model_assumption", ["air_gap", "inductance"])
    if bsat_status == "estimated":
        result.assume("bsat_interpolated", f"Bsat at {t_k - 273.15:.0f} degC linearly interpolated between datasheet points.",
                      "engineering_judgement", ["flux_density_peak"])
    result.add_value("flux_density_allowed", "Allowed peak flux density (policy)", Quantity(value=b_allow, unit="T"), status="assumed")
    result.add_value("saturation_flux_density", f"Bsat at {t_k - 273.15:.0f} degC", Quantity(value=bsat, unit="T"), status=bsat_status)
    ripple = request.current_ripple_peak_to_peak.value if request.current_ripple_peak_to_peak is not None else None
    waveform = WaveformSpec(kind="triangular", duty=request.ripple_duty)
    if ripple is None:
        result.warn("no_ripple_given", "No ripple current supplied; core loss not estimated (only DC flux is known).", "info")
    else:
        result.data["flux_waveform"] = waveform.model_dump(mode="json", exclude_none=True)
        result.assume("ripple_waveform_triangular",
                      f"Core loss uses a triangular AC flux waveform with rising fraction (duty) {request.ripple_duty:g} of the period and "
                      "amplitude dB/2 from the ripple current; the iGSE result depends on this shape (an asymmetric ripple raises it).",
                      "default" if abs(request.ripple_duty - 0.5) < 1e-12 else "user_input", ["loss_density", "core_loss"])

    candidates: list[dict[str, Any]] = []
    coded_warnings: dict[str, list[tuple[str, str]]] = {}

    def cwarn(row: dict[str, Any], code: str, message: str) -> None:
        row["warnings"].append(message)
        coded_warnings.setdefault(row["core"], []).append((code, message))

    for core_record in cores:
        core = CoreGeometryProperties.model_validate(core_record.properties)
        row: dict[str, Any] = {"core": core_record.id, "core_name": core_record.name, "feasible": True, "evaluation": "complete",
                               "reasons": [], "warnings": []}
        if core.effective_area is None or core.effective_length is None or core.effective_volume is None:
            row.update(feasible=False, evaluation="incomplete", reasons=["core record lacks effective area/length/volume"])
            candidates.append(row)
            continue
        ae, le, ve = core.effective_area.value, core.effective_length.value, core.effective_volume.value
        row.update(effective_area_m2=ae, effective_length_m=le, effective_volume_m3=ve, mass_kg=core.mass.value if core.mass else None)
        try:
            n_energy = F.turns_for_energy_limit(L, i_pk, b_allow, ae)
            n_min_ungapped = int(math.ceil(math.sqrt(L * le / (MU_0 * mu_i * ae))))
            n = max(n_energy, n_min_ungapped)
            gap = F.air_gap_for_inductance(L, n, ae, le, mu_i)
            b_pk = F.flux_density_from_current(L, i_pk, n, ae)
        except AIPEMagneticsError as error:
            row.update(feasible=False, evaluation="incomplete", reasons=[error.message])
            candidates.append(row)
            continue
        row.update(turns=n, turns_energy_method=n_energy, turns_min_ungapped=n_min_ungapped, air_gap_m=gap, flux_density_peak_T=b_pk)
        if gap > 0.1 * le:
            cwarn(row, "large_air_gap", f"air gap {gap * 1e3:.2f} mm exceeds 10 % of the magnetic path; fringing will be significant")
        screening = screen_saturation(flux_density_peak=b_pk, temperature=t_k, material=material, policy=request.flux_limit)
        row["saturation_margin"] = screening.margin_to_saturation
        if screening.saturated:
            row.update(feasible=False)
            row["reasons"].append("core saturates at the peak current")
        # core loss from the AC flux swing
        row["core_loss_W"] = None
        if ripple is not None and ripple > 0:
            db = L * ripple / (n * ae)
            row["flux_density_ac_amplitude_T"] = db / 2.0
            loss_req = CoreLossRequest(material=material_record.id, core=core_record.id, model=request.loss_model, policy=request.policy,
                                       operating_point=OperatingPointSpec(frequency=request.frequency, flux_density_peak=Quantity(value=db / 2.0, unit="T"),
                                                                          temperature=request.temperature, waveform=waveform))
            loss = estimate_core_loss(loss_req, catalog=cat)
            if loss.status == "ok":
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
                result.limit("dc_bias_ignored_in_loss", "The core-loss estimate uses the AC flux swing only; the DC bias of an inductor "
                             "increases ferrite losses and is not captured by Steinmetz/iGSE.")
            else:
                row.update(feasible=False)
                row["reasons"].append("core loss: " + "; ".join(e.message for e in loss.errors))
        # winding and window fill (mean turn length and window area are required for a complete evaluation)
        mlt = core.mean_turn_length.value if core.mean_turn_length else None
        window = core.bobbin_window_area or core.window_area
        window_area = window.value if window else None
        row["copper_loss_W"] = None
        row["window_fill"] = None
        missing = [name for name, value in (("mean turn length", mlt), ("window area", window_area)) if value is None]
        if missing:
            row.update(feasible=False, evaluation="incomplete")
            row["reasons"].append(f"not evaluable: core record has no {' and no '.join(missing)}; copper loss and window fill cannot be assessed")
        else:
            dims = {k: q.value for k, q in core.dimensions.values.items()} if core.dimensions else None
            try:
                wd = design_winding(turns=n, current_rms=i_rms, frequency=f, mean_turn_length=mlt, spec=request.winding,
                                    temperature=t_k, window_area=window_area, dimensions=dims)
            except AIPEMagneticsError as error:
                row.update(feasible=False, evaluation="incomplete")
                row["reasons"].append(f"winding not evaluable: {error.message}")
            else:
                row["winding"] = {"conductor_type": wd.conductor_type, "conductor_area_m2": wd.conductor_area, "layers": wd.layers,
                                  "resistance_dc_Ohm": wd.resistance_dc, "ac_resistance_factor": wd.ac_resistance_factor,
                                  "copper_loss_W": wd.copper_loss, "window_fill": wd.window_fill}
                row["copper_loss_W"] = wd.copper_loss
                row["window_fill"] = wd.window_fill
                for x in wd.warnings:
                    cwarn(row, x.code, x.message)
                    if x.code in INFEASIBLE_WINDING_WARNINGS:
                        row.update(feasible=False)
                        row["reasons"].append(x.message)
                for a in wd.assumptions:
                    result.assume(a.id, a.statement, a.basis, ["copper_loss"])
                for lim in wd.limitations:
                    result.limit(lim.code, lim.statement)
                for s in wd.sources:
                    result.cite(s)
                if wd.window_fill is not None and wd.window_fill > request.winding.fill_factor:
                    row.update(feasible=False)
                    row["reasons"].append(f"window fill {wd.window_fill:.2f} exceeds the fill factor {request.winding.fill_factor:.2f}")
                result.limit("ac_copper_loss_fundamental_only", "Copper loss uses the RMS current with the Dowell factor at the switching "
                             "frequency; the DC component of an inductor current sees R_dc only, so this is conservative for high-DC designs.")
        losses = [x for x in (row.get("core_loss_W"), row.get("copper_loss_W")) if x is not None]
        row["total_loss_W"] = sum(losses) if losses else None
        row["total_loss_scope"] = ("core + copper" if len(losses) == 2 else "copper only (no ripple given)" if row.get("copper_loss_W") is not None
                                   else None)
        if row["feasible"] and row.get("copper_loss_W") is None:  # defensive: an incomplete evaluation is never feasible
            row.update(feasible=False, evaluation="incomplete")
            row["reasons"].append("copper loss could not be evaluated")
        candidates.append(row)

    feasible = [c for c in candidates if c["feasible"] and c.get("total_loss_W") is not None]
    feasible.sort(key=lambda c: (c["total_loss_W"], c.get("mass_kg") or 0.0))
    ranked = feasible + [c for c in candidates if c not in feasible]
    result.data["candidates"] = ranked[: max(request.max_candidates, len(feasible))] if request.core is None else ranked
    result.data["material"] = material_record.id
    if not feasible:
        reasons = sorted({reason for c in candidates for reason in c["reasons"]})
        shown = "; ".join(reasons[:6]) + (f"; ... ({len(reasons) - 6} more in data.candidates)" if len(reasons) > 6 else "")
        result.status = "rejected"
        result.errors.append(ErrorRecord(code="no_feasible_candidate",
                                         message="No candidate core satisfied the saturation, window-fill and loss-model constraints with a "
                                                 f"complete evaluation. Reasons: {shown}",
                                         details={"candidates": [c["core"] for c in candidates], "reasons": {c["core"]: c["reasons"] for c in candidates}}))
    else:
        best = feasible[0]
        result.data["selected"] = best["core"]
        for code, message in coded_warnings.get(best["core"], []):
            result.warn(code, f"{best['core_name']}: {message}")
        result.add_value("turns", "Turns", Quantity(value=float(best["turns"]), unit="1"), status="calculated",
                         derived_from=["inductance", "current_peak", "flux_density_allowed", "effective_area"])
        result.add_value("air_gap", "Total air gap length (reluctance model)", Quantity(value=best["air_gap_m"], unit="m"), status="calculated",
                         derived_from=["turns", "inductance", "effective_length"])
        result.add_value("flux_density_peak", "Peak flux density at I_pk", Quantity(value=best["flux_density_peak_T"], unit="T"), status="calculated")
        result.add_value("saturation_margin", "Margin to saturation", Quantity(value=best["saturation_margin"], unit="1"), status="calculated")
        if best.get("flux_density_ac_amplitude_T") is not None:
            result.add_value("flux_density_ac_amplitude", "AC flux density amplitude from ripple", Quantity(value=best["flux_density_ac_amplitude_T"], unit="T"), status="calculated")
        total_label = "Total estimated loss (core + copper)" if best.get("core_loss_W") is not None else "Total estimated loss (copper only; no ripple given)"
        for key, label, unit in (("loss_density_W_per_m3", "Core loss density (AC swing)", "W/m^3"), ("core_loss_W", "Core loss", "W"),
                                 ("copper_loss_W", "Copper loss", "W"), ("total_loss_W", total_label, "W"), ("window_fill", "Window fill", "1")):
            if best.get(key) is not None:
                result.add_value(key.replace("_W_per_m3", "").replace("_W", "").replace("_m", ""), label, Quantity(value=best[key], unit=unit), status="estimated")
        if best.get("loss_model_identity"):
            result.model = ModelIdentity.model_validate(best["loss_model_identity"])  # the model behind the estimated loss values
        from aipe_magnetics.engineering._verification import capability_status, combine
        design_status = capability_status(None, CAP_INDUCTOR, frequency=f, temperature=t_k,
                                          flux_density_peak=best["flux_density_peak_T"], material=material_record.id)
        loss_status = VerificationStatus.model_validate(best["loss_verification"]) if best.get("loss_verification") else None
        result.verification = combine(design_status, loss_status,
                                      design_label="Sizing relations (reluctance model, energy method, Dowell factor)",
                                      model_label="Core-loss model of the selected core")
    result.limit("analytical_design", "Analytical sizing only: no finite-element, fringing, thermal, insulation or manufacturing analysis was performed.")
    result.limit("no_hardware_validation", "No hardware was built or measured; results are design estimates, not qualification data.")
    result.limit("candidate_enumeration", "Candidate cores are enumerated from the catalogue (every core offered in the material) and "
                 "ranked by total estimated loss; no area-product or thermal pre-selection is applied.")
    return result
