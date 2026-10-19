"""Design validation: saturation screening, operating-point checks, loss-model domain checks for an existing design."""

from __future__ import annotations

from typing import Any

from aipe_magnetics.api.types import CoreLossRequest, DesignValidationRequest, OperatingPointSpec
from aipe_magnetics.common.errors import AIPEMagneticsError, DataNotFoundError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import EngineeringResult
from aipe_magnetics.common.units import MU_0
from aipe_magnetics.data.records import CoreGeometryProperties
from aipe_magnetics.engineering import formulas as F
from aipe_magnetics.engineering.core_loss import estimate_core_loss
from aipe_magnetics.engineering.materials import evidence_sources, lookup_material
from aipe_magnetics.engineering.operating_point import validate_operating_point
from aipe_magnetics.engineering.transformer import flux_waveform_for

CAP_VALIDATE = "aipe.magnetics.validate_design"


def _catalog(catalog: Any | None):
    if catalog is not None:
        return catalog
    from aipe_magnetics.data.catalog import Catalog
    return Catalog.default()


def validate_design(request: DesignValidationRequest, catalog: Any | None = None) -> EngineeringResult:
    inputs = request.model_dump(mode="json", exclude_none=True)
    cat = _catalog(catalog)
    try:
        material_record, material = lookup_material(request.material, cat)
        core_record = cat.get(request.core)
        if core_record.kind != "core_geometry":
            raise DataNotFoundError(f"{request.core} is not a core_geometry record")
        core = CoreGeometryProperties.model_validate(core_record.properties)
        if core.effective_area is None:
            raise DataNotFoundError(f"core {request.core} has no effective area")
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(CAP_VALIDATE, error, inputs=inputs)

    result = EngineeringResult(capability=CAP_VALIDATE, inputs=inputs)
    for s in evidence_sources(material_record) + evidence_sources(core_record):
        result.cite(s)
    f, t_k, ae = request.frequency.value, request.temperature.value, core.effective_area.value
    # peak flux density: from voltage/turns (Faraday) or directly specified
    if request.flux_density_peak is not None:
        b_pk = request.flux_density_peak.value
        result.add_value("flux_density_peak", "Peak flux density (specified)", Quantity(value=b_pk, unit="T"), status="looked_up")
    else:
        v = request.voltage_peak.value  # type: ignore[union-attr]
        if request.voltage_waveform == "sinusoidal":
            b_pk = F.flux_density_peak_sine(v, f, request.turns, ae)
        else:
            b_pk = F.flux_density_peak_square(v, f, request.turns, ae, request.duty)
        result.add_value("flux_density_peak", "Peak flux density from volt-seconds", Quantity(value=b_pk, unit="T"), status="calculated",
                         derived_from=["voltage_peak", "frequency", "turns", "effective_area"])
    # DC/current-driven flux check through the gapped reluctance
    if request.current_peak is not None and material.initial_permeability is not None and core.effective_length is not None:
        gap = request.air_gap.value if request.air_gap is not None else 0.0
        b_dc = MU_0 * request.turns * request.current_peak.value / (gap + core.effective_length.value / material.initial_permeability.value.value)
        result.add_value("flux_density_from_current", "Flux density from N*I through the (gapped) reluctance", Quantity(value=b_dc, unit="T"),
                         status="estimated", derived_from=["current_peak", "turns", "air_gap"])
        result.assume("reluctance_model", "Current-driven flux uses mu_0 N I / (g + l_e/mu_i) without fringing.", "model_assumption",
                      ["flux_density_from_current"])
        b_check = max(b_pk, b_dc)
    else:
        b_check = b_pk
    validation = validate_operating_point(material=material, frequency=f, flux_density_peak=b_check, temperature=t_k,
                                          flux_limit=request.flux_limit)
    result.data["checks"] = [c.model_dump(mode="json") for c in validation.checks]
    if validation.saturation is not None:
        for key, (label, q, status) in validation.saturation.quantities().items():
            result.add_value(key, label, q, status=status)
        for a in validation.saturation.assumptions:
            result.assume(a.id, a.statement, a.basis, ["saturation_margin"])
        for lim in validation.saturation.limitations:
            result.limit(lim.code, lim.statement)
    for check in validation.checks:
        if check.status == "fail":
            result.warn(check.id, check.message, "error")
        elif check.status == "warning":
            result.warn(check.id, check.message, "warning")
        elif check.status == "not_evaluated":
            result.warn(check.id, check.message, "info")
    # loss estimate with domain warnings
    waveform = flux_waveform_for(request.voltage_waveform, request.duty)
    loss = estimate_core_loss(CoreLossRequest(material=material_record.id, core=request.core, model=request.loss_model, policy="warn",
                                              operating_point=OperatingPointSpec(frequency=request.frequency, flux_density_peak=Quantity(value=b_pk, unit="T"),
                                                                                 temperature=request.temperature, waveform=waveform)), catalog=cat)
    result.data["core_loss"] = loss.to_json_dict()
    if loss.status == "ok":
        result.add_value("loss_density", "Core loss density", loss.values["loss_density"].quantity, status="estimated")
        if "core_loss" in loss.values:
            result.add_value("core_loss", "Core loss", loss.values["core_loss"].quantity, status="estimated")
        result.model = loss.model
        for w in loss.warnings:
            result.warn(w.code, f"core loss: {w.message}", w.severity)
        for lim in loss.limitations:
            result.limit(lim.code, lim.statement)
        result.verification = loss.verification
    else:
        result.warn("core_loss_unavailable", "core loss not evaluated: " + "; ".join(e.message for e in loss.errors), "warning")
    failed = [c.id for c in validation.checks if c.status == "fail"]
    result.data["passed"] = not failed
    result.data["failed_checks"] = failed
    if failed:
        result.status = "rejected"
        from aipe_magnetics.common.results import ErrorRecord
        result.errors.append(ErrorRecord(code="design_checks_failed", message=f"failed checks: {', '.join(failed)}", details={"failed": failed}))
    result.limit("validation_scope", "Validation covers saturation, Curie margin, recommended frequency range and loss-model coverage; "
                 "thermal, leakage, insulation and mechanical aspects are not validated.")
    return result
