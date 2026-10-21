"""Stable, typed Python API of AIPE Magnetics (usable without any AI agent).

Every function accepts either a request object from :mod:`aipe_magnetics.api.types` or the
equivalent keyword arguments, validates them, and returns an
:class:`~aipe_magnetics.common.results.EngineeringResult`. Engineering rejections (invalid or
out-of-domain inputs, missing data) are returned as ``status="rejected"`` results with explicit
error records; programming errors still raise.

Capabilities
------------
``list_materials``, ``get_material``, ``compare_materials``, ``estimate_core_loss``,
``evaluate_hysteresis``, ``design_inductor``, ``design_transformer``, ``design_transformer_from_waveforms``,
``validate_design``, ``measurement_requests``, ``generate_report``, ``verification_status``, ``list_capabilities``.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from aipe_magnetics import __version__
from aipe_magnetics.api.types import (
    CoreLossRequest,
    DesignValidationRequest,
    HysteresisRequest,
    InductorDesignRequest,
    MeasurementRequestSpec,
    OperatingPointSpec,
    ReportRequest,
    TransformerDesignRequest,
    WaveformInformedTransformerRequest,
    WaveformSpec,
)
from aipe_magnetics.api.types import _q as _typed_q
from aipe_magnetics.common.errors import AIPEMagneticsError, InvalidInputError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import EngineeringResult

__all__ = [
    "list_materials", "get_material", "compare_materials", "estimate_core_loss", "evaluate_hysteresis",
    "design_inductor", "design_transformer", "design_transformer_from_waveforms", "validate_design", "measurement_requests",
    "generate_report", "verification_status", "list_capabilities", "CAPABILITIES", "__version__",
]

CAPABILITIES: dict[str, dict[str, str]] = {
    "list_materials": {"id": "aipe.magnetics.list_materials", "kind": "read_only",
                       "description": "List catalogue materials (optionally filtered by query or class)."},
    "get_material": {"id": "aipe.magnetics.get_material", "kind": "read_only",
                     "description": "Datasheet parameters, Steinmetz sets and sources of one material."},
    "compare_materials": {"id": "aipe.magnetics.compare_materials", "kind": "deterministic",
                          "description": "Compare materials at an operating point (Steinmetz loss, saturation margin)."},
    "estimate_core_loss": {"id": "aipe.magnetics.estimate_core_loss", "kind": "deterministic",
                           "description": "Core-loss density (and core loss) with Steinmetz, iGSE or Bertotti models."},
    "evaluate_hysteresis": {"id": "aipe.magnetics.evaluate_hysteresis", "kind": "deterministic",
                            "description": "Jiles-Atherton B-H loop evaluation through the ODE framework."},
    "design_inductor": {"id": "aipe.magnetics.design_inductor", "kind": "deterministic",
                        "description": "Basic gapped inductor sizing: turns, gap, flux density, losses."},
    "design_transformer": {"id": "aipe.magnetics.design_transformer", "kind": "deterministic",
                           "description": "Basic transformer sizing: candidate cores, turns, flux density, losses."},
    "design_transformer_from_waveforms": {"id": "aipe.magnetics.design_transformer_from_waveforms", "kind": "deterministic",
                                          "description": "Waveform-informed transformer design: actual (sampled or SPS-synthesised) winding waveforms -> "
                                                         "exact RMS / harmonics, flux from the voltage integral, iGSE core loss, per-layer harmonic Dowell "
                                                         "copper loss, leakage / magnetising inductance, fill, thermal estimate; ranked integer-turn "
                                                         "candidates, square-current comparison, material comparison, optional Core state export."},
    "validate_design": {"id": "aipe.magnetics.validate_design", "kind": "deterministic",
                        "description": "Saturation screening and operating-point validation of a design."},
    "measurement_requests": {"id": "aipe.magnetics.measurement_requests", "kind": "read_only",
                             "description": "Structured magnetic-material measurement requests (B-H, core loss vs f/B/T, DC bias, saturation vs T, "
                                            "complex permeability, winding AC resistance, thermal conductivity) with test conditions and priority."},
    "generate_report": {"id": "aipe.magnetics.generate_report", "kind": "write_optional",
                        "description": "Render Markdown/JSON engineering reports (writing requires authorisation)."},
    "verification_status": {"id": "aipe.magnetics.verification_status", "kind": "read_only",
                            "description": "Verification evidence per capability and operating range."},
}


def _coerce(model: type[BaseModel], request: Any, kwargs: dict[str, Any]) -> BaseModel:
    if request is not None and kwargs:
        raise InvalidInputError("pass either a request object or keyword arguments, not both")
    if isinstance(request, model):
        return request
    data = request if isinstance(request, dict) else kwargs
    try:
        return model.model_validate(data)
    except ValidationError as error:
        raise InvalidInputError(f"invalid {model.__name__}: " + "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors())) from None


def _guard(capability: str, func: Callable[[], EngineeringResult], inputs: dict[str, Any] | None = None) -> EngineeringResult:
    try:
        return func()
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(capability, error, inputs=inputs)


def _q(value: Any, unit: str, name: str) -> Quantity | None:
    """Dimension-checked, SI-normalised optional quantity (the same rules as the typed request models)."""
    return _typed_q(value, unit, name, optional=True)


# -------------------------------------------------------------------------------------------------
# materials
# -------------------------------------------------------------------------------------------------
def list_materials(query: str | None = None, material_class: str | None = None, *, catalog: Any | None = None) -> EngineeringResult:
    from aipe_magnetics.engineering import materials as m
    return _guard(m.CAP_LIST, lambda: m.list_materials(query, material_class, catalog), {"query": query})


def get_material(material: str, *, catalog: Any | None = None) -> EngineeringResult:
    from aipe_magnetics.engineering import materials as m
    return _guard(m.CAP_GET, lambda: m.get_material(material, catalog), {"material": material})


def compare_materials(materials: list[str], frequency: Any = None, flux_density_peak: Any = None, temperature: Any = None, *,
                      catalog: Any | None = None) -> EngineeringResult:
    from aipe_magnetics.engineering import materials as m

    items = list(materials) if isinstance(materials, (list, tuple, set)) else [materials]

    def run() -> EngineeringResult:
        return m.compare_materials(items, _q(frequency, "Hz", "frequency"), _q(flux_density_peak, "T", "flux_density_peak"),
                                   _q(temperature, "K", "temperature"), catalog)

    return _guard(m.CAP_COMPARE, run, {"materials": items})


# -------------------------------------------------------------------------------------------------
# models
# -------------------------------------------------------------------------------------------------
def estimate_core_loss(request: CoreLossRequest | dict | None = None, *, catalog: Any | None = None, **kwargs: Any) -> EngineeringResult:
    """``estimate_core_loss(material="N87", operating_point={...})`` or
    ``estimate_core_loss(material="N87", frequency={"value": 100, "unit": "kHz"}, flux_density_peak=..., temperature=..., waveform={...})``."""
    from aipe_magnetics.engineering import core_loss as cl

    if request is None and "operating_point" not in kwargs:
        op_keys = ("frequency", "flux_density_peak", "temperature", "waveform")
        op = {k: kwargs.pop(k) for k in op_keys if k in kwargs}
        if "waveform" in op and isinstance(op["waveform"], str):
            op["waveform"] = {"kind": op["waveform"]}
        kwargs["operating_point"] = op
    try:
        req = _coerce(CoreLossRequest, request, kwargs)
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(cl.CAP_CORE_LOSS, error, inputs=kwargs)
    return _guard(cl.CAP_CORE_LOSS, lambda: cl.estimate_core_loss(req, catalog=catalog), req.model_dump(mode="json", exclude_none=True))


def evaluate_hysteresis(request: HysteresisRequest | dict | None = None, **kwargs: Any) -> EngineeringResult:
    from aipe_magnetics.engineering import hysteresis as hy

    try:
        req = _coerce(HysteresisRequest, request, kwargs)
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(hy.CAP_HYSTERESIS, error, inputs=kwargs)
    return _guard(hy.CAP_HYSTERESIS, lambda: hy.evaluate_hysteresis(req), req.model_dump(mode="json", exclude_none=True))


# -------------------------------------------------------------------------------------------------
# design
# -------------------------------------------------------------------------------------------------
def design_inductor(request: InductorDesignRequest | dict | None = None, *, catalog: Any | None = None, **kwargs: Any) -> EngineeringResult:
    from aipe_magnetics.engineering import inductor as ind

    try:
        req = _coerce(InductorDesignRequest, request, kwargs)
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(ind.CAP_INDUCTOR, error, inputs=kwargs)
    return _guard(ind.CAP_INDUCTOR, lambda: ind.design_inductor(req, catalog=catalog), req.model_dump(mode="json", exclude_none=True))


def design_transformer(request: TransformerDesignRequest | dict | None = None, *, catalog: Any | None = None, **kwargs: Any) -> EngineeringResult:
    from aipe_magnetics.engineering import transformer as tr

    try:
        req = _coerce(TransformerDesignRequest, request, kwargs)
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(tr.CAP_TRANSFORMER, error, inputs=kwargs)
    return _guard(tr.CAP_TRANSFORMER, lambda: tr.design_transformer(req, catalog=catalog), req.model_dump(mode="json", exclude_none=True))


def design_transformer_from_waveforms(request: WaveformInformedTransformerRequest | dict | None = None, *, created_at: str | None = None,
                                      export_state: bool = False, **kwargs: Any) -> EngineeringResult:
    """Design a transformer from the converter's actual waveforms (HDF5 / CSV / JSON) or SPS parameters.

    ``created_at`` (RFC 3339, supplied by the caller) is recorded as ``produced_at`` and is required when
    ``export_state`` asks for the AIPE-Core Engineering State of the selected design (``data.engineering_state``).
    """
    from aipe_magnetics.design.transformers import waveform_informed as wd

    try:
        req = _coerce(WaveformInformedTransformerRequest, request, kwargs)
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(wd.CAP_WAVEFORM_DESIGN, error, inputs=kwargs)
    return _guard(wd.CAP_WAVEFORM_DESIGN, lambda: wd.design_transformer_from_waveforms(req, created_at=created_at, export_state=export_state),
                  req.model_dump(mode="json", exclude_none=True))


def validate_design(request: DesignValidationRequest | dict | None = None, *, catalog: Any | None = None, **kwargs: Any) -> EngineeringResult:
    from aipe_magnetics.engineering import design_validation as dv

    try:
        req = _coerce(DesignValidationRequest, request, kwargs)
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(dv.CAP_VALIDATE, error, inputs=kwargs)
    return _guard(dv.CAP_VALIDATE, lambda: dv.validate_design(req, catalog=catalog), req.model_dump(mode="json", exclude_none=True))


def measurement_requests(request: MeasurementRequestSpec | dict | None = None, *, catalog: Any | None = None,
                         winding_context: dict[str, Any] | None = None, **kwargs: Any) -> EngineeringResult:
    """Magnetic-material measurement requests (missing / partial catalogue data for an excitation) as structured JSON."""
    from aipe_magnetics.engineering import measurement_requests as mr

    try:
        req = _coerce(MeasurementRequestSpec, request, kwargs)
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(mr.CAP_MEASUREMENT_REQUESTS, error, inputs=kwargs)
    return _guard(mr.CAP_MEASUREMENT_REQUESTS, lambda: mr.measurement_requests(req, catalog, winding_context=winding_context),
                  req.model_dump(mode="json", exclude_none=True))


# -------------------------------------------------------------------------------------------------
# reporting and verification
# -------------------------------------------------------------------------------------------------
def generate_report(results: list[EngineeringResult] | EngineeringResult, request: ReportRequest | dict | None = None, *,
                    allowed_directory: Any | None = None, **kwargs: Any) -> EngineeringResult:
    from aipe_magnetics.engineering import report as rp

    items = [results] if isinstance(results, EngineeringResult) else list(results)
    try:
        req = _coerce(ReportRequest, request, kwargs)
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(rp.CAP_REPORT, error, inputs=kwargs)
    return _guard(rp.CAP_REPORT, lambda: rp.generate_report(items, req, allowed_directory=allowed_directory))


def verification_status(capability: str | None = None, *, frequency: Any = None, temperature: Any = None,
                        flux_density_peak: Any = None, material: str | None = None, excitation: str | None = None) -> EngineeringResult:
    """Verification evidence (per capability and operating range) from the bundled release evidence file.

    Without a capability the envelope summarises the registry; its verification block then reports whether the
    bundled evidence file exists at all (``unverified`` with an explicit statement when it does not).
    """
    from aipe_magnetics.verification.suite import bundled_evidence_path, load_bundled_registry

    def run() -> EngineeringResult:
        registry = load_bundled_registry()
        result = EngineeringResult(capability="aipe.magnetics.verification_status",
                                   inputs={"capability": capability, "material": material, "excitation": excitation})
        path = bundled_evidence_path()
        result.data["evidence_file"] = str(path)
        result.data["evidence_file_exists"] = path.exists()
        result.data["capabilities"] = registry.summary()
        result.data["record_count"] = len(registry.records)
        if not path.exists():
            result.warn("bundled_evidence_missing", f"The bundled evidence file {path} is missing; every capability reports L0 unverified "
                        "until a maintainer records it with 'aipe-magnetics verify --update-bundled'.", "warning")
        if capability:
            f, t, b = _q(frequency, "Hz", "frequency"), _q(temperature, "K", "temperature"), _q(flux_density_peak, "T", "flux_density_peak")
            status = registry.status_for(capability, frequency=f.value if f else None, temperature=t.value if t else None,
                                         flux_density_peak=b.value if b else None, material=material, excitation=excitation)
            result.verification = status
            result.data["records"] = [r.model_dump(mode="json", exclude_none=True) for r in registry.for_capability(capability)]
        elif registry.records:
            result.verification.level = "L1"
            result.verification.state = "verified"
            result.verification.statement = (f"Summary of {len(registry.records)} recorded evidence record(s); query a capability for a "
                                             "range-specific status.")
        else:
            result.verification.level = "L0"
            result.verification.state = "unverified"
            result.verification.statement = ("No verification evidence is recorded" +
                                             (f": the bundled evidence file {path} is missing." if not path.exists() else " (empty registry)."))
        result.limit("evidence_scope", "Verification evidence covers software, numerical and physical consistency (L1/L2) unless an L3 "
                     "record states otherwise; no hardware qualification is implied.")
        return result

    return _guard("aipe.magnetics.verification_status", run, {"capability": capability})


def list_capabilities() -> EngineeringResult:
    result = EngineeringResult(capability="aipe.magnetics.list_capabilities")
    result.data["capabilities"] = CAPABILITIES
    result.data["tool_version"] = __version__
    result.verification.level = "L1"
    result.verification.state = "verified"
    result.verification.statement = "Static capability table."
    return result
