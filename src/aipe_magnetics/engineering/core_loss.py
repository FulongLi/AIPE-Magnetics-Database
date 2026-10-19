"""Core-loss estimation capability: material + operating point + waveform -> loss density / core loss.

Model selection (``model="auto"``): classical Steinmetz for sinusoidal flux, iGSE otherwise, both
parameterised from the material's Steinmetz parameter set that covers the requested temperature.
No parameter set covering the temperature → the request is *rejected* with the available
temperatures listed (no silent nearest-neighbour substitution). Bertotti loss separation requires
user-supplied coefficients (``model_parameters``) because no catalogue material carries them.
"""

from __future__ import annotations

from typing import Any

from aipe_magnetics.api.types import CoreLossRequest, WaveformSpec
from aipe_magnetics.common.errors import (
    AIPEMagneticsError,
    DataNotFoundError,
    InvalidInputError,
    OutOfDomainError,
    UnsupportedOperationError,
)
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import EngineeringResult, SourceReference
from aipe_magnetics.data.records import CoreGeometryProperties, SteinmetzParameterSet
from aipe_magnetics.engineering.materials import evidence_sources, lookup_material

CAP_CORE_LOSS = "aipe.magnetics.estimate_core_loss"


def build_waveform(spec: WaveformSpec, frequency: float, flux_density_peak: float):
    """Translate an API waveform spec into a :class:`~aipe_magnetics.models.waveforms.Waveform` (SI)."""
    from aipe_magnetics.models.waveforms import Waveform

    if spec.kind == "sinusoidal":
        return Waveform.sinusoidal(flux_density_peak, frequency)
    if spec.kind == "triangular":
        return Waveform.triangular(flux_density_peak, frequency, duty=spec.duty if spec.duty is not None else 0.5)
    if spec.kind == "trapezoidal":
        return Waveform.trapezoidal(flux_density_peak, frequency, duty_rise=spec.duty_rise, duty_flat=spec.duty_flat)  # type: ignore[arg-type]
    if spec.kind == "piecewise_linear":
        points = [(float(t), float(b) * flux_density_peak) for t, b in spec.breakpoints or []]
        return Waveform.piecewise_linear(frequency, points)
    if spec.kind == "sampled":
        return Waveform.sampled(frequency, spec.samples or [])
    raise InvalidInputError(f"unsupported waveform kind {spec.kind}")  # pragma: no cover


def _user_parameter_set(params: dict[str, float], temperature_K: float | None) -> SteinmetzParameterSet:
    missing = [k for k in ("k", "alpha", "beta") if k not in params]
    if missing:
        raise InvalidInputError(f"model_parameters for steinmetz/igse need k, alpha, beta (missing: {', '.join(missing)})")
    return SteinmetzParameterSet(
        id="set.user-supplied", k=Quantity(value=float(params["k"]), unit="W/m^3"), alpha=float(params["alpha"]),
        beta=float(params["beta"]), temperature=None if temperature_K is None else Quantity(value=temperature_K, unit="K"),
        basis="publication", source_ref=None, note="user-supplied coefficients (SI: W/m^3, Hz, T)")


def select_model(request: CoreLossRequest, props, material_id: str, sources: list[SourceReference]):
    """Return ``(model, parameter_set | None, selection_notes)`` for the request."""
    from aipe_magnetics.models.empirical.igse import IGSEModel
    from aipe_magnetics.models.empirical.steinmetz import SteinmetzModel, select_parameter_set

    op = request.operating_point
    t_k = None if op.temperature is None else op.temperature.value
    notes: list[str] = []
    name = request.model
    if name == "auto":
        name = "steinmetz" if op.waveform.kind == "sinusoidal" else "igse"
        notes.append(f"model 'auto' resolved to '{name}' for a {op.waveform.kind} flux waveform")
    if name == "bertotti":
        from aipe_magnetics.models.physics.bertotti import BertottiLossModel
        if not request.model_parameters:
            raise UnsupportedOperationError("Bertotti loss separation needs model_parameters (k_h, beta_h, k_c, k_e); "
                                            "catalogue materials carry no Bertotti coefficients")
        model = BertottiLossModel.from_parameters(request.model_parameters, material_id=material_id, sources=sources)
        notes.append("Bertotti coefficients supplied by the user")
        return model, None, notes
    if request.model_parameters:
        pset = _user_parameter_set(request.model_parameters, t_k)
        notes.append("Steinmetz coefficients supplied by the user (not from the catalogue)")
    else:
        sets = list(props.steinmetz_parameter_sets)
        if request.parameter_set:
            sets = [s for s in sets if s.id == request.parameter_set]
            if not sets:
                raise DataNotFoundError(f"parameter set {request.parameter_set!r} not found for {material_id}")
        if not sets:
            raise DataNotFoundError(f"{material_id} has no Steinmetz parameter set; core loss cannot be estimated")
        pset = select_parameter_set(sets, t_k)
        notes.append(f"parameter set {pset.id} selected for temperature "
                     f"{'unspecified' if t_k is None else f'{t_k - 273.15:.0f} degC'}")
    cls = SteinmetzModel if name == "steinmetz" else IGSEModel
    model = cls.from_parameter_set(pset, material_id=material_id, sources=sources)
    return model, pset, notes


def estimate_core_loss(request: CoreLossRequest, catalog: Any | None = None, registry: Any | None = None) -> EngineeringResult:
    """Capability implementation returning an :class:`EngineeringResult` (never raises for engineering rejections)."""
    inputs = request.model_dump(mode="json", exclude_none=True)
    try:
        record, props = lookup_material(request.material, catalog)
        sources = evidence_sources(record)
        model, pset, notes = select_model(request, props, record.id, sources)
        op = request.operating_point
        f, bpk = op.frequency.value, op.flux_density_peak.value
        t_k = None if op.temperature is None else op.temperature.value
        waveform = build_waveform(op.waveform, f, bpk)
        kwargs: dict[str, Any] = {}
        if model.identity.id.endswith(".igse"):
            kwargs["waveform"] = waveform
        elif op.waveform.kind != "sinusoidal":
            raise OutOfDomainError(f"model {model.identity.id} supports sinusoidal flux only; use model='igse' for "
                                   f"{op.waveform.kind} waveforms")
        output = model.loss_density(frequency=f, flux_density_peak=bpk, temperature=t_k, policy=request.policy, **kwargs)
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(CAP_CORE_LOSS, error, inputs=inputs)

    result = EngineeringResult(capability=CAP_CORE_LOSS, inputs=inputs, model=output.model)
    pv = output.values["loss_density"]
    result.add_value("loss_density", "Volumetric core loss density", pv, status="estimated",
                     derived_from=["frequency", "flux_density_peak", "temperature", "waveform"])
    result.data["model"] = model.describe()
    result.data["domain_check"] = output.domain.model_dump(mode="json")
    result.data["diagnostics"] = _jsonable(output.diagnostics)
    result.data["waveform"] = waveform.describe() if hasattr(waveform, "describe") else {"kind": op.waveform.kind}
    if pset is not None:
        result.data["parameter_set"] = pset.model_dump(mode="json", exclude_none=True)
        result.assume("parameter_set_selection", "; ".join(notes) + f". Set basis: {pset.basis}; validity "
                      f"f {pset.frequency_range.model_dump(mode='json', exclude_none=True) if pset.frequency_range else 'unspecified'}, "
                      f"B {pset.flux_density_range.model_dump(mode='json', exclude_none=True) if pset.flux_density_range else 'unspecified'}.",
                      "user_input" if request.model_parameters else "datasheet", ["loss_density"])
    else:
        result.assume("model_selection", "; ".join(notes), "user_input", ["loss_density"])
    for a in output.assumptions:
        result.assume(a.id, a.statement, a.basis, a.affects or ["loss_density"])
    for lim in output.limitations:
        result.limit(lim.code, lim.statement)
    for w in output.warnings:
        result.warn(w.code, w.message, w.severity)
    for s in output.sources + sources:
        result.cite(s)
    if request.core:
        try:
            core_record = _catalog(catalog).get(request.core)
            if core_record.kind != "core_geometry":
                raise DataNotFoundError(f"{request.core} is not a core_geometry record")
            core = CoreGeometryProperties.model_validate(core_record.properties)
            if core.effective_volume is None:
                raise DataNotFoundError(f"core {request.core} has no effective volume")
            p_core = pv.value * core.effective_volume.value
            result.add_value("core_loss", f"Core loss in {core_record.name}", Quantity(value=p_core, unit="W"),
                             status="estimated", derived_from=["loss_density", "core"])
            result.add_value("effective_volume", "Core effective volume", core.effective_volume, status="looked_up")
            result.assume("uniform_flux", "Loss density is assumed uniform over the effective volume Ve (IEC 60205 "
                          "effective parameters); local flux crowding is ignored.", "model_assumption", ["core_loss"])
            for s in evidence_sources(core_record):
                result.cite(s)
        except AIPEMagneticsError as error:
            result.warn(error.code, f"core loss not computed: {error.message}", "error")
    if record.synthetic:
        result.warn("synthetic_material", "Material record is synthetic; the estimate is illustrative only.")
    result.verification = _verification(registry, model.identity.id, f, t_k, bpk, record.id, op.waveform.kind)
    return result


def _catalog(catalog: Any | None):
    if catalog is not None:
        return catalog
    from aipe_magnetics.data.catalog import Catalog
    return Catalog.default()


def _verification(registry: Any | None, capability: str, f: float, t_k: float | None, bpk: float, material: str, excitation: str):
    if registry is None:
        from aipe_magnetics.verification.suite import load_bundled_registry
        registry = load_bundled_registry()
    return registry.status_for(capability, frequency=f, temperature=t_k, flux_density_peak=bpk, material=material,
                               excitation=excitation)


def _jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 64 else {"array": True, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return value
