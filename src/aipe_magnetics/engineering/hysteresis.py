"""Hysteresis evaluation capability: Jiles-Atherton loop through the dynamic (ODE) framework."""

from __future__ import annotations

from typing import Any

import numpy as np

from aipe_magnetics.api.types import HysteresisRequest
from aipe_magnetics.common.errors import AIPEMagneticsError, InvalidInputError
from aipe_magnetics.common.results import EngineeringResult

CAP_HYSTERESIS = "aipe.magnetics.evaluate_hysteresis"
JA_PARAMETER_NAMES = ("Ms", "a", "alpha", "k", "c")


def _model(parameters: dict[str, float] | None):
    from aipe_magnetics.models.physics.jiles_atherton import JilesAthertonModel

    if not parameters:
        return JilesAthertonModel.illustrative(), "illustrative"
    missing = [p for p in JA_PARAMETER_NAMES if p not in parameters]
    unknown = [p for p in parameters if p not in JA_PARAMETER_NAMES]
    if missing or unknown:
        raise InvalidInputError(f"Jiles-Atherton parameters must be exactly {JA_PARAMETER_NAMES}; missing {missing}, unknown {unknown}")
    return JilesAthertonModel(**{k: float(parameters[k]) for k in JA_PARAMETER_NAMES}), "user_input"


def evaluate_hysteresis(request: HysteresisRequest, registry: Any | None = None) -> EngineeringResult:
    inputs = request.model_dump(mode="json", exclude_none=True)
    try:
        from aipe_magnetics.models.dynamic.excitation import Excitation

        model, basis = _model(request.parameters)
        h_peak, f = request.field_peak.value, request.frequency.value
        offset = request.field_dc.value if request.field_dc is not None else 0.0
        if request.waveform == "sinusoidal":
            excitation = Excitation.sinusoidal(h_peak, f, offset=offset)
        else:
            excitation = Excitation.triangular(h_peak, f, offset=offset)
        from aipe_magnetics.models.dynamic.ode import SolverConfig

        config = SolverConfig(method="RK45", rtol=request.rtol, atol=request.atol, max_step=1.0 / f / 200.0)
        output = model.trajectory(excitation, policy=request.policy, config=config, max_periods=request.periods,
                                  steady_state_tol=request.steady_state_tol, points_per_period=request.samples_per_period)
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(CAP_HYSTERESIS, error, inputs=inputs)

    result = EngineeringResult(capability=CAP_HYSTERESIS, inputs=inputs, model=output.model)
    labels = {"field_peak": "Peak applied field", "flux_density_peak": "Peak flux density", "remanence": "Remanent flux density",
              "coercivity": "Coercive field", "loop_energy_density": "Hysteresis loop energy density (closed-loop integral of H dB)",
              "magnetisation_peak": "Peak magnetisation", "magnetization_peak": "Peak magnetisation"}
    for key, q in output.values.items():
        result.add_value(key, labels.get(key, key.replace("_", " ")), q, status="estimated", derived_from=["field_peak", "parameters"])
    from aipe_magnetics.engineering._json import jsonable

    diagnostics = dict(output.diagnostics)
    diagnostics.pop("solution", None)  # raw ODESolution objects are not serialised; summary keys remain
    result.data["diagnostics"] = jsonable(diagnostics)
    result.data["model"] = model.describe()
    result.data["domain_check"] = output.domain.model_dump(mode="json")
    if basis == "illustrative":
        result.assume("ja_illustrative_parameters", "Jiles-Atherton parameters are the illustrative demonstration set shipped with the "
                      "package (Ms, a, alpha, k, c); they are not identified for any material and must not be used for material claims.",
                      "model_assumption", list(output.values))
        result.warn("illustrative_parameters", "Results use illustrative Jiles-Atherton parameters, not a material fit.", "warning")
    else:
        result.assume("ja_user_parameters", "Jiles-Atherton parameters supplied by the user.", "user_input", list(output.values))
    for a in output.assumptions:
        result.assume(a.id, a.statement, a.basis, a.affects)
    for lim in output.limitations:
        result.limit(lim.code, lim.statement)
    for w in output.warnings:
        result.warn(w.code, w.message, w.severity)
    for s in output.sources:
        result.cite(s)
    for lim in model.domain.known_limitations:
        result.limit("ja_" + lim.split(":")[0].split(" ")[0].lower(), lim)
    loop = output.values.get("loop_energy_density")
    if loop is not None and loop.value < 0:
        result.warn("negative_loop_energy", "Loop energy density is negative: sign convention or non-converged loop; result is nonphysical.", "error")
        result.status = "error"
    if registry is None:
        from aipe_magnetics.verification.suite import load_bundled_registry
        registry = load_bundled_registry()
    result.verification = registry.status_for(model.identity.id, excitation=request.waveform)
    return result
