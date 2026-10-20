"""Shared helpers of the Design layer: material/core context (through the Database interface only), constraint checks,
temperature-dependent loss scaling and the electrothermal fixed point."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from aipe_magnetics.common.errors import ConvergenceError, DataNotFoundError, OutOfDomainError
from aipe_magnetics.database import CoreData, FluxShape, LossEvaluator, MaterialData, core_data, list_core_data, material_data, physical_property
from aipe_magnetics.engineering import formulas as F


@dataclass
class MaterialContext:
    data: MaterialData
    loss: LossEvaluator

    @property
    def id(self) -> str:
        return self.data.id

    def bsat(self, temperature: float) -> tuple[float, bool]:
        """``(Bsat, clamped)``: Bsat(T) from the datasheet points; outside the points the nearest point is used and
        ``clamped`` is True (the caller records it - above the highest point this is non-conservative)."""
        try:
            return self.data.saturation_flux_density(temperature)[0], False
        except OutOfDomainError:
            temps = [p[0] for p in self.data.saturation_points if not math.isnan(p[0])]
            t = min(max(temperature, min(temps)), max(temps))
            return self.data.saturation_flux_density(t)[0], True

    def thermal_conductivity(self) -> float:
        return physical_property(self.data.material_class or "ferrite_mnzn", "thermal_conductivity").value

    def density(self) -> float:
        if self.data.density is None:
            raise DataNotFoundError(f"{self.id} has no density; core mass cannot be computed")
        return self.data.density


def material_contexts(queries: list[str] | None, frequency: float, temperature: float | tuple[float, float]) -> tuple[list[MaterialContext], list[dict[str, Any]]]:
    """Stage-1 database filter: materials with Bsat, initial permeability, density and loss sets covering ``frequency`` at
    ``temperature`` or, for a ``(low, high)`` window, at some temperature inside it (evaluations outside the fitted
    temperatures are flagged ``temperature_extrapolated`` by the loss evaluator)."""
    window = temperature if isinstance(temperature, tuple) else (temperature, temperature)
    from aipe_magnetics.database import search_materials
    if queries:
        ids = [material_data(q).id for q in queries]
    else:
        ids = [m["id"] for m in search_materials(limit=200).data["items"]]
    keep, rejected = [], []
    for mid in ids:
        md = material_data(mid)
        reasons = []
        if not md.has_saturation_data():
            reasons.append("no saturation flux density data")
        if md.initial_permeability is None:
            reasons.append("no initial permeability")
        if not md.steinmetz_sets:
            reasons.append("no Steinmetz parameter set")
        else:
            ev = LossEvaluator(md.id, md.steinmetz_sets)
            errors = []
            ok = False
            for t in sorted({window[0], 0.5 * (window[0] + window[1]), window[1]}):
                try:
                    ev.select(frequency, t, policy="reject")
                    ok = True
                    break
                except Exception as error:  # noqa: BLE001 - recorded as a rejection reason
                    errors.append(getattr(error, "message", str(error)))
            if not ok:
                reasons.append(f"loss model: {errors[0]}")
        if md.density is None:
            reasons.append("no density")
        if reasons:
            rejected.append({"material": mid, "reasons": reasons})
        else:
            keep.append(MaterialContext(data=md, loss=LossEvaluator(md.id, md.steinmetz_sets)))
    return keep, rejected


def core_candidates(queries: list[str] | None, materials: list[MaterialContext], policy: str = "catalogue") -> tuple[list[CoreData], list[dict[str, Any]]]:
    """Stage-1 filter: cores with the parametric E-core outline, offered in at least one candidate material (``policy='catalogue'``)."""
    cores = [core_data(c) for c in queries] if queries else list_core_data()
    keep, rejected = [], []
    mids = {m.id for m in materials}
    for c in cores:
        reasons = []
        if not c.has_e_geometry():
            reasons.append(f"no parametric outline for shape {c.shape} (E-family outline dimensions missing)")
        if not (set(c.material_refs) & mids) and not queries and policy == "catalogue":
            reasons.append("not offered in a candidate material")
        if c.effective_area is None or c.effective_volume is None:
            reasons.append("no effective parameters")
        (rejected if reasons else keep).append(c if not reasons else {"core": c.id, "reasons": reasons})
    return keep, rejected


def pack_conductivities() -> dict[str, float]:
    return {"litz": physical_property("winding_pack_litz", "thermal_conductivity").value,
            "round": physical_property("winding_pack_round", "thermal_conductivity").value,
            "foil": physical_property("winding_pack_foil", "thermal_conductivity").value}


def thermal_conductivity_map(material: MaterialContext) -> dict[str, float]:
    kp = pack_conductivities()
    return {material.id: material.thermal_conductivity(), "air": physical_property("air", "thermal_conductivity").value,
            "bobbin_thermoplastic": physical_property("bobbin_thermoplastic", "thermal_conductivity").value,
            "polyimide_film": physical_property("polyimide_film", "thermal_conductivity").value,
            "winding_pack_litz": kp["litz"], "winding_pack_round": kp["round"], "winding_pack_foil": kp["foil"]}


def resistivity_ratio(t: float, t_ref: float) -> float:
    return F.copper_resistivity(t) / F.copper_resistivity(t_ref)


@dataclass
class ElectrothermalResult:
    converged: bool
    iterations: int
    history: list[dict[str, float]] = field(default_factory=list)
    core_loss: float = 0.0
    winding_losses: dict[str, float] = field(default_factory=dict)
    temperatures: dict[str, float] = field(default_factory=dict)
    thermal: Any = None


def electrothermal(losses: Callable[[float, float], tuple[float, dict[str, float]]],
                   thermal: Callable[[float, dict[str, float]], tuple[float, float, Any]],
                   t_core: float, t_winding: float, *, tol: float = 0.5, max_iterations: int = 12,
                   relaxation: float = 0.7) -> ElectrothermalResult:
    """Fixed point ``T -> losses(T) -> thermal(losses) -> T`` on the (core mean, winding mean) temperatures.

    ``losses(T_core, T_winding) -> (P_core, {winding: P})``; ``thermal(P_core, P_w) -> (T_core_mean, T_winding_mean, result)``.
    """
    history: list[dict[str, float]] = []
    result = None
    for it in range(1, max_iterations + 1):
        p_core, p_w = losses(t_core, t_winding)
        tc, tw, result = thermal(p_core, p_w)
        history.append({"iteration": it, "T_core_in_K": t_core, "T_winding_in_K": t_winding, "P_core_W": p_core,
                        "P_winding_W": sum(p_w.values()), "T_core_out_K": tc, "T_winding_out_K": tw})
        if abs(tc - t_core) < tol and abs(tw - t_winding) < tol:
            return ElectrothermalResult(True, it, history, p_core, p_w, {"core": tc, "winding": tw}, result)
        if tc > 700 or tw > 700:
            return ElectrothermalResult(False, it, history, p_core, p_w, {"core": tc, "winding": tw}, result)
        t_core = t_core + relaxation * (tc - t_core)
        t_winding = t_winding + relaxation * (tw - t_winding)
    return ElectrothermalResult(False, max_iterations, history, p_core, p_w, {"core": tc, "winding": tw}, result)


def flux_shape_square(duty: float = 0.5) -> FluxShape:
    """Flux of a bipolar square voltage applying +V for ``duty`` of the period (triangular / symmetric trapezoid)."""
    return FluxShape.triangular(0.5) if abs(duty - 0.5) < 1e-12 else FluxShape.trapezoidal(duty)


__all__ = ["ConvergenceError", "ElectrothermalResult", "MaterialContext", "core_candidates", "electrothermal", "flux_shape_square",
           "material_contexts", "pack_conductivities", "resistivity_ratio", "thermal_conductivity_map"]
