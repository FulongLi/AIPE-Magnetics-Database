"""Lumped steady-state thermal network of an E-core component (analytical fidelity; *not* finite elements).

Nodes: the core (volume-average temperature) and one node per winding section (radially ordered, at the section
mid-build). Conductances:

* core -> ambient: ``h(T_core) A_core,exposed`` (outer box surface minus window openings);
* outermost winding section -> ambient: ``h(T) A_end-turns`` (end-turn surface outside the core);
* innermost section -> core (centre leg): half the section build in the winding pack, the bobbin tube and the leg
  clearance (air), over the inner turn perimeter x winding height;
* section i -> section i+1: half builds of both packs plus the insulation between them;
* ``h`` from :class:`~aipe_magnetics.simulation.thermal.convection.Cooling` at the node temperature (fixed-point
  iteration, converged to ``tol``).

Hot-spot estimates: winding sections - the node (mid-build) temperature plus ``P t / (8 k A)`` (uniformly
heated slab of build ``t``); core - node plus ``q (F/2)^2 / (2 k)`` across the centre-leg half width. Both are
explicitly labelled estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aipe_magnetics.common.errors import ConvergenceError, InvalidInputError
from aipe_magnetics.simulation.thermal.convection import Cooling

THERMAL_NETWORK_ID = "aipe.magnetics.thermal.network"
THERMAL_NETWORK_VERSION = "0.3.0"


@dataclass
class ThermalNetworkResult:
    temperatures: dict[str, float]
    hotspots: dict[str, float]
    conductances: dict[str, float]
    heat_flows: dict[str, float]
    iterations: int
    balance_residual_W: float
    assumptions: list[str] = field(default_factory=list)
    method: str = "analytical_thermal_network"

    def summary(self) -> dict[str, Any]:
        return {"method": self.method, "model": f"{THERMAL_NETWORK_ID}@{THERMAL_NETWORK_VERSION}",
                "temperatures_K": self.temperatures, "hotspots_K": self.hotspots,
                "hotspot_core_K": self.hotspots.get("core"), "hotspot_winding_K": max(v for k, v in self.hotspots.items() if k != "core"),
                "conductances_W_per_K": self.conductances, "heat_flows_W": self.heat_flows, "iterations": self.iterations,
                "balance_residual_W": self.balance_residual_W, "assumptions": self.assumptions}


def solve_thermal_network(geometry: Any, *, core_loss: float, winding_losses: dict[str, float], cooling: Cooling,
                          k_core: float, k_bobbin: float, k_insulation: float, k_pack: dict[str, float], k_air: float = 0.03,
                          tol: float = 0.01, max_iterations: int = 100) -> ThermalNetworkResult:
    """``winding_losses`` per winding name (W) are split over that winding's sections in proportion to their copper length."""
    model = geometry.model
    c = model.core
    layers = geometry.layers
    if not layers:
        raise InvalidInputError("geometry has no winding layers")
    derived = geometry.derived
    space = model.winding_space()
    sections = sorted({l.section for l in layers})
    sec = []
    for s in sections:
        ls = [l for l in layers if l.section == s]
        length = sum(l.turns * l.mean_turn_length for l in ls)
        sec.append({"id": f"{ls[0].winding}.s{s}", "winding": ls[0].winding, "d_in": ls[0].d_in, "d_out": ls[-1].d_out,
                    "height": max(l.z1 for l in ls) - min(l.z0 for l in ls), "length": length, "kind": ls[0].conductor.kind})
    for w in {s["winding"] for s in sec}:
        if w not in winding_losses:
            raise InvalidInputError(f"no loss given for winding {w!r}")
        total = sum(s["length"] for s in sec if s["winding"] == w)
        for s in sec:
            if s["winding"] == w:
                s["loss"] = winding_losses[w] * s["length"] / total
    hx, hy = 0.5 * c.centre_leg_width, 0.5 * model.depth

    def perimeter(d: float) -> float:
        return 4 * hx + 4 * hy + 2 * math.pi * d

    names = ["core"] + [s["id"] for s in sec]
    n = len(names)
    height_total = 2 * c.height_b_half + model.gap.spacer
    a_core = derived["core_exposed_area_m2"]
    a_end = derived["winding_exposed_area_m2"]
    g_fixed: dict[tuple[int, int], float] = {}
    # inner section to core: half build of the pack + bobbin tube + leg clearance (air)
    s0 = sec[0]
    area_in = perimeter(s0["d_in"]) * s0["height"]
    r = (0.5 * (s0["d_out"] - s0["d_in"])) / (k_pack[s0["kind"]] * area_in)
    if model.bobbin.present:
        r += model.bobbin.tube_thickness / (k_bobbin * perimeter(model.bobbin.leg_clearance + 0.5 * model.bobbin.tube_thickness) * s0["height"])
    r += model.bobbin.leg_clearance / (k_air * perimeter(0.5 * model.bobbin.leg_clearance) * s0["height"])
    g_fixed[(0, 1)] = 1.0 / r
    for i in range(len(sec) - 1):
        a, b = sec[i], sec[i + 1]
        area = perimeter(a["d_out"]) * min(a["height"], b["height"])
        r = (0.5 * (a["d_out"] - a["d_in"]) / (k_pack[a["kind"]] * area) + max(b["d_in"] - a["d_out"], 0.0) / (k_insulation * area)
             + 0.5 * (b["d_out"] - b["d_in"]) / (k_pack[b["kind"]] * area))
        g_fixed[(i + 1, i + 2)] = 1.0 / r
    p = np.array([core_loss] + [s["loss"] for s in sec])
    t = np.full(n, cooling.ambient + 40.0)
    g_core_amb = g_out_amb = 0.0
    for it in range(1, max_iterations + 1):
        g_core_amb = cooling.h_total(t[0], height_total) * a_core
        g_out_amb = cooling.h_total(t[-1], height_total) * a_end
        G = np.zeros((n, n))
        for (i, j), g in g_fixed.items():
            G[i, i] += g; G[j, j] += g; G[i, j] -= g; G[j, i] -= g
        G[0, 0] += g_core_amb
        G[-1, -1] += g_out_amb
        rhs = p.copy()
        rhs[0] += g_core_amb * cooling.ambient
        rhs[-1] += g_out_amb * cooling.ambient
        t_new = np.linalg.solve(G, rhs)
        if np.max(np.abs(t_new - t)) < tol:
            t = t_new
            break
        t = 0.5 * t + 0.5 * t_new
    else:
        raise ConvergenceError(f"thermal network did not converge in {max_iterations} iterations")
    q_core_amb = g_core_amb * (t[0] - cooling.ambient)
    q_out_amb = g_out_amb * (t[-1] - cooling.ambient)
    residual = float(p.sum() - q_core_amb - q_out_amb)
    temps = {name: float(v) for name, v in zip(names, t)}
    hot: dict[str, float] = {}
    v_core = derived["core_volume_m3"]
    hot["core"] = float(t[0] + (core_loss / v_core) * (0.5 * c.centre_leg_width) ** 2 / (2 * k_core))
    for i, s in enumerate(sec):
        build = s["d_out"] - s["d_in"]
        area_mid = perimeter(0.5 * (s["d_in"] + s["d_out"])) * s["height"]
        hot[s["id"]] = float(t[i + 1] + s["loss"] * build / (8 * k_pack[s["kind"]] * area_mid))
    for w in {s["winding"] for s in sec}:
        hot[w] = max(hot[s["id"]] for s in sec if s["winding"] == w)
    cond = {"core_to_ambient": g_core_amb, "outer_winding_to_ambient": g_out_amb}
    for (i, j), g in g_fixed.items():
        cond[f"{names[i]}<->{names[j]}"] = g
    flows = {"core_to_ambient": q_core_amb, "outer_winding_to_ambient": q_out_amb}
    for (i, j), g in g_fixed.items():
        flows[f"{names[j]}->{names[i]}"] = g * (t[j] - t[i])
    assumptions = [cooling.describe(), "core node = volume average; winding nodes at the section mid-build",
                   "end turns of the outermost winding are its only surface to ambient; inner windings reach ambient through the "
                   "outer windings and the core", "hot spots: uniformly heated slab estimates (labelled estimates)"]
    return ThermalNetworkResult(temperatures=temps, hotspots=hot, conductances=cond, heat_flows=flows, iterations=it,
                                balance_residual_W=residual, assumptions=assumptions)


__all__ = ["THERMAL_NETWORK_ID", "THERMAL_NETWORK_VERSION", "ThermalNetworkResult", "solve_thermal_network"]
