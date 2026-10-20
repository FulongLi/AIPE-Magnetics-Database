"""2.5-D steady-state thermal finite-element model on the component cross-section (genuine FE conduction solve).

Equation on the solid part of the half section (core, gaps, bobbin, insulation, window air, winding packs)::

    -div(k grad T) + s (T - T_amb) = q

* ``k`` per region: ferrite, bobbin polymer, insulation film, stagnant air (window, gaps) and the homogenised winding
  pack of each conductor kind (property library, :mod:`aipe_magnetics.database.properties`);
* ``q``: element core-loss density from the electromagnetic field post-processing (W/m^3) and each winding's loss spread
  uniformly over its pack regions (``P_w / (A_pack x mean turn length)``);
* ``s`` (W/m^3/K): out-of-plane heat removal - the core slab loses heat through its front/back faces
  (``s = 2 h / C``), the outermost winding pack through its end turns outside the core
  (``s = h A_end / (A_pack MLT)``, heat carried along the copper to the end turns);
* boundary: convection + radiation ``-k dT/dn = h (T - T_amb)`` on the exterior core edges (outer-leg face, top and
  bottom); ``x = 0`` is a symmetry plane (adiabatic).

``h`` follows the surface temperatures (fixed-point iteration). The solution reports the temperature field, region
maxima / means and the energy balance (sources = in-plane boundary + out-of-plane sinks), which is a verification
quantity. The out-of-plane terms make this a *2.5-D* model; it is labelled ``fea_2p5d`` and never presented as a 3-D
thermal simulation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aipe_magnetics.common.errors import ConvergenceError, InvalidInputError
from aipe_magnetics.simulation import fem
from aipe_magnetics.simulation.mesh import Mesh2D
from aipe_magnetics.simulation.thermal.convection import Cooling

THERMAL_FEM_ID = "aipe.magnetics.thermal.fem2p5d"
THERMAL_FEM_VERSION = "0.3.0"
SOLID_ROLES = ("core", "gap", "conductor", "stranded", "winding_pack", "bobbin", "insulation", "window_air")


@dataclass
class ThermalFEMResult:
    mesh: Mesh2D
    T: np.ndarray
    region_stats: dict[str, dict[str, float]]
    hotspots: dict[str, dict[str, Any]]
    balance: dict[str, float]
    iterations: int
    solve_time_s: float
    settings: dict[str, Any]
    element_source: np.ndarray
    element_k: np.ndarray
    method: str = "fea_2p5d"
    assumptions: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {"method": self.method, "model": f"{THERMAL_FEM_ID}@{THERMAL_FEM_VERSION}", "hotspots": self.hotspots,
                "region_stats": self.region_stats, "balance": self.balance, "iterations": self.iterations,
                "solve_time_s": self.solve_time_s, "elements": self.mesh.n_elements, "settings": self.settings,
                "assumptions": self.assumptions}


def solve_thermal_fem(geometry: Any, em_mesh: Mesh2D, *, core_loss_density: np.ndarray, winding_losses: dict[str, float],
                      cooling: Cooling, conductivity: dict[str, float], tol: float = 0.01, max_iterations: int = 50) -> ThermalFEMResult:
    """``core_loss_density``: per element of ``em_mesh`` (W/m^3). ``conductivity``: material key -> k (W/m/K)."""
    t0 = time.perf_counter()
    solid = em_mesh.region_mask(*SOLID_ROLES)
    mesh, _ = em_mesh.submesh(solid)
    q_core = np.asarray(core_loss_density)[solid]
    regions = mesh.section.regions
    model = geometry.model
    depth = mesh.section.depth
    area = mesh.areas()
    reg = mesh.element_region
    # conductivities
    missing = sorted({r.material for r in regions if r.role in SOLID_ROLES and r.material not in conductivity and r.role != "core"})
    k_region = np.zeros(len(regions))
    for i, r in enumerate(regions):
        if r.role == "core":
            k_region[i] = conductivity.get(r.material, conductivity.get("core", np.nan))
        elif r.role in ("conductor", "stranded"):
            # turns sit inside their pack: use the pack conductivity of the winding's conductor kind
            kind = r.meta.get("kind", "litz")
            k_region[i] = conductivity.get({"litz": "winding_pack_litz", "round": "winding_pack_round"}.get(kind, "winding_pack_foil"), np.nan)
        elif r.role in SOLID_ROLES:
            k_region[i] = conductivity.get(r.material, np.nan)
    if np.any(np.isnan(k_region[np.unique(reg)])):
        raise InvalidInputError(f"thermal conductivity missing for material(s) {missing}")
    k = k_region[reg]
    # winding heat sources: per winding uniform over its pack (+ conductor) regions
    q = q_core.copy()
    mlt = mesh.section.winding_length
    for w, p_w in winding_losses.items():
        m = np.array([(r.winding == w and r.role in ("winding_pack", "conductor", "stranded")) for r in regions])[reg]
        a_pack = float(area[m].sum())
        if a_pack <= 0:
            raise InvalidInputError(f"winding {w!r} has no pack region in the section")
        q[m] += p_w / (a_pack * mlt[w])
    # out-of-plane sinks
    core_mask = np.array([r.role == "core" for r in regions])[reg]
    layers = geometry.layers
    outer_section = max(l.section for l in layers)
    outer_mask = np.array([(r.role in ("winding_pack", "conductor", "stranded")) and r.meta.get("section", None) == outer_section
                           or (r.role in ("conductor", "stranded") and r.layer and r.layer.split(".")[1] == f"s{outer_section}")
                           for r in regions])[reg]
    outer_winding = next(l.winding for l in layers if l.section == outer_section)
    a_outer = float(area[outer_mask].sum())
    a_end = geometry.derived["winding_exposed_area_m2"]
    height_total = 2 * model.core.height_b_half + model.gap.spacer
    # exterior edges for in-plane convection (exclude the symmetry plane x = 0)
    edges = mesh.boundary_edges()
    x_edge = mesh.nodes[edges].mean(axis=1)[:, 0]
    exterior = edges[x_edge > 1e-9]
    t_amb = cooling.ambient
    T = np.full(mesh.n_nodes, t_amb + 40.0)
    geometry_cache = fem.p1_gradients(mesh.nodes, mesh.triangles)
    K = fem.stiffness(mesh.nodes, mesh.triangles, k, geometry=geometry_cache)
    f_src = fem.load(mesh.nodes, mesh.triangles, q, geometry=geometry_cache)
    it = 0
    for it in range(1, max_iterations + 1):
        t_core_mean = float(np.sum(T[mesh.triangles[core_mask]].mean(axis=1) * area[core_mask]) / np.sum(area[core_mask]))
        t_out_mean = float(np.sum(T[mesh.triangles[outer_mask]].mean(axis=1) * area[outer_mask]) / max(a_outer, 1e-30))
        h_core = cooling.h_total(t_core_mean, height_total)
        h_end = cooling.h_total(t_out_mean, height_total)
        s = np.zeros(mesh.n_elements)
        s[core_mask] = 2.0 * h_core / depth
        s[outer_mask] = h_end * a_end / (a_outer * mlt[outer_winding])
        M = fem.mass(mesh.nodes, mesh.triangles, s, geometry=geometry_cache)
        B, fb = fem.robin_edges(mesh.nodes, exterior, np.full(len(exterior), h_core), t_amb)
        f_sink = fem.load(mesh.nodes, mesh.triangles, s * t_amb, geometry=geometry_cache)
        A = (K + M + B).tocsc()
        import scipy.sparse.linalg as spla
        T_new = spla.spsolve(A, f_src + f_sink + fb)
        if not np.all(np.isfinite(T_new)):
            raise ConvergenceError("thermal FE solve produced non-finite temperatures")
        delta = float(np.max(np.abs(T_new - T)))
        T = T_new
        if delta < tol:
            break
    else:
        raise ConvergenceError(f"thermal FE h(T) iteration did not converge in {max_iterations} iterations")
    # energy balance per unit length (2-D) and scaled sums
    te = T[mesh.triangles].mean(axis=1)
    src = float(np.sum(q * area))
    sink = float(np.sum(s * (te - t_amb) * area))  # mean-value quadrature; consistent mass used in the solve
    sink_exact = float((T - t_amb) @ (M @ np.ones(mesh.n_nodes)))
    p_edges = mesh.nodes[exterior]
    length = np.linalg.norm(p_edges[:, 1] - p_edges[:, 0], axis=1)
    t_edge = T[exterior].mean(axis=1)
    edge_flow = float(np.sum(h_core * (t_edge - t_amb) * length))
    balance = {"sources_W_per_m": src, "out_of_plane_sink_W_per_m": sink_exact, "boundary_convection_W_per_m": edge_flow,
               "relative_residual": (src - sink_exact - edge_flow) / src if src > 0 else 0.0,
               "h_core_W_per_m2K": h_core, "h_end_turns_W_per_m2K": h_end, "out_of_plane_sink_midpoint_W_per_m": sink}
    stats: dict[str, dict[str, float]] = {}
    hot: dict[str, dict[str, Any]] = {}
    groups = {"core": core_mask, "bobbin": np.array([r.role == "bobbin" for r in regions])[reg]}
    for w in mlt:
        groups[f"winding:{w}"] = np.array([(r.winding == w and r.role in ("winding_pack", "conductor", "stranded")) for r in regions])[reg]
    for name, m in groups.items():
        if not m.any():
            continue
        tri = mesh.triangles[m]
        node_idx = np.unique(tri)
        imax = node_idx[np.argmax(T[node_idx])]
        stats[name] = {"T_max_K": float(T[node_idx].max()), "T_mean_K": float(np.sum(te[m] * area[m]) / np.sum(area[m])),
                       "T_min_K": float(T[node_idx].min())}
        hot[name] = {"T_K": float(T[imax]), "x_m": float(mesh.nodes[imax, 0]), "z_m": float(mesh.nodes[imax, 1])}
    assumptions = [cooling.describe(), "2.5-D: cross-section conduction with out-of-plane sinks (core front/back faces, end turns of the "
                   "outermost winding)", "winding packs homogenised (isotropic effective conductivity)",
                   "window air and gaps: stagnant-air conduction, no internal convection"]
    return ThermalFEMResult(mesh=mesh, T=T, region_stats=stats, hotspots=hot, balance=balance, iterations=it,
                            solve_time_s=time.perf_counter() - t0, settings={"tol_K": tol, "conductivity_W_per_mK": conductivity},
                            element_source=q, element_k=k, assumptions=assumptions)


__all__ = ["SOLID_ROLES", "THERMAL_FEM_ID", "THERMAL_FEM_VERSION", "ThermalFEMResult", "solve_thermal_fem"]
