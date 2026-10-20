"""2-D time-harmonic eddy-current solver with imposed conductor currents (skin and proximity effect).

Formulation (planar, phasors of amplitude, ``exp(j omega t)``)::

    -div(nu grad A) + j omega sigma A - sigma u_k = J_s     (solid conductor k: J = sigma (-j omega A + u_k))
    integral_{Omega_k} sigma (-j omega A + u_k) dS = I_k    (one constraint per conductor group)

``u_k`` (V/m) is the uniform driving field of conductor group ``k``. Solid conductors (role ``conductor``,
copper foil / round wire as an equal-area square / rectangular wire) carry eddy currents; stranded regions
(role ``stranded``, homogenised litz bundles) carry the uniform source density ``J_s = I / area`` without
eddy currents (strand-level losses are evaluated in post-processing from the local field, see
:mod:`aipe_magnetics.simulation.electromagnetic.postprocess`). Parallel conductors of one turn form one group
(equal driving field, the sum of their currents is the turn current). Boundary: ``A = 0`` on the outer boundary
and on ``x = 0`` (antisymmetric half model).

The extended sparse system is solved directly. Ohmic loss per unit length of conductor group ``k`` is
``P'_k = 1/(2 sigma) integral |J|^2 dS`` (exact quadrature for the element-linear current density); the AC
resistance per unit length is ``2 P'_k / |I_k|^2``. The per-turn resistance of the component uses the layer's
mean turn length (window cross-section assumed representative along the whole turn - conservative outside the
core where the proximity field is weaker).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from aipe_magnetics.common.errors import InvalidInputError, NumericalError
from aipe_magnetics.simulation import fem
from aipe_magnetics.simulation.electromagnetic.magnetostatic import EMProblem, element_flux_density
from aipe_magnetics.simulation.mesh import Mesh2D


@dataclass
class HarmonicSolution:
    problem: EMProblem
    frequency: float
    A: np.ndarray                     # complex nodal potential
    B: np.ndarray                     # (M, 2) complex element flux density amplitude
    groups: list[dict]                # conductor groups: winding, turn, regions, current, u, loss_per_length, r_ac_per_length
    solve_time_s: float
    stranded_density: np.ndarray      # (n_regions,) complex source density of stranded regions
    meta: dict = field(default_factory=dict)

    @property
    def mesh(self) -> Mesh2D:
        return self.problem.mesh

    def winding_loss_per_length(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for g in self.groups:
            out[g["winding"]] = out.get(g["winding"], 0.0) + g["loss_per_length_W_per_m"]
        return out

    def winding_loss(self, turn_length: dict[tuple[str, int], float]) -> dict[str, float]:
        """Total eddy/ohmic loss of the solid conductors per winding (W) with per-turn lengths (m)."""
        out: dict[str, float] = {}
        for g in self.groups:
            length = turn_length[(g["winding"], g["turn"])]
            out[g["winding"]] = out.get(g["winding"], 0.0) + g["loss_per_length_W_per_m"] * length
        return out

    def element_current_density(self) -> np.ndarray:
        """Element-mean complex current density (A/m^2) of solid conductors and stranded regions."""
        mesh = self.mesh
        j = np.zeros(mesh.n_elements, dtype=complex)
        sigma_e = self.meta["sigma_element"]
        a_mean = self.A[mesh.triangles].mean(axis=1)
        for g in self.groups:
            mask = np.isin(mesh.element_region, g["regions"])
            j[mask] = sigma_e[mask] * (-1j * 2 * np.pi * self.frequency * a_mean[mask] + g["u"])
        strand = self.stranded_density[mesh.element_region]
        return np.where(strand != 0, strand, j)


def _quad_abs2(values: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Exact integral of |f|^2 over triangles for element-linear f given at the vertices (M, 3)."""
    return area / 12.0 * (np.sum(np.abs(values) ** 2, axis=1) + np.abs(np.sum(values, axis=1)) ** 2)


def solve_harmonic(problem: EMProblem, frequency: float) -> HarmonicSolution:
    if not frequency > 0:
        raise InvalidInputError("frequency must be positive")
    mesh = problem.mesh
    t0 = time.perf_counter()
    omega = 2 * np.pi * frequency
    geometry = fem.p1_gradients(mesh.nodes, mesh.triangles)
    area = geometry[0]
    regions = mesh.section.regions
    nu = problem.element_reluctivity()
    sigma_region = np.array([problem.sigma.get(r.material, 0.0) if r.role == "conductor" else 0.0 for r in regions])
    sigma_e = sigma_region[mesh.element_region]
    group_regions: dict[tuple[str, int], list[int]] = {}
    for i, r in enumerate(regions):
        if r.role == "conductor" and r.winding is not None:
            if sigma_region[i] <= 0:
                raise InvalidInputError(f"conductor region {r.name} has no conductivity (material {r.material!r})")
            group_regions.setdefault((r.winding, r.turn), []).append(i)
    j_region = problem.region_current_density()
    stranded = np.array([j_region[i] if r.role == "stranded" else 0.0 for i, r in enumerate(regions)], dtype=complex)
    K = fem.stiffness(mesh.nodes, mesh.triangles, nu, geometry=geometry).astype(complex)
    M = fem.mass(mesh.nodes, mesh.triangles, sigma_e, geometry=geometry)
    f = fem.load(mesh.nodes, mesh.triangles, stranded[mesh.element_region], geometry=geometry).astype(complex)
    keys = list(group_regions)
    n = mesh.n_nodes
    ng = len(keys)
    cols_b = []
    G = np.zeros(ng)
    I = np.zeros(ng, dtype=complex)
    for k, key in enumerate(keys):
        mask = np.isin(mesh.element_region, group_regions[key])
        cols_b.append(fem.element_integral_vector(mesh.nodes, mesh.triangles, mask, sigma_e, geometry=geometry))
        G[k] = float(np.sum(sigma_e[mask] * area[mask]))
        I[k] = problem.polarity.get(key[0], 1.0) * problem.currents.get(key[0], 0.0)
    fixed = problem.dirichlet_nodes()
    free = np.setdiff1d(np.arange(n), fixed)
    Kc = (K + 1j * omega * M).tocsr()
    if ng:
        Bmat = sp.csr_matrix(np.column_stack(cols_b))
        top = sp.hstack([Kc[free][:, free], -Bmat[free]])
        bottom = sp.hstack([-1j * omega * Bmat[free].T, sp.diags(G)])
        system = sp.vstack([top, bottom]).tocsc()
        rhs = np.concatenate([f[free], I])
    else:
        system = Kc[free][:, free].tocsc()
        rhs = f[free]
    sol = spla.spsolve(system, rhs)
    if not np.all(np.isfinite(sol)):
        raise NumericalError("harmonic solve produced non-finite values")
    A = np.zeros(n, dtype=complex)
    A[free] = sol[: free.size]
    u = sol[free.size:]
    Bfield = element_flux_density(mesh, A, geometry[1])
    groups = []
    for k, key in enumerate(keys):
        mask = np.isin(mesh.element_region, group_regions[key])
        jv = sigma_e[mask, None] * (-1j * omega * A[mesh.triangles[mask]] + u[k])
        loss = float(np.sum(_quad_abs2(jv, area[mask]) / (2.0 * sigma_e[mask])))
        current = complex(np.sum(sigma_e[mask] * area[mask] * (-1j * omega * A[mesh.triangles[mask]].mean(axis=1) + u[k])))
        r_dc = 1.0 / G[k]
        r_ac = 2 * loss / abs(I[k]) ** 2 if abs(I[k]) > 0 else float("nan")
        groups.append({"winding": key[0], "turn": key[1], "regions": group_regions[key], "current": I[k], "current_check": current,
                       "u": complex(u[k]), "loss_per_length_W_per_m": loss, "r_dc_per_length_Ohm_per_m": r_dc,
                       "r_ac_per_length_Ohm_per_m": r_ac, "fr": r_ac / r_dc if abs(I[k]) > 0 else float("nan")})
    return HarmonicSolution(problem=problem, frequency=frequency, A=A, B=Bfield, groups=groups, solve_time_s=time.perf_counter() - t0,
                            stranded_density=stranded, meta={"sigma_element": sigma_e, "unknowns": int(free.size + ng)})


__all__ = ["HarmonicSolution", "solve_harmonic"]
