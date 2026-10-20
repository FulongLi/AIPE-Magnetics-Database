"""2-D magnetostatic finite-element solver (planar ``A_y`` formulation, optional nonlinear core).

Equation (planar, current along +y, ``B = curl(A_y e_y)``, so ``B_x = -dA/dz`` and ``B_z = dA/dx``)::

    -div( nu grad A ) = J        in the section,       A = 0 on the outer boundary and on x = 0 (half model)

``J`` is uniform in every conductor / stranded region: ``J = polarity * I_winding / (area * parallel)`` (all
turns of a winding in series). The right window represents the ``+y`` side of every turn; the antisymmetric
continuation to ``x < 0`` (the ``-y`` side) is implied by ``A(0, z) = 0``.

Nonlinear cores use the secant (Picard) iteration ``nu_e <- (1 - r) nu_e + r nu(|B_e|)`` with relaxation ``r``
until the relative change of ``nu`` in the core is below ``tol`` (failure to converge raises
:class:`~aipe_magnetics.common.errors.ConvergenceError`; it is never hidden).

Derived quantities (:class:`MagnetostaticSolution`):

* element flux density ``B`` and field ``H = nu B``;
* energy per unit depth of any element set ``W' = sum_e 1/2 nu_e |B_e|^2 A_e`` (linear materials);
* flux linkage of a winding per unit depth ``lambda' = 2 * sum_turns <A>_turn`` (the factor 2 accounts for the
  mirrored ``-y`` side; ``<A>`` is the area average over the conductor cross-section);
* :meth:`MagnetostaticSolution.energy_3d` - the 2.5-D energy of the component: core / gap / outer-air energy
  times ``2 x depth`` plus window energy times the winding's mean turn length (the window field is assumed to
  continue around the end turns; conservative for leakage inductance).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aipe_magnetics.common.errors import ConvergenceError, InvalidInputError
from aipe_magnetics.common.units import MU_0
from aipe_magnetics.simulation import fem
from aipe_magnetics.simulation.electromagnetic.materials import SaturationCurve
from aipe_magnetics.simulation.mesh import Mesh2D

WINDOW_ROLES = ("window_air", "conductor", "stranded", "winding_pack", "bobbin", "insulation")


@dataclass
class EMProblem:
    """Material assignment and excitation of a section mesh."""

    mesh: Mesh2D
    mu_r: dict[str, float]                                  # material key -> relative permeability (default 1)
    currents: dict[str, complex | float]                    # winding -> current (A; phasor amplitude for harmonic)
    polarity: dict[str, float] = field(default_factory=dict)   # winding -> +1 / -1 (default +1)
    curves: dict[str, SaturationCurve] = field(default_factory=dict)  # material key -> nonlinear law
    sigma: dict[str, float] = field(default_factory=dict)  # material key -> conductivity (harmonic solver)

    def region_mu(self) -> np.ndarray:
        return np.array([self.mu_r.get(r.material, 1.0) for r in self.mesh.section.regions])

    def element_reluctivity(self) -> np.ndarray:
        return 1.0 / (MU_0 * self.region_mu()[self.mesh.element_region])

    def region_current_density(self) -> np.ndarray:
        """Uniform current density per region (A/m^2): series turns, ``parallel`` conductors share the turn current."""
        regions = self.mesh.section.regions
        parallel: dict[tuple[str, int], int] = {}
        for r in regions:
            if r.role in ("conductor", "stranded") and r.winding is not None:
                parallel[(r.winding, r.turn)] = parallel.get((r.winding, r.turn), 0) + 1
        j = np.zeros(len(regions), dtype=complex)
        for i, r in enumerate(regions):
            if r.role in ("conductor", "stranded") and r.winding in self.currents:
                pol = self.polarity.get(r.winding, 1.0)
                j[i] = pol * self.currents[r.winding] / (r.area * parallel[(r.winding, r.turn)])
        return j

    def dirichlet_nodes(self) -> np.ndarray:
        nodes = self.mesh.nodes
        x0, x1, z0, z1 = self.mesh.section.bounds
        tol = 1e-12 * max(x1 - x0, z1 - z0)
        # outer boundary (A = 0, flux confined to the domain); for the half model x = x0 is the antisymmetry plane (A = 0 too)
        on = ((np.abs(nodes[:, 0] - x1) < tol) | (np.abs(nodes[:, 1] - z0) < tol) | (np.abs(nodes[:, 1] - z1) < tol)
              | (np.abs(nodes[:, 0] - x0) < tol))
        return np.where(on)[0]


@dataclass
class MagnetostaticSolution:
    problem: EMProblem
    A: np.ndarray
    B: np.ndarray            # (M, 2) element Bx, Bz
    nu: np.ndarray           # (M,) element reluctivity used (final iteration)
    iterations: int
    converged: bool
    residual: float
    solve_time_s: float
    nonlinear: bool
    history: list[float] = field(default_factory=list)

    @property
    def mesh(self) -> Mesh2D:
        return self.problem.mesh

    @property
    def B_magnitude(self) -> np.ndarray:
        return np.hypot(self.B[:, 0], self.B[:, 1])

    def energy_density(self) -> np.ndarray:
        """``1/2 nu |B|^2`` per element (J/m^3; for nonlinear cores the secant value, used only for reporting)."""
        return 0.5 * self.nu * self.B_magnitude ** 2

    def energy_per_depth(self, mask: np.ndarray | None = None) -> float:
        w = self.energy_density() * self.mesh.areas()
        return float(w.sum() if mask is None else w[mask].sum())

    def window_mask(self) -> np.ndarray:
        win = self.mesh.section.meta.get("window")
        c = self.mesh.centroids()
        roles = self.mesh.region_mask(*WINDOW_ROLES)
        if win is None:
            return roles
        inside = (c[:, 0] >= win["x0"]) & (c[:, 0] <= win["x1"]) & (c[:, 1] >= win["z0"]) & (c[:, 1] <= win["z1"])
        return roles & inside

    def energy_3d(self, window_length: float) -> dict[str, float]:
        """2.5-D energy: (core + gaps + outer air) x 2 x depth + window x ``window_length`` (J)."""
        win = self.window_mask()
        w_core = self.energy_per_depth(~win)
        w_win = self.energy_per_depth(win)
        depth = self.mesh.section.depth
        factor = 2.0 if self.mesh.section.symmetry == "half_x" else 1.0
        return {"core_gap_air_J": factor * depth * w_core, "window_J": w_win * window_length * (1.0 if factor == 2.0 else 0.5),
                "total_J": factor * depth * w_core + w_win * window_length * (1.0 if factor == 2.0 else 0.5),
                "core_gap_air_per_depth_J_per_m": w_core, "window_per_length_J_per_m": w_win}

    def mean_potential(self, element_mask: np.ndarray) -> float:
        area = self.mesh.areas()[element_mask]
        a_e = self.A[self.mesh.triangles[element_mask]].mean(axis=1)
        return float(np.real(np.sum(a_e * area) / np.sum(area)))

    def flux_linkage_per_depth(self, winding: str) -> float:
        """``lambda' = factor * sum_turns <A>_turn`` (Wb/m); factor 2 for the half model (mirrored side)."""
        regions = self.mesh.section.regions
        turns: dict[int, list[int]] = {}
        for i, r in enumerate(regions):
            if r.role in ("conductor", "stranded") and r.winding == winding:
                turns.setdefault(r.turn, []).append(i)
        if not turns:
            raise InvalidInputError(f"no conductor of winding {winding!r} in the section")
        pol = self.problem.polarity.get(winding, 1.0)
        total = 0.0
        for idx in turns.values():
            mask = np.isin(self.mesh.element_region, idx)
            total += self.mean_potential(mask)
        factor = 2.0 if self.mesh.section.symmetry == "half_x" else 1.0
        return pol * factor * total

    def region_stats(self, *roles: str, names: list[str] | None = None) -> dict[str, float]:
        mask = self.mesh.region_mask(*roles, names=names)
        if not mask.any():
            return {}
        b = self.B_magnitude[mask]
        a = self.mesh.areas()[mask]
        return {"B_max_T": float(b.max()), "B_mean_T": float(np.sum(b * a) / np.sum(a)), "B_rms_T": float(math.sqrt(np.sum(b ** 2 * a) / np.sum(a))),
                "area_m2": float(a.sum())}

    def field_arrays(self) -> dict[str, np.ndarray]:
        return {"A": np.real(self.A), "B": self.B, "B_magnitude": self.B_magnitude, "nu": self.nu}


def element_flux_density(mesh: Mesh2D, A: np.ndarray, grad: np.ndarray | None = None) -> np.ndarray:
    if grad is None:
        _, grad = fem.p1_gradients(mesh.nodes, mesh.triangles)
    a = A[mesh.triangles]
    dadx = np.einsum("ei,ei->e", a, grad[:, :, 0])
    dadz = np.einsum("ei,ei->e", a, grad[:, :, 1])
    return np.column_stack([-dadz, dadx])


def solve_magnetostatic(problem: EMProblem, *, nonlinear: bool | None = None, tol: float = 1e-4, max_iterations: int = 200,
                        relaxation: float = 0.5) -> MagnetostaticSolution:
    """Solve the magnetostatic problem (nonlinear when ``problem.curves`` is set and ``nonlinear`` is not False)."""
    mesh = problem.mesh
    t0 = time.perf_counter()
    geometry = fem.p1_gradients(mesh.nodes, mesh.triangles)
    nu = problem.element_reluctivity()
    j_region = problem.region_current_density()
    if np.any(np.abs(j_region.imag) > 0):
        raise InvalidInputError("magnetostatic currents must be real")
    j = j_region.real[mesh.element_region]
    f = fem.load(mesh.nodes, mesh.triangles, j, geometry=geometry)
    fixed = problem.dirichlet_nodes()
    use_nl = bool(problem.curves) if nonlinear is None else (nonlinear and bool(problem.curves))
    curve_of_elem: dict[str, np.ndarray] = {}
    if use_nl:
        materials = np.array([r.material for r in mesh.section.regions])[mesh.element_region]
        for key in problem.curves:
            curve_of_elem[key] = materials == key
    history: list[float] = []
    iterations = 0
    converged = True
    residual = 0.0
    while True:
        K = fem.stiffness(mesh.nodes, mesh.triangles, nu, geometry=geometry)
        A = fem.solve_dirichlet(K, f, fixed, 0.0)
        B = element_flux_density(mesh, A, geometry[1])
        iterations += 1
        if not use_nl:
            break
        bmag = np.hypot(B[:, 0], B[:, 1])
        new_nu = nu.copy()
        for key, mask in curve_of_elem.items():
            new_nu[mask] = problem.curves[key].reluctivity(bmag[mask])
        sel = np.zeros(nu.shape, dtype=bool)
        for mask in curve_of_elem.values():
            sel |= mask
        residual = float(np.max(np.abs(new_nu[sel] - nu[sel]) / new_nu[sel])) if sel.any() else 0.0
        history.append(residual)
        if residual < tol:
            nu = new_nu
            K = fem.stiffness(mesh.nodes, mesh.triangles, nu, geometry=geometry)
            A = fem.solve_dirichlet(K, f, fixed, 0.0)
            B = element_flux_density(mesh, A, geometry[1])
            break
        if iterations >= max_iterations:
            converged = False
            raise ConvergenceError(f"nonlinear magnetostatic iteration did not converge in {max_iterations} iterations "
                                   f"(relative reluctivity change {residual:.3g} > {tol:g})", details={"history": history[-10:]})
        nu = (1.0 - relaxation) * nu + relaxation * new_nu
    return MagnetostaticSolution(problem=problem, A=A, B=B, nu=nu, iterations=iterations, converged=converged, residual=residual,
                                 solve_time_s=time.perf_counter() - t0, nonlinear=use_nl, history=history)


__all__ = ["EMProblem", "MagnetostaticSolution", "WINDOW_ROLES", "element_flux_density", "solve_magnetostatic"]
