"""Linear-triangle (P1) finite-element kernels: scalar diffusion operators on a :class:`Mesh2D`.

All electromagnetic and thermal solvers of the package reduce to sparse systems assembled here:

* stiffness ``K_ij = sum_e c_e w_e A_e grad(phi_i) . grad(phi_j)``
* consistent mass ``M_ij = sum_e c_e w_e A_e (1 + delta_ij) / 12``
* load ``f_i = sum_e s_e w_e A_e / 3``
* Robin boundary terms on edges (``h L / 6 [[2, 1], [1, 2]]`` and ``h T_inf L / 2``)

``w_e`` is the coordinate weight: 1 for planar problems, the centroid radius ``r_e`` for axisymmetric
problems written in cylindrical coordinates (``x`` = r). The axisymmetric magnetostatic formulation uses
``u = r A_phi`` with coefficient ``nu / r`` (see :mod:`aipe_magnetics.simulation.electromagnetic.magnetostatic`).
Gradients of P1 basis functions are constant per element; derived fields are element-wise constants.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from aipe_magnetics.common.errors import NumericalError


def p1_gradients(nodes: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(area (M,), grad (M, 3, 2))``; raises on degenerate or clockwise elements."""
    p = nodes[triangles]
    x, z = p[:, :, 0], p[:, :, 1]
    det = (x[:, 1] - x[:, 0]) * (z[:, 2] - z[:, 0]) - (x[:, 2] - x[:, 0]) * (z[:, 1] - z[:, 0])
    if np.any(det <= 0):
        raise NumericalError(f"{int(np.sum(det <= 0))} degenerate or clockwise element(s) in the mesh")
    area = 0.5 * det
    # grad(phi_i) = (z_j - z_k, x_k - x_j) / (2A) with (i, j, k) cyclic
    gx = np.stack([z[:, 1] - z[:, 2], z[:, 2] - z[:, 0], z[:, 0] - z[:, 1]], axis=1) / det[:, None]
    gz = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1) / det[:, None]
    return area, np.stack([gx, gz], axis=2)


def _scatter(triangles: np.ndarray, local: np.ndarray, n: int) -> sp.csr_matrix:
    rows = np.repeat(triangles, 3, axis=1).ravel()
    cols = np.tile(triangles, (1, 3)).ravel()
    return sp.csr_matrix((local.ravel(), (rows, cols)), shape=(n, n))


def stiffness(nodes: np.ndarray, triangles: np.ndarray, coef: np.ndarray, weight: np.ndarray | None = None,
              geometry: tuple[np.ndarray, np.ndarray] | None = None) -> sp.csr_matrix:
    area, grad = geometry if geometry is not None else p1_gradients(nodes, triangles)
    c = np.asarray(coef, dtype=complex if np.iscomplexobj(coef) else float) * area
    if weight is not None:
        c = c * weight
    local = np.einsum("eid,ejd->eij", grad, grad) * c[:, None, None]
    return _scatter(triangles, local, nodes.shape[0])


_MASS = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 12.0


def mass(nodes: np.ndarray, triangles: np.ndarray, coef: np.ndarray, weight: np.ndarray | None = None,
         geometry: tuple[np.ndarray, np.ndarray] | None = None) -> sp.csr_matrix:
    area, _ = geometry if geometry is not None else p1_gradients(nodes, triangles)
    c = np.asarray(coef) * area
    if weight is not None:
        c = c * weight
    local = _MASS[None, :, :] * c[:, None, None]
    return _scatter(triangles, local, nodes.shape[0])


def load(nodes: np.ndarray, triangles: np.ndarray, source: np.ndarray, weight: np.ndarray | None = None,
         geometry: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    area, _ = geometry if geometry is not None else p1_gradients(nodes, triangles)
    s = np.asarray(source) * area / 3.0
    if weight is not None:
        s = s * weight
    f = np.zeros(nodes.shape[0], dtype=s.dtype)
    np.add.at(f, triangles.ravel(), np.repeat(s, 3))
    return f


def element_integral_vector(nodes: np.ndarray, triangles: np.ndarray, mask: np.ndarray, coef: float | np.ndarray = 1.0,
                            weight: np.ndarray | None = None, geometry: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Vector ``b_i = integral over masked elements of coef * phi_i`` (used for conductor current constraints)."""
    area, _ = geometry if geometry is not None else p1_gradients(nodes, triangles)
    s = np.where(mask, np.broadcast_to(coef, mask.shape), 0.0) * area / 3.0
    if weight is not None:
        s = s * weight
    b = np.zeros(nodes.shape[0], dtype=np.result_type(s, float))
    np.add.at(b, triangles.ravel(), np.repeat(s, 3))
    return b


def robin_edges(nodes: np.ndarray, edges: np.ndarray, h: np.ndarray, t_inf: np.ndarray | float,
                weight: np.ndarray | None = None) -> tuple[sp.csr_matrix, np.ndarray]:
    """Boundary matrix and load for ``-k dT/dn = h (T - T_inf)`` on ``edges`` (E, 2)."""
    n = nodes.shape[0]
    p = nodes[edges]
    length = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    hl = np.asarray(h) * length
    if weight is not None:
        hl = hl * weight
    local = np.array([[2.0, 1.0], [1.0, 2.0]])[None] * (hl / 6.0)[:, None, None]
    rows = np.repeat(edges, 2, axis=1).ravel()
    cols = np.tile(edges, (1, 2)).ravel()
    B = sp.csr_matrix((local.ravel(), (rows, cols)), shape=(n, n))
    f = np.zeros(n)
    np.add.at(f, edges.ravel(), np.repeat(hl * np.broadcast_to(t_inf, hl.shape) / 2.0, 2))
    return B, f


def solve_dirichlet(K: sp.spmatrix, f: np.ndarray, fixed: np.ndarray, values: np.ndarray | float = 0.0) -> np.ndarray:
    """Solve ``K u = f`` with ``u[fixed] = values`` by elimination (sparse direct solver)."""
    n = K.shape[0]
    fixed = np.asarray(fixed, dtype=np.int64)
    vals = np.broadcast_to(np.asarray(values), fixed.shape)
    u = np.zeros(n, dtype=np.result_type(K.dtype, f.dtype, vals.dtype))
    u[fixed] = vals
    fixed = np.unique(fixed)
    free = np.setdiff1d(np.arange(n), fixed)
    K = K.tocsr()
    rhs = f[free] - K[free][:, fixed] @ u[fixed]
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    if not np.all(np.isfinite(u)):
        raise NumericalError("finite-element solve produced non-finite values (singular system?)")
    return u


__all__ = ["element_integral_vector", "load", "mass", "p1_gradients", "robin_edges", "solve_dirichlet", "stiffness"]
