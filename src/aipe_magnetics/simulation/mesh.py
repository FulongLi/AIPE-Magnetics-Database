"""2-D triangular meshes of a :class:`~aipe_magnetics.geometry.section.Section2D`.

Two generators produce the same :class:`Mesh2D` data structure (linear triangles, one region index per
element, region assignment by the section's paint order evaluated at the element centroid):

``structured`` (built-in, no dependency, deterministic)
    A graded tensor-product grid whose lines include every region edge, so every grid cell lies inside exactly
    one region (the mesh conforms to the geometry exactly). Within each interval between region edges the
    spacing follows ``s(t) = min(h_max, h_left + g t, h_right + g (L - t))`` (linear growth of the cell size away
    from a fine neighbour, growth slope ``g``), i.e. a smoothly graded mesh. Each cell is split into two
    triangles with alternating diagonals.
``gmsh`` (optional, ``pip install aipe-magnetics[mesh]``; Gmsh is GPL-2.0-or-later and is used as a separate
    library, never bundled)
    Unstructured Frontal-Delaunay mesh of the fragmented rectangles with the same per-region target sizes
    (mesh-size callback).

Target sizes come from :class:`MeshSettings` (per role, overridable per region). ``refinement`` scales every
size, which is how mesh-convergence studies are run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from aipe_magnetics.common.errors import UnsupportedOperationError
from aipe_magnetics.geometry.section import Section2D

MesherName = Literal["structured", "gmsh"]


@dataclass(frozen=True)
class MeshSettings:
    """Target element sizes (m) per region role; ``refinement`` < 1 refines everything."""

    core: float = 1.5e-3
    gap: float | None = None            # None: gap thickness / 2 (at least two element rows across the gap)
    conductor: float = 0.5e-3
    stranded: float = 0.5e-3
    winding_pack: float = 1.0e-3
    bobbin: float = 1.0e-3
    insulation: float = 0.5e-3
    window_air: float = 1.0e-3
    air: float = 6.0e-3
    growth: float = 0.35                # size growth slope g (cell-size increase per unit distance)
    refinement: float = 1.0
    max_elements: int = 600_000

    def size_for(self, region: Any, axis: str | None = None) -> float:
        """Target size of ``region``; ``axis='z'`` honours an along-height override ``meta['mesh_size_z']`` (foil layers)."""
        if axis == "z" and region.meta.get("mesh_size_z") is not None:
            return float(region.meta["mesh_size_z"]) * self.refinement
        if region.mesh_size is not None:
            h = region.mesh_size
        elif region.role == "gap":
            wx, wz = region.x1 - region.x0, region.z1 - region.z0
            thin_axis = "x" if wx < wz else "z"
            if axis is not None and axis != thin_axis:
                h = self.core  # along the gap: core size (the gap is resolved across its thickness only)
            else:
                h = self.gap if self.gap is not None else min(wx, wz) / 2.0
        else:
            h = getattr(self, region.role)
        return h * self.refinement

    def to_json(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("core", "gap", "conductor", "stranded", "winding_pack", "bobbin", "insulation", "window_air", "air",
                                              "growth", "refinement", "max_elements")}


@dataclass
class Mesh2D:
    nodes: np.ndarray                   # (N, 2) x, z
    triangles: np.ndarray               # (M, 3) node indices, counter-clockwise
    element_region: np.ndarray          # (M,) index into section.regions
    section: Section2D
    generator: str
    settings: dict[str, Any] = field(default_factory=dict)
    _areas: np.ndarray | None = None

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.triangles.shape[0])

    def areas(self) -> np.ndarray:
        if self._areas is None:
            p = self.nodes[self.triangles]
            self._areas = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
        return self._areas

    def centroids(self) -> np.ndarray:
        return self.nodes[self.triangles].mean(axis=1)

    def region_mask(self, *roles: str, names: list[str] | None = None, winding: str | None = None) -> np.ndarray:
        regions = self.section.regions
        sel = np.array([(not roles or r.role in roles) and (names is None or r.name in names) and (winding is None or r.winding == winding)
                        for r in regions], dtype=bool)
        return sel[self.element_region]

    def boundary_edges(self, element_mask: np.ndarray | None = None) -> np.ndarray:
        """Edges (node pairs) used by exactly one element of the masked set, oriented as in their element."""
        tris = self.triangles if element_mask is None else self.triangles[element_mask]
        e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
        key = np.sort(e, axis=1)
        _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        return e[counts[inverse.ravel()] == 1]

    def quality(self) -> dict[str, float]:
        p = self.nodes[self.triangles]
        lengths = np.stack([np.linalg.norm(p[:, 1] - p[:, 0], axis=1), np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
                            np.linalg.norm(p[:, 0] - p[:, 2], axis=1)], axis=1)
        area = self.areas()
        # radius ratio quality 2 r_in / r_circ in (0, 1]
        s = lengths.sum(axis=1) / 2
        r_in = area / s
        r_circ = lengths.prod(axis=1) / (4 * area)
        q = 2 * r_in / r_circ
        return {"elements": float(self.n_elements), "nodes": float(self.n_nodes), "min_quality": float(q.min()),
                "mean_quality": float(q.mean()), "min_area": float(area.min()), "max_edge": float(lengths.max())}

    def submesh(self, element_mask: np.ndarray) -> tuple["Mesh2D", np.ndarray]:
        """Mesh of the masked elements with renumbered nodes; returns ``(mesh, old_node_index_of_new_nodes)``."""
        tris = self.triangles[element_mask]
        used = np.unique(tris)
        remap = -np.ones(self.n_nodes, dtype=np.int64)
        remap[used] = np.arange(used.size)
        sub = Mesh2D(nodes=self.nodes[used], triangles=remap[tris], element_region=self.element_region[element_mask],
                     section=self.section, generator=self.generator + "+submesh", settings=self.settings)
        return sub, used

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        return {"nodes": self.nodes, "triangles": self.triangles, "element_region": self.element_region,
                "region_names": np.array([r.name for r in self.section.regions])}


# -------------------------------------------------------------------------------------------------
# structured graded mesher
# -------------------------------------------------------------------------------------------------
def _graded_points(a: float, b: float, h_left: float, h_right: float, h_max: float, growth: float) -> np.ndarray:
    L = b - a
    if L <= 0:
        return np.array([a])
    t = np.linspace(0.0, L, 2001)
    s = np.minimum(h_max, np.minimum(h_left + growth * t, h_right + growth * (L - t)))
    s = np.maximum(s, 1e-12)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (1 / s[1:] + 1 / s[:-1]) * np.diff(t))])
    n = max(1, int(math.ceil(cum[-1] - 1e-9)))
    targets = np.linspace(0.0, cum[-1], n + 1)
    pts = a + np.interp(targets, cum, t)
    pts[0], pts[-1] = a, b
    return pts


def _axis_lines(breaks: np.ndarray, interval_size: np.ndarray, growth: float) -> np.ndarray:
    node_size = np.empty(breaks.size)
    node_size[0] = interval_size[0]
    node_size[-1] = interval_size[-1]
    node_size[1:-1] = np.minimum(interval_size[:-1], interval_size[1:])
    out = [np.array([breaks[0]])]
    for i in range(breaks.size - 1):
        pts = _graded_points(breaks[i], breaks[i + 1], node_size[i], node_size[i + 1], interval_size[i], growth)
        out.append(pts[1:])
    return np.concatenate(out)


def structured_mesh(section: Section2D, settings: MeshSettings | None = None) -> Mesh2D:
    settings = settings or MeshSettings()
    regions = section.regions
    x0, x1, z0, z1 = section.bounds
    xb = np.unique(np.round(np.array([v for r in regions for v in (r.x0, r.x1)]), 12))
    zb = np.unique(np.round(np.array([v for r in regions for v in (r.z0, r.z1)]), 12))
    def interval_sizes(breaks: np.ndarray, lo_attr: str, hi_attr: str) -> np.ndarray:
        sizes = [settings.size_for(r, "z" if lo_attr == "z0" else "x") for r in regions]
        out = np.empty(breaks.size - 1)
        for i in range(breaks.size - 1):
            a, b = breaks[i], breaks[i + 1]
            h = math.inf
            for r, size in zip(regions, sizes):
                if getattr(r, lo_attr) < b - 1e-15 and getattr(r, hi_attr) > a + 1e-15:
                    h = min(h, size)
            out[i] = h if math.isfinite(h) else settings.air * settings.refinement
        return out

    xs = _axis_lines(xb, interval_sizes(xb, "x0", "x1"), settings.growth)
    zs = _axis_lines(zb, interval_sizes(zb, "z0", "z1"), settings.growth)
    nx, nz = xs.size, zs.size
    n_elem = 2 * (nx - 1) * (nz - 1)
    if n_elem > settings.max_elements:
        raise UnsupportedOperationError(f"structured mesh would have {n_elem} elements (> max_elements {settings.max_elements}); "
                                        "coarsen the settings or raise max_elements", details={"elements": n_elem})
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    nodes = np.column_stack([X.ravel(), Z.ravel()])

    def nid(i: np.ndarray, k: np.ndarray) -> np.ndarray:
        return i * nz + k

    I, K = np.meshgrid(np.arange(nx - 1), np.arange(nz - 1), indexing="ij")
    I, K = I.ravel(), K.ravel()
    n00, n10, n11, n01 = nid(I, K), nid(I + 1, K), nid(I + 1, K + 1), nid(I, K + 1)
    alt = ((I + K) % 2 == 0)
    t1 = np.where(alt[:, None], np.column_stack([n00, n10, n11]), np.column_stack([n00, n10, n01]))
    t2 = np.where(alt[:, None], np.column_stack([n00, n11, n01]), np.column_stack([n10, n11, n01]))
    tris = np.concatenate([t1, t2])
    # cell-centre classification (cells never straddle a region edge)
    cx = 0.5 * (xs[I] + xs[I + 1])
    cz = 0.5 * (zs[K] + zs[K + 1])
    cell_region = _classify(section, cx, cz)
    elem_region = np.concatenate([cell_region, cell_region])
    keep = elem_region >= 0
    mesh = Mesh2D(nodes=nodes, triangles=tris[keep], element_region=elem_region[keep], section=section, generator="structured",
                  settings=settings.to_json() | {"nx": int(nx), "nz": int(nz)})
    return mesh


def _classify(section: Section2D, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    out = -np.ones(x.shape, dtype=np.int64)
    for i, r in enumerate(section.regions):  # paint order: later regions overwrite earlier ones
        inside = (x >= r.x0) & (x <= r.x1) & (z >= r.z0) & (z <= r.z1)
        out[inside] = i
    return out


# -------------------------------------------------------------------------------------------------
# gmsh mesher
# -------------------------------------------------------------------------------------------------
def gmsh_available() -> tuple[bool, str]:
    try:
        import gmsh  # noqa: F401
        return True, getattr(gmsh, "__version__", "?")
    except Exception as error:  # noqa: BLE001 - availability probe
        return False, f"{type(error).__name__}: {error}"


def gmsh_mesh(section: Section2D, settings: MeshSettings | None = None) -> Mesh2D:
    ok, info = gmsh_available()
    if not ok:
        raise UnsupportedOperationError(f"gmsh mesher unavailable ({info}); install with pip install 'aipe-magnetics[mesh]' "
                                        "or use mesher='structured'")
    import gmsh

    settings = settings or MeshSettings()
    regions = section.regions
    sizes = np.array([settings.size_for(r) for r in regions])
    rx0 = np.array([r.x0 for r in regions]); rx1 = np.array([r.x1 for r in regions])
    rz0 = np.array([r.z0 for r in regions]); rz1 = np.array([r.z1 for r in regions])

    def size_at(x: float, z: float) -> float:
        inside = (x >= rx0 - 1e-12) & (x <= rx1 + 1e-12) & (z >= rz0 - 1e-12) & (z <= rz1 + 1e-12)
        return float(sizes[inside].min()) if inside.any() else float(settings.air * settings.refinement)

    initialised_here = not gmsh.isInitialized()
    if initialised_here:
        gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("aipe_section")
        tags = [(2, gmsh.model.occ.addRectangle(r.x0, r.z0, 0.0, r.x1 - r.x0, r.z1 - r.z0)) for r in regions]
        gmsh.model.occ.fragment(tags[:1], tags[1:])
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: size_at(x, y))
        gmsh.model.mesh.generate(2)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = coords.reshape(-1, 3)[:, :2]
        index = {int(t): i for i, t in enumerate(node_tags)}
        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        tri = None
        for et, en in zip(etypes, enodes):
            if et == 2:
                tri = np.array([index[int(t)] for t in en], dtype=np.int64).reshape(-1, 3)
        if tri is None:
            raise UnsupportedOperationError("gmsh produced no triangles")
    finally:
        gmsh.model.remove()
        if initialised_here:
            gmsh.finalize()
    nodes = coords
    p = nodes[tri]
    signed = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    flip = signed < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    c = nodes[tri].mean(axis=1)
    region = _classify(section, c[:, 0], c[:, 1])
    keep = region >= 0
    return Mesh2D(nodes=nodes, triangles=tri[keep], element_region=region[keep], section=section, generator=f"gmsh {info}",
                  settings=settings.to_json())


def generate_mesh(section: Section2D, settings: MeshSettings | None = None, mesher: MesherName = "structured") -> Mesh2D:
    if mesher == "structured":
        return structured_mesh(section, settings)
    if mesher == "gmsh":
        return gmsh_mesh(section, settings)
    raise UnsupportedOperationError(f"unknown mesher {mesher!r}; use 'structured' or 'gmsh'")


__all__ = ["Mesh2D", "MeshSettings", "MesherName", "generate_mesh", "gmsh_available", "gmsh_mesh", "structured_mesh"]
