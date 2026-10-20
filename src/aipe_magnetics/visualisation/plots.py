"""Engineering plots (matplotlib, optional: ``pip install aipe-magnetics[viz]``).

Every function writes one PNG and returns its path. Plots are generated from computed data only (geometry, mesh
fields, thermal solutions, optimisation records); no plot is decorative or synthetic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from aipe_magnetics.common.errors import UnsupportedOperationError

ROLE_COLOURS = {"core": "#4a4a52", "gap": "#ffffff", "conductor": "#c8692b", "stranded": "#d98a4b", "winding_pack": "#f2c9a6",
                "bobbin": "#e8e4d0", "insulation": "#f0c040", "window_air": "#eef3f8", "air": "#ffffff"}


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as error:  # noqa: BLE001
        raise UnsupportedOperationError(f"plots need matplotlib ({error}); pip install 'aipe-magnetics[viz]'") from None


def matplotlib_available() -> bool:
    try:
        _plt()
        return True
    except UnsupportedOperationError:
        return False


def _save(fig: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": "aipe-magnetics"})
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def plot_section(section: Any, path: str | Path, title: str = "Cross-section (y = 0, x >= 0)") -> Path:
    plt = _plt()
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(6, 7))
    x0, x1, z0, z1 = section.bounds
    for r in section.regions:
        if r.role == "air":
            continue
        ax.add_patch(Rectangle((r.x0 * 1e3, r.z0 * 1e3), (r.x1 - r.x0) * 1e3, (r.z1 - r.z0) * 1e3, facecolor=ROLE_COLOURS.get(r.role, "#999"),
                               edgecolor="#333" if r.role in ("conductor", "stranded", "core") else "none", linewidth=0.3))
    win = section.meta.get("window", {})
    pad = 2.0
    xmax = max(r.x1 for r in section.regions if r.role == "core") * 1e3 + pad
    zmax = max(r.z1 for r in section.regions if r.role == "core") * 1e3 + pad
    ax.set_xlim(0, xmax)
    ax.set_ylim(-zmax, zmax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(title)
    handles = [Rectangle((0, 0), 1, 1, facecolor=ROLE_COLOURS[k]) for k in ("core", "conductor", "stranded", "winding_pack", "bobbin", "insulation")]
    ax.legend(handles, ["core", "solid conductor", "litz bundle", "winding pack", "bobbin", "insulation"], loc="upper right", fontsize=7)
    del win
    return _save(fig, path)


def plot_geometry_3d(geometry: Any, path: str | Path, title: str = "Parametric 3-D model", arc_segments: int = 8) -> Path:
    plt = _plt()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from aipe_magnetics.geometry.primitives import solid_triangles
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    colours = {"core": "#4a4a52", "bobbin": "#d9d4bc", "winding": "#c8692b", "insulation": "#f0c040"}
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for solid in geometry.solids:
        if solid.role == "insulation":  # thin wraps would hide the copper; they are in the CAD files
            continue
        tris = np.asarray(solid_triangles(solid, arc_segments)) * 1e3
        lo = np.minimum(lo, tris.reshape(-1, 3).min(axis=0))
        hi = np.maximum(hi, tris.reshape(-1, 3).max(axis=0))
        coll = Poly3DCollection(tris, facecolor=colours.get(solid.role, "#888"), edgecolor="none",
                                alpha={"core": 0.22, "bobbin": 0.55}.get(solid.role, 0.95))
        ax.add_collection3d(coll)
    c = 0.5 * (lo + hi)
    r = 0.5 * float(np.max(hi - lo))
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.view_init(elev=22, azim=-58)
    ax.set_title(title + " (core and bobbin translucent; insulation wraps not drawn)")
    return _save(fig, path)


def plot_element_field(mesh: Any, values: np.ndarray, path: str | Path, *, title: str, label: str, roles: Iterable[str] | None = None,
                       cmap: str = "viridis", vmax: float | None = None) -> Path:
    plt = _plt()
    import matplotlib.tri as mtri
    fig, ax = plt.subplots(figsize=(6, 7))
    keep = mesh.region_mask(*roles) if roles else np.ones(mesh.n_elements, dtype=bool)
    tri = mtri.Triangulation(mesh.nodes[:, 0] * 1e3, mesh.nodes[:, 1] * 1e3, mesh.triangles[keep])
    v = np.asarray(values)[keep]
    tpc = ax.tripcolor(tri, facecolors=v, cmap=cmap, vmin=0.0, vmax=vmax if vmax is not None else float(np.percentile(v, 99.5)), shading="flat")
    fig.colorbar(tpc, ax=ax, label=label, shrink=0.8)
    core = mesh.region_mask("core")
    xmax = mesh.nodes[mesh.triangles[core]][:, :, 0].max() * 1e3 + 2
    zmax = mesh.nodes[mesh.triangles[core]][:, :, 1].max() * 1e3 + 2
    ax.set_xlim(0, xmax)
    ax.set_ylim(-zmax, zmax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(title)
    return _save(fig, path)


def plot_temperature(thermal: Any, path: str | Path, title: str = "Temperature (2.5-D thermal FE)") -> Path:
    plt = _plt()
    import matplotlib.tri as mtri
    mesh = thermal.mesh
    fig, ax = plt.subplots(figsize=(6, 7))
    tri = mtri.Triangulation(mesh.nodes[:, 0] * 1e3, mesh.nodes[:, 1] * 1e3, mesh.triangles)
    t = thermal.T - 273.15
    cs = ax.tricontourf(tri, t, levels=24, cmap="inferno")
    fig.colorbar(cs, ax=ax, label="T (degC)", shrink=0.8)
    for name, hs in thermal.hotspots.items():
        ax.plot(hs["x_m"] * 1e3, hs["z_m"] * 1e3, "c^", ms=6)
        ax.annotate(f"{name}: {hs['T_K'] - 273.15:.0f} degC", (hs["x_m"] * 1e3, hs["z_m"] * 1e3), fontsize=7, color="cyan",
                    xytext=(4, 4), textcoords="offset points")
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(title)
    return _save(fig, path)


def plot_loss_breakdown(breakdowns: dict[str, dict[str, float]], path: str | Path, title: str = "Loss breakdown") -> Path:
    """``{design_label: {component: W}}`` stacked bars."""
    plt = _plt()
    labels = list(breakdowns)
    parts = sorted({k for d in breakdowns.values() for k in d})
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(labels)), 4.5))
    bottom = np.zeros(len(labels))
    palette = ["#4a4a52", "#c8692b", "#2b7bc8", "#6aa84f", "#9e5ec4", "#e0a030"]
    for i, part in enumerate(parts):
        vals = np.array([breakdowns[l].get(part, 0.0) for l in labels])
        ax.bar(labels, vals, bottom=bottom, label=part, color=palette[i % len(palette)])
        bottom += vals
    for x, total in enumerate(bottom):
        ax.text(x, total, f"{total:.1f} W", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("loss (W)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    return _save(fig, path)


def plot_candidates(rows: list[dict[str, Any]], path: str | Path, *, x: str = "volume_cm3", y: str = "total_loss_W",
                    pareto_ids: set[str] | None = None, selected_id: str | None = None, title: str = "Candidate designs") -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for fidelity, marker in (("analytical", "o"), ("fea", "s")):
        feas = [r for r in rows if r["fidelity"] == fidelity and r["feasible"] and x in r["objectives"] and y in r["objectives"]]
        inf = [r for r in rows if r["fidelity"] == fidelity and not r["feasible"] and x in r["objectives"] and y in r["objectives"]]
        if inf:
            ax.scatter([r["objectives"][x] for r in inf], [r["objectives"][y] for r in inf], marker=marker, c="#cccccc", s=14,
                       label=f"infeasible ({fidelity})")
        if feas:
            ax.scatter([r["objectives"][x] for r in feas], [r["objectives"][y] for r in feas], marker=marker,
                       c="#2b7bc8" if fidelity == "analytical" else "#c8692b", s=18 if fidelity == "analytical" else 40,
                       label=f"feasible ({fidelity})")
    if pareto_ids:
        pr = sorted([r for r in rows if r["id"] in pareto_ids and r["fidelity"] == "analytical"], key=lambda r: r["objectives"][x])
        if pr:
            ax.plot([r["objectives"][x] for r in pr], [r["objectives"][y] for r in pr], "k--", lw=0.8, label="Pareto front (analytical)")
    if selected_id:
        sel = [r for r in rows if r["id"] == selected_id and r["fidelity"] == "fea"]
        if sel:
            ax.scatter([sel[0]["objectives"][x]], [sel[0]["objectives"][y]], marker="*", s=220, c="gold", edgecolors="k", label="selected")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend(fontsize=7)
    return _save(fig, path)


def plot_convergence(history: list[dict[str, Any]], path: str | Path, title: str = "Optimisation progress", key: str = "best_primary") -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    stages = [h["stage"] for h in history if h.get(key) is not None]
    vals = [h[key] for h in history if h.get(key) is not None]
    ax.plot(range(len(vals)), vals, "o-")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(stages, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(key)
    ax.set_title(title)
    return _save(fig, path)


def plot_waveforms(waveforms: dict[str, Any], path: str | Path, *, period: float, title: str = "Excitation waveforms") -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    t = np.linspace(0, 1, 801)
    for label, w in waveforms.items():
        ax.plot(t * period * 1e6, w.value(t), label=f"{label} (rms {w.rms():.2f} {w.unit})")
    ax.set_xlabel("t (us)")
    ax.set_ylabel("current (A)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_electrothermal(history: list[dict[str, float]], path: str | Path, title: str = "Electrothermal iteration") -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    it = [h["iteration"] for h in history]
    ax.plot(it, [h["T_core_out_K"] - 273.15 for h in history], "o-", label="core mean T")
    ax.plot(it, [h["T_winding_out_K"] - 273.15 for h in history], "s-", label="winding mean T")
    ax2 = ax.twinx()
    ax2.plot(it, [h["P_core_W"] + h["P_winding_W"] for h in history], "k:", label="total loss")
    ax.set_xlabel("iteration")
    ax.set_ylabel("T (degC)")
    ax2.set_ylabel("loss (W)")
    ax.set_title(title)
    ax.legend(loc="center right", fontsize=8)
    return _save(fig, path)


__all__ = ["matplotlib_available", "plot_candidates", "plot_convergence", "plot_electrothermal", "plot_element_field", "plot_geometry_3d",
           "plot_loss_breakdown", "plot_section", "plot_temperature", "plot_waveforms"]
