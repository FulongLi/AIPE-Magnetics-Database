"""CadQuery / OpenCascade backend: exact B-rep solids (true arcs) and a named STEP assembly.

Optional dependency (``pip install aipe-magnetics[cad]``; CadQuery is Apache-2.0, OCP/OCCT LGPL-2.1 with
exception). Imported lazily by :func:`aipe_magnetics.geometry.export.export_cad`; nothing in the Database layer
imports it.

Each primitive of the common representation maps to one CadQuery solid:

* :class:`~aipe_magnetics.geometry.primitives.ExtrudedProfile` - polyline in the XZ plane extruded along +Y;
* :class:`~aipe_magnetics.geometry.primitives.RoundedRectRing` - offset rectangle wire (``Wire.offset2D`` with
  arc corners) minus the inner offset, extruded along Z (square corners when ``rounded=False`` or for the
  explicit outer rectangle of a flange).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from aipe_magnetics.geometry.primitives import ExtrudedProfile, RoundedRectRing, Solid

MM = 1e3

_COLOURS = {"core": (0.25, 0.25, 0.28), "bobbin": (0.9, 0.9, 0.85), "winding": (0.85, 0.45, 0.2), "insulation": (0.95, 0.75, 0.2)}


def _offset_face(cq: Any, hx: float, hy: float, d: float, rounded: bool):
    base = cq.Workplane("XY").rect(2 * hx * MM, 2 * hy * MM)
    if d <= 0:
        return base.val() if False else cq.Wire.makePolygon([cq.Vector(-hx * MM, -hy * MM, 0), cq.Vector(hx * MM, -hy * MM, 0),
                                                            cq.Vector(hx * MM, hy * MM, 0), cq.Vector(-hx * MM, hy * MM, 0)], close=True)
    if rounded:
        wire = cq.Wire.makePolygon([cq.Vector(-hx * MM, -hy * MM, 0), cq.Vector(hx * MM, -hy * MM, 0),
                                    cq.Vector(hx * MM, hy * MM, 0), cq.Vector(-hx * MM, hy * MM, 0)], close=True)
        return wire.offset2D(d * MM, "arc")[0]
    ex, ey = (hx + d) * MM, (hy + d) * MM
    return cq.Wire.makePolygon([cq.Vector(-ex, -ey, 0), cq.Vector(ex, -ey, 0), cq.Vector(ex, ey, 0), cq.Vector(-ex, ey, 0)], close=True)


def build_solid(solid: Solid):
    import cadquery as cq

    if isinstance(solid, ExtrudedProfile):
        pts = [(x * MM, z * MM) for x, z in solid.ccw_profile]
        # XZ workplane: local x -> global X, local y -> global Z, normal -> -Y; extrude then translate to [y0, y1]
        wp = cq.Workplane("XZ").polyline(pts).close().extrude(-(solid.y1 - solid.y0) * MM)
        return wp.translate((0, solid.y0 * MM, 0)).val()
    if isinstance(solid, RoundedRectRing):
        height = (solid.z1 - solid.z0) * MM
        if solid.outer_override is not None:
            ox, oy = solid.outer_override
            outer = cq.Wire.makePolygon([cq.Vector(-ox * MM, -oy * MM, 0), cq.Vector(ox * MM, -oy * MM, 0),
                                         cq.Vector(ox * MM, oy * MM, 0), cq.Vector(-ox * MM, oy * MM, 0)], close=True)
        else:
            outer = _offset_face(cq, solid.hx, solid.hy, solid.d_out, solid.rounded)
        inner = _offset_face(cq, solid.hx, solid.hy, solid.d_in, solid.rounded)
        face = cq.Face.makeFromWires(outer, [inner])
        body = cq.Solid.extrudeLinear(face, cq.Vector(0, 0, height))
        return body.translate(cq.Vector(0, 0, solid.z0 * MM))
    raise TypeError(f"unsupported primitive {type(solid).__name__}")


def export_cadquery(geometry: Any, directory: str | Path, name: str, formats: Iterable[str]) -> list[dict[str, Any]]:
    import cadquery as cq

    from aipe_magnetics.geometry.export import file_record

    directory = Path(directory)
    assembly = cq.Assembly(name=name)
    shapes = []
    for solid in geometry.solids:
        shape = build_solid(solid)
        shapes.append(shape)
        colour = _COLOURS.get(solid.role, (0.6, 0.6, 0.6))
        assembly.add(shape, name=solid.name, color=cq.Color(*colour))
    files: list[dict[str, Any]] = []
    formats = list(formats)
    if "step" in formats:
        p = directory / f"{name}.step"
        assembly.export(str(p))
        files.append(file_record(p, "model/step", "cad", "STEP AP214 assembly, exact B-rep (CadQuery/OpenCascade, mm)") | {"backend": "cadquery"})
    compound = cq.Compound.makeCompound(shapes)
    if "stl" in formats:
        p = directory / f"{name}.stl"
        cq.exporters.export(cq.Workplane().add(compound), str(p), exportType="STL", tolerance=0.01, angularTolerance=0.1)
        files.append(file_record(p, "model/stl", "cad", "STL triangulation (CadQuery, mm)") | {"backend": "cadquery"})
    if "brep" in formats:
        p = directory / f"{name}.brep"
        compound.exportBrep(str(p))
        files.append(file_record(p, "application/octet-stream", "cad", "OpenCascade BREP (mm)") | {"backend": "cadquery"})
    return files


def step_volumes(path: str | Path) -> list[float]:
    """Volumes (mm^3) of the solids in a STEP file, read back with OpenCascade (verification helper)."""
    import cadquery as cq
    shape = cq.importers.importStep(str(path))
    return [s.Volume() for s in shape.solids().vals()]


__all__ = ["build_solid", "export_cadquery", "step_volumes"]
