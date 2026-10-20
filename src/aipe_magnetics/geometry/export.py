"""CAD export of the common geometry representation.

Backends
--------
``builtin`` (always available, no dependency)
    * **STEP AP214** (``AUTOMOTIVE_DESIGN``) faceted boundary representation: every solid becomes a
      ``MANIFOLD_SOLID_BREP`` of planar ``ADVANCED_FACE`` s bounded by straight edges; circular corners of the
      winding / bobbin rings are discretised with ``arc_segments`` segments per quarter circle (exact for the
      core halves, which are polygon prisms). One product, one named B-rep body per part.
    * **STL** (ASCII) triangulated surfaces, one ``solid`` block per part.
``cadquery`` (optional, ``pip install aipe-magnetics[cad]``)
    Exact B-rep (true circular arcs) built with CadQuery / OpenCascade, exported as a named STEP assembly with
    one part per solid, plus STL and BREP files. See :mod:`aipe_magnetics.geometry.cad_cadquery`.

Both backends write the same parts with the same names and coordinates (millimetres in the files); the tests
re-import the built-in STEP file with OpenCascade (when installed) and compare every body volume with the exact
analytical volume.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Literal

from aipe_magnetics.geometry.primitives import Face, Solid, solid_triangles

MM = 1e3  # metres -> millimetres in exported files


def _real(v: float) -> str:
    """STEP real literal (always with a decimal point)."""
    if v == 0.0:
        return "0."
    text = f"{v:.12G}"
    if "E" in text:
        mant, exp = text.split("E")
        if "." not in mant:
            mant += "."
        return f"{mant}E{int(exp)}"
    return text if "." in text else text + "."


def _string(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


class _StepWriter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.n = 0

    def add(self, text: str) -> int:
        self.n += 1
        self.lines.append(f"#{self.n}={text};")
        return self.n


def _norm(v: tuple[float, float, float]) -> tuple[float, float, float]:
    import math
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return (v[0] / n, v[1] / n, v[2] / n)


def _ref_direction(normal: tuple[float, float, float]) -> tuple[float, float, float]:
    """A unit vector orthogonal to ``normal``."""
    import numpy as np
    n = np.asarray(normal)
    trial = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    r = trial - n * float(np.dot(trial, n))
    r /= np.linalg.norm(r)
    return (float(r[0]), float(r[1]), float(r[2]))


def write_step(solids: Iterable[Solid], path: str | Path, *, product: str = "aipe_magnetic", arc_segments: int = 16,
               timestamp: str = "2026-01-01T00:00:00", author: str = "AIPE Magnetics") -> Path:
    """Write a faceted AP214 STEP file (millimetres). ``timestamp`` is fixed by default so output is reproducible."""
    from aipe_magnetics import __version__
    from aipe_magnetics.geometry.primitives import face_normal

    w = _StepWriter()
    ctx = w.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
    w.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{ctx})")
    pctx = w.add(f"PRODUCT_CONTEXT('',#{ctx},'mechanical')")
    pdctx = w.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{ctx},'design')")
    prod = w.add(f"PRODUCT({_string(product)},{_string(product)},'',(#{pctx}))")
    pdf = w.add(f"PRODUCT_DEFINITION_FORMATION('','',#{prod})")
    pd = w.add(f"PRODUCT_DEFINITION('design','',#{pdf},#{pdctx})")
    pds = w.add(f"PRODUCT_DEFINITION_SHAPE('','',#{pd})")
    u_len = w.add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
    u_ang = w.add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    u_sol = w.add("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    unc = w.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-06),#{u_len},'distance_accuracy_value','confusion accuracy')")
    gctx = w.add(f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc}))"
                 f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{u_len},#{u_ang},#{u_sol}))REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY'))")
    origin = w.add("CARTESIAN_POINT('',(0.,0.,0.))")
    dz = w.add("DIRECTION('',(0.,0.,1.))")
    dx = w.add("DIRECTION('',(1.,0.,0.))")
    world = w.add(f"AXIS2_PLACEMENT_3D('',#{origin},#{dz},#{dx})")
    bodies: list[int] = []
    for solid in solids:
        points: dict[tuple[float, float, float], int] = {}
        vertices: dict[tuple[float, float, float], int] = {}
        edges: dict[tuple[tuple[float, ...], tuple[float, ...]], int] = {}

        def key(p: tuple[float, float, float]) -> tuple[float, float, float]:
            return (round(p[0] * MM, 9), round(p[1] * MM, 9), round(p[2] * MM, 9))

        def vertex(p: tuple[float, float, float]) -> int:
            k = key(p)
            if k not in vertices:
                cp = w.add(f"CARTESIAN_POINT('',({_real(k[0])},{_real(k[1])},{_real(k[2])}))")
                points[k] = cp
                vertices[k] = w.add(f"VERTEX_POINT('',#{cp})")
            return vertices[k]

        def edge(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[int, bool]:
            ka, kb = key(a), key(b)
            if (kb, ka) in edges:
                return edges[(kb, ka)], False
            if (ka, kb) not in edges:
                va, vb = vertex(a), vertex(b)
                d = (kb[0] - ka[0], kb[1] - ka[1], kb[2] - ka[2])
                import math
                length = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
                u = _norm(d)
                direc = w.add(f"DIRECTION('',({_real(u[0])},{_real(u[1])},{_real(u[2])}))")
                vec = w.add(f"VECTOR('',#{direc},{_real(length)})")
                line = w.add(f"LINE('',#{points[ka]},#{vec})")
                edges[(ka, kb)] = w.add(f"EDGE_CURVE('',#{va},#{vb},#{line},.T.)")
            return edges[(ka, kb)], True

        def loop(pts: list[tuple[float, float, float]]) -> int:
            oes = []
            clean = [p for i, p in enumerate(pts) if key(p) != key(pts[i - 1])]
            for i in range(len(clean)):
                e, sense = edge(clean[i], clean[(i + 1) % len(clean)])
                oes.append(w.add(f"ORIENTED_EDGE('',*,*,#{e},{'.T.' if sense else '.F.'})"))
            return w.add("EDGE_LOOP('',(" + ",".join(f"#{o}" for o in oes) + "))")

        face_ids = []
        for face in solid.faces(arc_segments):
            n = face_normal(face)
            nrm = (float(n[0]), float(n[1]), float(n[2]))
            ref = _ref_direction(nrm)
            p0 = key(face.outer[0])
            loc = w.add(f"CARTESIAN_POINT('',({_real(p0[0])},{_real(p0[1])},{_real(p0[2])}))")
            ax = w.add(f"DIRECTION('',({_real(nrm[0])},{_real(nrm[1])},{_real(nrm[2])}))")
            rf = w.add(f"DIRECTION('',({_real(ref[0])},{_real(ref[1])},{_real(ref[2])}))")
            placement = w.add(f"AXIS2_PLACEMENT_3D('',#{loc},#{ax},#{rf})")
            plane = w.add(f"PLANE('',#{placement})")
            bounds = [w.add(f"FACE_OUTER_BOUND('',#{loop(list(face.outer))},.T.)")]
            for hole in face.holes:
                bounds.append(w.add(f"FACE_BOUND('',#{loop(list(hole))},.T.)"))
            face_ids.append(w.add("ADVANCED_FACE('',(" + ",".join(f"#{b}" for b in bounds) + f"),#{plane},.T.)"))
        shell = w.add("CLOSED_SHELL('',(" + ",".join(f"#{f}" for f in face_ids) + "))")
        bodies.append(w.add(f"MANIFOLD_SOLID_BREP({_string(solid.name)},#{shell})"))
    rep = w.add("ADVANCED_BREP_SHAPE_REPRESENTATION('',(#" + str(world) + "," + ",".join(f"#{b}" for b in bodies) + f"),#{gctx})")
    w.add(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")
    header = ["ISO-10303-21;", "HEADER;",
              f"FILE_DESCRIPTION(('AIPE Magnetics parametric magnetic component (faceted B-rep, mm)'),'2;1');",
              f"FILE_NAME({_string(Path(path).name)},{_string(timestamp)},({_string(author)}),(''),"
              f"{_string('aipe-magnetics ' + __version__)},'aipe-magnetics builtin STEP writer','');",
              "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));", "ENDSEC;", "DATA;"]
    text = "\n".join(header + w.lines + ["ENDSEC;", "END-ISO-10303-21;"]) + "\n"
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="ascii", newline="\n")
    return out


def write_stl(solids: Iterable[Solid], path: str | Path, *, arc_segments: int = 16) -> Path:
    """ASCII STL (millimetres), one ``solid <name>`` block per part, outward normals."""
    import numpy as np
    lines: list[str] = []
    for solid in solids:
        lines.append(f"solid {solid.name}")
        for a, b, c in solid_triangles(solid, arc_segments):
            pa, pb, pc = (np.asarray(p) * MM for p in (a, b, c))
            n = np.cross(pb - pa, pc - pa)
            norm = np.linalg.norm(n)
            n = n / norm if norm > 0 else n
            lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
            lines.append("    outer loop")
            for p in (pa, pb, pc):
                lines.append(f"      vertex {p[0]:.9e} {p[1]:.9e} {p[2]:.9e}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append(f"endsolid {solid.name}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    return out


def cad_backends() -> dict[str, dict[str, Any]]:
    """Availability of the CAD backends (never raises)."""
    out: dict[str, dict[str, Any]] = {"builtin": {"available": True, "formats": ["step", "stl"], "exact_arcs": False,
                                                  "notes": "faceted planar B-rep; no dependency"}}
    try:
        import cadquery  # noqa: F401
        out["cadquery"] = {"available": True, "version": getattr(cadquery, "__version__", "?"), "formats": ["step", "stl", "brep"],
                           "exact_arcs": True, "notes": "OpenCascade exact B-rep, named STEP assembly"}
    except Exception as error:  # noqa: BLE001 - availability probe
        out["cadquery"] = {"available": False, "reason": f"{type(error).__name__}: {error}", "install": "pip install 'aipe-magnetics[cad]'"}
    return out


def file_record(path: Path, media_type: str, role: str, description: str) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "media_type": media_type, "role": role, "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "description": description}


def export_cad(geometry: Any, directory: str | Path, *, basename: str | None = None, formats: Iterable[str] = ("step", "stl"),
               backend: Literal["auto", "builtin", "cadquery"] = "auto", arc_segments: int = 16) -> dict[str, Any]:
    """Export ``geometry`` (a :class:`~aipe_magnetics.geometry.ecore.MagneticGeometry`) and return file records.

    ``auto`` uses CadQuery when it is installed (exact arcs) and *additionally* always writes the built-in faceted
    STEP so a dependency-free file exists; the record states which backend wrote which file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = basename or geometry.model.name
    formats = [f.lower() for f in formats]
    unknown = sorted(set(formats) - {"step", "stl", "brep"})
    if unknown:
        raise ValueError(f"unsupported CAD format(s) {unknown}")
    backends = cad_backends()
    use_cq = backend == "cadquery" or (backend == "auto" and backends["cadquery"]["available"])
    if backend == "cadquery" and not backends["cadquery"]["available"]:
        from aipe_magnetics.common.errors import UnsupportedOperationError
        raise UnsupportedOperationError(f"CadQuery backend unavailable: {backends['cadquery']['reason']}; "
                                        "install with pip install 'aipe-magnetics[cad]' or use backend='builtin'")
    files: list[dict[str, Any]] = []
    if "step" in formats and (backend in ("auto", "builtin")):
        p = write_step(geometry.solids, directory / f"{name}.faceted.step", product=name, arc_segments=arc_segments)
        files.append(file_record(p, "model/step", "cad", "STEP AP214 faceted B-rep (builtin writer, mm)") | {"backend": "builtin"})
    if "stl" in formats and not use_cq:
        p = write_stl(geometry.solids, directory / f"{name}.stl", arc_segments=arc_segments)
        files.append(file_record(p, "model/stl", "cad", "ASCII STL triangulation (builtin writer, mm)") | {"backend": "builtin"})
    if use_cq:
        from aipe_magnetics.geometry.cad_cadquery import export_cadquery
        files.extend(export_cadquery(geometry, directory, name, formats))
    return {"directory": str(directory), "files": files, "backends": {k: v.get("available") for k, v in backends.items()},
            "units": "mm", "parts": [{"name": s.name, "role": s.role, "material": s.material, "volume_mm3": s.volume() * 1e9}
                                     for s in geometry.solids]}


__all__ = ["cad_backends", "export_cad", "file_record", "write_step", "write_stl", "Face"]
