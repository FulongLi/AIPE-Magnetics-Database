"""Backend-neutral solid primitives of the common geometry representation.

Two primitives describe every part of a gapped E-core magnetic component exactly:

* :class:`ExtrudedProfile` - a simple polygon in the XZ plane extruded along Y (core halves).
* :class:`RoundedRectRing` - the region between two offset curves of a rectangle (an annulus that follows a
  rectangular centre leg at a constant distance, with circular corners), extruded along Z (bobbin tube and
  flanges, winding layers, insulation wraps). With ``corner_radius=0`` offsets the ring has square corners.

Coordinates: metres; Z is the centre-leg axis, X runs along the core length, Y along the core depth; the
core mating plane is ``z = 0``.

Every primitive provides its exact analytical volume, a polygonal boundary representation (planar faces with
optional holes; circular arcs discretised with a stated number of segments) for the dependency-free STEP / STL
writers, and the exact parameters used by the CadQuery backend (true arcs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class Face:
    """A planar face: outer loop (counter-clockwise seen from outside) and hole loops (clockwise)."""

    outer: list[Point3]
    holes: list[list[Point3]] = field(default_factory=list)


def polygon_area(points: list[Point2]) -> float:
    """Signed shoelace area (positive for counter-clockwise)."""
    s = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        s += x0 * y1 - x1 * y0
    return 0.5 * s


@dataclass(frozen=True)
class ExtrudedProfile:
    """Polygon ``profile`` (x, z) extruded from ``y0`` to ``y1``."""

    name: str
    role: str
    material: str
    profile: tuple[Point2, ...]
    y0: float
    y1: float
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.profile) < 3:
            raise ValueError(f"{self.name}: profile needs at least 3 points")
        if not self.y1 > self.y0:
            raise ValueError(f"{self.name}: y1 must exceed y0")
        if abs(polygon_area(list(self.profile))) <= 0:
            raise ValueError(f"{self.name}: degenerate profile")

    @property
    def ccw_profile(self) -> list[Point2]:
        pts = list(self.profile)
        return pts if polygon_area(pts) > 0 else pts[::-1]

    def volume(self) -> float:
        return abs(polygon_area(list(self.profile))) * (self.y1 - self.y0)

    def bounds(self) -> tuple[Point3, Point3]:
        xs = [p[0] for p in self.profile]
        zs = [p[1] for p in self.profile]
        return (min(xs), self.y0, min(zs)), (max(xs), self.y1, max(zs))

    def faces(self, arc_segments: int = 8) -> list[Face]:  # noqa: ARG002 - no arcs in a polygon prism
        pts = self.ccw_profile  # CCW in (x, z)
        # (x, z) CCW seen from -y looking towards +y is CW seen from +y ... orientation is fixed explicitly below.
        front = [(x, self.y0, z) for x, z in pts]          # face at y0, outward normal -y
        back = [(x, self.y1, z) for x, z in pts[::-1]]     # face at y1, outward normal +y
        faces = [Face(outer=_orient(front, (0.0, -1.0, 0.0))), Face(outer=_orient(back, (0.0, 1.0, 0.0)))]
        n = len(pts)
        for i in range(n):
            (xa, za), (xb, zb) = pts[i], pts[(i + 1) % n]
            quad = [(xa, self.y0, za), (xb, self.y0, zb), (xb, self.y1, zb), (xa, self.y1, za)]
            # outward normal of a CCW (x, z) polygon edge: (dz, -dx) in (x, z)
            normal = (zb - za, 0.0, -(xb - xa))
            faces.append(Face(outer=_orient(quad, normal)))
        return faces


@dataclass(frozen=True)
class RoundedRectRing:
    """Ring between the offset curves at distances ``d_in`` and ``d_out`` of the rectangle ``[-hx, hx] x [-hy, hy]``.

    The rectangle is the *base* (e.g. the centre-leg cross-section); the ring occupies the points whose distance
    to the base rectangle lies in ``[d_in, d_out]`` when ``rounded=True`` (circular corners of radius ``d``), or the
    difference of the two axis-aligned rectangles grown by ``d_in`` and ``d_out`` when ``rounded=False``. It is
    extruded from ``z0`` to ``z1``. ``outer_override`` (hx_out, hy_out) replaces the outer offset rectangle by an
    explicit rectangle with square corners (bobbin flanges).
    """

    name: str
    role: str
    material: str
    hx: float
    hy: float
    d_in: float
    d_out: float
    z0: float
    z1: float
    rounded: bool = True
    outer_override: tuple[float, float] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.hx <= 0 or self.hy <= 0:
            raise ValueError(f"{self.name}: base half-sizes must be positive")
        if self.d_in < 0 or not self.d_out > self.d_in:
            raise ValueError(f"{self.name}: need 0 <= d_in < d_out (got {self.d_in}, {self.d_out})")
        if not self.z1 > self.z0:
            raise ValueError(f"{self.name}: z1 must exceed z0")
        if self.outer_override is not None:
            ox, oy = self.outer_override
            if ox <= self.hx + self.d_in or oy <= self.hy + self.d_in:
                raise ValueError(f"{self.name}: outer rectangle must enclose the inner offset curve")

    @staticmethod
    def offset_area(hx: float, hy: float, d: float, rounded: bool) -> float:
        """Area enclosed by the offset curve at distance ``d`` of the base rectangle."""
        if rounded:
            return 4 * hx * hy + 2 * d * (2 * hx + 2 * hy) + math.pi * d * d
        return (2 * hx + 2 * d) * (2 * hy + 2 * d)

    @staticmethod
    def offset_perimeter(hx: float, hy: float, d: float, rounded: bool) -> float:
        if rounded:
            return 4 * hx + 4 * hy + 2 * math.pi * d
        return 4 * (hx + d) + 4 * (hy + d)

    def mean_turn_length(self) -> float:
        """Perimeter of the offset curve at the mid-distance (the mean turn length of a winding layer)."""
        return self.offset_perimeter(self.hx, self.hy, 0.5 * (self.d_in + self.d_out), self.rounded)

    def area(self) -> float:
        inner = self.offset_area(self.hx, self.hy, self.d_in, self.rounded)
        if self.outer_override is not None:
            outer = 4 * self.outer_override[0] * self.outer_override[1]
        else:
            outer = self.offset_area(self.hx, self.hy, self.d_out, self.rounded)
        return outer - inner

    def volume(self) -> float:
        return self.area() * (self.z1 - self.z0)

    def bounds(self) -> tuple[Point3, Point3]:
        if self.outer_override is not None:
            ox, oy = self.outer_override
        else:
            ox, oy = self.hx + self.d_out, self.hy + self.d_out
        return (-ox, -oy, self.z0), (ox, oy, self.z1)

    def loop(self, d: float, arc_segments: int, rounded: bool | None = None) -> list[Point2]:
        """Counter-clockwise polygon of the offset curve at distance ``d`` (x, y)."""
        rounded = self.rounded if rounded is None else rounded
        hx, hy = self.hx, self.hy
        if not rounded or d == 0.0:
            ex, ey = hx + d, hy + d
            return [(ex, -ey), (ex, ey), (-ex, ey), (-ex, -ey)]
        pts: list[Point2] = []
        corners = [((hx, hy), 0.0), ((-hx, hy), 0.5 * math.pi), ((-hx, -hy), math.pi), ((hx, -hy), 1.5 * math.pi)]
        for (cx, cy), start in corners:
            for k in range(arc_segments + 1):
                a = start + 0.5 * math.pi * k / arc_segments
                pts.append((cx + d * math.cos(a), cy + d * math.sin(a)))
        return pts

    def outer_loop(self, arc_segments: int) -> list[Point2]:
        """Outer boundary (CCW). For an explicit outer rectangle the radial projections of the inner-loop points are inserted on
        its edges so that cap triangulation and side faces share vertices (watertight surface meshes)."""
        if self.outer_override is None:
            return self.loop(self.d_out, arc_segments)
        ox, oy = self.outer_override
        pts: list[tuple[float, Point2]] = [(math.atan2(y, x) % (2 * math.pi), (x, y)) for x, y in ((ox, -oy), (ox, oy), (-ox, oy), (-ox, -oy))]
        for x, y in self.loop(self.d_in, arc_segments):
            t = min(ox / abs(x) if abs(x) > 1e-15 else math.inf, oy / abs(y) if abs(y) > 1e-15 else math.inf)
            q = (x * t, y * t)
            if all(abs(q[0] - p[0]) > 1e-12 or abs(q[1] - p[1]) > 1e-12 for _, p in pts):
                pts.append((math.atan2(q[1], q[0]) % (2 * math.pi), q))
        pts.sort(key=lambda e: e[0])
        return [p for _, p in pts]

    def faces(self, arc_segments: int = 8) -> list[Face]:
        outer = self.outer_loop(arc_segments)
        inner = self.loop(self.d_in, arc_segments)
        faces: list[Face] = []
        top_outer = [(x, y, self.z1) for x, y in outer]
        top_hole = [(x, y, self.z1) for x, y in inner[::-1]]
        bot_outer = [(x, y, self.z0) for x, y in outer[::-1]]
        bot_hole = [(x, y, self.z0) for x, y in inner]
        faces.append(Face(outer=top_outer, holes=[top_hole]))
        faces.append(Face(outer=bot_outer, holes=[bot_hole]))
        for loop, outward in ((outer, 1.0), (inner, -1.0)):
            n = len(loop)
            for i in range(n):
                (xa, ya), (xb, yb) = loop[i], loop[(i + 1) % n]
                if (xa, ya) == (xb, yb):
                    continue
                quad = [(xa, ya, self.z0), (xb, yb, self.z0), (xb, yb, self.z1), (xa, ya, self.z1)]
                normal = (outward * (yb - ya), -outward * (xb - xa), 0.0)
                faces.append(Face(outer=_orient(quad, normal)))
        return faces


    def cap_triangles(self, arc_segments: int) -> list[tuple[Point3, Point3, Point3]]:
        """Top and bottom annulus by an angular zipper between the (star-shaped) inner and outer loops."""
        inner = self.loop(self.d_in, arc_segments)
        outer = self.outer_loop(arc_segments)

        def start_at_min_angle(loop: list[Point2]) -> list[tuple[float, Point2]]:
            angs = [math.atan2(y, x) % (2 * math.pi) for x, y in loop]
            k = int(np.argmin(angs))
            ordered = loop[k:] + loop[:k]
            out, prev = [], None
            for p in ordered:
                a = math.atan2(p[1], p[0]) % (2 * math.pi)
                if prev is not None and a < prev - 1e-12:
                    a += 2 * math.pi
                out.append((a, p))
                prev = a
            out.append((out[0][0] + 2 * math.pi, out[0][1]))
            return out

        I, O = start_at_min_angle(inner), start_at_min_angle(outer)
        flat: list[tuple[Point2, Point2, Point2]] = []
        i = j = 0
        while i < len(I) - 1 or j < len(O) - 1:
            if j >= len(O) - 1 or (i < len(I) - 1 and I[i + 1][0] <= O[j + 1][0]):
                flat.append((O[j][1], I[i + 1][1], I[i][1]))
                i += 1
            else:
                flat.append((O[j][1], O[j + 1][1], I[i][1]))
                j += 1
        tris: list[tuple[Point3, Point3, Point3]] = []
        for z, up in ((self.z1, True), (self.z0, False)):
            for a, b, c in flat:
                pa, pb, pc = (*a, z), (*b, z), (*c, z)
                if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) < 1e-30:
                    continue
                tris.append((pa, pb, pc) if up else (pa, pc, pb))
        return tris


Solid = ExtrudedProfile | RoundedRectRing


def _newell(points: list[Point3]) -> np.ndarray:
    n = np.zeros(3)
    for i in range(len(points)):
        x0, y0, z0 = points[i]
        x1, y1, z1 = points[(i + 1) % len(points)]
        n += ((y0 - y1) * (z0 + z1), (z0 - z1) * (x0 + x1), (x0 - x1) * (y0 + y1))
    return n


def _orient(points: list[Point3], outward: tuple[float, float, float]) -> list[Point3]:
    """Order ``points`` so that their right-hand normal points along ``outward``."""
    n = _newell(points)
    return points if float(np.dot(n, outward)) >= 0 else points[::-1]


def face_normal(face: Face) -> np.ndarray:
    n = _newell(face.outer)
    norm = float(np.linalg.norm(n))
    return n / norm if norm > 0 else n


def triangulate_face(face: Face) -> list[tuple[Point3, Point3, Point3]]:
    """Triangles of a planar face (ear clipping on the 2-D projection; holes bridged)."""
    normal = face_normal(face)
    axis = int(np.argmax(np.abs(normal)))
    keep = [i for i in range(3) if i != axis]

    def proj(p: Point3) -> Point2:
        return (p[keep[0]], p[keep[1]])

    loop = list(face.outer)
    for hole in face.holes:
        loop = _bridge(loop, list(hole), proj)
    sign = 1.0 if normal[axis] >= 0 else -1.0
    if keep == [0, 2]:  # projecting onto (x, z) flips handedness relative to (y-axis)
        sign = -sign
    return _ear_clip(loop, proj, sign)


def _bridge(outer: list[Point3], hole: list[Point3], proj) -> list[Point3]:
    """Join a hole into the outer loop through the mutually closest vertex pair (simple, adequate for convex-ish rings)."""
    best = None
    for i, p in enumerate(outer):
        for j, q in enumerate(hole):
            a, b = proj(p), proj(q)
            d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            if best is None or d < best[0]:
                best = (d, i, j)
    assert best is not None
    _, i, j = best
    return outer[: i + 1] + hole[j:] + hole[: j + 1] + outer[i:]


def _ear_clip(loop: list[Point3], proj, sign: float) -> list[tuple[Point3, Point3, Point3]]:
    pts = list(loop)
    tris: list[tuple[Point3, Point3, Point3]] = []

    def cross(o: Point2, a: Point2, b: Point2) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p: Point2, a: Point2, b: Point2, c: Point2) -> bool:
        d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
        return (d1 > 1e-18 and d2 > 1e-18 and d3 > 1e-18) or (d1 < -1e-18 and d2 < -1e-18 and d3 < -1e-18)

    guard = 0
    while len(pts) > 3 and guard < 100000:
        guard += 1
        n = len(pts)
        clipped = False
        for i in range(n):
            a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
            pa, pb, pc = proj(a), proj(b), proj(c)
            area = cross(pa, pb, pc) * sign
            if area <= 1e-24:
                continue
            if any(inside(proj(q), pa, pb, pc) for q in pts if q not in (a, b, c)):
                continue
            tris.append((a, b, c))
            del pts[i]
            clipped = True
            break
        if not clipped:  # degenerate remainder (collinear bridge edges): fan it
            for k in range(1, len(pts) - 1):
                if abs(cross(proj(pts[0]), proj(pts[k]), proj(pts[k + 1]))) > 1e-24:
                    tris.append((pts[0], pts[k], pts[k + 1]))
            pts = []
            break
    if len(pts) == 3:
        tris.append((pts[0], pts[1], pts[2]))
    return tris


def solid_triangles(solid: Solid, arc_segments: int = 8) -> list[tuple[Point3, Point3, Point3]]:
    tris: list[tuple[Point3, Point3, Point3]] = []
    faces = solid.faces(arc_segments)
    if isinstance(solid, RoundedRectRing):
        tris.extend(solid.cap_triangles(arc_segments))
        faces = [f for f in faces if not f.holes]  # caps replaced by the matched-annulus triangulation
    for face in faces:
        tris.extend(triangulate_face(face))
    return tris


def mesh_volume(tris: list[tuple[Point3, Point3, Point3]]) -> float:
    """Signed volume of a closed triangle surface (divergence theorem); positive for outward orientation."""
    v = 0.0
    for a, b, c in tris:
        v += float(np.dot(a, np.cross(b, c))) / 6.0
    return v


__all__ = ["ExtrudedProfile", "Face", "RoundedRectRing", "Solid", "face_normal", "mesh_volume", "polygon_area",
           "solid_triangles", "triangulate_face"]
