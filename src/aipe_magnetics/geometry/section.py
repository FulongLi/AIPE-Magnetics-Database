"""2-D cross-section model shared by the electromagnetic and thermal solvers.

A :class:`Section2D` is an ordered list of axis-aligned rectangular :class:`Region` objects in the (x, z)
plane plus the computational domain. Later regions take precedence over earlier ones where they overlap (a
"paint" order). The section is
the single geometric input of both mesh generators (the built-in structured mesher and Gmsh), so CAD and
simulation share one geometry definition.

The E-core section is the plane ``y = 0`` through the centre leg. Only ``x >= 0`` is represented
(``symmetry='half_x'``): geometry and currents are mirror-antisymmetric about ``x = 0`` (the winding current in
the left window flows opposite to the right window), so the planar vector potential satisfies ``A = 0`` there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RegionRole = Literal["core", "gap", "conductor", "stranded", "winding_pack", "bobbin", "insulation", "window_air", "air"]


@dataclass(frozen=True)
class Region:
    """Rectangle ``[x0, x1] x [z0, z1]`` with a role and a material key."""

    name: str
    role: RegionRole
    material: str
    x0: float
    x1: float
    z0: float
    z1: float
    winding: str | None = None
    turn: int | None = None          # index of the turn inside its winding (0-based)
    layer: str | None = None         # id of the placed layer
    mesh_size: float | None = None   # target element size (m); None = role default
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (self.x1 > self.x0 and self.z1 > self.z0):
            raise ValueError(f"region {self.name}: empty rectangle ({self.x0}, {self.x1}) x ({self.z0}, {self.z1})")

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.z1 - self.z0)

    @property
    def centre(self) -> tuple[float, float]:
        return 0.5 * (self.x0 + self.x1), 0.5 * (self.z0 + self.z1)

    def contains(self, x: float, z: float) -> bool:
        return self.x0 <= x <= self.x1 and self.z0 <= z <= self.z1


@dataclass
class Section2D:
    """Ordered regions inside a rectangular domain (the first region normally *is* the domain)."""

    regions: list[Region]
    symmetry: Literal["none", "half_x"] = "half_x"
    coordinates: Literal["planar", "axisymmetric"] = "planar"
    depth: float = 1.0                       # out-of-plane length used for core / gap / outer-air energy (m)
    winding_length: dict[str, float] = field(default_factory=dict)   # mean turn length per winding (m)
    description: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs0 = min(r.x0 for r in self.regions)
        xs1 = max(r.x1 for r in self.regions)
        zs0 = min(r.z0 for r in self.regions)
        zs1 = max(r.z1 for r in self.regions)
        return xs0, xs1, zs0, zs1

    def region(self, name: str) -> Region:
        for r in self.regions:
            if r.name == name:
                return r
        raise KeyError(name)

    def by_role(self, *roles: str) -> list[Region]:
        return [r for r in self.regions if r.role in roles]

    def conductors(self, winding: str | None = None) -> list[Region]:
        return [r for r in self.regions if r.role in ("conductor", "stranded") and (winding is None or r.winding == winding)]

    def windings(self) -> list[str]:
        seen: list[str] = []
        for r in self.regions:
            if r.winding and r.winding not in seen:
                seen.append(r.winding)
        return seen

    def classify(self, x: float, z: float) -> int:
        """Index of the top-most region containing ``(x, z)`` (-1 when outside every region)."""
        for i in range(len(self.regions) - 1, -1, -1):
            if self.regions[i].contains(x, z):
                return i
        return -1

    def to_json(self) -> dict[str, Any]:
        return {"symmetry": self.symmetry, "coordinates": self.coordinates, "depth_m": self.depth,
                "winding_length_m": self.winding_length, "description": self.description,
                "regions": [{"name": r.name, "role": r.role, "material": r.material, "x0": r.x0, "x1": r.x1, "z0": r.z0, "z1": r.z1,
                             **({"winding": r.winding} if r.winding else {}), **({"turn": r.turn} if r.turn is not None else {}),
                             **({"layer": r.layer} if r.layer else {})} for r in self.regions]}


__all__ = ["Region", "RegionRole", "Section2D"]
