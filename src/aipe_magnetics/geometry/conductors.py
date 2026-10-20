"""Conductor specifications (bare copper and outer envelope) used by the winding layout.

Lengths are metres. ``axial_pitch`` is the conductor envelope along the window height (Z), ``radial_build``
the envelope across the window width (X). Round and litz conductors use square packing (pitch = outer
diameter in both directions); orthocyclic nesting is not modelled.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ConductorKind = Literal["round", "litz", "foil", "rectangular"]
DEFAULT_ENAMEL_BUILD = 1.1          # outer / bare diameter of grade-2 enamelled wire (assumption, as in v0.2)
DEFAULT_LITZ_PACKING = 1.25         # bundle diameter = packing * sqrt(n) * strand diameter (assumption, as in v0.2)


class ConductorSpec(BaseModel):
    """One conductor (one turn's cross-section; ``parallel`` conductors side by side form one turn)."""

    model_config = ConfigDict(extra="forbid")
    kind: ConductorKind
    bare_diameter: float | None = Field(default=None, gt=0, description="round: copper diameter (m)")
    outer_diameter: float | None = Field(default=None, gt=0, description="round/litz: insulated envelope (m)")
    strand_diameter: float | None = Field(default=None, gt=0, description="litz: strand copper diameter (m)")
    strand_count: int | None = Field(default=None, ge=1, description="litz: strands per bundle")
    thickness: float | None = Field(default=None, gt=0, description="foil / rectangular: copper thickness across the window (m)")
    width: float | None = Field(default=None, gt=0, description="foil / rectangular: copper width along the window height (m); "
                                                               "foil: None = fill the available winding height")
    insulation: float = Field(default=0.0, ge=0, description="foil / rectangular: insulation thickness on each side (m)")
    parallel: int = Field(default=1, ge=1, description="conductors in parallel per turn (laid side by side along the height)")

    @model_validator(mode="after")
    def _complete(self) -> "ConductorSpec":
        if self.kind == "round" and self.bare_diameter is None:
            raise ValueError("round conductor needs bare_diameter")
        if self.kind == "litz" and (self.strand_diameter is None or self.strand_count is None):
            raise ValueError("litz conductor needs strand_diameter and strand_count")
        if self.kind in ("foil", "rectangular") and self.thickness is None:
            raise ValueError(f"{self.kind} conductor needs thickness")
        if self.kind == "rectangular" and self.width is None:
            raise ValueError("rectangular conductor needs width")
        if self.kind == "foil" and self.parallel != 1:
            raise ValueError("foil turns are single conductors (use a thicker foil instead of parallel foils)")
        return self

    # -- copper ---------------------------------------------------------------------------------------
    def copper_area(self, foil_width: float | None = None) -> float:
        """Copper cross-section of one turn (all parallel conductors)."""
        if self.kind == "round":
            a = math.pi * self.bare_diameter ** 2 / 4.0
        elif self.kind == "litz":
            a = self.strand_count * math.pi * self.strand_diameter ** 2 / 4.0
        elif self.kind == "foil":
            w = self.width if self.width is not None else foil_width
            if w is None:
                raise ValueError("foil width unknown: give width or the available winding height")
            a = self.thickness * w
        else:
            a = self.thickness * self.width
        return a * self.parallel

    def envelope_diameter(self) -> float:
        if self.kind == "round":
            return self.outer_diameter or self.bare_diameter * DEFAULT_ENAMEL_BUILD
        if self.kind == "litz":
            return self.outer_diameter or DEFAULT_LITZ_PACKING * math.sqrt(self.strand_count) * self.strand_diameter
        raise ValueError("envelope_diameter applies to round and litz conductors")

    def radial_build(self) -> float:
        if self.kind in ("round", "litz"):
            return self.envelope_diameter()
        return self.thickness + 2 * self.insulation

    def axial_pitch(self, foil_width: float | None = None) -> float:
        """Envelope of one turn along the window height (all parallel conductors)."""
        if self.kind in ("round", "litz"):
            return self.envelope_diameter() * self.parallel
        if self.kind == "foil":
            w = self.width if self.width is not None else foil_width
            return (w or 0.0) + 2 * self.insulation
        return (self.width + 2 * self.insulation) * self.parallel

    def equivalent_square_side(self) -> float:
        """Side of the copper square of equal area (Dowell's equivalent conductor for round wire / one strand bundle)."""
        if self.kind == "round":
            return math.sqrt(math.pi) / 2.0 * self.bare_diameter
        if self.kind == "litz":
            return math.sqrt(self.copper_area() / self.parallel)
        raise ValueError("equivalent_square_side applies to round and litz conductors")

    def describe(self) -> str:
        if self.kind == "round":
            return f"round {self.bare_diameter * 1e3:.3g} mm" + (f" x{self.parallel}" if self.parallel > 1 else "")
        if self.kind == "litz":
            return f"litz {self.strand_count} x {self.strand_diameter * 1e3:.3g} mm" + (f" x{self.parallel}" if self.parallel > 1 else "")
        if self.kind == "foil":
            return f"foil {self.thickness * 1e3:.3g} mm" + (f" x {self.width * 1e3:.3g} mm" if self.width else "")
        return f"rectangular {self.thickness * 1e3:.3g} x {self.width * 1e3:.3g} mm"


def litz_for_area(area: float, strand_diameter: float) -> ConductorSpec:
    """Litz bundle with the smallest strand count whose copper area is at least ``area``."""
    n = max(1, math.ceil(area / (math.pi * strand_diameter ** 2 / 4.0)))
    return ConductorSpec(kind="litz", strand_diameter=strand_diameter, strand_count=n)


def round_for_area(area: float) -> ConductorSpec:
    return ConductorSpec(kind="round", bare_diameter=2.0 * math.sqrt(area / math.pi))


__all__ = ["ConductorKind", "ConductorSpec", "DEFAULT_ENAMEL_BUILD", "DEFAULT_LITZ_PACKING", "litz_for_area", "round_for_area"]
