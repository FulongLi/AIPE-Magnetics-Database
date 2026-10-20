"""Parametric gapped E-core magnetic component: core halves, gaps, bobbin, layered windings, insulation.

:class:`ECoreMagnetic` is the editable parameter set (all lengths in metres, validated); :meth:`ECoreMagnetic.build`
returns a :class:`MagneticGeometry` with

* the 3-D solids of the common representation (:mod:`aipe_magnetics.geometry.primitives`) for CAD export,
* the placed winding layers and turns (positions, mean turn length per layer, fit / clearance checks),
* the 2-D half cross-section (:class:`~aipe_magnetics.geometry.section.Section2D`) for the solvers,
* derived quantities (core volume and mass, copper volume and mass, winding resistances' geometric inputs,
  exposed surface areas, bounding volume) with the formula used for each.

E-core outline symbols (IEC 62317 / TDK datasheet convention, per *half* core):
``A`` length, ``B`` height of one half (``height_b_half``), ``C`` depth, ``D`` window height of one half
(``window_height_half``), ``E`` distance between the inner faces of the outer legs (``window_width_total``),
``F`` centre-leg width. Outer-leg width ``(A - E)/2``, back thickness ``B - D``, window width per side
``(E - F)/2``, total window height ``2 D + spacer``.

Gaps: ``spacer`` separates the two halves at every leg (shims, or the residual mating gap of an "ungapped"
set); ``centre_grind`` is the additional total centre-leg gap ground into the halves (split equally). The
centre-leg gap is ``spacer + centre_grind`` and each outer-leg gap is ``spacer``.

Winding turns follow the offset curve of the centre leg at their radial distance ``d`` from the leg surface:
mean turn length ``2F + 2C + 2 pi d`` (a turn hugging a rectangular leg with circular corners). The bobbin
and winding solids use the same offset curves.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aipe_magnetics.geometry.conductors import ConductorSpec
from aipe_magnetics.geometry.primitives import ExtrudedProfile, RoundedRectRing, Solid
from aipe_magnetics.geometry.section import Region, Section2D

GEOMETRY_MODEL_ID = "aipe.magnetics.geometry.ecore"
GEOMETRY_MODEL_VERSION = "0.3.0"


class ECoreDimensions(BaseModel):
    """Outline of one E half (metres)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    length_a: float = Field(gt=0)
    height_b_half: float = Field(gt=0)
    depth_c: float = Field(gt=0)
    window_height_half: float = Field(gt=0)
    window_width_total: float = Field(gt=0)
    centre_leg_width: float = Field(gt=0)
    source: str | None = Field(default=None, description="record id the dimensions were taken from")

    @model_validator(mode="after")
    def _consistent(self) -> "ECoreDimensions":
        if not self.window_width_total > self.centre_leg_width:
            raise ValueError("window_width_total (E) must exceed centre_leg_width (F)")
        if not self.length_a > self.window_width_total:
            raise ValueError("length_a (A) must exceed window_width_total (E)")
        if not self.height_b_half > self.window_height_half:
            raise ValueError("height_b_half (B) must exceed window_height_half (D)")
        return self

    @property
    def outer_leg_width(self) -> float:
        return 0.5 * (self.length_a - self.window_width_total)

    @property
    def back_thickness(self) -> float:
        return self.height_b_half - self.window_height_half

    @property
    def window_width(self) -> float:
        """Window width of one side ``(E - F) / 2``."""
        return 0.5 * (self.window_width_total - self.centre_leg_width)

    @classmethod
    def from_core_data(cls, core: Any) -> "ECoreDimensions":
        """From :class:`aipe_magnetics.database.CoreData` (raises when the outline is not recorded)."""
        if not core.has_e_geometry():
            missing = [k for k in ("length_a", "height_b_half", "depth_c", "window_height_half", "window_width_total", "centre_leg_width")
                       if k not in core.dimensions]
            raise ValueError(f"core {core.id} lacks E-core outline dimensions {missing}; the parametric E model cannot be built")
        d = core.dimensions
        return cls(length_a=d["length_a"], height_b_half=d["height_b_half"], depth_c=d["depth_c"],
                   window_height_half=d["window_height_half"], window_width_total=d["window_width_total"],
                   centre_leg_width=d["centre_leg_width"], source=core.id)


class GapSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    spacer: float = Field(default=0.0, ge=0, description="separation of the halves at every leg (m)")
    centre_grind: float = Field(default=0.0, ge=0, description="additional total centre-leg gap (m)")
    note: str | None = None

    @property
    def centre_gap(self) -> float:
        return self.spacer + self.centre_grind

    @property
    def outer_gap(self) -> float:
        return self.spacer


class BobbinSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    present: bool = True
    leg_clearance: float = Field(default=0.3e-3, ge=0, description="radial air between the centre leg and the tube (m)")
    tube_thickness: float = Field(default=1.3e-3, ge=0, description="coil-former tube wall (m)")
    flange_thickness: float = Field(default=1.7e-3, ge=0, description="flange wall (m)")
    outer_clearance: float = Field(default=0.5e-3, ge=0, description="minimum clearance between winding and outer legs (m)")
    material: str = "bobbin_thermoplastic"
    note: str | None = None


class InsulationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    layer_insulation: float = Field(default=0.05e-3, ge=0, description="tape between layers of one winding (m)")
    section_insulation: float = Field(default=0.2e-3, ge=0, description="tape between winding sections (m)")
    outer_wrap: float = Field(default=0.1e-3, ge=0, description="tape over the outermost layer (m)")
    creepage_margin: float = Field(default=0.0, ge=0, description="margin at each end of every layer (m)")
    material: str = "polyimide_film"


class WindingSection(BaseModel):
    """A contiguous block of layers of one winding (several sections of one winding are connected in series)."""

    model_config = ConfigDict(extra="forbid")
    winding: str = Field(min_length=1)
    turns: int = Field(ge=1)
    conductor: ConductorSpec
    layers: int | None = Field(default=None, ge=1, description="None = as few layers as fit")


class WindingLayout(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sections: list[WindingSection] = Field(min_length=1)
    insulation: InsulationSpec = Field(default_factory=InsulationSpec)
    arrangement: str | None = Field(default=None, description="label, e.g. 'P-S' or 'P-S-P'")


@dataclass
class PlacedLayer:
    id: str
    winding: str
    section: int
    index: int                    # layer index inside its section
    d_in: float                   # distance of the layer's inner face from the centre-leg surface (m)
    d_out: float
    z0: float
    z1: float
    turns: int
    turn_z: list[tuple[float, float]]   # axial extent of every turn of the layer
    conductor: ConductorSpec
    mean_turn_length: float
    first_turn: int               # index of the first turn of this layer inside its winding


@dataclass
class LayoutCheck:
    code: str
    passed: bool
    message: str
    value: float | None = None
    limit: float | None = None


@dataclass
class MagneticGeometry:
    """Result of :meth:`ECoreMagnetic.build`."""

    model: "ECoreMagnetic"
    solids: list[Solid]
    layers: list[PlacedLayer]
    section: Section2D
    checks: list[LayoutCheck]
    derived: dict[str, Any] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return all(c.passed for c in self.checks)

    def winding_names(self) -> list[str]:
        names: list[str] = []
        for layer in self.layers:
            if layer.winding not in names:
                names.append(layer.winding)
        return names

    def winding_turns(self, name: str) -> int:
        return sum(layer.turns for layer in self.layers if layer.winding == name)

    def winding_length(self, name: str) -> float:
        """Total conductor length of a winding (sum over layers of turns x layer mean turn length)."""
        return sum(layer.turns * layer.mean_turn_length for layer in self.layers if layer.winding == name)

    def summary(self) -> dict[str, Any]:
        return {"model": GEOMETRY_MODEL_ID, "version": GEOMETRY_MODEL_VERSION, "parameter_hash": self.model.parameter_hash(),
                "feasible": self.feasible, "checks": [c.__dict__ for c in self.checks], "derived": self.derived,
                "solids": [{"name": s.name, "role": s.role, "material": s.material, "volume_mm3": s.volume() * 1e9} for s in self.solids],
                "layers": [{"id": l.id, "winding": l.winding, "turns": l.turns, "d_in_mm": l.d_in * 1e3, "d_out_mm": l.d_out * 1e3,
                            "z0_mm": l.z0 * 1e3, "z1_mm": l.z1 * 1e3, "mean_turn_length_mm": l.mean_turn_length * 1e3,
                            "conductor": l.conductor.describe()} for l in self.layers]}


class ECoreMagnetic(BaseModel):
    """Editable parameter set of a gapped E-core inductor or transformer."""

    model_config = ConfigDict(extra="forbid")
    name: str = "magnetic"
    core: ECoreDimensions
    core_material: str = Field(description="catalogue material id of the core")
    gap: GapSpec = Field(default_factory=GapSpec)
    bobbin: BobbinSpec = Field(default_factory=BobbinSpec)
    layout: WindingLayout | None = None
    stack: int = Field(default=1, ge=1, le=4, description="core sets stacked along the depth (C multiplied)")

    # -- parameters ------------------------------------------------------------------------------------
    def parameter_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def parameters(self) -> dict[str, dict[str, Any]]:
        """Flat editable parameter table ``path -> {value, unit}`` (lengths in mm for readability)."""
        out: dict[str, dict[str, Any]] = {}
        for k, v in self.core.model_dump(exclude={"source"}).items():
            out[f"core.{k}"] = {"value": v * 1e3, "unit": "mm"}
        out["core.stack"] = {"value": self.stack, "unit": "1"}
        for k, v in self.gap.model_dump(exclude={"note"}).items():
            out[f"gap.{k}"] = {"value": v * 1e3, "unit": "mm"}
        for k, v in self.bobbin.model_dump(exclude={"note", "material", "present"}).items():
            out[f"bobbin.{k}"] = {"value": v * 1e3, "unit": "mm"}
        if self.layout:
            for k, v in self.layout.insulation.model_dump(exclude={"material"}).items():
                out[f"insulation.{k}"] = {"value": v * 1e3, "unit": "mm"}
            for i, s in enumerate(self.layout.sections):
                out[f"sections.{i}.turns"] = {"value": s.turns, "unit": "1"}
                if s.layers is not None:
                    out[f"sections.{i}.layers"] = {"value": s.layers, "unit": "1"}
                for k in ("bare_diameter", "strand_diameter", "thickness", "width", "outer_diameter"):
                    v = getattr(s.conductor, k)
                    if v is not None:
                        out[f"sections.{i}.conductor.{k}"] = {"value": v * 1e3, "unit": "mm"}
                if s.conductor.strand_count is not None:
                    out[f"sections.{i}.conductor.strand_count"] = {"value": s.conductor.strand_count, "unit": "1"}
        return out

    def with_parameters(self, changes: dict[str, Any]) -> "ECoreMagnetic":
        """A new model with parameters changed by path (``{'gap.centre_grind': 0.8, 'sections.0.turns': 22}``).

        Length values are interpreted in **mm** (as listed by :meth:`parameters`) unless given as
        ``{'value': x, 'unit': 'm'|'mm'|'um'}``; counts are integers.
        """
        data = self.model_dump(mode="python")
        for path, raw in changes.items():
            if isinstance(raw, dict):
                from aipe_magnetics.common.quantity import Quantity
                q = Quantity.parse(raw, name=path)
                value: Any = q.magnitude("m") if q.unit != "1" else q.value
                is_length = q.unit != "1"
            else:
                value, is_length = raw, None
            parts = path.split(".")
            target: Any = data
            keys: list[Any] = []
            for p in parts:
                keys.append(int(p) if p.isdigit() else p)
            if keys[0] == "sections":
                if not data.get("layout"):
                    raise ValueError("model has no winding layout")
                target = data["layout"]["sections"]
                keys = keys[1:]
            elif keys[0] == "insulation":
                if not data.get("layout"):
                    raise ValueError("model has no winding layout")
                target = data["layout"]["insulation"]
                keys = keys[1:]
            elif keys[0] == "core" and keys[1] == "stack":
                target = data
                keys = ["stack"]
            for k in keys[:-1]:
                target = target[k]
            leaf = keys[-1]
            if leaf not in (target if isinstance(target, dict) else {}) and not isinstance(target, list):
                raise ValueError(f"unknown parameter path {path!r}")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if leaf in ("turns", "layers", "strand_count", "stack", "parallel"):
                    if float(value) != int(value):
                        raise ValueError(f"{path} must be an integer")
                    value = int(value)
                elif is_length is None:
                    value = float(value) * 1e-3  # mm
            target[leaf] = value
        return ECoreMagnetic.model_validate(data)

    # -- geometry ---------------------------------------------------------------------------------------
    @property
    def depth(self) -> float:
        return self.core.depth_c * self.stack

    def _z_frame(self) -> dict[str, float]:
        c, g = self.core, self.gap
        s = g.spacer
        return {"z_back_bot": -(c.height_b_half + s / 2), "z_win_bot": -(c.window_height_half + s / 2),
                "z_win_top": c.window_height_half + s / 2, "z_back_top": c.height_b_half + s / 2,
                "z_cgap": g.centre_gap / 2, "z_ogap": s / 2}

    def winding_space(self) -> dict[str, float]:
        """Radial and axial space available to the windings (distances from the centre-leg surface)."""
        c, b = self.core, self.bobbin
        zf = self._z_frame()
        if b.present:
            d0 = b.leg_clearance + b.tube_thickness
            z0, z1 = zf["z_win_bot"] + b.flange_thickness, zf["z_win_top"] - b.flange_thickness
        else:
            d0 = b.leg_clearance
            z0, z1 = zf["z_win_bot"], zf["z_win_top"]
        return {"d_start": d0, "d_limit": c.window_width - b.outer_clearance, "z0": z0, "z1": z1, "height": z1 - z0}

    def place_layers(self) -> tuple[list[PlacedLayer], list[LayoutCheck]]:
        checks: list[LayoutCheck] = []
        layers: list[PlacedLayer] = []
        if self.layout is None:
            return layers, checks
        space = self.winding_space()
        ins = self.layout.insulation
        margin = ins.creepage_margin
        z0, z1 = space["z0"] + margin, space["z1"] - margin
        height = z1 - z0
        if height <= 0:
            checks.append(LayoutCheck("winding_height", False, "creepage margins leave no winding height", height, 0.0))
            return layers, checks
        d = space["d_start"]
        turn_counter: dict[str, int] = {}
        hx, hy = 0.5 * self.core.centre_leg_width, 0.5 * self.depth
        for si, section in enumerate(self.layout.sections):
            cond = section.conductor
            if si > 0:
                d += ins.section_insulation
            if cond.kind == "foil":
                width = cond.width if cond.width is not None else height - 2 * cond.insulation
                if width + 2 * cond.insulation > height * (1 + 1e-9):
                    checks.append(LayoutCheck(f"{section.winding}.foil_width", False,
                                              f"{section.winding}: foil width {width * 1e3:.2f} mm exceeds the winding height {height * 1e3:.2f} mm",
                                              width, height))
                cond = cond.model_copy(update={"width": width})
                per_layer = 1
                n_layers = section.turns if section.layers is None else section.layers
                if n_layers != section.turns:
                    checks.append(LayoutCheck(f"{section.winding}.foil_layers", False, "a foil section has one turn per layer"))
                    n_layers = section.turns
            else:
                pitch = cond.axial_pitch()
                fit = int(math.floor(height / pitch + 1e-9))
                if fit < 1:
                    checks.append(LayoutCheck(f"{section.winding}.turn_fit", False,
                                              f"{section.winding}: one turn ({pitch * 1e3:.2f} mm) does not fit the winding height {height * 1e3:.2f} mm",
                                              pitch, height))
                    fit = 1
                if section.layers is None:
                    n_layers = int(math.ceil(section.turns / fit))
                    per_layer = int(math.ceil(section.turns / n_layers))
                else:
                    n_layers = section.layers
                    per_layer = int(math.ceil(section.turns / n_layers))
                    if per_layer > fit:
                        checks.append(LayoutCheck(f"{section.winding}.turns_per_layer", False,
                                                  f"{section.winding}: {per_layer} turns per layer need {per_layer * pitch * 1e3:.2f} mm, "
                                                  f"winding height is {height * 1e3:.2f} mm", per_layer * pitch, height))
            remaining = section.turns
            for li in range(n_layers):
                if li > 0:
                    d += ins.layer_insulation
                n_here = min(per_layer, remaining)
                remaining -= n_here
                build = cond.radial_build()
                if cond.kind == "foil":
                    turn_z = [(z0 + (height - cond.width) / 2, z0 + (height + cond.width) / 2)]
                else:
                    pitch = cond.axial_pitch()
                    span = n_here * pitch
                    start = z0 + 0.5 * (height - span)
                    turn_z = [(start + k * pitch, start + (k + 1) * pitch) for k in range(n_here)]
                first = turn_counter.get(section.winding, 0)
                turn_counter[section.winding] = first + n_here
                lid = f"{section.winding}.s{si}.l{li}"
                mlt = RoundedRectRing.offset_perimeter(hx, hy, d + 0.5 * build, True)
                layers.append(PlacedLayer(id=lid, winding=section.winding, section=si, index=li, d_in=d, d_out=d + build,
                                          z0=turn_z[0][0], z1=turn_z[-1][1], turns=n_here, turn_z=turn_z, conductor=cond,
                                          mean_turn_length=mlt, first_turn=first))
                d += build
        d_end = d + ins.outer_wrap
        checks.append(LayoutCheck("radial_fit", d_end <= space["d_limit"] * (1 + 1e-9),
                                  f"radial build {d_end * 1e3:.2f} mm (incl. bobbin and insulation) vs available "
                                  f"{space['d_limit'] * 1e3:.2f} mm", d_end, space["d_limit"]))
        return layers, checks

    def build(self, *, arc_segments: int = 8) -> MagneticGeometry:  # noqa: ARG002 - solids keep exact arcs
        c, g, b = self.core, self.gap, self.bobbin
        A, F, E = c.length_a, c.centre_leg_width, c.window_width_total
        depth = self.depth
        zf = self._z_frame()
        solids: list[Solid] = []

        def half_profile(sign: float) -> tuple[tuple[float, float], ...]:
            zb = zf["z_back_bot"] if sign < 0 else zf["z_back_top"]
            zw = zf["z_win_bot"] if sign < 0 else zf["z_win_top"]
            zo = -zf["z_ogap"] if sign < 0 else zf["z_ogap"]
            zc = -zf["z_cgap"] if sign < 0 else zf["z_cgap"]
            pts = [(-A / 2, zb), (A / 2, zb), (A / 2, zo), (E / 2, zo), (E / 2, zw), (F / 2, zw), (F / 2, zc), (-F / 2, zc),
                   (-F / 2, zw), (-E / 2, zw), (-E / 2, zo), (-A / 2, zo)]
            return tuple(pts)

        for name, sign in (("core_bottom", -1.0), ("core_top", 1.0)):
            solids.append(ExtrudedProfile(name=name, role="core", material=self.core_material, profile=half_profile(sign),
                                          y0=-depth / 2, y1=depth / 2, meta={"half": "bottom" if sign < 0 else "top"}))
        hx, hy = F / 2, depth / 2
        space = self.winding_space()
        if b.present:
            solids.append(RoundedRectRing(name="bobbin_tube", role="bobbin", material=b.material, hx=hx, hy=hy, d_in=b.leg_clearance,
                                          d_out=b.leg_clearance + b.tube_thickness, z0=space["z0"], z1=space["z1"]))
            flange_out = c.window_width - b.outer_clearance * 0.0  # flange spans to the outer-leg face in X
            ox = hx + flange_out - 1e-6
            oy = hy + flange_out
            for name, z0, z1 in (("bobbin_flange_bottom", zf["z_win_bot"], zf["z_win_bot"] + b.flange_thickness),
                                 ("bobbin_flange_top", zf["z_win_top"] - b.flange_thickness, zf["z_win_top"])):
                solids.append(RoundedRectRing(name=name, role="bobbin", material=b.material, hx=hx, hy=hy, d_in=b.leg_clearance,
                                              d_out=flange_out, z0=z0, z1=z1, rounded=False, outer_override=(ox, oy)))
        layers, checks = self.place_layers()
        ins = self.layout.insulation if self.layout else InsulationSpec()
        for layer in layers:
            solids.append(RoundedRectRing(name=f"winding_{layer.id}", role="winding", material="copper", hx=hx, hy=hy, d_in=layer.d_in,
                                          d_out=layer.d_out, z0=layer.z0, z1=layer.z1,
                                          meta={"winding": layer.winding, "turns": layer.turns, "conductor": layer.conductor.describe(),
                                                "homogenised": True}))
        # insulation wraps after each layer (layer tape, section tape, outer wrap)
        for i, layer in enumerate(layers):
            nxt = layers[i + 1] if i + 1 < len(layers) else None
            gap = (nxt.d_in - layer.d_out) if nxt else ins.outer_wrap
            if gap > 1e-9:
                solids.append(RoundedRectRing(name=f"insulation_after_{layer.id}", role="insulation", material=ins.material, hx=hx, hy=hy,
                                              d_in=layer.d_out, d_out=layer.d_out + gap, z0=space["z0"], z1=space["z1"]))
        section = self._section(layers)
        derived = self._derived(layers, solids)
        if g.centre_gap > 0.1 * c.centre_leg_width:
            checks.append(LayoutCheck("gap_aspect", True, f"centre gap {g.centre_gap * 1e3:.2f} mm exceeds 10 % of the leg width: "
                                      "fringing is strong (informational)", g.centre_gap, 0.1 * c.centre_leg_width))
        return MagneticGeometry(model=self, solids=solids, layers=layers, section=section, checks=checks, derived=derived)

    # -- 2-D section ----------------------------------------------------------------------------------
    def _section(self, layers: list[PlacedLayer], *, air_padding: float | None = None) -> Section2D:
        c, g, b = self.core, self.gap, self.bobbin
        A, F, E = c.length_a, c.centre_leg_width, c.window_width_total
        zf = self._z_frame()
        pad = air_padding if air_padding is not None else 0.6 * max(A / 2, zf["z_back_top"])
        regions: list[Region] = [Region("air", "air", "air", 0.0, A / 2 + pad, zf["z_back_bot"] - pad, zf["z_back_top"] + pad)]
        regions.append(Region("window", "window_air", "air", F / 2, E / 2, zf["z_win_bot"], zf["z_win_top"]))
        cm = self.core_material
        # core halves (bottom: back, centre leg, outer leg; top mirrored)
        for half, sgn in (("bottom", -1.0), ("top", 1.0)):
            zb = zf["z_back_bot"] if sgn < 0 else zf["z_back_top"]
            zw = zf["z_win_bot"] if sgn < 0 else zf["z_win_top"]
            zo = -zf["z_ogap"] if sgn < 0 else zf["z_ogap"]
            zc = -zf["z_cgap"] if sgn < 0 else zf["z_cgap"]
            lo, hi = sorted((zb, zw))
            regions.append(Region(f"core_{half}_back", "core", cm, 0.0, A / 2, lo, hi))
            lo, hi = sorted((zw, zc))
            if hi - lo > 0:
                regions.append(Region(f"core_{half}_centre_leg", "core", cm, 0.0, F / 2, lo, hi))
            lo, hi = sorted((zw, zo))
            regions.append(Region(f"core_{half}_outer_leg", "core", cm, E / 2, A / 2, lo, hi))
        if g.centre_gap > 0:
            regions.append(Region("gap_centre", "gap", "air", 0.0, F / 2, -zf["z_cgap"], zf["z_cgap"]))
        if g.outer_gap > 0:
            regions.append(Region("gap_outer", "gap", "air", E / 2, A / 2, -zf["z_ogap"], zf["z_ogap"]))
        if b.present:
            sp = self.winding_space()
            x_t0 = F / 2 + b.leg_clearance
            regions.append(Region("bobbin_tube", "bobbin", b.material, x_t0, x_t0 + b.tube_thickness, sp["z0"], sp["z1"]))
            regions.append(Region("bobbin_flange_bottom", "bobbin", b.material, x_t0, E / 2, zf["z_win_bot"], zf["z_win_bot"] + b.flange_thickness))
            regions.append(Region("bobbin_flange_top", "bobbin", b.material, x_t0, E / 2, zf["z_win_top"] - b.flange_thickness, zf["z_win_top"]))
        ins_material = self.layout.insulation.material if self.layout else "polyimide_film"
        sp = self.winding_space()
        # homogenised winding packs (one per section): the thermal conduction path of the layers incl. inter-layer films
        for si in sorted({l.section for l in layers}):
            ls = [l for l in layers if l.section == si]
            kind = ls[0].conductor.kind
            pack_material = {"litz": "winding_pack_litz", "round": "winding_pack_round"}.get(kind, "winding_pack_foil")
            regions.append(Region(f"pack_{ls[0].winding}.s{si}", "winding_pack", pack_material, F / 2 + ls[0].d_in, F / 2 + ls[-1].d_out,
                                  min(l.z0 for l in ls), max(l.z1 for l in ls), winding=ls[0].winding, meta={"section": si}))
        # insulation between sections and the outer wrap (layer films are part of the pack)
        for i, layer in enumerate(layers):
            nxt = layers[i + 1] if i + 1 < len(layers) else None
            if nxt is not None and nxt.section == layer.section:
                continue
            gap = (nxt.d_in - layer.d_out) if nxt else (self.layout.insulation.outer_wrap if self.layout else 0.0)
            if gap > 1e-9:
                regions.append(Region(f"insulation_{layer.id}", "insulation", ins_material, F / 2 + layer.d_out, F / 2 + layer.d_out + gap,
                                      sp["z0"], sp["z1"]))
        for layer in layers:
            cond = layer.conductor
            x0 = F / 2 + layer.d_in
            for k, (tz0, tz1) in enumerate(layer.turn_z):
                turn = layer.first_turn + k
                if cond.kind in ("foil", "rectangular"):
                    rx0, rx1 = x0 + cond.insulation, x0 + cond.insulation + cond.thickness
                    if cond.kind == "foil":
                        rz0, rz1 = tz0 + cond.insulation, tz1 - cond.insulation
                        regions.append(Region(f"{layer.winding}.t{turn}", "conductor", "copper", rx0, rx1, rz0, rz1,
                                              winding=layer.winding, turn=turn, layer=layer.id,
                                              meta={"kind": cond.kind, "mesh_size_z": max((rz1 - rz0) / 40.0, 0.5e-3)}))
                    else:
                        w = cond.width
                        for p in range(cond.parallel):
                            pz0 = tz0 + cond.insulation + p * (w + 2 * cond.insulation)
                            regions.append(Region(f"{layer.winding}.t{turn}.p{p}", "conductor", "copper", rx0, rx1, pz0, pz0 + w,
                                                  winding=layer.winding, turn=turn, layer=layer.id, meta={"kind": cond.kind, "parallel_index": p}))
                else:
                    env = cond.envelope_diameter()
                    side = cond.equivalent_square_side() if cond.kind == "round" else math.sqrt(math.pi) / 2.0 * env
                    role = "conductor" if cond.kind == "round" else "stranded"
                    for p in range(cond.parallel):
                        cx = x0 + env / 2
                        cz = tz0 + env / 2 + p * env
                        regions.append(Region(f"{layer.winding}.t{turn}" + (f".p{p}" if cond.parallel > 1 else ""), role,
                                              "copper" if role == "conductor" else "litz", cx - side / 2, cx + side / 2,
                                              cz - side / 2, cz + side / 2, winding=layer.winding, turn=turn, layer=layer.id,
                                              meta={"kind": cond.kind, "copper_area": cond.copper_area() / cond.parallel,
                                                    "envelope": env, **({"strand_diameter": cond.strand_diameter,
                                                                         "strand_count": cond.strand_count} if cond.kind == "litz" else {})}))
        lengths: dict[str, float] = {}
        for name in {l.winding for l in layers}:
            ls = [l for l in layers if l.winding == name]
            n = sum(l.turns for l in ls)
            lengths[name] = sum(l.turns * l.mean_turn_length for l in ls) / n
        return Section2D(regions=regions, symmetry="half_x", coordinates="planar", depth=self.depth, winding_length=lengths,
                         description=f"{self.name}: half cross-section y=0 of the E-core (x >= 0)",
                         meta={"core_depth": self.depth, "window": {"x0": F / 2, "x1": E / 2, "z0": zf["z_win_bot"], "z1": zf["z_win_top"]}})

    # -- derived quantities ---------------------------------------------------------------------------
    def _derived(self, layers: list[PlacedLayer], solids: list[Solid]) -> dict[str, Any]:
        c = self.core
        zf = self._z_frame()
        core_volume = sum(s.volume() for s in solids if s.role == "core")
        windings: dict[str, Any] = {}
        for name in dict.fromkeys(l.winding for l in layers):
            ls = [l for l in layers if l.winding == name]
            cond = ls[0].conductor
            turns = sum(l.turns for l in ls)
            length = sum(l.turns * l.mean_turn_length for l in ls)
            area = cond.copper_area()
            windings[name] = {"turns": turns, "layers": len(ls), "conductor": cond.describe(), "conductor_kind": cond.kind,
                              "copper_area_per_turn_m2": area, "conductor_length_m": length, "mean_turn_length_m": length / turns,
                              "copper_volume_m3": length * area, "layer_mean_turn_lengths_m": [l.mean_turn_length for l in ls]}
        height_total = zf["z_back_top"] - zf["z_back_bot"]
        d_out = max((l.d_out for l in layers), default=0.0) + (self.layout.insulation.outer_wrap if self.layout else 0.0)
        bobbin_extent = (self.bobbin.leg_clearance + self.bobbin.tube_thickness) if self.bobbin.present else 0.0
        y_extent = max(self.depth, self.depth + 2 * max(d_out, self.winding_space()["d_limit"] if self.bobbin.present else d_out, bobbin_extent))
        box = c.length_a * height_total * y_extent
        window_area_side = c.window_width * (zf["z_win_top"] - zf["z_win_bot"])
        core_box_area = 2 * (c.length_a * height_total + c.length_a * self.depth + height_total * self.depth)
        core_exposed = core_box_area - 4 * window_area_side
        hx, hy = 0.5 * c.centre_leg_width, 0.5 * self.depth
        winding_exposed = 0.0
        if layers:
            h = max(l.z1 for l in layers) - min(l.z0 for l in layers)
            winding_exposed = h * (2 * c.centre_leg_width + 2 * math.pi * d_out)  # end-turn outer surface outside the core
        copper_volume = sum(w["copper_volume_m3"] for w in windings.values())
        space = self.winding_space()
        copper_area_window = sum(l.turns * l.conductor.copper_area() for l in layers)
        return {
            "core_volume_m3": core_volume, "core_volume_formula": "exact volume of the two extruded half profiles",
            "bounding_volume_m3": box, "bounding_volume_formula": "A x (2B + spacer) x (C + 2 x winding/bobbin overhang)",
            "window_area_per_side_m2": window_area_side,
            "winding_space": space,
            "copper_area_in_window_m2": copper_area_window,
            "window_fill_bare_copper": copper_area_window / window_area_side if window_area_side > 0 else None,
            "copper_volume_m3": copper_volume,
            "core_exposed_area_m2": core_exposed, "core_exposed_area_formula": "outer box surface minus the four window openings",
            "winding_exposed_area_m2": winding_exposed,
            "winding_exposed_area_formula": "outer end-turn surface outside the core: height x (2F + 2 pi d_out)",
            "centre_leg_area_m2": c.centre_leg_width * self.depth, "outer_leg_area_m2": 2 * c.outer_leg_width * self.depth,
            "centre_gap_m": self.gap.centre_gap, "outer_gap_m": self.gap.outer_gap, "windings": windings,
            "turn_offset_curve": {"hx": hx, "hy": hy},
        }


def arrangement_sections(arrangement: str, windings: dict[str, tuple[int, ConductorSpec]]) -> list[WindingSection]:
    """Winding sections for an arrangement label such as ``"P-S"``, ``"P-S-P"``, ``"S-P-S"`` or ``"P-S-P-S"``.

    ``windings`` maps the letter (``P``, ``S``, ``T`` ...) to ``(turns, conductor)``; a winding that appears ``k``
    times is split into ``k`` series sections with turns as equal as possible (larger parts first).
    """
    letters = [p.strip() for p in arrangement.split("-") if p.strip()]
    if not letters:
        raise ValueError("empty arrangement")
    unknown = sorted(set(letters) - set(windings))
    if unknown:
        raise ValueError(f"arrangement {arrangement!r} names unknown winding(s) {unknown}")
    missing = sorted(set(windings) - set(letters))
    if missing:
        raise ValueError(f"arrangement {arrangement!r} does not place winding(s) {missing}")
    counts = {k: letters.count(k) for k in windings}
    parts: dict[str, list[int]] = {}
    for k, (turns, _) in windings.items():
        n = counts[k]
        if turns < n:
            raise ValueError(f"winding {k} has {turns} turns, cannot be split into {n} sections")
        base, extra = divmod(turns, n)
        parts[k] = [base + (1 if i < extra else 0) for i in range(n)]
    used = {k: 0 for k in windings}
    sections = []
    names = {"P": "primary", "S": "secondary", "T": "tertiary"}
    for letter in letters:
        turns = parts[letter][used[letter]]
        used[letter] += 1
        sections.append(WindingSection(winding=names.get(letter, letter.lower()), turns=turns, conductor=windings[letter][1]))
    return sections


__all__ = ["BobbinSpec", "ECoreDimensions", "ECoreMagnetic", "GEOMETRY_MODEL_ID", "GEOMETRY_MODEL_VERSION", "GapSpec", "InsulationSpec",
           "LayoutCheck", "MagneticGeometry", "PlacedLayer", "WindingLayout", "WindingSection", "arrangement_sections"]
