"""Basic winding resistance and copper-loss estimation.

Model: DC resistance from mean turn length and conductor area at the conductor temperature;
AC/DC ratio from Dowell's 1-D layered-winding model (sinusoidal current at the fundamental);
copper loss ``I_rms^2 R_ac``. Harmonic currents, 2-D/3-D field effects, gap fringing fields and
terminations are not modelled: the result carries these as limitations.

Assumptions are explicit and returned as :class:`~aipe_magnetics.common.results.Assumption` objects
so the caller can list them in the engineering result.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aipe_magnetics.api.types import WindingSpec
from aipe_magnetics.common.errors import InvalidInputError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import Assumption, Limitation, SourceReference, Warning_
from aipe_magnetics.engineering import formulas as F

DEFAULT_CURRENT_DENSITY = 4.0e6  # A/m^2 (4 A/mm^2)
DEFAULT_LITZ_STRAND_DIAMETER = 0.1e-3  # m
DEFAULT_FOIL_THICKNESS = 0.1e-3  # m
INSULATION_BUILD_FACTOR = 1.1  # outer/bare diameter for enamelled wire (assumption)

DOWELL_REFERENCE = SourceReference(
    id="ref.dowell-1966", kind="publication",
    reference="P.L. Dowell, 'Effects of eddy currents in transformer windings', Proc. IEE 113(8), 1966, pp. 1387-1394",
    license="NOASSERTION", note="1-D layered winding AC resistance factor")
SULLIVAN_LITZ_REFERENCE = SourceReference(
    id="ref.sullivan-1999-litz", kind="publication",
    reference="C.R. Sullivan, 'Optimal choice for number of strands in a litz-wire transformer winding', IEEE Trans. Power Electron. 14(2), 1999, pp. 283-291",
    license="NOASSERTION", note="strand-level eddy-current treatment of litz wire")


class WindingDesign(BaseModel):
    """Outcome of :func:`design_winding` (SI floats plus explanatory metadata)."""

    model_config = ConfigDict(extra="forbid")

    conductor_type: str
    turns: int
    current_rms: float
    conductor_area: float = Field(description="total copper cross-section per turn (m^2)")
    conductor_diameter: float | None = None
    strand_diameter: float | None = None
    strand_count: int | None = None
    foil_thickness: float | None = None
    foil_width: float | None = None
    layers: int
    turns_per_layer: int | None = None
    mean_turn_length: float
    conductor_length: float
    conductor_temperature: float
    resistivity: float
    resistance_dc: float
    skin_depth: float
    dowell_penetration_ratio: float
    ac_resistance_factor: float
    resistance_ac: float
    copper_loss: float
    window_fill: float | None = None
    assumptions: list[Assumption] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)
    warnings: list[Warning_] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)

    def quantities(self, prefix: str = "") -> dict[str, tuple[str, Quantity, str]]:
        """``key -> (label, quantity, status)`` for result envelopes."""
        p = prefix
        out: dict[str, tuple[str, Quantity, str]] = {
            f"{p}conductor_area": ("Conductor area per turn", Quantity(value=self.conductor_area, unit="m^2"), "calculated"),
            f"{p}resistance_dc": ("DC resistance", Quantity(value=self.resistance_dc, unit="Ohm"), "calculated"),
            f"{p}ac_resistance_factor": ("AC/DC resistance factor (Dowell)", Quantity(value=self.ac_resistance_factor, unit="1"), "estimated"),
            f"{p}resistance_ac": ("AC resistance at the fundamental", Quantity(value=self.resistance_ac, unit="Ohm"), "estimated"),
            f"{p}copper_loss": ("Copper loss", Quantity(value=self.copper_loss, unit="W"), "estimated"),
            f"{p}skin_depth": ("Skin depth", Quantity(value=self.skin_depth, unit="m"), "calculated"),
            f"{p}conductor_length": ("Conductor length", Quantity(value=self.conductor_length, unit="m"), "calculated"),
        }
        if self.window_fill is not None:
            out[f"{p}window_fill"] = ("Window fill (bare copper / window area)", Quantity(value=self.window_fill, unit="1"), "estimated")
        return out


def _window_breadth(window_area: float | None, dimensions: dict[str, float] | None) -> tuple[float | None, str | None]:
    if dimensions:
        for key in ("window_breadth", "winding_breadth", "bobbin_breadth"):
            if key in dimensions and dimensions[key] > 0:
                return float(dimensions[key]), None
        # datasheet outline dimensions: winding breadth ~ 90 % of the core window height (bobbin flanges)
        if dimensions.get("window_height_half", 0) > 0:
            return 0.9 * 2.0 * float(dimensions["window_height_half"]), "winding breadth taken as 90 % of the core window height (2 x half height)"
        if dimensions.get("window_height", 0) > 0:
            return 0.9 * float(dimensions["window_height"]), "winding breadth taken as 90 % of the core window height"
    if window_area:
        # assumption: window aspect ratio breadth:height = 2:1 -> breadth = sqrt(2 * A_w)
        return math.sqrt(2.0 * window_area), "window breadth estimated as sqrt(2*A_w) (2:1 aspect assumption)"
    return None, None


def design_winding(*, turns: int, current_rms: float, frequency: float, mean_turn_length: float, spec: WindingSpec,
                   temperature: float, window_area: float | None = None, dimensions: dict[str, float] | None = None,
                   layers_override: int | None = None) -> WindingDesign:
    """Size one winding and estimate its DC/AC resistance and copper loss.

    Parameters are SI floats: ``current_rms`` A, ``frequency`` Hz, ``mean_turn_length`` m,
    ``temperature`` K (conductor temperature), ``window_area`` m^2 (bobbin window available to all
    windings; used only for the fill estimate and the layer estimate).
    """
    if isinstance(turns, bool) or not isinstance(turns, int) or turns < 1:
        raise InvalidInputError(f"turns must be a positive integer, got {turns!r}")
    i_rms = F._positive("current_rms", current_rms)
    f = F._positive("frequency", frequency)
    mlt = F._positive("mean_turn_length", mean_turn_length)
    t_cond = spec.conductor_temperature.to("K").value if spec.conductor_temperature is not None else F._positive("temperature", temperature)

    assumptions: list[Assumption] = []
    limitations: list[Limitation] = []
    warnings: list[Warning_] = []
    sources: list[SourceReference] = [DOWELL_REFERENCE]

    if spec.current_density is not None:
        j = spec.current_density.to("A/m^2").value
        assumptions.append(Assumption(id="current_density_user", statement=f"Current density {j / 1e6:.2f} A/mm^2 supplied by the user.",
                                      basis="user_input", affects=["conductor_area", "copper_loss"]))
    else:
        j = DEFAULT_CURRENT_DENSITY
        assumptions.append(Assumption(id="current_density_default", statement="Current density 4 A/mm^2 (default sizing assumption).",
                                      basis="default", affects=["conductor_area", "copper_loss"]))
    area = F.conductor_area_for_current_density(i_rms, j)
    rho = F.copper_resistivity(t_cond)
    assumptions.append(Assumption(id="copper_resistivity_model",
                                  statement=f"Annealed copper, rho(T) = 1.68e-8 Ohm*m * (1 + 0.00393/K * (T - 20 degC)) at {t_cond - 273.15:.0f} degC.",
                                  basis="default", affects=["resistance_dc", "copper_loss"]))
    delta = F.skin_depth(f, rho)
    breadth, breadth_note = _window_breadth(window_area, dimensions)
    if breadth_note:
        assumptions.append(Assumption(id="window_breadth_estimate", statement=breadth_note, basis="engineering_judgement",
                                      affects=["layers", "ac_resistance_factor"]))

    conductor_diameter = strand_diameter = foil_thickness = foil_width = None
    strand_count = turns_per_layer = None
    porosity = 1.0

    if spec.conductor_type == "round":
        conductor_diameter = F.round_wire_diameter(area)
        pitch = conductor_diameter * INSULATION_BUILD_FACTOR
        assumptions.append(Assumption(id="insulation_build", statement="Enamelled wire outer diameter = 1.1 x bare diameter.",
                                      basis="default", affects=["layers", "window_fill"]))
        if layers_override or spec.layers:
            layers = int(layers_override or spec.layers)  # type: ignore[arg-type]
        elif breadth:
            turns_per_layer = max(1, int(math.floor(breadth / pitch)))
            layers = int(math.ceil(turns / turns_per_layer))
        else:
            layers = int(math.ceil(math.sqrt(turns)))
            assumptions.append(Assumption(id="layers_sqrt_estimate", statement="Layers estimated as ceil(sqrt(N)) (no window dimensions available).",
                                          basis="engineering_judgement", affects=["ac_resistance_factor"]))
        h = F.round_wire_equivalent_thickness(conductor_diameter)
        porosity = min(1.0, conductor_diameter / pitch)
        f_r = F.dowell_ac_resistance_factor(layers, h, delta, porosity)
        copper_area = area
    elif spec.conductor_type == "litz":
        strand_diameter = spec.strand_diameter.to("m").value if spec.strand_diameter is not None else DEFAULT_LITZ_STRAND_DIAMETER
        if spec.strand_diameter is None:
            assumptions.append(Assumption(id="litz_strand_default", statement="Litz strand diameter 0.1 mm (default).", basis="default",
                                          affects=["ac_resistance_factor"]))
        if strand_diameter > 2 * delta:
            warnings.append(Warning_(code="litz_strand_too_thick", message=f"Litz strand diameter {strand_diameter * 1e3:.3f} mm exceeds "
                                     f"2 x skin depth ({2 * delta * 1e3:.3f} mm); strand-level eddy loss will be significant."))
        strand_area = F.round_wire_area(strand_diameter)
        strand_count = int(math.ceil(area / strand_area))
        copper_area = strand_count * strand_area
        bundle_diameter = strand_diameter * math.sqrt(strand_count) * 1.25  # packing factor assumption
        assumptions.append(Assumption(id="litz_bundle_packing", statement="Litz bundle diameter = 1.25 x sqrt(n) x strand diameter (packing/insulation assumption).",
                                      basis="engineering_judgement", affects=["layers", "window_fill"]))
        if layers_override or spec.layers:
            bundle_layers = int(layers_override or spec.layers)  # type: ignore[arg-type]
        elif breadth:
            turns_per_layer = max(1, int(math.floor(breadth / bundle_diameter)))
            bundle_layers = int(math.ceil(turns / turns_per_layer))
        else:
            bundle_layers = int(math.ceil(math.sqrt(turns)))
            assumptions.append(Assumption(id="layers_sqrt_estimate", statement="Bundle layers estimated as ceil(sqrt(N)) (no window dimensions available).",
                                          basis="engineering_judgement", affects=["ac_resistance_factor"]))
        # strand-level Dowell: effective strand layers = bundle layers x sqrt(strands per bundle)
        layers = max(1, int(round(bundle_layers * math.sqrt(strand_count))))
        h = F.round_wire_equivalent_thickness(strand_diameter)
        porosity = min(1.0, (strand_count * strand_area) / (math.pi * bundle_diameter ** 2 / 4.0)) if bundle_diameter > 0 else 1.0
        porosity = max(0.1, porosity)
        f_r = F.dowell_ac_resistance_factor(layers, h, delta, porosity)
        sources.append(SULLIVAN_LITZ_REFERENCE)
        limitations.append(Limitation(code="litz_approximation",
                                      statement="Litz AC resistance uses Dowell's model with bundle_layers*sqrt(strands) equivalent strand layers "
                                                "and an estimated porosity; strand twisting/transposition effects are not modelled."))
    elif spec.conductor_type == "foil":
        foil_thickness = spec.foil_thickness.to("m").value if spec.foil_thickness is not None else DEFAULT_FOIL_THICKNESS
        if spec.foil_thickness is None:
            assumptions.append(Assumption(id="foil_thickness_default", statement="Foil thickness 0.1 mm (default).", basis="default",
                                          affects=["ac_resistance_factor", "window_fill"]))
        foil_width = area / foil_thickness
        if breadth and foil_width > breadth:
            warnings.append(Warning_(code="foil_width_exceeds_window", message=f"Required foil width {foil_width * 1e3:.1f} mm exceeds the "
                                     f"estimated window breadth {breadth * 1e3:.1f} mm; a thicker foil or parallel foils are needed."))
        layers = turns if not (layers_override or spec.layers) else int(layers_override or spec.layers)  # type: ignore[arg-type]
        turns_per_layer = 1
        f_r = F.dowell_ac_resistance_factor(layers, foil_thickness, delta, 1.0)
        copper_area = area
    else:  # pragma: no cover - pydantic restricts the literal
        raise InvalidInputError(f"unsupported conductor type {spec.conductor_type}")

    length = turns * mlt
    r_dc = F.dc_resistance(rho, length, copper_area)
    r_ac = r_dc * f_r
    loss = F.copper_loss(i_rms, r_ac)
    fill = None
    if window_area:
        fill = F.window_fill(turns, copper_area, window_area)
    if f_r > 3:
        warnings.append(Warning_(code="high_ac_resistance_factor", message=f"Dowell AC/DC factor {f_r:.2f} is high; reduce layers, use thinner "
                                 "conductors or litz wire."))
    limitations.append(Limitation(code="dowell_1d", statement="Dowell 1-D model: sinusoidal current at the fundamental, layers spanning the window, "
                                  "no gap fringing, no harmonic content, no termination effects."))
    limitations.append(Limitation(code="no_thermal_model", statement="Conductor temperature is an input; no thermal model couples losses to temperature."))
    return WindingDesign(
        conductor_type=spec.conductor_type, turns=turns, current_rms=i_rms, conductor_area=copper_area,
        conductor_diameter=conductor_diameter, strand_diameter=strand_diameter, strand_count=strand_count,
        foil_thickness=foil_thickness, foil_width=foil_width, layers=layers, turns_per_layer=turns_per_layer,
        mean_turn_length=mlt, conductor_length=length, conductor_temperature=t_cond, resistivity=rho,
        resistance_dc=r_dc, skin_depth=delta, dowell_penetration_ratio=(math.sqrt(porosity) * (foil_thickness if foil_thickness else
                                                                         F.round_wire_equivalent_thickness(strand_diameter or conductor_diameter)) / delta),
        ac_resistance_factor=f_r, resistance_ac=r_ac, copper_loss=loss, window_fill=fill,
        assumptions=assumptions, limitations=limitations, warnings=warnings, sources=sources)


def winding_to_json(design: WindingDesign) -> dict[str, Any]:
    return design.model_dump(mode="json", exclude_none=True)
