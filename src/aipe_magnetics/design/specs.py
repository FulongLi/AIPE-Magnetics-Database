"""Engineering specifications of the Design workflows, with explicit assumptions and open questions.

A specification is a validated pydantic model whose fields carry SI values. Fields the brief does not define have
*documented* defaults; :meth:`DABTransformerSpec.assumption_report` / :meth:`InductorSpec.assumption_report` list
every value that was assumed rather than specified, and the questions that must be answered before the design can
be qualified (waveform / modulation, operating range, isolation, cooling). ``strict=True`` turns unanswered
critical questions into a rejection instead of an assumption.

Quantities may be given as ``{"value": x, "unit": "..."}`` in any registered unit (normalised to SI).
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aipe_magnetics.common.errors import InvalidInputError
from aipe_magnetics.common.quantity import Quantity


def _si(value: Any, unit: str, name: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    q = Quantity.parse(value, name=name)
    from aipe_magnetics.common.units import units
    if not units.compatible(q.unit, unit):
        raise InvalidInputError(f"{name} must have the dimension of {unit!r}, got {q.unit!r}")
    return q.magnitude(unit)


class CoolingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["natural", "forced"] = "forced"
    ambient: float = Field(default=313.15, description="K")
    forced_h: float | None = Field(default=40.0, description="W/m^2/K (forced mode)")
    radiation: bool = True
    emissivity: float = 0.9

    @field_validator("ambient", mode="before")
    @classmethod
    def _t(cls, v: Any) -> float:
        return _si(v, "K", "ambient")

    def to_cooling(self):
        from aipe_magnetics.simulation.thermal.convection import Cooling
        return Cooling(mode=self.mode, ambient=self.ambient, forced_h=self.forced_h if self.mode == "forced" else None,
                       radiation=self.radiation, emissivity=self.emissivity)


class IsolationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    working_voltage: float | None = Field(default=None, description="V (DC or peak)")
    insulation_class: Literal["functional", "basic", "supplementary", "reinforced"] | None = None
    creepage: float | None = Field(default=None, description="required creepage distance (m)")
    clearance: float | None = Field(default=None, description="required clearance (m)")
    standard: str | None = None
    pollution_degree: int | None = None


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_winding_hotspot: float = Field(default=393.15, description="K")
    max_core_hotspot: float = Field(default=383.15, description="K")
    flux_fraction_of_bsat: float = Field(default=0.7, gt=0, le=1)
    max_current_density: float = Field(default=6.0e6, description="A/m^2 (RMS, bare copper)")
    max_window_fill: float = Field(default=0.45, gt=0, le=1, description="bare copper / window area (one side)")
    turns_ratio_tolerance: float = Field(default=0.05, ge=0)
    inductance_tolerance: float = Field(default=0.1, ge=0, description="inductor: relative tolerance of L at the design point")
    min_inductance_at_peak: float = Field(default=0.8, gt=0, le=1, description="inductor: L(I_pk) / L_target lower limit")

    @field_validator("max_winding_hotspot", "max_core_hotspot", mode="before")
    @classmethod
    def _t(cls, v: Any) -> float:
        return _si(v, "K", "temperature limit")

    @field_validator("max_current_density", mode="before")
    @classmethod
    def _j(cls, v: Any) -> float:
        return _si(v, "A/m^2", "max_current_density")


class DesignSpace(BaseModel):
    """Discrete optimisation variables (catalogue filters, conductor and arrangement options)."""

    model_config = ConfigDict(extra="forbid")
    materials: list[str] | None = Field(default=None, description="material ids/grades; None = every catalogue material with the data needed")
    cores: list[str] | None = Field(default=None, description="core ids; None = every catalogue core with parametric E geometry")
    conductors: list[Literal["litz", "foil", "round"]] = Field(default_factory=lambda: ["litz"])
    litz_strand_diameters: list[float] = Field(default_factory=lambda: [0.071e-3, 0.1e-3])
    foil_thicknesses: list[float] = Field(default_factory=lambda: [0.1e-3, 0.2e-3, 0.3e-3])
    round_diameters: list[float] = Field(default_factory=lambda: [0.8e-3, 1.0e-3, 1.25e-3, 1.6e-3, 2.0e-3])
    current_densities: list[float] = Field(default_factory=lambda: [3.0e6, 4.0e6, 5.0e6])
    arrangements: list[str] = Field(default_factory=lambda: ["P-S", "P-S-P"])
    primary_turns: list[int] | None = Field(default=None, description="None = derived from the flux limit (minimum N1 and the next steps)")
    turn_steps: int = Field(default=4, ge=1, le=12, description="number of N1 values above the minimum explored")
    gaps: list[float] | None = Field(default=None, description="inductor: centre gaps (m); None = solved for the target inductance")
    frequencies: list[float] | None = Field(default=None, description="permitted switching frequencies (Hz); None = the specified one")
    parallel_round: list[int] = Field(default_factory=lambda: [1, 2, 3])
    material_offer_policy: Literal["catalogue", "standard_size_equivalent"] = Field(
        default="catalogue",
        description="'catalogue': a core is combined only with the materials its record lists; 'standard_size_equivalent': an IEC 62317 "
                    "standard outline is assumed available in every candidate material (dimensions and effective parameters of the "
                    "catalogue record; recorded as an assumption)")


class DABTransformerSpec(BaseModel):
    """10 kW-class DAB isolation transformer specification (defaults = the AIPE v0.3 demonstration brief)."""

    model_config = ConfigDict(extra="forbid")
    name: str = "dab-10kw"
    power: float = 10e3
    frequency: float = 100e3
    v1: float = 800.0
    v2: float = 400.0
    v2_range: tuple[float, float] | None = (360.0, 440.0)
    v1_range: tuple[float, float] | None = None
    modulation: Literal["sps"] = "sps"
    rated_phase_shift: float | None = Field(default=math.pi / 6, description="rad; sets the series inductance when it is not given")
    series_inductance: float | None = Field(default=None, description="total series inductance referred to the primary (H)")
    cooling: CoolingSpec = Field(default_factory=CoolingSpec)
    isolation: IsolationSpec = Field(default_factory=IsolationSpec)
    limits: Limits = Field(default_factory=Limits)
    space: DesignSpace = Field(default_factory=DesignSpace)
    loss_temperature_initial: float = 373.15
    specified: list[str] = Field(default_factory=lambda: ["power", "frequency", "v1", "v2"],
                                 description="fields given by the brief; every other value is an assumption")

    @field_validator("power", mode="before")
    @classmethod
    def _p(cls, v: Any) -> float:
        return _si(v, "W", "power")

    @field_validator("frequency", mode="before")
    @classmethod
    def _f(cls, v: Any) -> float:
        return _si(v, "Hz", "frequency")

    @field_validator("v1", "v2", mode="before")
    @classmethod
    def _v(cls, v: Any) -> float:
        return _si(v, "V", "voltage")

    @field_validator("series_inductance", mode="before")
    @classmethod
    def _l(cls, v: Any) -> float | None:
        return None if v is None else _si(v, "H", "series_inductance")

    @model_validator(mode="after")
    def _rules(self) -> "DABTransformerSpec":
        if min(self.power, self.frequency, self.v1, self.v2) <= 0:
            raise ValueError("power, frequency and voltages must be positive")
        if self.series_inductance is None and self.rated_phase_shift is None:
            raise ValueError("give series_inductance or rated_phase_shift")
        if self.v2_range is not None and not (0 < self.v2_range[0] <= self.v2 <= self.v2_range[1]):
            raise ValueError("v2_range must bracket v2")
        return self

    def assumption_report(self) -> dict[str, Any]:
        assumed, questions = [], []

        def a(field_: str, statement: str, critical: bool, question: str | None = None) -> None:
            if field_ not in self.specified:
                assumed.append({"field": field_, "statement": statement, "critical": critical})
                if question:
                    questions.append({"field": field_, "question": question, "critical": critical, "assumed_answer": statement})

        a("modulation", "single-phase-shift (SPS) modulation, 50 % square voltages on both bridges, dead time neglected", True,
          "Which modulation (SPS, extended / dual / triple phase shift) and dead time does the converter use?")
        a("series_inductance", (f"series inductance {self.series_inductance * 1e6:.1f} uH given" if self.series_inductance else
                                f"series inductance chosen for a {math.degrees(self.rated_phase_shift or 0):.0f} deg phase shift at rated power"), True,
          "What total series (leakage + external) inductance or rated phase shift is intended?")
        a("v2_range", f"secondary voltage range {self.v2_range} V at rated power" if self.v2_range else "fixed secondary voltage", True,
          "What is the operating voltage range of both DC links, and at which power levels?")
        a("v1_range", "primary DC link fixed at the nominal value" if self.v1_range is None else f"primary range {self.v1_range} V", False)
        a("cooling", f"{self.cooling.mode} cooling, ambient {self.cooling.ambient - 273.15:.0f} degC"
                     + (f", h = {self.cooling.forced_h:g} W/m^2/K" if self.cooling.mode == "forced" else ""), True,
          "What cooling (natural / forced air speed / cold plate) and ambient temperature apply?")
        a("isolation", "no isolation requirement defined: insulation is not designed to a standard (creepage margin 0, functional "
                       "inter-winding film only)", True,
          "Which isolation class (basic / reinforced), working voltage, standard, pollution degree and creepage / clearance apply?")
        a("limits", f"winding hot spot <= {self.limits.max_winding_hotspot - 273.15:.0f} degC, core <= "
                    f"{self.limits.max_core_hotspot - 273.15:.0f} degC, B <= {self.limits.flux_fraction_of_bsat:.0%} Bsat(T), J <= "
                    f"{self.limits.max_current_density / 1e6:.1f} A/mm^2, window fill <= {self.limits.max_window_fill:.2f}", False)
        a("frequency_tolerance", "switching frequency fixed", False)
        return {"assumed": assumed, "open_questions": questions,
                "critical_open": [q["field"] for q in questions if q["critical"]]}


class InductorSpec(BaseModel):
    """Gapped high-frequency inductor specification (buck-type operating point by default)."""

    model_config = ConfigDict(extra="forbid")
    name: str = "buck-inductor"
    inductance: float = 100e-6
    dc_current: float = 20.0
    frequency: float = 100e3
    topology: Literal["buck", "boost", "given_ripple"] = "buck"
    v_in: float | None = 400.0
    v_out: float | None = 200.0
    ripple_peak_to_peak: float | None = Field(default=None, description="A (topology='given_ripple')")
    duty: float | None = Field(default=None, description="rising fraction of the ripple; derived for buck/boost")
    cooling: CoolingSpec = Field(default_factory=lambda: CoolingSpec(mode="natural"))
    limits: Limits = Field(default_factory=Limits)
    space: DesignSpace = Field(default_factory=lambda: DesignSpace(conductors=["round", "foil"], arrangements=["L"]))
    specified: list[str] = Field(default_factory=lambda: ["inductance", "dc_current", "frequency", "topology", "v_in", "v_out"])

    @field_validator("inductance", mode="before")
    @classmethod
    def _l(cls, v: Any) -> float:
        return _si(v, "H", "inductance")

    @field_validator("dc_current", mode="before")
    @classmethod
    def _i(cls, v: Any) -> float:
        return _si(v, "A", "dc_current")

    @field_validator("frequency", mode="before")
    @classmethod
    def _f(cls, v: Any) -> float:
        return _si(v, "Hz", "frequency")

    @model_validator(mode="after")
    def _rules(self) -> "InductorSpec":
        if self.topology in ("buck", "boost"):
            if not (self.v_in and self.v_out):
                raise ValueError(f"{self.topology} needs v_in and v_out")
            if self.topology == "buck" and not self.v_out < self.v_in:
                raise ValueError("buck needs v_out < v_in")
            if self.topology == "boost" and not self.v_out > self.v_in:
                raise ValueError("boost needs v_out > v_in")
        elif self.ripple_peak_to_peak is None or self.duty is None:
            raise ValueError("given_ripple needs ripple_peak_to_peak and duty")
        return self

    def duty_cycle(self) -> float:
        if self.topology == "buck":
            return self.v_out / self.v_in  # type: ignore[operator]
        if self.topology == "boost":
            return 1.0 - self.v_in / self.v_out  # type: ignore[operator]
        return float(self.duty)  # type: ignore[arg-type]

    def ripple(self, frequency: float | None = None, inductance: float | None = None) -> float:
        f = frequency or self.frequency
        L = inductance or self.inductance
        d = self.duty_cycle()
        if self.topology == "buck":
            return (self.v_in - self.v_out) * d / (L * f)  # type: ignore[operator]
        if self.topology == "boost":
            return self.v_in * d / (L * f)  # type: ignore[operator]
        return float(self.ripple_peak_to_peak) * self.frequency / f * self.inductance / L  # type: ignore[arg-type]

    def current_waveform(self, frequency: float | None = None, inductance: float | None = None):
        from aipe_magnetics.engineering.periodic import triangular
        return triangular(self.ripple(frequency, inductance), self.duty_cycle(), self.dc_current, unit="A", label="inductor current")

    def assumption_report(self) -> dict[str, Any]:
        assumed, questions = [], []
        if "cooling" not in self.specified:
            assumed.append({"field": "cooling", "statement": f"{self.cooling.mode} cooling, ambient {self.cooling.ambient - 273.15:.0f} degC",
                            "critical": True})
            questions.append({"field": "cooling", "question": "What cooling and ambient temperature apply?", "critical": True,
                              "assumed_answer": assumed[-1]["statement"]})
        assumed.append({"field": "operating_point", "statement": f"continuous conduction, steady state at I_dc = {self.dc_current:g} A, "
                        f"ripple from the {self.topology} relation at the switching frequency", "critical": False})
        assumed.append({"field": "limits", "statement": f"L within +-{self.limits.inductance_tolerance:.0%} at low current, L(I_pk) >= "
                        f"{self.limits.min_inductance_at_peak:.0%} L, hot spot <= {self.limits.max_winding_hotspot - 273.15:.0f} degC",
                        "critical": False})
        return {"assumed": assumed, "open_questions": questions, "critical_open": [q["field"] for q in questions if q["critical"]]}


__all__ = ["CoolingSpec", "DABTransformerSpec", "DesignSpace", "InductorSpec", "IsolationSpec", "Limits"]
