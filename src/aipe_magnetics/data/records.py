"""AIPE Magnetic Data Standard v0.2 - normative record models.

These pydantic models are the *normative* definition of the v0.2 record standard; the JSON Schema
files in ``aipe_magnetics/schemas/v0.2`` are generated from and tested against them.

Design principles (see ``docs/data-standard-v0.2.md``):

* Every record has a common **envelope**: ``schema_version``, stable ``id``, ``kind``, ``name``,
  ``synthetic`` flag, record-level ``license``, ``evidence`` + ``evidence_refs`` (v0.1 compatible),
  optional ``provenance``, ``quality``, ``artifacts`` (with checksums) and namespaced ``extensions``.
* **All physical quantities are SI** ``{value, unit}`` objects. Where a value was converted, the
  ``original`` ``{value, unit}`` is retained so normalisation is reproducible.
* **Missing values are omitted**, never zero-filled. ``null`` is not a value.
* **Uncertainty** is optional and never fabricated (``kind: unknown`` when not quantified).
* Large numerical arrays (waveforms, B-H loops with many points) are **not stored in JSON**: a
  ``waveform_dataset`` record references an HDF5 artifact and describes its layout, units, sampling
  rate, timebase and sign conventions.
* v0.1 kinds remain valid v0.2 kinds (``core_material``, ``winding``, ``inductor``, ``transformer``,
  ``loss_model`` ...); the migration tool upgrades v0.1 files without changing their meaning.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.units import units

SCHEMA_VERSION = "0.2.0"
ID_PATTERN = r"^[a-z][a-z0-9]*(?:[.:-][a-z0-9]+(?:-[a-z0-9]+)*)+$"
_ID_RE = re.compile(ID_PATTERN)

RecordKind = Literal[
    # identities and materials
    "manufacturer", "magnetic_material", "core_material",
    # geometry
    "core_family", "core_geometry", "winding", "winding_geometry",
    # components
    "inductor", "transformer", "magnetic_component",
    # conditions and excitation
    "operating_condition", "excitation_waveform",
    # measurements and datasets
    "measurement", "core_loss_measurement", "bh_loop", "waveform_dataset", "dataset",
    # models
    "loss_model", "model_metadata",
]

EvidenceKind = Literal["assumption", "source", "measurement", "analytical", "simulation"]
ReviewStatus = Literal["unreviewed", "reviewed", "rejected"]


def _check_id(value: str, label: str = "id") -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise ValueError(f"{label} {value!r} must match {ID_PATTERN}")
    return value


def _si(q: Quantity, unit: str, label: str) -> Quantity:
    """Require canonical SI unit for a stored quantity (the schema is an SI standard)."""
    if q.unit != unit:
        raise ValueError(f"{label} must be stored in SI unit {unit!r}; got {q.unit!r} (use Quantity.si to normalise)")
    return q


def _si_dim(q: Quantity, unit: str, label: str) -> Quantity:
    if not units.compatible(q.unit, unit) or not units.is_si(q.unit):
        raise ValueError(f"{label} must be stored in a canonical SI unit compatible with {unit!r}; got {q.unit!r}")
    return q


def _positive(q: Quantity, label: str) -> Quantity:
    """v0.1 parity: physical magnitudes such as inductance, areas and turns ratio are strictly positive."""
    if q.value <= 0:
        raise ValueError(f"{label} must be positive, got {q.value} {q.unit}")
    return q


def _nonnegative(q: Quantity, label: str) -> Quantity:
    if q.value < 0:
        raise ValueError(f"{label} must be non-negative, got {q.value} {q.unit}")
    return q


_RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")


def _timestamp(value: str, label: str) -> str:
    """RFC 3339 date-time with an explicit offset (Core and v0.1 schema parity)."""
    if not isinstance(value, str) or not _RFC3339.match(value):
        raise ValueError(f"{label} must be an RFC 3339 date-time with time zone, e.g. 2026-10-19T00:00:00Z; got {value!r}")
    from datetime import datetime
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{label} is not a valid date-time: {value!r}") from None
    return value


# --------------------------------------------------------------------------------------------------
# Envelope parts
# --------------------------------------------------------------------------------------------------
class Evidence(BaseModel):
    """v0.1-compatible evidence item with optional v0.2 locator/url/checksum fields."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: EvidenceKind
    reference: str = Field(min_length=1)
    description: str = Field(min_length=1)
    created_at: str = Field(description="RFC 3339 timestamp of the evidence event (not build time)")
    review_status: ReviewStatus
    license: str = Field(min_length=1, description="SPDX identifier/expression or NOASSERTION")
    locator: str | None = Field(default=None, description="page/table/figure/column/record field")
    url: str | None = None
    version: str | None = Field(default=None, description="edition, revision, commit hash or dataset version")
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    accessed_at: str | None = None

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        return _check_id(v, "evidence id")

    @field_validator("created_at")
    @classmethod
    def _created(cls, v: str) -> str:
        return _timestamp(v, "evidence.created_at")

    @field_validator("accessed_at")
    @classmethod
    def _accessed(cls, v: str | None) -> str | None:
        return None if v is None else _timestamp(v, "evidence.accessed_at")


class Attribution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    version: str = Field(min_length=1)
    name: str | None = None


class Provenance(BaseModel):
    """How this record came to exist (distinct from the evidence it cites)."""

    model_config = ConfigDict(extra="forbid")
    created_at: str | None = None
    activity: str | None = None
    tool: Attribution | None = None
    agent: Attribution | None = None
    inputs: list[str] = Field(default_factory=list, description="ids of input records or evidence")
    source_commit: str | None = Field(default=None, description="upstream commit hash when imported from a repository")
    dataset_version: str | None = None
    normalization: str | None = Field(default=None, description="description of unit/format normalisation applied")

    @field_validator("created_at")
    @classmethod
    def _created(cls, v: str | None) -> str | None:
        return None if v is None else _timestamp(v, "provenance.created_at")


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uri: str = Field(min_length=1, description="URI or path relative to the record file")
    media_type: str = Field(min_length=1)
    role: Literal["raw", "normalized", "derived", "documentation", "model_weights", "waveform_data", "other"] = "other"
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    size_bytes: int | None = Field(default=None, ge=0)
    description: str | None = None


class Quality(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["raw", "normalized", "curated", "reviewed", "rejected"] = "normalized"
    completeness: float | None = Field(default=None, ge=0, le=1, description="fraction of expected fields present")
    flags: list[str] = Field(default_factory=list, description="e.g. outlier_suspected, missing_temperature")
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------------------
# Property building blocks
# --------------------------------------------------------------------------------------------------
class RangeSI(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min: float | None = None
    max: float | None = None
    unit: str

    @model_validator(mode="after")
    def _ordered(self) -> "RangeSI":
        units.get(self.unit)
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("range min exceeds max")
        return self


class ConditionedValue(BaseModel):
    """A datasheet-style value valid at stated conditions, with its own source reference."""

    model_config = ConfigDict(extra="forbid")
    value: Quantity
    temperature: Quantity | None = None
    frequency: Quantity | None = None
    field: Quantity | None = Field(default=None, description="magnetic field strength H")
    flux_density_peak: Quantity | None = None
    source_ref: str | None = Field(default=None, description="evidence id")
    note: str | None = None

    @model_validator(mode="after")
    def _si_conditions(self) -> "ConditionedValue":
        if self.temperature is not None:
            _si(self.temperature, "K", "temperature")
        if self.frequency is not None:
            _si(self.frequency, "Hz", "frequency")
        if self.field is not None:
            _si(self.field, "A/m", "field")
        if self.flux_density_peak is not None:
            _si(self.flux_density_peak, "T", "flux_density_peak")
        return self


class SteinmetzParameterSet(BaseModel):
    """Classical Steinmetz parameters in SI form: ``Pv [W/m^3] = k * f[Hz]^alpha * Bpk[T]^beta``.

    Parameters fitted in other unit systems (kHz, mT, kW/m^3) must be converted on import; the
    ``original`` field of ``k`` records the pre-conversion value for reproducibility.
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    k: Quantity = Field(description="W/m^3 with f in Hz and B in T")
    alpha: float = Field(gt=0)
    beta: float = Field(gt=0)
    temperature: Quantity | None = Field(default=None, description="temperature at which the set was fitted")
    frequency_range: RangeSI | None = None
    flux_density_range: RangeSI | None = None
    temperature_range: RangeSI | None = None
    excitation: Literal["sinusoidal"] = "sinusoidal"
    basis: Literal["datasheet_fit", "measurement_fit", "publication", "manufacturer_supplied", "synthetic"]
    source_ref: str | None = None
    fit_metrics: dict[str, float] = Field(default_factory=dict, description="e.g. max_rel_error, n_points")
    note: str | None = None

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        return _check_id(v, "parameter set id")

    @model_validator(mode="after")
    def _si_units(self) -> "SteinmetzParameterSet":
        _si(self.k, "W/m^3", "k")
        if self.temperature is not None:
            _si(self.temperature, "K", "temperature")
        return self


class LossReferencePoint(BaseModel):
    """A single (f, Bpk, T) -> Pv point, typically a datasheet typical value or a measurement."""

    model_config = ConfigDict(extra="forbid")
    frequency: Quantity
    flux_density_peak: Quantity
    temperature: Quantity | None = None
    loss_density: Quantity
    dc_bias_field: Quantity | None = None
    excitation: Literal["sinusoidal", "triangular", "trapezoidal", "square", "piecewise_linear", "sampled"] = "sinusoidal"
    duty: float | None = Field(default=None, gt=0, lt=1)
    source_ref: str | None = None
    basis: Literal["datasheet_typical", "measurement", "publication", "synthetic"] = "datasheet_typical"
    note: str | None = None

    @model_validator(mode="after")
    def _si_units(self) -> "LossReferencePoint":
        _positive(_si(self.frequency, "Hz", "frequency"), "frequency")
        _nonnegative(_si(self.flux_density_peak, "T", "flux_density_peak"), "flux_density_peak")
        _nonnegative(_si(self.loss_density, "W/m^3", "loss_density"), "loss_density")
        if self.temperature is not None:
            _positive(_si(self.temperature, "K", "temperature"), "temperature")
        if self.dc_bias_field is not None:
            _si(self.dc_bias_field, "A/m", "dc_bias_field")
        return self


class TemperaturePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: Quantity
    value: Quantity
    source_ref: str | None = None

    @model_validator(mode="after")
    def _si_units(self) -> "TemperaturePoint":
        _si(self.temperature, "K", "temperature")
        return self


# --------------------------------------------------------------------------------------------------
# Kind-specific property models
# --------------------------------------------------------------------------------------------------
class ManufacturerProperties(BaseModel):
    model_config = ConfigDict(extra="forbid")
    legal_name: str | None = None
    brand_names: list[str] = Field(default_factory=list)
    country: str | None = None
    website: str | None = None
    description: str | None = None


class MaterialProperties(BaseModel):
    """Datasheet-level parameters of a soft magnetic material."""

    model_config = ConfigDict(extra="forbid")

    manufacturer_ref: str | None = None
    grade: str | None = Field(default=None, description="manufacturer grade designation, e.g. N87")
    material_class: Literal["ferrite_mnzn", "ferrite_nizn", "powder_iron", "powder_alloy", "nanocrystalline",
                            "amorphous", "silicon_steel", "other"] | None = None
    initial_permeability: ConditionedValue | None = None
    amplitude_permeability: list[ConditionedValue] = Field(default_factory=list)
    saturation_flux_density: list[ConditionedValue] = Field(default_factory=list,
                                                            description="Bsat at stated H and T")
    remanence: list[ConditionedValue] = Field(default_factory=list)
    coercivity: list[ConditionedValue] = Field(default_factory=list)
    curie_temperature: ConditionedValue | None = None
    density: ConditionedValue | None = None
    resistivity: ConditionedValue | None = None
    recommended_frequency_range: RangeSI | None = None
    steinmetz_parameter_sets: list[SteinmetzParameterSet] = Field(default_factory=list)
    loss_reference_points: list[LossReferencePoint] = Field(default_factory=list)
    permeability_vs_temperature: list[TemperaturePoint] = Field(default_factory=list)
    loss_density_vs_temperature: list[LossReferencePoint] = Field(default_factory=list)
    dc_bias_points: list[ConditionedValue] = Field(default_factory=list,
                                                   description="permeability (or B) at stated DC field H")
    relative_permeability: Quantity | None = Field(default=None, description="v0.1 compatibility field")
    description: str | None = None

    @model_validator(mode="after")
    def _si_units(self) -> "MaterialProperties":
        if self.initial_permeability is not None:
            _positive(_si(self.initial_permeability.value, "1", "initial_permeability"), "initial_permeability")
        for item in self.amplitude_permeability:
            _positive(_si(item.value, "1", "amplitude_permeability"), "amplitude_permeability")
        for item in self.saturation_flux_density + self.remanence:
            _positive(_si(item.value, "T", "saturation_flux_density/remanence"), "saturation_flux_density/remanence")
        for item in self.coercivity:
            _nonnegative(_si(item.value, "A/m", "coercivity"), "coercivity")
        if self.curie_temperature is not None:
            _positive(_si(self.curie_temperature.value, "K", "curie_temperature"), "curie_temperature")
        if self.density is not None:
            _positive(_si(self.density.value, "kg/m^3", "density"), "density")
        if self.resistivity is not None:
            _positive(_si(self.resistivity.value, "Ohm*m", "resistivity"), "resistivity")
        if self.recommended_frequency_range is not None and self.recommended_frequency_range.unit != "Hz":
            raise ValueError("recommended_frequency_range must use Hz")
        if self.relative_permeability is not None:
            _positive(_si(self.relative_permeability, "1", "relative_permeability"), "relative_permeability")
        return self


class CoreMaterialProperties(BaseModel):
    """v0.1 ``core_material`` kind: a material as used in a core (kept for compatibility)."""

    model_config = ConfigDict(extra="forbid")
    material_ref: str | None = None
    relative_permeability: Quantity | None = None
    description: str | None = None

    @model_validator(mode="after")
    def _si_units(self) -> "CoreMaterialProperties":
        if self.relative_permeability is not None:
            _positive(_si(self.relative_permeability, "1", "relative_permeability"), "relative_permeability")
        return self


class CoreFamilyProperties(BaseModel):
    model_config = ConfigDict(extra="forbid")
    manufacturer_ref: str | None = None
    shape: Literal["E", "EE", "EI", "ETD", "EFD", "ER", "EQ", "PQ", "PM", "RM", "P", "U", "UU", "UI", "toroid",
                   "planar_E", "block", "other"] | None = None
    standard: str | None = Field(default=None, description="e.g. IEC 62317")
    description: str | None = None


class CoreDimensions(BaseModel):
    """Named outline dimensions in metres (datasheet letter designations)."""

    model_config = ConfigDict(extra="forbid")
    values: dict[str, Quantity] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _si_units(self) -> "CoreDimensions":
        for key, q in self.values.items():
            _positive(_si(q, "m", f"dimension {key}"), f"dimension {key}")
        return self


class CoreGeometryProperties(BaseModel):
    """Effective magnetic parameters of a core (set or single part) per IEC 60205 conventions."""

    model_config = ConfigDict(extra="forbid")

    family_ref: str | None = None
    manufacturer_ref: str | None = None
    part_number: str | None = None
    material_refs: list[str] = Field(default_factory=list, description="materials this core is offered in")
    set_description: str | None = Field(default=None, description="e.g. 'core set of 2 halves', 'single E'")
    effective_area: Quantity | None = Field(default=None, description="Ae")
    effective_length: Quantity | None = Field(default=None, description="le")
    effective_volume: Quantity | None = Field(default=None, description="Ve")
    minimum_area: Quantity | None = Field(default=None, description="Amin")
    core_constant_c1: Quantity | None = Field(default=None, description="sum(l/A), 1/m")
    window_area: Quantity | None = Field(default=None, description="core window area (per datasheet definition)")
    bobbin_window_area: Quantity | None = Field(default=None, description="usable winding area with bobbin")
    mean_turn_length: Quantity | None = None
    mass: Quantity | None = Field(default=None, description="mass of the core set")
    surface_area: Quantity | None = None
    dimensions: CoreDimensions | None = None
    air_gap_options: list[Quantity] = Field(default_factory=list)
    description: str | None = None

    @model_validator(mode="after")
    def _si_units(self) -> "CoreGeometryProperties":
        pairs = (("effective_area", "m^2"), ("effective_length", "m"), ("effective_volume", "m^3"),
                 ("minimum_area", "m^2"), ("window_area", "m^2"), ("bobbin_window_area", "m^2"),
                 ("mean_turn_length", "m"), ("mass", "kg"), ("surface_area", "m^2"))
        for name, unit in pairs:
            q = getattr(self, name)
            if q is not None:
                _positive(_si(q, unit, name), name)
        if self.core_constant_c1 is not None:
            _positive(_si(self.core_constant_c1, "1/m", "core_constant_c1"), "core_constant_c1")
        for g in self.air_gap_options:
            _nonnegative(_si(g, "m", "air_gap_options"), "air_gap_options")
        return self


class WindingProperties(BaseModel):
    """Winding geometry (v0.1 ``winding`` plus conductor detail)."""

    model_config = ConfigDict(extra="forbid")

    turns: Quantity | None = None
    conductor_area: Quantity | None = None
    conductor_type: Literal["round", "foil", "litz", "rectangular", "pcb_trace", "other"] | None = None
    strand_diameter: Quantity | None = None
    strand_count: int | None = Field(default=None, ge=1)
    conductor_diameter: Quantity | None = None
    foil_thickness: Quantity | None = None
    foil_width: Quantity | None = None
    layers: int | None = Field(default=None, ge=1)
    parallel_conductors: int | None = Field(default=None, ge=1)
    material: Literal["copper", "aluminium", "other"] | None = None
    mean_turn_length: Quantity | None = None
    insulation_thickness: Quantity | None = None
    description: str | None = None

    @model_validator(mode="after")
    def _si_units(self) -> "WindingProperties":
        if self.turns is not None:
            _si(self.turns, "1", "turns")
            if abs(self.turns.value - round(self.turns.value)) > 0 or self.turns.value <= 0:
                raise ValueError("turns must be a positive integer count")
        for name, unit in (("conductor_area", "m^2"), ("strand_diameter", "m"), ("conductor_diameter", "m"),
                           ("foil_thickness", "m"), ("foil_width", "m"), ("mean_turn_length", "m")):
            q = getattr(self, name)
            if q is not None:
                _positive(_si(q, unit, name), name)
        if self.insulation_thickness is not None:
            _nonnegative(_si(self.insulation_thickness, "m", "insulation_thickness"), "insulation_thickness")
        return self


class ComponentProperties(BaseModel):
    """Inductor / transformer / generic magnetic component."""

    model_config = ConfigDict(extra="forbid")

    component_type: Literal["inductor", "transformer", "coupled_inductor", "other"] | None = None
    core_ref: str | None = None
    material_ref: str | None = None
    winding_refs: list[str] = Field(default_factory=list)
    inductance: Quantity | None = None
    leakage_inductance: Quantity | None = None
    magnetizing_inductance: Quantity | None = None
    turns_ratio: Quantity | None = None
    air_gap: Quantity | None = None
    rated_current_rms: Quantity | None = None
    rated_voltage_rms: Quantity | None = None
    rated_power: Quantity | None = None
    description: str | None = None

    @model_validator(mode="after")
    def _si_units(self) -> "ComponentProperties":
        for name, unit in (("inductance", "H"), ("leakage_inductance", "H"), ("magnetizing_inductance", "H"),
                           ("turns_ratio", "1"), ("rated_current_rms", "A"), ("rated_voltage_rms", "V"), ("rated_power", "W")):
            q = getattr(self, name)
            if q is not None:
                _positive(_si(q, unit, name), name)
        if self.air_gap is not None:
            _nonnegative(_si(self.air_gap, "m", "air_gap"), "air_gap")
        return self


class OperatingConditionProperties(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frequency: Quantity | None = None
    flux_density_peak: Quantity | None = None
    flux_density_dc: Quantity | None = None
    field_dc: Quantity | None = Field(default=None, description="DC bias field H_dc")
    temperature: Quantity | None = None
    ambient_temperature: Quantity | None = None
    current_rms: Quantity | None = None
    current_dc: Quantity | None = None
    voltage_rms: Quantity | None = None
    duty: float | None = Field(default=None, gt=0, lt=1)
    excitation_ref: str | None = Field(default=None, description="excitation_waveform record id")
    description: str | None = None

    @model_validator(mode="after")
    def _si_units(self) -> "OperatingConditionProperties":
        for name, unit in (("frequency", "Hz"), ("temperature", "K"), ("ambient_temperature", "K")):
            q = getattr(self, name)
            if q is not None:
                _positive(_si(q, unit, name), name)
        for name, unit in (("flux_density_peak", "T"), ("current_rms", "A"), ("voltage_rms", "V")):
            q = getattr(self, name)
            if q is not None:
                _nonnegative(_si(q, unit, name), name)
        for name, unit in (("flux_density_dc", "T"), ("field_dc", "A/m"), ("current_dc", "A")):
            q = getattr(self, name)
            if q is not None:
                _si(q, unit, name)  # signed: DC bias may be negative relative to the reference direction
        return self


class ExcitationWaveformProperties(BaseModel):
    """Parametric or sampled time-domain excitation description.

    Sign convention: positive ``B``/``H``/voltage follow the right-hand rule around the winding
    reference direction stated in ``reference_direction``. Time starts at ``t=0`` at the beginning of
    one steady-state period unless ``timebase`` says otherwise.
    """

    model_config = ConfigDict(extra="forbid")

    waveform_type: Literal["sinusoidal", "triangular", "trapezoidal", "square", "piecewise_linear", "sampled", "dc"]
    controlled_quantity: Literal["flux_density", "field", "voltage", "current"] = "flux_density"
    frequency: Quantity | None = None
    amplitude_peak: Quantity | None = None
    dc_offset: Quantity | None = None
    duty: float | None = Field(default=None, gt=0, lt=1)
    rise_fraction: float | None = Field(default=None, ge=0, le=1, description="trapezoid rise time / period")
    breakpoints: list[list[float]] | None = Field(
        default=None, description="piecewise-linear [t_fraction_of_period, value_fraction_of_peak] pairs")
    timebase: Literal["periodic_from_zero", "absolute"] = "periodic_from_zero"
    reference_direction: str | None = None
    dataset_ref: str | None = Field(default=None, description="waveform_dataset record holding the samples")
    description: str | None = None

    @model_validator(mode="after")
    def _si_units(self) -> "ExcitationWaveformProperties":
        if self.frequency is not None:
            _si(self.frequency, "Hz", "frequency")
        expected = {"flux_density": "T", "field": "A/m", "voltage": "V", "current": "A"}[self.controlled_quantity]
        for name in ("amplitude_peak", "dc_offset"):
            q = getattr(self, name)
            if q is not None:
                _si(q, expected, name)
        if self.breakpoints is not None:
            for pair in self.breakpoints:
                if len(pair) != 2 or not (0 <= pair[0] <= 1):
                    raise ValueError("breakpoints must be [t_fraction, value_fraction] with 0<=t_fraction<=1")
        return self


class MeasurementProperties(BaseModel):
    """v0.1 ``measurement`` kind (artifact-based) with optional instrument detail."""

    model_config = ConfigDict(extra="forbid")

    component_ref: str | None = None
    material_ref: str | None = None
    operating_condition_ref: str | None = None
    artifact: str | None = None
    tool: Attribution | None = None
    instrument: str | None = None
    method: str | None = Field(default=None, description="e.g. two-winding wattmeter, calorimetric")
    specimen: str | None = Field(default=None, description="specimen identity/geometry (toroid dimensions ...)")
    acquired_at: str | None = None
    description: str | None = None


class CoreLossMeasurementProperties(BaseModel):
    """Table of scalar core-loss points measured on one specimen."""

    model_config = ConfigDict(extra="forbid")

    material_ref: str
    specimen_core_ref: str | None = None
    specimen_description: str | None = None
    method: str | None = None
    instrument: str | None = None
    points: list[LossReferencePoint] = Field(min_length=1)
    dataset_ref: str | None = None
    description: str | None = None


class BHLoopProperties(BaseModel):
    """Summary of a B-H hysteresis loop; the sampled loop itself lives in a waveform dataset."""

    model_config = ConfigDict(extra="forbid")

    material_ref: str
    frequency: Quantity | None = None
    temperature: Quantity | None = None
    field_peak: Quantity | None = None
    flux_density_peak: Quantity | None = None
    remanence: Quantity | None = None
    coercivity: Quantity | None = None
    loop_energy_density: Quantity | None = Field(default=None, description="integral H dB over one loop, J/m^3")
    loop_type: Literal["major", "minor", "initial_curve", "anhysteretic"] = "major"
    dataset_ref: str | None = None
    sign_convention: str = Field(default="B and H positive along the same winding reference direction")
    description: str | None = None

    @model_validator(mode="after")
    def _si_units(self) -> "BHLoopProperties":
        for name, unit in (("frequency", "Hz"), ("temperature", "K"), ("field_peak", "A/m"),
                           ("flux_density_peak", "T"), ("remanence", "T"), ("coercivity", "A/m"),
                           ("loop_energy_density", "J/m^3")):
            q = getattr(self, name)
            if q is not None:
                _si(q, unit, name)
        return self


class ArrayLayout(BaseModel):
    """Description of one array inside an HDF5 waveform file."""

    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, description="HDF5 dataset path, e.g. /sequences/flux_density")
    quantity: str = Field(min_length=1, description="physical meaning, e.g. flux_density, field, time, voltage")
    unit: str
    original_unit: str | None = None
    dtype: str = Field(default="float64")
    shape: list[int] = Field(default_factory=list, description="[n_sequences, n_samples] or [n_samples]")
    axis_semantics: list[str] = Field(default_factory=list, description="meaning of each axis")
    missing_value: str | None = Field(default=None, description="sentinel description, e.g. 'NaN'")

    @field_validator("unit")
    @classmethod
    def _known(cls, v: str) -> str:
        units.get(v)
        return v


class WaveformDatasetProperties(BaseModel):
    """Metadata for an HDF5 waveform container (large numerical arrays never go into JSON)."""

    model_config = ConfigDict(extra="forbid")

    material_ref: str | None = None
    source_dataset_ref: str | None = Field(default=None, description="dataset record of the upstream source")
    format: Literal["hdf5"] = "hdf5"
    layout_version: str = Field(default="aipe.magnetics.waveforms/0.2")
    sampling_rate: Quantity | None = Field(default=None, description="Hz; omit if per-sequence time arrays exist")
    samples_per_sequence: int | None = Field(default=None, ge=2)
    timebase: Literal["uniform_per_period", "explicit_time_array", "uniform_absolute"] = "uniform_per_period"
    sign_convention: str = "B, H, v, i positive along the winding reference direction (right-hand rule)"
    coordinate_definition: str = "scalar quantities along the magnetic path; no vector components"
    arrays: list[ArrayLayout] = Field(min_length=1)
    conditions: dict[str, str] = Field(default_factory=dict,
                                       description="scalar per-sequence condition arrays: name -> HDF5 path")
    n_sequences: int | None = Field(default=None, ge=0)
    description: str | None = None

    @model_validator(mode="after")
    def _si_units(self) -> "WaveformDatasetProperties":
        if self.sampling_rate is not None:
            _si(self.sampling_rate, "Hz", "sampling_rate")
        return self


class DatasetProperties(BaseModel):
    """Collection-level description of an external or local dataset (source, version, licence)."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1, description="adapter/source identifier, e.g. princeton_magnet")
    title: str = Field(min_length=1)
    publisher: str | None = None
    url: str | None = None
    version: str | None = Field(default=None, description="release tag or dataset version")
    source_commit: str | None = None
    retrieved_at: str | None = None
    license: str = Field(default="NOASSERTION")
    redistribution_permitted: bool | None = Field(default=None, description="None = unknown / not assessed")
    citation: str | None = None
    materials: list[str] = Field(default_factory=list)
    checksums: dict[str, str] = Field(default_factory=dict, description="relative path -> sha256 of original bytes")
    record_count: int | None = Field(default=None, ge=0)
    description: str | None = None


class LossModelProperties(BaseModel):
    """v0.1 ``loss_model`` kind kept for compatibility."""

    model_config = ConfigDict(extra="forbid")
    method: str
    model_artifact: str
    operating_condition_refs: list[str] = Field(default_factory=list)
    description: str | None = None


class ModelMetadataProperties(BaseModel):
    """Identity, architecture, parameters and provenance of a model (empirical, physics, ODE or data-driven)."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1, description="e.g. aipe.magnetics.model.igse")
    family: Literal["empirical", "physics", "dynamic", "data_driven"]
    architecture: str | None = Field(default=None, description="e.g. steinmetz, igse, jiles_atherton, mlp, lstm")
    implementation_version: str | None = None
    method_version: str | None = None
    material_refs: list[str] = Field(default_factory=list)
    parameter_set_ref: str | None = None
    weights_artifact: str | None = Field(default=None, description="artifact uri of trained parameters")
    inference_config: dict[str, Any] = Field(default_factory=dict)
    training_data_refs: list[str] = Field(default_factory=list)
    validity: dict[str, Any] = Field(default_factory=dict, description="ValidityDomain JSON")
    verification_refs: list[str] = Field(default_factory=list, description="evidence record ids")
    description: str | None = None


PROPERTY_MODELS: dict[str, type[BaseModel]] = {
    "manufacturer": ManufacturerProperties,
    "magnetic_material": MaterialProperties,
    "core_material": CoreMaterialProperties,
    "core_family": CoreFamilyProperties,
    "core_geometry": CoreGeometryProperties,
    "winding": WindingProperties,
    "winding_geometry": WindingProperties,
    "inductor": ComponentProperties,
    "transformer": ComponentProperties,
    "magnetic_component": ComponentProperties,
    "operating_condition": OperatingConditionProperties,
    "excitation_waveform": ExcitationWaveformProperties,
    "measurement": MeasurementProperties,
    "core_loss_measurement": CoreLossMeasurementProperties,
    "bh_loop": BHLoopProperties,
    "waveform_dataset": WaveformDatasetProperties,
    "dataset": DatasetProperties,
    "loss_model": LossModelProperties,
    "model_metadata": ModelMetadataProperties,
}

REFERENCE_KINDS: dict[str, set[str]] = {
    "manufacturer_ref": {"manufacturer"},
    "material_ref": {"magnetic_material", "core_material"},
    "material_refs": {"magnetic_material", "core_material"},
    "family_ref": {"core_family"},
    "core_ref": {"core_geometry"},
    "specimen_core_ref": {"core_geometry"},
    "winding_refs": {"winding", "winding_geometry"},
    "component_ref": {"inductor", "transformer", "magnetic_component"},
    "operating_condition_ref": {"operating_condition"},
    "operating_condition_refs": {"operating_condition"},
    "excitation_ref": {"excitation_waveform"},
    "dataset_ref": {"waveform_dataset", "dataset"},
    "source_dataset_ref": {"dataset"},
    "training_data_refs": {"waveform_dataset", "core_loss_measurement", "dataset"},
}


# --------------------------------------------------------------------------------------------------
# The record envelope
# --------------------------------------------------------------------------------------------------
class MagneticRecord(BaseModel):
    """A v0.2 record: envelope + kind-specific validated properties."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.2.0"] = SCHEMA_VERSION
    id: str
    kind: RecordKind
    name: str = Field(min_length=1)
    synthetic: bool
    description: str | None = None
    license: str = Field(default="NOASSERTION", min_length=1, description="record-level SPDX or NOASSERTION")
    properties: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)
    provenance: Provenance | None = None
    quality: Quality | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    dataset_ref: str | None = None
    tags: list[str] = Field(default_factory=list)
    extensions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        return _check_id(v)

    @field_validator("extensions")
    @classmethod
    def _namespaced(cls, v: dict[str, Any]) -> dict[str, Any]:
        for key in v:
            if not re.match(r"^[a-z][a-z0-9-]*[.][a-z][a-z0-9_.-]*$", key):
                raise ValueError(f"extension key {key!r} must be namespaced like 'aipe.import'")
        return v

    @model_validator(mode="after")
    def _semantics(self) -> "MagneticRecord":
        model = PROPERTY_MODELS[self.kind]
        model.model_validate(self.properties)  # raises on unknown fields / wrong units
        if len({e.id for e in self.evidence}) != len(self.evidence):
            raise ValueError(f"{self.id}: duplicate evidence id")
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError(f"{self.id}: duplicate evidence_refs")
        by_id = {e.id: e for e in self.evidence}
        for ref in self.evidence_refs:
            if ref not in by_id:
                raise ValueError(f"{self.id}: unknown evidence {ref}")
            if by_id[ref].review_status == "rejected":
                raise ValueError(f"{self.id}: rejected evidence {ref}")
        selected = [by_id[r] for r in self.evidence_refs]
        if not self.synthetic and all(e.kind == "assumption" for e in selected):
            raise ValueError(f"{self.id}: real records require non-assumption source evidence")
        if self.kind in ("measurement", "core_loss_measurement", "bh_loop") and not self.synthetic and not any(
                e.kind == "measurement" for e in selected):
            raise ValueError(f"{self.id}: measurement-type records require measurement evidence")
        return self

    # -- typed access -----------------------------------------------------------------------------
    def typed_properties(self) -> BaseModel:
        return PROPERTY_MODELS[self.kind].model_validate(self.properties)

    def references(self) -> list[tuple[str, str]]:
        """``(field, target_id)`` pairs for every *_ref/*_refs field in ``properties``."""
        out: list[tuple[str, str]] = []
        for key, value in self.properties.items():
            if key.endswith("_ref") and isinstance(value, str):
                out.append((key, value))
            elif key.endswith("_refs") and isinstance(value, list):
                out.extend((key, v) for v in value if isinstance(v, str))
        return out

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True, exclude_defaults=False)


def validate_catalogue(records: list[MagneticRecord]) -> list[str]:
    """Cross-record checks: unique ids, resolvable references, reference kinds."""
    errors: list[str] = []
    by_id: dict[str, MagneticRecord] = {}
    for record in records:
        if record.id in by_id:
            errors.append(f"Duplicate record ID: {record.id}")
        by_id[record.id] = record
    for record in records:
        for field, target in record.references():
            if target not in by_id:
                errors.append(f"{record.id}: unknown record reference {target} ({field})")
                continue
            allowed = REFERENCE_KINDS.get(field)
            if allowed is not None and by_id[target].kind not in allowed:
                errors.append(f"{record.id}: {field} points to wrong record kind {by_id[target].kind}")
        if record.dataset_ref is not None:
            if record.dataset_ref not in by_id:
                errors.append(f"{record.id}: unknown dataset_ref {record.dataset_ref}")
            elif by_id[record.dataset_ref].kind != "dataset":
                errors.append(f"{record.id}: dataset_ref must point to a dataset record")
    return errors
