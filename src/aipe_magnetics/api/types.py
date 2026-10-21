"""Typed, validated input structures of the stable engineering API.

Every API function takes one of these specifications (or plain keyword arguments that are
validated into one). Physical values are :class:`~aipe_magnetics.common.quantity.Quantity`
objects or ``{"value": ..., "unit": ...}`` dicts in any registered unit; they are normalised to SI
on validation and the original unit is retained. Bare numbers are rejected.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, field_validator, model_validator

from aipe_magnetics.common.errors import InvalidInputError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.units import units

DomainPolicy = Literal["reject", "warn"]
LossModelName = Literal["auto", "steinmetz", "igse", "bertotti"]


def _q(value: Any, unit: str, name: str, *, positive: bool = False, nonnegative: bool = False,
       optional: bool = False) -> Quantity | None:
    """Parse a quantity-like value, check dimension, normalise to SI, enforce sign."""
    if value is None:
        if optional:
            return None
        raise InvalidInputError(f"{name} is required")
    q = Quantity.parse(value, name=name)
    if not units.compatible(q.unit, unit):
        raise InvalidInputError(f"{name} must have the dimension of {unit!r}, got {q.unit!r}", details={"name": name})
    q = q.to_si()
    if positive and q.value <= 0:
        raise InvalidInputError(f"{name} must be positive, got {q.value} {q.unit}", details={"name": name})
    if nonnegative and q.value < 0:
        raise InvalidInputError(f"{name} must be non-negative, got {q.value} {q.unit}", details={"name": name})
    return q


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class WaveformSpec(_Spec):
    """Periodic flux-density waveform shape for loss estimation (amplitude comes from the operating point)."""

    kind: Literal["sinusoidal", "triangular", "trapezoidal", "piecewise_linear", "sampled"] = "sinusoidal"
    duty: float | None = Field(default=None, gt=0, lt=1, strict=True, description="triangular: fraction of the period with rising flux")
    duty_rise: float | None = Field(default=None, gt=0, le=0.5, strict=True,
                                    description="trapezoidal (symmetric: rise, flat, fall, flat): rising fraction of the period")
    duty_flat: float | None = Field(default=None, ge=0, lt=0.5, strict=True,
                                    description="trapezoidal: flat fraction after the rise; must equal 0.5 - duty_rise (derived when omitted)")
    breakpoints: list[list[float]] | None = Field(default=None,
                                                  description="piecewise_linear: [[t_fraction, B_fraction_of_peak], ...]")
    samples: list[float] | None = Field(default=None,
                                        description="sampled: one period of B(t) in tesla; with endpoint_included=true (default) the samples "
                                                    "span [0, T] and the last must equal the first (closed period), with false they span [0, T)")
    endpoint_included: bool | None = Field(default=None, strict=True,
                                           description="sampled: whether samples[-1] is B(T) (closed period; the default when omitted) or "
                                                       "the samples span [0, T) (false)")
    dc_bias_field: Quantity | None = Field(default=None, description="H_dc (A/m), reported to models that support it")

    @model_validator(mode="after")
    def _consistent(self) -> "WaveformSpec":
        if self.kind == "triangular" and self.duty is None:
            object.__setattr__(self, "duty", 0.5)
        if self.kind == "trapezoidal":
            if self.duty_rise is None:
                raise ValueError("trapezoidal waveform needs duty_rise (duty_flat = 0.5 - duty_rise is derived when omitted)")
            if self.duty_flat is None:
                object.__setattr__(self, "duty_flat", 0.5 - self.duty_rise)
            if abs(2 * (self.duty_rise + self.duty_flat) - 1) > 1e-9:  # type: ignore[operator]
                raise ValueError("trapezoidal: the engine implements the symmetric trapezoid (rise, flat, fall, flat), which requires "
                                 f"2*(duty_rise + duty_flat) == 1; got duty_rise={self.duty_rise}, duty_flat={self.duty_flat} "
                                 "(omit duty_flat to derive it)")
        if self.kind == "piecewise_linear" and not self.breakpoints:
            raise ValueError("piecewise_linear waveform needs breakpoints")
        if self.kind == "sampled" and (not self.samples or len(self.samples) < 8):
            raise ValueError("sampled waveform needs at least 8 samples of one period")
        if self.dc_bias_field is not None:
            object.__setattr__(self, "dc_bias_field", _q(self.dc_bias_field, "A/m", "dc_bias_field"))
        return self


class OperatingPointSpec(_Spec):
    """Operating point for a loss or validation query."""

    frequency: Quantity
    flux_density_peak: Quantity
    temperature: Quantity | None = None
    waveform: WaveformSpec = Field(default_factory=WaveformSpec)

    @field_validator("frequency", mode="before")
    @classmethod
    def _f(cls, v: Any) -> Quantity:
        return _q(v, "Hz", "frequency", positive=True)  # type: ignore[return-value]

    @field_validator("flux_density_peak", mode="before")
    @classmethod
    def _b(cls, v: Any) -> Quantity:
        # zero excitation is not a loss query (the waveform constructors need a positive amplitude): rejected explicitly
        return _q(v, "T", "flux_density_peak", positive=True)  # type: ignore[return-value]

    @field_validator("temperature", mode="before")
    @classmethod
    def _t(cls, v: Any) -> Quantity | None:
        q = _q(v, "K", "temperature", optional=True)
        if q is not None and q.value <= 0:
            raise InvalidInputError("temperature must be above absolute zero")
        return q


class CoreLossRequest(_Spec):
    material: str = Field(min_length=1, description="catalogue id, name or grade (e.g. 'N87')")
    operating_point: OperatingPointSpec
    model: LossModelName = "auto"
    core: str | None = Field(default=None, description="core id to convert loss density into total core loss")
    parameter_set: str | None = Field(default=None, description="force a specific Steinmetz parameter set id")
    model_parameters: dict[str, StrictFloat] | None = Field(
        default=None,
        description="user-supplied SI model coefficients (steinmetz/igse: k, alpha, beta; bertotti: k_h, beta_h, k_c, k_e); "
                    "recorded as a user_input assumption and bypasses the catalogue parameter sets")
    policy: DomainPolicy = "reject"


class HysteresisRequest(_Spec):
    model: Literal["jiles_atherton"] = "jiles_atherton"
    field_peak: Quantity = Field(description="peak applied field H (A/m)")
    frequency: Quantity = Field(default_factory=lambda: Quantity(value=1.0, unit="Hz"),
                                description="excitation frequency; the model is rate independent")
    field_dc: Quantity | None = None
    waveform: Literal["sinusoidal", "triangular"] = "sinusoidal"
    periods: int = Field(default=10, ge=1, le=50, strict=True, description="maximum periods integrated to reach periodic steady state")
    steady_state_tol: float = Field(default=1e-4, gt=0, lt=1, strict=True,
                                    description="relative change of the state at period boundaries accepted as steady")
    parameters: dict[str, StrictFloat] | None = Field(default=None, description="Ms, a, alpha, k, c in SI; default illustrative set")
    rtol: float = Field(default=1e-8, gt=0, lt=1e-2, strict=True, description="solver relative tolerance (default matches the model's own)")
    atol: float = Field(default=1e-3, gt=0, strict=True, description="solver absolute tolerance on M (A/m)")
    samples_per_period: int = Field(default=400, ge=50, le=20000, strict=True)
    policy: DomainPolicy = "reject"

    @model_validator(mode="after")
    def _tolerances(self) -> "HysteresisRequest":
        if self.steady_state_tol < 100 * self.rtol:
            raise ValueError("steady_state_tol must be at least 100 x rtol so that the steady-state test is not dominated by solver noise")
        return self

    @field_validator("field_peak", mode="before")
    @classmethod
    def _h(cls, v: Any) -> Quantity:
        return _q(v, "A/m", "field_peak", positive=True)  # type: ignore[return-value]

    @field_validator("frequency", mode="before")
    @classmethod
    def _f(cls, v: Any) -> Quantity:
        return _q(v, "Hz", "frequency", positive=True)  # type: ignore[return-value]

    @field_validator("field_dc", mode="before")
    @classmethod
    def _hdc(cls, v: Any) -> Quantity | None:
        return _q(v, "A/m", "field_dc", optional=True)


class WindingSpec(_Spec):
    conductor_type: Literal["round", "foil", "litz"] = "round"
    current_density: Quantity | None = Field(default=None, description="target J (A/m^2); default 4 A/mm^2 assumption")
    fill_factor: float = Field(default=0.4, gt=0, le=1, strict=True, description="usable fraction of the window area")
    strand_diameter: Quantity | None = Field(default=None, description="litz strand diameter (m)")
    foil_thickness: Quantity | None = None
    layers: int | None = Field(default=None, ge=1, strict=True, description="number of layers; estimated from turns when omitted")
    conductor_temperature: Quantity | None = Field(default=None, description="winding temperature for resistivity")

    @field_validator("current_density", mode="before")
    @classmethod
    def _j(cls, v: Any) -> Quantity | None:
        return _q(v, "A/m^2", "current_density", positive=True, optional=True)

    @field_validator("strand_diameter", "foil_thickness", mode="before")
    @classmethod
    def _len(cls, v: Any) -> Quantity | None:
        return _q(v, "m", "length", positive=True, optional=True)

    @field_validator("conductor_temperature", mode="before")
    @classmethod
    def _t(cls, v: Any) -> Quantity | None:
        return _q(v, "K", "conductor_temperature", positive=True, optional=True)


class FluxLimitSpec(_Spec):
    """How the allowed peak flux density is chosen."""

    mode: Literal["fraction_of_saturation", "absolute"] = "fraction_of_saturation"
    fraction: float = Field(default=0.7, gt=0, le=1, strict=True)
    absolute: Quantity | None = None

    @field_validator("absolute", mode="before")
    @classmethod
    def _b(cls, v: Any) -> Quantity | None:
        return _q(v, "T", "flux limit", positive=True, optional=True)

    @model_validator(mode="after")
    def _mode(self) -> "FluxLimitSpec":
        if self.mode == "absolute" and self.absolute is None:
            raise ValueError("absolute flux limit requires 'absolute'")
        return self


class InductorDesignRequest(_Spec):
    inductance: Quantity
    current_peak: Quantity
    current_rms: Quantity
    current_ripple_peak_to_peak: Quantity | None = Field(default=None, description="AC ripple for core-loss estimation")
    ripple_duty: float = Field(default=0.5, gt=0, lt=1, strict=True,
                               description="rising fraction of the period of the triangular ripple flux used for core loss (0.5 = symmetric)")
    frequency: Quantity
    material: str
    core: str | None = Field(default=None, description="core id; None = evaluate every catalogue core offered in the material")
    temperature: Quantity = Field(default_factory=lambda: Quantity(value=373.15, unit="K"))
    flux_limit: FluxLimitSpec = Field(default_factory=FluxLimitSpec)
    winding: WindingSpec = Field(default_factory=WindingSpec)
    loss_model: LossModelName = "auto"
    max_candidates: int = Field(default=5, ge=1, le=50, strict=True)
    policy: DomainPolicy = "warn"

    @field_validator("inductance", mode="before")
    @classmethod
    def _l(cls, v: Any) -> Quantity:
        return _q(v, "H", "inductance", positive=True)  # type: ignore[return-value]

    @field_validator("current_peak", "current_rms", mode="before")
    @classmethod
    def _i(cls, v: Any) -> Quantity:
        return _q(v, "A", "current", positive=True)  # type: ignore[return-value]

    @field_validator("current_ripple_peak_to_peak", mode="before")
    @classmethod
    def _ir(cls, v: Any) -> Quantity | None:
        return _q(v, "A", "current_ripple_peak_to_peak", nonnegative=True, optional=True)

    @field_validator("frequency", mode="before")
    @classmethod
    def _f(cls, v: Any) -> Quantity:
        return _q(v, "Hz", "frequency", positive=True)  # type: ignore[return-value]

    @field_validator("temperature", mode="before")
    @classmethod
    def _t(cls, v: Any) -> Quantity:
        return _q(v, "K", "temperature", positive=True)  # type: ignore[return-value]

    @model_validator(mode="after")
    def _currents(self) -> "InductorDesignRequest":
        if self.current_rms.value > self.current_peak.value * (1 + 1e-9):
            raise ValueError("current_rms cannot exceed current_peak")
        if self.current_ripple_peak_to_peak is not None and self.current_ripple_peak_to_peak.value > 2 * self.current_peak.value * (1 + 1e-9):
            raise ValueError("current_ripple_peak_to_peak cannot exceed 2 x current_peak (the AC flux amplitude would exceed the peak flux density)")
        return self


class TransformerWindingSpec(_Spec):
    name: str = Field(min_length=1)
    voltage: Quantity = Field(description="winding voltage amplitude (peak of the square/trapezoidal wave, or RMS for sinusoidal)")
    current_rms: Quantity | None = Field(default=None, description="winding RMS current; derived from power when omitted")
    power_share: float | None = Field(default=None, gt=0, le=1, strict=True,
                                      description="secondaries only: fraction of the throughput power carried by this winding when its "
                                                  "current is derived from power; omitted = equal split between the secondaries")
    winding: WindingSpec = Field(default_factory=WindingSpec)

    @field_validator("voltage", mode="before")
    @classmethod
    def _v(cls, v: Any) -> Quantity:
        return _q(v, "V", "voltage", positive=True)  # type: ignore[return-value]

    @field_validator("current_rms", mode="before")
    @classmethod
    def _i(cls, v: Any) -> Quantity | None:
        return _q(v, "A", "current_rms", positive=True, optional=True)


class TransformerDesignRequest(_Spec):
    power: Quantity
    frequency: Quantity
    primary: TransformerWindingSpec
    secondaries: list[TransformerWindingSpec] = Field(min_length=1)
    voltage_waveform: Literal["square", "sinusoidal", "trapezoidal"] = "square"
    duty: float = Field(default=0.5, gt=0, le=0.5, strict=True,
                        description="square/trapezoidal: fraction of the full period during which +V is applied; the flux rises for "
                                    "duty*T and is flat for (0.5 - duty)*T (0.5 = full square wave without zero-voltage intervals)")
    material: str
    cores: list[str] | None = Field(default=None, description="candidate core ids; None = evaluate every catalogue core offered in the material")
    temperature: Quantity = Field(default_factory=lambda: Quantity(value=373.15, unit="K"))
    flux_limit: FluxLimitSpec = Field(default_factory=FluxLimitSpec)
    loss_density_target: Quantity | None = Field(default=None, description="optional core-loss density ceiling (W/m^3)")
    loss_model: LossModelName = "auto"
    max_candidates: int = Field(default=5, ge=1, le=50, strict=True)
    policy: DomainPolicy = "warn"

    @model_validator(mode="after")
    def _power_shares(self) -> "TransformerDesignRequest":
        if self.primary.power_share is not None:
            raise ValueError("power_share applies to secondaries only (the primary carries the full throughput power)")
        names = [w.name for w in [self.primary, *self.secondaries]]
        if len(set(names)) != len(names):
            raise ValueError("winding names must be unique")
        shares = [w.power_share for w in self.secondaries]
        if any(s is not None for s in shares):
            if any(s is None for s in shares):
                raise ValueError("when power_share is given for one secondary it must be given for every secondary")
            total = sum(shares)  # type: ignore[arg-type]
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"secondary power shares must sum to 1 (got {total:g})")
        return self

    @field_validator("power", mode="before")
    @classmethod
    def _p(cls, v: Any) -> Quantity:
        return _q(v, "W", "power", positive=True)  # type: ignore[return-value]

    @field_validator("frequency", mode="before")
    @classmethod
    def _f(cls, v: Any) -> Quantity:
        return _q(v, "Hz", "frequency", positive=True)  # type: ignore[return-value]

    @field_validator("temperature", mode="before")
    @classmethod
    def _t(cls, v: Any) -> Quantity:
        return _q(v, "K", "temperature", positive=True)  # type: ignore[return-value]

    @field_validator("loss_density_target", mode="before")
    @classmethod
    def _pv(cls, v: Any) -> Quantity | None:
        return _q(v, "W/m^3", "loss_density_target", positive=True, optional=True)


class DesignValidationRequest(_Spec):
    """Validate an existing design (turns, core, material) at an operating point."""

    material: str
    core: str
    turns: int = Field(ge=1, strict=True, description="turns of the winding that sets the flux (primary)")
    frequency: Quantity
    temperature: Quantity = Field(default_factory=lambda: Quantity(value=373.15, unit="K"))
    voltage_peak: Quantity | None = Field(default=None, description="square/trapezoidal voltage amplitude applied to 'turns'")
    voltage_waveform: Literal["square", "sinusoidal", "trapezoidal"] = "square"
    duty: float = Field(default=0.5, gt=0, le=0.5, strict=True,
                        description="square/trapezoidal: fraction of the full period during which +V is applied (0.5 = full square wave)")
    current_peak: Quantity | None = Field(default=None, description="peak magnetising/DC current for H-based saturation check")
    air_gap: Quantity | None = None
    flux_density_peak: Quantity | None = Field(default=None, description="directly specified peak flux density")
    flux_limit: FluxLimitSpec = Field(default_factory=FluxLimitSpec)
    loss_model: LossModelName = "auto"
    policy: DomainPolicy = "warn"

    @field_validator("frequency", mode="before")
    @classmethod
    def _f(cls, v: Any) -> Quantity:
        return _q(v, "Hz", "frequency", positive=True)  # type: ignore[return-value]

    @field_validator("temperature", mode="before")
    @classmethod
    def _t(cls, v: Any) -> Quantity:
        return _q(v, "K", "temperature", positive=True)  # type: ignore[return-value]

    @field_validator("voltage_peak", mode="before")
    @classmethod
    def _v(cls, v: Any) -> Quantity | None:
        return _q(v, "V", "voltage_peak", positive=True, optional=True)

    @field_validator("current_peak", mode="before")
    @classmethod
    def _i(cls, v: Any) -> Quantity | None:
        return _q(v, "A", "current_peak", nonnegative=True, optional=True)

    @field_validator("air_gap", mode="before")
    @classmethod
    def _g(cls, v: Any) -> Quantity | None:
        return _q(v, "m", "air_gap", nonnegative=True, optional=True)

    @field_validator("flux_density_peak", mode="before")
    @classmethod
    def _b(cls, v: Any) -> Quantity | None:
        return _q(v, "T", "flux_density_peak", nonnegative=True, optional=True)

    @model_validator(mode="after")
    def _one_source_of_flux(self) -> "DesignValidationRequest":
        if self.voltage_peak is None and self.flux_density_peak is None:
            raise ValueError("provide voltage_peak (with turns) or flux_density_peak")
        return self


class ReportRequest(_Spec):
    title: str = Field(default="AIPE Magnetics engineering report", min_length=1)
    format: Literal["markdown", "json"] = "markdown"
    output_path: str | None = Field(default=None, description="explicit file path; requires authorization to write")
    allow_write: bool = Field(default=False, description="explicit authorization to create the file")
    overwrite: bool = False
    include_inputs: bool = True


# -------------------------------------------------------------------------------------------------
# waveform-informed transformer design (v0.3)
# -------------------------------------------------------------------------------------------------
def _angle_rad(value: Any, name: str) -> Quantity | None:
    """Angle as a ``rad`` quantity (``deg`` converted); plain ratios are rejected."""
    if value is None:
        return None
    q = Quantity.parse(value, name=name)
    if q.unit not in ("rad", "deg"):
        raise InvalidInputError(f"{name} must be an angle in 'rad' or 'deg', got {q.unit!r}", details={"name": name})
    q = q.to("rad")
    if not (0 < q.value <= 1.5707963267948966 + 1e-12):
        raise InvalidInputError(f"{name} must lie in (0, pi/2] rad for single-phase-shift operation, got {q.value:g} rad")
    return q


def _length_list(values: Any, name: str) -> list[Quantity]:
    if not isinstance(values, (list, tuple)) or not values:
        raise InvalidInputError(f"{name} must be a non-empty list of lengths")
    return [_q(v, "m", name, positive=True) for v in values]  # type: ignore[misc]


class WaveformChannelMap(_Spec):
    """Roles -> channel / column / dataset names of the sampled source (nothing is inferred from names)."""

    primary_current: str = Field(min_length=1, description="transformer primary winding current (A)")
    secondary_current: str | None = Field(default=None, description="transformer secondary winding current (A); derived from the "
                                                                      "primary current and the turns-ratio target when omitted")
    primary_voltage: str | None = Field(default=None, description="transformer primary winding voltage (V); its integral gives the "
                                                                    "flux linkage; an ideal square wave of 'primary_voltage' amplitude "
                                                                    "is assumed when omitted")
    magnetising_current: str | None = Field(default=None, description="optional magnetising current channel (A), reported for comparison")


class WaveformWindowSpec(_Spec):
    """Steady-state window ``[t0, t1]`` of a multi-period record; the last complete period of the window is used."""

    t0: Quantity | None = None
    t1: Quantity | None = None

    @field_validator("t0", "t1", mode="before")
    @classmethod
    def _t(cls, v: Any) -> Quantity | None:
        return _q(v, "s", "window time", nonnegative=True, optional=True)


class SPSWaveformSpec(_Spec):
    """Single-phase-shift DAB parameters used to synthesise the transformer waveforms analytically."""

    power: Quantity
    primary_voltage: Quantity
    secondary_voltage: Quantity
    phase_shift: Quantity | None = Field(default=None, description="rated phase shift (rad or deg); fixes the series inductance")
    series_inductance: Quantity | None = Field(default=None, description="total series inductance referred to the primary (H); fixes "
                                                                           "the phase shift")

    @field_validator("power", mode="before")
    @classmethod
    def _p(cls, v: Any) -> Quantity:
        return _q(v, "W", "power", positive=True)  # type: ignore[return-value]

    @field_validator("primary_voltage", "secondary_voltage", mode="before")
    @classmethod
    def _v(cls, v: Any) -> Quantity:
        return _q(v, "V", "voltage", positive=True)  # type: ignore[return-value]

    @field_validator("phase_shift", mode="before")
    @classmethod
    def _phi(cls, v: Any) -> Quantity | None:
        return _angle_rad(v, "phase_shift")

    @field_validator("series_inductance", mode="before")
    @classmethod
    def _l(cls, v: Any) -> Quantity | None:
        return _q(v, "H", "series_inductance", positive=True, optional=True)

    @model_validator(mode="after")
    def _one_of(self) -> "SPSWaveformSpec":
        if (self.phase_shift is None) == (self.series_inductance is None):
            raise ValueError("give exactly one of phase_shift or series_inductance")
        return self


class WaveformSourceSpec(_Spec):
    """Where the transformer waveforms come from: a sampled file / document or the analytical SPS model."""

    kind: Literal["hdf5", "csv", "json", "sps"]
    path: str | None = Field(default=None, description="hdf5 / csv / json: file path")
    document: dict[str, Any] | None = Field(default=None, description="json: in-memory document instead of a file")
    channels: WaveformChannelMap | None = Field(default=None, description="hdf5 / csv / json: role -> channel name")
    units: dict[str, str] | None = Field(default=None, description="csv: unit per column (required for every mapped column)")
    time_column: str | None = Field(default=None, description="csv: header of the time column")
    time_unit: str | None = Field(default=None, description="csv: unit of the time column")
    time_dataset: str = Field(default="time", description="hdf5: time dataset path")
    channel_group: str = Field(default="channels", description="hdf5: group holding the channel datasets")
    window: WaveformWindowSpec | None = None
    delimiter: str = ","
    sps: SPSWaveformSpec | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "WaveformSourceSpec":
        if self.kind == "sps":
            if self.sps is None:
                raise ValueError("kind 'sps' needs the 'sps' parameters")
            return self
        if self.channels is None:
            raise ValueError(f"kind {self.kind!r} needs 'channels' (role -> channel name)")
        if self.kind == "json":
            if (self.path is None) == (self.document is None):
                raise ValueError("kind 'json' needs exactly one of path or document")
        elif not self.path:
            raise ValueError(f"kind {self.kind!r} needs a file path")
        if self.kind == "csv":
            if not self.time_column or not self.time_unit:
                raise ValueError("kind 'csv' needs time_column and time_unit")
            names = [n for n in (self.channels.primary_current, self.channels.secondary_current, self.channels.primary_voltage,
                                 self.channels.magnetising_current) if n]
            missing = [n for n in names if not (self.units or {}).get(n)]
            if missing:
                raise ValueError(f"kind 'csv' needs a unit for every mapped column; missing: {missing}")
        return self


class ConductorOptionsSpec(_Spec):
    """Winding design space explored for every core / turns combination."""

    conductors: list[Literal["litz", "foil"]] = Field(default_factory=lambda: ["litz"])
    litz_strand_diameters: list[Quantity] = Field(default_factory=lambda: [Quantity(value=0.1e-3, unit="m")])
    foil_thicknesses: list[Quantity] = Field(default_factory=lambda: [Quantity(value=0.1e-3, unit="m"), Quantity(value=0.2e-3, unit="m")])
    current_densities: list[Quantity] = Field(default_factory=lambda: [Quantity(value=4.0e6, unit="A/m^2")],
                                              description="RMS current densities used to size the conductors")
    arrangements: list[str] = Field(default_factory=lambda: ["P-S", "P-S-P"], description="radial winding order, e.g. 'P-S', 'P-S-P', 'S-P-S'")
    max_window_fill: float = Field(default=0.45, gt=0, le=1, strict=True, description="bare copper area / coil-former window area")
    max_current_density: Quantity = Field(default_factory=lambda: Quantity(value=6.0e6, unit="A/m^2"))

    @field_validator("litz_strand_diameters", "foil_thicknesses", mode="before")
    @classmethod
    def _lengths(cls, v: Any, info: Any) -> list[Quantity]:
        return _length_list(v, info.field_name)

    @field_validator("current_densities", mode="before")
    @classmethod
    def _js(cls, v: Any) -> list[Quantity]:
        if not isinstance(v, (list, tuple)) or not v:
            raise InvalidInputError("current_densities must be a non-empty list")
        return [_q(j, "A/m^2", "current_density", positive=True) for j in v]  # type: ignore[misc]

    @field_validator("max_current_density", mode="before")
    @classmethod
    def _jmax(cls, v: Any) -> Quantity:
        return _q(v, "A/m^2", "max_current_density", positive=True)  # type: ignore[return-value]

    @field_validator("conductors")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("conductors must not be empty")
        return v

    @field_validator("arrangements")
    @classmethod
    def _arr(cls, v: list[str]) -> list[str]:
        if not v or any(not isinstance(a, str) or not a.strip() for a in v):
            raise ValueError("arrangements must be non-empty strings such as 'P-S'")
        return v


class ThermalSpec(_Spec):
    """Cooling boundary condition and temperature limits of the thermal estimate."""

    cooling: Literal["natural", "forced"] = "forced"
    ambient: Quantity = Field(default_factory=lambda: Quantity(value=313.15, unit="K"))
    heat_transfer_coefficient: Quantity | None = Field(default=None, description="forced: h in W/m^2/K (default 40 W/m^2/K)")
    radiation: bool = True
    emissivity: float = Field(default=0.9, gt=0, le=1, strict=True)
    max_winding_hotspot: Quantity = Field(default_factory=lambda: Quantity(value=393.15, unit="K"))
    max_core_hotspot: Quantity = Field(default_factory=lambda: Quantity(value=383.15, unit="K"))

    @field_validator("ambient", "max_winding_hotspot", "max_core_hotspot", mode="before")
    @classmethod
    def _t(cls, v: Any) -> Quantity:
        return _q(v, "K", "temperature", positive=True)  # type: ignore[return-value]

    @field_validator("heat_transfer_coefficient", mode="before")
    @classmethod
    def _h(cls, v: Any) -> Quantity | None:
        return _q(v, "W/m^2/K", "heat_transfer_coefficient", positive=True, optional=True)


class IsolationSpec(_Spec):
    """Insulation inputs that affect the layout (the insulation system itself is not designed here)."""

    creepage_margin: Quantity = Field(default_factory=lambda: Quantity(value=0.0, unit="m"),
                                      description="margin at both ends of every layer (reduces the winding height)")
    interwinding_insulation: Quantity = Field(default_factory=lambda: Quantity(value=0.2e-3, unit="m"),
                                              description="insulation thickness between winding sections")
    working_voltage: Quantity | None = None
    insulation_class: Literal["functional", "basic", "supplementary", "reinforced"] | None = None
    standard: str | None = None

    @field_validator("creepage_margin", "interwinding_insulation", mode="before")
    @classmethod
    def _len(cls, v: Any) -> Quantity:
        return _q(v, "m", "insulation length", nonnegative=True)  # type: ignore[return-value]

    @field_validator("working_voltage", mode="before")
    @classmethod
    def _v(cls, v: Any) -> Quantity | None:
        return _q(v, "V", "working_voltage", positive=True, optional=True)


class WaveformInformedTransformerRequest(_Spec):
    """Transformer design driven by the converter's actual (sampled or SPS-synthesised) winding waveforms."""

    frequency: Quantity = Field(description="switching frequency (also the period-detection hint for sampled sources)")
    waveforms: WaveformSourceSpec
    turns_ratio_target: float = Field(gt=0, description="N1 / N2 target, e.g. 2.0 for 800 V : 400 V")
    turns_ratio_tolerance: float = Field(default=0.05, ge=0, lt=1, strict=True)
    primary_voltage: Quantity | None = Field(default=None, description="square-wave amplitude; required when the source has no voltage channel")
    series_inductance_total: Quantity | None = Field(default=None, description="total series inductance referred to the primary (H); taken "
                                                                                 "from the HDF5 metadata (l_ext + leakages) when omitted")
    materials: list[str] | None = Field(default=None, description="candidate materials; None = every catalogue material with sufficient data")
    cores: list[str] | None = Field(default=None, description="candidate core ids; None = every catalogue core offered in the material")
    material_offer_policy: Literal["catalogue", "standard_size_equivalent"] = "catalogue"
    temperature: Quantity = Field(default_factory=lambda: Quantity(value=373.15, unit="K"), description="design (core) temperature")
    flux_limit: FluxLimitSpec = Field(default_factory=FluxLimitSpec)
    loss_density_target: Quantity | None = Field(default=None, description="optional core-loss density ceiling (W/m^3)")
    windings: ConductorOptionsSpec = Field(default_factory=ConductorOptionsSpec)
    thermal: ThermalSpec = Field(default_factory=ThermalSpec)
    isolation: IsolationSpec = Field(default_factory=IsolationSpec)
    turn_steps: int = Field(default=4, ge=0, le=20, strict=True, description="primary turn counts above the minimum that are explored")
    harmonics: int = Field(default=41, ge=1, le=201, strict=True, description="highest current harmonic in the winding-loss sums")
    max_candidates: int = Field(default=10, ge=1, le=100, strict=True)
    reference_core: str | None = Field(default=None, description="core for the material comparison (default: the selected core)")
    policy: DomainPolicy = "warn"

    @field_validator("frequency", mode="before")
    @classmethod
    def _f(cls, v: Any) -> Quantity:
        return _q(v, "Hz", "frequency", positive=True)  # type: ignore[return-value]

    @field_validator("primary_voltage", mode="before")
    @classmethod
    def _v(cls, v: Any) -> Quantity | None:
        return _q(v, "V", "primary_voltage", positive=True, optional=True)

    @field_validator("series_inductance_total", mode="before")
    @classmethod
    def _l(cls, v: Any) -> Quantity | None:
        return _q(v, "H", "series_inductance_total", positive=True, optional=True)

    @field_validator("temperature", mode="before")
    @classmethod
    def _t(cls, v: Any) -> Quantity:
        return _q(v, "K", "temperature", positive=True)  # type: ignore[return-value]

    @field_validator("loss_density_target", mode="before")
    @classmethod
    def _pv(cls, v: Any) -> Quantity | None:
        return _q(v, "W/m^3", "loss_density_target", positive=True, optional=True)

    @model_validator(mode="after")
    def _voltage_available(self) -> "WaveformInformedTransformerRequest":
        src = self.waveforms
        if src.kind != "sps" and src.channels is not None and src.channels.primary_voltage is None and self.primary_voltage is None:
            raise ValueError("without a primary_voltage channel the request needs primary_voltage (square-wave amplitude) to derive the flux")
        return self


class MeasurementRequestSpec(_Spec):
    """Excitation conditions for which magnetic-material measurement requests are generated."""

    materials: list[str] | None = Field(default=None, description="catalogue materials; None = every non-synthetic material")
    frequency: Quantity
    flux_density_peak: Quantity | None = Field(default=None, description="design peak flux density; sets the requested B range")
    temperature: Quantity = Field(default_factory=lambda: Quantity(value=373.15, unit="K"), description="design temperature")
    temperature_range: list[Quantity] | None = Field(default=None, description="[min, max] operating temperature range")
    excitation: Literal["sinusoidal", "triangular", "trapezoidal", "piecewise_linear", "sampled"] = "triangular"
    dc_bias_field_max: Quantity | None = Field(default=None, description="largest DC bias field the application can produce (A/m)")
    harmonics: int = Field(default=5, ge=1, le=51, strict=True, description="highest current harmonic for winding AC-resistance tests")
    include_available: bool = Field(default=True, description="also list categories the catalogue already covers")

    @field_validator("frequency", mode="before")
    @classmethod
    def _f(cls, v: Any) -> Quantity:
        return _q(v, "Hz", "frequency", positive=True)  # type: ignore[return-value]

    @field_validator("flux_density_peak", mode="before")
    @classmethod
    def _b(cls, v: Any) -> Quantity | None:
        return _q(v, "T", "flux_density_peak", positive=True, optional=True)

    @field_validator("temperature", mode="before")
    @classmethod
    def _t(cls, v: Any) -> Quantity:
        return _q(v, "K", "temperature", positive=True)  # type: ignore[return-value]

    @field_validator("temperature_range", mode="before")
    @classmethod
    def _tr(cls, v: Any) -> list[Quantity] | None:
        if v is None:
            return None
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise InvalidInputError("temperature_range must be [min, max]")
        lo, hi = (_q(x, "K", "temperature_range", positive=True) for x in v)
        if lo.value >= hi.value:  # type: ignore[union-attr]
            raise InvalidInputError("temperature_range min must be below max")
        return [lo, hi]  # type: ignore[list-item]

    @field_validator("dc_bias_field_max", mode="before")
    @classmethod
    def _h(cls, v: Any) -> Quantity | None:
        return _q(v, "A/m", "dc_bias_field_max", nonnegative=True, optional=True)
