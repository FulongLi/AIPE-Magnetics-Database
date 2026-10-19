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
