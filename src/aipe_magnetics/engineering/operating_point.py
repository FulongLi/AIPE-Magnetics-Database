"""Operating-point validation against material limits and model validity domains.

Checks are reported individually (pass / fail / not_evaluated with reason); nothing is clipped.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aipe_magnetics.api.types import FluxLimitSpec
from aipe_magnetics.common.errors import AIPEMagneticsError
from aipe_magnetics.data.records import MaterialProperties
from aipe_magnetics.engineering.saturation import SaturationScreening, screen_saturation

CURIE_MARGIN_K = 20.0


class OperatingPointCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    status: str = Field(description="pass | fail | warning | not_evaluated")
    message: str
    severity: str = "warning"


class OperatingPointValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checks: list[OperatingPointCheck] = Field(default_factory=list)
    saturation: SaturationScreening | None = None

    @property
    def passed(self) -> bool:
        return all(c.status != "fail" for c in self.checks)

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


def validate_operating_point(*, material: MaterialProperties, frequency: float, flux_density_peak: float | None,
                             temperature: float, flux_limit: FluxLimitSpec | None = None,
                             parameter_sets: list | None = None) -> OperatingPointValidation:
    """SI inputs: Hz, T, K. Returns individual checks; callers map them into result warnings."""
    checks: list[OperatingPointCheck] = []
    screening = None

    # temperature versus Curie point
    if material.curie_temperature is not None:
        tc = material.curie_temperature.value.value
        if temperature >= tc:
            checks.append(OperatingPointCheck(id="curie_temperature", status="fail", severity="error",
                                              message=f"operating temperature {temperature - 273.15:.0f} degC is at or above the Curie "
                                                      f"temperature {tc - 273.15:.0f} degC"))
        elif temperature >= tc - CURIE_MARGIN_K:
            checks.append(OperatingPointCheck(id="curie_temperature", status="warning",
                                              message=f"operating temperature within {CURIE_MARGIN_K:.0f} K of the Curie temperature "
                                                      f"({tc - 273.15:.0f} degC)"))
        else:
            checks.append(OperatingPointCheck(id="curie_temperature", status="pass", severity="info",
                                              message=f"{tc - temperature:.0f} K below the Curie temperature"))
    else:
        checks.append(OperatingPointCheck(id="curie_temperature", status="not_evaluated", severity="info",
                                          message="material record has no Curie temperature"))

    # frequency versus recommended range
    rng = material.recommended_frequency_range
    if rng is not None:
        lo, hi = rng.min, rng.max
        inside = (lo is None or frequency >= lo) and (hi is None or frequency <= hi)
        checks.append(OperatingPointCheck(
            id="frequency_range", status="pass" if inside else "warning", severity="info" if inside else "warning",
            message=(f"{frequency / 1e3:.0f} kHz is {'inside' if inside else 'outside'} the recommended range "
                     f"{'' if lo is None else f'{lo / 1e3:.0f}'}..{'' if hi is None else f'{hi / 1e3:.0f}'} kHz")))
    else:
        checks.append(OperatingPointCheck(id="frequency_range", status="not_evaluated", severity="info",
                                          message="material record has no recommended frequency range"))

    # saturation
    if flux_density_peak is not None:
        try:
            screening = screen_saturation(flux_density_peak=flux_density_peak, temperature=temperature, material=material,
                                          policy=flux_limit)
            if screening.saturated:
                checks.append(OperatingPointCheck(id="saturation", status="fail", severity="error",
                                                  message=f"B_pk {flux_density_peak:.3f} T >= Bsat {screening.saturation_flux_density:.3f} T"))
            elif not screening.within_limit:
                checks.append(OperatingPointCheck(id="saturation", status="warning",
                                                  message=f"B_pk {flux_density_peak:.3f} T exceeds the flux limit {screening.flux_limit:.3f} T "
                                                          f"(margin to Bsat {screening.margin_to_saturation:.1%})"))
            else:
                checks.append(OperatingPointCheck(id="saturation", status="pass", severity="info",
                                                  message=f"B_pk {flux_density_peak:.3f} T within limit {screening.flux_limit:.3f} T "
                                                          f"(margin to Bsat {screening.margin_to_saturation:.1%})"))
        except AIPEMagneticsError as error:
            checks.append(OperatingPointCheck(id="saturation", status="not_evaluated", severity="warning",
                                              message=f"saturation not evaluated: {error.message}"))
    else:
        checks.append(OperatingPointCheck(id="saturation", status="not_evaluated", severity="info",
                                          message="no peak flux density supplied"))

    # loss-parameter coverage
    sets = parameter_sets if parameter_sets is not None else list(material.steinmetz_parameter_sets)
    covering = []
    for s in sets:
        ok = True
        if s.temperature_range is not None:
            ok &= (s.temperature_range.min is None or temperature >= s.temperature_range.min) and \
                  (s.temperature_range.max is None or temperature <= s.temperature_range.max)
        elif s.temperature is not None:
            ok &= abs(s.temperature.value - temperature) <= 5.0
        if s.frequency_range is not None:
            ok &= (s.frequency_range.min is None or frequency >= s.frequency_range.min) and \
                  (s.frequency_range.max is None or frequency <= s.frequency_range.max)
        if flux_density_peak is not None and s.flux_density_range is not None:
            ok &= (s.flux_density_range.min is None or flux_density_peak >= s.flux_density_range.min) and \
                  (s.flux_density_range.max is None or flux_density_peak <= s.flux_density_range.max)
        if ok:
            covering.append(s.id)
    if not sets:
        checks.append(OperatingPointCheck(id="loss_parameters", status="not_evaluated", severity="warning",
                                          message="no Steinmetz parameter set recorded for this material"))
    elif covering:
        checks.append(OperatingPointCheck(id="loss_parameters", status="pass", severity="info",
                                          message=f"parameter set(s) covering the operating point: {', '.join(covering)}"))
    else:
        checks.append(OperatingPointCheck(id="loss_parameters", status="warning",
                                          message="no Steinmetz parameter set covers this temperature/frequency/flux density; "
                                                  "loss estimates would be extrapolations and are rejected by default"))
    return OperatingPointValidation(checks=checks, saturation=screening)
