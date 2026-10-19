"""Magnetic saturation screening.

Compares a peak flux density with the material's saturation flux density at the operating
temperature (datasheet points, linear interpolation between them) and with the user's flux limit
policy. Screening is a pass/fail check with margins; it does not model the B-H curve knee or
local saturation at gaps/corners.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aipe_magnetics.api.types import FluxLimitSpec
from aipe_magnetics.common.errors import InvalidInputError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import Assumption, Limitation, Warning_
from aipe_magnetics.data.records import MaterialProperties
from aipe_magnetics.engineering import formulas as F
from aipe_magnetics.engineering.materials import saturation_flux_density_at


class SaturationScreening(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flux_density_peak: float
    temperature: float
    saturation_flux_density: float
    saturation_status: str = Field(description="looked_up or estimated (interpolated between datasheet points)")
    saturation_source_refs: list[str] = Field(default_factory=list)
    margin_to_saturation: float = Field(description="1 - B_pk / B_sat")
    flux_limit: float
    flux_limit_policy: str
    within_limit: bool
    saturated: bool
    assumptions: list[Assumption] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)
    warnings: list[Warning_] = Field(default_factory=list)

    def quantities(self, prefix: str = "") -> dict[str, tuple[str, Quantity, str]]:
        p = prefix
        return {
            f"{p}saturation_flux_density": (f"Saturation flux density at {self.temperature - 273.15:.0f} degC",
                                            Quantity(value=self.saturation_flux_density, unit="T"), self.saturation_status),
            f"{p}saturation_margin": ("Margin to saturation (1 - Bpk/Bsat)", Quantity(value=self.margin_to_saturation, unit="1"), "calculated"),
            f"{p}flux_density_limit": ("Allowed peak flux density (policy)", Quantity(value=self.flux_limit, unit="T"), "assumed"),
        }


def flux_limit_from_policy(policy: FluxLimitSpec, saturation_flux_density: float) -> tuple[float, Assumption]:
    if policy.mode == "absolute":
        assert policy.absolute is not None
        limit = policy.absolute.to("T").value
        assumption = Assumption(id="flux_limit_absolute", statement=f"Peak flux density limit {limit:.3f} T supplied by the user.",
                                basis="user_input", affects=["turns", "flux_density_peak"])
    else:
        limit = policy.fraction * saturation_flux_density
        assumption = Assumption(id="flux_limit_fraction",
                                statement=f"Peak flux density limited to {policy.fraction:.0%} of Bsat at the operating temperature "
                                          f"({limit:.3f} T).", basis="default" if policy.fraction == 0.7 else "user_input",
                                affects=["turns", "flux_density_peak"])
    return limit, assumption


def screen_saturation(*, flux_density_peak: float, temperature: float, material: MaterialProperties,
                      policy: FluxLimitSpec | None = None) -> SaturationScreening:
    """Screen ``flux_density_peak`` (T) at ``temperature`` (K) against the material's Bsat and the flux-limit policy."""
    b = F._nonnegative("flux_density_peak", flux_density_peak)
    t = F._positive("temperature", temperature)
    policy = policy or FluxLimitSpec()
    bsat_q, status, refs = saturation_flux_density_at(material, Quantity(value=t, unit="K"))
    bsat = bsat_q.value
    if bsat <= 0:
        raise InvalidInputError("saturation flux density must be positive")
    limit, assumption = flux_limit_from_policy(policy, bsat)
    margin = F.saturation_margin(b, bsat)
    assumptions = [assumption]
    limitations = [Limitation(code="saturation_screening_scope",
                              statement="Screening compares peak flux density with datasheet Bsat; the B-H knee, local saturation at "
                                        "gaps/corners and DC-bias permeability roll-off are not modelled.")]
    warnings: list[Warning_] = []
    if status == "estimated":
        assumptions.append(Assumption(id="bsat_interpolated", statement=f"Bsat at {t - 273.15:.0f} degC linearly interpolated between "
                                      "datasheet temperature points.", basis="engineering_judgement", affects=["saturation_margin"]))
    saturated = b >= bsat
    within = b <= limit * (1 + 1e-12)
    if saturated:
        warnings.append(Warning_(code="saturated", message=f"Peak flux density {b:.3f} T reaches or exceeds Bsat {bsat:.3f} T.", severity="error"))
    elif not within:
        warnings.append(Warning_(code="flux_limit_exceeded", message=f"Peak flux density {b:.3f} T exceeds the policy limit {limit:.3f} T "
                                 f"(margin to saturation {margin:.1%}).", severity="warning"))
    return SaturationScreening(flux_density_peak=b, temperature=t, saturation_flux_density=bsat, saturation_status=status,
                               saturation_source_refs=refs, margin_to_saturation=margin, flux_limit=limit,
                               flux_limit_policy=policy.mode, within_limit=within, saturated=saturated,
                               assumptions=assumptions, limitations=limitations, warnings=warnings)


def screening_to_json(screening: SaturationScreening) -> dict[str, Any]:
    return screening.model_dump(mode="json", exclude_none=True)
