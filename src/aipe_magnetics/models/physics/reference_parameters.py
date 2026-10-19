"""Source references and reference parameter sets for the physics-based model family.

Engineering-integrity rules applied here:

* Every parameter set carries a ``basis`` (``'illustrative'`` / ``'published'`` / ``'fitted'``) and a
  :class:`SourceReference`.  The only set shipped with v0.2 is **illustrative**: the widely
  reproduced Jiles-Atherton demonstration values ``Ms = 1.6e6 A/m, a = 1100 A/m, alpha = 1.6e-3,
  k = 400 A/m, c = 0.2``.  They are *not* a fitted material and must not be attributed to one;
  predictions made with them are demonstrations of the model, not material data.
* Publication references are given as full citations; DOIs are included only where known with
  confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from aipe_magnetics.common.errors import DataNotFoundError
from aipe_magnetics.common.results import SourceReference
from aipe_magnetics.models.contract import ModelParameter

__all__ = ["JA_1986", "JTD_1992", "JILES_1992_MINOR_LOOPS", "BERTOTTI_1988", "LAMINATION_EXAMPLE",
           "JA_ILLUSTRATIVE_SOURCE", "JilesAthertonParameterSet", "JA_ILLUSTRATIVE", "REFERENCE_PARAMETER_SETS",
           "get_parameter_set"]

JA_1986 = SourceReference(
    id="pub.jiles-atherton-1986",
    kind="publication",
    reference=("D. C. Jiles and D. L. Atherton, 'Theory of ferromagnetic hysteresis', Journal of Magnetism and "
               "Magnetic Materials 61 (1986) 48-60"),
    url="https://doi.org/10.1016/0304-8853(86)90066-1",
    license="NOASSERTION",
    note="Original model: effective field, Langevin anhysteretic, irreversible/reversible decomposition.",
)

JTD_1992 = SourceReference(
    id="pub.jiles-thoelke-devine-1992",
    kind="publication",
    reference=("D. C. Jiles, J. B. Thoelke and M. K. Devine, 'Numerical determination of hysteresis parameters for "
               "the modeling of magnetic properties using the theory of ferromagnetic hysteresis', IEEE "
               "Transactions on Magnetics 28 (1992) 27-35"),
    license="NOASSERTION",
    note="Standard field-driven differential form dM_irr/dH = (M_an - M_irr)/(delta k - alpha (M_an - M_irr)) "
         "with k in A/m, M_rev = c (M_an - M_irr), and the parameter-determination relations.",
)

JILES_1992_MINOR_LOOPS = SourceReference(
    id="pub.jiles-1992-minor-loops",
    kind="publication",
    reference=("D. C. Jiles, 'A self consistent generalized model for the calculation of minor loop excursions in "
               "the theory of hysteresis', IEEE Transactions on Magnetics 28 (1992) 2602-2604"),
    license="NOASSERTION",
    note="Source of the delta_M factor that suppresses the nonphysical negative susceptibility after a field "
         "reversal (irreversible change only when (M_an - M_irr) * delta > 0).",
)

BERTOTTI_1988 = SourceReference(
    id="pub.bertotti-1988",
    kind="publication",
    reference=("G. Bertotti, 'General properties of power losses in soft ferromagnetic materials', IEEE "
               "Transactions on Magnetics 24 (1988) 621-630"),
    url="https://doi.org/10.1109/20.43994",
    license="NOASSERTION",
    note="Statistical loss separation: hysteresis + classical eddy-current + excess loss.",
)

LAMINATION_EXAMPLE = SourceReference(
    id="src.example.lamination-si-steel",
    kind="assumption",
    reference=("Example lamination values for the classical eddy-current coefficient: conductivity sigma = 2.0e6 S/m "
               "(resistivity 50 uOhm*cm, representative of ~3 % Si-Fe electrical steel) and thickness d = 0.35 mm "
               "(a standard lamination gauge). Illustrative example values, not a specific product datasheet."),
    license="NOASSERTION",
    note="basis: example values for verification of the k_c = pi^2 sigma d^2 / 6 formula only",
)

JA_ILLUSTRATIVE_SOURCE = SourceReference(
    id="src.ja.illustrative-demo-parameters",
    kind="assumption",
    reference=("Illustrative Jiles-Atherton parameter set commonly used in demonstrations of the model "
               "(Ms = 1.6e6 A/m, a = 1100 A/m, alpha = 1.6e-3, k = 400 A/m, c = 0.2); not a fitted material"),
    license="NOASSERTION",
    note="basis: illustrative. Do not attribute to a specific material; identify parameters per material "
         "(e.g. after Jiles, Thoelke & Devine 1992) before any engineering use.",
)

ParameterBasis = Literal["illustrative", "published", "fitted"]


@dataclass(frozen=True)
class JilesAthertonParameterSet:
    """Named Jiles-Atherton parameter set with provenance."""

    id: str
    Ms: float
    a: float
    alpha: float
    k: float
    c: float
    basis: ParameterBasis
    source: SourceReference
    description: str = ""

    def as_model_parameters(self) -> list[ModelParameter]:
        ref = self.source.id
        return [
            ModelParameter(name="Ms", value=self.Ms, unit="A/m", description="saturation magnetisation", source_ref=ref),
            ModelParameter(name="a", value=self.a, unit="A/m", description="anhysteretic shape parameter", source_ref=ref),
            ModelParameter(name="alpha", value=self.alpha, unit="1", description="inter-domain coupling", source_ref=ref),
            ModelParameter(name="k", value=self.k, unit="A/m", description="pinning (irreversible loss) parameter",
                           source_ref=ref),
            ModelParameter(name="c", value=self.c, unit="1", description="reversibility coefficient", source_ref=ref),
        ]

    def as_kwargs(self) -> dict[str, float]:
        return {"Ms": self.Ms, "a": self.a, "alpha": self.alpha, "k": self.k, "c": self.c}

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "basis": self.basis, "description": self.description, **self.as_kwargs(),
                "source": self.source.model_dump(mode="json", exclude_none=True)}


JA_ILLUSTRATIVE = JilesAthertonParameterSet(
    id="ja.illustrative-demo",
    Ms=1.6e6, a=1100.0, alpha=1.6e-3, k=400.0, c=0.2,
    basis="illustrative",
    source=JA_ILLUSTRATIVE_SOURCE,
    description="Illustrative Jiles-Atherton demonstration parameters (not a material).",
)

REFERENCE_PARAMETER_SETS: dict[str, JilesAthertonParameterSet] = {JA_ILLUSTRATIVE.id: JA_ILLUSTRATIVE}


def get_parameter_set(set_id: str) -> JilesAthertonParameterSet:
    try:
        return REFERENCE_PARAMETER_SETS[set_id]
    except KeyError:
        raise DataNotFoundError(f"unknown Jiles-Atherton parameter set {set_id!r}; available: "
                                f"{sorted(REFERENCE_PARAMETER_SETS)}", details={"id": set_id}) from None
