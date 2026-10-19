"""Classical Steinmetz equation (CSE) for the volumetric core loss under sinusoidal excitation.

Equation
--------
::

    Pv = k * f**alpha * Bpk**beta                                                    (1)

``Pv`` is the time-average volumetric loss density in W/m^3, ``f`` the frequency in Hz, ``Bpk`` the
peak of the AC flux-density component in T, and ``k`` (W/m^3 for ``f`` in Hz and ``B`` in T),
``alpha`` and ``beta`` (dimensionless, > 0) are the material parameters of a sinusoidal loss fit.
Parameters fitted in other unit systems (kHz, mT, kW/m^3, mW/cm^3) must be converted before
construction: ``k_SI = k_orig * P_unit / (F_unit**alpha * B_unit**beta)`` where ``P_unit``,
``F_unit`` and ``B_unit`` are the multiplicative factors of the original units to W/m^3, Hz and T
(see :class:`aipe_magnetics.data.records.SteinmetzParameterSet`, which stores SI values).

Assumptions and limitations (stated in every output)
---------------------------------------------------
* Sinusoidal excitation only - the parameters are a fit to sinusoidal loss data.
* Parameters are temperature specific. A set applies at its fit temperature; when no explicit
  temperature range is given, the declared validity is ``fit_temperature +- 5 K``. No temperature
  correction is applied.
* No DC-bias dependence, no minor-loop treatment, no relaxation effects.
* Validity is limited to the frequency / flux-density range of the fit. Outside it, the evaluation is
  rejected (policy ``reject``) or flagged (policy ``warn``) - never clipped.

Numerical behaviour
-------------------
* ``Bpk == 0`` is zero excitation and returns exactly ``0 W/m^3`` with an ``info`` warning
  ``zero_excitation``.
* ``f <= 0``, ``Bpk < 0``, non-finite or non-real inputs raise ``InvalidInputError``.
* Declared validity ranges must be physically admissible after unit conversion: frequency bounds
  ``> 0 Hz``, flux-density bounds ``>= 0 T``, temperature bounds ``> 0 K``; otherwise ``InvalidInputError``.
  Open (``None``) bounds are allowed.
* Overflow raises ``NumericalError`` (``OverflowError`` from ``math.pow`` is caught and the product is
  checked with ``math.isfinite``); the vectorised helper uses ``numpy.errstate(over='raise')``.

Reference
---------
C. P. Steinmetz, "On the law of hysteresis", Transactions of the American Institute of Electrical
Engineers, vol. IX, no. 1, pp. 1-64, 1892, doi:10.1109/T-AIEE.1892.5570437. Steinmetz established the
power-law dependence on the flux-density amplitude; the frequency exponent ``alpha`` is the modern
generalisation used throughout the core-loss literature.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

from aipe_magnetics.common.errors import DataNotFoundError, InvalidInputError, NumericalError, OutOfDomainError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import Assumption, AssumptionBasis, Limitation, ModelIdentity, SourceReference, Warning_
from aipe_magnetics.common.units import units
from aipe_magnetics.data.records import SteinmetzParameterSet
from aipe_magnetics.models.contract import (
    CoreLossModel,
    DomainCheck,
    DomainPolicy,
    ModelOutput,
    ModelParameter,
    ValidityDomain,
    ValidityRange,
)
from aipe_magnetics.models.waveforms import validate_bool, validate_real

__all__ = [
    "MODEL_VERSION",
    "STEINMETZ_MODEL_ID",
    "STEINMETZ_SOURCE",
    "FIT_TEMPERATURE_TOLERANCE_K",
    "SteinmetzParameterisedModel",
    "SteinmetzModel",
    "select_parameter_set",
]

MODEL_VERSION = "0.2.0"
STEINMETZ_MODEL_ID = "aipe.magnetics.model.steinmetz"
FIT_TEMPERATURE_TOLERANCE_K = 5.0
"""Half-width of the implied temperature validity band around a stated fit temperature (K)."""

STEINMETZ_SOURCE = SourceReference(
    id="publication.steinmetz-1892",
    kind="publication",
    reference=('C. P. Steinmetz, "On the law of hysteresis," Transactions of the American Institute of Electrical '
               "Engineers, vol. IX, no. 1, pp. 1-64, 1892."),
    url="https://doi.org/10.1109/T-AIEE.1892.5570437",
    note="Equation reference (power-law loss description). Parameter values come from separate sources.",
)

_POLICIES = ("reject", "warn")


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def _validate_policy(policy: Any) -> DomainPolicy:
    if policy not in _POLICIES:
        raise InvalidInputError(f"policy must be one of {_POLICIES}, got {policy!r}", details={"policy": str(policy)})
    return policy  # type: ignore[return-value]


def _reject_unknown_kwargs(kwargs: dict[str, Any]) -> None:
    if kwargs:
        raise InvalidInputError(f"unexpected keyword arguments: {', '.join(sorted(kwargs))}",
                                details={"unexpected": sorted(kwargs)})


def _optional_str(name: str, value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InvalidInputError(f"{name} must be a non-empty string or None", details={"name": name})
    return value


def _coerce_range(value: Any, unit: str, name: str, *, si_positive: bool = False,
                  si_nonnegative: bool = False) -> ValidityRange | None:
    """Accept ``ValidityRange``, ``RangeSI`` (any object with min/max/unit) or ``(min, max)`` in ``unit``.

    Bounds are converted to ``unit`` and must be physically admissible there: ``si_positive`` requires
    ``> 0`` (frequency, absolute temperature), ``si_nonnegative`` requires ``>= 0`` (flux density).
    """
    if value is None:
        return None
    if hasattr(value, "min") and hasattr(value, "max") and hasattr(value, "unit"):
        low, high, source_unit = value.min, value.max, value.unit
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        low, high, source_unit = value[0], value[1], unit
    else:
        raise InvalidInputError(f"{name} must be a ValidityRange, a RangeSI or a (min, max) pair in {unit}",
                                details={"name": name})
    if not units.compatible(source_unit, unit):  # raises UnitError (an InvalidInputError) for unknown units
        raise InvalidInputError(f"{name} must have the dimension of {unit!r}, got {source_unit!r}",
                                details={"name": name, "unit": source_unit})

    def _bound(bound: Any, which: str) -> float | None:
        if bound is None:
            return None
        si_value = units.convert(validate_real(f"{name}.{which}", bound), source_unit, unit)
        if si_positive and si_value <= 0.0:
            raise InvalidInputError(
                f"{name}.{which} must be > 0 {unit} after conversion, got {si_value:g} {unit} "
                f"(from {float(bound):g} {source_unit}); the bound is not physically admissible",
                details={"name": name, "bound": which, "value": si_value, "unit": unit})
        if si_nonnegative and si_value < 0.0:
            raise InvalidInputError(
                f"{name}.{which} must be >= 0 {unit} after conversion, got {si_value:g} {unit} "
                f"(from {float(bound):g} {source_unit}); the bound is not physically admissible",
                details={"name": name, "bound": which, "value": si_value, "unit": unit})
        return si_value

    low_si, high_si = _bound(low, "min"), _bound(high, "max")
    if low_si is not None and high_si is not None and low_si > high_si:
        raise InvalidInputError(f"{name}: min {low_si:g} exceeds max {high_si:g} {unit}", details={"name": name})
    return ValidityRange(min=low_si, max=high_si, unit=unit)


def _validate_sources(sources: Iterable[SourceReference]) -> list[SourceReference]:
    """Deduplicate identical references; two *different* references with the same id are an error."""
    result: list[SourceReference] = []
    for item in sources:
        if not isinstance(item, SourceReference):
            raise InvalidInputError(f"sources must contain SourceReference objects, got {type(item).__name__}")
        existing = next((kept for kept in result if kept.id == item.id), None)
        if existing is None:
            result.append(item)
        elif existing != item:
            raise InvalidInputError(
                f"source id {item.id!r} is used by two different SourceReference objects; source ids must be unique "
                "(the built-in equation references use the ids "
                f"{', '.join(repr(src.id) for src in (STEINMETZ_SOURCE,))} and others)",
                details={"id": item.id})
    return result


def _real_grid(name: str, values: Any) -> np.ndarray:
    try:
        array = np.asarray(values)
    except Exception as error:  # noqa: BLE001
        raise InvalidInputError(f"{name} could not be converted to an array: {error}", details={"name": name}) from None
    if array.dtype == object or array.dtype.kind not in "iuf":
        raise InvalidInputError(f"{name} must contain real numbers, got dtype {array.dtype}", details={"name": name})
    array = array.astype(float, copy=True)
    if not np.all(np.isfinite(array)):
        raise InvalidInputError(f"{name} must be finite everywhere", details={"name": name})
    return array


# --------------------------------------------------------------------------------------------------
# shared parameterised base
# --------------------------------------------------------------------------------------------------
class SteinmetzParameterisedModel(CoreLossModel):
    """Common parameter handling for models defined by a Steinmetz triple ``(k, alpha, beta)``.

    Subclasses set ``MODEL_ID``, ``identity``, ``EXCITATION``, ``DC_BIAS``, ``LIMITATIONS`` and
    ``NUMERICAL_CONSTRAINTS`` and implement :meth:`loss_density`.
    """

    MODEL_ID: str = ""
    identity: ModelIdentity
    EXCITATION: tuple[str, ...] = ("sinusoidal",)
    DC_BIAS: str = "unsupported"
    EQUATION_SOURCES: tuple[SourceReference, ...] = (STEINMETZ_SOURCE,)
    LIMITATIONS: tuple[tuple[str, str], ...] = ()
    NUMERICAL_CONSTRAINTS: tuple[str, ...] = ()

    def __init__(self, k: float, alpha: float, beta: float, *, material_id: str | None = None,
                 fit_temperature: float | None = None, frequency_range: Any = None, flux_density_range: Any = None,
                 temperature_range: Any = None, sources: Iterable[SourceReference] = (),
                 parameter_set_id: str | None = None, parameter_basis: str | None = None, synthetic: bool = False,
                 source_ref: str | None = None, note: str | None = None):
        self.k = validate_real("k", k, positive=True)
        self.alpha = validate_real("alpha", alpha, positive=True)
        self.beta = validate_real("beta", beta, positive=True)
        self.material_id = _optional_str("material_id", material_id)
        self.parameter_set_id = _optional_str("parameter_set_id", parameter_set_id)
        self.parameter_basis = _optional_str("parameter_basis", parameter_basis)
        self.source_ref = _optional_str("source_ref", source_ref)
        self.note = _optional_str("note", note)
        self.synthetic = validate_bool("synthetic", synthetic)
        self.fit_temperature = (None if fit_temperature is None
                                else validate_real("fit_temperature", fit_temperature, positive=True))
        self.frequency_range = _coerce_range(frequency_range, "Hz", "frequency_range", si_positive=True)
        self.flux_density_range = _coerce_range(flux_density_range, "T", "flux_density_range", si_nonnegative=True)
        explicit_temperature_range = _coerce_range(temperature_range, "K", "temperature_range", si_positive=True)
        self.temperature_range_implied = explicit_temperature_range is None and self.fit_temperature is not None
        if self.temperature_range_implied:
            assert self.fit_temperature is not None
            explicit_temperature_range = ValidityRange(min=self.fit_temperature - FIT_TEMPERATURE_TOLERANCE_K,
                                                       max=self.fit_temperature + FIT_TEMPERATURE_TOLERANCE_K, unit="K")
        self.temperature_range = explicit_temperature_range
        self.sources = _validate_sources(list(self.EQUATION_SOURCES) + list(sources))
        self.parameters = [
            ModelParameter(name="k", value=self.k, unit="W/m^3", source_ref=self.source_ref,
                           description="Steinmetz coefficient for f in Hz and Bpk in T"),
            ModelParameter(name="alpha", value=self.alpha, unit="1", source_ref=self.source_ref,
                           description="frequency exponent"),
            ModelParameter(name="beta", value=self.beta, unit="1", source_ref=self.source_ref,
                           description="flux-density exponent"),
        ]
        limitations = [statement for _, statement in self.LIMITATIONS]
        if self.synthetic:
            limitations.append("Parameters are synthetic / illustrative and do not describe any real material.")
        self.domain = ValidityDomain(
            materials=[self.material_id] if self.material_id else None,
            excitation=list(self.EXCITATION),  # type: ignore[arg-type]
            frequency=self.frequency_range,
            temperature=self.temperature_range,
            flux_density_peak=self.flux_density_range,
            dc_bias=self.DC_BIAS,  # type: ignore[arg-type]
            requires_history=False,
            numerical_constraints=list(self.NUMERICAL_CONSTRAINTS),
            known_limitations=limitations,
        )

    # -- construction from the data standard ------------------------------------------------------
    @classmethod
    def from_parameter_set(cls, parameter_set: SteinmetzParameterSet, material_id: str | None = None,
                           sources: Iterable[SourceReference] = ()):
        """Build the model from a :class:`SteinmetzParameterSet` record (SI values, validated on load).

        ``sources`` should carry the :class:`SourceReference` of the fit (the record only stores an id in
        ``source_ref``; nothing is fabricated here). A set with ``basis == 'synthetic'`` marks the model
        as synthetic and adds the corresponding limitation.
        """
        if not isinstance(parameter_set, SteinmetzParameterSet):
            raise InvalidInputError(f"parameter_set must be a SteinmetzParameterSet, got {type(parameter_set).__name__}")
        fit_temperature = None if parameter_set.temperature is None else parameter_set.temperature.magnitude("K")
        return cls(
            parameter_set.k.magnitude("W/m^3"), parameter_set.alpha, parameter_set.beta,
            material_id=material_id, fit_temperature=fit_temperature,
            frequency_range=parameter_set.frequency_range, flux_density_range=parameter_set.flux_density_range,
            temperature_range=parameter_set.temperature_range, sources=sources,
            parameter_set_id=parameter_set.id, parameter_basis=parameter_set.basis,
            synthetic=(parameter_set.basis == "synthetic"), source_ref=parameter_set.source_ref,
            note=parameter_set.note,
        )

    # -- validation -------------------------------------------------------------------------------
    @staticmethod
    def _validate_conditions(frequency: Any, flux_density_peak: Any,
                             temperature: Any) -> tuple[float, float, float | None]:
        f = validate_real("frequency", frequency, positive=True)
        b = validate_real("flux_density_peak", flux_density_peak, nonnegative=True)
        t = None if temperature is None else validate_real("temperature", temperature, positive=True)
        return f, b, t

    # -- closed form ------------------------------------------------------------------------------
    def closed_form(self, frequency: float, flux_density_peak: float) -> float:
        """``k * f**alpha * Bpk**beta`` in W/m^3 with input validation and overflow detection (no domain logic)."""
        f = validate_real("frequency", frequency, positive=True)
        b = validate_real("flux_density_peak", flux_density_peak, nonnegative=True)
        if b == 0.0:
            return 0.0
        try:
            value = self.k * math.pow(f, self.alpha) * math.pow(b, self.beta)
        except OverflowError:
            raise NumericalError(
                f"{self.MODEL_ID}: k*f^alpha*Bpk^beta overflows for f={f:g} Hz, Bpk={b:g} T",
                details={"frequency": f, "flux_density_peak": b, "alpha": self.alpha, "beta": self.beta}) from None
        return self.check_output_finite("loss_density", value)

    def loss_density_array(self, frequencies: Any, peaks: Any, *, temperature: float | None = None,
                           policy: DomainPolicy = "reject") -> np.ndarray:
        """Vectorised sinusoidal evaluation ``k * f**alpha * Bpk**beta`` on broadcastable NumPy grids (W/m^3).

        Applies the same validation as :meth:`loss_density`: finite real inputs, ``f > 0``, ``Bpk >= 0``
        (zero gives zero), overflow raises ``NumericalError``. Domain policy: with ``reject`` a temperature
        outside the declared range raises ``OutOfDomainError`` naming the temperature, and any grid point
        outside the frequency / flux-density ranges raises ``OutOfDomainError`` with the count of such
        points; with ``warn`` all points are evaluated and :meth:`in_domain_mask` tells which ones are inside
        (an array cannot carry warnings).
        """
        policy = _validate_policy(policy)
        f = _real_grid("frequencies", frequencies)
        b = _real_grid("peaks", peaks)
        t = None if temperature is None else validate_real("temperature", temperature, positive=True)
        if np.any(f <= 0.0):
            raise InvalidInputError("frequencies must be positive", details={"name": "frequencies"})
        if np.any(b < 0.0):
            raise InvalidInputError("peaks must be non-negative", details={"name": "peaks"})
        try:
            f, b = np.broadcast_arrays(f, b)
        except ValueError as error:
            raise InvalidInputError(f"frequencies and peaks are not broadcastable: {error}") from None
        if policy == "reject":
            if t is not None and self.domain.temperature is not None and not self.domain.temperature.contains(t):
                violation = f"temperature={t:g} K outside {self.domain.temperature.describe()}"
                raise OutOfDomainError(f"{self.MODEL_ID}: inputs outside declared validity domain: {violation}",
                                       details={"model": self.MODEL_ID, "violations": [violation], "temperature": t})
            inside = self.in_domain_mask(f, b, temperature=t)
            if not np.all(inside):
                count = int(np.count_nonzero(~inside))
                raise _out_of_domain_error(self, count, int(inside.size))
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            try:
                result = self.k * np.power(f, self.alpha) * np.power(b, self.beta)
            except FloatingPointError as error:
                raise NumericalError(f"{self.MODEL_ID}: overflow in vectorised evaluation ({error})",
                                     details={"alpha": self.alpha, "beta": self.beta}) from None
        result = np.asarray(result, dtype=float)
        if not np.all(np.isfinite(result)):
            raise NumericalError(f"{self.MODEL_ID}: non-finite loss density in vectorised evaluation")
        return result

    def in_domain_mask(self, frequencies: Any, peaks: Any, *, temperature: float | None = None) -> np.ndarray:
        """Boolean mask of grid points inside the declared frequency / flux-density (/ temperature) domain.

        Zero-flux points are not checked against the flux-density range (they are zero excitation).
        """
        f = _real_grid("frequencies", frequencies)
        b = _real_grid("peaks", peaks)
        f, b = np.broadcast_arrays(f, b)
        inside = np.ones(f.shape, dtype=bool)
        if self.domain.frequency is not None:
            rng = self.domain.frequency
            if rng.min is not None:
                inside &= f >= rng.min
            if rng.max is not None:
                inside &= f <= rng.max
        if self.domain.flux_density_peak is not None:
            rng = self.domain.flux_density_peak
            nonzero = b > 0.0
            if rng.min is not None:
                inside &= ~nonzero | (b >= rng.min)
            if rng.max is not None:
                inside &= ~nonzero | (b <= rng.max)
        if temperature is not None and self.domain.temperature is not None:
            t = validate_real("temperature", temperature, positive=True)
            if not self.domain.temperature.contains(t):
                inside &= False
        return inside

    # -- output assembly --------------------------------------------------------------------------
    def _parameter_assumption(self) -> Assumption:
        basis: AssumptionBasis = "user_input"
        if self.parameter_basis in ("datasheet_fit", "manufacturer_supplied"):
            basis = "datasheet"
        origin = f"parameter set {self.parameter_set_id!r}" if self.parameter_set_id else "caller-supplied parameters"
        if self.parameter_basis:
            origin += f" (basis: {self.parameter_basis})"
        if self.synthetic:
            origin += " - SYNTHETIC, illustrative only"
        return Assumption(
            id="steinmetz_parameters",
            statement=(f"Steinmetz parameters k={self.k:.6g} W/m^3 (f in Hz, Bpk in T), alpha={self.alpha:.6g}, "
                       f"beta={self.beta:.6g} from {origin}."),
            basis=basis, affects=["loss_density"])

    def _temperature_assumption(self, temperature: float | None) -> Assumption:
        if temperature is None:
            if self.fit_temperature is not None:
                statement = (f"No temperature given; the parameters are applied as fitted at "
                             f"{self.fit_temperature:g} K and the result refers to that temperature.")
            else:
                statement = ("No temperature given and no fit temperature recorded; the result refers to the "
                             "unknown temperature of the parameter fit.")
        else:
            statement = (f"Parameters applied without temperature correction at {temperature:g} K"
                         + (f" (fit temperature {self.fit_temperature:g} K)." if self.fit_temperature is not None
                            else "."))
        return Assumption(id="temperature_handling", statement=statement, basis="model_assumption",
                          affects=["loss_density"])

    def _new_output(self, domain: DomainCheck, *, frequency: float, flux_density_peak: float,
                    temperature: float | None, excitation: str, policy: str) -> ModelOutput:
        output = ModelOutput(model=self.identity, domain=domain, sources=list(self.sources))
        output.assumptions.append(self._parameter_assumption())
        output.assumptions.append(self._temperature_assumption(temperature))
        for code, statement in self.LIMITATIONS:
            output.limitations.append(Limitation(code=code, statement=statement))
        if self.synthetic:
            output.limitations.append(Limitation(
                code="synthetic_parameters",
                statement="Parameters are synthetic / illustrative and do not describe any real material."))
        if not domain.inside:
            output.warnings.append(Warning_(
                code="out_of_domain",
                message="inputs outside the declared validity domain: " + "; ".join(domain.violations),
                severity="warning"))
        output.diagnostics.update({
            "k": self.k, "alpha": self.alpha, "beta": self.beta,
            "frequency": frequency, "flux_density_peak": flux_density_peak, "temperature": temperature,
            "excitation": excitation, "policy": policy, "out_of_domain": not domain.inside,
            "zero_excitation": False,
            "material_id": self.material_id, "parameter_set_id": self.parameter_set_id,
        })
        return output

    @staticmethod
    def _mark_zero_excitation(output: ModelOutput) -> None:
        output.diagnostics["zero_excitation"] = True
        output.warnings.append(Warning_(
            code="zero_excitation",
            message="flux_density_peak is 0 T: no periodic excitation, loss density is exactly 0 W/m^3",
            severity="info"))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"{type(self).__name__}(k={self.k:g}, alpha={self.alpha:g}, beta={self.beta:g}, "
                f"material_id={self.material_id!r})")


def _out_of_domain_error(model: SteinmetzParameterisedModel, count: int, total: int) -> OutOfDomainError:
    return OutOfDomainError(
        f"{model.MODEL_ID}: {count} of {total} grid points lie outside the declared validity domain "
        f"(frequency {model.domain.frequency.describe() if model.domain.frequency else 'unbounded'}, "
        f"flux density {model.domain.flux_density_peak.describe() if model.domain.flux_density_peak else 'unbounded'})",
        details={"model": model.MODEL_ID, "outside": count, "total": total})


# --------------------------------------------------------------------------------------------------
# classical Steinmetz equation
# --------------------------------------------------------------------------------------------------
class SteinmetzModel(SteinmetzParameterisedModel):
    """Classical Steinmetz equation ``Pv = k f^alpha Bpk^beta`` (sinusoidal excitation, SI units)."""

    MODEL_ID = STEINMETZ_MODEL_ID
    identity = ModelIdentity(id=STEINMETZ_MODEL_ID, name="Classical Steinmetz equation", version=MODEL_VERSION,
                             family="empirical", method_version=None)
    EXCITATION = ("sinusoidal",)
    DC_BIAS = "unsupported"
    LIMITATIONS = (
        ("sinusoidal_only", "Classical Steinmetz equation: valid for sinusoidal flux density only; non-sinusoidal "
                            "waveforms require a waveform-aware model such as iGSE."),
        ("no_dc_bias", "No DC-bias dependence: the loss change under premagnetisation is not modelled."),
        ("no_minor_loops", "Minor hysteresis loops are not represented."),
        ("temperature_specific_parameters", "Parameters apply at their fit temperature; no temperature correction "
                                            "is applied."),
        ("fit_range_only", "Extrapolation outside the fitted frequency / flux-density range is not validated; such "
                           "inputs are rejected or flagged according to the domain policy."),
    )
    NUMERICAL_CONSTRAINTS = ("closed form; overflow is detected and raised, never returned as inf",)

    def loss_density(self, *, frequency: float, flux_density_peak: float, temperature: float | None = None,
                     policy: DomainPolicy = "reject", **kwargs: Any) -> ModelOutput:
        """Evaluate equation (1). Inputs in Hz, T (peak of the AC component), K. See module docstring."""
        _reject_unknown_kwargs(kwargs)
        policy = _validate_policy(policy)
        f, b, t = self._validate_conditions(frequency, flux_density_peak, temperature)
        zero = b == 0.0
        domain = self.check_domain(frequency=f, temperature=t, flux_density_peak=None if zero else b,
                                   excitation="sinusoidal")
        if not domain.inside and policy == "reject":
            domain.raise_if_outside(self.identity.id)
        value = 0.0 if zero else self.closed_form(f, b)
        output = self._new_output(domain, frequency=f, flux_density_peak=b, temperature=t, excitation="sinusoidal",
                                  policy=policy)
        output.values["loss_density"] = Quantity(value=value, unit="W/m^3")
        if zero:
            self._mark_zero_excitation(output)
        return output


# --------------------------------------------------------------------------------------------------
# parameter-set selection
# --------------------------------------------------------------------------------------------------
def _range_bounds_k(rng: Any) -> tuple[float | None, float | None]:
    low = None if rng.min is None else units.convert(float(rng.min), rng.unit, "K")
    high = None if rng.max is None else units.convert(float(rng.max), rng.unit, "K")
    return low, high


def select_parameter_set(sets: Sequence[SteinmetzParameterSet], temperature: float) -> SteinmetzParameterSet:
    """Return the parameter set valid at ``temperature`` (K); never a nearest-neighbour guess.

    A set matches when its ``temperature_range`` contains ``temperature`` or, when it has no range but a
    fit ``temperature``, when ``|temperature - fit| <= 5 K``. Sets stating neither are never selected.
    When several sets match, the one whose fit temperature is closest wins; sets without a fit
    temperature rank after those with one (narrower ranges first); remaining ties keep input order.
    Raises :class:`DataNotFoundError` listing the available temperatures when nothing matches.
    """
    t = validate_real("temperature", temperature, positive=True)
    if isinstance(sets, SteinmetzParameterSet) or not isinstance(sets, Sequence):
        raise InvalidInputError("sets must be a sequence of SteinmetzParameterSet records")
    candidates: list[tuple[tuple[int, float], int, SteinmetzParameterSet]] = []
    available: list[str] = []
    for index, item in enumerate(sets):
        if not isinstance(item, SteinmetzParameterSet):
            raise InvalidInputError(f"sets[{index}] is not a SteinmetzParameterSet (got {type(item).__name__})")
        fit = None if item.temperature is None else item.temperature.magnitude("K")
        if item.temperature_range is not None:
            low, high = _range_bounds_k(item.temperature_range)
            available.append(f"{item.id}: [{'-inf' if low is None else f'{low:g}'}, "
                             f"{'+inf' if high is None else f'{high:g}'}] K")
            if (low is None or t >= low) and (high is None or t <= high):
                if fit is not None:
                    key = (0, abs(t - fit))
                else:
                    width = math.inf if low is None or high is None else high - low
                    key = (1, width)
                candidates.append((key, index, item))
        elif fit is not None:
            available.append(f"{item.id}: {fit:g} K (+-{FIT_TEMPERATURE_TOLERANCE_K:g} K)")
            if abs(t - fit) <= FIT_TEMPERATURE_TOLERANCE_K:
                candidates.append(((0, abs(t - fit)), index, item))
        else:
            available.append(f"{item.id}: no temperature stated (never selected)")
    if not candidates:
        raise DataNotFoundError(
            f"no Steinmetz parameter set is valid at {t:g} K; available: " + ("; ".join(available) or "none"),
            details={"temperature": t, "available": available})
    candidates.sort(key=lambda entry: (entry[0], entry[1]))
    return candidates[0][2]
