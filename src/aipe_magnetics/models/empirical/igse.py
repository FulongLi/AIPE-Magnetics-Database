"""Improved Generalized Steinmetz Equation (iGSE) for non-sinusoidal flux-density waveforms.

Reference
---------
K. Venkatachalam, C. R. Sullivan, T. Abdallah and H. Tacca, "Accurate prediction of ferrite core loss
with nonsinusoidal waveforms using only Steinmetz parameters," in Proc. IEEE Workshop on Computers in
Power Electronics (COMPEL 2002), Mayaguez, PR, USA, June 2002, pp. 36-41, doi:10.1109/CIPE.2002.1196712.

Equations (SI: W/m^3, Hz, T, s)
-------------------------------
::

    Pv  = (1/T) * Integral_0^T  k_i * |dB/dt|^alpha * (DeltaB)^(beta - alpha) dt                 (1)
    k_i = k / ( (2*pi)^(alpha - 1) * Integral_0^{2*pi} |cos(theta)|^alpha * 2^(beta - alpha) dtheta ) (2)

``DeltaB`` is the peak-to-peak flux density of the major loop, ``T = 1/f`` the period and
``(k, alpha, beta)`` are the classical Steinmetz parameters of the sinusoidal fit (see
:mod:`aipe_magnetics.models.empirical.steinmetz`). ``k_i`` is defined so that (1) reproduces the
classical equation ``k f^alpha Bpk^beta`` exactly for a sinusoid.

Evaluation paths
----------------
* **sinusoidal** ``B = dc + Bpk sin(omega t)``: ``dB/dt = Bpk omega cos(omega t)``, ``DeltaB = 2 Bpk``, and
  (1) reduces analytically to ``Pv = k_i (2 pi f)^alpha Bpk^alpha (2 Bpk)^(beta-alpha) I(alpha) / (2 pi)``
  with ``I(alpha) = Integral_0^{2 pi} |cos theta|^alpha dtheta`` (the same quadrature value used in ``k_i``).
* **piecewise-linear kinds** (``triangular``, ``trapezoidal``, ``piecewise_linear``): on segment ``i`` with
  period fraction ``x_i = dt_i / T`` (taken directly from the breakpoints) and increment ``dB_i`` the slope
  is constant, so::

      Pv = k_i DeltaB^(beta-alpha) (1/T) Sum_i |dB_i / dt_i|^alpha dt_i
         = k_i DeltaB^(beta-alpha) f^alpha Sum_i |dB_i|^alpha x_i^(1-alpha)                         (3)

  which is exact for the waveform (no sampling). A segment of zero fraction cannot be evaluated (it would
  carry an infinite ``dB/dt``) and raises ``NumericalError``; the waveform constructors reject such
  breakpoints, so this is a defence in depth.
* **sampled** waveforms (``n`` stored samples spanning one period, both ends included; see
  :meth:`~aipe_magnetics.models.waveforms.Waveform.sampled` for the ``[0, T)`` convention) are interpreted as
  their piecewise-linear interpolant: on sampling interval ``i`` the slope is the forward difference
  ``(B_{i+1} - B_i) / dt`` with ``dt = T/(n-1)``, so ``|dB/dt|^alpha`` is piecewise constant and its time
  integral is **exactly** ``Sum_i |B_{i+1}-B_i|^alpha dt^(1-alpha)`` (no further quadrature rule is involved).
  Relative to the underlying smooth waveform this is a midpoint-type rule - the forward difference is the
  second-order central difference at the interval midpoint - hence ``O(n^-2)`` for smooth waveforms with
  ``alpha >= 1`` (observed order 2.00 at alpha = 1.0, 1.5, 2.0 for a sinusoid). For ``alpha < 1`` the
  integrand ``|dB/dt|^alpha`` has a cusp wherever ``dB/dt = 0`` and the observed order is irregular (1.6-2.8
  at alpha = 0.7). Waveforms whose corners fall on samples are reproduced exactly. The output records
  ``n_samples`` and warns (``sampling_sensitivity``) when ``n < 256``. Noise is not filtered: it inflates
  ``|dB/dt|^alpha`` and must be removed (smoothing / decimation) by the caller.

Closed forms used as analytic references (tests and verification only)
----------------------------------------------------------------------
For a triangular flux with rise fraction ``D`` (``0 < D < 1``) and ``DeltaB = 2 Bpk``::

    Pv = k_i (2 Bpk)^beta f^alpha [ D^(1-alpha) + (1-D)^(1-alpha) ]                                  (4)

For the symmetric trapezoid of :meth:`~aipe_magnetics.models.waveforms.Waveform.trapezoidal` (two sloped
segments of fraction ``D_r`` each; the flats contribute nothing; ``0 < D_r <= 1/2``)::

    Pv = 2 k_i (2 Bpk)^beta f^alpha D_r^(1-alpha)                                                    (5)

which coincides with (4) at ``D = D_r = 1/2`` (no flat).

Numerics of ``k_i``
-------------------
The angular integral in (2) is evaluated with ``scipy.integrate.quad`` separately on the four quarter
periods ``[0, pi/2], [pi/2, pi], [pi, 3pi/2], [3pi/2, 2pi]`` (the integrand is not smooth at odd
multiples of ``pi/2``), relative tolerance 1e-12, and cached per ``(k, alpha, beta)``. The exact value
``I(alpha) = 2 sqrt(pi) Gamma((alpha+1)/2) / Gamma(alpha/2 + 1)`` (evaluated through ``lgamma`` so that it
cannot overflow) and Venkatachalam's closed-form approximation
``k_i ~= k / (2^(beta+1) pi^(alpha-1) (0.2761 + 1.7061/(alpha+1.354)))`` are provided as *references only*
(tests / verification) and are never used in the production path.

Minor loops
-----------
Minor loops are not decomposed. A waveform with more than two ``dB/dt`` reversals per period (counted by
:meth:`~aipe_magnetics.models.waveforms.Waveform.count_reversals` with ``reversal_tolerance``, default
:data:`~aipe_magnetics.models.waveforms.DEFAULT_REVERSAL_TOLERANCE`) is rejected under policy ``reject``
(``OutOfDomainError``) or evaluated with the major-loop ``DeltaB`` applied to every segment under policy
``warn`` (warning ``minor_loops_ignored``). Compared with the loop-decomposed evaluation of the cited paper
(each minor loop weighted with its own, smaller ``DeltaB_minor``) the two evaluations share every
``|dB/dt|^alpha`` integral and differ only through the factor ``DeltaB^(beta-alpha)`` on the minor-loop
segments, so the undecomposed value **overestimates** the decomposed one when ``beta > alpha`` (the usual
ferrite case), underestimates it when ``beta < alpha`` and equals it when ``beta == alpha``.

Limitations
-----------
* DC bias is ignored (``dc_bias='ignored'``): only ``dB/dt`` and ``DeltaB`` enter (1). A DC component
  larger than :data:`DC_BIAS_RELATIVE_THRESHOLD` (0.1 %) of the peak-to-peak value is reported with the
  warning ``dc_bias_ignored``; smaller offsets (e.g. discretisation residuals of sampled data) are below any
  documented premagnetisation effect and appear in the diagnostics only.
* Minor loops are not decomposed (see above).
* No relaxation term (i2GSE): losses during zero-``dB/dt`` intervals are not modelled.
* ``(k, alpha, beta)`` are a sinusoidal fit; the waveform generalisation is the iGSE hypothesis and is
  validated in the literature for ferrites to within typically tens of percent, not an exact law.
* Sampled waveforms are used as given; measurement noise inflates the loss integral and may be counted as
  minor loops. ``reversal_tolerance`` affects only the loop count, never the integral.
"""

from __future__ import annotations

import math
import warnings as _warnings
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.integrate import IntegrationWarning, quad

from aipe_magnetics.common.errors import InvalidInputError, NumericalError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import Assumption, ModelIdentity, SourceReference, Warning_
from aipe_magnetics.models.contract import DomainCheck, DomainPolicy, ModelOutput
from aipe_magnetics.models.empirical.steinmetz import (
    MODEL_VERSION,
    STEINMETZ_SOURCE,
    SteinmetzParameterisedModel,
    _reject_unknown_kwargs,
    _validate_policy,
)
from aipe_magnetics.models.waveforms import DEFAULT_REVERSAL_TOLERANCE, Waveform, validate_real

__all__ = [
    "DC_BIAS_RELATIVE_THRESHOLD",
    "IGSE_MODEL_ID",
    "IGSE_SOURCE",
    "SAMPLING_WARNING_THRESHOLD",
    "IGSEModel",
    "abs_cos_power_integral",
    "abs_cos_power_integral_closed_form",
    "igse_coefficient",
    "igse_coefficient_venkatachalam_approximation",
    "igse_trapezoidal_closed_form",
    "igse_triangular_closed_form",
]

IGSE_MODEL_ID = "aipe.magnetics.model.igse"
SAMPLING_WARNING_THRESHOLD = 256
"""Sampled waveforms with fewer samples per period trigger the ``sampling_sensitivity`` warning."""
DC_BIAS_RELATIVE_THRESHOLD = 1e-3
"""A DC component with ``|dc| <= DC_BIAS_RELATIVE_THRESHOLD * peak_to_peak`` counts as no DC bias.

0.1 % of the excursion is far below the resolution of flux-density measurements and below any documented
effect of premagnetisation on core loss, and well above the discretisation residual of the trapezoid mean
of sampled waveforms (about 1e-4 of the excursion at 65 samples per period, 1e-7 at 1024). The exact
``dc_component`` is always reported in the diagnostics; only the warning is thresholded.
"""

IGSE_SOURCE = SourceReference(
    id="publication.venkatachalam-2002-igse",
    kind="publication",
    reference=('K. Venkatachalam, C. R. Sullivan, T. Abdallah and H. Tacca, "Accurate prediction of ferrite core loss '
               'with nonsinusoidal waveforms using only Steinmetz parameters," Proc. IEEE Workshop on Computers in '
               "Power Electronics (COMPEL 2002), Mayaguez, PR, USA, 2002, pp. 36-41."),
    url="https://doi.org/10.1109/CIPE.2002.1196712",
    note="Defines the iGSE loss integral and the coefficient k_i; also gives the closed-form approximation of k_i.",
)

_QUAD_EPSREL = 1e-12
_QUAD_ERROR_LIMIT = 1e-9  # accepted total quad error estimate relative to the integral
_QUARTERS = tuple((q * math.pi / 2.0, (q + 1) * math.pi / 2.0) for q in range(4))
_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------------------------------
# k_i: production quadrature, exact reference and published approximation
# --------------------------------------------------------------------------------------------------
@lru_cache(maxsize=256)
def _abs_cos_power_integral_cached(alpha: float) -> float:
    total = 0.0
    error_total = 0.0
    for lower, upper in _QUARTERS:
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", IntegrationWarning)
            try:
                value, error = quad(lambda theta: abs(math.cos(theta)) ** alpha, lower, upper,
                                    epsabs=0.0, epsrel=_QUAD_EPSREL, limit=200)
            except IntegrationWarning as warning:
                raise NumericalError(f"quadrature of |cos(theta)|^alpha did not converge for alpha={alpha:g}: {warning}",
                                     details={"alpha": alpha}) from None
        total += value
        error_total += error
    if not math.isfinite(total) or total <= 0.0 or error_total > _QUAD_ERROR_LIMIT * total:
        raise NumericalError(f"quadrature of |cos(theta)|^alpha unreliable for alpha={alpha:g} "
                             f"(value {total!r}, error estimate {error_total!r})", details={"alpha": alpha})
    return total


def abs_cos_power_integral(alpha: float) -> float:
    """``I(alpha) = Integral_0^{2 pi} |cos theta|^alpha dtheta`` by adaptive quadrature (production path, cached)."""
    return _abs_cos_power_integral_cached(validate_real("alpha", alpha, positive=True))


def abs_cos_power_integral_closed_form(alpha: float) -> float:
    """Exact ``I(alpha) = 2 sqrt(pi) Gamma((alpha+1)/2) / Gamma(alpha/2 + 1)`` (reference for tests only).

    Evaluated as ``2 sqrt(pi) exp(lgamma((alpha+1)/2) - lgamma(alpha/2 + 1))`` so that it cannot overflow for
    any admissible ``alpha`` (``math.gamma`` itself overflows beyond ``alpha ~ 341``).
    """
    a = validate_real("alpha", alpha, positive=True)
    return 2.0 * math.sqrt(math.pi) * math.exp(math.lgamma((a + 1.0) / 2.0) - math.lgamma(a / 2.0 + 1.0))


@lru_cache(maxsize=256)
def _igse_coefficient_cached(k: float, alpha: float, beta: float) -> float:
    integral = _abs_cos_power_integral_cached(alpha)
    try:
        denominator = math.pow(_TWO_PI, alpha - 1.0) * math.pow(2.0, beta - alpha) * integral
        value = k / denominator
    except (OverflowError, ZeroDivisionError):
        raise NumericalError(f"k_i overflows for k={k:g}, alpha={alpha:g}, beta={beta:g}",
                             details={"k": k, "alpha": alpha, "beta": beta}) from None
    if not math.isfinite(value) or value <= 0.0:
        raise NumericalError(f"k_i is not a finite positive number ({value!r}) for k={k:g}, alpha={alpha:g}, "
                             f"beta={beta:g}", details={"k": k, "alpha": alpha, "beta": beta})
    return value


def igse_coefficient(k: float, alpha: float, beta: float) -> float:
    """``k_i`` of equation (2) via quadrature (production path, cached per ``(k, alpha, beta)``)."""
    return _igse_coefficient_cached(validate_real("k", k, positive=True), validate_real("alpha", alpha, positive=True),
                                    validate_real("beta", beta, positive=True))


def igse_coefficient_venkatachalam_approximation(k: float, alpha: float, beta: float) -> float:
    """Closed-form approximation of ``k_i`` from Venkatachalam et al. (2002). REFERENCE ONLY.

    ``k_i ~= k / (2^(beta+1) pi^(alpha-1) (0.2761 + 1.7061 / (alpha + 1.354)))``. Never used by the model;
    provided so tests and verification can confirm the quadrature against the published approximation.
    """
    kk = validate_real("k", k, positive=True)
    a = validate_real("alpha", alpha, positive=True)
    b = validate_real("beta", beta, positive=True)
    return kk / (math.pow(2.0, b + 1.0) * math.pow(math.pi, a - 1.0) * (0.2761 + 1.7061 / (a + 1.354)))


def igse_triangular_closed_form(k: float, alpha: float, beta: float, *, peak: float, frequency: float,
                                duty: float) -> float:
    """Equation (4): ``k_i (2 Bpk)^beta f^alpha [D^(1-alpha) + (1-D)^(1-alpha)]`` (analytic reference)."""
    a = validate_real("alpha", alpha, positive=True)
    b = validate_real("beta", beta, positive=True)
    p = validate_real("peak", peak, positive=True)
    f = validate_real("frequency", frequency, positive=True)
    d = validate_real("duty", duty)
    if not 0.0 < d < 1.0:
        raise InvalidInputError(f"duty must lie strictly between 0 and 1, got {d:g}")
    k_i = igse_coefficient(k, a, b)
    try:
        return k_i * math.pow(2.0 * p, b) * math.pow(f, a) * (math.pow(d, 1.0 - a) + math.pow(1.0 - d, 1.0 - a))
    except OverflowError:
        raise NumericalError("triangular closed form overflows", details={"peak": p, "frequency": f}) from None


def igse_trapezoidal_closed_form(k: float, alpha: float, beta: float, *, peak: float, frequency: float,
                                 duty_rise: float) -> float:
    """Equation (5): ``2 k_i (2 Bpk)^beta f^alpha D_r^(1-alpha)`` for the symmetric trapezoid (analytic reference).

    ``duty_rise`` is the period fraction of each of the two sloped segments, ``0 < duty_rise <= 0.5``
    (``0.5`` is the symmetric triangle without flats, where (5) equals (4) at ``D = 0.5``).
    """
    a = validate_real("alpha", alpha, positive=True)
    b = validate_real("beta", beta, positive=True)
    p = validate_real("peak", peak, positive=True)
    f = validate_real("frequency", frequency, positive=True)
    d = validate_real("duty_rise", duty_rise)
    if not 0.0 < d <= 0.5:
        raise InvalidInputError(f"duty_rise must lie in (0, 0.5], got {d:g}")
    k_i = igse_coefficient(k, a, b)
    try:
        return 2.0 * k_i * math.pow(2.0 * p, b) * math.pow(f, a) * math.pow(d, 1.0 - a)
    except OverflowError:
        raise NumericalError("trapezoidal closed form overflows", details={"peak": p, "frequency": f}) from None


# --------------------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------------------
class IGSEModel(SteinmetzParameterisedModel):
    """Improved Generalized Steinmetz Equation (equations (1)-(3) of the module docstring)."""

    MODEL_ID = IGSE_MODEL_ID
    identity = ModelIdentity(id=IGSE_MODEL_ID, name="Improved Generalized Steinmetz Equation (iGSE)",
                             version=MODEL_VERSION, family="empirical", method_version="igse-quad-ki/1.0.0")
    EXCITATION = ("sinusoidal", "triangular", "trapezoidal", "piecewise_linear", "sampled")
    DC_BIAS = "ignored"
    EQUATION_SOURCES = (IGSE_SOURCE, STEINMETZ_SOURCE)
    LIMITATIONS = (
        ("dc_bias_ignored", "iGSE ignores DC bias: only dB/dt and the peak-to-peak flux density enter the loss "
                            "integral; the loss change under premagnetisation is not modelled."),
        ("no_minor_loops", "Minor loops are not decomposed: waveforms with more than two dB/dt reversals per period "
                           "are rejected (policy reject) or evaluated with the major-loop peak-to-peak flux density "
                           "applied to every segment (policy warn), which overestimates the loop-decomposed iGSE "
                           "result when beta > alpha and underestimates it when beta < alpha."),
        ("no_relaxation", "No relaxation term (i2GSE): losses during zero-dB/dt intervals after a flux transition "
                          "are not modelled."),
        ("sinusoidal_parameter_fit", "(k, alpha, beta) are a sinusoidal fit; the waveform generalisation is the "
                                     "iGSE hypothesis, validated in the literature for ferrites to within typically "
                                     "tens of percent, not an exact law."),
        ("temperature_specific_parameters", "Parameters apply at their fit temperature; no temperature correction "
                                            "is applied."),
        ("fit_range_only", "Extrapolation outside the fitted frequency / flux-density range is not validated; such "
                           "inputs are rejected or flagged according to the domain policy."),
        ("noisy_sampled_data", "Sampled waveforms are used as given: measurement noise inflates |dB/dt|^alpha and can "
                               "be counted as minor loops. Smooth or decimate measured data before evaluation; the "
                               "reversal_tolerance argument only affects the loop count, never the loss integral."),
    )
    NUMERICAL_CONSTRAINTS = (
        "k_i from adaptive quadrature of |cos(theta)|^alpha over four quarter periods (relative tolerance 1e-12)",
        "piecewise-linear waveforms are integrated exactly per segment (every segment fraction must be > 0)",
        "sampled waveforms: exact integral of the piecewise-linear interpolant of the samples (midpoint-type rule on "
        f"|dB/dt|^alpha, O(n^-2) for smooth waveforms with alpha >= 1); n >= {SAMPLING_WARNING_THRESHOLD} samples per "
        "period recommended; noise is not filtered",
    )

    @property
    def k_i(self) -> float:
        """iGSE coefficient ``k_i`` (equation (2)) for this parameter set."""
        return igse_coefficient(self.k, self.alpha, self.beta)

    @classmethod
    def from_steinmetz(cls, model: SteinmetzParameterisedModel) -> "IGSEModel":
        """Build an iGSE model that shares the parameters, ranges and sources of ``model``."""
        if not isinstance(model, SteinmetzParameterisedModel):
            raise InvalidInputError(f"model must be a Steinmetz-parameterised model, got {type(model).__name__}")
        extra_sources = [s for s in model.sources if s.id not in {src.id for src in cls.EQUATION_SOURCES}]
        return cls(model.k, model.alpha, model.beta, material_id=model.material_id,
                   fit_temperature=model.fit_temperature, frequency_range=model.frequency_range,
                   flux_density_range=model.flux_density_range,
                   temperature_range=None if model.temperature_range_implied else model.temperature_range,
                   sources=extra_sources, parameter_set_id=model.parameter_set_id,
                   parameter_basis=model.parameter_basis, synthetic=model.synthetic, source_ref=model.source_ref,
                   note=model.note)

    # -- evaluation -------------------------------------------------------------------------------
    def loss_density(self, *, frequency: float | None = None, flux_density_peak: float | None = None,
                     temperature: float | None = None, waveform: Waveform | None = None,
                     policy: DomainPolicy = "reject", reversal_tolerance: float = DEFAULT_REVERSAL_TOLERANCE,
                     **kwargs: Any) -> ModelOutput:
        """Evaluate equation (1) for ``waveform`` (default: sinusoid of the given peak and frequency).

        When a waveform is given it is authoritative; ``frequency`` / ``flux_density_peak`` may be omitted
        or must agree with ``waveform.frequency`` / ``waveform.peak`` (half the peak-to-peak excursion)
        within 1e-9 relative, otherwise ``InvalidInputError`` is raised (nothing is rescaled silently).

        ``reversal_tolerance`` (fraction of the peak-to-peak value, default 1e-9) is forwarded to
        :meth:`Waveform.count_reversals` for the minor-loop classification. Raising it above the default is
        recorded in the diagnostics and flagged (``reversal_tolerance_raised``) because it changes the loop
        count only, never the loss integral; a tolerance that suppresses every change of the waveform is
        rejected (``InvalidInputError``) since the classification is then meaningless.
        """
        _reject_unknown_kwargs(kwargs)
        policy = _validate_policy(policy)
        t = None if temperature is None else validate_real("temperature", temperature, positive=True)
        tol = validate_real("reversal_tolerance", reversal_tolerance, nonnegative=True)
        if waveform is None:
            if frequency is None or flux_density_peak is None:
                raise InvalidInputError("frequency and flux_density_peak are required when no waveform is given")
            f, b, _ = self._validate_conditions(frequency, flux_density_peak, None)
            wf = None if b == 0.0 else Waveform.sinusoidal(peak=b, frequency=f)
        else:
            if not isinstance(waveform, Waveform):
                raise InvalidInputError(f"waveform must be a Waveform, got {type(waveform).__name__}")
            wf = waveform
            f, b = wf.frequency, wf.peak
            if frequency is not None:
                given = validate_real("frequency", frequency, positive=True)
                if not math.isclose(given, f, rel_tol=1e-9, abs_tol=0.0):
                    raise InvalidInputError(
                        f"frequency={given:g} Hz does not match waveform.frequency={f:g} Hz; pass the waveform's "
                        "frequency or omit the argument", details={"frequency": given, "waveform_frequency": f})
            if flux_density_peak is not None:
                given_b = validate_real("flux_density_peak", flux_density_peak, nonnegative=True)
                if not math.isclose(given_b, b, rel_tol=1e-9, abs_tol=0.0):
                    raise InvalidInputError(
                        f"flux_density_peak={given_b:g} T does not match waveform.peak={b:g} T (half the "
                        "peak-to-peak excursion); pass the waveform's peak or omit the argument",
                        details={"flux_density_peak": given_b, "waveform_peak": b})

        zero = wf is None
        kind = "sinusoidal" if wf is None else wf.kind
        dc = 0.0 if wf is None else wf.dc_component
        has_dc = False if wf is None else abs(dc) > DC_BIAS_RELATIVE_THRESHOLD * wf.peak_to_peak
        reversals = 2 if wf is None else wf.count_reversals(tol)
        if wf is not None and reversals == 0:
            raise InvalidInputError(
                f"reversal_tolerance={tol:g} suppresses every dB/dt change of the waveform (no interval exceeds "
                f"{tol:g} x peak_to_peak), so minor loops cannot be classified; use a smaller tolerance or smooth / "
                "decimate the data instead", details={"reversal_tolerance": tol})
        minor = reversals > 2

        base = self.check_domain(frequency=f, temperature=t, flux_density_peak=None if zero else b, excitation=kind,
                                 dc_bias=has_dc)
        violations = list(base.violations)
        checked = dict(base.checked)
        checked["minor_loops"] = "unsupported (single major loop per period)"
        if minor:
            violations.append(f"waveform has {reversals} dB/dt reversals per period (minor loops); iGSE does not "
                              "treat minor loops")
        domain = DomainCheck(inside=not violations, violations=violations, checked=checked)
        if not domain.inside and policy == "reject":
            domain.raise_if_outside(self.identity.id)

        if zero:
            value, details = 0.0, {"integration_method": "none (zero excitation)", "delta_b": 0.0}
        elif kind == "sinusoidal":
            value, details = self._sinusoidal(f, b)
        elif wf is not None and wf.is_piecewise_linear:
            value, details = self._piecewise_linear(wf)
        else:
            assert wf is not None
            value, details = self._sampled(wf)

        output = self._new_output(domain, frequency=f, flux_density_peak=b, temperature=t, excitation=kind,
                                  policy=policy)
        output.values["loss_density"] = Quantity(value=value, unit="W/m^3")
        output.assumptions.append(Assumption(
            id="major_loop_delta_b",
            statement="DeltaB is the peak-to-peak flux density over the whole period (major loop).",
            basis="model_assumption", affects=["loss_density"]))
        output.diagnostics.update({
            "k_i": self.k_i, "waveform_kind": kind, "dc_component": dc, "dc_bias_present": has_dc,
            "dc_bias_threshold": DC_BIAS_RELATIVE_THRESHOLD, "n_reversals": reversals, "has_minor_loops": minor,
            "reversal_tolerance": tol,
            "waveform": None if wf is None else wf.describe(reversal_tolerance=tol),
        })
        output.diagnostics.update(details)
        if zero:
            self._mark_zero_excitation(output)
        if has_dc:
            output.warnings.append(Warning_(
                code="dc_bias_ignored",
                message=f"waveform has a DC component of {dc:g} T; iGSE ignores DC bias", severity="warning"))
        if minor and policy == "warn":
            if self.beta > self.alpha:
                relation, direction = ">", "overestimates"
            elif self.beta < self.alpha:
                relation, direction = "<", "underestimates"
            else:
                relation, direction = "==", "equals"
            output.warnings.append(Warning_(
                code="minor_loops_ignored",
                message=(f"waveform has {reversals} dB/dt reversals per period; minor loops are not decomposed and the "
                         f"major-loop peak-to-peak flux density DeltaB={details['delta_b']:g} T was applied to every "
                         f"segment, which {direction} the loop-decomposed iGSE result because beta {relation} alpha "
                         "(the two evaluations differ only through DeltaB^(beta-alpha) on the minor-loop segments, "
                         "whose own DeltaB is smaller)"),
                severity="warning"))
        if wf is not None and tol > DEFAULT_REVERSAL_TOLERANCE:
            output.warnings.append(Warning_(
                code="reversal_tolerance_raised",
                message=(f"dB/dt changes with |dB| <= {tol:g} x peak_to_peak were ignored when counting loop reversals "
                         f"(reversal_tolerance above the default {DEFAULT_REVERSAL_TOLERANCE:g}); the loss integral "
                         "still includes every change of the waveform, so this does not remove the effect of noise - "
                         "smooth or decimate measured data instead"),
                severity="warning"))
        n_samples = details.get("n_samples")
        if n_samples is not None and n_samples < SAMPLING_WARNING_THRESHOLD:
            output.warnings.append(Warning_(
                code="sampling_sensitivity",
                message=(f"sampled waveform has only {n_samples} samples per period (< {SAMPLING_WARNING_THRESHOLD}); "
                         "the finite-difference |dB/dt|^alpha integral may be sampling sensitive"),
                severity="warning"))
        return output

    # -- evaluation paths -------------------------------------------------------------------------
    def _sinusoidal(self, frequency: float, peak: float) -> tuple[float, dict[str, Any]]:
        k_i = self.k_i
        integral = abs_cos_power_integral(self.alpha)
        delta_b = 2.0 * peak
        try:
            value = (k_i * math.pow(_TWO_PI * frequency, self.alpha) * math.pow(peak, self.alpha)
                     * math.pow(delta_b, self.beta - self.alpha) * integral / _TWO_PI)
        except OverflowError:
            raise NumericalError(f"{self.MODEL_ID}: sinusoidal evaluation overflows for f={frequency:g} Hz, "
                                 f"Bpk={peak:g} T", details={"frequency": frequency, "flux_density_peak": peak}) from None
        return self.check_output_finite("loss_density", value), {
            "integration_method": "analytic (sinusoid, angular integral from cached quadrature)",
            "delta_b": delta_b}

    def _piecewise_linear(self, waveform: Waveform) -> tuple[float, dict[str, Any]]:
        k_i = self.k_i
        f = waveform.frequency
        delta_b = waveform.peak_to_peak
        points = waveform.breakpoints
        fractions = np.diff(points[:, 0])  # x_i = dt_i / T, exact from the breakpoints (equation (3))
        increments = np.diff(points[:, 1])
        if np.any(fractions <= 0.0):
            raise NumericalError(f"{self.MODEL_ID}: a piecewise-linear segment has zero duration (infinite dB/dt); "
                                 "the waveform cannot be evaluated", details={"frequency": f, "delta_b": delta_b})
        try:
            total = 0.0
            for x, db in zip(fractions.tolist(), increments.tolist()):
                if db == 0.0:
                    continue
                total += math.pow(abs(db), self.alpha) * math.pow(x, 1.0 - self.alpha)
            value = k_i * math.pow(delta_b, self.beta - self.alpha) * math.pow(f, self.alpha) * total
        except (OverflowError, ValueError, ZeroDivisionError) as error:
            raise NumericalError(f"{self.MODEL_ID}: piecewise-linear evaluation overflows for f={f:g} Hz, "
                                 f"DeltaB={delta_b:g} T ({error})", details={"frequency": f, "delta_b": delta_b}) from None
        return self.check_output_finite("loss_density", value), {
            "integration_method": "analytic per piecewise-linear segment", "delta_b": delta_b,
            "n_segments": int(fractions.size)}

    def _sampled(self, waveform: Waveform) -> tuple[float, dict[str, Any]]:
        k_i = self.k_i
        f = waveform.frequency
        period = 1.0 / f
        delta_b = waveform.peak_to_peak
        _, samples = waveform.samples()
        n = int(samples.size)
        dt = period / (n - 1)
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            try:
                slopes = np.abs(np.diff(samples)) / dt
                integral = float(np.sum(np.power(slopes, self.alpha)) * dt)  # Integral_0^T |dB/dt|^alpha dt, exact for the interpolant
                value = k_i * math.pow(delta_b, self.beta - self.alpha) * integral / period
            except (FloatingPointError, OverflowError):
                raise NumericalError(f"{self.MODEL_ID}: sampled evaluation overflows for f={f:g} Hz, "
                                     f"DeltaB={delta_b:g} T", details={"frequency": f, "delta_b": delta_b}) from None
        return self.check_output_finite("loss_density", value), {
            "integration_method": "exact integral of the piecewise-linear interpolant of the samples (forward "
                                  "differences on the uniform grid; midpoint-type rule on |dB/dt|^alpha)",
            "delta_b": delta_b, "n_samples": n, "sample_interval": dt}
