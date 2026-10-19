"""Closed-form magnetic design relations (SI, scalar, well documented).

Every function here is a pure function of SI floats, raises :class:`InvalidInputError` for
impossible inputs and never clips. The result-producing engineering capabilities
(:mod:`aipe_magnetics.engineering.transformer`, ``inductor`` ...) wrap these relations, attach units,
assumptions and sources, and never re-implement them.

Conventions
-----------
* ``frequency`` in Hz, lengths in m, areas in m^2, flux density in T, field in A/m, current in A,
  voltage in V, resistivity in Ohm*m, temperatures in K.
* A *square* voltage wave of amplitude ``V`` applied for a fraction ``duty`` of each half period
  (``duty = 0.5`` is a full 50 % square wave) produces a symmetric triangular/trapezoidal flux swing
  ``2*B_pk = V * duty * T / (N * A_e)`` (Faraday's law, no resistive drop).
* Fringing, leakage, winding capacitance and thermal effects are **not** modelled here.

References
----------
* Faraday's law and the transformer EMF equation: any standard text, e.g. W.G. Hurley, W.H. Wolfle,
  *Transformers and Inductors for Power Electronics*, Wiley 2013, ch. 1-2.
* Gapped-core inductance and energy method: Hurley & Wolfle 2013, ch. 2; C.W.T. McLyman,
  *Transformer and Inductor Design Handbook*, 4th ed., CRC 2011 (area product).
* Dowell, P.L., "Effects of eddy currents in transformer windings", Proc. IEE 113(8), 1966,
  pp. 1387-1394 (1-D foil-layer AC resistance factor).
* Copper resistivity: IACS reference 1.7241e-8 Ohm*m at 20 degC; commonly tabulated annealed copper
  1.68e-8 Ohm*m; temperature coefficient 0.00393 1/K near 20 degC.
"""

from __future__ import annotations

import math

from aipe_magnetics.common.errors import InvalidInputError
from aipe_magnetics.common.units import COPPER_RESISTIVITY_20C, COPPER_TEMPERATURE_COEFFICIENT, MU_0

SQRT2 = math.sqrt(2.0)


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInputError(f"{name} must be a real number")
    if not math.isfinite(value) or value <= 0:
        raise InvalidInputError(f"{name} must be a positive finite number, got {value}")
    return float(value)


def _nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInputError(f"{name} must be a real number")
    if not math.isfinite(value) or value < 0:
        raise InvalidInputError(f"{name} must be a non-negative finite number, got {value}")
    return float(value)


def _duty(duty: float) -> float:
    if not isinstance(duty, (int, float)) or not math.isfinite(duty) or not (0 < duty <= 0.5):
        raise InvalidInputError(f"duty must be in (0, 0.5], got {duty}")
    return float(duty)


# -------------------------------------------------------------------------------------------------
# Flux density <-> volt-seconds / turns
# -------------------------------------------------------------------------------------------------
def flux_density_peak_square(voltage_peak: float, frequency: float, turns: int, area: float, duty: float = 0.5) -> float:
    """Peak flux density for a bipolar square/trapezoidal voltage of amplitude ``voltage_peak``.

    ``B_pk = V * duty / (2 * f * N * A_e)``. With ``duty = 0.5``: ``B_pk = V / (4 f N A_e)``.
    """
    v, f, a = _positive("voltage_peak", voltage_peak), _positive("frequency", frequency), _positive("area", area)
    n = _turns(turns)
    return v * _duty(duty) / (2.0 * f * n * a)


def flux_density_peak_sine(voltage_rms: float, frequency: float, turns: int, area: float) -> float:
    """Transformer EMF equation: ``V_rms = sqrt(2) pi f N A_e B_pk`` (the 4.44 f N A_e B_pk form)."""
    v, f, a = _positive("voltage_rms", voltage_rms), _positive("frequency", frequency), _positive("area", area)
    return v / (SQRT2 * math.pi * f * _turns(turns) * a)


def turns_for_flux_density_square(voltage_peak: float, frequency: float, area: float, flux_density_peak: float,
                                  duty: float = 0.5) -> int:
    """Smallest integer turn count keeping the peak flux density at or below ``flux_density_peak``."""
    v, f, a, b = (_positive("voltage_peak", voltage_peak), _positive("frequency", frequency),
                  _positive("area", area), _positive("flux_density_peak", flux_density_peak))
    exact = v * _duty(duty) / (2.0 * f * a * b)
    return _ceil_turns(exact)


def turns_for_flux_density_sine(voltage_rms: float, frequency: float, area: float, flux_density_peak: float) -> int:
    v, f, a, b = (_positive("voltage_rms", voltage_rms), _positive("frequency", frequency),
                  _positive("area", area), _positive("flux_density_peak", flux_density_peak))
    return _ceil_turns(v / (SQRT2 * math.pi * f * a * b))


def flux_density_from_current(inductance: float, current: float, turns: int, area: float) -> float:
    """``B = L I / (N A_e)`` (flux linkage ``L I = N A_e B``); valid for a linear inductor."""
    return _positive("inductance", inductance) * _nonnegative("current", current) / (_turns(turns) * _positive("area", area))


def field_strength(turns: int, current: float, path_length: float) -> float:
    """Ampere's law around the mean magnetic path of an ungapped core: ``H = N I / l_e``."""
    return _turns(turns) * _nonnegative("current", current) / _positive("path_length", path_length)


# -------------------------------------------------------------------------------------------------
# Gapped inductors (reluctance model, no fringing)
# -------------------------------------------------------------------------------------------------
def gapped_inductance(turns: int, area: float, air_gap: float, path_length: float, relative_permeability: float) -> float:
    """``L = mu_0 N^2 A_e / (g + l_e / mu_r)`` - series reluctance of gap and core, fringing neglected."""
    n, a = _turns(turns), _positive("area", area)
    g, le, mur = _nonnegative("air_gap", air_gap), _positive("path_length", path_length), _positive("relative_permeability", relative_permeability)
    return MU_0 * n * n * a / (g + le / mur)


def air_gap_for_inductance(inductance: float, turns: int, area: float, path_length: float, relative_permeability: float) -> float:
    """Total air gap ``g = mu_0 N^2 A_e / L - l_e / mu_r``; raises when the ungapped core cannot reach ``L``."""
    L, n, a = _positive("inductance", inductance), _turns(turns), _positive("area", area)
    le, mur = _positive("path_length", path_length), _positive("relative_permeability", relative_permeability)
    gap = MU_0 * n * n * a / L - le / mur
    if gap < 0:
        raise InvalidInputError(
            f"requested inductance {L:g} H exceeds the ungapped inductance {MU_0 * n * n * a * mur / le:g} H for N={n}",
            details={"inductance": L, "turns": n})
    return gap


def turns_for_energy_limit(inductance: float, current_peak: float, flux_density_max: float, area: float) -> int:
    """Energy method: ``N = L I_pk / (B_max A_e)`` rounded up to an integer."""
    L, i, b, a = (_positive("inductance", inductance), _positive("current_peak", current_peak),
                  _positive("flux_density_max", flux_density_max), _positive("area", area))
    return _ceil_turns(L * i / (b * a))


def inductor_area_product(inductance: float, current_peak: float, current_rms: float, flux_density_max: float,
                          current_density: float, fill_factor: float) -> float:
    """Area product ``A_p = L I_pk I_rms / (B_max J k_u)`` (m^4), McLyman's inductor sizing relation."""
    L, ipk, irms = _positive("inductance", inductance), _positive("current_peak", current_peak), _positive("current_rms", current_rms)
    b, j, ku = _positive("flux_density_max", flux_density_max), _positive("current_density", current_density), _fill(fill_factor)
    return L * ipk * irms / (b * j * ku)


def transformer_area_product(apparent_power_total: float, frequency: float, flux_density_max: float,
                             current_density: float, fill_factor: float, waveform_factor: float) -> float:
    """``A_p = P_t / (K_f f B_max J k_u)`` with ``P_t`` the sum of winding VA and ``K_f`` 4.0 (square) or 4.44 (sine)."""
    pt, f = _positive("apparent_power_total", apparent_power_total), _positive("frequency", frequency)
    b, j, ku, kf = (_positive("flux_density_max", flux_density_max), _positive("current_density", current_density),
                    _fill(fill_factor), _positive("waveform_factor", waveform_factor))
    return pt / (kf * f * b * j * ku)


# -------------------------------------------------------------------------------------------------
# Windings: resistance and copper loss
# -------------------------------------------------------------------------------------------------
def copper_resistivity(temperature: float, rho_20: float = COPPER_RESISTIVITY_20C,
                       alpha: float = COPPER_TEMPERATURE_COEFFICIENT) -> float:
    """Linear temperature model ``rho(T) = rho_20 (1 + alpha (T - 293.15 K))``; valid roughly -50..200 degC."""
    t = _positive("temperature", temperature)
    rho = rho_20 * (1.0 + alpha * (t - 293.15))
    if rho <= 0:
        raise InvalidInputError(f"temperature {t} K is outside the linear resistivity model")
    return rho


def dc_resistance(resistivity: float, length: float, conductor_area: float) -> float:
    """``R = rho l / A``."""
    return _positive("resistivity", resistivity) * _positive("length", length) / _positive("conductor_area", conductor_area)


def skin_depth(frequency: float, resistivity: float, relative_permeability: float = 1.0) -> float:
    """``delta = sqrt(rho / (pi f mu_0 mu_r))``."""
    return math.sqrt(_positive("resistivity", resistivity) /
                     (math.pi * _positive("frequency", frequency) * MU_0 * _positive("relative_permeability", relative_permeability)))


def dowell_ac_resistance_factor(layers: int, layer_thickness: float, skin_depth_value: float, porosity: float = 1.0) -> float:
    """Dowell (1966) 1-D AC/DC resistance ratio of a winding portion with ``m`` layers.

    ``Delta = sqrt(eta) h / delta``;
    ``F_R = Delta [ (sinh 2Delta + sin 2Delta)/(cosh 2Delta - cos 2Delta)
                     + (2/3)(m^2 - 1)(sinh Delta - sin Delta)/(cosh Delta + cos Delta) ]``.

    ``h`` is the foil thickness (or the equivalent square-conductor height ``sqrt(pi)/2 d`` for round
    wire of diameter ``d``); ``eta`` is the layer porosity (conductor width / layer width).
    Assumptions: 1-D field, layers fully spanning the window height, sinusoidal current.
    """
    m = layers
    if isinstance(m, bool) or not isinstance(m, int) or m < 1:
        raise InvalidInputError(f"layers must be a positive integer, got {layers}")
    h, d = _positive("layer_thickness", layer_thickness), _positive("skin_depth", skin_depth_value)
    if not isinstance(porosity, (int, float)) or not (0 < porosity <= 1):
        raise InvalidInputError(f"porosity must be in (0, 1], got {porosity}")
    delta = math.sqrt(porosity) * h / d
    if delta > 700:  # hyperbolic overflow guard: report instead of silently saturating
        raise InvalidInputError(f"Dowell penetration ratio {delta:g} is outside the numerically valid range")
    s2, c2 = math.sinh(2 * delta), math.cosh(2 * delta)
    s1, c1 = math.sinh(delta), math.cosh(delta)
    term1 = (s2 + math.sin(2 * delta)) / (c2 - math.cos(2 * delta))
    term2 = (2.0 * (m * m - 1) / 3.0) * (s1 - math.sin(delta)) / (c1 + math.cos(delta))
    return delta * (term1 + term2)


def round_wire_equivalent_thickness(diameter: float) -> float:
    """Dowell equivalent square-conductor height for round wire: ``h = sqrt(pi)/2 * d``."""
    return math.sqrt(math.pi) / 2.0 * _positive("diameter", diameter)


def conductor_area_for_current_density(current_rms: float, current_density: float) -> float:
    """``A = I_rms / J``."""
    return _positive("current_rms", current_rms) / _positive("current_density", current_density)


def round_wire_diameter(area: float) -> float:
    return 2.0 * math.sqrt(_positive("area", area) / math.pi)


def round_wire_area(diameter: float) -> float:
    d = _positive("diameter", diameter)
    return math.pi * d * d / 4.0


def copper_loss(current_rms: float, resistance_ac: float) -> float:
    """``P = I_rms^2 R_ac``."""
    return _nonnegative("current_rms", current_rms) ** 2 * _nonnegative("resistance_ac", resistance_ac)


def window_fill(turns: int, conductor_area: float, window_area: float) -> float:
    """Fraction of the window occupied by bare conductor: ``N A_cu / A_w``."""
    return _turns(turns) * _positive("conductor_area", conductor_area) / _positive("window_area", window_area)


# -------------------------------------------------------------------------------------------------
# Core loss and screening helpers
# -------------------------------------------------------------------------------------------------
def core_loss_from_density(loss_density: float, effective_volume: float) -> float:
    """``P = P_v V_e``."""
    return _nonnegative("loss_density", loss_density) * _positive("effective_volume", effective_volume)


def saturation_margin(flux_density_peak: float, saturation_flux_density: float) -> float:
    """``1 - B_pk / B_sat`` (negative when saturated)."""
    return 1.0 - _nonnegative("flux_density_peak", flux_density_peak) / _positive("saturation_flux_density", saturation_flux_density)


def rms_of_square_current(current_peak: float, duty: float = 0.5) -> float:
    """RMS of a bipolar square current with amplitude ``I`` conducting ``2*duty`` of the period."""
    return _positive("current_peak", current_peak) * math.sqrt(2.0 * _duty(duty))


def rms_of_triangular_current(current_peak: float) -> float:
    """RMS of a symmetric triangular current of amplitude ``I_pk``: ``I_pk / sqrt(3)``."""
    return _positive("current_peak", current_peak) / math.sqrt(3.0)


def dab_primary_rms_current_square_approximation(power: float, voltage: float) -> float:
    """First-order DAB current estimate assuming square-wave voltage and current: ``I_rms ~ P / V``.

    The real DAB current is trapezoidal and depends on the phase shift and leakage inductance; this
    is an explicit *assumption* used for winding sizing in the basic design and must be reported as
    such.
    """
    return _positive("power", power) / _positive("voltage", voltage)


# -------------------------------------------------------------------------------------------------
# internal
# -------------------------------------------------------------------------------------------------
def _turns(turns: int) -> int:
    if isinstance(turns, bool) or not isinstance(turns, int) or turns < 1:
        raise InvalidInputError(f"turns must be a positive integer, got {turns!r}")
    return turns


def _fill(fill_factor: float) -> float:
    if not isinstance(fill_factor, (int, float)) or not (0 < fill_factor <= 1):
        raise InvalidInputError(f"fill_factor must be in (0, 1], got {fill_factor}")
    return float(fill_factor)


def _ceil_turns(exact: float) -> int:
    if not math.isfinite(exact) or exact <= 0:
        raise InvalidInputError(f"computed turn count {exact} is not a positive finite number")
    # tolerate floating-point noise just above an integer (e.g. 10.0000000001 -> 10)
    rounded = round(exact)
    if abs(exact - rounded) < 1e-9:
        return int(rounded)
    return int(math.ceil(exact))
