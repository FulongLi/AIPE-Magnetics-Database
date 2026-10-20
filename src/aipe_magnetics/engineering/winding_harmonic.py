"""Analytical winding-loss and leakage relations for layered windings with non-sinusoidal currents.

These are the *analytical* references that the finite-element results are compared with:

* :func:`dowell_harmonic_loss` - Dowell's 1-D layer model evaluated per current harmonic,
  ``P = R_dc (I_dc^2 + sum_n I_n,rms^2 F_R(n f))`` (Dowell 1966; harmonic superposition as in
  Hurley, Gath & Breslin, IEEE Trans. Power Electron. 15(2), 2000, "Optimizing the AC resistance of multilayer
  transformer windings with arbitrary current waveforms").
* :func:`leakage_inductance_1d` - energy of the 1-D window field ``H(x) = F(x)/h_w`` across the radial build of
  a layered winding stack with ampere-turn balance: ``L = mu_0 MLT / (h_w I^2) integral F(x)^2 dx`` (exact for the
  1-D field model; reduces to the classical ``mu_0 N^2 MLT/h_w (a_1/3 + a_2/3 + delta)`` for two windings).
* :func:`litz_strand_losses` - strand-level skin factor ``1 + (r/delta)^4/48`` and the low-frequency proximity
  loss per unit length of a round strand in a uniform transverse field, ``P' = pi sigma omega^2 B^2 d^4 / 128``
  (peak ``B``; time-averaged), used by the field-based litz post-processing (Sullivan, IEEE Trans. Power Electron.
  14(2), 1999).
"""

from __future__ import annotations

import math
from typing import Iterable

from aipe_magnetics.common.units import MU_0
from aipe_magnetics.engineering import formulas as F

HURLEY_2000 = ("W.G. Hurley, E. Gath, J.G. Breslin, 'Optimizing the AC resistance of multilayer transformer windings with arbitrary "
               "current waveforms', IEEE Trans. Power Electron. 15(2), 2000, pp. 369-376")


def dowell_layer_factor(layers: int, conductor_kind: str, size: float, pitch: float, skin_depth: float) -> float:
    """Dowell factor for a winding portion: ``size`` = foil thickness or bare round diameter, ``pitch`` = axial pitch."""
    if conductor_kind == "foil":
        return F.dowell_ac_resistance_factor(layers, size, skin_depth, 1.0)
    h = F.round_wire_equivalent_thickness(size)
    porosity = min(1.0, h / pitch)
    return F.dowell_ac_resistance_factor(layers, h, skin_depth, porosity)


def dowell_harmonic_loss(*, r_dc: float, dc: float, harmonics: Iterable[tuple[int, float]], frequency: float, temperature: float,
                         layers: int, conductor_kind: str, size: float, pitch: float) -> dict[str, float]:
    """``harmonics``: ``(n, peak amplitude)``. Returns total loss, DC part, AC part and the effective AC factor."""
    rho = F.copper_resistivity(temperature)
    p_dc = r_dc * dc * dc
    p_ac = 0.0
    i_ac2 = 0.0
    for n, amp in harmonics:
        if amp <= 0:
            continue
        delta = F.skin_depth(n * frequency, rho)
        fr = dowell_layer_factor(layers, conductor_kind, size, pitch, delta)
        p_ac += r_dc * (amp * amp / 2.0) * fr
        i_ac2 += amp * amp / 2.0
    return {"loss_W": p_dc + p_ac, "dc_loss_W": p_dc, "ac_loss_W": p_ac,
            "effective_ac_factor": (p_ac / (r_dc * i_ac2)) if i_ac2 > 0 else 1.0}


def leakage_inductance_1d(layers: list[dict], window_height: float, mean_turn_length: float) -> float:
    """1-D window-field leakage inductance referred to the winding with ``reference_turns`` (H).

    ``layers``: radially ordered ``{"x0", "x1", "ampere_turns"}`` (ampere-turns of the layer for 1 A in the reference
    winding, signed; the stack must be ampere-turn balanced). Gaps between layers are field regions with constant MMF.
    """
    total = 0.0
    mmf = 0.0
    x_prev = None
    for layer in layers:
        x0, x1, at = layer["x0"], layer["x1"], layer["ampere_turns"]
        if x_prev is not None and x0 > x_prev:
            total += mmf * mmf * (x0 - x_prev)
        # linear rise of the MMF across the layer: integral of (mmf + at*s/w)^2 ds
        w = x1 - x0
        total += w * (mmf * mmf + mmf * at + at * at / 3.0)
        mmf += at
        x_prev = x1
    # W = mu_0 MLT / (2 h_w) integral F^2 dx  ->  L = 2 W / (1 A)^2
    return MU_0 * mean_turn_length / window_height * total


def litz_strand_skin_factor(strand_diameter: float, skin_depth: float) -> float:
    """``R_ac/R_dc = 1 + (r/delta)^4 / 48`` (isolated round strand, low-frequency expansion, r = d/2)."""
    r = strand_diameter / 2.0
    return 1.0 + (r / skin_depth) ** 4 / 48.0


def strand_proximity_loss_per_length(strand_diameter: float, conductivity: float, omega: float, b_peak: float) -> float:
    """``pi sigma omega^2 B^2 d^4 / 128`` (W/m) for a round strand in a uniform transverse field of amplitude ``b_peak``."""
    return math.pi * conductivity * omega * omega * b_peak * b_peak * strand_diameter ** 4 / 128.0


def dowell_layer_resistance_factor(m: float, delta_ratio: float) -> float:
    """Dowell factor of one layer whose outer-face MMF is ``m`` times its own ampere-turns (inner face ``m - 1``).

    ``F = D [ (sinh 2D + sin 2D)/(cosh 2D - cos 2D) + 2 m (m - 1) (sinh D - sin D)/(cosh D + cos D) ]`` with ``D`` the
    porosity-corrected penetration ratio. Averaging over ``m = 1..M`` gives the classical portion formula; ``m`` may be
    any real number (interleaved windings, fractional MMF steps).
    """
    d = delta_ratio
    if d > 700:
        raise ValueError(f"penetration ratio {d:g} outside the numerically valid range")
    s1 = (math.sinh(2 * d) + math.sin(2 * d)) / (math.cosh(2 * d) - math.cos(2 * d))
    s2 = (math.sinh(d) - math.sin(d)) / (math.cosh(d) + math.cos(d))
    return d * (s1 + 2.0 * m * (m - 1.0) * s2)


def dowell_layer_stack_loss(stack: list[dict], harmonics: Iterable[tuple[int, float]], frequency: float, temperature: float,
                            dc: float = 0.0) -> dict[str, float]:
    """Per-layer Dowell loss of a radially ordered, ampere-turn balanced layer stack (W per winding).

    ``stack`` entries: ``{"winding", "ampere_turns" (per 1 A reference, signed), "turns", "r_dc" (Ohm, layer), "kind",
    "size" (foil thickness / bare diameter), "pitch" (axial pitch; foil: winding height)}``. Each layer's ``m`` follows from
    the MMF at its faces; ``harmonics`` are ``(n, peak amplitude of the reference current)``.
    """
    rho = F.copper_resistivity(temperature)
    harmonics = list(harmonics)
    out: dict[str, float] = {}
    mmf = 0.0
    for layer in stack:
        at = layer["ampere_turns"]
        f_in, f_out = mmf, mmf + at
        mmf = f_out
        if at == 0:
            continue
        m = f_out / at
        i_per_turn = abs(at) / layer["turns"]
        if layer["kind"] == "foil":
            h, porosity = layer["size"], 1.0
        else:
            h = F.round_wire_equivalent_thickness(layer["size"])
            porosity = min(1.0, h / layer["pitch"])
        loss = layer["r_dc"] * (dc * i_per_turn) ** 2
        for n, amp in harmonics:
            delta = F.skin_depth(n * frequency, rho)
            factor = dowell_layer_resistance_factor(m, math.sqrt(porosity) * h / delta)
            loss += layer["r_dc"] * (amp * i_per_turn) ** 2 / 2.0 * factor
        out[layer["winding"]] = out.get(layer["winding"], 0.0) + loss
        del f_in
    return out


def litz_proximity_loss_1d(layers: list[dict], window_height: float, harmonics: Iterable[tuple[int, float]], frequency: float,
                           temperature: float) -> float:
    """1-D analytical litz proximity loss of a balanced layer stack (W).

    ``layers``: radially ordered ``{"x0", "x1", "ampere_turns", "turns", "strands", "strand_diameter", "mlt"}`` with ampere-turns
    per 1 A (peak) of the reference current; ``harmonics``: ``(n, peak amplitude of the reference current)``. Every strand of a
    layer sees the layer's mean square 1-D field ``<H^2> = (F0^2 + F0 dF + dF^2/3) / h_w^2``.
    """
    sigma = 1.0 / F.copper_resistivity(temperature)
    harmonics = list(harmonics)
    total = 0.0
    mmf = 0.0
    for layer in layers:
        f0, df = mmf, layer["ampere_turns"]
        mean_f2 = f0 * f0 + f0 * df + df * df / 3.0
        mmf += df
        if not layer.get("strands"):
            continue
        strand_length = layer["turns"] * layer["strands"] * layer["mlt"]
        for n, amp in harmonics:
            omega = 2 * math.pi * n * frequency
            b2 = (MU_0 * amp) ** 2 * mean_f2 / window_height ** 2
            total += strand_length * math.pi * sigma * omega * omega * b2 * layer["strand_diameter"] ** 4 / 128.0
    return total


__all__ = ["HURLEY_2000", "dowell_layer_resistance_factor", "dowell_layer_stack_loss", "litz_proximity_loss_1d", "dowell_harmonic_loss", "dowell_layer_factor", "leakage_inductance_1d", "litz_strand_skin_factor",
           "strand_proximity_loss_per_length"]
