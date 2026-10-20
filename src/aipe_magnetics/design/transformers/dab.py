"""Dual-active-bridge (DAB) operating waveforms under single-phase-shift (SPS) modulation.

The 800 V / 400 V rating alone does not define the transformer waveforms. This model makes the missing
choices explicit and computes the transformer excitation from them:

* both bridges produce 50 % bipolar square voltages ``+-V1`` and ``+-V2``; the secondary bridge lags by ``phi``;
* the total series inductance ``L`` (transformer leakage + external inductor), referred to the primary, carries
  ``i_L(t)`` with ``L di/dt = v_1(t) - n v_2(t - phi)``, ``n = N1/N2``; dead times and the magnetising current
  are neglected in ``i_L`` (the magnetising current is added separately by the electromagnetic study);
* transferred power ``P = n V1 V2 phi (pi - |phi|) / (2 pi^2 f L)`` (lossless SPS relation), so a rated phase
  shift ``phi_rated`` fixes ``L``, or a given ``L`` fixes ``phi`` for every operating point;
* operating points over a voltage range give the RMS / peak currents used for winding sizing (worst case) and
  report soft-switching (ZVS) indications from the switching-instant currents.

Reference: R.W.A.A. De Doncker, D.M. Divan, M.H. Kheraluwala, "A three-phase soft-switched high-power-density
DC/DC converter for high-power applications", IEEE Trans. Ind. Appl. 27(1), 1991; M.N. Kheraluwala et al.,
"Performance characterization of a high-power dual active bridge DC-to-DC converter", IEEE Trans. Ind. Appl.
28(6), 1992 (SPS power equation and current waveforms).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from aipe_magnetics.common.errors import InvalidInputError
from aipe_magnetics.engineering.periodic import PiecewiseLinear


def sps_power(v1: float, v2: float, n: float, frequency: float, inductance: float, phi: float) -> float:
    return n * v1 * v2 * phi * (math.pi - abs(phi)) / (2 * math.pi ** 2 * frequency * inductance)


def inductance_for_phase_shift(power: float, v1: float, v2: float, n: float, frequency: float, phi: float) -> float:
    if not (0 < phi <= math.pi / 2):
        raise InvalidInputError("rated phase shift must be in (0, pi/2] rad")
    return n * v1 * v2 * phi * (math.pi - phi) / (2 * math.pi ** 2 * frequency * power)


def phase_shift_for_power(power: float, v1: float, v2: float, n: float, frequency: float, inductance: float) -> float:
    p_max = sps_power(v1, v2, n, frequency, inductance, math.pi / 2)
    if power > p_max * (1 + 1e-12):
        raise InvalidInputError(f"power {power:g} W exceeds the SPS maximum {p_max:g} W for L = {inductance:g} H at V1={v1:g}, V2={v2:g}")
    # P(phi) monotonic on [0, pi/2]: solve phi (pi - phi) = c
    c = power * 2 * math.pi ** 2 * frequency * inductance / (n * v1 * v2)
    return 0.5 * (math.pi - math.sqrt(max(math.pi ** 2 - 4 * c, 0.0)))


def inductor_current(v1: float, v2: float, n: float, frequency: float, inductance: float, phi: float) -> PiecewiseLinear:
    """Series-inductor (= transformer primary load) current over one period, exact piecewise-linear."""
    T = 1.0 / frequency
    s = phi / (2 * math.pi)
    if not (0 <= s <= 0.25):
        raise InvalidInputError("phi must be in [0, pi/2]")
    vs = n * v2
    # intervals (fractions of T): [0, s): v1 - (-vs); [s, 0.5): v1 - vs; [0.5, 0.5+s): -v1 - vs; [0.5+s, 1): -v1 + vs
    times = [0.0, s, 0.5, 0.5 + s, 1.0]
    slopes = [v1 + vs, v1 - vs, -v1 - vs, -v1 + vs]
    rel = [0.0]
    for (t0, t1), k in zip(zip(times, times[1:]), slopes):
        rel.append(rel[-1] + k * (t1 - t0) * T / inductance)
    i0 = -0.5 * rel[2]  # half-wave antisymmetry: i(T/2) = -i(0)
    values = [i0 + r for r in rel]
    values[-1] = values[0]
    pts = [(t, v) for t, v in zip(times, values)]
    # drop duplicate time points (phi = 0)
    clean = [pts[0]] + [p for a, p in zip(pts, pts[1:]) if p[0] > a[0]]
    return PiecewiseLinear(points=tuple(clean), unit="A", label="DAB series-inductor current (primary)")


@dataclass
class DABOperatingPoint:
    v1: float
    v2: float
    power: float
    phi: float
    current: PiecewiseLinear
    label: str

    def summary(self, frequency: float, n: float) -> dict[str, Any]:
        i = self.current
        s = self.phi / (2 * math.pi)
        i_sw_primary = float(i.value(0.0))       # current when the primary bridge switches (t = 0)
        i_sw_secondary = float(i.value(s))       # current when the secondary bridge switches (t = phi/2pi)
        return {"label": self.label, "v1_V": self.v1, "v2_V": self.v2, "power_W": self.power, "phase_shift_rad": self.phi,
                "phase_shift_deg": math.degrees(self.phi), "voltage_conversion_ratio": n * self.v2 / self.v1,
                "primary_current_rms_A": i.rms(), "primary_current_peak_A": i.peak(),
                "secondary_current_rms_A": n * i.rms(), "secondary_current_peak_A": n * i.peak(),
                "primary_switching_current_A": i_sw_primary, "secondary_switching_current_A": n * i_sw_secondary,
                # ZVS indication (lossless, ideal switches): primary needs i(0) < 0, secondary needs i(phi) > 0
                "zvs_primary_indicated": i_sw_primary < 0, "zvs_secondary_indicated": i_sw_secondary > 0}


def operating_points(*, power: float, v1: float, v2_nominal: float, v2_range: tuple[float, float] | None, n: float, frequency: float,
                     inductance: float) -> list[DABOperatingPoint]:
    out = []
    v2s = [("nominal", v2_nominal)]
    if v2_range is not None:
        v2s += [("V2 min", v2_range[0]), ("V2 max", v2_range[1])]
    for label, v2 in v2s:
        phi = phase_shift_for_power(power, v1, v2, n, frequency, inductance)
        out.append(DABOperatingPoint(v1=v1, v2=v2, power=power, phi=phi, current=inductor_current(v1, v2, n, frequency, inductance, phi),
                                     label=label))
    return out


__all__ = ["DABOperatingPoint", "inductance_for_phase_shift", "inductor_current", "operating_points", "phase_shift_for_power", "sps_power"]
