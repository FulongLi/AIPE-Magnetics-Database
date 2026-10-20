"""Gapped high-frequency inductor: candidates, analytical evaluation, FEA gap tuning and detailed evaluation.

Analytical fidelity: centre-gap length solved for the target inductance with the McLyman fringing factor
``L = mu_0 N^2 A_e F(g) / (g + l_e/mu_r)``, ``F = 1 + g/sqrt(A_e) ln(2 G / g)`` (G = window height); peak flux density
``L I_pk / (N A_e)``; iGSE core loss of the triangular ripple flux on ``V_e``; Dowell per harmonic of the ripple plus the
DC loss; thermal network with electrothermal fixed point. Fringing-field proximity loss in the winding is *not* in the
analytical model (flagged); the FEA includes it.

Detailed fidelity: the gap is re-tuned until the 2-D FEA inductance (energy method, fringing included) matches the target
(secant iteration, 1 % tolerance); then the full inductor study (saturation with the assumed nonlinear curve, field-based
core loss, harmonic eddy-current winding loss including the gap fringing field) and the 2.5-D thermal FE are coupled
electrothermally, followed by a final re-solve at the converged temperatures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aipe_magnetics.common.units import MU_0
from aipe_magnetics.database import CoreData
from aipe_magnetics.design.common import MaterialContext, electrothermal, pack_conductivities, resistivity_ratio, thermal_conductivity_map
from aipe_magnetics.design.specs import InductorSpec
from aipe_magnetics.engineering import formulas as F
from aipe_magnetics.engineering.winding_harmonic import dowell_harmonic_loss, litz_strand_skin_factor
from aipe_magnetics.database import FluxShape
from aipe_magnetics.geometry.conductors import ConductorSpec, litz_for_area
from aipe_magnetics.geometry.ecore import BobbinSpec, ECoreDimensions, ECoreMagnetic, GapSpec, WindingLayout, WindingSection
from aipe_magnetics.simulation.thermal.network import solve_thermal_network

RESIDUAL_SPACER = 10e-6
MAX_GAP_FRACTION_OF_LEG = 0.3


@dataclass
class InductorCandidate:
    material: MaterialContext
    core: CoreData
    turns: int
    conductor: ConductorSpec
    frequency: float
    gap: float | None = None          # centre grind (m); None = solved analytically
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return (f"{self.core.name.replace(' ', '')}|{self.material.data.grade or self.material.id}|N{self.turns}|{self.conductor.describe()}|"
                f"{self.frequency / 1e3:g}kHz" + (f"|g{self.gap * 1e3:.3f}mm" if self.tags.get("gap_fixed") else ""))

    def params(self) -> dict[str, Any]:
        return {"material": self.material.id, "core": self.core.id, "turns": self.turns, "conductor": self.conductor.describe(),
                "conductor_kind": self.conductor.kind, "frequency_Hz": self.frequency, "gap_m": self.gap, **self.tags}

    def model(self, gap: float, name: str = "gapped-inductor") -> ECoreMagnetic:
        return ECoreMagnetic(name=name, core=ECoreDimensions.from_core_data(self.core), core_material=self.material.id,
                             gap=GapSpec(spacer=RESIDUAL_SPACER, centre_grind=max(gap - RESIDUAL_SPACER, 0.0), note="ground centre gap"),
                             bobbin=BobbinSpec(), layout=WindingLayout(sections=[WindingSection(winding="L", turns=self.turns, conductor=self.conductor)],
                                                                       arrangement="L"))


def fringing_factor(gap: float, area: float, window_height: float) -> float:
    return 1.0 + gap / math.sqrt(area) * math.log(2.0 * window_height / gap)


def inductance_analytic(turns: int, gap: float, core: CoreData, mu_r: float, window_height: float, spacer: float = RESIDUAL_SPACER) -> float:
    """McLyman: centre gap with fringing factor plus the outer-leg spacer (same area assumption) and the core path."""
    ae, le = core.effective_area, core.effective_length
    g_total = gap + spacer  # outer-leg spacer in series (outer legs ~ same total area as the centre leg)
    return MU_0 * turns ** 2 * ae * fringing_factor(gap, ae, window_height) / (g_total + le / mu_r)


def solve_gap_analytic(target: float, turns: int, core: CoreData, mu_r: float, window_height: float) -> float | None:
    lo, hi = 1e-6, 0.5 * window_height
    f_lo = inductance_analytic(turns, lo, core, mu_r, window_height) - target
    f_hi = inductance_analytic(turns, hi, core, mu_r, window_height) - target
    if f_lo < 0 or f_hi > 0:
        return None
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if inductance_analytic(turns, mid, core, mu_r, window_height) > target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def _conductors(spec: InductorSpec, i_rms: float) -> list[tuple[ConductorSpec, dict]]:
    sp = spec.space
    out = []
    for kind in sp.conductors:
        if kind == "round":
            for d in sp.round_diameters:
                for p in sp.parallel_round:
                    out.append((ConductorSpec(kind="round", bare_diameter=d, parallel=p), {"round_mm": d * 1e3, "parallel": p}))
        elif kind == "foil":
            for t in sp.foil_thicknesses:
                out.append((ConductorSpec(kind="foil", thickness=t), {"foil_mm": t * 1e3}))
        elif kind == "litz":
            for d in sp.litz_strand_diameters:
                for j in sp.current_densities:
                    out.append((litz_for_area(i_rms / j, d), {"strand_mm": d * 1e3, "J_design_A_per_mm2": j / 1e6}))
    return out


def generate_candidates(spec: InductorSpec, materials: list[MaterialContext], cores: list[CoreData]) -> list[InductorCandidate]:
    freqs = spec.space.frequencies or [spec.frequency]
    out = []
    for m in materials:
        bsat, _ = m.bsat(373.15)
        b_allow = spec.limits.flux_fraction_of_bsat * bsat
        for c in cores:
            if m.id not in c.material_refs and spec.space.cores is None and spec.space.material_offer_policy == "catalogue":
                continue
            for f in freqs:
                wave = spec.current_waveform(f)
                i_pk, i_rms = wave.peak(), wave.rms()
                n_min = F.turns_for_energy_limit(spec.inductance, i_pk, b_allow, c.effective_area)
                for n in range(n_min, n_min + spec.space.turn_steps + 1):
                    for cond, tags in _conductors(spec, i_rms):
                        out.append(InductorCandidate(material=m, core=c, turns=n, conductor=cond, frequency=f, tags=dict(tags)))
    return out


def _winding_loss_analytic(geo: Any, cand: InductorCandidate, wave, temperature: float) -> dict[str, float]:
    layers = [l for l in geo.layers if l.winding == "L"]
    cond = layers[0].conductor  # placed conductor (foil width resolved by the layout)
    rho = F.copper_resistivity(temperature)
    length = sum(l.turns * l.mean_turn_length for l in layers)
    r_dc = rho * length / cond.copper_area()
    harm = [(n, a) for n, a, _ in wave.harmonic_amplitudes(41) if a > 1e-9]
    dc = wave.mean()
    if cond.kind in ("round", "foil"):
        size = cond.bare_diameter if cond.kind == "round" else cond.thickness
        pitch = cond.axial_pitch() if cond.kind == "round" else geo.model.winding_space()["height"]
        d = dowell_harmonic_loss(r_dc=r_dc, dc=dc, harmonics=harm, frequency=cand.frequency, temperature=temperature, layers=len(layers),
                                 conductor_kind=cond.kind, size=size, pitch=pitch)
        return {"loss_W": d["loss_W"], "dc_loss_W": d["dc_loss_W"], "ac_loss_W": d["ac_loss_W"], "r_dc_Ohm": r_dc, "method": "dowell_harmonic"}
    ac = sum(r_dc * a * a / 2 * litz_strand_skin_factor(cond.strand_diameter, F.skin_depth(n * cand.frequency, rho)) for n, a in harm)
    return {"loss_W": r_dc * dc * dc + ac, "dc_loss_W": r_dc * dc * dc, "ac_loss_W": ac, "r_dc_Ohm": r_dc,
            "method": "litz_dc_skin_only (proximity not in the analytical model)"}


def evaluate_analytic(cand: InductorCandidate, spec: InductorSpec) -> dict[str, Any]:
    lim = spec.limits
    violations: list[str] = []
    metrics: dict[str, Any] = {}
    mu = cand.material.data.initial_permeability
    dims = ECoreDimensions.from_core_data(cand.core)
    probe = cand.model(1e-3)
    wh = 2 * dims.window_height_half
    gap = cand.gap if cand.gap is not None else solve_gap_analytic(spec.inductance, cand.turns, cand.core, mu, wh)
    if gap is None:
        return {"feasible": False, "violations": ["gap: no gap gives the target inductance with this turn count"], "objectives": {},
                "metrics": metrics, "fidelity": "analytical"}
    metrics["gap_m"] = gap
    metrics["fringing_factor_mclyman"] = fringing_factor(gap, cand.core.effective_area, wh)
    if gap > MAX_GAP_FRACTION_OF_LEG * dims.centre_leg_width:
        violations.append(f"gap: {gap * 1e3:.2f} mm exceeds {MAX_GAP_FRACTION_OF_LEG:.0%} of the centre-leg width")
    geo = cand.model(gap).build()
    del probe
    for chk in geo.checks:
        if not chk.passed:
            violations.append(f"geometry: {chk.message}")
    if any(v.startswith("geometry") for v in violations):
        return {"feasible": False, "violations": violations, "objectives": {}, "metrics": metrics, "geometry": geo, "fidelity": "analytical"}
    wave = spec.current_waveform(cand.frequency)
    i_pk, i_rms, ripple = wave.peak(), wave.rms(), wave.peak_to_peak()
    ae, ve = cand.core.effective_area, cand.core.effective_volume
    L = spec.inductance
    b_pk = L * i_pk / (cand.turns * ae)
    b_ac = L * ripple / (2 * cand.turns * ae)
    shape = FluxShape.triangular(spec.duty_cycle())
    metrics.update({"current_peak_A": i_pk, "current_rms_A": i_rms, "ripple_A": ripple, "flux_density_peak_T": b_pk,
                    "flux_density_ac_amplitude_T": b_ac})
    kp = pack_conductivities()
    cooling = spec.cooling.to_cooling()

    def losses(tc: float, tw: float):
        ev = cand.material.loss.loss_density(cand.frequency, b_ac, tc, shape, policy="warn")
        wl = _winding_loss_analytic(geo, cand, wave, tw)
        losses.last = {"core_sets": ev.sets, "core_warnings": ev.warnings, "winding": wl}  # type: ignore[attr-defined]
        return float(ev.loss_density) * ve, {"L": wl["loss_W"]}

    def thermal(pc: float, pw: dict[str, float]):
        res = solve_thermal_network(geo, core_loss=pc, winding_losses=pw, cooling=cooling, k_core=cand.material.thermal_conductivity(),
                                    k_bobbin=0.3, k_insulation=0.12, k_pack=kp)
        return res.temperatures["core"], max(v for k, v in res.temperatures.items() if k != "core"), res

    et = electrothermal(losses, thermal, 353.15, 353.15)
    pc, pw = losses(et.temperatures["core"], et.temperatures["winding"])
    detail = losses.last  # type: ignore[attr-defined]
    net = et.thermal
    metrics.update({"core_loss_W": pc, "winding_loss_W": pw["L"], "total_loss_W": pc + pw["L"], "winding_detail": detail["winding"],
                    "loss_model_warnings": sorted(set(detail["core_warnings"])), "thermal": net.summary(),
                    "electrothermal_converged": et.converged})
    bsat, clamped = cand.material.bsat(et.temperatures["core"])
    metrics.update({"bsat_T": bsat, "bsat_clamped": clamped})
    if b_pk > lim.flux_fraction_of_bsat * bsat:
        violations.append(f"saturation: B_pk {b_pk:.3f} T > {lim.flux_fraction_of_bsat:.0%} of Bsat {bsat:.3f} T")
    j = i_rms / cand.conductor.copper_area(geo.model.winding_space()["height"])
    metrics["current_density_A_per_mm2"] = j / 1e6
    if j > lim.max_current_density:
        violations.append(f"winding: current density {j / 1e6:.2f} A/mm^2 > {lim.max_current_density / 1e6:.1f}")
    fill = geo.derived["window_fill_bare_copper"]
    metrics["window_fill"] = fill
    if fill > lim.max_window_fill:
        violations.append(f"window fill {fill:.2f} > {lim.max_window_fill:.2f}")
    hs_w = max(v for k, v in net.hotspots.items() if k != "core")
    hs_c = net.hotspots["core"]
    metrics.update({"hotspot_winding_K": hs_w, "hotspot_core_K": hs_c})
    if not et.converged:
        violations.append("electrothermal iteration did not converge (thermal runaway or > 700 K)")
    if hs_w > lim.max_winding_hotspot:
        violations.append(f"winding hot spot {hs_w - 273.15:.0f} degC > {lim.max_winding_hotspot - 273.15:.0f} degC")
    if hs_c > lim.max_core_hotspot:
        violations.append(f"core hot spot {hs_c - 273.15:.0f} degC > {lim.max_core_hotspot - 273.15:.0f} degC")
    core_mass = cand.core.mass or cand.material.density() * geo.derived["core_volume_m3"]
    copper_mass = 8933.0 * geo.derived["copper_volume_m3"]
    objectives = {"total_loss_W": pc + pw["L"], "volume_cm3": geo.derived["bounding_volume_m3"] * 1e6, "mass_g": (core_mass + copper_mass) * 1e3,
                  "hotspot_C": max(hs_w, hs_c) - 273.15}
    return {"feasible": not violations, "violations": violations, "objectives": objectives, "metrics": metrics, "geometry": geo,
            "fidelity": "analytical", "gap": gap}


def fea_inductance(geo: Any, mu_r: float, material_id: str, settings: Any = None) -> float:
    from aipe_magnetics.simulation.electromagnetic.magnetostatic import EMProblem, solve_magnetostatic
    from aipe_magnetics.simulation.mesh import MeshSettings, generate_mesh
    st = settings or MeshSettings(conductor=1.0e-3, stranded=1.0e-3, core=min(1.5e-3, geo.model.core.centre_leg_width / 12))
    mesh = generate_mesh(geo.section, st)
    sol = solve_magnetostatic(EMProblem(mesh=mesh, mu_r={material_id: mu_r}, currents={"L": 1.0}))
    return 2 * sol.energy_3d(geo.section.winding_length["L"])["total_J"]


def tune_gap_fea(cand: InductorCandidate, target: float, gap0: float, mu_r: float, *, tol: float = 0.01, max_iterations: int = 8) -> dict[str, Any]:
    """Secant iteration on the centre gap so that the FEA inductance matches ``target`` within ``tol``."""
    history = []
    g_prev, l_prev = None, None
    g = gap0
    for it in range(1, max_iterations + 1):
        geo = cand.model(g).build()
        L = fea_inductance(geo, mu_r, cand.material.id)
        history.append({"iteration": it, "gap_m": g, "inductance_fea_H": L, "relative_error": L / target - 1})
        if abs(L / target - 1) < tol:
            return {"gap_m": g, "inductance_fea_H": L, "converged": True, "history": history, "geometry": geo}
        if g_prev is None:
            g_new = g * (L / target)  # first step: L ~ 1/g
        else:
            # secant on 1/L (closer to linear in g)
            s = (1 / L - 1 / l_prev) / (g - g_prev)
            g_new = g + (1 / target - 1 / L) / s if s != 0 else g * (L / target)
        g_prev, l_prev = g, L
        g = float(min(max(g_new, 0.2 * g), 5 * g, 0.45 * cand.model(g).core.centre_leg_width))
    return {"gap_m": g, "inductance_fea_H": L, "converged": False, "history": history, "geometry": cand.model(g).build()}


def evaluate_detailed(cand: InductorCandidate, spec: InductorSpec, *, analytic: dict[str, Any] | None = None, em_settings: Any = None) -> dict[str, Any]:
    from aipe_magnetics.simulation.electromagnetic.analysis import CoreModel, EMSettings, analyse_inductor
    from aipe_magnetics.simulation.electromagnetic.materials import SaturationCurve
    from aipe_magnetics.simulation.thermal.fem2d import solve_thermal_fem

    analytic = analytic or evaluate_analytic(cand, spec)
    mu = cand.material.data.initial_permeability
    tuned = tune_gap_fea(cand, spec.inductance, analytic.get("gap") or analytic["metrics"]["gap_m"], mu)
    geo = tuned["geometry"]
    wave = spec.current_waveform(cand.frequency)
    shape = FluxShape.triangular(spec.duty_cycle())
    settings = em_settings or EMSettings()
    th0 = analytic["metrics"]["thermal"]["temperatures_K"]
    t_core0, t_w0 = th0["core"], max(v for k, v in th0.items() if k != "core")

    def core_model(t: float) -> CoreModel:
        bsat, _ = cand.material.bsat(t)
        sat_pts = [p for p in cand.material.data.saturation_points if not math.isnan(p[0])]
        b_sat_curve, h_sat = (min(sat_pts, key=lambda p: abs(p[0] - t))[1], min(sat_pts, key=lambda p: abs(p[0] - t))[2] or 1200.0) if sat_pts else (bsat, 1200.0)
        curve = SaturationCurve(mu_i=mu, b_sat=b_sat_curve, h_sat=h_sat)
        return CoreModel(material_id=cand.material.id, mu_r=mu, bsat=bsat, loss=cand.material.loss, shape=shape, temperature=t, curve=curve)

    rep0 = analyse_inductor(geo, frequency=cand.frequency, core=core_model(t_core0), current=wave, winding_temperature=t_w0, settings=settings,
                            winding="L", datasheet_area=cand.core.effective_area, datasheet_length=cand.core.effective_length)
    b_ac = rep0.fields["B_ac_amplitude_T"]
    mesh = rep0.mesh
    core_mask = mesh.region_mask("core")
    area = mesh.areas()
    depth = geo.section.depth
    w0 = rep0.winding_losses["L"]
    p_dc0 = w0["dc_loss_W"]
    p_ac0 = w0["selected_W"] - p_dc0
    a_ref = _winding_loss_analytic(geo, cand, wave, t_w0)
    state: dict[str, Any] = {}

    def losses(tc: float, tw: float):
        ev = cand.material.loss.loss_density(cand.frequency, b_ac[core_mask], tc, shape, policy="warn")
        pv = np.zeros(mesh.n_elements)
        pv[core_mask] = ev.loss_density
        state["pv"] = pv
        r = resistivity_ratio(tw, t_w0)
        a_t = _winding_loss_analytic(geo, cand, wave, tw)
        ac_scale = (a_t["ac_loss_W"] / a_ref["ac_loss_W"]) if a_ref["ac_loss_W"] > 0 else 1.0
        return 2 * depth * float(np.sum(pv * area)), {"L": p_dc0 * r + p_ac0 * ac_scale}

    cooling = spec.cooling.to_cooling()
    cond_map = thermal_conductivity_map(cand.material)

    def thermal(pc: float, pw: dict[str, float]):
        th = solve_thermal_fem(geo, mesh, core_loss_density=state["pv"], winding_losses=pw, cooling=cooling, conductivity=cond_map)
        return th.region_stats["core"]["T_mean_K"], th.region_stats["winding:L"]["T_mean_K"], th

    et = electrothermal(losses, thermal, t_core0, t_w0, tol=0.5, max_iterations=10)
    th = et.thermal
    tc, tw = et.temperatures["core"], et.temperatures["winding"]
    rep = analyse_inductor(geo, frequency=cand.frequency, core=core_model(tc), current=wave, winding_temperature=tw, settings=settings,
                           winding="L", datasheet_area=cand.core.effective_area, datasheet_length=cand.core.effective_length)
    lim = spec.limits
    violations = []
    L_fea = rep.value("inductance")
    if abs(L_fea / spec.inductance - 1) > lim.inductance_tolerance:
        violations.append(f"inductance {L_fea * 1e6:.1f} uH outside +-{lim.inductance_tolerance:.0%} of {spec.inductance * 1e6:.1f} uH")
    if "inductance_incremental_at_peak_nonlinear" in rep.quantities:
        l_pk = rep.value("inductance_incremental_at_peak_nonlinear")
        if l_pk < lim.min_inductance_at_peak * spec.inductance:
            violations.append(f"incremental inductance at peak current {l_pk * 1e6:.1f} uH < {lim.min_inductance_at_peak:.0%} of target")
    bsat, clamped = cand.material.bsat(tc)
    if rep.value("flux_density_peak_local") > bsat:
        violations.append(f"local saturation at peak current: {rep.value('flux_density_peak_local'):.3f} T > Bsat {bsat:.3f} T")
    hs_w = th.region_stats["winding:L"]["T_max_K"]
    hs_c = th.region_stats["core"]["T_max_K"]
    if hs_w > lim.max_winding_hotspot:
        violations.append(f"winding hot spot {hs_w - 273.15:.0f} degC > {lim.max_winding_hotspot - 273.15:.0f} degC (2.5-D FE)")
    if hs_c > lim.max_core_hotspot:
        violations.append(f"core hot spot {hs_c - 273.15:.0f} degC > {lim.max_core_hotspot - 273.15:.0f} degC (2.5-D FE)")
    if not tuned["converged"]:
        violations.append("FEA gap tuning did not converge")
    if not et.converged:
        violations.append("electrothermal FE iteration did not converge")
    violations += [v for v in analytic["violations"] if v.startswith(("winding: current", "window fill", "geometry"))]
    objectives = dict(analytic["objectives"])
    objectives.update({"total_loss_W": rep.value("total_loss"), "hotspot_C": max(hs_w, hs_c) - 273.15})
    surrogate = et.core_loss + sum(et.winding_losses.values())
    metrics = {"gap_tuning": {k: v for k, v in tuned.items() if k != "geometry"}, "gap_m": tuned["gap_m"], "gap_analytic_m": analytic["metrics"]["gap_m"],
               "em": rep.compact(), "thermal_fe": th.summary(), "hotspot_winding_K": hs_w, "hotspot_core_K": hs_c, "bsat_T": bsat,
               "electrothermal": {"converged": et.converged, "iterations": et.iterations, "history": et.history,
                                  "surrogate_total_loss_W": surrogate, "re_solved_total_loss_W": rep.value("total_loss"),
                                  "surrogate_relative_error": (surrogate - rep.value("total_loss")) / rep.value("total_loss")}}
    return {"feasible": not violations, "violations": violations, "objectives": objectives, "metrics": metrics, "geometry": geo,
            "fidelity": "fea", "reports": {"em": rep, "thermal": th}}


def neighbours(cand: InductorCandidate, spec: InductorSpec) -> list[InductorCandidate]:
    out = []
    for dn in (-1, 1):
        if cand.turns + dn >= 1:
            out.append(InductorCandidate(**{**cand.__dict__, "turns": cand.turns + dn, "gap": None, "tags": {**cand.tags, "move": f"N{dn:+d}"}}))
    c = cand.conductor
    if c.kind == "round":
        for p in (c.parallel - 1, c.parallel + 1):
            if p >= 1:
                out.append(InductorCandidate(**{**cand.__dict__, "conductor": c.model_copy(update={"parallel": p}), "gap": None,
                                                "tags": {**cand.tags, "move": f"parallel {p}"}}))
    elif c.kind == "foil":
        for factor in (0.8, 1.25):
            out.append(InductorCandidate(**{**cand.__dict__, "conductor": c.model_copy(update={"thickness": c.thickness * factor}), "gap": None,
                                            "tags": {**cand.tags, "move": f"foil x{factor:g}"}}))
    return out


__all__ = ["InductorCandidate", "evaluate_analytic", "evaluate_detailed", "fea_inductance", "fringing_factor", "generate_candidates",
           "inductance_analytic", "neighbours", "solve_gap_analytic", "tune_gap_fea"]
