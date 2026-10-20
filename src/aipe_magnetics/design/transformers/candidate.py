"""DAB transformer candidates: generation, analytical (fast) evaluation and detailed (FEA + thermal FE) evaluation.

Analytical fidelity (every candidate): parametric geometry and layer layout (exact build-up, mean turn length per layer),
SPS current waveforms with the realised turns ratio over the voltage range, Faraday peak flux density on the datasheet
``A_e``, iGSE core loss on ``V_e``, per-layer Dowell (solid conductors) or 1-D litz model (litz) with the current
harmonics, 1-D leakage inductance, thermal network with an electrothermal fixed point.

Detailed fidelity (shortlist): 2-D FEA (magnetising and leakage fields, harmonic eddy-current solves for solid
conductors, field-based litz losses, field-based core loss), 2.5-D thermal FE coupled electrothermally, and a final
electromagnetic re-solve at the converged temperatures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from aipe_magnetics.common.units import MU_0
from aipe_magnetics.database import CoreData
from aipe_magnetics.design.common import (
    MaterialContext,
    electrothermal,
    flux_shape_square,
    pack_conductivities,
    resistivity_ratio,
    thermal_conductivity_map,
)
from aipe_magnetics.design.specs import DABTransformerSpec
from aipe_magnetics.design.transformers.dab import inductance_for_phase_shift, operating_points
from aipe_magnetics.engineering import formulas as F
from aipe_magnetics.engineering.winding_harmonic import (
    dowell_layer_stack_loss,
    leakage_inductance_1d,
    litz_proximity_loss_1d,
    litz_strand_skin_factor,
)
from aipe_magnetics.geometry.conductors import ConductorSpec, litz_for_area
from aipe_magnetics.geometry.ecore import BobbinSpec, ECoreDimensions, ECoreMagnetic, GapSpec, InsulationSpec, WindingLayout, arrangement_sections
from aipe_magnetics.simulation.thermal.network import solve_thermal_network

DEFAULT_SPACER = 10e-6  # residual mating gap per leg of an "ungapped" set (assumption; ~20 um total, consistent with the TDK AL value)


@dataclass
class TransformerCandidate:
    material: MaterialContext
    core: CoreData
    n1: int
    n2: int
    kind: str
    primary: ConductorSpec
    secondary: ConductorSpec
    arrangement: str
    frequency: float
    spacer: float = DEFAULT_SPACER
    creepage_margin: float = 0.0
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return (f"{self.core.name.replace(' ', '')}|{self.material.data.grade or self.material.id}|{self.n1}:{self.n2}|"
                f"{self.primary.describe()}/{self.secondary.describe()}|{self.arrangement}|{self.frequency / 1e3:g}kHz")

    @property
    def standard_size_equivalent(self) -> bool:
        return self.material.id not in self.core.material_refs

    def params(self) -> dict[str, Any]:
        return {"material": self.material.id, "core": self.core.id, "n1": self.n1, "n2": self.n2, "conductor": self.kind,
                "standard_size_equivalent": self.standard_size_equivalent,
                "primary_conductor": self.primary.describe(), "secondary_conductor": self.secondary.describe(),
                "arrangement": self.arrangement, "frequency_Hz": self.frequency, "spacer_m": self.spacer, **self.tags}

    def model(self, name: str = "dab-transformer") -> ECoreMagnetic:
        dims = ECoreDimensions.from_core_data(self.core)
        sections = arrangement_sections(self.arrangement, {"P": (self.n1, self.primary), "S": (self.n2, self.secondary)})
        return ECoreMagnetic(name=name, core=dims, core_material=self.material.id, gap=GapSpec(spacer=self.spacer, note="residual mating gap"),
                             bobbin=BobbinSpec(), layout=WindingLayout(sections=sections, arrangement=self.arrangement,
                                                                       insulation=InsulationSpec(creepage_margin=self.creepage_margin)))


@dataclass
class DABContext:
    spec: DABTransformerSpec
    series_inductance: float

    @classmethod
    def from_spec(cls, spec: DABTransformerSpec) -> "DABContext":
        n_nom = spec.v1 / spec.v2
        L = spec.series_inductance or inductance_for_phase_shift(spec.power, spec.v1, spec.v2, n_nom, spec.frequency, spec.rated_phase_shift)
        return cls(spec=spec, series_inductance=L)

    def operating_points(self, n: float, frequency: float):
        s = self.spec
        # the series inductance is fixed by the spec (rated phase shift at the nominal ratio and frequency); with a different
        # realised turns ratio or frequency the phase shift adapts
        L = self.series_inductance * self.spec.frequency / frequency if s.series_inductance is None else self.series_inductance
        return operating_points(power=s.power, v1=s.v1, v2_nominal=s.v2, v2_range=s.v2_range, n=n, frequency=frequency, inductance=L), L


# -------------------------------------------------------------------------------------------------
# generation
# -------------------------------------------------------------------------------------------------
def minimum_primary_turns(spec: DABTransformerSpec, material: MaterialContext, core: CoreData, frequency: float, t_core: float) -> int:
    bsat, _ = material.bsat(t_core)
    b_allow = spec.limits.flux_fraction_of_bsat * bsat
    return F.turns_for_flux_density_square(spec.v1, frequency, core.effective_area, b_allow, 0.5)


def _conductor_options(spec: DABTransformerSpec, i1: float, i2: float, winding_height: float) -> list[tuple[str, ConductorSpec, ConductorSpec, dict]]:
    sp = spec.space
    out = []
    for kind in sp.conductors:
        if kind == "litz":
            for d in sp.litz_strand_diameters:
                for j in sp.current_densities:
                    out.append((kind, litz_for_area(i1 / j, d), litz_for_area(i2 / j, d), {"J_design_A_per_mm2": j / 1e6, "strand_mm": d * 1e3}))
        elif kind == "foil":
            for t in sp.foil_thicknesses:
                out.append((kind, ConductorSpec(kind="foil", thickness=t), ConductorSpec(kind="foil", thickness=t * i2 / i1),
                            {"foil_mm": t * 1e3}))
        elif kind == "round":
            for d in sp.round_diameters:
                for p in sp.parallel_round:
                    a1 = p * math.pi * d * d / 4
                    p2 = max(1, round(a1 * i2 / i1 / (math.pi * d * d / 4)))
                    out.append((kind, ConductorSpec(kind="round", bare_diameter=d, parallel=p), ConductorSpec(kind="round", bare_diameter=d, parallel=p2),
                                {"round_mm": d * 1e3, "parallel_primary": p}))
    return out


def generate_candidates(spec: DABTransformerSpec, materials: list[MaterialContext], cores: list[CoreData]) -> list[TransformerCandidate]:
    ctx = DABContext.from_spec(spec)
    freqs = spec.space.frequencies or [spec.frequency]
    n_nom = spec.v1 / spec.v2
    out: list[TransformerCandidate] = []
    for m in materials:
        for c in cores:
            if m.id not in c.material_refs and spec.space.cores is None and spec.space.material_offer_policy == "catalogue":
                continue
            for f in freqs:
                n1_min = minimum_primary_turns(spec, m, c, f, spec.loss_temperature_initial)
                n1_values = spec.space.primary_turns or list(range(n1_min, n1_min + spec.space.turn_steps + 1))
                for n1 in n1_values:
                    n2 = max(1, round(n1 / n_nom))
                    if abs((n1 / n2) / n_nom - 1) > spec.limits.turns_ratio_tolerance + 1e-9:
                        continue
                    ops, _ = ctx.operating_points(n1 / n2, f)
                    i1 = max(op.current.rms() for op in ops)
                    i2 = i1 * n1 / n2
                    for kind, pc, sc, tags in _conductor_options(spec, i1, i2, 0.0):
                        for arr in spec.space.arrangements:
                            out.append(TransformerCandidate(material=m, core=c, n1=n1, n2=n2, kind=kind, primary=pc, secondary=sc,
                                                            arrangement=arr, frequency=f, tags=dict(tags)))
    return out


# -------------------------------------------------------------------------------------------------
# analytical evaluation
# -------------------------------------------------------------------------------------------------
def _stack(geometry: Any, n1: int, n2: int, temperature: float, winding_height: float) -> list[dict]:
    rho = F.copper_resistivity(temperature)
    unit = {"primary": 1.0, "secondary": -n1 / n2}
    out = []
    for layer in sorted(geometry.layers, key=lambda l: l.d_in):
        c = layer.conductor
        out.append({"winding": layer.winding, "x0": layer.d_in, "x1": layer.d_out, "ampere_turns": unit[layer.winding] * layer.turns,
                    "turns": layer.turns, "r_dc": rho * layer.turns * layer.mean_turn_length / c.copper_area(),
                    "kind": c.kind if c.kind in ("foil", "round") else "foil", "size": c.bare_diameter if c.kind == "round" else (c.thickness or 1e-6),
                    "pitch": c.axial_pitch() if c.kind == "round" else winding_height, "strands": c.strand_count if c.kind == "litz" else 0,
                    "strand_diameter": c.strand_diameter, "mlt": layer.mean_turn_length, "litz": c.kind == "litz"})
    return out


def winding_losses_analytic(geometry: Any, cand: TransformerCandidate, current, temperature: float) -> dict[str, dict[str, float]]:
    """Per-winding analytical loss for the primary current waveform ``current`` (secondary = n x primary)."""
    space = geometry.model.winding_space()
    stack = _stack(geometry, cand.n1, cand.n2, temperature, space["height"])
    harmonics = [(n, a) for n, a, _ in current.harmonic_amplitudes(41) if a > 0]
    out: dict[str, dict[str, float]] = {}
    if cand.kind in ("round", "foil"):
        dow = dowell_layer_stack_loss(stack, harmonics, cand.frequency, temperature)
        for w, v in dow.items():
            out[w] = {"loss_W": v, "method": "dowell_per_layer"}
    else:
        rho = F.copper_resistivity(temperature)
        for w in ("primary", "secondary"):
            layers = [s for s in stack if s["winding"] == w]
            r_dc = sum(s["r_dc"] for s in layers)
            ratio = 1.0 if w == "primary" else cand.n1 / cand.n2
            cond = cand.primary if w == "primary" else cand.secondary
            dc_skin = sum(r_dc * (a * ratio) ** 2 / 2 * litz_strand_skin_factor(cond.strand_diameter, F.skin_depth(n * cand.frequency, rho))
                          for n, a in harmonics)
            prox = litz_proximity_loss_1d([s if s["winding"] == w else {**s, "strands": 0} for s in stack], space["height"], harmonics,
                                          cand.frequency, temperature)
            out[w] = {"loss_W": dc_skin + prox, "dc_and_skin_W": dc_skin, "proximity_W": prox, "method": "litz_1d", "r_dc_Ohm": r_dc}
    return out


def evaluate_analytic(cand: TransformerCandidate, spec: DABTransformerSpec) -> dict[str, Any]:
    """Fast evaluation: returns ``{feasible, violations, objectives, metrics, geometry}``."""
    ctx = DABContext.from_spec(spec)
    lim = spec.limits
    violations: list[str] = []
    metrics: dict[str, Any] = {}
    model = cand.model()
    geo = model.build()
    for chk in geo.checks:
        if not chk.passed:
            violations.append(f"geometry: {chk.message}")
    if violations:
        return {"feasible": False, "violations": violations, "objectives": {}, "metrics": metrics, "geometry": geo, "fidelity": "analytical"}
    n = cand.n1 / cand.n2
    ops, L_series = ctx.operating_points(n, cand.frequency)
    nominal = ops[0]
    worst = max(ops, key=lambda op: op.current.rms())
    metrics["operating_points"] = [op.summary(cand.frequency, n) for op in ops]
    metrics["series_inductance_H"] = L_series
    ae, ve = cand.core.effective_area, cand.core.effective_volume
    b_pk = F.flux_density_peak_square(spec.v1, cand.frequency, cand.n1, ae, 0.5)
    metrics["flux_density_peak_T"] = b_pk
    metrics["turns_ratio_error"] = n / (spec.v1 / spec.v2) - 1
    shape = flux_shape_square(0.5)
    kp = pack_conductivities()
    k_core = cand.material.thermal_conductivity()
    cooling = spec.cooling.to_cooling()
    wh = model.winding_space()["height"]
    t_ref = spec.loss_temperature_initial

    def losses_at(op, t_core: float, t_w: float) -> tuple[float, dict[str, float], dict]:
        ev = cand.material.loss.loss_density(cand.frequency, b_pk, t_core, shape, policy="warn")
        wl = winding_losses_analytic(geo, cand, op.current, t_w)
        return float(ev.loss_density) * ve, {w: v["loss_W"] for w, v in wl.items()}, {"core_sets": ev.sets, "core_warnings": ev.warnings, "winding": wl}

    def thermal(pc: float, pw: dict[str, float]):
        res = solve_thermal_network(geo, core_loss=pc, winding_losses=pw, cooling=cooling, k_core=k_core,
                                    k_bobbin=0.3, k_insulation=0.12, k_pack=kp)
        tw = max(v for k, v in res.temperatures.items() if k != "core")
        return res.temperatures["core"], tw, res

    # electrothermal at the worst-case current (thermal limit), then nominal losses at those temperatures (efficiency)
    et = electrothermal(lambda tc, tw: losses_at(worst, tc, tw)[:2], thermal, t_ref, t_ref)
    net = et.thermal
    p_core_nom, p_w_nom, detail = losses_at(nominal, et.temperatures["core"], et.temperatures["winding"])
    p_core_w, p_w_w, _ = losses_at(worst, et.temperatures["core"], et.temperatures["winding"])
    metrics["loss_model_warnings"] = sorted(set(detail["core_warnings"]))
    metrics.update({"core_loss_W": p_core_nom, "winding_loss_W": sum(p_w_nom.values()), "winding_losses_W": p_w_nom,
                    "total_loss_W": p_core_nom + sum(p_w_nom.values()), "worst_case_total_loss_W": p_core_w + sum(p_w_w.values()),
                    "worst_case_label": worst.label, "electrothermal_converged": et.converged, "electrothermal_iterations": et.iterations,
                    "thermal": net.summary() if net else None, "loss_detail": detail})
    bsat, clamped = cand.material.bsat(et.temperatures["core"])
    metrics["bsat_T"] = bsat
    metrics["bsat_clamped"] = clamped
    if b_pk > lim.flux_fraction_of_bsat * bsat:
        violations.append(f"saturation: B_pk {b_pk:.3f} T > {lim.flux_fraction_of_bsat:.0%} of Bsat {bsat:.3f} T at {et.temperatures['core'] - 273.15:.0f} degC")
    if abs(metrics["turns_ratio_error"]) > lim.turns_ratio_tolerance + 1e-9:
        violations.append(f"turns ratio error {metrics['turns_ratio_error']:+.1%}")
    i1 = worst.current.rms()
    for w, cond, i in (("primary", cand.primary, i1), ("secondary", cand.secondary, i1 * n)):
        a = cond.copper_area(wh)
        j = i / a
        metrics[f"current_density_{w}_A_per_mm2"] = j / 1e6
        if j > lim.max_current_density:
            violations.append(f"{w}: current density {j / 1e6:.2f} A/mm^2 > {lim.max_current_density / 1e6:.1f}")
    fill = geo.derived["window_fill_bare_copper"]
    metrics["window_fill"] = fill
    if fill > lim.max_window_fill:
        violations.append(f"window fill {fill:.2f} > {lim.max_window_fill:.2f}")
    leak = leakage_inductance_1d([{k: s[k] for k in ("x0", "x1", "ampere_turns")} for s in _stack(geo, cand.n1, cand.n2, t_ref, wh)],
                                 wh, sum(l.mean_turn_length * l.turns for l in geo.layers) / sum(l.turns for l in geo.layers))
    metrics["leakage_inductance_1d_H"] = leak
    metrics["external_series_inductance_H"] = L_series - leak
    if leak > L_series:
        violations.append(f"leakage {leak * 1e6:.1f} uH exceeds the required series inductance {L_series * 1e6:.1f} uH")
    hs_w = max(v for k, v in net.hotspots.items() if k != "core")
    hs_c = net.hotspots["core"]
    metrics["hotspot_winding_K"], metrics["hotspot_core_K"] = hs_w, hs_c
    if not et.converged:
        violations.append("electrothermal iteration did not converge (thermal runaway or > 700 K)")
    if hs_w > lim.max_winding_hotspot:
        violations.append(f"winding hot spot {hs_w - 273.15:.0f} degC > {lim.max_winding_hotspot - 273.15:.0f} degC")
    if hs_c > lim.max_core_hotspot:
        violations.append(f"core hot spot {hs_c - 273.15:.0f} degC > {lim.max_core_hotspot - 273.15:.0f} degC")
    iso = spec.isolation
    if iso.creepage is not None and cand.creepage_margin < iso.creepage:
        violations.append(f"creepage margin {cand.creepage_margin * 1e3:.1f} mm < required {iso.creepage * 1e3:.1f} mm")
    metrics["insulation_assessed"] = iso.creepage is not None or iso.clearance is not None
    density = cand.material.density()
    core_mass = cand.core.mass or density * geo.derived["core_volume_m3"]
    copper_mass = 8933.0 * geo.derived["copper_volume_m3"]
    objectives = {"total_loss_W": metrics["total_loss_W"], "volume_cm3": geo.derived["bounding_volume_m3"] * 1e6,
                  "mass_g": (core_mass + copper_mass) * 1e3, "hotspot_C": max(hs_w, hs_c) - 273.15}
    metrics.update({"core_mass_g": core_mass * 1e3, "copper_mass_g": copper_mass * 1e3})
    return {"feasible": not violations, "violations": violations, "objectives": objectives, "metrics": metrics, "geometry": geo,
            "fidelity": "analytical"}


# -------------------------------------------------------------------------------------------------
# detailed evaluation (FEA + 2.5-D thermal FE, electrothermal)
# -------------------------------------------------------------------------------------------------
def evaluate_detailed(cand: TransformerCandidate, spec: DABTransformerSpec, *, analytic: dict[str, Any] | None = None,
                      em_settings: Any = None, mu_r: float | None = None) -> dict[str, Any]:
    from aipe_magnetics.simulation.electromagnetic.analysis import CoreModel, EMSettings, analyse_transformer
    from aipe_magnetics.simulation.thermal.fem2d import solve_thermal_fem

    analytic = analytic or evaluate_analytic(cand, spec)
    geo = analytic["geometry"]
    ctx = DABContext.from_spec(spec)
    n = cand.n1 / cand.n2
    ops, L_series = ctx.operating_points(n, cand.frequency)
    nominal, worst = ops[0], max(ops, key=lambda op: op.current.rms())
    lam = spec.v1 * 0.5 / (2 * cand.frequency)
    mu = mu_r or cand.material.data.initial_permeability
    settings = em_settings or EMSettings()
    cooling = spec.cooling.to_cooling()
    cond_map = thermal_conductivity_map(cand.material)
    t_core0 = analytic["metrics"].get("thermal", {}).get("temperatures_K", {}).get("core", spec.loss_temperature_initial)
    t_w0 = max(v for k, v in analytic["metrics"]["thermal"]["temperatures_K"].items() if k != "core") if analytic["metrics"].get("thermal") else t_core0

    def core_model(t: float) -> CoreModel:
        bsat, _ = cand.material.bsat(t)
        return CoreModel(material_id=cand.material.id, mu_r=mu, bsat=bsat, loss=cand.material.loss, shape=flux_shape_square(0.5), temperature=t)

    rep_w = analyse_transformer(geo, frequency=cand.frequency, core=core_model(t_core0), primary_current=worst.current,
                                primary_flux_linkage_peak=lam, turns_ratio=n, winding_temperature=t_w0, settings=settings)
    b_mag = rep_w.fields["B_magnetising_peak_T"]
    core_mask = rep_w.mesh.region_mask("core")
    area = rep_w.mesh.areas()
    depth = geo.section.depth

    def core_loss_at(t: float) -> tuple[float, Any]:
        import numpy as np
        ev = cand.material.loss.loss_density(cand.frequency, b_mag[core_mask], t, flux_shape_square(0.5), policy="warn")
        pv = np.zeros(rep_w.mesh.n_elements)
        pv[core_mask] = ev.loss_density
        return 2 * depth * float(np.sum(pv * area)), pv

    w_ref = {w: e["selected_W"] for w, e in rep_w.winding_losses.items()}
    split = {}
    for w, e in rep_w.winding_losses.items():
        if "litz_dc_and_skin_W" in e:
            split[w] = (e["litz_dc_and_skin_W"], e["litz_proximity_W"])
    analytic_ref = {w: v for w, v in winding_losses_analytic(geo, cand, worst.current, t_w0).items()}

    def winding_at(t: float) -> dict[str, float]:
        out = {}
        r = resistivity_ratio(t, t_w0)
        for w, p in w_ref.items():
            if w in split:  # litz: DC/skin ~ rho, proximity ~ 1/rho
                out[w] = split[w][0] * r + split[w][1] / r
            else:  # solid conductors: scaled with the per-layer Dowell model
                a_t = winding_losses_analytic(geo, cand, worst.current, t)[w]["loss_W"]
                out[w] = p * a_t / analytic_ref[w]["loss_W"]
        return out

    state: dict[str, Any] = {}

    def losses(tc: float, tw: float):
        pc, pv = core_loss_at(tc)
        state["pv"] = pv
        return pc, winding_at(tw)

    def thermal(pc: float, pw: dict[str, float]):
        th = solve_thermal_fem(geo, rep_w.mesh, core_loss_density=state["pv"], winding_losses=pw, cooling=cooling, conductivity=cond_map)
        tw = max(v["T_mean_K"] for k, v in th.region_stats.items() if k.startswith("winding:"))
        return th.region_stats["core"]["T_mean_K"], tw, th

    et = electrothermal(losses, thermal, t_core0, t_w0, tol=0.5, max_iterations=10)
    th = et.thermal
    # final consistency re-solve at the converged temperatures, nominal operating point (efficiency) and worst case
    tc, tw = et.temperatures["core"], et.temperatures["winding"]
    rep_nom = analyse_transformer(geo, frequency=cand.frequency, core=core_model(tc), primary_current=nominal.current,
                                  primary_flux_linkage_peak=lam, turns_ratio=n, winding_temperature=tw, settings=settings)
    rep_final_w = analyse_transformer(geo, frequency=cand.frequency, core=core_model(tc), primary_current=worst.current,
                                      primary_flux_linkage_peak=lam, turns_ratio=n, winding_temperature=tw, settings=settings)
    surrogate_w = et.core_loss + sum(et.winding_losses.values())
    resolved_w = rep_final_w.value("total_loss")
    hs_w = max(v["T_max_K"] for k, v in th.region_stats.items() if k.startswith("winding:"))
    hs_c = th.region_stats["core"]["T_max_K"]
    lim = spec.limits
    violations = []
    bsat, clamped = cand.material.bsat(tc)
    b_local = rep_nom.value("flux_density_peak_local")
    if rep_nom.value("flux_density_peak_centre_leg") > lim.flux_fraction_of_bsat * bsat:
        violations.append(f"saturation: centre-leg B {rep_nom.value('flux_density_peak_centre_leg'):.3f} T > {lim.flux_fraction_of_bsat:.0%} Bsat")
    if b_local > bsat:
        violations.append(f"local saturation: B_local {b_local:.3f} T > Bsat {bsat:.3f} T")
    if hs_w > lim.max_winding_hotspot:
        violations.append(f"winding hot spot {hs_w - 273.15:.0f} degC > {lim.max_winding_hotspot - 273.15:.0f} degC (2.5-D FE)")
    if hs_c > lim.max_core_hotspot:
        violations.append(f"core hot spot {hs_c - 273.15:.0f} degC > {lim.max_core_hotspot - 273.15:.0f} degC (2.5-D FE)")
    leak = rep_nom.value("leakage_inductance")
    if leak > L_series:
        violations.append(f"FEA leakage {leak * 1e6:.1f} uH exceeds the series inductance {L_series * 1e6:.1f} uH")
    violations += [v for v in analytic["violations"] if v.startswith(("geometry", "turns ratio", "primary: current", "secondary: current", "window fill", "creepage"))]
    if not et.converged:
        violations.append("electrothermal FE iteration did not converge")
    objectives = dict(analytic["objectives"])
    objectives.update({"total_loss_W": rep_nom.value("total_loss"), "hotspot_C": max(hs_w, hs_c) - 273.15})
    metrics = {"em_nominal": rep_nom.compact(), "em_worst_case": rep_final_w.compact(), "thermal_fe": th.summary(),
               "electrothermal": {"converged": et.converged, "iterations": et.iterations, "history": et.history,
                                  "surrogate_total_loss_W": surrogate_w, "re_solved_total_loss_W": resolved_w,
                                  "surrogate_relative_error": (surrogate_w - resolved_w) / resolved_w if resolved_w else None},
               "hotspot_winding_K": hs_w, "hotspot_core_K": hs_c, "bsat_T": bsat, "bsat_clamped": clamped, "series_inductance_H": L_series,
               "external_series_inductance_H": L_series - leak, "operating_points": [op.summary(cand.frequency, n) for op in ops]}
    return {"feasible": not violations, "violations": violations, "objectives": objectives, "metrics": metrics, "geometry": geo,
            "fidelity": "fea", "reports": {"nominal": rep_nom, "worst_case": rep_final_w, "thermal": th}}


def neighbours(cand: TransformerCandidate, spec: DABTransformerSpec) -> list[TransformerCandidate]:
    """Local refinement moves: N1 +-1 (N2 re-rounded), conductor size +-1 option step, arrangement swap."""
    out = []
    n_nom = spec.v1 / spec.v2
    for dn in (-1, 1):
        n1 = cand.n1 + dn
        if n1 < 1:
            continue
        n2 = max(1, round(n1 / n_nom))
        if abs((n1 / n2) / n_nom - 1) > spec.limits.turns_ratio_tolerance + 1e-9:
            continue
        out.append(TransformerCandidate(**{**cand.__dict__, "n1": n1, "n2": n2, "tags": {**cand.tags, "move": f"N1{dn:+d}"}}))
    if cand.kind == "litz":
        for factor in (0.9, 1.1):
            p = litz_for_area(cand.primary.copper_area() * factor, cand.primary.strand_diameter)
            s = litz_for_area(cand.secondary.copper_area() * factor, cand.secondary.strand_diameter)
            out.append(TransformerCandidate(**{**cand.__dict__, "primary": p, "secondary": s, "tags": {**cand.tags, "move": f"litz x{factor:g}"}}))
    if cand.kind == "foil":
        for factor in (0.8, 1.25):
            out.append(TransformerCandidate(**{**cand.__dict__, "primary": cand.primary.model_copy(update={"thickness": cand.primary.thickness * factor}),
                                               "secondary": cand.secondary.model_copy(update={"thickness": cand.secondary.thickness * factor}),
                                               "tags": {**cand.tags, "move": f"foil x{factor:g}"}}))
    for arr in spec.space.arrangements:
        if arr != cand.arrangement:
            out.append(TransformerCandidate(**{**cand.__dict__, "arrangement": arr, "tags": {**cand.tags, "move": f"arrangement {arr}"}}))
    return out


__all__ = ["DABContext", "DEFAULT_SPACER", "TransformerCandidate", "evaluate_analytic", "evaluate_detailed", "generate_candidates",
           "minimum_primary_turns", "neighbours", "winding_losses_analytic"]
