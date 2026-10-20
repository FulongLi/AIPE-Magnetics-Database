"""Electromagnetic studies of a parametric E-core transformer or inductor (2-D finite elements + post-processing).

Every reported quantity carries a ``method`` tag:

``fea``                  computed directly from a finite-element solution (energies, flux linkages, local B, eddy-current
                         losses of solid conductors);
``fea_post_processed``   a material / strand model evaluated on the finite-element field (core loss from the local flux
                         density via Steinmetz/iGSE, litz strand proximity loss from the local field);
``analytical``           closed-form references (reluctance model, McLyman fringing factor, Dowell per harmonic,
                         1-D leakage integral, Ve-based core loss) reported next to the numerical values.

Limitations of the 2-D planar half-section model (also returned with every report):

* the section ``y = 0`` represents the core over its depth ``C`` and the winding window over the whole turn
  (mean turn length); end-turn fields outside the core, 3-D fringing at the front/back faces of the gaps and
  the round corners of the turns are not resolved;
* the core is linear (``mu_r`` constant) for the flux-density distribution, core loss and inductances; the
  nonlinear solve (inductor) uses an *assumed* tanh saturation curve and is reported separately;
* round wire is modelled as the equal-area square conductor (Dowell's equivalence); litz bundles are homogenised
  (strand eddy currents from the low-frequency strand formulas on the local field);
* magnetostatic fields are used for core loss and litz proximity loss (eddy-current reaction of the core and of
  nearby litz is neglected; ferrite conductivity is not modelled).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aipe_magnetics.common.errors import InvalidInputError
from aipe_magnetics.common.units import MU_0
from aipe_magnetics.database.loss import FluxShape, LossEvaluator
from aipe_magnetics.engineering import formulas as F
from aipe_magnetics.engineering.periodic import PiecewiseLinear
from aipe_magnetics.engineering.winding_harmonic import (
    dowell_harmonic_loss,
    dowell_layer_stack_loss,
    leakage_inductance_1d,
    litz_proximity_loss_1d,
    litz_strand_skin_factor,
    strand_proximity_loss_per_length,
)
from aipe_magnetics.simulation.electromagnetic.harmonic import HarmonicSolution, solve_harmonic
from aipe_magnetics.simulation.electromagnetic.magnetostatic import EMProblem, MagnetostaticSolution, solve_magnetostatic
from aipe_magnetics.simulation.electromagnetic.materials import SaturationCurve, copper_conductivity, skin_depth
from aipe_magnetics.simulation.mesh import Mesh2D, MeshSettings, generate_mesh

SOLVER_ID = "aipe.magnetics.fem2d"
SOLVER_VERSION = "0.3.0"

LIMITATIONS_2D = [
    "2-D planar half-section (y = 0): core energy scaled by the core depth, window energy by the mean turn length; "
    "end-turn fields, 3-D gap fringing at the front/back core faces and turn corners are not resolved.",
    "Linear core permeability for the field distribution, inductances and core loss; ferrite eddy currents and "
    "dimensional resonance are not modelled.",
    "Round wire as equal-area square conductors; litz bundles homogenised with strand losses from low-frequency "
    "strand formulas evaluated on the local field.",
    "Winding losses are per unit length of the window cross-section multiplied by the layer mean turn length "
    "(conservative outside the core).",
]


@dataclass
class CoreModel:
    material_id: str
    mu_r: float
    bsat: float
    loss: LossEvaluator
    shape: FluxShape
    temperature: float
    curve: SaturationCurve | None = None


@dataclass
class EMSettings:
    mesher: str = "structured"
    mesh: MeshSettings | None = None
    harmonics_fea: int = 7               # highest harmonic solved by harmonic FEA (solid conductors)
    harmonic_threshold: float = 0.02     # skip FEA harmonics whose amplitude is below this fraction of the fundamental
    harmonics_analytic: int = 41         # harmonics included in the analytical / litz post-processing sums
    nonlinear: bool = True
    padding: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {"mesher": self.mesher, "mesh": self.mesh.to_json() if self.mesh else None, "harmonics_fea": self.harmonics_fea,
                "harmonic_threshold": self.harmonic_threshold, "harmonics_analytic": self.harmonics_analytic, "nonlinear": self.nonlinear}


def _qv(value: float, unit: str, method: str, description: str, **extra: Any) -> dict[str, Any]:
    return {"value": float(value), "unit": unit, "method": method, "description": description, **extra}


@dataclass
class EMReport:
    kind: str
    quantities: dict[str, dict[str, Any]]
    fields: dict[str, np.ndarray]
    mesh: Mesh2D
    solver: dict[str, Any]
    diagnostics: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=lambda: list(LIMITATIONS_2D))
    winding_losses: dict[str, dict[str, float]] = field(default_factory=dict)

    def value(self, key: str) -> float:
        return float(self.quantities[key]["value"])

    def compact(self) -> dict[str, Any]:
        return {"kind": self.kind, "quantities": self.quantities, "winding_losses": self.winding_losses, "warnings": self.warnings,
                "limitations": self.limitations, "diagnostics": {k: v for k, v in self.diagnostics.items() if k != "groups"},
                "solver": self.solver}

    def save_fields(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = self.mesh.to_npz_dict()
        arrays.update({k: np.asarray(v) for k, v in self.fields.items()})
        np.savez_compressed(path, **arrays)
        return path


# -------------------------------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------------------------------
def turn_lengths(geometry: Any) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    for layer in geometry.layers:
        for k in range(layer.turns):
            out[(layer.winding, layer.first_turn + k)] = layer.mean_turn_length
    return out


def _winding_info(geometry: Any) -> dict[str, dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}
    for name in geometry.winding_names():
        layers = [l for l in geometry.layers if l.winding == name]
        cond = layers[0].conductor
        turns = sum(l.turns for l in layers)
        length = sum(l.turns * l.mean_turn_length for l in layers)
        info[name] = {"turns": turns, "layers": len(layers), "conductor": cond, "length": length, "mlt": length / turns,
                      "copper_area": cond.copper_area(), "pitch": cond.axial_pitch() if cond.kind != "foil" else cond.width}
    return info


def _r_dc(info: dict[str, Any], temperature: float) -> float:
    return F.copper_resistivity(temperature) * info["length"] / info["copper_area"]


def default_mesh_settings(geometry: Any, frequency: float, temperature: float, harmonics_fea: int) -> MeshSettings:  # noqa: ARG001
    """Element sizes resolving the skin depth of the *fundamental* in solid conductors (h <= delta_1 / 2.5).

    Higher harmonics are solved by FEA when the mesh resolves their skin depth (h <= delta_n / 2); the remaining ones are
    estimated with the FEA-calibrated per-harmonic Dowell model (reported per harmonic)."""
    has_solid = any(l.conductor.kind in ("round", "foil", "rectangular") for l in geometry.layers)
    delta = skin_depth(frequency, temperature)
    smallest = min((min(r.x1 - r.x0, r.z1 - r.z0) for r in geometry.section.conductors()), default=1e-3)
    h_cond = min(delta / 2.5, smallest / 6.0) if has_solid else min(1.0e-3, smallest / 3.0)
    core = geometry.model.core
    return MeshSettings(core=min(1.5e-3, core.centre_leg_width / 12), conductor=max(h_cond, 1.0e-5), stranded=max(smallest / 3.0, 2e-4),
                        window_air=1.0e-3, winding_pack=1.0e-3, insulation=0.5e-3, air=0.25 * core.length_a / 2)


def _mesh_with_budget(section: Any, settings: MeshSettings, mesher: str, warnings: list[str]) -> Mesh2D:
    """Mesh; when the structured grid exceeds its element budget use Gmsh (if installed) or coarsen the conductors (warned)."""
    from aipe_magnetics.common.errors import UnsupportedOperationError
    from aipe_magnetics.simulation.mesh import gmsh_available
    try:
        return generate_mesh(section, settings, mesher)  # type: ignore[arg-type]
    except UnsupportedOperationError as error:
        if mesher != "structured" or "max_elements" not in str(error):
            raise
    if gmsh_available()[0]:
        warnings.append("structured mesh exceeded its element budget; the unstructured Gmsh mesher was used instead")
        return generate_mesh(section, settings, "gmsh")
    st = settings
    for _ in range(6):
        st = MeshSettings(**{**st.to_json(), "conductor": st.conductor * 1.5, "stranded": st.stranded * 1.5})
        try:
            mesh = generate_mesh(section, st, "structured")
            warnings.append(f"structured mesh coarsened to conductor size {st.conductor * 1e3:.3f} mm to stay inside the element budget")
            return mesh
        except UnsupportedOperationError:
            continue
    raise UnsupportedOperationError("no mesh within the element budget; install gmsh (pip install 'aipe-magnetics[mesh]')")


def _core_mask(mesh: Mesh2D) -> np.ndarray:
    return mesh.region_mask("core")


def _core_loss_from_field(mesh: Mesh2D, b_amplitude: np.ndarray, core: CoreModel, frequency: float) -> tuple[float, np.ndarray, dict[str, float], list[str]]:
    mask = _core_mask(mesh)
    pv = np.zeros(mesh.n_elements)
    ev = core.loss.loss_density(frequency, b_amplitude[mask], core.temperature, core.shape)
    pv[mask] = np.asarray(ev.loss_density)
    # domain warnings from the representative (99.5 % area percentile) amplitude, not from singular corner elements
    b_rep = _area_percentile(b_amplitude[mask], mesh.areas()[mask], 0.995)
    ev_rep = core.loss.loss_density(frequency, b_rep, core.temperature, core.shape)
    factor = 2.0 if mesh.section.symmetry == "half_x" else 1.0
    depth = mesh.section.depth
    area = mesh.areas()
    total = factor * depth * float(np.sum(pv * area))
    by_region: dict[str, float] = {}
    for i, r in enumerate(mesh.section.regions):
        if r.role == "core":
            m = mesh.element_region == i
            by_region[r.name] = factor * depth * float(np.sum(pv[m] * area[m]))
    return total, pv, by_region, list(ev_rep.warnings)


def _area_percentile(values: np.ndarray, areas: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    cum = np.cumsum(areas[order]) / np.sum(areas)
    return float(values[order][min(int(np.searchsorted(cum, q)), values.size - 1)])


def _saturation(mesh: Mesh2D, b_peak: np.ndarray, bsat: float) -> dict[str, float]:
    """Saturation metrics. The raw maximum sits at the re-entrant window corners, where the linear field of the idealised
    sharp-cornered outline is singular (mesh dependent); the 99.5 % area percentile is the robust local peak."""
    mask = _core_mask(mesh)
    b = b_peak[mask]
    a = mesh.areas()[mask]
    p995 = _area_percentile(b, a, 0.995)
    return {"B_peak_max_T": float(b.max()), "B_peak_p995_T": p995, "B_peak_area_mean_T": float(np.sum(b * a) / np.sum(a)), "bsat_T": bsat,
            "fraction_above_0p8_bsat": float(np.sum(a[b > 0.8 * bsat]) / np.sum(a)),
            "fraction_above_bsat": float(np.sum(a[b > bsat]) / np.sum(a)),
            "local_margin": 1.0 - p995 / bsat}


def _stack_layers_1d(geometry: Any, polarity: dict[str, float], currents: dict[str, float]) -> list[dict[str, float]]:
    out = []
    for layer in sorted(geometry.layers, key=lambda l: l.d_in):
        at = polarity.get(layer.winding, 1.0) * currents.get(layer.winding, 0.0) * layer.turns
        out.append({"x0": layer.d_in, "x1": layer.d_out, "ampere_turns": at})
    return out


def _litz_losses(mesh: Mesh2D, geometry: Any, b_by_harmonic: dict[int, np.ndarray], harmonic_amplitude: dict[str, dict[int, float]],
                 frequency: float, temperature: float, lengths: dict[tuple[str, int], float]) -> dict[str, dict[str, float]]:
    """Field-based litz strand proximity loss + strand skin and DC loss per winding (fea_post_processed)."""
    sigma = copper_conductivity(temperature)
    area = mesh.areas()
    out: dict[str, dict[str, float]] = {}
    info = _winding_info(geometry)
    for i, r in enumerate(mesh.section.regions):
        if r.role != "stranded":
            continue
        name = r.winding
        cond = info[name]["conductor"]
        d, ns = cond.strand_diameter, cond.strand_count
        density = ns / r.area  # strands per unit area of the homogenised bundle
        m = mesh.element_region == i
        length = lengths[(name, r.turn)]
        prox = 0.0
        beyond = 0.0
        for n, b in b_by_harmonic.items():
            omega = 2 * math.pi * n * frequency
            p_n = float(np.sum(strand_proximity_loss_per_length(d, sigma, omega, np.abs(b[m])) * density * area[m])) * length
            prox += p_n
            if d > skin_depth(n * frequency, temperature):
                beyond += p_n
        entry = out.setdefault(name, {"proximity_W": 0.0, "proximity_beyond_strand_validity_W": 0.0})
        entry["proximity_W"] += prox
        entry["proximity_beyond_strand_validity_W"] += beyond
    for name, data in info.items():
        cond = data["conductor"]
        if cond.kind != "litz":
            continue
        r_dc = _r_dc(data, temperature)
        amps = harmonic_amplitude[name]
        dc = amps.get(0, 0.0)
        skin = r_dc * dc * dc
        for n, a in amps.items():
            if n == 0:
                continue
            fs = litz_strand_skin_factor(cond.strand_diameter, skin_depth(n * frequency, temperature))
            skin += r_dc * a * a / 2.0 * fs
        entry = out.setdefault(name, {"proximity_W": 0.0})
        entry.update({"dc_and_skin_W": skin, "r_dc_Ohm": r_dc})
        entry["total_W"] = entry["dc_and_skin_W"] + entry["proximity_W"]
        delta_max = skin_depth(max(amps) * frequency, temperature) if amps else math.inf
        entry["strand_to_skin_depth_at_highest_harmonic"] = cond.strand_diameter / delta_max
    return out


def _harmonics(waveform: PiecewiseLinear, n_max: int) -> dict[int, float]:
    out = {0: abs(waveform.mean())} if abs(waveform.mean()) > 1e-12 * max(1.0, waveform.peak()) else {}
    for n, amp, _ in waveform.harmonic_amplitudes(n_max):
        if amp > 1e-9 * max(1.0, waveform.peak()):
            out[n] = amp
    return out


def _solver_record(kind: str, mesh: Mesh2D, settings: EMSettings, problem_materials: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    from aipe_magnetics import __version__
    return {"id": SOLVER_ID, "version": SOLVER_VERSION, "package_version": __version__, "kind": kind,
            "formulation": "2-D planar magnetic vector potential A_y, linear triangles (P1); magnetostatic and time-harmonic "
                           "eddy-current (solid conductors with per-turn current constraints)",
            "symmetry": "half model x >= 0 with A = 0 on x = 0 (antisymmetric winding currents)",
            "boundary_conditions": "A = 0 on the outer air box", "mesher": mesh.generator, "mesh_settings": mesh.settings,
            "elements": mesh.n_elements, "nodes": mesh.n_nodes, "mesh_quality": mesh.quality(), "materials": problem_materials,
            "settings": settings.to_json(), "linear_solver": "scipy.sparse.linalg.spsolve (SuperLU)", **extra}


# -------------------------------------------------------------------------------------------------
# transformer
# -------------------------------------------------------------------------------------------------
def analyse_transformer(geometry: Any, *, frequency: float, core: CoreModel, primary_current: PiecewiseLinear,
                        primary_flux_linkage_peak: float, turns_ratio: float, winding_temperature: float,
                        settings: EMSettings | None = None, primary: str = "primary", secondary: str = "secondary") -> EMReport:
    """Two-winding transformer study.

    ``primary_current`` is the load current of the primary (the secondary carries ``turns_ratio x`` it, opposite sign;
    magnetising current is computed separately). ``primary_flux_linkage_peak`` is the peak flux linkage imposed by the
    primary voltage (``V D / (2 f)`` for a square wave of duty ``D``), which scales the magnetising field.
    """
    settings = settings or EMSettings()
    t_start = time.perf_counter()
    info = _winding_info(geometry)
    if primary not in info or secondary not in info:
        raise InvalidInputError(f"geometry needs windings {primary!r} and {secondary!r}; has {list(info)}")
    mesh_settings = settings.mesh or default_mesh_settings(geometry, frequency, winding_temperature, settings.harmonics_fea)
    warnings: list[str] = []
    mesh = _mesh_with_budget(geometry.section, mesh_settings, settings.mesher, warnings)
    mu = {core.material_id: core.mu_r}
    sigma = {"copper": copper_conductivity(winding_temperature)}
    lengths = turn_lengths(geometry)
    q: dict[str, dict[str, Any]] = {}
    n1, n2 = info[primary]["turns"], info[secondary]["turns"]
    depth = geometry.section.depth

    # 1. magnetising field (primary only, 1 A)
    mag = solve_magnetostatic(EMProblem(mesh=mesh, mu_r=mu, currents={primary: 1.0}))
    lam_unit = mag.flux_linkage_per_depth(primary) * depth
    e_mag = mag.energy_3d(info[primary]["mlt"])
    q["magnetising_inductance"] = _qv(lam_unit, "H", "fea", "Magnetising inductance (primary), flux linkage x core depth")
    q["magnetising_inductance_energy"] = _qv(2 * e_mag["total_J"], "H", "fea", "Self inductance of the primary from the 2.5-D field energy "
                                             "(includes the primary's own leakage around the end turns)")
    scale = primary_flux_linkage_peak / lam_unit
    q["magnetising_current_peak"] = _qv(scale, "A", "fea", "Peak magnetising current for the imposed flux linkage")
    b_mag_pk = scale * mag.B_magnitude
    sat = _saturation(mesh, b_mag_pk, core.bsat)
    q["flux_density_peak_max"] = _qv(sat["B_peak_max_T"], "T", "fea", "Maximum element peak flux density (at the sharp inner window corners: "
                                     "singular in the idealised outline, mesh dependent; informational)")
    q["flux_density_peak_local"] = _qv(sat["B_peak_p995_T"], "T", "fea", "Local peak flux density, 99.5 % core-area percentile (robust)")
    q["flux_density_peak_mean"] = _qv(sat["B_peak_area_mean_T"], "T", "fea", "Area-averaged peak flux density in the core section")
    q["saturation_local_margin"] = _qv(sat["local_margin"], "1", "fea", "1 - B_peak,local (99.5 % percentile) / Bsat(T)")
    q["core_fraction_above_0p8_bsat"] = _qv(sat["fraction_above_0p8_bsat"], "1", "fea", "Core section area fraction with B_peak > 0.8 Bsat")
    centre = mesh.region_mask("core", names=["core_bottom_centre_leg", "core_top_centre_leg"])
    q["flux_density_peak_centre_leg"] = _qv(float(np.sum(b_mag_pk[centre] * mesh.areas()[centre]) / np.sum(mesh.areas()[centre])), "T", "fea",
                                            "Area-averaged peak flux density in the centre leg")
    p_core, pv, core_by_region, cw = _core_loss_from_field(mesh, b_mag_pk, core, frequency)
    warnings += [f"core loss: {w}" for w in cw]
    q["core_loss"] = _qv(p_core, "W", "fea_post_processed", f"Core loss: {core.loss.material_id} Steinmetz/iGSE evaluated element by "
                         f"element on the FEA peak flux density ({core.shape.kind} flux), integrated over the core volume",
                         by_region=core_by_region)
    # 2. leakage field (ampere-turn balanced, 1 A primary)
    pol = {secondary: -1.0}
    cur_unit = {primary: 1.0, secondary: turns_ratio}
    leak = solve_magnetostatic(EMProblem(mesh=mesh, mu_r=mu, currents=cur_unit, polarity=pol))
    mlt_mean = 0.5 * (info[primary]["mlt"] + info[secondary]["mlt"])
    e_leak = leak.energy_3d(mlt_mean)
    q["leakage_inductance"] = _qv(2 * e_leak["total_J"], "H", "fea", "Leakage inductance referred to the primary (field energy with "
                                  "ampere-turn balanced currents; window energy x mean turn length)")
    space = geometry.model.winding_space()
    l1d = leakage_inductance_1d(_stack_layers_1d(geometry, pol, cur_unit), space["height"], mlt_mean)
    q["leakage_inductance_1d"] = _qv(l1d, "H", "analytical", "1-D window-field leakage inductance (layer stack, ampere-turn balance)")
    # 3. winding losses
    i_p = primary_current
    h_p = _harmonics(i_p, settings.harmonics_analytic)
    amps = {primary: h_p, secondary: {n: a * turns_ratio for n, a in h_p.items()}}
    winding_losses: dict[str, dict[str, float]] = {}
    for name in (primary, secondary):
        data = info[name]
        r_dc = _r_dc(data, winding_temperature)
        cond = data["conductor"]
        a = amps[name]
        harm = [(n, v) for n, v in a.items() if n > 0]
        entry: dict[str, float] = {"r_dc_Ohm": r_dc, "current_rms_A": math.sqrt(a.get(0, 0.0) ** 2 + sum(v * v / 2 for _, v in harm))}
        winding_losses[name] = entry
    # analytical Dowell per layer with the ampere-turn balanced MMF profile (interleaving accounted for)
    rho_w = F.copper_resistivity(winding_temperature)
    stack = []
    for layer in sorted(geometry.layers, key=lambda l: l.d_in):
        c = layer.conductor
        if c.kind not in ("round", "foil"):
            stack.append({"winding": layer.winding, "ampere_turns": pol.get(layer.winding, 1.0) * cur_unit[layer.winding] * layer.turns,
                          "turns": layer.turns, "r_dc": 0.0, "kind": "foil", "size": 1e-6, "pitch": 1.0})
            continue
        stack.append({"winding": layer.winding, "ampere_turns": pol.get(layer.winding, 1.0) * cur_unit[layer.winding] * layer.turns,
                      "turns": layer.turns, "r_dc": rho_w * layer.turns * layer.mean_turn_length / c.copper_area(), "kind": c.kind,
                      "size": c.bare_diameter if c.kind == "round" else c.thickness,
                      "pitch": c.axial_pitch() if c.kind == "round" else space["height"]})
    dow = dowell_layer_stack_loss(stack, [(k, v) for k, v in h_p.items() if k > 0], frequency, winding_temperature, dc=h_p.get(0, 0.0))
    for name in (primary, secondary):
        if info[name]["conductor"].kind in ("round", "foil"):
            winding_losses[name]["dowell_W"] = dow.get(name, 0.0)
    solid = any(info[w]["conductor"].kind in ("round", "foil", "rectangular") for w in (primary, secondary))
    b_by_harmonic: dict[int, np.ndarray] = {}
    harmonic_solutions: list[dict[str, Any]] = []
    fundamental = h_p.get(1, max(h_p.values()) if h_p else 0.0)
    for n, amp in sorted(h_p.items()):
        if n == 0:
            continue
        b_by_harmonic[n] = amp * leak.B_magnitude
    calibrated_share = 0.0
    if solid:
        h_cond = mesh_settings.conductor
        fea_loss = {primary: 0.0, secondary: 0.0}
        per_harmonic: dict[int, dict[str, Any]] = {}
        deferred: list[tuple[int, float]] = []
        skipped_energy = 0.0
        for n, amp in sorted(h_p.items()):
            if n == 0:
                continue
            if amp < settings.harmonic_threshold * fundamental:
                skipped_energy += amp * amp / 2
                continue
            if n <= settings.harmonics_fea and skin_depth(n * frequency, winding_temperature) >= 2.0 * h_cond:
                hs = solve_harmonic(EMProblem(mesh=mesh, mu_r=mu, currents={primary: amp, secondary: amp * turns_ratio}, polarity=pol,
                                              sigma=sigma), n * frequency)
                losses = hs.winding_loss(lengths)
                for w in losses:
                    fea_loss[w] += losses[w]
                per_harmonic[n] = {"method": "fea", "losses_W": losses}
                b_by_harmonic[n] = np.hypot(np.abs(hs.B[:, 0]), np.abs(hs.B[:, 1]))  # eddy reaction of solid conductors included
                harmonic_solutions.append({"harmonic": n, "amplitude_A": amp, "frequency_Hz": n * frequency, "solve_time_s": hs.solve_time_s,
                                           "losses_W": losses, "unknowns": hs.meta["unknowns"], "method": "fea"})
            else:
                deferred.append((n, amp))
        # FEA-calibrated per-harmonic Dowell for harmonics the mesh does not resolve (or above harmonics_fea)
        dow_by_n = {n: dowell_layer_stack_loss(stack, [(n, a)], frequency, winding_temperature) for n, a in h_p.items() if n > 0}
        calib: dict[str, float] = {}
        for w in (primary, secondary):
            ref_fea = sum(v["losses_W"].get(w, 0.0) for v in per_harmonic.values())
            ref_dow = sum(dow_by_n[n].get(w, 0.0) for n in per_harmonic)
            calib[w] = ref_fea / ref_dow if ref_dow > 0 else 1.0
        deferred_total = 0.0
        for n, amp in deferred:
            est = {w: dow_by_n[n].get(w, 0.0) * calib[w] for w in (primary, secondary)}
            for w in est:
                fea_loss[w] += est[w]
            deferred_total += sum(est.values())
            harmonic_solutions.append({"harmonic": n, "amplitude_A": amp, "frequency_Hz": n * frequency, "losses_W": est,
                                       "method": "fea_calibrated_dowell", "calibration": dict(calib)})
        for w in (primary, secondary):
            if info[w]["conductor"].kind in ("round", "foil", "rectangular"):
                dc = amps[w].get(0, 0.0)
                winding_losses[w]["fea_W"] = fea_loss[w] + winding_losses[w]["r_dc_Ohm"] * dc * dc
                winding_losses[w]["fea_calibration_factor"] = calib[w]
        total_fea = sum(fea_loss.values())
        calibrated_share = deferred_total / total_fea if total_fea > 0 else 0.0
        total_ac = sum(v * v / 2 for n, v in h_p.items() if n > 0)
        if total_ac > 0 and skipped_energy / total_ac > 0.005:
            warnings.append(f"harmonics below the amplitude threshold carry {100 * skipped_energy / total_ac:.1f} % of the primary AC "
                            "mean-square current; their loss is not included")
    litz = _litz_losses(mesh, geometry, b_by_harmonic, amps, frequency, winding_temperature, lengths)
    for w, entry in litz.items():
        if info[w]["conductor"].kind == "litz":
            winding_losses[w].update({f"litz_{k}": v for k, v in entry.items()})
            winding_losses[w]["fea_post_processed_W"] = entry["total_W"]
            share = entry.get("proximity_beyond_strand_validity_W", 0.0) / entry["total_W"] if entry["total_W"] > 0 else 0.0
            if share > 0.05:
                warnings.append(f"{w}: {100 * share:.0f} % of the litz loss comes from harmonics where the strand diameter exceeds the skin "
                                "depth; the low-frequency strand formula overestimates those terms (conservative)")
    total_w = 0.0
    method = "fea"
    for w, entry in winding_losses.items():
        v = entry.get("fea_W", entry.get("fea_post_processed_W"))
        if v is None:
            v = entry.get("dowell_W", 0.0)
            method = "analytical"
        elif "fea_post_processed_W" in entry and method == "fea":
            method = "fea_post_processed"
        entry["selected_W"] = v
        total_w += v
    if method == "fea" and calibrated_share > 0.01:
        method = "fea_post_processed"
    q["winding_loss"] = _qv(total_w, "W", method, "Winding loss (all windings; per-winding method in winding_losses)",
                            fea_calibrated_harmonic_share=calibrated_share)
    if any(info[w]["conductor"].kind == "litz" for w in (primary, secondary)):
        stack = []
        for layer in sorted(geometry.layers, key=lambda l: l.d_in):
            c = layer.conductor
            stack.append({"x0": layer.d_in, "x1": layer.d_out, "ampere_turns": pol.get(layer.winding, 1.0) * cur_unit[layer.winding] * layer.turns,
                          "turns": layer.turns, "strands": c.strand_count if c.kind == "litz" else 0, "strand_diameter": c.strand_diameter,
                          "mlt": layer.mean_turn_length})
        p1d = litz_proximity_loss_1d(stack, space["height"], [(k, v) for k, v in h_p.items() if k > 0], frequency, winding_temperature)
        q["litz_proximity_loss_1d"] = _qv(p1d, "W", "analytical", "Litz strand proximity loss with the 1-D window field (all litz windings, "
                                          "same harmonics) - reference for the field-based value")
    q["total_loss"] = _qv(total_w + p_core, "W", "fea_post_processed", "Core + winding loss from the field solutions")
    # analytical references
    ae = geometry.derived["centre_leg_area_m2"]
    b_avg = primary_flux_linkage_peak / (n1 * ae)
    ev = core.loss.loss_density(frequency, b_avg, core.temperature, core.shape)
    q["core_loss_uniform_leg_flux"] = _qv(float(ev.loss_density) * geometry.derived["core_volume_m3"], "W", "analytical",
                                          "Core loss with the centre-leg flux density applied uniformly to the geometric core volume")
    q["flux_density_peak_faraday"] = _qv(b_avg, "T", "analytical", "Faraday peak flux density in the centre leg (lambda / (N A_leg))")
    # element fields for export and thermal coupling
    fields = {"B_magnetising_unit_T_per_A": mag.B, "B_magnetising_peak_T": b_mag_pk, "B_leakage_unit_T_per_A": leak.B,
              "core_loss_density_W_per_m3": pv, "A_magnetising_unit": mag.A, "A_leakage_unit": leak.A}
    solver = _solver_record("transformer", mesh, settings, {"core": {"material": core.material_id, "mu_r": core.mu_r},
                                                            "copper_conductivity_S_per_m": sigma["copper"]},
                            {"excitations": {"magnetising": {primary: "1 A, scaled to the imposed flux linkage"},
                                             "leakage": {primary: "1 A", secondary: f"{turns_ratio:g} A (opposite)"},
                                             "harmonics": harmonic_solutions},
                             "temperatures_K": {"core": core.temperature, "winding": winding_temperature}})
    diagnostics = {"solve_time_s": time.perf_counter() - t_start, "mesh": mesh.quality(), "harmonic_solves": harmonic_solutions,
                   "harmonic_energy_captured": i_p.harmonic_energy_fraction(settings.harmonics_analytic)}
    return EMReport(kind="transformer", quantities=q, fields=fields, mesh=mesh, solver=solver, diagnostics=diagnostics,
                    warnings=warnings, winding_losses=winding_losses)


# -------------------------------------------------------------------------------------------------
# inductor
# -------------------------------------------------------------------------------------------------
def analyse_inductor(geometry: Any, *, frequency: float, core: CoreModel, current: PiecewiseLinear, winding_temperature: float,
                     settings: EMSettings | None = None, winding: str | None = None,
                     datasheet_area: float | None = None, datasheet_length: float | None = None) -> EMReport:
    """Gapped inductor study: inductance (with fringing), saturation at peak current, core loss of the ripple, winding loss."""
    settings = settings or EMSettings()
    t_start = time.perf_counter()
    info = _winding_info(geometry)
    winding = winding or next(iter(info))
    data = info[winding]
    n = data["turns"]
    mesh_settings = settings.mesh or default_mesh_settings(geometry, frequency, winding_temperature, settings.harmonics_fea)
    warnings: list[str] = []
    mesh = _mesh_with_budget(geometry.section, mesh_settings, settings.mesher, warnings)
    mu = {core.material_id: core.mu_r}
    sigma = {"copper": copper_conductivity(winding_temperature)}
    lengths = turn_lengths(geometry)
    q: dict[str, dict[str, Any]] = {}
    depth = geometry.section.depth
    unit = solve_magnetostatic(EMProblem(mesh=mesh, mu_r=mu, currents={winding: 1.0}))
    e = unit.energy_3d(data["mlt"])
    l_energy = 2 * e["total_J"]
    l_flux = unit.flux_linkage_per_depth(winding) * depth
    q["inductance"] = _qv(l_energy, "H", "fea", "Inductance from the 2.5-D field energy (core/gap/fringing x depth, window x mean turn length)")
    q["inductance_flux_linkage"] = _qv(l_flux, "H", "fea", "Inductance from the flux linkage over the core depth only (lower bound)")
    g = geometry.model.gap
    ae_leg = geometry.derived["centre_leg_area_m2"]
    ae = datasheet_area or ae_leg
    le = datasheet_length or 2 * (geometry.model.core.window_height_half + geometry.model.core.back_thickness)
    # series reluctance: centre gap (area A_leg) + outer-leg gap (area of both outer legs) + core path l_e / (mu_r A_e)
    ao = geometry.derived["outer_leg_area_m2"]
    rel = g.centre_gap / (MU_0 * ae_leg) + g.outer_gap / (MU_0 * ao) + le / (MU_0 * core.mu_r * ae)
    l_rel = n * n / rel
    q["inductance_reluctance"] = _qv(l_rel, "H", "analytical", "Reluctance model without fringing (gap(s) + l_e/mu_r)")
    if g.centre_gap > 0:
        fr = 1 + g.centre_gap / math.sqrt(ae_leg) * math.log(2 * (2 * geometry.model.core.window_height_half) / g.centre_gap)
        q["fringing_factor_mclyman"] = _qv(fr, "1", "analytical", "McLyman fringing factor 1 + g/sqrt(A) ln(2 G / g)")
        q["inductance_reluctance_fringing"] = _qv(l_rel * fr, "H", "analytical", "Reluctance model x McLyman fringing factor")
    q["fringing_factor_fea"] = _qv(l_energy / l_rel, "1", "fea", "FEA inductance / reluctance-model inductance")
    i_pk = current.peak()
    i_dc = current.mean()
    ripple = current.peak_to_peak()
    b_pk = i_pk * unit.B_magnitude
    sat = _saturation(mesh, b_pk, core.bsat)
    q["flux_density_peak_max"] = _qv(sat["B_peak_max_T"], "T", "fea", "Maximum element flux density at peak current (sharp-corner singularity, "
                                     "mesh dependent; informational)")
    q["flux_density_peak_local"] = _qv(sat["B_peak_p995_T"], "T", "fea", "Local flux density at peak current, 99.5 % core-area percentile (linear core)")
    q["flux_density_peak_mean"] = _qv(sat["B_peak_area_mean_T"], "T", "fea", "Area-averaged flux density in the core at peak current")
    q["saturation_local_margin"] = _qv(sat["local_margin"], "1", "fea", "1 - B_local (99.5 % percentile) / Bsat(T) at the peak current (linear core)")
    q["core_fraction_above_0p8_bsat"] = _qv(sat["fraction_above_0p8_bsat"], "1", "fea", "Core area fraction above 0.8 Bsat at peak current")
    nl_info: dict[str, Any] = {}
    if settings.nonlinear and core.curve is not None:
        try:
            sol_pk = solve_magnetostatic(EMProblem(mesh=mesh, mu_r=mu, currents={winding: i_pk}, curves={core.material_id: core.curve}))
            sol_lo = solve_magnetostatic(EMProblem(mesh=mesh, mu_r=mu, currents={winding: 0.97 * i_pk}, curves={core.material_id: core.curve}))
            lam_pk = sol_pk.flux_linkage_per_depth(winding) * depth
            lam_lo = sol_lo.flux_linkage_per_depth(winding) * depth
            l_inc = (lam_pk - lam_lo) / (0.03 * i_pk)
            ratio = l_inc / l_flux
            q["inductance_incremental_at_peak_nonlinear"] = _qv(l_inc * l_energy / l_flux, "H", "fea",
                                                                 "Incremental inductance at the peak current (nonlinear solve with the assumed tanh curve, "
                                                                 "scaled from the flux-linkage to the energy-based inductance)")
            q["inductance_drop_at_peak"] = _qv(1.0 - ratio, "1", "fea", "Relative drop of the incremental inductance at peak current (nonlinear)")
            bmax_nl = float(sol_pk.B_magnitude[_core_mask(mesh)].max())
            q["flux_density_peak_max_nonlinear"] = _qv(bmax_nl, "T", "fea", "Maximum local flux density at peak current (nonlinear core)")
            nl_info = {"iterations": sol_pk.iterations, "residual": sol_pk.residual, "curve": core.curve.to_json()}
        except Exception as error:  # noqa: BLE001 - reported, never hidden
            warnings.append(f"nonlinear saturation solve failed: {type(error).__name__}: {error}")
    # core loss of the ripple (small-signal linear field)
    b_ac = 0.5 * ripple * unit.B_magnitude
    p_core, pv, core_by_region, cw = _core_loss_from_field(mesh, b_ac, core, frequency)
    warnings += [f"core loss: {w}" for w in cw]
    q["core_loss"] = _qv(p_core, "W", "fea_post_processed", "Core loss of the ripple flux (iGSE on the local AC flux amplitude; DC bias "
                         "effect on loss not modelled)", by_region=core_by_region)
    b_ac_avg = l_energy * ripple / 2 / (n * ae_leg)
    ev = core.loss.loss_density(frequency, b_ac_avg, core.temperature, core.shape)
    q["core_loss_uniform_leg_flux"] = _qv(float(ev.loss_density) * geometry.derived["core_volume_m3"], "W", "analytical",
                                          "Core loss with the centre-leg AC flux amplitude applied uniformly to the core volume")
    q["flux_density_ac_amplitude_centre_leg"] = _qv(b_ac_avg, "T", "analytical", "L dI / (2 N A_leg)")
    # winding loss
    harm = _harmonics(current, settings.harmonics_analytic)
    r_dc = _r_dc(data, winding_temperature)
    cond = data["conductor"]
    entry: dict[str, float] = {"r_dc_Ohm": r_dc, "current_rms_A": current.rms(), "dc_loss_W": r_dc * i_dc * i_dc}
    harm_ac = [(k, v) for k, v in harm.items() if k > 0]
    if cond.kind in ("round", "foil"):
        size = cond.bare_diameter if cond.kind == "round" else cond.thickness
        pitch = cond.axial_pitch() if cond.kind == "round" else geometry.model.winding_space()["height"]
        dow = dowell_harmonic_loss(r_dc=r_dc, dc=harm.get(0, 0.0), harmonics=harm_ac, frequency=frequency, temperature=winding_temperature,
                                   layers=data["layers"], conductor_kind=cond.kind, size=size, pitch=pitch)
        entry["dowell_W"] = dow["loss_W"]
        entry["dowell_effective_ac_factor"] = dow["effective_ac_factor"]
    b_by_harmonic: dict[int, np.ndarray] = {k: v * unit.B_magnitude for k, v in harm_ac}
    harmonic_solutions = []
    calibrated_share = 0.0
    if cond.kind in ("round", "foil", "rectangular"):
        fundamental = harm.get(1, 0.0)
        h_cond = mesh_settings.conductor
        fea_ac = 0.0
        skipped = 0.0
        solved_fea, solved_dow = 0.0, 0.0
        deferred = []

        def dowell_n(k: int, amp: float) -> float:
            if cond.kind not in ("round", "foil"):
                return 0.0
            size = cond.bare_diameter if cond.kind == "round" else cond.thickness
            pitch = cond.axial_pitch() if cond.kind == "round" else geometry.model.winding_space()["height"]
            return dowell_harmonic_loss(r_dc=r_dc, dc=0.0, harmonics=[(k, amp)], frequency=frequency, temperature=winding_temperature,
                                        layers=data["layers"], conductor_kind=cond.kind, size=size, pitch=pitch)["loss_W"]

        for k, amp in harm_ac:
            if amp < settings.harmonic_threshold * fundamental:
                skipped += amp * amp / 2
                continue
            if k <= settings.harmonics_fea and skin_depth(k * frequency, winding_temperature) >= 2.0 * h_cond:
                hs = solve_harmonic(EMProblem(mesh=mesh, mu_r=mu, currents={winding: amp}, sigma=sigma), k * frequency)
                losses = hs.winding_loss(lengths)
                fea_ac += losses.get(winding, 0.0)
                solved_fea += losses.get(winding, 0.0)
                solved_dow += dowell_n(k, amp)
                harmonic_solutions.append({"harmonic": k, "amplitude_A": amp, "frequency_Hz": k * frequency, "loss_W": losses.get(winding, 0.0),
                                           "solve_time_s": hs.solve_time_s, "unknowns": hs.meta["unknowns"], "method": "fea"})
            else:
                deferred.append((k, amp))
        calib = solved_fea / solved_dow if solved_dow > 0 else 1.0
        deferred_total = 0.0
        for k, amp in deferred:
            est = dowell_n(k, amp) * calib
            fea_ac += est
            deferred_total += est
            harmonic_solutions.append({"harmonic": k, "amplitude_A": amp, "frequency_Hz": k * frequency, "loss_W": est,
                                       "method": "fea_calibrated_dowell", "calibration": calib})
        entry["fea_ac_W"] = fea_ac
        entry["fea_W"] = fea_ac + entry["dc_loss_W"]
        entry["fea_calibration_factor"] = calib
        calibrated_share = deferred_total / entry["fea_W"] if entry["fea_W"] > 0 else 0.0
        total_ac = sum(v * v / 2 for _, v in harm_ac)
        if total_ac > 0 and skipped / total_ac > 0.005:
            warnings.append(f"harmonics below the amplitude threshold carry {100 * skipped / total_ac:.1f} % of the AC mean-square current")
    litz = _litz_losses(mesh, geometry, b_by_harmonic, {winding: harm}, frequency, winding_temperature, lengths)
    if cond.kind == "litz" and winding in litz:
        entry.update({f"litz_{k}": v for k, v in litz[winding].items()})
        entry["fea_post_processed_W"] = litz[winding]["total_W"]
    selected = entry.get("fea_W", entry.get("fea_post_processed_W", entry.get("dowell_W")))
    method = "fea" if "fea_W" in entry else ("fea_post_processed" if "fea_post_processed_W" in entry else "analytical")
    if method == "fea" and calibrated_share > 0.01:
        method = "fea_post_processed"
    entry["selected_W"] = selected
    q["winding_loss"] = _qv(selected, "W", method, "Winding loss: DC + ripple harmonics (gap fringing field included in the FEA)",
                            fea_calibrated_harmonic_share=calibrated_share)
    q["total_loss"] = _qv(selected + p_core, "W", "fea_post_processed", "Core + winding loss from the field solutions")
    fields = {"B_unit_T_per_A": unit.B, "B_peak_T": b_pk, "B_ac_amplitude_T": b_ac, "core_loss_density_W_per_m3": pv, "A_unit": unit.A}
    solver = _solver_record("inductor", mesh, settings, {"core": {"material": core.material_id, "mu_r": core.mu_r},
                                                         "copper_conductivity_S_per_m": sigma["copper"]},
                            {"excitations": {"unit": {winding: "1 A"}, "peak_current_A": i_pk, "ripple_A": ripple, "harmonics": harmonic_solutions},
                             "nonlinear": nl_info, "temperatures_K": {"core": core.temperature, "winding": winding_temperature}})
    diagnostics = {"solve_time_s": time.perf_counter() - t_start, "mesh": mesh.quality(), "harmonic_solves": harmonic_solutions,
                   "harmonic_energy_captured": current.harmonic_energy_fraction(settings.harmonics_analytic)}
    return EMReport(kind="inductor", quantities=q, fields=fields, mesh=mesh, solver=solver, diagnostics=diagnostics, warnings=warnings,
                    winding_losses={winding: entry})


__all__ = ["CoreModel", "EMReport", "EMSettings", "LIMITATIONS_2D", "SOLVER_ID", "SOLVER_VERSION", "analyse_inductor", "analyse_transformer",
           "default_mesh_settings", "turn_lengths"]
