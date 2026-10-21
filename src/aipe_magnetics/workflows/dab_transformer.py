"""End-to-end workflow, case 1: 10 kW / 800 V : 400 V / 100 kHz dual-active-bridge transformer.

Steps (each recorded in the Magnetic Engineering State and the report):

1. read and validate the specification; list every assumption and open question (waveform / modulation, operating
   range, isolation, cooling are *not* defined by "800 V / 400 V" and are stated explicitly);
2. retrieve materials and cores through the Database interface (stage-1 filter with rejection reasons);
3. generate analytical design candidates (turns, conductor, arrangement, material, core);
4. select feasible winding arrangements (P-S, interleaved P-S-P, ... per the design space);
5. generate parametric CAD geometry of the selected design (STEP / STL) and of the v0.2 baseline;
6. run the electromagnetic finite-element analysis (magnetising and leakage fields, harmonic eddy currents / litz
   strand losses, field-based core loss);
7. estimate losses and run the thermal analysis (2.5-D thermal FE coupled electrothermally; thermal network for
   comparison and a natural-convection sensitivity case);
8. optimise feasible design parameters (multi-fidelity engine);
9. compare against the v0.2 baseline (E 65/32/27, N87, 21 : 10, litz) - both its v0.2 reported numbers and its v0.3
   re-evaluation with the same models as the optimised design;
10. export engineering results and evidence (MES JSON, AIPE-Core state, CAD, solver configurations, field data,
    plots, BOM, engineering report, verification report).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from aipe_magnetics import __version__
from aipe_magnetics.design.common import core_candidates, material_contexts
from aipe_magnetics.design.specs import DABTransformerSpec
from aipe_magnetics.design.transformers import candidate as tc
from aipe_magnetics.geometry.conductors import litz_for_area
from aipe_magnetics.optimisation.engine import Objective, run_multifidelity
from aipe_magnetics.state.model import (
    Component,
    Excitation,
    HistoryEvent,
    LossBreakdown,
    MagneticEngineeringState,
    OpenQuestion,
    OperatingPoint,
    OptimisationRecord,
    ProvenanceRecord,
    Requirements,
    StateAssumption,
    ToolRef,
    UncertaintyItem,
    VerificationLevelStatus,
    VerificationRecord,
    Waveform,
)
from aipe_magnetics.workflows import common as wc

V02_BASELINE = {"core": "core.tdk-e65-32-27", "material": "N87", "n1": 21, "n2": 10, "strand_diameter": 0.1e-3, "current_density": 4e6,
                "arrangement": "P-S", "reported": {"total_loss_W": 39.2, "core_loss_W": 21.5, "copper_loss_W": 17.7, "flux_density_peak_T": 0.178,
                                                   "window_fill": 0.30, "source": "docs/demo-dab-10kw-report.md (v0.2.0 release run)"}}


@dataclass
class WorkflowOptions:
    shortlist: int = 3
    refine_iterations: int = 2
    cad_formats: tuple[str, ...] = ("step", "stl")
    cad_backend: str = "auto"
    plots: bool = True
    mesh_refinement_check: bool = True
    natural_convection_sensitivity: bool = True
    baseline_comparison: bool = True
    created_at: str | None = None   # RFC 3339, supplied by the caller (the actual run time); never a build-time literal
    em_settings: Any = None
    log: Callable[[str], None] | None = None
    max_candidates: int | None = None


@dataclass
class WorkflowResult:
    status: str
    output_dir: Path
    state: MagneticEngineeringState | None
    summary: dict[str, Any]
    artifacts: list[dict[str, Any]]
    errors: list[str] = field(default_factory=list)

    def compact(self) -> dict[str, Any]:
        return {"status": self.status, "output_dir": str(self.output_dir), "summary": self.summary,
                "artifacts": self.artifacts, "errors": self.errors}


def _waveform(pl: Any, quantity: str = "current", description: str | None = None) -> Waveform:
    return Waveform(quantity=quantity, unit=pl.unit, breakpoints=[[t, v] for t, v in pl.points], rms=pl.rms(), peak=pl.peak(), mean=pl.mean(),
                    description=description or pl.label)


def run_dab_workflow(output_dir: str | Path, spec: DABTransformerSpec | dict | None = None, options: WorkflowOptions | None = None,
                     *, strict: bool = False) -> WorkflowResult:
    t0 = time.perf_counter()
    opts = options or WorkflowOptions()
    if not opts.created_at:
        from aipe_magnetics.common.errors import InvalidInputError
        raise InvalidInputError("WorkflowOptions.created_at (RFC 3339 timestamp of this run) must be supplied by the caller; "
                                "the workflow never invents a timestamp")
    log = opts.log or (lambda m: None)
    spec = spec if isinstance(spec, DABTransformerSpec) else DABTransformerSpec.model_validate(spec or {})
    store = wc.ArtifactStore(Path(output_dir))
    history: list[HistoryEvent] = []
    errors: list[str] = []

    # 1. specification ------------------------------------------------------------------------------
    report = spec.assumption_report()
    spec_id = store.json("specification/spec.json", spec.model_dump(mode="json"), "specification", "Validated DAB transformer specification (SI)")
    store.json("specification/assumptions.json", report, "specification", "Assumed values and open questions of the specification")
    history.append(HistoryEvent(stage="specification", description=f"specification validated; {len(report['assumed'])} assumed values, "
                                f"{len(report['open_questions'])} open questions ({len(report['critical_open'])} critical)"))
    log(f"1 specification: {len(report['open_questions'])} open questions")
    if strict and report["critical_open"]:
        return WorkflowResult("rejected", store.root, None, {"open_questions": report["open_questions"]}, store.compact(),
                              [f"strict mode: critical open questions {report['critical_open']}"])
    ctx = tc.DABContext.from_spec(spec)

    # 2. database retrieval -------------------------------------------------------------------------
    window = (spec.cooling.ambient, spec.limits.max_core_hotspot)
    mats, mat_rej = material_contexts(spec.space.materials, spec.frequency, window)
    cores, core_rej = core_candidates(spec.space.cores, mats, spec.space.material_offer_policy)
    sel_id = store.json("selection/database-selection.json", {
        "materials_accepted": [{"id": m.id, "grade": m.data.grade, "bsat_points": m.data.saturation_points, "loss_sets": [s.id for s in m.data.steinmetz_sets],
                                "sources": [s.reference for s in m.data.sources][:4]} for m in mats],
        "materials_rejected": mat_rej, "cores_accepted": [c.id for c in cores], "cores_rejected": core_rej,
        "filter": {"frequency_Hz": spec.frequency, "temperature_window_K": window, "material_offer_policy": spec.space.material_offer_policy}},
        "selection_record", "Stage-1 database filter: accepted / rejected materials and cores with reasons")
    history.append(HistoryEvent(stage="database_selection", description=f"{len(mats)} material(s), {len(cores)} core(s) accepted"))
    log(f"2 database: materials {[m.id for m in mats]}, cores {[c.id for c in cores]}")
    if not mats or not cores:
        return WorkflowResult("rejected", store.root, None, {"materials_rejected": mat_rej, "cores_rejected": core_rej}, store.compact(),
                              ["no material or core passes the database filter"])

    # 3-4 + 8. candidates, arrangements, optimisation -------------------------------------------------
    cands = tc.generate_candidates(spec, mats, cores)
    if opts.max_candidates:
        cands = cands[: opts.max_candidates]
    history.append(HistoryEvent(stage="candidate_generation", description=f"{len(cands)} analytical candidates generated"))
    objectives = [Objective("total_loss_W", "W", description="nominal-point core + winding loss"),
                  Objective("volume_cm3", "cm^3", description="bounding volume"), Objective("mass_g", "g", description="core + copper mass"),
                  Objective("hotspot_C", "degC", description="hottest point (worst-case operating point)")]
    em_settings = opts.em_settings
    res = run_multifidelity(candidates=cands, candidate_id=lambda c: c.id, params=lambda c: c.params(),
                            evaluate_fast=lambda c: tc.evaluate_analytic(c, spec),
                            evaluate_detailed=lambda c, a: tc.evaluate_detailed(c, spec, analytic=a, em_settings=em_settings),
                            neighbours=lambda c: tc.neighbours(c, spec), objectives=objectives, primary="total_loss_W",
                            shortlist=opts.shortlist, refine_iterations=opts.refine_iterations, log=log,
                            group_key=lambda c: (c.core.id, c.material.id, c.kind, c.arrangement),
                            filters={"materials": [m.id for m in mats], "cores": [c.id for c in cores]})
    rows = [e.row() for e in res.evaluations]
    cand_id = store.csv("optimisation/candidates.csv", [{"id": r["id"], "fidelity": r["fidelity"], "stage": r["stage"], "feasible": r["feasible"],
                                                         **{k: round(v, 6) for k, v in r["objectives"].items()}, "violations": "; ".join(r["violations"]),
                                                         **{f"p_{k}": v for k, v in r["params"].items()}} for r in rows],
                        "results", "Every evaluated candidate with fidelity, feasibility, objectives and violations")
    opt_summary = res.summary(top=10)
    opt_id = store.json("optimisation/summary.json", opt_summary, "results", "Optimisation summary: stages, ranking, Pareto fronts, history")
    arrangements = {}
    for e in res.ranked("analytical"):
        arrangements.setdefault(e.params["arrangement"], e.row())
    history.append(HistoryEvent(stage="optimisation", description=f"{res.stages['analytical_evaluations']} analytical and "
                                f"{res.stages['detailed_evaluations']} detailed evaluations; selected {res.selected.candidate_id if res.selected else 'none'}"))
    log(f"8 optimisation: {res.stages}")
    if res.selected is None:
        return WorkflowResult("rejected", store.root, None, {"optimisation": opt_summary}, store.compact(),
                              ["no candidate satisfied the detailed constraints"])
    sel = res.selected
    cand = sel.candidate
    raw = sel.raw
    geo = raw["geometry"]
    rep_nom = raw["reports"]["nominal"]
    rep_w = raw["reports"]["worst_case"]
    thermal = raw["reports"]["thermal"]

    # 5. CAD ---------------------------------------------------------------------------------------------
    from aipe_magnetics.geometry.export import export_cad
    cad = export_cad(geo, store.root / "cad", basename="dab-transformer-selected", formats=opts.cad_formats, backend=opts.cad_backend)  # type: ignore[arg-type]
    cad_ids = [store.file(Path(f["path"]), "cad", f["description"] + f" [{f['backend']}]", "geometry.export") for f in cad["files"]]
    geom_id = store.json("geometry/selected-geometry.json", {"parameters": geo.model.parameters(), "model": geo.model.model_dump(mode="json"),
                                                             "summary": geo.summary(), "section": geo.section.to_json()},
                         "specification", "Editable parametric model (regenerate with ECoreMagnetic.model_validate(model).build())")
    history.append(HistoryEvent(stage="geometry", description=f"parametric model {geo.model.parameter_hash()} exported: "
                                + ", ".join(Path(f["path"]).name for f in cad["files"])))

    # 6-7. EM and thermal artifacts ---------------------------------------------------------------------
    em_cfg = store.json("solver/em-solver-config.json", rep_nom.solver, "solver_configuration", "2-D FEA configuration (nominal point)")
    em_fields = store.file(rep_nom.save_fields(store.root / "fields/em-nominal.npz"), "field_data", "Nodes, triangles, regions, A and B fields, "
                           "element core-loss density (nominal point)", "simulation.electromagnetic")
    em_res_id = store.json("results/em-nominal.json", rep_nom.compact(), "results", "Electromagnetic results, nominal operating point")
    em_w_res_id = store.json("results/em-worst-case.json", rep_w.compact(), "results", "Electromagnetic results, worst-case operating point")
    th_cfg = store.json("solver/thermal-solver-config.json", {**thermal.summary(), "cooling": spec.cooling.model_dump()}, "solver_configuration",
                        "2.5-D thermal FE configuration")
    th_fields = store.file(wc.save_thermal_fields(thermal, store.root / "fields/thermal.npz"), "field_data", "Thermal FE temperature field",
                           "simulation.thermal")
    th_res_id = store.json("results/thermal.json", thermal.summary(), "results", "Thermal FE results (worst case, converged electrothermal)")
    history.append(HistoryEvent(stage="electromagnetic_analysis", description=f"2-D FEA: {rep_nom.mesh.n_elements} elements, "
                                f"{len(rep_nom.diagnostics['harmonic_solves'])} harmonic solve(s)"))
    history.append(HistoryEvent(stage="thermal_analysis", description="2.5-D thermal FE coupled electrothermally; energy balance residual "
                                f"{thermal.balance['relative_residual']:.1e}"))

    # natural-convection sensitivity (analytical network on the selected geometry with the FEA losses)
    sensitivity: dict[str, Any] = {}
    if opts.natural_convection_sensitivity:
        from aipe_magnetics.design.common import pack_conductivities
        from aipe_magnetics.simulation.thermal.convection import Cooling
        from aipe_magnetics.simulation.thermal.network import solve_thermal_network
        wl = {w: e["selected_W"] for w, e in rep_w.winding_losses.items()}
        nat = solve_thermal_network(geo, core_loss=rep_w.value("core_loss"), winding_losses=wl, cooling=Cooling(mode="natural", ambient=spec.cooling.ambient),
                                    k_core=cand.material.thermal_conductivity(), k_bobbin=0.3, k_insulation=0.12, k_pack=pack_conductivities())
        forced = solve_thermal_network(geo, core_loss=rep_w.value("core_loss"), winding_losses=wl, cooling=spec.cooling.to_cooling(),
                                       k_core=cand.material.thermal_conductivity(), k_bobbin=0.3, k_insulation=0.12, k_pack=pack_conductivities())
        sensitivity = {"natural_convection_hotspot_C": max(nat.hotspots.values()) - 273.15,
                       "assumed_cooling_hotspot_C_network": max(forced.hotspots.values()) - 273.15,
                       "losses_W": rep_w.value("core_loss") + sum(wl.values()), "note": "fixed losses (no electrothermal update) - indicative"}
        store.json("results/cooling-sensitivity.json", {"natural": nat.summary(), "assumed": forced.summary(), **sensitivity}, "results",
                   "Cooling sensitivity: thermal network under natural convection vs the assumed cooling")

    # mesh refinement check ---------------------------------------------------------------------------
    convergence: dict[str, Any] = {}
    if opts.mesh_refinement_check:
        from aipe_magnetics.simulation.electromagnetic.analysis import EMSettings, analyse_transformer, default_mesh_settings
        from dataclasses import replace
        base = default_mesh_settings(geo, cand.frequency, thermal.region_stats.get("winding:primary", {"T_mean_K": 373.15})["T_mean_K"], 7)
        fine = replace(base, refinement=0.6)
        rep_fine = analyse_transformer(geo, frequency=cand.frequency, core=_core_model(cand, rep_nom), primary_current=_nominal_current(ctx, cand),
                                       primary_flux_linkage_peak=spec.v1 * 0.5 / (2 * cand.frequency), turns_ratio=cand.n1 / cand.n2,
                                       winding_temperature=rep_nom.solver["temperatures_K"]["winding"], settings=EMSettings(mesh=fine))
        rep_base = analyse_transformer(geo, frequency=cand.frequency, core=_core_model(cand, rep_nom), primary_current=_nominal_current(ctx, cand),
                                       primary_flux_linkage_peak=spec.v1 * 0.5 / (2 * cand.frequency), turns_ratio=cand.n1 / cand.n2,
                                       winding_temperature=rep_nom.solver["temperatures_K"]["winding"], settings=EMSettings(mesh=base))
        for key in ("core_loss", "winding_loss", "leakage_inductance", "magnetising_inductance", "flux_density_peak_centre_leg"):
            a, b = rep_base.value(key), rep_fine.value(key)
            convergence[key] = {"default_mesh": a, "refined_mesh": b, "relative_change": (b - a) / a if a else None}
        convergence["elements"] = {"default": rep_base.mesh.n_elements, "refined": rep_fine.mesh.n_elements}
        store.json("results/mesh-refinement.json", convergence, "results", "Mesh refinement check (element size x 0.6)")

    # 9. baseline comparison ------------------------------------------------------------------------------
    baseline: dict[str, Any] = {}
    if opts.baseline_comparison:
        baseline = _baseline(spec, mats, store, opts, em_settings, log)

    # plots --------------------------------------------------------------------------------------------
    plot_ids: list[str] = []
    if opts.plots:
        from aipe_magnetics.visualisation import plots as pl
        if pl.matplotlib_available():
            n = cand.n1 / cand.n2
            ops, _ = ctx.operating_points(n, cand.frequency)
            plot_ids += [store.file(pl.plot_geometry_3d(geo, store.root / "plots/geometry-3d.png", title="Selected DAB transformer"), "plot", "3-D model"),
                         store.file(pl.plot_section(geo.section, store.root / "plots/section.png", title="Selected design: half cross-section"), "plot", "2-D section"),
                         store.file(pl.plot_element_field(rep_nom.mesh, rep_nom.fields["B_magnetising_peak_T"], store.root / "plots/flux-density.png",
                                                          title="Peak magnetising flux density (2-D FEA)", label="B (T)", roles=("core",)), "plot", "B field"),
                         store.file(pl.plot_element_field(rep_nom.mesh, abs_b(rep_nom.fields["B_leakage_unit_T_per_A"]) * 1e3,
                                                          store.root / "plots/leakage-field.png", title="Leakage field per ampere (primary)",
                                                          label="|B| (mT/A)"), "plot", "leakage field"),
                         store.file(pl.plot_temperature(thermal, store.root / "plots/temperature.png"), "plot", "temperature field"),
                         store.file(pl.plot_waveforms({f"primary ({op.label})": op.current for op in ops}, store.root / "plots/waveforms.png",
                                                      period=1 / cand.frequency, title="DAB primary current (SPS)"), "plot", "waveforms"),
                         store.file(pl.plot_candidates(rows, store.root / "plots/candidates.png", pareto_ids={e.candidate_id for e in res.pareto_analytic},
                                                       selected_id=sel.candidate_id, title="DAB candidates: loss vs volume"), "plot", "candidates"),
                         store.file(pl.plot_convergence(res.history, store.root / "plots/optimisation.png"), "plot", "optimisation progress"),
                         store.file(pl.plot_electrothermal(raw["metrics"]["electrothermal"]["history"], store.root / "plots/electrothermal.png"), "plot",
                                    "electrothermal iteration")]
            breakdown = {"selected (v0.3 FEA)": {"core": rep_nom.value("core_loss"), **{f"winding {w}": e["selected_W"] for w, e in rep_nom.winding_losses.items()}}}
            if baseline.get("v03_detailed"):
                b = baseline["v03_detailed"]
                breakdown["v0.2 design (v0.3 FEA)"] = {"core": b["core_loss_W"], **{f"winding {w}": v for w, v in b["winding_losses_W"].items()}}
            if baseline.get("v02_reported"):
                r = baseline["v02_reported"]
                breakdown["v0.2 reported (analytical)"] = {"core": r["core_loss_W"], "winding (all)": r["copper_loss_W"]}
            plot_ids.append(store.file(pl.plot_loss_breakdown(breakdown, store.root / "plots/loss-breakdown.png", title="Loss breakdown (nominal point)"),
                                       "plot", "loss breakdown"))
        else:
            errors.append("plots not generated: matplotlib is not installed (pip install 'aipe-magnetics[viz]')")

    # BOM -------------------------------------------------------------------------------------------------
    bom = _bom(geo, cand)
    store.csv("bom/bom.csv", bom, "bom", "Bill of materials (preliminary; quantities from the parametric model)")

    # 10. state + reports --------------------------------------------------------------------------------
    n = cand.n1 / cand.n2
    ops, L_series = ctx.operating_points(n, cand.frequency)
    op_models = [OperatingPoint(id=f"op.{op.label.replace(' ', '-').lower()}", label=op.label, frequency=wc.qv(cand.frequency, "Hz", "specified"),
                                waveforms={"primary_current": _waveform(op.current), "secondary_current": _waveform(op.current.scaled(n, label="secondary current"))},
                                quantities={"v1": wc.qv(op.v1, "V", "specified"), "v2": wc.qv(op.v2, "V", "specified"),
                                            "phase_shift": wc.qv(op.phi, "rad", "analytical"), "power": wc.qv(op.power, "W", "specified")},
                                role="nominal" if op.label == "nominal" else "corner") for op in ops]
    worst = max(ops, key=lambda op: op.current.rms())
    for m in op_models:
        if m.label == worst.label and m.role != "nominal":
            m.role = "worst_case"
    assumptions = [StateAssumption(id=f"spec.{a['field']}", statement=a["statement"], status="assumed", critical=a["critical"],
                                   basis="default") for a in report["assumed"]]
    assumptions += [StateAssumption(id="model.residual-gap", statement=f"residual mating gap {tc.DEFAULT_SPACER * 1e6:.0f} um per leg (~20 um total, "
                                    "consistent with the datasheet AL of the ungapped set)", basis="engineering_judgement"),
                    StateAssumption(id="model.mu-linear", statement=f"core permeability linear, mu_r = {cand.material.data.initial_permeability:g} "
                                    "(datasheet initial permeability)", basis="datasheet"),
                    StateAssumption(id="model.series-inductance", statement=f"total series inductance {L_series * 1e6:.1f} uH referred to the primary; "
                                    f"external inductor {raw['metrics']['external_series_inductance_H'] * 1e6:.1f} uH after the transformer leakage",
                                    basis="model_assumption")]
    if cand.standard_size_equivalent:
        assumptions.append(StateAssumption(id="model.standard-size-equivalent", statement=f"{cand.core.name} outline assumed available in "
                                           f"{cand.material.data.grade} (IEC 62317 standard size; dimensions of record {cand.core.id})",
                                           basis="engineering_judgement", critical=True))
    questions = [OpenQuestion(id=f"q.{q['field']}", field=q["field"], question=q["question"], assumed_answer=q["assumed_answer"], critical=q["critical"])
                 for q in report["open_questions"]]
    em_solver = wc.em_solver_record("solver.em-fea", rep_nom, em_cfg)
    th_solver = wc.thermal_solver_record("solver.thermal-fe", thermal, th_cfg)
    from aipe_magnetics.state.model import SolverConfiguration
    an_solver = SolverConfiguration(id="solver.analytical", domain="electromagnetic", backend="aipe.magnetics.design.transformers.candidate",
                                    version=__version__, fidelity="analytical", formulation="Faraday / iGSE on Ve / per-layer Dowell or 1-D litz / "
                                    "1-D leakage / thermal network", status="completed")
    em_set = wc.em_result_set("results.em-nominal", "solver.em-fea", "op.nominal", rep_nom, [em_fields, em_res_id])
    em_set_w = wc.em_result_set("results.em-worst-case", "solver.em-fea", next(m.id for m in op_models if m.label == worst.label), rep_w, [em_w_res_id])
    an_quant = {k: wc.qv(v, u, "analytical") for k, v, u in (
        ("flux_density_peak", sel.metrics.get("flux_density_peak_T") or res_analytic_metric(res, sel, "flux_density_peak_T"), "T"),
        ("leakage_inductance_1d", res_analytic_metric(res, sel, "leakage_inductance_1d_H"), "H"),
        ("total_loss", res_analytic_metric(res, sel, "total_loss_W"), "W")) if v is not None}
    from aipe_magnetics.state.model import ResultSet
    an_set = ResultSet(id="results.analytical", solver_ref="solver.analytical", operating_point_ref="op.nominal", quantities=an_quant)
    th_set = wc.thermal_result_set("results.thermal", "solver.thermal-fe", next(m.id for m in op_models if m.label == worst.label), thermal,
                                   [th_fields, th_res_id])
    losses = LossBreakdown(operating_point_ref="op.nominal", core=wc.qv(rep_nom.value("core_loss"), "W", "fea_post_processed"),
                           windings={w: wc.qv(e["selected_W"], "W", rep_nom.quantities["winding_loss"]["method"]) for w, e in rep_nom.winding_losses.items()},
                           total=wc.qv(rep_nom.value("total_loss"), "W", "fea_post_processed"))
    checks = [{"kind": "detailed_constraints", "id": "detailed-constraints", "status": "pass" if sel.feasible else "fail", "violations": sel.violations},
              {"kind": "energy_balance", "id": "thermal-energy-balance", "status": "pass" if abs(thermal.balance["relative_residual"]) < 1e-6 else "fail",
               "value": thermal.balance["relative_residual"]},
              {"kind": "electrothermal", "id": "electrothermal-convergence", "status": "pass" if raw["metrics"]["electrothermal"]["converged"] else "fail",
               "surrogate_relative_error": raw["metrics"]["electrothermal"]["surrogate_relative_error"]}]
    if convergence:
        worst_change = max(abs(v["relative_change"]) for k, v in convergence.items() if isinstance(v, dict) and v.get("relative_change") is not None)
        checks.append({"kind": "mesh_refinement", "id": "mesh-refinement", "status": "pass" if worst_change < 0.03 else "partial",
                       "max_relative_change": worst_change})
    ext_refs = {"leakage_fea_vs_1d": (rep_nom.value("leakage_inductance"), rep_nom.value("leakage_inductance_1d")),
                "core_loss_fea_vs_uniform": (rep_nom.value("core_loss"), rep_nom.value("core_loss_uniform_leg_flux"))}
    if "litz_proximity_loss_1d" in rep_nom.quantities:
        prox = sum(e.get("litz_proximity_W", 0.0) for e in rep_nom.winding_losses.values())
        ext_refs["litz_proximity_fea_vs_1d"] = (prox, rep_nom.value("litz_proximity_loss_1d"))
    checks.append({"kind": "reference_comparison", "id": "fea-vs-analytical", "status": "recorded",
                   "comparisons": {k: {"fea": a, "reference": b, "ratio": a / b if b else None} for k, (a, b) in ext_refs.items()}})
    verification = build_verification_record(checks, schema_validated=None)
    unc = [UncertaintyItem(quantity="ferrite thermal conductivity", kind="bounds", lower=1.0, upper=5.0, unit="W/m/K",
                           basis="TDK general data range; nominal 4 W/m/K used"),
           UncertaintyItem(quantity="winding pack thermal conductivity", kind="bounds", lower=0.3, upper=1.5, unit="W/m/K", basis="engineering judgement"),
           UncertaintyItem(quantity="core loss", kind="unknown", basis="Steinmetz / iGSE parameter sets; uncertainty not quantified by the sources")]
    if sensitivity:
        unc.append(UncertaintyItem(quantity="hot spot under natural convection", kind="sensitivity", lower=None,
                                   upper=sensitivity["natural_convection_hotspot_C"], unit="degC",
                                   basis="thermal network with natural convection instead of the assumed forced cooling"))
    state = MagneticEngineeringState(
        id=f"mes.{spec.name}", title="10 kW DAB transformer - AIPE Magnetics v0.3 workflow", created_at=opts.created_at, lifecycle_stage="exported",
        component=Component(type="transformer", application="DAB isolation transformer", topology="dual-active-bridge (SPS)",
                            windings=["primary", "secondary"]),
        requirements=Requirements(electrical={"power": wc.qv(spec.power, "W", "specified"), "frequency": wc.qv(spec.frequency, "Hz", "specified"),
                                              "v1": wc.qv(spec.v1, "V", "specified"), "v2": wc.qv(spec.v2, "V", "specified")},
                                  constraints={"max_winding_hotspot": wc.qv(spec.limits.max_winding_hotspot, "K", "assumed"),
                                               "max_core_hotspot": wc.qv(spec.limits.max_core_hotspot, "K", "assumed"),
                                               "max_current_density": wc.qv(spec.limits.max_current_density, "A/m^2", "assumed"),
                                               "flux_fraction_of_bsat": wc.qv(spec.limits.flux_fraction_of_bsat, "1", "assumed"),
                                               "max_window_fill": wc.qv(spec.limits.max_window_fill, "1", "assumed")},
                                  environment={"ambient": wc.qv(spec.cooling.ambient, "K", "assumed")}, cooling=spec.cooling.model_dump(),
                                  isolation=spec.isolation.model_dump(), specified_fields=spec.specified),
        excitation=Excitation(fundamental_frequency=wc.qv(cand.frequency, "Hz", "specified"), operating_points=op_models,
                              description="SPS DAB, square voltages, series-inductor current (magnetising current computed separately)"),
        assumptions=assumptions, open_questions=questions,
        materials=[wc.material_selection(m, selected=(m.id == cand.material.id)) for m in mats],
        cores=[wc.core_selection(c, selected=(c.id == cand.core.id)) for c in cores],
        windings=wc.winding_definitions(geo, rep_nom.solver["temperatures_K"]["winding"]),
        geometry=wc.geometry_record(geo, cad_ids + [geom_id]),
        solvers=[an_solver, em_solver, th_solver],
        results={"electromagnetic": [an_set, em_set, em_set_w], "thermal": [th_set], "losses": [losses],
                 "comparison": {"baseline": baseline, "mesh_refinement": convergence, "cooling_sensitivity": sensitivity}},
        optimisation=OptimisationRecord(strategy="multi-fidelity: database filter -> analytical screening -> Pareto / ranking -> FEA + thermal FE "
                                        "shortlist -> local refinement", objectives=[o.__dict__ for o in objectives], primary_objective="total_loss_W",
                                        constraints=["saturation", "current density", "window fill", "turns ratio", "radial fit", "hot spots",
                                                     "leakage <= series inductance", "creepage (when specified)"],
                                        variables={"materials": [m.id for m in mats], "cores": [c.id for c in cores], "conductors": spec.space.conductors,
                                                   "arrangements": spec.space.arrangements, "frequencies": spec.space.frequencies or [spec.frequency]},
                                        stages=res.stages, history=res.history, ranked=opt_summary["top_detailed"],
                                        pareto=opt_summary["pareto_analytic"], selected=sel.candidate_id,
                                        rejection_reasons=opt_summary["rejection_reasons"], artifacts=[cand_id, opt_id]),
        verification=verification,
        provenance=ProvenanceRecord(tool=ToolRef(version=__version__), inputs=[spec_id, sel_id],
                                    database={"package_version": __version__, "materials": [m.id for m in mats], "cores": [c.id for c in cores]},
                                    software=wc.software_versions(), sources=list(cand.material.data.sources)[:8] + list(cand.core.sources)[:4]),
        uncertainty=unc, artifacts=store.records, history=history)
    report_md = render_dab_report(spec, report, res, sel, rep_nom, rep_w, thermal, baseline, convergence, sensitivity, cad, arrangements, bom, L_series)
    store.text("report/engineering-report.md", report_md, "report", "Engineering summary report")
    ver_md = render_verification_report(state, checks, convergence, ext_refs)
    store.text("report/verification-report.md", ver_md, "verification_report", "Verification report")
    state = state.model_copy(update={"artifacts": list(store.records)})
    from aipe_magnetics.state.schema import dump_state, validate_state
    errs = validate_state(state.model_dump(mode="json", exclude_none=True))
    if errs:
        errors += [f"state validation: {e}" for e in errs[:10]]
    # the software/schema level is recorded from the validation that was actually executed on this state, never asserted up front
    state = state.model_copy(update={"verification": build_verification_record(checks, schema_validated=not errs)})
    final_errs = validate_state(state.model_dump(mode="json", exclude_none=True))
    if final_errs:
        errors += [f"state validation (after recording the verification outcome): {e}" for e in final_errs[:10]]
    dump_state(state, store.root / "state" / "magnetic-engineering-state.json", overwrite=True)
    store.file(store.root / "state" / "magnetic-engineering-state.json", "state", "Magnetic Engineering State 0.3.0")
    try:
        from aipe_magnetics.state.core_adapter import mes_to_core_state
        core_state = mes_to_core_state(state, created_at=opts.created_at, converter_topology="dual-active-bridge", role="DAB isolation transformer")
        store.json("state/aipe-core-engineering-state.json", core_state, "state", "AIPE-Core Engineering State 0.1.0 (validated)")
    except Exception as error:  # noqa: BLE001 - reported, the MES remains the primary record
        errors.append(f"AIPE-Core export failed: {type(error).__name__}: {error}")
    summary = {
        "selected": sel.candidate_id, "params": sel.params, "feasible": sel.feasible, "violations": sel.violations,
        "nominal": {k: rep_nom.quantities[k] for k in ("core_loss", "winding_loss", "total_loss", "leakage_inductance", "magnetising_inductance",
                                                        "flux_density_peak_centre_leg", "flux_density_peak_local") if k in rep_nom.quantities},
        "hotspots_C": {"winding": raw["metrics"]["hotspot_winding_K"] - 273.15, "core": raw["metrics"]["hotspot_core_K"] - 273.15},
        "series_inductance_uH": L_series * 1e6, "external_inductor_uH": raw["metrics"]["external_series_inductance_H"] * 1e6,
        "optimisation": res.stages, "baseline": {k: v for k, v in baseline.items() if k in ("v02_reported", "v03_analytic", "v03_detailed")},
        "mesh_refinement": convergence, "cooling_sensitivity": sensitivity, "open_questions": [q.question for q in questions if q.critical],
        "elapsed_s": round(time.perf_counter() - t0, 1)}
    store.json("results/workflow-summary.json", summary, "results", "Compact workflow summary")
    log(f"10 exported {len(store.records)} artifacts in {summary['elapsed_s']} s")
    return WorkflowResult("ok" if not errors else "ok_with_warnings", store.root, state, summary, store.compact(), errors)


def build_verification_record(checks: list[dict[str, Any]], *, schema_validated: bool | None) -> VerificationRecord:
    """Verification record whose statuses follow from what was executed.

    * ``software_schema``: ``passed`` only when ``schema_validated`` is True (the state validated against the MES 0.3.0 JSON Schema
      in this run), ``failed`` when False, ``partial`` while the validation has not run yet (``None``);
    * ``numerical_physical``: the workflow's own self-consistency checks (energy balance, electrothermal convergence, mesh refinement,
      FEA-vs-analytical comparison) - these are internal consistency checks, so the level is ``partial`` at best; it is ``failed``
      when one of them failed. Independent numerical verification lives in the bundled evidence of the solvers, not here;
    * ``external_reference``: ``partial`` (comparison with analytical references, discrepancies reported, not tuned);
    * ``experimental``: ``not_performed``.
    """
    self_checks = [c for c in checks if c.get("status") not in (None, "recorded")]
    if any(c["status"] == "fail" for c in self_checks):
        numerical_status = "failed"
    else:
        numerical_status = "partial"
    if schema_validated is None:
        schema_status, schema_statement = "partial", "JSON Schema validation of this state has not been recorded yet."
    elif schema_validated:
        schema_status, schema_statement = "passed", "This state validated against the MES 0.3.0 JSON Schema and the model's cross-reference rules in this run."
    else:
        schema_status, schema_statement = "failed", "This state did not validate against the MES 0.3.0 JSON Schema (see the workflow errors)."
    return VerificationRecord(levels=[
        VerificationLevelStatus(level="software_schema", status=schema_status, statement=schema_statement),
        VerificationLevelStatus(level="numerical_physical", status=numerical_status,
                                statement="Self-consistency checks of this run only (thermal energy balance, electrothermal convergence, mesh refinement, "
                                          "FEA versus analytical references; see checks). These are the workflow's own checks, not independent "
                                          "verification; the solvers' independent evidence is the bundled verification registry."),
        VerificationLevelStatus(level="external_reference", status="partial", statement="FEA compared with published analytical references "
                                "(1-D leakage, Dowell, Ve-based core loss, litz 1-D); discrepancies are reported, not tuned."),
        VerificationLevelStatus(level="experimental", status="not_performed", statement="No prototype was built or measured.")],
        checks=checks, manufacturing_readiness="preliminary_design",
        statement="Preliminary engineering design: insulation / creepage, mechanical, safety and production verification are not performed; "
                  "isolation requirements are an open question.")


def abs_b(b: Any):
    import numpy as np
    return np.hypot(b[:, 0], b[:, 1])


def res_analytic_metric(res: Any, sel: Any, key: str) -> float | None:
    for e in res.evaluations:
        if e.candidate_id == sel.candidate_id and e.fidelity == "analytical":
            return e.metrics.get(key)
    return None


def _nominal_current(ctx: Any, cand: Any):
    ops, _ = ctx.operating_points(cand.n1 / cand.n2, cand.frequency)
    return ops[0].current


def _core_model(cand: Any, rep: Any):
    from aipe_magnetics.design.common import flux_shape_square
    from aipe_magnetics.simulation.electromagnetic.analysis import CoreModel
    t = rep.solver["temperatures_K"]["core"]
    bsat, _ = cand.material.bsat(t)
    return CoreModel(material_id=cand.material.id, mu_r=cand.material.data.initial_permeability, bsat=bsat, loss=cand.material.loss,
                     shape=flux_shape_square(0.5), temperature=t)


def _baseline(spec: DABTransformerSpec, mats: list[Any], store: Any, opts: WorkflowOptions, em_settings: Any, log: Callable[[str], None]) -> dict[str, Any]:
    """v0.2 baseline: its reported analytical result, a fresh v0.2-API run, and its v0.3 re-evaluation (same models as the optimum)."""
    import aipe_magnetics.api as api
    from aipe_magnetics.database import core_data
    from aipe_magnetics.design.common import MaterialContext
    from aipe_magnetics.database import LossEvaluator, material_data
    from aipe_magnetics.engineering.demo_dab import SPEC as V02_SPEC
    winding = {"conductor_type": "litz", "strand_diameter": V02_SPEC["strand_diameter"], "current_density": V02_SPEC["current_density"],
               "fill_factor": V02_SPEC["fill_factor"]}
    v02 = api.design_transformer({"power": V02_SPEC["power"], "frequency": V02_SPEC["frequency"],
                                  "primary": {"name": "primary", "voltage": V02_SPEC["primary_voltage"], "winding": winding},
                                  "secondaries": [{"name": "secondary", "voltage": V02_SPEC["secondary_voltage"], "winding": winding}],
                                  "voltage_waveform": "square", "duty": 0.5, "material": "N87", "cores": None, "temperature": {"value": 100, "unit": "degC"},
                                  "flux_limit": {"mode": "fraction_of_saturation", "fraction": 0.7}, "loss_density_target": V02_SPEC["loss_density_target"],
                                  "loss_model": "auto", "max_candidates": 10, "policy": "warn"})
    out: dict[str, Any] = {"v02_reported": V02_BASELINE["reported"]}
    if v02.status == "ok":
        out["v02_rerun"] = {"selected": v02.data["selected"], "turns": {k: v.quantity.value for k, v in v02.values.items() if k.startswith("turns_")},
                            "core_loss_W": v02.value("core_loss").value, "copper_loss_W": v02.value("copper_loss").value,
                            "total_loss_W": v02.value("total_loss").value, "flux_density_peak_T": v02.value("flux_density_peak").value,
                            "current_basis": "P/V square-wave approximation", "mean_turn_length_basis": "datasheet lN = 188 mm"}
    store.json("baseline/v02-design-envelope.json", v02.to_json_dict(), "results", "v0.2 design_transformer envelope (re-run in v0.3)")
    md = material_data(V02_BASELINE["material"])
    mctx = next((m for m in mats if m.id == md.id), MaterialContext(data=md, loss=LossEvaluator(md.id, md.steinmetz_sets)))
    core = core_data(V02_BASELINE["core"])
    i1, i2 = 10e3 / 800, 10e3 / 400
    cand = tc.TransformerCandidate(material=mctx, core=core, n1=V02_BASELINE["n1"], n2=V02_BASELINE["n2"], kind="litz",
                                   primary=litz_for_area(i1 / V02_BASELINE["current_density"], V02_BASELINE["strand_diameter"]),
                                   secondary=litz_for_area(i2 / V02_BASELINE["current_density"], V02_BASELINE["strand_diameter"]),
                                   arrangement=V02_BASELINE["arrangement"], frequency=spec.frequency, tags={"baseline": "v0.2"})
    a = tc.evaluate_analytic(cand, spec)
    out["v03_analytic"] = {"id": cand.id, "feasible": a["feasible"], "violations": a["violations"], **a["objectives"],
                           "core_loss_W": a["metrics"].get("core_loss_W"), "winding_loss_W": a["metrics"].get("winding_loss_W")}
    try:
        d = tc.evaluate_detailed(cand, spec, analytic=a, em_settings=em_settings)
        rep = d["reports"]["nominal"]
        out["v03_detailed"] = {"id": cand.id, "feasible": d["feasible"], "violations": d["violations"], **d["objectives"],
                               "core_loss_W": rep.value("core_loss"), "winding_loss_W": rep.value("winding_loss"),
                               "winding_losses_W": {w: e["selected_W"] for w, e in rep.winding_losses.items()},
                               "leakage_inductance_H": rep.value("leakage_inductance"),
                               "hotspot_winding_C": d["metrics"]["hotspot_winding_K"] - 273.15, "hotspot_core_C": d["metrics"]["hotspot_core_K"] - 273.15,
                               "primary_current_rms_A": d["metrics"]["operating_points"][0]["primary_current_rms_A"]}
        from aipe_magnetics.geometry.export import export_cad
        cad = export_cad(d["geometry"], store.root / "cad", basename="dab-transformer-v02-baseline", formats=opts.cad_formats, backend=opts.cad_backend)  # type: ignore[arg-type]
        for f in cad["files"]:
            store.file(Path(f["path"]), "cad", f"v0.2 baseline geometry: {f['description']} [{f['backend']}]", "geometry.export")
    except Exception as error:  # noqa: BLE001 - reported
        out["v03_detailed_error"] = f"{type(error).__name__}: {error}"
    store.json("baseline/comparison.json", out, "results", "v0.2 baseline vs v0.3 re-evaluation")
    log("9 baseline comparison done")
    return out


def _bom(geo: Any, cand: Any) -> list[dict[str, Any]]:
    d = geo.derived
    rows = [{"item": 1, "part": f"Core set {cand.core.name}", "material": cand.material.data.grade, "quantity": geo.model.stack,
             "unit": "set", "note": ("standard-size equivalent of " + cand.core.id) if cand.standard_size_equivalent else cand.core.part_number or ""},
            {"item": 2, "part": "Coil former (bobbin)", "material": geo.model.bobbin.material, "quantity": 1, "unit": "pc",
             "note": "parametric bobbin; select a catalogue coil former with equal or better dimensions"}]
    for i, (name, w) in enumerate(d["windings"].items(), start=3):
        rows.append({"item": i, "part": f"Winding {name}: {w['conductor']}", "material": "copper", "quantity": round(w["conductor_length_m"], 3),
                     "unit": "m", "note": f"{w['turns']} turns, {w['layers']} layer(s), copper {w['copper_volume_m3'] * 8933 * 1e3:.1f} g"})
    ins = geo.model.layout.insulation if geo.model.layout else None
    if ins:
        rows.append({"item": len(rows) + 1, "part": "Insulation film", "material": ins.material, "quantity": "per layout", "unit": "-",
                     "note": f"layer {ins.layer_insulation * 1e3:.2f} mm, section {ins.section_insulation * 1e3:.2f} mm, wrap {ins.outer_wrap * 1e3:.2f} mm "
                             "(functional insulation only; isolation not designed)"})
    if geo.model.gap.spacer > 0:
        rows.append({"item": len(rows) + 1, "part": "Gap / mating", "material": "-", "quantity": geo.model.gap.spacer * 1e6, "unit": "um per leg",
                     "note": "residual mating gap (ungapped set)" if geo.model.gap.centre_grind == 0 else "spacer"})
    return rows


def render_dab_report(spec, report, res, sel, rep_nom, rep_w, thermal, baseline, convergence, sensitivity, cad, arrangements, bom, L_series) -> str:
    q = rep_nom.quantities
    m = sel.metrics
    lines = ["# 10 kW DAB transformer - AIPE Magnetics v0.3 engineering report", "",
             "**Status: preliminary engineering design.** Not manufacturing qualified: insulation / creepage / clearance, mechanical, safety and "
             "production verification are not performed and no prototype was measured.", "",
             "## 1. Specification and assumptions", "",
             f"Specified: {spec.power / 1e3:g} kW, {spec.v1:g} V / {spec.v2:g} V, {spec.frequency / 1e3:g} kHz. Everything else is assumed:", ""]
    lines += [f"- **{a['field']}**{' (critical)' if a['critical'] else ''}: {a['statement']}" for a in report["assumed"]]
    lines += ["", "Open questions (must be answered before qualification):", ""]
    lines += [f"- {q_['question']} *(assumed: {q_['assumed_answer']})*" for q_ in report["open_questions"]]
    lines += ["", "## 2. Operating waveforms (SPS model)", "",
              wc.md_table(["point", "V2 (V)", "phase (deg)", "I1 rms (A)", "I1 peak (A)", "ZVS prim./sec."],
                          [[o["label"], o["v2_V"], o["phase_shift_deg"], o["primary_current_rms_A"], o["primary_current_peak_A"],
                            f"{o['zvs_primary_indicated']}/{o['zvs_secondary_indicated']}"] for o in m["operating_points"]]),
              "", f"Total series inductance {L_series * 1e6:.1f} uH (referred to the primary); transformer leakage (FEA) "
                  f"{q['leakage_inductance']['value'] * 1e6:.2f} uH, external inductor {m['external_series_inductance_H'] * 1e6:.1f} uH.", "",
              "## 3. Optimisation", "",
              f"Generated {res.stages['generated']} candidates; {res.stages['analytical_feasible']} analytically feasible; "
              f"{res.stages['detailed_evaluations']} detailed (FEA + thermal FE) evaluations. Most frequent rejection reasons: "
              + ", ".join(f"{r['constraint']} ({r['candidates']})" for r in res.rejection_summary()[:5]) + ".", "",
              wc.md_table(["rank", "candidate", "fidelity", "loss (W)", "volume (cm3)", "mass (g)", "hot spot (degC)", "feasible"],
                          [[i + 1, e.candidate_id, e.fidelity, e.objectives.get("total_loss_W", float("nan")), e.objectives.get("volume_cm3", float("nan")),
                            e.objectives.get("mass_g", float("nan")), e.objectives.get("hotspot_C", float("nan")), e.feasible]
                           for i, e in enumerate(res.ranked("fea", feasible_only=False)[:6])]),
              "", "Best analytical candidate per winding arrangement:", "",
              wc.md_table(["arrangement", "candidate", "loss (W)", "hot spot (degC)"],
                          [[k, v["id"], v["objectives"]["total_loss_W"], v["objectives"]["hotspot_C"]] for k, v in arrangements.items()]),
              "", f"**Selected:** `{sel.candidate_id}`", "",
              "## 4. Electromagnetic results of the selected design (nominal point)", "",
              wc.md_table(["quantity", "value", "unit", "method"], [[k, v["value"], v["unit"], v["method"]] for k, v in q.items()]),
              "", "Winding losses:", "", wc.md_table(["winding", "loss (W)", "detail"],
                                                    [[w, e["selected_W"], ", ".join(f"{k}={v:.3g}" for k, v in e.items() if isinstance(v, float) and k != "selected_W")]
                                                     for w, e in rep_nom.winding_losses.items()]),
              "", "Warnings: " + ("; ".join(rep_nom.warnings) if rep_nom.warnings else "none"), "",
              "## 5. Thermal (worst-case point, electrothermal 2.5-D FE)", "",
              wc.md_table(["region", "T max (degC)", "T mean (degC)"], [[k, v["T_max_K"] - 273.15, v["T_mean_K"] - 273.15] for k, v in thermal.region_stats.items()]),
              "", f"Energy balance residual {thermal.balance['relative_residual']:.1e}; h(core) {thermal.balance['h_core_W_per_m2K']:.1f} W/m^2/K. "
                  f"Electrothermal iterations: {m['electrothermal']['iterations']}, surrogate vs re-solve error "
                  f"{100 * (m['electrothermal']['surrogate_relative_error'] or 0):+.2f} %.", ""]
    if sensitivity:
        lines += [f"Cooling sensitivity (thermal network, fixed losses {sensitivity['losses_W']:.1f} W): natural convection hot spot "
                  f"{sensitivity['natural_convection_hotspot_C']:.0f} degC vs {sensitivity['assumed_cooling_hotspot_C_network']:.0f} degC with the assumed "
                  "cooling - the cooling assumption is decisive.", ""]
    if convergence:
        lines += ["## 6. Numerical convergence (mesh refinement x 0.6)", "",
                  wc.md_table(["quantity", "default", "refined", "change"],
                              [[k, v["default_mesh"], v["refined_mesh"], f"{100 * v['relative_change']:+.2f} %"] for k, v in convergence.items()
                               if isinstance(v, dict) and "default_mesh" in v]), ""]
    if baseline:
        rows = []
        r = baseline.get("v02_reported", {})
        rows.append(["v0.2 reported (analytical, P/V current, datasheet lN)", r.get("total_loss_W"), r.get("core_loss_W"), r.get("copper_loss_W"), "-", "-"])
        if "v03_analytic" in baseline:
            a = baseline["v03_analytic"]
            rows.append(["v0.2 design, v0.3 analytical", a.get("total_loss_W"), a.get("core_loss_W"), a.get("winding_loss_W"), a.get("hotspot_C"),
                         "feasible" if a["feasible"] else "; ".join(a["violations"])[:80]])
        if "v03_detailed" in baseline:
            d = baseline["v03_detailed"]
            rows.append(["v0.2 design, v0.3 FEA + thermal FE", d.get("total_loss_W"), d.get("core_loss_W"), d.get("winding_loss_W"), d.get("hotspot_C"),
                         "feasible" if d["feasible"] else "; ".join(d["violations"])[:80]])
        rows.append(["v0.3 optimised (FEA + thermal FE)", rep_nom.value("total_loss"), rep_nom.value("core_loss"), rep_nom.value("winding_loss"),
                     sel.objectives.get("hotspot_C"), "feasible" if sel.feasible else "infeasible"])
        lines += ["## 7. Comparison with the v0.2 baseline (E 65/32/27, N87, 21 : 10, litz)", "",
                  wc.md_table(["design / model", "total (W)", "core (W)", "winding (W)", "hot spot (degC)", "status"], rows), "",
                  "Differences between the v0.2 report and the v0.3 re-evaluation of the same design come from the models, not from tuning: the "
                  "SPS current (RMS above P/V), current harmonics in the winding loss, the layer-resolved mean turn length of the parametric "
                  "geometry (instead of the datasheet coil-former lN) and the field-based core loss.", ""]
    lines += ["## 8. Artifacts", "", "CAD: " + ", ".join(Path(f["path"]).name + f" ({f['backend']})" for f in cad["files"]),
              "", "Bill of materials:", "", wc.md_table(["item", "part", "material", "quantity", "unit", "note"],
                                                       [[b["item"], b["part"], b["material"], b["quantity"], b["unit"], b["note"]] for b in bom]),
              "", "## 9. Limitations", ""]
    lines += [f"- {t}" for t in rep_nom.limitations + thermal.assumptions]
    lines += ["- Isolation (creepage / clearance / insulation system) is not designed: the brief does not define it.",
              "- Results are model estimates; no measurement-based (experimental) validation exists."]
    return "\n".join(lines) + "\n"


def render_verification_report(state: MagneticEngineeringState, checks: list[dict], convergence: dict, ext_refs: dict) -> str:
    lines = [f"# Verification report - {state.title}", "", f"State: `{state.id}` (MES {state.state_version}); manufacturing readiness: "
             f"**{state.verification.manufacturing_readiness}**.", "", "## Verification levels", ""]
    lines.append(wc.md_table(["level", "status", "statement"], [[l.level, l.status, l.statement] for l in state.verification.levels]))
    lines += ["", "## Checks", ""]
    for c in checks:
        lines.append(f"- **{c['id']}** ({c['kind']}): {c['status']} - " + ", ".join(f"{k}={v}" for k, v in c.items() if k not in ("id", "kind", "status")))
    lines += ["", "## FEA versus analytical / published references (discrepancies reported, not tuned)", ""]
    lines.append(wc.md_table(["comparison", "FEA", "reference", "ratio"], [[k, a, b, (a / b if b else float("nan"))] for k, (a, b) in ext_refs.items()]))
    lines += ["", "Experimental validation: **not performed** (no measurements exist for this design).", ""]
    return "\n".join(lines)


__all__ = ["V02_BASELINE", "WorkflowOptions", "WorkflowResult", "build_verification_record", "render_dab_report", "run_dab_workflow"]
