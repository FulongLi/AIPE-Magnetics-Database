"""Multi-fidelity design optimisation engine.

Strategy (never runs FEA on every database record):

1. **database filtering** - the caller supplies only candidates built from materials / cores that carry the data the
   models need (:func:`aipe_magnetics.design.common.material_contexts`, ``core_candidates``);
2. **fast analytical evaluation** of every generated candidate (closed forms, layer layout, thermal network);
3. **ranking and constraint screening** - feasibility, the primary objective and the Pareto front of all objectives;
4. **detailed evaluation** (2-D FEA + 2.5-D thermal FE, electrothermal) of a *shortlist*: the best feasible candidates by
   the primary objective plus the Pareto-front extremes, limited to ``shortlist`` designs;
   (optionally the best design of each design family first - ``group_key`` - so the shortlist is diverse);
5. **iterative refinement** - local moves around the best detailed design, screened analytically first and evaluated in
   detail only when the analytical model predicts an improvement, until no move improves or ``refine_iterations`` is
   reached; the best detailed design is the selection.

Every evaluation is kept in :attr:`OptimisationResult.evaluations` with its fidelity, so the ranking is traceable to
the model that produced it. The engine itself is problem independent: it calls user functions
``evaluate_fast(candidate)``, ``evaluate_detailed(candidate, fast_result)`` and ``neighbours(candidate)`` that return
``{"feasible", "violations", "objectives", "metrics"}`` dictionaries.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass
class Objective:
    name: str
    unit: str
    direction: str = "min"
    description: str = ""


@dataclass
class Evaluation:
    candidate_id: str
    params: dict[str, Any]
    fidelity: str
    feasible: bool
    violations: list[str]
    objectives: dict[str, float]
    metrics: dict[str, Any]
    stage: str
    duration_s: float
    candidate: Any = field(default=None, repr=False)
    raw: Any = field(default=None, repr=False)

    def row(self) -> dict[str, Any]:
        return {"id": self.candidate_id, "fidelity": self.fidelity, "stage": self.stage, "feasible": self.feasible,
                "violations": self.violations, "objectives": self.objectives, "params": self.params, "duration_s": round(self.duration_s, 4)}


def dominates(a: dict[str, float], b: dict[str, float], names: list[str]) -> bool:
    better = False
    for n in names:
        if a[n] > b[n] * (1 + 1e-12) + 1e-15:
            return False
        if a[n] < b[n] * (1 - 1e-12) - 1e-15:
            better = True
    return better


def pareto_front(evals: Iterable[Evaluation], names: list[str]) -> list[Evaluation]:
    pool = [e for e in evals if e.feasible and all(n in e.objectives and math.isfinite(e.objectives[n]) for n in names)]
    front = []
    for e in pool:
        if not any(dominates(o.objectives, e.objectives, names) for o in pool if o is not e):
            front.append(e)
    return sorted(front, key=lambda e: e.objectives[names[0]])


@dataclass
class OptimisationResult:
    objectives: list[Objective]
    primary: str
    evaluations: list[Evaluation]
    shortlist: list[str]
    selected: Evaluation | None
    pareto_analytic: list[Evaluation]
    pareto_detailed: list[Evaluation]
    history: list[dict[str, Any]]
    stages: dict[str, Any]
    filters: dict[str, Any] = field(default_factory=dict)

    def ranked(self, fidelity: str | None = None, feasible_only: bool = True) -> list[Evaluation]:
        pool = [e for e in self.evaluations if (fidelity is None or e.fidelity == fidelity) and (e.feasible or not feasible_only)]
        return sorted(pool, key=lambda e: (not e.feasible, e.objectives.get(self.primary, math.inf)))

    def summary(self, top: int = 10) -> dict[str, Any]:
        return {"primary_objective": self.primary, "objectives": [o.__dict__ for o in self.objectives], "stages": self.stages,
                "filters": self.filters,
                "selected": self.selected.row() if self.selected else None,
                "top_detailed": [e.row() for e in self.ranked("fea")[:top]],
                "top_analytic": [e.row() for e in self.ranked("analytical")[:top]],
                "pareto_analytic": [e.row() for e in self.pareto_analytic[:top]],
                "pareto_detailed": [e.row() for e in self.pareto_detailed],
                "history": self.history,
                "rejection_reasons": self.rejection_summary()}

    def rejection_summary(self, top: int = 12) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for e in self.evaluations:
            if e.fidelity == "analytical" and not e.feasible:
                for v in e.violations:
                    key = v.split(":")[0] if ":" in v else v.split(" ")[0]
                    counts[key] = counts.get(key, 0) + 1
        return [{"constraint": k, "candidates": v} for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:top]]


def run_multifidelity(*, candidates: list[Any], candidate_id: Callable[[Any], str], params: Callable[[Any], dict[str, Any]],
                      evaluate_fast: Callable[[Any], dict[str, Any]], evaluate_detailed: Callable[[Any, dict[str, Any]], dict[str, Any]],
                      neighbours: Callable[[Any], list[Any]] | None, objectives: list[Objective], primary: str,
                      shortlist: int = 4, refine_iterations: int = 3, log: Callable[[str], None] | None = None,
                      filters: dict[str, Any] | None = None, group_key: Callable[[Any], Any] | None = None) -> OptimisationResult:
    log = log or (lambda msg: None)
    names = [o.name for o in objectives]
    evaluations: list[Evaluation] = []
    seen: dict[str, Evaluation] = {}
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    def fast(c: Any, stage: str) -> Evaluation:
        cid = candidate_id(c)
        if cid in seen:
            return seen[cid]
        t = time.perf_counter()
        try:
            r = evaluate_fast(c)
        except Exception as error:  # noqa: BLE001 - an evaluation failure is a recorded infeasibility, never hidden
            r = {"feasible": False, "violations": [f"evaluation error: {type(error).__name__}: {error}"], "objectives": {}, "metrics": {}}
        e = Evaluation(cid, params(c), "analytical", r["feasible"], r["violations"], r["objectives"], r["metrics"], stage,
                       time.perf_counter() - t, candidate=c, raw=r)
        seen[cid] = e
        evaluations.append(e)
        return e

    def detailed(c: Any, fast_eval: Evaluation, stage: str) -> Evaluation:
        t = time.perf_counter()
        try:
            r = evaluate_detailed(c, fast_eval.raw)
        except Exception as error:  # noqa: BLE001
            r = {"feasible": False, "violations": [f"detailed evaluation error: {type(error).__name__}: {error}"], "objectives": {}, "metrics": {}}
        e = Evaluation(candidate_id(c), params(c), "fea", r["feasible"], r["violations"], r["objectives"], r["metrics"], stage,
                       time.perf_counter() - t, candidate=c, raw=r)
        evaluations.append(e)
        return e

    log(f"stage 2: analytical evaluation of {len(candidates)} candidate(s)")
    for c in candidates:
        fast(c, "analytical_screening")
    feasible = [e for e in evaluations if e.feasible]
    t_fast = time.perf_counter() - t0
    log(f"stage 3: {len(feasible)} feasible after constraint screening")
    ranked = sorted(feasible, key=lambda e: e.objectives[primary])
    front = pareto_front(feasible, names)
    history.append({"stage": "analytical_screening", "evaluated": len(evaluations), "feasible": len(feasible),
                    "best_primary": ranked[0].objectives[primary] if ranked else None, "elapsed_s": round(t_fast, 3)})
    picks: list[Evaluation] = []
    if group_key is not None:  # diversity: the best design of each design family first (core, material, conductor, arrangement ...)
        seen_groups: set[Any] = set()
        for e in ranked:
            g = group_key(e.candidate)
            if g not in seen_groups:
                seen_groups.add(g)
                picks.append(e)
            if len(picks) >= max(1, shortlist - 1):
                break
    else:
        for e in ranked[: max(1, shortlist - 1)]:
            picks.append(e)
    for name in names[1:]:  # Pareto extremes of the secondary objectives
        if front and len(picks) < shortlist:
            best = min(front, key=lambda e: e.objectives[name])
            if best not in picks:
                picks.append(best)
    for e in ranked:
        if len(picks) >= shortlist:
            break
        if e not in picks:
            picks.append(e)
    log(f"stage 4: detailed evaluation of {len(picks)} shortlisted candidate(s)")
    details = [detailed(e.candidate, e, "detailed_shortlist") for e in picks]
    feasible_d = [d for d in details if d.feasible]
    best = min(feasible_d, key=lambda e: e.objectives[primary]) if feasible_d else None
    history.append({"stage": "detailed_shortlist", "evaluated": len(details), "feasible": len(feasible_d),
                    "best_primary": best.objectives[primary] if best else None, "elapsed_s": round(time.perf_counter() - t0, 3)})
    if best is not None and neighbours is not None:
        for it in range(1, refine_iterations + 1):
            moves = neighbours(best.candidate)
            screened = [fast(m, f"refinement_{it}") for m in moves]
            reference = seen[best.candidate_id]
            promising = [s for s in screened if s.feasible and s.objectives.get(primary, math.inf) < reference.objectives[primary] * 0.999
                         and s.candidate_id not in {d.candidate_id for d in evaluations if d.fidelity == "fea"}]
            log(f"stage 5.{it}: {len(moves)} move(s), {len(promising)} promising")
            improved = False
            for s in sorted(promising, key=lambda s: s.objectives[primary])[:2]:
                d = detailed(s.candidate, s, f"refinement_{it}")
                if d.feasible and d.objectives[primary] < best.objectives[primary]:
                    best, improved = d, True
            history.append({"stage": f"refinement_{it}", "moves": len(moves), "promising": len(promising),
                            "best_primary": best.objectives[primary], "improved": improved, "elapsed_s": round(time.perf_counter() - t0, 3)})
            if not improved:
                break
    detailed_all = [e for e in evaluations if e.fidelity == "fea"]
    return OptimisationResult(objectives=objectives, primary=primary, evaluations=evaluations,
                              shortlist=[e.candidate_id for e in picks], selected=best, pareto_analytic=front,
                              pareto_detailed=pareto_front(detailed_all, names), history=history, filters=filters or {},
                              stages={"generated": len(candidates), "analytical_evaluations": sum(e.fidelity == "analytical" for e in evaluations),
                                      "analytical_feasible": sum(e.fidelity == "analytical" and e.feasible for e in evaluations),
                                      "detailed_evaluations": len(detailed_all), "detailed_feasible": sum(e.feasible for e in detailed_all),
                                      "elapsed_s": round(time.perf_counter() - t0, 2)})


__all__ = ["Evaluation", "Objective", "OptimisationResult", "dominates", "pareto_front", "run_multifidelity"]
