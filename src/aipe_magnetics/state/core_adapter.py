"""Adapter between the Magnetic Engineering State (MES 0.3.0) and the AIPE-Core Engineering State 0.1.0.

Export (:func:`mes_to_core_state`):

* the magnetics entry, the converter requirements and the assumption / source / analytical evidence are produced by
  the tested v0.2 adapter (:func:`aipe_magnetics.integrations.aipe_core.export.to_engineering_state`); the entry carries
  the *analytical* inductance / leakage / turns ratio of the selected design, and the finite-element values travel in its
  ``aipe.magnetics-detail`` extension with their method tags;
* every completed solver of the MES with artifacts becomes a Core ``simulation`` evidence record (versioned tool
  attribution, artifacts with SHA-256) and a completed ``simulation.runs[]`` entry (model = solver configuration file,
  outputs = field / result files); a ``simulation`` validation check records whether the detailed constraints passed;
* the v0.2 default limitation ("no simulation ... claimed") is replaced by the MES verification statement and its 2-D /
  2.5-D limitations;
* the result is validated against the vendored Core 0.1.0 contract before it is returned.

The magnetics entry keeps referencing only assumption / analytical / source evidence, so the v0.2 importer
(:func:`~aipe_magnetics.integrations.aipe_core.export.from_engineering_state`) still round-trips it.

Import (:func:`mes_from_core_state`): converter requirements (power, voltages, switching frequency) and the magnetics
entries of a contract-valid Core state seed a new MES at the ``specification`` stage.
"""

from __future__ import annotations

import re
from typing import Any

from aipe_magnetics import __version__
from aipe_magnetics.common.errors import InvalidInputError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.state.model import MagneticEngineeringState


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or "x"


def mes_to_core_state(state: MagneticEngineeringState, *, created_at: str, project_id: str | None = None,
                      converter_topology: str = "unspecified", role: str | None = None) -> dict[str, Any]:
    from aipe_magnetics.integrations.aipe_core.export import (
        DEFAULT_LIMITATION,
        CoreValidationCheck,
        MagneticComponentSummary,
        to_engineering_state,
    )
    from aipe_magnetics.integrations.aipe_core.state import validate_engineering_state

    comp_id = f"magnetic.{_slug(state.id)}"
    tool = {"id": "aipe.magnetics", "name": "AIPE Magnetics", "version": __version__}
    evidence: list[dict[str, Any]] = []
    for a in state.assumptions[:40]:
        evidence.append({"id": f"evidence.{_slug(comp_id)}-assumption-{_slug(a.id)}", "kind": "assumption", "description": a.statement,
                         "source": {"type": "user_requirement" if a.basis == "user_input" else "document",
                                    "reference": f"aipe-magnetics MES {state.id} assumption {a.id} (status {a.status})"},
                         "provenance": {"created_at": created_at, "activity": "Design assumption recorded in the MES", "inputs": [], "tool": tool}})
    for s in state.provenance.sources[:20]:
        if s.kind in ("datasheet", "publication", "repository", "dataset"):
            evidence.append({"id": f"evidence.{_slug(comp_id)}-source-{_slug(s.id)}", "kind": "source", "description": s.reference,
                             "source": {"type": "datasheet" if s.kind == "datasheet" else ("publication" if s.kind == "publication" else "repository"),
                                        "reference": s.reference, **({"locator": s.locator} if s.locator else {})},
                             "provenance": {"created_at": created_at, "activity": "Source cited by the design", "inputs": [], "tool": tool}})
    seen: set[str] = set()
    evidence = [e for e in evidence if not (e["id"] in seen or seen.add(e["id"]))]
    analytical_quantities = {}
    fea_quantities = {}
    for rs in state.results.electromagnetic:
        for k, v in rs.quantities.items():
            target = fea_quantities if (v.method or "").startswith("fea") else analytical_quantities
            target.setdefault(k, {"value": v.value, "unit": v.unit, "method": v.method, "result": rs.id})
    analytical_id = f"evidence.{_slug(comp_id)}-analytical"
    evidence.append({"id": analytical_id, "kind": "analytical",
                     "description": "Analytical design quantities of the selected candidate (aipe-magnetics MES): "
                                    + "; ".join(f"{k} = {v['value']:.6g} {v['unit']}" for k, v in list(analytical_quantities.items())[:12]),
                     "source": {"type": "repository", "reference": "https://github.com/FulongLi/AIPE-Magnetics-Database", "locator": state.id},
                     "provenance": {"created_at": created_at, "activity": f"aipe-magnetics {__version__} analytical evaluation",
                                    "inputs": [e["id"] for e in evidence] or [], "tool": tool}})
    if not evidence[-1]["provenance"]["inputs"]:
        raise InvalidInputError("the MES needs at least one assumption or source to export analytical evidence to Core")

    def first(names: tuple[str, ...], pool: dict[str, dict]) -> Quantity | None:
        for n in names:
            if n in pool:
                return Quantity(value=pool[n]["value"], unit=pool[n]["unit"])
        return None

    turns = {w.name: w.turns for w in state.windings}
    turns_ratio = None
    if state.component.type == "transformer" and "primary" in turns and "secondary" in turns:
        turns_ratio = Quantity(value=turns["primary"] / turns["secondary"], unit="1")
    inductance = first(("inductance", "magnetising_inductance"), analytical_quantities)
    leakage = first(("leakage_inductance_1d", "leakage_inductance"), analytical_quantities)
    material = next((m.record_id for m in state.materials if m.role == "core" and m.selected), None)
    summary = MagneticComponentSummary(
        id=comp_id, role=role or (state.component.application or state.component.type), kind=state.component.type,
        design_status="selected" if state.lifecycle_stage in ("optimisation", "verification", "exported") else "candidate",
        inductance=inductance, leakage_inductance=leakage, turns_ratio=turns_ratio,
        evidence=evidence, limitations=[lvl.statement for lvl in state.verification.levels if lvl.status != "passed"][:6] or
        ["preliminary design; see the MES verification record"],
        detail={"mes_id": state.id, "lifecycle_stage": state.lifecycle_stage, "fea_quantities": fea_quantities,
                "analytical_quantities": analytical_quantities, "manufacturing_readiness": state.verification.manufacturing_readiness},
        core_material_ref=material)
    req: dict[str, Any] = {}
    elec = state.requirements.electrical
    mapping = {"nominal_power": ("power",), "input_voltage": ("v1", "primary_voltage", "input_voltage"),
               "output_voltage": ("v2", "secondary_voltage", "output_voltage"), "switching_frequency": ("frequency",)}
    for core_name, names in mapping.items():
        for n in names:
            if n in elec:
                req[core_name] = Quantity(value=elec[n].value, unit=elec[n].unit)
                break
    requirements = None
    if req:
        brief_id = f"evidence.{_slug(comp_id)}-requirements"
        requirements = {**req, "evidence": [{"id": brief_id, "kind": "assumption",
                                              "description": f"Requirements of MES {state.id} (specified fields: "
                                                             f"{', '.join(state.requirements.specified_fields) or 'none'})",
                                              "source": {"type": "user_requirement", "reference": f"aipe-magnetics MES {state.id}"},
                                              "provenance": {"created_at": created_at, "activity": "Record the design requirements", "inputs": []}}]}
    checks = [CoreValidationCheck(id=f"check.{_slug(comp_id)}-analytical", status="pass" if analytical_quantities else "not_run",
                                  description="Analytical design relations evaluated for the selected candidate",
                                  evidence_refs=[analytical_id] if analytical_quantities else [])]
    core_state = to_engineering_state(summary, created_at=created_at, project_id=project_id or f"project.{_slug(state.id)}",
                                      project_name=state.title, converter_topology=converter_topology, requirements=requirements,
                                      checks=checks)
    # simulation evidence and runs
    artifacts = {a.id: a for a in state.artifacts}
    runs, sim_ids = [], []
    for solver in state.solvers:
        if solver.status != "completed" or not (solver.fidelity or "").startswith("fea"):
            continue
        result_artifacts = [aid for rs in state.results.electromagnetic + state.results.thermal if rs.solver_ref == solver.id for aid in rs.artifacts]
        outs = [artifacts[a] for a in result_artifacts if a in artifacts]
        if not outs or solver.configuration_artifact not in artifacts:
            continue
        sim_id = f"evidence.{_slug(comp_id)}-{_slug(solver.id)}"
        core_art = [{"uri": a.path, "media_type": a.media_type, **({"sha256": a.sha256} if a.sha256 else {}),
                     **({"description": a.description} if a.description else {})} for a in outs]
        core_state["evidence"].append({"id": sim_id, "kind": "simulation",
                                       "description": f"{solver.domain} {solver.fidelity} by {solver.backend} {solver.version}: {solver.formulation}",
                                       "source": {"type": "repository", "reference": "https://github.com/FulongLi/AIPE-Magnetics-Database",
                                                  "locator": solver.id},
                                       "provenance": {"created_at": created_at, "activity": f"{solver.backend} run recorded in MES {state.id}",
                                                      "inputs": [analytical_id], "tool": {"id": "aipe.magnetics", "name": "AIPE Magnetics",
                                                                                          "version": __version__}},
                                       "artifacts": core_art, "review_status": "unreviewed"})
        cfg = artifacts[solver.configuration_artifact]
        runs.append({"id": f"run.{_slug(comp_id)}-{_slug(solver.id)}", "status": "completed",
                     "tool": {"id": "aipe.magnetics", "name": solver.backend, "version": solver.version},
                     "model": {"uri": cfg.path, "media_type": cfg.media_type, **({"sha256": cfg.sha256} if cfg.sha256 else {})},
                     "outputs": core_art, "evidence_refs": [sim_id]})
        sim_ids.append(sim_id)
    if runs:
        core_state["simulation"]["runs"].extend(runs)
        core_state["simulation"]["status"] = "completed"
        core_state["simulation"]["evidence_refs"] = sorted(set(core_state["simulation"].get("evidence_refs", [])) | set(sim_ids))
        passed = all(c.get("status") != "fail" for c in state.verification.checks if c.get("kind") == "detailed_constraints")
        detailed = [c for c in state.verification.checks if c.get("kind") == "detailed_constraints"]
        status = "pass" if detailed and passed else ("fail" if detailed else "not_run")
        core_state["validation"]["checks"].append({"id": f"check.{_slug(comp_id)}-simulation", "kind": "simulation", "status": status,
                                                    "description": "Detailed (2-D FEA + 2.5-D thermal FE) constraint check of the selected design",
                                                    "evidence_refs": sim_ids})
        if status == "fail":
            core_state["validation"]["status"] = "failed"
        elif status == "pass" and core_state["validation"]["status"] != "failed":
            core_state["validation"]["status"] = "simulated"
    lim = core_state["validation"]["limitations"]
    if DEFAULT_LIMITATION in lim and runs:
        lim.remove(DEFAULT_LIMITATION)
    statement = (f"aipe.magnetics MES {state.id}: {state.verification.statement} Manufacturing readiness: "
                 f"{state.verification.manufacturing_readiness}.")
    if statement not in lim:
        lim.append(statement)
    for solver in state.solvers:
        for text in solver.limitations[:4]:
            if text not in lim:
                lim.append(text)
    errors = validate_engineering_state(core_state)
    if errors:
        raise InvalidInputError("MES export violates the Core Engineering State contract", details={"errors": errors})
    return core_state


def mes_from_core_state(core_state: dict[str, Any], *, state_id: str = "mes.from-core", title: str | None = None) -> MagneticEngineeringState:
    from aipe_magnetics.integrations.aipe_core.export import extract_requirements
    req = extract_requirements(core_state)
    names = {"nominal_power": "power", "input_voltage": "v1", "output_voltage": "v2", "switching_frequency": "frequency"}
    electrical = {names[k]: {"value": q.value, "unit": q.unit, "method": "specified"} for k, q in req.items()}
    magnetics = core_state.get("magnetics", [])
    kind = magnetics[0]["kind"] if magnetics else "transformer"
    return MagneticEngineeringState.model_validate({
        "id": state_id, "title": title or f"MES seeded from Core state {core_state.get('id')}", "lifecycle_stage": "specification",
        "component": {"type": kind if kind in ("inductor", "transformer", "coupled_inductor") else "transformer",
                      "topology": core_state.get("converter", {}).get("topology"),
                      "description": f"seeded from AIPE-Core state {core_state.get('id')}"},
        "requirements": {"electrical": electrical, "specified_fields": sorted(electrical)},
        "provenance": {"tool": {"version": __version__}, "inputs": [f"aipe-core-state:{core_state.get('id')}"]},
        "history": [{"stage": "specification", "description": "requirements imported from an AIPE-Core Engineering State 0.1.0"}],
    })


__all__ = ["mes_from_core_state", "mes_to_core_state"]
