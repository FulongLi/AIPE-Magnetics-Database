"""Bridge between engineering results and the AIPE-Core round-trip adapter.

``component_summary_from_design`` turns a ``design_transformer`` / ``design_inductor`` result
envelope into a :class:`MagneticComponentSummary` whose evidence records the assumptions and the
analytical derivation (with the assumption ids as provenance inputs), so that
:func:`to_engineering_state` can emit a contract-valid Core Engineering State.
"""

from __future__ import annotations

import re
from typing import Any

from aipe_magnetics import TOOL_ID, __version__
from aipe_magnetics.common.errors import InvalidInputError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import EngineeringResult
from aipe_magnetics.integrations.aipe_core.export import CoreEvidence, MagneticComponentSummary


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "item"


def component_summary_from_design(result: EngineeringResult, *, role: str, created_at: str,
                                  component_id: str | None = None, design_status: str = "candidate") -> MagneticComponentSummary:
    """Summarise a successful design result as a Core magnetics component with assumption + analytical evidence."""
    if result.status != "ok":
        raise InvalidInputError(f"cannot export a {result.status} design result")
    # design_transformer and the waveform-informed design_transformer_from_waveforms both describe transformers
    is_transformer = "design_transformer" in result.capability or result.data.get("component_kind") == "transformer"
    kind = "transformer" if is_transformer else "inductor"
    selected = result.data.get("selected")
    component_id = component_id or f"magnetic.{_slug(role)}"
    evidence: list[CoreEvidence] = []
    tool = {"id": TOOL_ID, "version": __version__, "name": "AIPE Magnetics"}
    assumption_ids: list[str] = []
    for a in result.assumptions:
        eid = f"evidence.{_slug(component_id)}-assumption-{_slug(a.id)}"
        assumption_ids.append(eid)
        evidence.append(CoreEvidence(id=eid, kind="assumption", description=a.statement,
                                     source={"type": "user_requirement" if a.basis == "user_input" else "document",
                                             "reference": f"aipe-magnetics {result.capability} assumption {a.id} (basis: {a.basis})"},
                                     provenance={"created_at": created_at, "activity": f"Design assumption recorded by {result.capability}",
                                                 "inputs": [], "tool": tool}))
    for s in result.sources:
        if s.kind in ("datasheet", "publication", "repository", "dataset"):
            eid = f"evidence.{_slug(component_id)}-source-{_slug(s.id)}"
            if any(e.id == eid for e in evidence):
                continue
            evidence.append(CoreEvidence(id=eid, kind="source", description=s.reference,
                                         source={"type": "datasheet" if s.kind == "datasheet" else ("publication" if s.kind == "publication" else "repository"),
                                                 "reference": s.reference, **({"locator": s.locator} if s.locator else {})},
                                         provenance={"created_at": created_at, "activity": "Source cited by the design calculation", "inputs": [], "tool": tool}))
    source_ids = [e.id for e in evidence if e.kind == "source"]
    analytical_id = f"evidence.{_slug(component_id)}-analytical"
    values_text = "; ".join(f"{k} = {v.quantity.value:.6g} {v.quantity.unit} ({v.status})" for k, v in result.values.items())
    evidence.append(CoreEvidence(id=analytical_id, kind="analytical",
                                 description=(f"Analytical design by {result.capability} (model {result.model.id}@{result.model.version} "
                                              if result.model else f"Analytical design by {result.capability} (")
                                 + f"selected core {selected}): {values_text}",
                                 source={"type": "repository", "reference": "https://github.com/FulongLi/AIPE-Magnetics-Database",
                                         "locator": result.capability},
                                 provenance={"created_at": created_at, "activity": f"{result.capability} executed by aipe-magnetics {__version__}",
                                             "inputs": assumption_ids + source_ids, "tool": tool}))
    turns_ratio = None
    if is_transformer:
        turns = {k[len("turns_"):]: v.quantity.value for k, v in result.values.items() if k.startswith("turns_")}
        primary = turns.get("primary")
        secondaries = [v for k, v in turns.items() if k != "primary"]
        if primary and secondaries:
            turns_ratio = Quantity(value=primary / secondaries[0], unit="1")
    detail: dict[str, Any] = {
        "capability": result.capability, "selected_core": selected,
        "values": {k: {"value": v.quantity.value, "unit": v.quantity.unit, "status": v.status} for k, v in result.values.items()},
        "verification": result.verification.model_dump(mode="json"),
        "warnings": [w.model_dump(mode="json") for w in result.warnings],
    }
    limitations = [lim.statement for lim in result.limitations] or ["analytical design estimate; no hardware validation"]
    return MagneticComponentSummary(id=component_id, role=role, kind=kind, design_status=design_status, turns_ratio=turns_ratio,
                                    core_material_ref=result.data.get("material"), database_ref=selected, evidence=evidence,
                                    limitations=limitations, detail=detail)
