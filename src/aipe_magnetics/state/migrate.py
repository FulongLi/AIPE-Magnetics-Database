"""Version migration into the Magnetic Engineering State 0.3.0.

Supported sources:

* ``aipe.magnetics.engineering-state/0.3.0`` - returned unchanged (deep copy);
* ``aipe.magnetics.result/0.2`` result envelopes of ``design_transformer`` / ``design_inductor`` (the v0.2 design
  record): converted into an MES at the ``candidate_generation`` stage with the selected core, turns, analytical
  losses (method ``analytical``), the envelope's assumptions, sources and verification statement. Nothing is
  invented: values the envelope does not contain stay absent.

Every migration records itself in ``history`` and in ``extensions['aipe.magnetics.migration']``.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from aipe_magnetics import __version__
from aipe_magnetics.common.errors import InvalidInputError, UnsupportedOperationError
from aipe_magnetics.state.model import STATE_SCHEMA


def _qv(value: Any, unit: str, method: str, description: str | None = None) -> dict[str, Any]:
    out = {"value": float(value), "unit": unit, "method": method}
    if description:
        out["description"] = description
    return out


def _from_v02_design(envelope: dict[str, Any]) -> dict[str, Any]:
    cap = envelope.get("capability", "")
    if envelope.get("status") != "ok":
        raise InvalidInputError(f"cannot migrate a {envelope.get('status')!r} v0.2 result (only successful designs carry a design)")
    is_tr = cap.endswith("design_transformer")
    if not (is_tr or cap.endswith("design_inductor")):
        raise UnsupportedOperationError(f"v0.2 envelopes of capability {cap!r} are not design records")
    inputs = envelope.get("inputs", {})
    values = envelope.get("values", {})
    data = envelope.get("data", {})
    tool_version = envelope.get("tool", {}).get("version", "0.2.x")
    selected = data.get("selected")
    method_of = {"calculated": "analytical", "estimated": "analytical", "assumed": "assumed", "specified": "specified",
                 "looked_up": "looked_up", "measured": "measured"}

    def q(key: str) -> dict[str, Any] | None:
        v = values.get(key)
        if not v:
            return None
        qty = v["quantity"]
        return _qv(qty["value"], qty["unit"], method_of.get(v.get("status", "calculated"), "analytical"), v.get("label"))

    electrical: dict[str, Any] = {}
    for key in ("power", "frequency", "inductance", "current_peak", "current_rms"):
        if isinstance(inputs.get(key), dict) and "value" in inputs[key]:
            electrical[key] = _qv(inputs[key]["value"], inputs[key]["unit"], "specified")
    if is_tr:
        for role in ("primary",):
            w = inputs.get(role, {})
            if isinstance(w.get("voltage"), dict):
                electrical["primary_voltage"] = _qv(w["voltage"]["value"], w["voltage"]["unit"], "specified")
        for i, w in enumerate(inputs.get("secondaries", [])):
            if isinstance(w.get("voltage"), dict):
                electrical[f"secondary_voltage_{i}"] = _qv(w["voltage"]["value"], w["voltage"]["unit"], "specified")
    windings = []
    for key, v in values.items():
        if key.startswith("turns_") or key == "turns":
            name = key[len("turns_"):] if key.startswith("turns_") else "L"
            windings.append({"name": name, "turns": int(round(v["quantity"]["value"])), "conductor": inputs.get("winding", {}) or
                             next((w.get("winding", {}) for w in [inputs.get("primary", {}), *inputs.get("secondaries", [])]
                                   if w.get("name") == name), {})})
    losses = {"core": q("core_loss"), "total": q("total_loss")}
    copper = q("copper_loss")
    loss = {"operating_point_ref": None, "notes": [f"migrated from {cap} (aipe-magnetics {tool_version}, analytical sizing)"]}
    if losses["core"]:
        loss["core"] = losses["core"]
    if losses["total"]:
        loss["total"] = losses["total"]
    if copper:
        loss["windings"] = {"all": copper}
    quantities = {k: q(k) for k in values if q(k) is not None}
    state = {
        "state_schema": STATE_SCHEMA, "state_version": "0.3.0",
        "id": f"migrated.{cap.split('.')[-1]}.{selected or 'none'}",
        "title": f"Migrated v0.2 {('transformer' if is_tr else 'inductor')} design ({selected})",
        "lifecycle_stage": "candidate_generation",
        "component": {"type": "transformer" if is_tr else "inductor", "windings": [w["name"] for w in windings],
                      "description": "migrated from an aipe.magnetics.result/0.2 envelope"},
        "requirements": {"electrical": electrical, "specified_fields": sorted(electrical)},
        "assumptions": [{"id": a["id"], "statement": a["statement"], "basis": a.get("basis", "default"), "status": "assumed",
                         "affects": a.get("affects", [])} for a in envelope.get("assumptions", [])],
        "materials": [{"role": "core", "record_id": data.get("material", inputs.get("material", "unknown")), "origin": "catalogue"}],
        "cores": [{"record_id": c["core"], "name": c.get("core_name", c["core"]), "selected": c["core"] == selected,
                   "notes": c.get("reasons", [])} for c in data.get("candidates", [])],
        "windings": windings,
        "solvers": [{"id": "solver.v02-analytical", "domain": "electromagnetic", "backend": cap, "version": tool_version, "fidelity": "analytical",
                     "formulation": "v0.2 analytical sizing (Faraday, reluctance, Dowell 1-D, Steinmetz/iGSE on Ve)"}],
        "results": {"electromagnetic": [{"id": "results.v02-analytical", "solver_ref": "solver.v02-analytical", "quantities": quantities,
                                         "warnings": [w["message"] for w in envelope.get("warnings", [])]}],
                    "losses": [loss]},
        "verification": {"levels": [{"level": "numerical_physical", "status": "partial",
                                     "statement": envelope.get("verification", {}).get("statement", "v0.2 verification statement unavailable")},
                                    {"level": "experimental", "status": "not_performed", "statement": "no measurements"}],
                         "manufacturing_readiness": "preliminary_design",
                         "statement": "Migrated analytical design; no FEA, thermal or insulation analysis in the source record."},
        "provenance": {"tool": {"version": __version__}, "inputs": [cap], "sources": envelope.get("sources", []),
                       "software": {"aipe-magnetics (source record)": tool_version}},
        "history": [{"stage": "candidate_generation", "description": f"v0.2 {cap} result migrated to MES 0.3.0"}],
        "extensions": {"aipe.magnetics.migration": {"from": envelope.get("result_schema"), "capability": cap, "tool_version": tool_version}},
    }
    return state


MIGRATIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "aipe.magnetics.result/0.2": _from_v02_design,
}


def migrate_state(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise InvalidInputError("state must be a JSON object")
    if data.get("state_schema") == STATE_SCHEMA:
        return copy.deepcopy(data)
    source = data.get("state_schema") or data.get("result_schema")
    if source in MIGRATIONS:
        return MIGRATIONS[source](data)
    raise UnsupportedOperationError(f"no migration from {source!r} to {STATE_SCHEMA}; supported: {sorted(MIGRATIONS)} and {STATE_SCHEMA}")


__all__ = ["MIGRATIONS", "migrate_state"]
