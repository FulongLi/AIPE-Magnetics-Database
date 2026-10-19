"""Plain-text presentation of result envelopes (shared by the CLI and the executable demonstration).

Kept outside ``cli/`` so that the engineering layer never imports an interaction layer.
"""

from __future__ import annotations

from aipe_magnetics.common.results import EngineeringResult


def summary(result: EngineeringResult) -> str:
    """Human-readable one-screen summary of a result envelope (values, errors, warnings, tables, verification)."""
    lines = [f"[{result.status}] {result.capability}" + (f"  model={result.model.id}@{result.model.version}" if result.model else "")]
    for key, v in result.values.items():
        lines.append(f"  {v.label} ({key}): {v.quantity.value:.6g} {v.quantity.unit}  [{v.status}]")
    for e in result.errors:
        lines.append(f"  ERROR {e.code}: {e.message}")
    for w in result.warnings:
        lines.append(f"  {w.severity}: {w.code}: {w.message}")
    if result.data.get("materials") is not None:
        for item in result.data["materials"]:
            lines.append(f"  - {item['id']}  grade={item.get('grade')}  class={item.get('material_class')}  sets={item.get('steinmetz_sets')}"
                         f"  synthetic={item.get('synthetic')}")
    if result.data.get("comparison") is not None:
        for row in result.data["comparison"]:
            pv = row.get("loss_density_W_per_m3")
            margin = row.get("saturation_margin")
            pv_text = "n/a" if pv is None else "%.1f kW/m^3" % (pv / 1e3)
            margin_text = "n/a" if margin is None else "%.1f%%" % (margin * 100)
            lines.append("  - %s: Pv=%s  margin=%s  %s" % (row["id"], pv_text, margin_text, "; ".join(row.get("notes", []))))
    if result.data.get("checks") is not None:
        verdict = result.data.get("passed")
        verdict_text = "PASSED" if verdict is True else ("FAILED" if verdict is False else "INCONCLUSIVE")
        lines.append("  checks: " + verdict_text)
        for check in result.data["checks"]:
            lines.append("  - %s: %s - %s" % (check.get("id"), check.get("status"), check.get("message")))
    if result.data.get("candidates") is not None:
        for c in result.data["candidates"]:
            total = c.get("total_loss_W")
            bpk = c.get("flux_density_peak_T")
            bpk_text = "n/a" if bpk is None else "%.3f T" % bpk
            total_text = "n/a" if total is None else "%.1f W" % total
            lines.append("  - %s: feasible=%s Bpk=%s total=%s %s" % (c.get("core_name", c.get("core")), c.get("feasible"), bpk_text,
                                                                     total_text, "; ".join(c.get("reasons", []))))
    lines.append(f"  verification: {result.verification.level} {result.verification.state} - {result.verification.statement}")
    for lim in result.limitations:
        lines.append(f"  limitation: {lim.statement}")
    return "\n".join(lines)
