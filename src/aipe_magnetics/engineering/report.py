"""Engineering report generation (Markdown and JSON) from result envelopes.

The report separates **calculated** values, **model-based estimates**, **looked-up data**, **specified**
inputs and **assumed** parameters, lists sources, verification status per capability, warnings,
limitations and errors, and renders the structured payloads of the capabilities (candidate tables,
material comparisons, validation check verdicts) so that nothing in ``data`` is silently dropped.
Quantified uncertainties are printed next to the value; a value without an uncertainty statement has
unknown uncertainty. Writing to disk requires explicit authorisation (``allow_write=True``) and an
explicit output path; existing files are never overwritten unless ``overwrite=True``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from aipe_magnetics import __version__
from aipe_magnetics.api.types import ReportRequest
from aipe_magnetics.common.errors import AuthorizationError, DestinationExistsError, InvalidInputError
from aipe_magnetics.common.jsonio import dumps
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import EngineeringResult

CAP_REPORT = "aipe.magnetics.generate_report"

_PREFERRED_UNITS = {"T": "mT", "W/m^3": "kW/m^3", "m": "mm", "m^2": "mm^2", "m^3": "cm^3", "H": "uH", "A/m": "A/m",
                    "Hz": "kHz", "K": "degC", "Ohm": "mOhm"}


def _cell(text: Any) -> str:
    """Escape a Markdown table cell: pipes would add columns, newlines would break the row."""
    return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _uncertainty(q: Quantity) -> str:
    u = q.uncertainty
    if u is None:
        return ""
    unit = "" if q.unit == "1" else f" {q.unit}"
    if u.kind == "standard" and u.value is not None:
        return f" +/- {u.value:.3g}{unit} (standard uncertainty)"
    if u.kind == "expanded" and u.value is not None:
        k = f", k={u.coverage_factor:g}" if u.coverage_factor is not None else ""
        return f" +/- {u.value:.3g}{unit} (expanded uncertainty{k})"
    if u.kind == "bounds" and u.lower is not None and u.upper is not None:
        return f" (bounds {u.lower:.6g}..{u.upper:.6g}{unit})"
    return " (uncertainty unknown)"


def _fmt(q: Quantity) -> str:
    if q.unit == "1":
        return f"{q.value:.6g}" + _uncertainty(q)
    unit = _PREFERRED_UNITS.get(q.unit)
    text = f"{q.value:.6g} {q.unit}"
    if unit and unit != q.unit:
        try:
            text = f"{q.value:.6g} {q.unit} ({q.magnitude(unit):.5g} {unit})"
        except Exception:  # pragma: no cover - conversion always valid for the table above
            pass
    return text + _uncertainty(q)


def _num(x: Any, fmt: str = ".3g") -> str:
    return "-" if x is None else format(x, fmt)


def _verdict(passed: Any) -> str:
    return "PASSED" if passed is True else ("FAILED" if passed is False else "INCONCLUSIVE")


def _candidate_table(candidates: list[dict[str, Any]]) -> list[str]:
    lines = ["### Candidate comparison", "",
             "Feasible = saturation, window fill and loss model satisfied *and* completely evaluated; a candidate whose copper loss or "
             "fill could not be evaluated is listed as not feasible with the reason.", "",
             "| Core | Feasible | Bpk [T] | Core loss [W] | Copper loss [W] | Total [W] | Fill | Notes |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in candidates:
        notes = "; ".join(list(c.get("reasons", [])) + list(c.get("warnings", [])))
        lines.append(f"| {_cell(c.get('core_name') or c.get('core'))} | {'yes' if c.get('feasible') else 'no'} | "
                     f"{_num(c.get('flux_density_peak_T'))} | {_num(c.get('core_loss_W'))} | {_num(c.get('copper_loss_W'))} | "
                     f"{_num(c.get('total_loss_W'))} | {_num(c.get('window_fill'), '.2f')} | {_cell(notes)} |")
    return lines + [""]


def _comparison_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["### Material comparison", "",
             "| Material | Loss density [kW/m^3] | Loss model | Parameter set | Bsat [T] | Saturation margin | Verification | Notes |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        pv = r.get("loss_density_W_per_m3")
        verification = r.get("loss_verification") or {}
        ver_text = f"{verification.get('level', '-')} {verification.get('state', '')}".strip() if verification else "-"
        lines.append(f"| {_cell(r.get('id'))} | {'-' if pv is None else format(pv / 1e3, '.4g')} | {_cell(r.get('loss_model') or '-')} | "
                     f"{_cell(r.get('parameter_set') or '-')} | {_num(r.get('saturation_flux_density_T'))} | "
                     f"{_num(r.get('saturation_margin'), '.3f')} | {_cell(ver_text)} | {_cell('; '.join(r.get('notes', [])) or '-')} |")
    return lines + [""]


def _checks_table(data: dict[str, Any]) -> list[str]:
    lines = ["### Validation checks", "", f"Verdict: **{_verdict(data.get('passed'))}**"]
    if data.get("failed_checks"):
        lines.append(f"Failed checks: {', '.join(data['failed_checks'])}")
    if data.get("inconclusive_checks"):
        lines.append(f"Not evaluated: {', '.join(data['inconclusive_checks'])}")
    lines += ["", "| Check | Status | Message |", "| --- | --- | --- |"]
    for check in data["checks"]:
        status = str(check.get("status", "")).upper()
        lines.append(f"| {_cell(check.get('id'))} | {status} | {_cell(check.get('message'))} |")
    return lines + [""]


def render_markdown(results: Iterable[EngineeringResult], request: ReportRequest | None = None) -> str:
    request = request or ReportRequest()
    results = list(results)
    lines = [f"# {request.title}", "", f"Generated by AIPE Magnetics {__version__} (aipe-magnetics). "
             "All numbers below come from the engineering engine; no value was edited by hand.", ""]
    lines += ["## Reading guide", "",
              "* **calculated** - follows from inputs through verified closed-form relations (e.g. Faraday's law).",
              "* **estimated** - produced by a model (Steinmetz/iGSE/Dowell ...) whose identity, version and validity are stated.",
              "* **looked_up** - transcribed catalogue/datasheet data with source references.",
              "* **specified** - an input supplied by the caller, echoed for traceability.",
              "* **assumed** - parameters chosen by policy or default; listed under Assumptions with their effect.",
              "* Uncertainties are printed when a source or fit quantifies them; a value without one has unknown uncertainty.",
              "* Verification levels: L1 software/schema correctness, L2 numerical and physical consistency, "
              "L3 engineering application validation. No hardware qualification is implied.", ""]
    for index, r in enumerate(results, 1):
        lines += [f"## {index}. {r.capability}", "", f"Status: **{r.status}**" +
                  (f" - model `{r.model.id}` v{r.model.version} ({r.model.family})" if r.model else ""), ""]
        if r.errors:
            lines += ["### Errors", ""]
            lines += [f"- `{e.code}`: {e.message}" for e in r.errors]
            lines.append("")
        if r.values:
            lines += ["### Results", "", "| Quantity | Value | Status |", "| --- | --- | --- |"]
            for key, v in r.values.items():
                note = f" ({v.note})" if v.note else ""
                lines.append(f"| {_cell(v.label)} (`{key}`){_cell(note)} | {_cell(_fmt(v.quantity))} | {v.status} |")
            lines.append("")
        if request.include_inputs and r.inputs:
            lines += ["### Inputs", "", "```json", dumps(r.inputs).rstrip(), "```", ""]
        if r.assumptions:
            lines += ["### Assumptions", ""]
            lines += [f"- `{a.id}` ({a.basis}): {a.statement}" + (f" _affects: {', '.join(a.affects)}_" if a.affects else "")
                      for a in r.assumptions]
            lines.append("")
        lines += ["### Verification", "", f"- level {r.verification.level}, state **{r.verification.state}**: {r.verification.statement}"]
        if r.verification.evidence_refs:
            lines.append(f"- evidence: {', '.join(r.verification.evidence_refs)}")
        lines.append("")
        if r.warnings:
            lines += ["### Warnings", ""] + [f"- [{w.severity}] `{w.code}`: {w.message}" for w in r.warnings] + [""]
        if r.limitations:
            lines += ["### Limitations", ""] + [f"- `{lim.code}`: {lim.statement}" for lim in r.limitations] + [""]
        if r.sources:
            lines += ["### Sources", ""]
            for s in r.sources:
                extra = "; ".join(x for x in (s.locator, s.version, s.url) if x)
                lines.append(f"- `{s.id}` ({s.kind}, licence {s.license}): {s.reference}" + (f" - {extra}" if extra else ""))
            lines.append("")
        data = r.data if isinstance(r.data, dict) else {}
        if data.get("candidates"):
            lines += _candidate_table(data["candidates"])
        if data.get("comparison"):
            lines += _comparison_table(data["comparison"])
        if data.get("checks"):
            lines += _checks_table(data)
    return "\n".join(lines).rstrip() + "\n"


def render_json(results: Iterable[EngineeringResult], request: ReportRequest | None = None) -> str:
    request = request or ReportRequest()
    payload = {"report_schema": "aipe.magnetics.report/0.2", "title": request.title, "tool": {"id": "aipe.magnetics", "version": __version__},
               "results": [r.to_json_dict() for r in results]}
    return dumps(payload)


def generate_report(results: list[EngineeringResult], request: ReportRequest | None = None, *,
                    allowed_directory: str | Path | None = None) -> EngineeringResult:
    """Render a report; optionally write it (explicit authorisation + path inside ``allowed_directory`` when given)."""
    request = request or ReportRequest()
    if not results:
        return EngineeringResult.rejected(CAP_REPORT, InvalidInputError("generate_report needs at least one result"))
    text = render_markdown(results, request) if request.format == "markdown" else render_json(results, request)
    out = EngineeringResult(capability=CAP_REPORT, inputs=request.model_dump(mode="json", exclude_none=True))
    out.data["format"] = request.format
    out.data["content"] = text
    out.data["result_count"] = len(results)
    out.verification.level = "L1"
    out.verification.state = "verified"
    out.verification.statement = "Report rendering is a deterministic transformation of result envelopes (L1)."
    if request.output_path:
        try:
            path = _authorised_path(request, allowed_directory)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w" if request.overwrite else "x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
            out.data["written_to"] = str(path)
        except FileExistsError:
            return EngineeringResult.rejected(CAP_REPORT, DestinationExistsError(
                f"{request.output_path} exists; pass overwrite=true to replace it"), inputs=out.inputs)
        except AuthorizationError as error:
            return EngineeringResult.rejected(CAP_REPORT, error, inputs=out.inputs)
    return out


def _authorised_path(request: ReportRequest, allowed_directory: str | Path | None) -> Path:
    if not request.allow_write:
        raise AuthorizationError("writing a report requires allow_write=true and an explicit output_path")
    path = Path(request.output_path).expanduser()  # type: ignore[arg-type]
    if allowed_directory is not None:
        root = Path(allowed_directory).resolve()
        resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            raise AuthorizationError(f"output path {path} is outside the allowed directory {root}") from None
        return resolved
    return path.resolve()
