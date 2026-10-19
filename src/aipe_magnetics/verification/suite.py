"""Verification suite: registration and execution of checks that produce evidence records.

Usage in a check module (``aipe_magnetics/verification/checks/check_<topic>.py``)::

    from aipe_magnetics.verification.suite import check, CheckResult

    @check(
        id="igse.sinusoid-reduces-to-steinmetz",
        capability="aipe.magnetics.model.igse",
        level="L2", kind="limiting_case",
        description="For sinusoidal excitation iGSE equals the classical Steinmetz equation.",
        method="Evaluate both models on a grid of (f, Bpk) and compare relative error.",
        range=OperatingRange(frequency=ValidityRange(min=1e3, max=1e6, unit="Hz")),
    )
    def igse_sinusoid() -> CheckResult:
        ...
        return CheckResult(passed=max_err < 1e-9, metrics={"max_rel_error": max_err})

Every check must be deterministic, offline, and must not modify repository files. The suite
runner imports all ``aipe_magnetics.verification.checks.check_*`` modules, executes the checks,
and builds a :class:`VerificationRegistry` / :class:`VerificationReport`. ``pytest`` runs the same
functions, so a failing check fails CI, and the committed evidence file is compared against a fresh
run so that stale evidence cannot be claimed.
"""

from __future__ import annotations

import importlib
import pkgutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from pydantic import BaseModel, ConfigDict, Field

from aipe_magnetics import __version__
from aipe_magnetics.common.results import SourceReference
from aipe_magnetics.verification.evidence import (
    EvidenceKind,
    EvidenceLevel,
    EvidenceRecord,
    OperatingRange,
    VerificationRegistry,
    VerificationReport,
)


class CheckResult(BaseModel):
    """What a check function returns."""

    model_config = ConfigDict(extra="forbid")
    passed: bool
    metrics: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    reference: SourceReference | None = None


@dataclass(frozen=True)
class CheckSpec:
    id: str
    capability: str
    level: EvidenceLevel
    kind: EvidenceKind
    description: str
    method: str
    func: Callable[[], CheckResult]
    range: OperatingRange | None = None
    reference: SourceReference | None = None
    tags: tuple[str, ...] = ()
    module: str = ""


@dataclass
class CheckOutcome:
    spec: CheckSpec
    result: CheckResult | None
    error: str | None = None
    duration_s: float = 0.0
    extra_records: list[EvidenceRecord] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.error is not None:
            return "fail"
        assert self.result is not None
        return "pass" if self.result.passed else "fail"

    def to_record(self) -> EvidenceRecord:
        metrics = dict(self.result.metrics) if self.result else {}
        notes = list(self.result.notes) if self.result else []
        if self.error:
            notes.append(f"check raised: {self.error.strip().splitlines()[-1]}")
        reference = (self.result.reference if self.result and self.result.reference else self.spec.reference)
        return EvidenceRecord(
            id=f"check:{self.spec.id}", capability=self.spec.capability, level=self.spec.level, kind=self.spec.kind,
            status=self.status, description=self.spec.description, method=self.spec.method, range=self.spec.range,
            reference=reference, check_id=self.spec.id, tool_version=__version__, metrics=metrics, notes=notes,
        )


_REGISTRY: dict[str, CheckSpec] = {}


def check(*, id: str, capability: str, level: EvidenceLevel, kind: EvidenceKind, description: str, method: str,
          range: OperatingRange | None = None, reference: SourceReference | None = None,
          tags: Iterable[str] = ()) -> Callable[[Callable[[], CheckResult]], Callable[[], CheckResult]]:
    """Register a verification check. ``id`` must be unique across the package."""

    def decorator(func: Callable[[], CheckResult]) -> Callable[[], CheckResult]:
        if id in _REGISTRY and _REGISTRY[id].func is not func:
            raise ValueError(f"duplicate check id {id!r} ({_REGISTRY[id].module} vs {func.__module__})")
        _REGISTRY[id] = CheckSpec(id=id, capability=capability, level=level, kind=kind, description=description,
                                  method=method, func=func, range=range, reference=reference, tags=tuple(tags),
                                  module=func.__module__)
        func.check_id = id  # type: ignore[attr-defined]
        return func

    return decorator


def discover_checks(package: str = "aipe_magnetics.verification.checks") -> dict[str, CheckSpec]:
    """Import every ``check_*`` module of ``package`` so decorators register their checks."""
    module = importlib.import_module(package)
    for info in pkgutil.iter_modules(module.__path__):
        if info.name.startswith("check_"):
            importlib.import_module(f"{package}.{info.name}")
    return dict(_REGISTRY)


def registered_checks() -> dict[str, CheckSpec]:
    return dict(_REGISTRY)


def run_check(spec: CheckSpec) -> CheckOutcome:
    start = time.perf_counter()
    try:
        result = spec.func()
        if not isinstance(result, CheckResult):
            raise TypeError(f"check {spec.id} returned {type(result).__name__}, expected CheckResult")
        return CheckOutcome(spec=spec, result=result, duration_s=time.perf_counter() - start)
    except Exception:  # noqa: BLE001 - a crashing check is a failing check, reported explicitly
        return CheckOutcome(spec=spec, result=None, error=traceback.format_exc(), duration_s=time.perf_counter() - start)


def run_suite(*, capability: str | None = None, tags: Iterable[str] | None = None,
              ids: Iterable[str] | None = None) -> list[CheckOutcome]:
    specs = discover_checks()
    selected = list(specs.values())
    if capability:
        selected = [s for s in selected if s.capability == capability]
    if tags:
        wanted = set(tags)
        selected = [s for s in selected if wanted & set(s.tags)]
    if ids:
        wanted_ids = set(ids)
        selected = [s for s in selected if s.id in wanted_ids]
    selected.sort(key=lambda s: s.id)
    return [run_check(spec) for spec in selected]


def build_registry(outcomes: Iterable[CheckOutcome]) -> VerificationRegistry:
    registry = VerificationRegistry()
    for outcome in outcomes:
        registry.add(outcome.to_record())
        for extra in outcome.extra_records:
            registry.add(extra)
    return registry


def build_report(outcomes: Iterable[CheckOutcome], *, produced_at: str | None = None) -> VerificationReport:
    outcomes = list(outcomes)
    registry = build_registry(outcomes)
    summary = registry.summary()
    summary["_totals"] = {
        "checks": len(outcomes),
        "pass": sum(o.status == "pass" for o in outcomes),
        "fail": sum(o.status == "fail" for o in outcomes),
        "duration_s": round(sum(o.duration_s for o in outcomes), 3),
    }
    return VerificationReport(tool_version=__version__, produced_at=produced_at, records=registry.records, summary=summary)


def report_markdown(report: VerificationReport) -> str:
    lines = ["# AIPE Magnetics verification report", "",
             f"Tool version: `{report.tool_version}`" + (f"; produced at {report.produced_at}" if report.produced_at else ""),
             "", "Levels: L1 software/schema correctness, L2 numerical and physical consistency, "
             "L3 engineering application validation. Evidence is tracked per capability and operating range; "
             "there is no global validated flag.", ""]
    totals = report.summary.get("_totals", {})
    if totals:
        lines += [f"Checks: {totals.get('checks', 0)}, pass: {totals.get('pass', 0)}, fail: {totals.get('fail', 0)}", ""]
    lines += ["| Capability | Highest passing level | Pass | Fail | Not run |", "| --- | --- | --- | --- | --- |"]
    for cap, info in sorted(report.summary.items()):
        if cap.startswith("_"):
            continue
        lines.append(f"| `{cap}` | {info['highest_passing_level']} | {info['pass']} | {info['fail']} | {info['not_run']} |")
    lines += ["", "## Evidence records", ""]
    for record in sorted(report.records, key=lambda r: (r.capability, r.id)):
        rng = f" range: {record.range.model_dump(mode='json', exclude_none=True)}" if record.range else ""
        metrics = ", ".join(f"{k}={v}" for k, v in record.metrics.items())
        lines.append(f"- **{record.status.upper()}** `{record.id}` ({record.level}, {record.kind}) - {record.description}"
                     f" Method: {record.method}{rng}" + (f" Metrics: {metrics}" if metrics else ""))
        for note in record.notes:
            lines.append(f"  - note: {note}")
    return "\n".join(lines) + "\n"


def write_reports(report: VerificationReport, directory: str | Path, *, overwrite: bool = True) -> tuple[Path, Path]:
    from aipe_magnetics.common.jsonio import dump
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = dump(report.model_dump(mode="json", exclude_none=True), directory / "verification-report.json",
                     overwrite=overwrite)
    md_path = directory / "verification-report.md"
    md_path.write_text(report_markdown(report), encoding="utf-8", newline="\n")
    return json_path, md_path


def bundled_evidence_path() -> Path:
    """Location of the committed evidence file produced by the release verification run."""
    return Path(__file__).resolve().parent / "evidence" / f"v{__version__}.json"


def bundled_evidence_available() -> bool:
    """True when the committed evidence file for this version exists."""
    return bundled_evidence_path().exists()


def load_bundled_registry() -> VerificationRegistry:
    """The bundled evidence registry; when the file is missing, an *empty* registry that says so in every status.

    The absence is never silent: :attr:`VerificationRegistry.missing_source` names the expected file, every
    :meth:`VerificationRegistry.status_for` statement mentions it, and ``aipe-magnetics info`` /
    ``verification_status`` report it.
    """
    path = bundled_evidence_path()
    if not path.exists():
        return VerificationRegistry(missing_source=path)
    return VerificationRegistry.load(path)
