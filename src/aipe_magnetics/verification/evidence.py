"""Capability-level verification evidence.

Verification is tracked per *capability* and per *operating range*, never as a single global flag.

Levels:

* **L1** software and schema correctness (unit tests, schema validation, determinism)
* **L2** numerical and physical consistency (analytical references, independent numerical
  implementations, published reference results, energy consistency, convergence, limiting cases)
* **L3** engineering application validation (comparison with experimental data or reviewed
  engineering cases)

An :class:`EvidenceRecord` says what was checked, how, over which range, with what outcome and
metrics. :class:`VerificationRegistry` answers "what is the verification status of capability X
under conditions Y?" and returns a :class:`VerificationStatus` suitable for result envelopes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

from aipe_magnetics.common.jsonio import dump, load
from aipe_magnetics.common.results import SourceReference, VerificationStatus
from aipe_magnetics.models.contract import ValidityRange

EvidenceLevel = Literal["L1", "L2", "L3"]
EvidenceKind = Literal[
    "software", "schema", "determinism",
    "analytical_reference", "independent_numerical", "published_reference", "physical_consistency",
    "limiting_case", "convergence", "sampling_sensitivity", "energy_consistency", "invalid_input_rejection",
    "experimental_comparison", "engineering_application", "review",
]
EvidenceStatus = Literal["pass", "fail", "not_run"]


class OperatingRange(BaseModel):
    """Range over which an evidence record applies (SI units)."""

    model_config = ConfigDict(extra="forbid")
    materials: list[str] | None = None
    excitation: list[str] | None = None
    frequency: ValidityRange | None = None
    temperature: ValidityRange | None = None
    flux_density_peak: ValidityRange | None = None
    note: str | None = None

    def contains(self, *, frequency: float | None = None, temperature: float | None = None,
                 flux_density_peak: float | None = None, material: str | None = None,
                 excitation: str | None = None) -> bool | None:
        """True/False when the range can be decided; None when no range dimension is declared."""
        decided = False
        for value, rng in ((frequency, self.frequency), (temperature, self.temperature),
                           (flux_density_peak, self.flux_density_peak)):
            if rng is None:
                continue
            decided = True
            if value is None or not rng.contains(float(value)):
                return False
        if self.materials is not None:
            decided = True
            if material is None or material not in self.materials:
                return False
        if self.excitation is not None:
            decided = True
            if excitation is None or excitation not in self.excitation:
                return False
        return True if decided else None


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    capability: str = Field(description="capability or model id, e.g. aipe.magnetics.model.igse")
    level: EvidenceLevel
    kind: EvidenceKind
    status: EvidenceStatus
    description: str
    method: str = Field(description="how the check was performed")
    range: OperatingRange | None = None
    reference: SourceReference | None = None
    check_id: str = Field(description="id of the check function or test that produced this record")
    tool_version: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    produced_at: str | None = None
    notes: list[str] = Field(default_factory=list)


class VerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    report_schema: Literal["aipe.magnetics.verification-report/0.2"] = "aipe.magnetics.verification-report/0.2"
    tool_version: str
    produced_at: str | None = None
    records: list[EvidenceRecord] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)

    def passed(self) -> bool:
        return all(r.status == "pass" for r in self.records if r.status != "not_run") and bool(self.records)


LEVEL_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


class VerificationRegistry:
    """Queryable collection of evidence records."""

    def __init__(self, records: Iterable[EvidenceRecord] = ()):
        self.records: list[EvidenceRecord] = list(records)

    @classmethod
    def from_report(cls, report: VerificationReport) -> "VerificationRegistry":
        return cls(report.records)

    @classmethod
    def load(cls, path: str | Path) -> "VerificationRegistry":
        data = load(path)
        report = VerificationReport.model_validate(data)
        return cls.from_report(report)

    def save(self, path: str | Path, *, tool_version: str, produced_at: str | None = None, overwrite: bool = False) -> Path:
        report = VerificationReport(tool_version=tool_version, produced_at=produced_at, records=self.records,
                                    summary=self.summary())
        return dump(report.model_dump(mode="json", exclude_none=True), path, overwrite=overwrite)

    def add(self, record: EvidenceRecord) -> None:
        if any(r.id == record.id for r in self.records):
            raise ValueError(f"duplicate evidence id {record.id}")
        self.records.append(record)

    def for_capability(self, capability: str) -> list[EvidenceRecord]:
        return [r for r in self.records if r.capability == capability]

    def capabilities(self) -> list[str]:
        return sorted({r.capability for r in self.records})

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for cap in self.capabilities():
            recs = self.for_capability(cap)
            out[cap] = {
                "records": len(recs),
                "pass": sum(r.status == "pass" for r in recs),
                "fail": sum(r.status == "fail" for r in recs),
                "not_run": sum(r.status == "not_run" for r in recs),
                "highest_passing_level": self.highest_level(cap),
            }
        return out

    def highest_level(self, capability: str) -> str:
        best = "L0"
        for r in self.for_capability(capability):
            if r.status == "pass" and LEVEL_ORDER[r.level] > LEVEL_ORDER[best]:
                best = r.level
        return best

    def status_for(self, capability: str, *, frequency: float | None = None, temperature: float | None = None,
                   flux_density_peak: float | None = None, material: str | None = None,
                   excitation: str | None = None) -> VerificationStatus:
        """Derive a :class:`VerificationStatus` for a capability under given SI conditions."""
        recs = self.for_capability(capability)
        if not recs:
            return VerificationStatus(level="L0", state="unverified",
                                      statement=f"No verification evidence recorded for {capability}.")
        if any(r.status == "fail" for r in recs):
            failed = [r.id for r in recs if r.status == "fail"]
            return VerificationStatus(level="L0", state="failed", evidence_refs=failed,
                                      statement=f"Failing verification evidence exists for {capability}: {', '.join(failed)}.")
        passing = [r for r in recs if r.status == "pass"]
        if not passing:
            return VerificationStatus(level="L0", state="not_run", evidence_refs=[r.id for r in recs],
                                      statement=f"Verification checks for {capability} have not been run.")
        in_range: list[EvidenceRecord] = []
        ranged = False
        for r in passing:
            if r.range is None:
                in_range.append(r)
                continue
            decision = r.range.contains(frequency=frequency, temperature=temperature,
                                        flux_density_peak=flux_density_peak, material=material, excitation=excitation)
            if decision is None:
                in_range.append(r)
            else:
                ranged = True
                if decision:
                    in_range.append(r)
        level = "L0"
        for r in in_range:
            if LEVEL_ORDER[r.level] > LEVEL_ORDER[level]:
                level = r.level
        ranged_records = [r for r in passing if r.range is not None]
        in_ranged = [r for r in in_range if r.range is not None]
        if ranged and ranged_records and not in_ranged:
            unranged_level = level
            return VerificationStatus(
                level=unranged_level, state="out_of_range", in_verified_range=False,
                evidence_refs=[r.id for r in in_range],
                statement=(f"Conditions lie outside every verified operating range for {capability}; "
                           f"only range-independent evidence (up to {unranged_level}) applies."))
        state = "verified" if level in ("L2", "L3") else "partially_verified"
        return VerificationStatus(level=level, state=state, in_verified_range=(True if ranged else None),
                                  evidence_refs=[r.id for r in in_range],
                                  statement=f"Passing evidence up to {level} applies to these conditions for {capability}.")
