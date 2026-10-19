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

    def decide(self, *, frequency: float | None = None, temperature: float | None = None,
               flux_density_peak: float | None = None, material: str | None = None,
               excitation: str | None = None) -> tuple[bool | None, list[str]]:
        """``(decision, undecided)`` for the supplied conditions.

        ``decision`` is ``False`` as soon as one *supplied* condition lies outside its declared range,
        ``True`` when every declared dimension was supplied and lies inside, and ``None`` when the range
        cannot be decided: no dimension is declared, or a declared dimension was not supplied (listed in
        ``undecided``). A missing condition is never treated as "outside".
        """
        inside = False
        undecided: list[str] = []
        for name, value, rng in (("frequency", frequency, self.frequency), ("temperature", temperature, self.temperature),
                                 ("flux_density_peak", flux_density_peak, self.flux_density_peak)):
            if rng is None:
                continue
            if value is None:
                undecided.append(name)
                continue
            if not rng.contains(float(value)):
                return False, undecided
            inside = True
        for name, value, allowed in (("material", material, self.materials), ("excitation", excitation, self.excitation)):
            if allowed is None:
                continue
            if value is None:
                undecided.append(name)
                continue
            if value not in allowed:
                return False, undecided
            inside = True
        if inside and not undecided:
            return True, []
        return None, undecided

    def contains(self, *, frequency: float | None = None, temperature: float | None = None,
                 flux_density_peak: float | None = None, material: str | None = None,
                 excitation: str | None = None) -> bool | None:
        """True/False when the range can be decided from the supplied conditions; None when it cannot."""
        return self.decide(frequency=frequency, temperature=temperature, flux_density_peak=flux_density_peak,
                           material=material, excitation=excitation)[0]


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

    def __init__(self, records: Iterable[EvidenceRecord] = (), *, missing_source: Path | None = None):
        self.records: list[EvidenceRecord] = list(records)
        #: path of a bundled evidence file that should have been loaded but does not exist (reported, never hidden)
        self.missing_source: Path | None = missing_source

    @property
    def evidence_available(self) -> bool:
        return bool(self.records) and self.missing_source is None

    def _missing_note(self) -> str:
        if self.missing_source is None:
            return ""
        return (f" The bundled evidence file {self.missing_source} is missing, so no verification evidence is available "
                "to any result; a maintainer records it with 'aipe-magnetics verify --update-bundled'.")

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
                                      statement=f"No verification evidence recorded for {capability}." + self._missing_note())
        if any(r.status == "fail" for r in recs):
            failed = [r.id for r in recs if r.status == "fail"]
            return VerificationStatus(level="L0", state="failed", evidence_refs=failed,
                                      statement=f"Failing verification evidence exists for {capability}: {', '.join(failed)}.")
        passing = [r for r in recs if r.status == "pass"]
        if not passing:
            return VerificationStatus(level="L0", state="not_run", evidence_refs=[r.id for r in recs],
                                      statement=f"Verification checks for {capability} have not been run.")
        applicable: list[EvidenceRecord] = []   # unranged, ranged-and-inside, or ranged-but-undecidable records
        confirmed: list[EvidenceRecord] = []    # ranged records whose declared range contains the supplied conditions
        unchecked: list[EvidenceRecord] = []    # ranged records whose range could not be decided (condition not supplied)
        outside: list[EvidenceRecord] = []
        undecided_dims: set[str] = set()
        for r in passing:
            if r.range is None:
                applicable.append(r)
                continue
            decision, undecided = r.range.decide(frequency=frequency, temperature=temperature,
                                                 flux_density_peak=flux_density_peak, material=material, excitation=excitation)
            if decision is True:
                applicable.append(r)
                confirmed.append(r)
            elif decision is False:
                outside.append(r)
            else:
                applicable.append(r)
                if undecided:
                    unchecked.append(r)
                    undecided_dims.update(undecided)
        level = "L0"
        for r in applicable:
            if LEVEL_ORDER[r.level] > LEVEL_ORDER[level]:
                level = r.level
        if outside and not confirmed and not unchecked:
            unranged_level = level
            return VerificationStatus(
                level=unranged_level, state="out_of_range", in_verified_range=False,
                evidence_refs=[r.id for r in applicable],
                statement=(f"Conditions lie outside every verified operating range for {capability}; "
                           f"only range-independent evidence (up to {unranged_level}) applies."))
        state = "verified" if level in ("L2", "L3") else "partially_verified"
        if confirmed:
            in_verified_range: bool | None = True
            statement = f"Passing evidence up to {level} applies to these conditions for {capability}."
        elif unchecked:
            in_verified_range = None
            dims = ", ".join(sorted(undecided_dims))
            statement = (f"Passing evidence up to {level} exists for {capability}; its operating range was not checked because "
                         f"no {dims} condition was supplied.")
        else:
            in_verified_range = None
            statement = f"Passing evidence up to {level} applies to {capability} (no operating range declared)."
        if outside:
            statement += f" {len(outside)} ranged record(s) do not cover the supplied conditions."
        return VerificationStatus(level=level, state=state, in_verified_range=in_verified_range,
                                  evidence_refs=[r.id for r in applicable], statement=statement)
