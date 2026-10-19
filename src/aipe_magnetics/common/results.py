"""The engineering result envelope.

Every API operation returns an :class:`EngineeringResult`. The envelope makes the following
explicit and machine-readable: result values with units, assumptions, the model identity and
version, input conditions, source references, verification status, warnings and limitations.

Results are deterministic: no wall-clock timestamp is inserted unless the caller supplies one.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aipe_magnetics import TOOL_ID, TOOL_NAME, __version__
from aipe_magnetics.common.quantity import Quantity

ValueStatus = Literal["calculated", "assumed", "estimated", "measured", "looked_up"]
VerificationLevel = Literal["L0", "L1", "L2", "L3"]
VerificationState = Literal["not_run", "verified", "partially_verified", "out_of_range", "unverified", "failed"]
AssumptionBasis = Literal["user_input", "default", "datasheet", "engineering_judgement", "model_assumption"]


class ToolIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    name: str
    version: str


def tool_identity() -> ToolIdentity:
    return ToolIdentity(id=TOOL_ID, name=TOOL_NAME, version=__version__)


class ModelIdentity(BaseModel):
    """Identity of the mathematical model that produced a value."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(description="stable model identifier, e.g. aipe.magnetics.model.igse")
    name: str
    version: str = Field(description="model implementation version (semver)")
    family: Literal["empirical", "physics", "dynamic", "data_driven", "analytical", "lookup"]
    method_version: str | None = Field(default=None, description="version of the numerical method, if separate")


class SourceReference(BaseModel):
    """Where a number, parameter or equation came from."""

    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["datasheet", "publication", "repository", "dataset", "standard", "assumption", "internal",
                  "measurement_record", "user_input"]
    reference: str = Field(description="human-readable citation or title")
    locator: str | None = Field(default=None, description="page, table, figure, column, record field")
    url: str | None = None
    version: str | None = Field(default=None, description="edition / commit / dataset version")
    license: str = Field(default="NOASSERTION", description="SPDX identifier or NOASSERTION")
    sha256: str | None = None
    note: str | None = None


class Assumption(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    statement: str
    basis: AssumptionBasis
    affects: list[str] = Field(default_factory=list, description="result value keys influenced by this assumption")


class Warning_(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"


class Limitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    statement: str


class VerificationStatus(BaseModel):
    """Verification state for *this* result, derived from the capability-level evidence registry."""

    model_config = ConfigDict(extra="forbid")
    level: VerificationLevel = Field(default="L0", description="highest verification level with passing evidence")
    state: VerificationState = "not_run"
    statement: str = Field(default="No verification evidence consulted.")
    evidence_refs: list[str] = Field(default_factory=list)
    in_verified_range: bool | None = Field(default=None, description="None when no range is declared")


class ResultValue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    quantity: Quantity
    status: ValueStatus = "calculated"
    derived_from: list[str] = Field(default_factory=list, description="input or value keys this depends on")
    note: str | None = None


class ErrorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class EngineeringResult(BaseModel):
    """Typed, validated output of every engineering capability."""

    model_config = ConfigDict(extra="forbid")

    result_schema: Literal["aipe.magnetics.result/0.2"] = "aipe.magnetics.result/0.2"
    capability: str = Field(description="capability id, e.g. aipe.magnetics.estimate_core_loss")
    status: Literal["ok", "rejected", "error"] = "ok"
    tool: ToolIdentity = Field(default_factory=tool_identity)
    model: ModelIdentity | None = None
    inputs: dict[str, Any] = Field(default_factory=dict, description="validated input conditions (JSON-serialisable)")
    values: dict[str, ResultValue] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict,
                                 description="structured non-scalar payload (tables, records, arrays), JSON-serialisable")
    assumptions: list[Assumption] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    verification: VerificationStatus = Field(default_factory=VerificationStatus)
    warnings: list[Warning_] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)
    errors: list[ErrorRecord] = Field(default_factory=list)
    produced_at: str | None = Field(default=None, description="RFC 3339 timestamp supplied by the caller, else None")
    extensions: dict[str, Any] = Field(default_factory=dict)

    # -- convenience ----------------------------------------------------------------------------
    def value(self, key: str) -> Quantity:
        return self.values[key].quantity

    def magnitude(self, key: str, unit: str) -> float:
        return self.values[key].quantity.magnitude(unit)

    def add_value(self, key: str, label: str, quantity: Quantity, *, status: ValueStatus = "calculated",
                  derived_from: list[str] | None = None, note: str | None = None) -> "EngineeringResult":
        self.values[key] = ResultValue(label=label, quantity=quantity, status=status,
                                       derived_from=list(derived_from or []), note=note)
        return self

    def warn(self, code: str, message: str, severity: Literal["info", "warning", "error"] = "warning") -> None:
        self.warnings.append(Warning_(code=code, message=message, severity=severity))

    def limit(self, code: str, statement: str) -> None:
        if not any(item.code == code for item in self.limitations):
            self.limitations.append(Limitation(code=code, statement=statement))

    def assume(self, id: str, statement: str, basis: AssumptionBasis, affects: list[str] | None = None) -> None:
        if not any(item.id == id for item in self.assumptions):
            self.assumptions.append(Assumption(id=id, statement=statement, basis=basis, affects=list(affects or [])))

    def cite(self, source: SourceReference) -> None:
        if not any(item.id == source.id for item in self.sources):
            self.sources.append(source)

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def rejected(cls, capability: str, error: Exception, *, inputs: dict[str, Any] | None = None) -> "EngineeringResult":
        from aipe_magnetics.common.errors import AIPEMagneticsError
        if isinstance(error, AIPEMagneticsError):
            record = ErrorRecord(code=error.code, message=error.message, details=error.details)
        else:
            record = ErrorRecord(code=type(error).__name__, message=str(error))
        return cls(capability=capability, status="rejected", inputs=dict(inputs or {}), errors=[record],
                   verification=VerificationStatus(state="not_run", statement="Input rejected; nothing evaluated."))
