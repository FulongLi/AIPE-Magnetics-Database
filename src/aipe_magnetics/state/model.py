"""Magnetic Engineering State (MES) 0.3.0 - the versioned, machine-readable record of a magnetic design lifecycle.

One JSON document holds the component definition, requirements, excitation, assumptions and open questions,
selected database records (with provenance), winding and geometry definitions, solver configurations, results
(each value tagged with the method that produced it), optimisation record, verification status, uncertainty and the
artifacts (files) produced. Physical values are ``{"value", "unit", "method"?}`` objects in registered units.

This is the *domain* state of AIPE Magnetics. It does not compete with the ecosystem-wide AIPE-Core Engineering State:
:mod:`aipe_magnetics.state.core_adapter` maps an MES onto a Core 0.1.0 state (and seeds an MES from one) through the
tested Core adapter of v0.2.
"""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aipe_magnetics.common.quantity import Uncertainty
from aipe_magnetics.common.results import SourceReference

STATE_SCHEMA = "aipe.magnetics.engineering-state/0.3.0"
STATE_VERSION = "0.3.0"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-|]*$")

Method = Literal["specified", "assumed", "looked_up", "analytical", "fea", "fea_post_processed", "fea_2p5d",
                 "analytical_thermal_network", "measured", "derived"]
LifecycleStage = Literal["specification", "database_selection", "candidate_generation", "geometry", "electromagnetic_analysis",
                         "thermal_analysis", "optimisation", "verification", "exported"]
ManufacturingReadiness = Literal["preliminary_design", "prototype_ready", "manufacturing_qualified"]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QV(_M):
    """Physical value with unit and the method that produced it."""

    value: float
    unit: str
    method: Method | None = None
    description: str | None = None
    uncertainty: Uncertainty | None = None

    @field_validator("value")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("value must be finite")
        return float(v)

    @field_validator("unit")
    @classmethod
    def _unit(cls, v: str) -> str:
        from aipe_magnetics.common.units import units
        units.get(v)
        return v


class ToolRef(_M):
    id: str = "aipe.magnetics"
    name: str = "AIPE Magnetics"
    version: str


class Component(_M):
    type: Literal["inductor", "transformer", "coupled_inductor"]
    application: str | None = None
    topology: str | None = None
    description: str | None = None
    windings: list[str] = Field(default_factory=list)


class Requirements(_M):
    electrical: dict[str, QV] = Field(default_factory=dict)
    constraints: dict[str, QV] = Field(default_factory=dict)
    environment: dict[str, QV] = Field(default_factory=dict)
    cooling: dict[str, Any] = Field(default_factory=dict)
    isolation: dict[str, Any] = Field(default_factory=dict)
    specified_fields: list[str] = Field(default_factory=list)


class Waveform(_M):
    quantity: Literal["current", "voltage", "flux_density", "flux_linkage"]
    kind: Literal["piecewise_linear", "sinusoidal", "square"] = "piecewise_linear"
    unit: str
    breakpoints: list[list[float]] = Field(default_factory=list, description="[[t/T, value], ...]")
    rms: float | None = None
    peak: float | None = None
    mean: float | None = None
    description: str | None = None


class OperatingPoint(_M):
    id: str
    label: str
    frequency: QV
    waveforms: dict[str, Waveform] = Field(default_factory=dict, description="winding / quantity name -> waveform")
    quantities: dict[str, QV] = Field(default_factory=dict)
    role: Literal["nominal", "worst_case", "corner", "other"] = "other"


class Excitation(_M):
    fundamental_frequency: QV
    operating_points: list[OperatingPoint] = Field(default_factory=list)
    description: str | None = None


class StateAssumption(_M):
    id: str
    statement: str
    basis: Literal["user_input", "default", "datasheet", "engineering_judgement", "model_assumption"] = "default"
    status: Literal["specified", "assumed", "requested"] = "assumed"
    critical: bool = False
    affects: list[str] = Field(default_factory=list)


class OpenQuestion(_M):
    id: str
    field: str
    question: str
    assumed_answer: str | None = None
    critical: bool = False


class MaterialSelection(_M):
    role: Literal["core", "conductor", "winding_pack", "bobbin", "insulation", "coolant"]
    record_id: str
    name: str | None = None
    origin: Literal["catalogue", "property_library", "user"] = "catalogue"
    record_schema_version: str | None = None
    properties: dict[str, QV] = Field(default_factory=dict)
    sources: list[SourceReference] = Field(default_factory=list)
    selected: bool = True
    notes: list[str] = Field(default_factory=list)


class CoreSelection(_M):
    record_id: str
    name: str
    shape: str | None = None
    part_number: str | None = None
    selected: bool = False
    dimensions: dict[str, QV] = Field(default_factory=dict)
    effective_parameters: dict[str, QV] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class WindingLayer(_M):
    id: str
    turns: int = Field(ge=1)
    inner_distance: QV
    outer_distance: QV
    mean_turn_length: QV


class WindingDefinition(_M):
    name: str
    turns: int = Field(ge=1)
    conductor: dict[str, Any]
    sections: int = Field(default=1, ge=1)
    layers: list[WindingLayer] = Field(default_factory=list)
    conductor_length: QV | None = None
    resistance_dc: QV | None = None


class GeometryRecord(_M):
    model_id: str
    model_version: str
    parameter_hash: str
    parameters: dict[str, QV] = Field(default_factory=dict, description="editable parameters (path -> value)")
    derived: dict[str, QV] = Field(default_factory=dict)
    checks: list[dict[str, Any]] = Field(default_factory=list)
    model: dict[str, Any] = Field(default_factory=dict, description="complete parameter set (ECoreMagnetic JSON) for regeneration")
    artifacts: list[str] = Field(default_factory=list)


class SolverConfiguration(_M):
    id: str
    domain: Literal["electromagnetic", "thermal", "optimisation", "geometry"]
    backend: str
    version: str
    fidelity: Method
    formulation: str
    settings: dict[str, Any] = Field(default_factory=dict)
    configuration_artifact: str | None = None
    status: Literal["configured", "completed", "failed", "not_run"] = "completed"
    limitations: list[str] = Field(default_factory=list)


class ResultSet(_M):
    id: str
    solver_ref: str
    operating_point_ref: str | None = None
    quantities: dict[str, QV] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LossBreakdown(_M):
    operating_point_ref: str | None = None
    core: QV | None = None
    windings: dict[str, QV] = Field(default_factory=dict)
    total: QV | None = None
    notes: list[str] = Field(default_factory=list)


class Results(_M):
    electromagnetic: list[ResultSet] = Field(default_factory=list)
    thermal: list[ResultSet] = Field(default_factory=list)
    losses: list[LossBreakdown] = Field(default_factory=list)
    comparison: dict[str, Any] = Field(default_factory=dict, description="e.g. against a baseline design")


class OptimisationRecord(_M):
    strategy: str
    objectives: list[dict[str, Any]] = Field(default_factory=list)
    primary_objective: str | None = None
    constraints: list[str] = Field(default_factory=list)
    variables: dict[str, Any] = Field(default_factory=dict)
    stages: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)
    ranked: list[dict[str, Any]] = Field(default_factory=list)
    pareto: list[dict[str, Any]] = Field(default_factory=list)
    selected: str | None = None
    rejection_reasons: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)


class VerificationLevelStatus(_M):
    level: Literal["software_schema", "numerical_physical", "external_reference", "experimental"]
    status: Literal["passed", "partial", "failed", "not_performed"]
    statement: str
    evidence: list[str] = Field(default_factory=list)


class VerificationRecord(_M):
    levels: list[VerificationLevelStatus] = Field(default_factory=list)
    checks: list[dict[str, Any]] = Field(default_factory=list)
    manufacturing_readiness: ManufacturingReadiness = "preliminary_design"
    statement: str = ""

    @model_validator(mode="after")
    def _no_unfounded_claims(self) -> "VerificationRecord":
        exp = [lvl for lvl in self.levels if lvl.level == "experimental"]
        if any(lvl.status in ("passed", "partial") and not lvl.evidence for lvl in exp):
            raise ValueError("experimental validation cannot be claimed without measurement evidence references")
        if self.manufacturing_readiness != "preliminary_design" and not any(
                lvl.level == "experimental" and lvl.status == "passed" for lvl in self.levels):
            raise ValueError("manufacturing readiness beyond 'preliminary_design' requires passed experimental validation")
        return self


class ArtifactRecord(_M):
    id: str
    path: str = Field(description="path relative to the state file directory")
    media_type: str
    role: Literal["specification", "cad", "mesh", "solver_configuration", "field_data", "results", "plot", "report", "bom",
                  "selection_record", "verification_report", "state", "other"]
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    size_bytes: int | None = Field(default=None, ge=0)
    description: str | None = None
    produced_by: str | None = None


class ProvenanceRecord(_M):
    tool: ToolRef
    inputs: list[str] = Field(default_factory=list)
    database: dict[str, Any] = Field(default_factory=dict)
    software: dict[str, str] = Field(default_factory=dict)
    sources: list[SourceReference] = Field(default_factory=list)


class UncertaintyItem(_M):
    quantity: str
    kind: Literal["bounds", "standard", "sensitivity", "unknown"]
    lower: float | None = None
    upper: float | None = None
    unit: str | None = None
    basis: str


class HistoryEvent(_M):
    stage: LifecycleStage
    description: str
    at: str | None = None


class MagneticEngineeringState(_M):
    state_schema: Literal["aipe.magnetics.engineering-state/0.3.0"] = STATE_SCHEMA
    state_version: Literal["0.3.0"] = STATE_VERSION
    id: str
    title: str
    created_at: str | None = Field(default=None, description="RFC 3339, caller supplied (never wall-clock by default)")
    lifecycle_stage: LifecycleStage = "specification"
    component: Component
    requirements: Requirements = Field(default_factory=Requirements)
    excitation: Excitation | None = None
    assumptions: list[StateAssumption] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    materials: list[MaterialSelection] = Field(default_factory=list)
    cores: list[CoreSelection] = Field(default_factory=list)
    windings: list[WindingDefinition] = Field(default_factory=list)
    geometry: GeometryRecord | None = None
    solvers: list[SolverConfiguration] = Field(default_factory=list)
    results: Results = Field(default_factory=Results)
    optimisation: OptimisationRecord | None = None
    verification: VerificationRecord = Field(default_factory=VerificationRecord)
    provenance: ProvenanceRecord
    uncertainty: list[UncertaintyItem] = Field(default_factory=list)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    history: list[HistoryEvent] = Field(default_factory=list)
    extensions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not _ID.match(v):
            raise ValueError(f"invalid state id {v!r}")
        return v

    @model_validator(mode="after")
    def _references(self) -> "MagneticEngineeringState":
        artifact_ids = {a.id for a in self.artifacts}
        if len(artifact_ids) != len(self.artifacts):
            raise ValueError("artifact ids must be unique")
        solver_ids = {s.id for s in self.solvers}
        if len(solver_ids) != len(self.solvers):
            raise ValueError("solver ids must be unique")
        op_ids = {op.id for op in self.excitation.operating_points} if self.excitation else set()
        refs: list[str] = []
        for rs in self.results.electromagnetic + self.results.thermal:
            if rs.solver_ref not in solver_ids:
                raise ValueError(f"result {rs.id} references unknown solver {rs.solver_ref}")
            if rs.operating_point_ref and rs.operating_point_ref not in op_ids:
                raise ValueError(f"result {rs.id} references unknown operating point {rs.operating_point_ref}")
            refs += rs.artifacts
        for s in self.solvers:
            if s.configuration_artifact:
                refs.append(s.configuration_artifact)
        if self.geometry:
            refs += self.geometry.artifacts
        if self.optimisation:
            refs += self.optimisation.artifacts
        missing = sorted(set(refs) - artifact_ids)
        if missing:
            raise ValueError(f"unknown artifact reference(s) {missing}")
        assumption_ids = [a.id for a in self.assumptions]
        if len(set(assumption_ids)) != len(assumption_ids):
            raise ValueError("assumption ids must be unique")
        return self


__all__ = [
    "ArtifactRecord", "Component", "CoreSelection", "Excitation", "GeometryRecord", "HistoryEvent", "LifecycleStage", "LossBreakdown",
    "MagneticEngineeringState", "MaterialSelection", "Method", "OpenQuestion", "OperatingPoint", "OptimisationRecord", "ProvenanceRecord",
    "QV", "Requirements", "ResultSet", "Results", "STATE_SCHEMA", "STATE_VERSION", "SolverConfiguration", "StateAssumption", "ToolRef",
    "UncertaintyItem", "VerificationLevelStatus", "VerificationRecord", "Waveform", "WindingDefinition", "WindingLayer",
]
