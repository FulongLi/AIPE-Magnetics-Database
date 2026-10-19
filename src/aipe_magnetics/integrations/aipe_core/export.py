"""Executable round-trip adapter between AIPE Magnetics summaries and the Core Engineering State 0.1.0.

Mapping (Core ``magnetics.schema.json`` / ``docs/metadata.md``; repository ``docs/core-integration.md``):

==========================  ==================================================================
AIPE Magnetics summary      Core Engineering State location / rule
==========================  ==================================================================
``id, role, kind``          ``magnetics[].id/role/kind`` (kind: inductor | transformer | coupled_inductor | other)
``design_status``           ``magnetics[].design_status`` (unspecified | candidate | selected)
``inductance``              ``magnetics[].inductance`` as SI ``{"value", "unit": "H"}``, value >= 0
``leakage_inductance``      ``magnetics[].leakage_inductance`` as SI ``H``, value >= 0
``turns_ratio``             ``magnetics[].turns_ratio`` as dimensionless ``{"value", "unit": "1"}``, value > 0
``core_material_ref``       ``magnetics[].core_material_ref`` (uri-reference; never invented material data)
``database_ref``            ``magnetics[].database_ref`` (uri-reference to a versioned domain record)
``evidence[]``              appended to ``state.evidence`` and listed in ``magnetics[].evidence_refs``
``limitations, detail``     ``magnetics[].extensions["aipe.magnetics-detail"]`` (namespaced; never Core fields)
original units/uncertainty  same extension, key ``quantities`` (Core fields always carry coherent SI)
==========================  ==================================================================

Evidence rules enforced before export (Core ``evidence.schema.json`` and ``validate.py``):

* kinds emitted by this tool are ``assumption``, ``analytical`` and ``source`` only; the magnetics
  package runs no simulation or measurement, so it never fabricates such evidence;
* analytical evidence must list the evidence ids it was derived from (``provenance.inputs``);
* ``provenance.created_at`` is an RFC 3339 timestamp supplied by the caller (never wall-clock time);
* rejected evidence cannot support a component; evidence ids are unique within a state.

Validation status: the exported state carries ``validation.status = "not_run"`` unless the caller
supplies at least one *passing* analytical check that references analytical evidence, in which
case the status is ``analytically_checked``. Any failing check forces ``failed``. An existing
base-state status is never downgraded and never promoted beyond what the supplied checks support.

Extending a base state (``base_state=...``) never modifies anything silently:

* evidence ids and validation-check ids must be new or byte-identical to the existing record
  (identical re-exports are no-ops, so a re-export is idempotent); a different record under an
  existing id is rejected;
* a requirement scalar already present in the base with a different value is a *conflict* and is
  rejected unless ``replace_requirements=True``; a replacement keeps the previous evidence refs
  (they may still support the unchanged scalars) but records, per replaced field, the previous
  value, the evidence refs that documented it and the replacement time in
  ``requirements.extensions["aipe.magnetics-detail"].superseded`` and adds a human-readable entry
  to ``validation.limitations`` (checks recorded earlier were evaluated against the previous
  requirements);
* ``replace_existing=True`` replaces the magnetics entry with the same id only. Evidence records
  and validation checks already in the state are never removed (Core evidence is an audit trail);
  a check that referenced the replaced entry's evidence therefore remains and keeps driving
  ``validation.status``.

Round trip: ``summary -> to_engineering_state -> validate_engineering_state == [] ->
from_engineering_state -> [summary]`` holds field for field (``None`` fields stay ``None``) for
every summary whose analytical inputs are among its own evidence (the summaries this package
produces). On import the evidence set is closed under ``provenance.inputs``: evidence that an
entry's analytical records were derived from (for example requirement evidence) is appended after
the entry's own ``evidence_refs``, so an imported summary always re-exports standalone; for such
non-closed entries the imported summary carries more evidence than the entry lists.
"""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any, Literal, Sequence

from jsonschema import FormatChecker
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from aipe_magnetics import TOOL_ID, TOOL_NAME, __version__
from aipe_magnetics.common.errors import DataIntegrityError, InvalidInputError, UnitError, UnsupportedOperationError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.integrations.aipe_core.state import (
    CORE_SCHEMA_VERSION,
    REQUIRED_FORMATS,
    validate_engineering_state,
)

#: Core ``common.schema.json#/$defs/id``.
CORE_ID_PATTERN = r"^[a-z][a-z0-9]*(?:[.:-][a-z0-9]+(?:-[a-z0-9]+)*)+$"
#: Core ``common.schema.json#/$defs/extensions`` property-name pattern.
CORE_EXTENSION_KEY_PATTERN = r"^[a-z][a-z0-9-]*[.][a-z][a-z0-9_.-]*$"
#: Namespaced extension that carries magnetics detail inside a Core magnetics entry.
EXTENSION_KEY = "aipe.magnetics-detail"
DETAIL_SCHEMA = "aipe.magnetics.core-detail/0.2"
DEFAULT_LIMITATION = ("Magnetics entries exported by aipe.magnetics are analytical estimates with stated "
                      "assumptions; no simulation, measurement, thermal analysis or hardware qualification is claimed.")

_CORE_ID = re.compile(CORE_ID_PATTERN)
_EXTENSION_KEY = re.compile(CORE_EXTENSION_KEY_PATTERN)
_FORMAT = FormatChecker()
_DETAIL_SCHEMA_PREFIX = DETAIL_SCHEMA.rsplit("/", 1)[0] + "/"
_STATUS_ORDER = {"not_run": 0, "schema_validated": 1, "analytically_checked": 2, "simulated": 3, "measured": 4,
                 "reviewed": 5, "failed": 6}
_REQUIREMENT_UNITS = (("nominal_power", "W"), ("input_voltage", "V"), ("output_voltage", "V"),
                      ("switching_frequency", "Hz"))
_COMPONENT_UNITS = (("inductance", "H"), ("leakage_inductance", "H"), ("turns_ratio", "1"))
#: Relative tolerance under which two Core SI values are the same number (re-export idempotency).
_SAME_VALUE_REL_TOL = 1e-9

ComponentKind = Literal["inductor", "transformer", "coupled_inductor", "other"]
DesignStatus = Literal["unspecified", "candidate", "selected"]
ExportedEvidenceKind = Literal["assumption", "analytical", "source"]
SourceType = Literal["user_requirement", "document", "repository", "datasheet", "publication", "measurement_record",
                     "synthetic_example"]
ReviewStatus = Literal["unreviewed", "reviewed", "rejected"]
ProjectPurpose = Literal["example", "research", "education", "engineering"]
SUPPORTED_EVIDENCE_KINDS = frozenset({"assumption", "analytical", "source"})


# --------------------------------------------------------------------------------------------------
# Field validators shared by the models
# --------------------------------------------------------------------------------------------------
def _core_id(value: str, label: str = "id") -> str:
    if not isinstance(value, str) or not _CORE_ID.match(value):
        raise ValueError(f"{label} {value!r} must match Core id pattern {CORE_ID_PATTERN}")
    return value


def _format_checker() -> FormatChecker:
    """The format checker, verified to carry every format this module relies on.

    ``FormatChecker.conforms`` returns True for a format it has no checker for, so without this
    guard an installation lacking ``jsonschema[format]`` would accept invalid timestamps and URI
    references. The adapter fails closed instead (same rule as ``state.state_validator``).
    """
    missing = sorted(REQUIRED_FORMATS - set(_FORMAT.checkers))
    if missing:
        raise UnsupportedOperationError(
            "JSON Schema format validators are unavailable: " + ", ".join(missing) + "; install jsonschema[format]",
            details={"missing_formats": missing})
    return _FORMAT


def _rfc3339(value: str, label: str = "created_at") -> str:
    if not isinstance(value, str) or not _format_checker().conforms(value, "date-time"):
        raise ValueError(f"{label} {value!r} must be an RFC 3339 date-time with time zone (e.g. 2026-10-07T00:00:00Z)")
    return value


def _uri_reference(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or not _format_checker().conforms(value, "uri-reference"):
        raise ValueError(f"{label} {value!r} must be a non-empty RFC 3986 URI reference")
    return value


def _unique(values: Sequence[str], label: str) -> None:
    seen: set[str] = set()
    for item in values:
        if item in seen:
            raise ValueError(f"duplicate {label}: {item}")
        seen.add(item)


def _namespaced_extensions(value: dict[str, Any]) -> dict[str, Any]:
    for key in value:
        if not _EXTENSION_KEY.match(key):
            raise ValueError(f"extension key {key!r} must be namespaced ({CORE_EXTENSION_KEY_PATTERN})")
    return _json_data(value, "extensions")


def _json_data(value: Any, label: str) -> Any:
    """Normalise through JSON so exported and re-imported payloads compare equal (tuples -> lists ...)."""
    try:
        return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be JSON data with finite numbers: {error}") from None


def _quantity_in(value: Quantity | None, unit: str, label: str, *, positive: bool, nonnegative: bool) -> Quantity | None:
    if value is None:
        return None
    try:
        si = value.to(unit)
    except UnitError as error:
        raise ValueError(f"{label}: {error.message}") from None
    if positive and si.value <= 0:
        raise ValueError(f"{label} must be positive, got {value.value} {value.unit}")
    if nonnegative and si.value < 0:
        raise ValueError(f"{label} must be non-negative, got {value.value} {value.unit}")
    return value


# --------------------------------------------------------------------------------------------------
# Core evidence models (Core evidence.schema.json / common.schema.json)
# --------------------------------------------------------------------------------------------------
class CoreAttribution(BaseModel):
    """Core ``attribution``: ``{id, version, name?}`` of an actual tool/skill/agent."""

    model_config = ConfigDict(extra="forbid")
    id: str
    version: str = Field(min_length=1)
    name: str | None = Field(default=None, min_length=1)

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        return _core_id(value, "attribution id")


def tool_attribution() -> CoreAttribution:
    """Attribution of this package (``aipe.magnetics`` at the installed version)."""
    return CoreAttribution(id=TOOL_ID, version=__version__, name=TOOL_NAME)


class CoreSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: SourceType
    reference: str = Field(min_length=1, description="original citation, document title, record id or URL")
    locator: str | None = Field(default=None, min_length=1, description="table, page, figure or record field")


class CoreArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uri: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    description: str | None = Field(default=None, min_length=1)

    @field_validator("uri")
    @classmethod
    def _uri(cls, value: str) -> str:
        return _uri_reference(value, "artifact uri")  # type: ignore[return-value]


class CoreProvenance(BaseModel):
    """Core evidence provenance; ``created_at`` is caller supplied (actual event time, never build time)."""

    model_config = ConfigDict(extra="forbid")
    created_at: str
    activity: str = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list, description="ids of the evidence this record was derived from")
    tool: CoreAttribution | None = None
    skill: CoreAttribution | None = None
    agent: CoreAttribution | None = None
    authors: list[str] | None = None

    @field_validator("created_at")
    @classmethod
    def _created(cls, value: str) -> str:
        return _rfc3339(value, "provenance.created_at")

    @field_validator("inputs")
    @classmethod
    def _inputs(cls, value: list[str]) -> list[str]:
        for item in value:
            _core_id(item, "provenance input")
        _unique(value, "provenance input")
        return value

    @field_validator("authors")
    @classmethod
    def _authors(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(not author for author in value):
            raise ValueError("authors must be non-empty strings")
        return value


class CoreEvidence(BaseModel):
    """One Core evidence record of a kind this tool may emit (assumption, analytical, source)."""

    model_config = ConfigDict(extra="forbid")
    id: str
    kind: ExportedEvidenceKind
    description: str = Field(min_length=1)
    source: CoreSource
    provenance: CoreProvenance
    artifacts: list[CoreArtifact] = Field(default_factory=list)
    review_status: ReviewStatus = "unreviewed"
    extensions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        return _core_id(value, "evidence id")

    @field_validator("extensions")
    @classmethod
    def _ext(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _namespaced_extensions(value)

    @model_validator(mode="after")
    def _rules(self) -> "CoreEvidence":
        # Core evidence.schema.json: analytical evidence requires at least one input evidence id.
        if self.kind == "analytical" and not self.provenance.inputs:
            raise ValueError(f"analytical evidence {self.id} must list the input evidence ids it was derived from")
        return self

    def to_core(self) -> dict[str, Any]:
        record = self.model_dump(mode="json", exclude_none=True)
        if not record.get("extensions"):
            record.pop("extensions", None)
        return record


class CoreValidationCheck(BaseModel):
    """A validation check entry (Core ``validation.schema.json``). This tool emits analytical checks only."""

    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["analytical"] = "analytical"
    status: Literal["not_run", "pass", "fail"]
    description: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        return _core_id(value, "check id")

    @field_validator("evidence_refs")
    @classmethod
    def _refs(cls, value: list[str]) -> list[str]:
        for item in value:
            _core_id(item, "check evidence reference")
        _unique(value, "check evidence reference")
        return value

    def to_core(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# --------------------------------------------------------------------------------------------------
# Design requirements and the component summary
# --------------------------------------------------------------------------------------------------
class DesignRequirements(BaseModel):
    """Converter-level design inputs (Core ``requirements.schema.json``) with the evidence that records them."""

    model_config = ConfigDict(extra="forbid")
    nominal_power: Quantity | None = None
    input_voltage: Quantity | None = None
    output_voltage: Quantity | None = None
    switching_frequency: Quantity | None = None
    evidence: list[CoreEvidence] = Field(default_factory=list,
                                         description="evidence recording where the requirements come from")

    @field_validator("nominal_power")
    @classmethod
    def _power(cls, value: Quantity | None) -> Quantity | None:
        return _quantity_in(value, "W", "nominal_power", positive=False, nonnegative=True)

    @field_validator("input_voltage", "output_voltage")
    @classmethod
    def _voltage(cls, value: Quantity | None) -> Quantity | None:
        return _quantity_in(value, "V", "voltage", positive=False, nonnegative=False)

    @field_validator("switching_frequency")
    @classmethod
    def _frequency(cls, value: Quantity | None) -> Quantity | None:
        return _quantity_in(value, "Hz", "switching_frequency", positive=True, nonnegative=True)

    @model_validator(mode="after")
    def _rules(self) -> "DesignRequirements":
        _unique([item.id for item in self.evidence], "requirements evidence id")
        quantities = [name for name, _ in _REQUIREMENT_UNITS if getattr(self, name) is not None]
        if quantities and not self.evidence:
            raise ValueError("Core requires evidence_refs for physical quantities; supply at least one evidence "
                             f"record for {', '.join(quantities)}")
        return self


class MagneticComponentSummary(BaseModel):
    """What AIPE Magnetics says about one magnetic component, in Core-compatible terms."""

    model_config = ConfigDict(extra="forbid")
    id: str
    role: str = Field(min_length=1, description="function in the converter, e.g. 'DAB isolation transformer'")
    kind: ComponentKind
    design_status: DesignStatus = "unspecified"
    inductance: Quantity | None = Field(default=None, description="magnetising or main inductance (dimension H)")
    leakage_inductance: Quantity | None = Field(default=None, description="dimension H")
    turns_ratio: Quantity | None = Field(default=None, description="dimensionless, > 0")
    core_material_ref: str | None = Field(default=None, description="uri-reference to a material record")
    database_ref: str | None = Field(default=None, description="uri-reference to a versioned domain record")
    evidence: list[CoreEvidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    detail: dict[str, Any] = Field(default_factory=dict,
                                   description="JSON domain detail kept in extensions['aipe.magnetics-detail']")

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        return _core_id(value, "component id")

    @field_validator("inductance", "leakage_inductance")
    @classmethod
    def _inductance(cls, value: Quantity | None, info: Any) -> Quantity | None:
        return _quantity_in(value, "H", info.field_name, positive=False, nonnegative=True)

    @field_validator("turns_ratio")
    @classmethod
    def _ratio(cls, value: Quantity | None) -> Quantity | None:
        return _quantity_in(value, "1", "turns_ratio", positive=True, nonnegative=True)

    @field_validator("core_material_ref", "database_ref")
    @classmethod
    def _refs(cls, value: str | None, info: Any) -> str | None:
        return _uri_reference(value, info.field_name)

    @field_validator("limitations")
    @classmethod
    def _limitations(cls, value: list[str]) -> list[str]:
        if any(not item for item in value):
            raise ValueError("limitations must be non-empty strings")
        return value

    @field_validator("detail")
    @classmethod
    def _detail(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _json_data(value, "detail")

    @model_validator(mode="after")
    def _rules(self) -> "MagneticComponentSummary":
        _unique([item.id for item in self.evidence], "component evidence id")
        quantities = [name for name, _ in _COMPONENT_UNITS if getattr(self, name) is not None]
        if quantities and not self.evidence:
            raise ValueError("Core requires evidence_refs for physical quantities; supply at least one evidence "
                             f"record for {', '.join(quantities)}")
        return self


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------
def _coerce(model: type[BaseModel], value: Any, label: str) -> Any:
    if isinstance(value, model):
        return value
    if not isinstance(value, dict):
        raise InvalidInputError(f"{label} must be a {model.__name__} or a dict, got {type(value).__name__}")
    try:
        return model.model_validate(value)
    except ValidationError as error:
        raise InvalidInputError(f"{label} is invalid: {error}", details={"error_count": error.error_count()}) from None


def _core_quantity(value: Quantity, unit: str) -> dict[str, Any]:
    """Core quantity object in the canonical SI spelling ``unit`` (conversion is dimension-checked)."""
    return {"value": value.to(unit).value, "unit": unit}


def _detail_extension(summary: MagneticComponentSummary) -> dict[str, Any]:
    return {
        "schema": DETAIL_SCHEMA,
        "tool": {"id": TOOL_ID, "name": TOOL_NAME, "version": __version__},
        "quantities": {name: getattr(summary, name).model_dump(mode="json", exclude_none=True)
                       for name, _ in _COMPONENT_UNITS if getattr(summary, name) is not None},
        "limitations": list(summary.limitations),
        "detail": copy.deepcopy(summary.detail),
    }


def _detail_schema_compatible(schema: str) -> bool:
    """Semantic-version compatibility of a ``aipe.magnetics.core-detail/<major>.<minor>`` payload.

    Same major version is required; while the major is ``0`` (pre-1.0, minors may break) the minor
    must match as well. ``DETAIL_SCHEMA`` is the version this module writes.
    """
    ours = DETAIL_SCHEMA[len(_DETAIL_SCHEMA_PREFIX):].split(".")
    theirs = schema[len(_DETAIL_SCHEMA_PREFIX):].split(".")
    if not theirs or not theirs[0].isdigit() or theirs[0] != ours[0]:
        return False
    return ours[0] != "0" or (len(theirs) > 1 and theirs[1] == ours[1])


def _parse_detail_extension(extension: Any, component_id: str) -> dict[str, Any]:
    empty = {"quantities": {}, "limitations": [], "detail": {}}
    if extension is None:
        return empty
    if not isinstance(extension, dict):
        raise DataIntegrityError(f"{component_id}: extensions[{EXTENSION_KEY!r}] must be an object")
    schema = extension.get("schema")
    if not isinstance(schema, str) or not schema:
        raise DataIntegrityError(f"{component_id}: malformed {EXTENSION_KEY} payload: 'schema' is missing or not a string",
                                 details={"keys": sorted(extension)})
    if not schema.startswith(_DETAIL_SCHEMA_PREFIX):
        raise UnsupportedOperationError(f"{component_id}: unsupported {EXTENSION_KEY} payload schema {schema!r} "
                                        "(not written by this adapter)", details={"supported": DETAIL_SCHEMA})
    if not _detail_schema_compatible(schema):
        raise UnsupportedOperationError(f"{component_id}: unsupported {EXTENSION_KEY} payload version {schema!r}; "
                                        f"this adapter reads {DETAIL_SCHEMA} (same major, same minor while 0.x)",
                                        details={"supported": DETAIL_SCHEMA, "found": schema})
    quantities = extension.get("quantities", {})
    limitations = extension.get("limitations", [])
    detail = extension.get("detail", {})
    if not isinstance(quantities, dict) or not isinstance(detail, dict) or not isinstance(limitations, list) \
            or any(not isinstance(item, str) or not item for item in limitations):
        raise DataIntegrityError(f"{component_id}: malformed {EXTENSION_KEY} payload",
                                 details={"keys": sorted(extension)})
    known = {name for name, _ in _COMPONENT_UNITS}
    unknown = sorted(set(quantities) - known)
    if unknown:
        raise DataIntegrityError(f"{component_id}: {EXTENSION_KEY}.quantities has unknown keys {unknown}; "
                                 f"this adapter records {sorted(known)} only", details={"unknown": unknown})
    return {"quantities": quantities, "limitations": list(limitations), "detail": detail}


def _core_quantity_object(core_value: dict[str, Any], label: str) -> Quantity:
    """A Core ``{value, unit}`` as :class:`Quantity`; a contract-valid but unrepresentable number is a data error."""
    try:
        return Quantity(value=core_value["value"], unit=core_value["unit"])
    except (ValidationError, ValueError) as error:
        raise DataIntegrityError(f"{label} is not representable as a finite float quantity: {error}",
                                 details={"path": label, "core": core_value}) from None


def _restore_quantity(core_value: dict[str, Any], recorded: Any, label: str) -> Quantity:
    """The quantity as originally supplied when the extension recorded it consistently; else the Core SI value."""
    base = _core_quantity_object(core_value, label)
    if recorded is None:
        return base
    try:
        quantity = Quantity.model_validate(recorded)
        converted = quantity.to(core_value["unit"])
    except (ValidationError, ValueError) as error:
        raise DataIntegrityError(f"{EXTENSION_KEY}.quantities.{label} is not a valid quantity: {error}",
                                 details={"path": label}) from None
    if not math.isclose(converted.value, base.value, rel_tol=_SAME_VALUE_REL_TOL, abs_tol=0.0):
        raise DataIntegrityError(f"{EXTENSION_KEY}.quantities.{label} disagrees with the Core field",
                                 details={"path": label, "core": core_value, "extension": recorded})
    return quantity


def _same_core_value(existing: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return existing["unit"] == candidate["unit"] and math.isclose(
        float(existing["value"]), float(candidate["value"]), rel_tol=_SAME_VALUE_REL_TOL, abs_tol=0.0)


def _entry_evidence(entry: dict[str, Any], evidence: dict[str, dict[str, Any]]) -> list[CoreEvidence]:
    """Evidence of a magnetics entry: its ``evidence_refs`` in order, then the transitive provenance inputs.

    Closing the set under ``provenance.inputs`` makes the imported summary re-exportable on its own
    (``to_engineering_state`` requires every analytical input to be present). Only the kinds this
    adapter emits may be carried; anything else is refused explicitly.
    """
    order: list[str] = []
    seen: set[str] = set()

    def admit(identifier: str, via: str) -> None:
        record = evidence[identifier]  # validate_engineering_state guarantees that references resolve
        if record["kind"] not in SUPPORTED_EVIDENCE_KINDS:
            raise UnsupportedOperationError(
                f"{entry['id']}: evidence {identifier} (reached via {via}) has kind {record['kind']!r}; this adapter "
                "imports assumption, analytical and source evidence only (it never re-emits simulation or "
                "measurement evidence it did not produce)", details={"evidence": identifier, "via": via})
        seen.add(identifier)
        order.append(identifier)

    for reference in entry["evidence_refs"]:
        if reference not in seen:
            admit(reference, entry["id"])
    pending = list(order)
    while pending:
        identifier = pending.pop(0)
        for dependency in evidence[identifier]["provenance"]["inputs"]:
            if dependency not in seen:
                admit(dependency, identifier)
                pending.append(dependency)
    records: list[CoreEvidence] = []
    for identifier in order:
        try:
            records.append(CoreEvidence.model_validate(evidence[identifier]))
        except ValidationError as error:
            raise DataIntegrityError(f"evidence {identifier} cannot be represented by this adapter: {error}",
                                     details={"path": f"$.evidence[{identifier}]"}) from None
    return records


def _optional_text(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InvalidInputError(f"{label} must be a non-empty string or None, got {value!r}")
    return value


def _new_state(*, state_id: str | None, project_id: str | None, project_name: str | None,
               project_description: str | None, project_purpose: str, converter_topology: str,
               created_at: str) -> dict[str, Any]:
    if project_id is None:
        raise InvalidInputError("project_id is required when no base_state is given")
    try:
        _core_id(project_id, "project_id")
        state_id = state_id or f"{project_id}.state"
        _core_id(state_id, "state_id")
    except ValueError as error:
        raise InvalidInputError(str(error)) from None
    if project_purpose not in ("example", "research", "education", "engineering"):
        raise InvalidInputError(f"project_purpose must be one of example, research, education, engineering; "
                                f"got {project_purpose!r}")
    if not isinstance(converter_topology, str) or not converter_topology:
        raise InvalidInputError("converter_topology must be a non-empty string")
    project_name = _optional_text(project_name, "project_name")
    project_description = _optional_text(project_description, "project_description")
    project: dict[str, Any] = {"id": project_id, "name": project_id if project_name is None else project_name,
                               "purpose": project_purpose, "created_at": created_at, "updated_at": created_at}
    if project_description is not None:
        project["description"] = project_description
    return {
        "id": state_id,
        "schema_version": CORE_SCHEMA_VERSION,
        "project": project,
        "requirements": {"evidence_refs": []},
        "architecture": {"stage": "concept", "evidence_refs": []},
        "converter": {"id": f"{project_id}.converter", "topology": converter_topology, "design_status": "concept",
                      "evidence_refs": []},
        "semiconductors": [],
        "magnetics": [],
        "passives": [],
        "control": {"evidence_refs": []},
        "thermal": {"evidence_refs": []},
        "simulation": {"status": "not_run", "runs": [], "evidence_refs": []},
        "validation": {"status": "not_run", "checks": [], "limitations": [], "evidence_refs": []},
        "evidence": [],
    }


# --------------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------------
def to_engineering_state(summary: MagneticComponentSummary | dict[str, Any], *, created_at: str,
                         project_id: str | None = None,
                         requirements: DesignRequirements | dict[str, Any] | None = None,
                         base_state: dict[str, Any] | None = None,
                         checks: Sequence[CoreValidationCheck | dict[str, Any]] = (),
                         state_id: str | None = None, project_name: str | None = None,
                         project_description: str | None = None, project_purpose: ProjectPurpose = "engineering",
                         converter_topology: str = "unspecified", replace_existing: bool = False,
                         replace_requirements: bool = False) -> dict[str, Any]:
    """Build (or extend a deep copy of) a Core Engineering State 0.1.0 containing ``summary``.

    ``created_at`` (RFC 3339, caller supplied) becomes the project timestamps of a new state or the
    ``updated_at`` of an extended one. ``requirements`` sets the converter-level design inputs and
    appends their evidence. ``checks`` are analytical validation checks; a passing one must reference
    analytical evidence. With ``base_state`` the base must itself be contract-valid; evidence and
    check ids must be new or byte-identical to existing records (idempotent re-export); an existing
    magnetics entry with the same id is replaced only when ``replace_existing`` is true; a
    requirement scalar whose value differs from the base is replaced only when
    ``replace_requirements`` is true, and the replacement is recorded (previous value and evidence
    refs in ``requirements.extensions["aipe.magnetics-detail"].superseded`` plus a
    ``validation.limitations`` entry) instead of being silent. Nothing already in the base is ever
    removed. The returned state is validated against the Core contract before it is returned; an
    invalid combination raises :class:`InvalidInputError` with every Core diagnostic in ``details``.
    """
    summary = _coerce(MagneticComponentSummary, summary, "summary")
    requirements = None if requirements is None else _coerce(DesignRequirements, requirements, "requirements")
    check_models = [_coerce(CoreValidationCheck, item, "check") for item in checks]
    try:
        _rfc3339(created_at, "created_at")
    except ValueError as error:
        raise InvalidInputError(str(error)) from None

    if base_state is None:
        state = _new_state(state_id=state_id, project_id=project_id, project_name=project_name,
                           project_description=project_description, project_purpose=project_purpose,
                           converter_topology=converter_topology, created_at=created_at)
    else:
        if not isinstance(base_state, dict):
            raise InvalidInputError(f"base_state must be a dict, got {type(base_state).__name__}")
        base_errors = validate_engineering_state(base_state)
        if base_errors:
            raise InvalidInputError("base_state is not a valid Core Engineering State 0.1.0",
                                    details={"errors": base_errors})
        state = copy.deepcopy(base_state)
        if project_id is not None and project_id != state["project"]["id"]:
            raise InvalidInputError(f"project_id {project_id!r} does not match base_state project {state['project']['id']!r}")
        if state_id is not None and state_id != state["id"]:
            raise InvalidInputError(f"state_id {state_id!r} does not match base_state id {state['id']!r}")
        state["project"]["updated_at"] = created_at

    # -- evidence (ids unique; identical re-exports are idempotent) ---------------------------------
    existing: dict[str, dict[str, Any]] = {item["id"]: item for item in state["evidence"]}
    added: list[CoreEvidence] = []

    def add_evidence(record: CoreEvidence, origin: str) -> None:
        if record.review_status == "rejected":
            raise InvalidInputError(f"{origin}: evidence {record.id} is rejected and cannot support state fields")
        core = record.to_core()
        if record.id in existing:
            if existing[record.id] == core:
                return
            raise InvalidInputError(f"{origin}: evidence id {record.id} already exists in base_state with different content")
        state["evidence"].append(core)
        existing[record.id] = core
        added.append(record)

    if requirements is not None:
        for record in requirements.evidence:
            add_evidence(record, "requirements")
    for record in summary.evidence:
        add_evidence(record, f"component {summary.id}")
    for record in added:
        for dependency in record.provenance.inputs:
            if dependency not in existing:
                raise InvalidInputError(f"evidence {record.id} lists unknown input evidence {dependency}")

    # -- requirements (conflicts with the base are rejected or recorded, never silent) ---------------
    if requirements is not None:
        target = state["requirements"]
        previous_refs = list(target.get("evidence_refs", []))
        supplied = {name: _core_quantity(getattr(requirements, name), unit)
                    for name, unit in _REQUIREMENT_UNITS if getattr(requirements, name) is not None}
        conflicts = [{"field": name, "base": target[name], "supplied": value}
                     for name, value in supplied.items()
                     if name in target and not _same_core_value(target[name], value)]
        if conflicts and not replace_requirements:
            raise InvalidInputError(
                "base_state already carries different values for requirements "
                + ", ".join(item["field"] for item in conflicts)
                + "; pass replace_requirements=True to replace them (the replacement is recorded in the state)",
                details={"conflicts": conflicts})
        target.update(supplied)
        references = target.setdefault("evidence_refs", [])
        for record in requirements.evidence:
            if record.id not in references:
                references.append(record.id)
        if conflicts:
            superseded = [{"field": item["field"], "previous": item["base"], "replacement": item["supplied"],
                           "previous_evidence_refs": previous_refs, "replaced_at": created_at}
                          for item in conflicts]
            extension = target.setdefault("extensions", {}).setdefault(EXTENSION_KEY, {"schema": DETAIL_SCHEMA})
            if not isinstance(extension, dict) or not isinstance(extension.get("schema"), str) \
                    or not extension["schema"].startswith(_DETAIL_SCHEMA_PREFIX):
                raise DataIntegrityError(f"base_state requirements.extensions[{EXTENSION_KEY!r}] is not a payload "
                                         "written by this adapter", details={"found": extension})
            extension.setdefault("superseded", []).extend(superseded)
            statement = ("Requirements replaced by aipe.magnetics export at " + created_at + ": "
                         + "; ".join(f"{item['field']} {item['previous']['value']:g} {item['previous']['unit']} -> "
                                     f"{item['replacement']['value']:g} {item['replacement']['unit']}"
                                     for item in superseded)
                         + ". Evidence refs recorded before the replacement ("
                         + (", ".join(previous_refs) or "none")
                         + ") documented the previous values; validation checks recorded earlier were evaluated "
                           "against the previous requirements.")
            state["validation"]["limitations"].append(statement)

    # -- magnetics entry ----------------------------------------------------------------------------
    entry: dict[str, Any] = {"id": summary.id, "role": summary.role, "kind": summary.kind,
                             "design_status": summary.design_status}
    if summary.database_ref is not None:
        entry["database_ref"] = summary.database_ref
    for name, unit in _COMPONENT_UNITS:
        quantity = getattr(summary, name)
        if quantity is not None:
            entry[name] = _core_quantity(quantity, unit)
    if summary.core_material_ref is not None:
        entry["core_material_ref"] = summary.core_material_ref
    entry["evidence_refs"] = [record.id for record in summary.evidence]
    entry["extensions"] = {EXTENSION_KEY: _detail_extension(summary)}
    positions = [index for index, item in enumerate(state["magnetics"]) if item["id"] == summary.id]
    if positions:
        if not replace_existing:
            raise InvalidInputError(f"base_state already contains magnetics entry {summary.id}; "
                                    "pass replace_existing=True to replace it")
        state["magnetics"][positions[0]] = entry
    else:
        state["magnetics"].append(entry)

    # -- validation checks and status (identical re-exports are no-ops, like evidence) ---------------
    validation = state["validation"]
    existing_checks = {item["id"]: item for item in validation["checks"]}
    for check in check_models:
        core_check = check.to_core()
        if check.id in existing_checks:
            if existing_checks[check.id] == core_check:
                continue
            raise InvalidInputError(f"validation check id {check.id} already exists in base_state with different content")
        for reference in check.evidence_refs:
            if reference not in existing:
                raise InvalidInputError(f"check {check.id} references unknown evidence {reference}")
        if check.status == "pass" and not any(existing[r]["kind"] == "analytical" for r in check.evidence_refs):
            raise InvalidInputError(f"passing analytical check {check.id} must reference analytical evidence")
        validation["checks"].append(core_check)
        existing_checks[check.id] = core_check
    if any(item["status"] == "fail" for item in validation["checks"]):
        validation["status"] = "failed"
    elif any(item["status"] == "pass" and item["kind"] == "analytical" for item in validation["checks"]):
        if _STATUS_ORDER[validation["status"]] < _STATUS_ORDER["analytically_checked"]:
            validation["status"] = "analytically_checked"
    if DEFAULT_LIMITATION not in validation["limitations"]:
        validation["limitations"].append(DEFAULT_LIMITATION)

    errors = validate_engineering_state(state)
    if errors:
        raise InvalidInputError("exported state violates the Core Engineering State contract",
                                details={"errors": errors})
    return state


def from_engineering_state(state: dict[str, Any]) -> list[MagneticComponentSummary]:
    """Import every ``magnetics[]`` entry of a contract-valid Core state as a summary (state order kept).

    Each summary carries the entry's evidence closed under provenance inputs (see module docstring).
    The ``aipe.magnetics-detail`` extension, when present, must agree with the Core fields: every
    recorded quantity must correspond to a Core field of the entry with the same SI value, otherwise
    :class:`DataIntegrityError` is raised rather than dropping or preferring either side.
    """
    errors = validate_engineering_state(state)
    if errors:
        raise InvalidInputError("state is not a valid Core Engineering State 0.1.0", details={"errors": errors})
    evidence = {item["id"]: item for item in state["evidence"]}
    summaries: list[MagneticComponentSummary] = []
    for entry in state["magnetics"]:
        records = _entry_evidence(entry, evidence)
        detail = _parse_detail_extension((entry.get("extensions") or {}).get(EXTENSION_KEY), entry["id"])
        present = [name for name, _ in _COMPONENT_UNITS if name in entry]
        orphaned = sorted(set(detail["quantities"]) - set(present))
        if orphaned:
            raise DataIntegrityError(f"{entry['id']}: {EXTENSION_KEY}.quantities records {orphaned} but the Core entry "
                                     "carries no such field", details={"orphaned": orphaned})
        quantities = {name: _restore_quantity(entry[name], detail["quantities"].get(name), f"{entry['id']}.{name}")
                      for name in present}
        try:
            summaries.append(MagneticComponentSummary(
                id=entry["id"], role=entry["role"], kind=entry["kind"], design_status=entry["design_status"],
                core_material_ref=entry.get("core_material_ref"), database_ref=entry.get("database_ref"),
                evidence=records, limitations=detail["limitations"], detail=detail["detail"], **quantities))
        except ValidationError as error:
            raise DataIntegrityError(f"{entry['id']}: Core entry cannot be represented as a component summary: {error}",
                                     details={"path": f"$.magnetics[{entry['id']}]"}) from None
    return summaries


def extract_requirements(state: dict[str, Any]) -> dict[str, Quantity]:
    """Design inputs of a contract-valid state as SI quantities; absent Core fields are omitted (unknown)."""
    errors = validate_engineering_state(state)
    if errors:
        raise InvalidInputError("state is not a valid Core Engineering State 0.1.0", details={"errors": errors})
    requirements = state["requirements"]
    return {name: _core_quantity_object(requirements[name], f"$.requirements.{name}")
            for name, _ in _REQUIREMENT_UNITS if name in requirements}


__all__ = [
    "CORE_ID_PATTERN", "DEFAULT_LIMITATION", "DETAIL_SCHEMA", "EXTENSION_KEY", "SUPPORTED_EVIDENCE_KINDS",
    "CoreArtifact", "CoreAttribution", "CoreEvidence", "CoreProvenance", "CoreSource", "CoreValidationCheck",
    "DesignRequirements", "MagneticComponentSummary", "extract_requirements", "from_engineering_state",
    "to_engineering_state", "tool_attribution",
]
