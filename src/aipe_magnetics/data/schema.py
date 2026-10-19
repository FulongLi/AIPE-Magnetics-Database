"""JSON Schema 2020-12 generation, loading and validation for the AIPE magnetic record standard.

The **normative** definition of the v0.2 standard is the set of pydantic models in
:mod:`aipe_magnetics.data.records`. This module derives the published JSON Schema artefacts from
them and validates records against both layers (structural JSON Schema + pydantic semantics).

Artefacts (canonical copies live in ``aipe_magnetics/schemas/v0.2`` inside the package; the
repository directory ``schemas/v0.2`` is a byte-identical mirror):

* ``magnetic-record.schema.json`` - the record envelope. Because the pydantic envelope declares
  ``properties: dict``, the kind-specific property objects are bound with
  ``allOf: [{if: {kind: X}, then: {properties: {properties: {$ref: '#/$defs/<XProperties>'}}}}]``,
  one block per entry of :data:`aipe_magnetics.data.records.PROPERTY_MODELS`.
* ``properties/<kind>.schema.json`` - the standalone property schema of every kind.
* ``quantity.schema.json`` - the ``{value, unit, original?, uncertainty?}`` object.

Conventions:

* ``$schema`` is draft 2020-12. ``$id`` is the raw GitHub URL of the ``v0.2.0`` tag. There are no
  remote ``$ref``s: every definition is inlined under the top-level ``$defs`` and named after the
  pydantic class (stable names such as ``Quantity``, ``Evidence``, ``MaterialProperties``).
* ``schema_version`` is *required* by the JSON Schema although the pydantic model defaults it,
  because it is the dispatch key of :func:`validate_record` and of every external validator.
* Every kind block whose property model has required fields also requires the ``properties``
  key, so a third-party validator cannot accept e.g. a ``core_loss_measurement`` without points.
* Format annotations (:data:`_FORMAT_ANNOTATIONS`): ``Evidence.created_at`` carries
  ``format: date-time`` (RFC 3339) and ``measurement.artifact`` / ``loss_model.model_artifact``
  carry ``format: uri-reference``, exactly as in v0.1; v0.2 applies ``uri-reference`` to the other
  URI-valued fields as well (``Artifact.uri``, ``Evidence.url``, ``dataset.url``,
  ``manufacturer.website``, ``model_metadata.weights_artifact``). Annotations are applied to
  ``type: string`` fields and to the string branch of optional ``anyOf`` fields.
* Optional fields are emitted by pydantic as ``anyOf: [<type>, {type: null}]``; the standard still
  says *omit* missing values - ``null`` is tolerated by the schema only because the normative model
  tolerates it (the JSON importer rejects explicit ``null`` values).
* Export is deterministic: sorted keys, two-space indent, UTF-8, LF line endings. The emitted
  shapes come from the pydantic JSON Schema generator, so the byte-reproducibility of an export is
  tied to the generator version recorded by :func:`generator_versions`.

Validation layers of :func:`validate_record` (dispatch on ``schema_version``):

* ``0.2.0``: ``[schema]`` JSON Schema errors, ``[semantic]`` pydantic (normative model) errors and
  ``[physical]`` plausibility errors (:func:`physical_errors`): every number anywhere in the record
  must be finite (v0.1 parity; the normative model does not check non-Quantity floats such as
  ``RangeSI.min``, ``alpha``/``beta`` or ``fit_metrics``), and the SI quantities listed in
  :data:`_SIGN_RULES` must be positive / non-negative. The sign table restates, on the JSON form and
  with the exact JSON path, the rules the normative model enforces since records.py gained its
  ``_positive`` / ``_nonnegative`` validators (frequencies, absolute temperatures, geometric sizes,
  inductances, permeabilities ... > 0; peaks, rms values, loss densities, air gaps, coercivity ... >= 0)
  and is never stricter than the model where the model has a rule; it additionally covers quantities
  the model leaves unconstrained (conditions inside ``ConditionedValue`` / ``TemperaturePoint``,
  Steinmetz ``k``, B-H loop summaries, excitation ``frequency`` / ``amplitude_peak``, waveform
  ``sampling_rate``). Signed DC quantities (``field_dc``, ``flux_density_dc``, ``current_dc``,
  ``dc_offset``, ``dc_bias_field``, ``field`` conditions) carry no sign rule. Where both layers state
  the same rule both report it; the ``[physical]`` message is the one with the field path.
* ``0.1.0``: the legacy rules of ``scripts/validate.py`` when the repository checkout is present,
  otherwise a replica of those rules (packaged v0.1 schema + evidence, reference and finiteness
  checks). The per-record subset of those rules is exposed as :func:`legacy_record_errors` for the
  migration tool.

Command line::

    python -m aipe_magnetics.data.schema --export schemas/v0.2 --export-package   # refresh both copies
    python -m aipe_magnetics.data.schema --check schemas/v0.2                      # exit 1 when stale
    python -m aipe_magnetics.data.schema --validate a.json b.json [...]            # the files form ONE set

``--validate`` validates the files on the command line as one catalogue (like ``scripts/validate.py``
did for v0.1): per-file structural / semantic / physical errors, plus cross-record checks in which
``*_ref`` fields must resolve inside the set or inside the bundled catalogue (``--no-bundled-catalogue``
drops the latter). A record that references another file therefore needs that file on the command
line; messages are attributed to the file they concern.
"""

from __future__ import annotations

import argparse
import copy
import functools
import importlib.metadata
import importlib.util
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import pydantic
from jsonschema import Draft202012Validator, FormatChecker
from pydantic import BaseModel, ValidationError
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue, models_json_schema

from aipe_magnetics import __version__
from aipe_magnetics.common.errors import DataIntegrityError, DataNotFoundError, InvalidInputError
from aipe_magnetics.common.jsonio import dumps, load
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.data import records as _records
from aipe_magnetics.data.records import PROPERTY_MODELS, SCHEMA_VERSION, MagneticRecord

DIALECT = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_BASE_URL = "https://raw.githubusercontent.com/FulongLi/AIPE-Magnetics-Database/v0.2.0/schemas/v0.2/"
SCHEMA_ID_V02 = SCHEMA_BASE_URL + "magnetic-record.schema.json"
SCHEMA_TITLE_V02 = "AIPE magnetic record v0.2"
LEGACY_SCHEMA_VERSION = "0.1.0"
SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION)
RECORD_FILENAME = "magnetic-record.schema.json"
QUANTITY_FILENAME = "quantity.schema.json"
PROPERTIES_DIRNAME = "properties"

#: (definition name, property name) -> JSON Schema ``format`` annotation. ``date-time`` and the two
#: ``uri-reference`` entries of ``measurement.artifact`` / ``loss_model.model_artifact`` are v0.1 parity;
#: the remaining ``uri-reference`` entries cover the other URI-valued fields of the v0.2 standard.
#: Only formats whose checkers :func:`_format_checker` requires are used, so validation never depends
#: on an optional format plugin.
_FORMAT_ANNOTATIONS: dict[tuple[str, str], str] = {
    ("Evidence", "created_at"): "date-time",
    ("Evidence", "url"): "uri-reference",
    ("Artifact", "uri"): "uri-reference",
    ("MeasurementProperties", "artifact"): "uri-reference",
    ("LossModelProperties", "model_artifact"): "uri-reference",
    ("DatasetProperties", "url"): "uri-reference",
    ("ManufacturerProperties", "website"): "uri-reference",
    ("ModelMetadataProperties", "weights_artifact"): "uri-reference",
}


def generator_versions() -> dict[str, str]:
    """Versions of the libraries whose output shapes the exported schema files (recorded in evidence)."""
    return {"pydantic": pydantic.VERSION, "jsonschema": importlib.metadata.version("jsonschema")}

_RECORD_DESCRIPTION = (
    "A record of the AIPE Magnetic Data Standard v0.2: a common envelope (identity, evidence, licence, "
    "provenance, quality, artifacts with checksums, namespaced extensions) plus a kind-specific 'properties' "
    "object bound by the allOf/if/then blocks below. All physical quantities are SI {value, unit} objects; a "
    "converted value retains its original {value, unit}. Missing values are omitted, never zero-filled. "
    "Large numerical arrays live in HDF5 artifacts described by waveform_dataset records. Generated from the "
    "normative pydantic models in aipe_magnetics.data.records - do not edit by hand."
)


# --------------------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------------------
def _annotate_formats(definitions: dict[str, Any]) -> None:
    """Apply :data:`_FORMAT_ANNOTATIONS` to plain ``type: string`` fields and to the string branch of optional fields."""
    for (definition, prop), fmt in _FORMAT_ANNOTATIONS.items():
        target = definitions.get(definition)
        if not isinstance(target, dict):
            continue
        field = target.get("properties", {}).get(prop)
        if not isinstance(field, dict):
            continue
        if field.get("type") == "string":
            field["format"] = fmt
        for branch in field.get("anyOf", []):
            if isinstance(branch, dict) and branch.get("type") == "string":
                branch["format"] = fmt


class V02SchemaGenerator(GenerateJsonSchema):
    """pydantic schema generator for the v0.2 standard (draft 2020-12 dialect, format annotations)."""

    schema_dialect = DIALECT

    def generate(self, schema: Any, mode: Any = "validation") -> JsonSchemaValue:  # type: ignore[override]
        json_schema = super().generate(schema, mode=mode)
        _annotate_formats(json_schema.get("$defs", {}))
        title = json_schema.get("title")
        if isinstance(title, str):
            _annotate_formats({title: json_schema})
        return json_schema

    def generate_definitions(self, inputs: Any) -> Any:  # type: ignore[override]
        key_map, definitions = super().generate_definitions(inputs)
        _annotate_formats(definitions)
        return key_map, definitions


def _unique_property_models() -> list[type[BaseModel]]:
    seen: list[type[BaseModel]] = []
    for model in PROPERTY_MODELS.values():
        if model not in seen:
            seen.append(model)
    return seen


def generate_schema_v02() -> dict[str, Any]:
    """Build the v0.2 ``magnetic-record.schema.json`` document from the normative pydantic models."""
    models: list[tuple[type[BaseModel], Any]] = [(MagneticRecord, "validation")]
    models += [(model, "validation") for model in _unique_property_models()]
    key_map, shared = models_json_schema(models, ref_template="#/$defs/{model}", schema_generator=V02SchemaGenerator)
    definitions: dict[str, Any] = dict(shared.get("$defs", {}))
    record_name = key_map[(MagneticRecord, "validation")]["$ref"].rsplit("/", 1)[-1]
    envelope = dict(definitions.pop(record_name))
    envelope.pop("title", None)
    envelope.pop("description", None)
    required = list(envelope.pop("required", []))
    if "schema_version" not in required:
        required.insert(0, "schema_version")
    all_of: list[dict[str, Any]] = []
    for kind, model in PROPERTY_MODELS.items():  # insertion order of the normative table
        then: dict[str, Any] = {"properties": {"properties": dict(key_map[(model, "validation")])}}
        if any(field.is_required() for field in model.model_fields.values()):
            then["required"] = ["properties"]  # the kind cannot be expressed without its mandatory fields
        all_of.append({"if": {"properties": {"kind": {"const": kind}}, "required": ["kind"]}, "then": then})
    schema: dict[str, Any] = {
        "$schema": DIALECT,
        "$id": SCHEMA_ID_V02,
        "title": SCHEMA_TITLE_V02,
        "description": _RECORD_DESCRIPTION,
    }
    schema.update(envelope)
    schema["required"] = required
    schema["allOf"] = all_of
    schema["$defs"] = definitions
    return schema


def generate_property_schema_v02(kind: str) -> dict[str, Any]:
    """Standalone JSON Schema of the ``properties`` object of one record kind."""
    try:
        model = PROPERTY_MODELS[kind]
    except KeyError:
        raise InvalidInputError(f"unknown record kind {kind!r}; known kinds: {', '.join(PROPERTY_MODELS)}",
                                details={"kind": kind}) from None
    generated = model.model_json_schema(schema_generator=V02SchemaGenerator, ref_template="#/$defs/{model}",
                                        mode="validation")
    description = generated.pop("description", "")
    generated.pop("title", None)
    schema: dict[str, Any] = {
        "$schema": DIALECT,
        "$id": f"{SCHEMA_BASE_URL}{PROPERTIES_DIRNAME}/{kind}.schema.json",
        "title": f"{SCHEMA_TITLE_V02} - {kind} properties",
        "description": (f"The 'properties' object of a '{kind}' record ({model.__name__}). " + description).strip(),
    }
    schema.update(generated)
    return schema


def generate_quantity_schema_v02() -> dict[str, Any]:
    generated = Quantity.model_json_schema(schema_generator=V02SchemaGenerator, ref_template="#/$defs/{model}",
                                           mode="validation")
    generated.pop("title", None)
    generated.pop("description", None)
    schema: dict[str, Any] = {
        "$schema": DIALECT,
        "$id": f"{SCHEMA_BASE_URL}{QUANTITY_FILENAME}",
        "title": f"{SCHEMA_TITLE_V02} - quantity",
        "description": ("A physical quantity: SI value and canonical SI unit symbol as registered in "
                        "aipe_magnetics.common.units, the original {value, unit} when converted, and an optional, "
                        "never fabricated uncertainty statement (kind 'unknown' when not quantified)."),
    }
    schema.update(generated)
    return schema


def schema_files_v02() -> dict[str, dict[str, Any]]:
    """All v0.2 schema documents keyed by their path relative to the schema directory (POSIX form)."""
    files: dict[str, dict[str, Any]] = {
        RECORD_FILENAME: generate_schema_v02(),
        QUANTITY_FILENAME: generate_quantity_schema_v02(),
    }
    for kind in PROPERTY_MODELS:
        files[f"{PROPERTIES_DIRNAME}/{kind}.schema.json"] = generate_property_schema_v02(kind)
    return files


def render_schema_files_v02() -> dict[str, bytes]:
    """Deterministic bytes (sorted keys, LF, UTF-8) of every v0.2 schema file."""
    return {relative: dumps(schema).encode("utf-8") for relative, schema in schema_files_v02().items()}


def export_schemas(directory: str | Path) -> list[Path]:
    """Write a fresh, deterministic export of all v0.2 schema files into ``directory`` (overwrites)."""
    directory = Path(directory)
    written: list[Path] = []
    for relative, content in sorted(render_schema_files_v02().items()):
        path = directory / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        written.append(path)
    return written


def check_exported_schemas(directory: str | Path) -> list[str]:
    """Differences between the files in ``directory`` and a fresh export (empty list = in sync)."""
    directory = Path(directory)
    if not directory.is_dir():
        return [f"{directory}: directory does not exist"]
    fresh = render_schema_files_v02()
    problems: list[str] = []
    for relative, content in sorted(fresh.items()):
        path = directory / Path(relative)
        if not path.is_file():
            problems.append(f"{relative}: missing")
            continue
        on_disk = path.read_bytes()
        if on_disk == content:
            continue
        if on_disk.replace(b"\r\n", b"\n") == content:
            problems.append(f"{relative}: differs only by line endings (CRLF on disk; the export uses LF)")
        else:
            problems.append(f"{relative}: content differs from a fresh export")
    for path in sorted(directory.rglob("*.schema.json")):
        relative = path.relative_to(directory).as_posix()
        if relative not in fresh:
            problems.append(f"{relative}: unexpected file (not produced by the exporter)")
    return problems


def compare_directories(canonical: str | Path, mirror: str | Path, *, pattern: str = "*") -> list[str]:
    """Byte-compare two directory trees (files matching ``pattern``); returns human-readable differences."""
    canonical, mirror = Path(canonical), Path(mirror)
    problems: list[str] = []
    if not canonical.is_dir():
        return [f"{canonical}: directory does not exist"]
    if not mirror.is_dir():
        return [f"{mirror}: directory does not exist"]

    def listing(root: Path) -> dict[str, Path]:
        return {p.relative_to(root).as_posix(): p for p in sorted(root.rglob(pattern))
                if p.is_file() and "__pycache__" not in p.parts}

    left, right = listing(canonical), listing(mirror)
    for relative in sorted(set(left) | set(right)):
        if relative not in right:
            problems.append(f"{relative}: present in {canonical} but missing from {mirror}")
        elif relative not in left:
            problems.append(f"{relative}: present in {mirror} but missing from {canonical}")
        elif left[relative].read_bytes() != right[relative].read_bytes():
            problems.append(f"{relative}: bytes differ between {canonical} and {mirror}")
    return problems


# --------------------------------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------------------------------
def package_schema_dir(version: str = SCHEMA_VERSION) -> Path:
    """Directory of the packaged schema files for ``version`` (``aipe_magnetics/schemas/v0.X``)."""
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise InvalidInputError(f"unsupported schema version {version!r}; supported: {SUPPORTED_SCHEMA_VERSIONS}",
                                details={"version": version})
    major_minor = ".".join(version.split(".")[:2])
    return Path(__file__).resolve().parent.parent / "schemas" / f"v{major_minor}"


def repository_root() -> Path | None:
    """Root of a source checkout (``pyproject.toml`` + legacy schema present), else ``None`` when installed."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "schemas" / RECORD_FILENAME).is_file() \
                and (candidate / "src" / "aipe_magnetics").is_dir():
            return candidate
    return None


def mirror_schema_dir() -> Path | None:
    """``schemas/v0.2`` of the repository checkout, or ``None`` when running from an installed package."""
    root = repository_root()
    return None if root is None else root / "schemas" / "v0.2"


@functools.lru_cache(maxsize=None)
def _load_schema_cached(version: str) -> dict[str, Any]:
    path = package_schema_dir(version) / RECORD_FILENAME
    if not path.is_file():
        raise DataNotFoundError(f"schema file missing: {path}; refresh it with "
                                f"'python -m aipe_magnetics.data.schema --export-package'", details={"path": str(path)})
    schema = load(path)
    Draft202012Validator.check_schema(schema)
    return schema


def load_schema(version: str = SCHEMA_VERSION) -> dict[str, Any]:
    """Load the packaged JSON Schema for ``version`` (``'0.1.0'`` legacy copy or ``'0.2.0'``)."""
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise InvalidInputError(f"unsupported schema version {version!r}; supported: {SUPPORTED_SCHEMA_VERSIONS}",
                                details={"version": version})
    return copy.deepcopy(_load_schema_cached(version))


def _format_checker() -> FormatChecker:
    checker = FormatChecker()
    missing = {"date-time", "uri-reference"} - set(checker.checkers)
    if missing:
        raise DataIntegrityError(f"jsonschema format checkers unavailable: {sorted(missing)}; install 'jsonschema[format]'",
                                 details={"missing": sorted(missing)})
    return checker


@functools.lru_cache(maxsize=None)
def _validator(version: str) -> Draft202012Validator:
    return Draft202012Validator(_load_schema_cached(version), format_checker=_format_checker())


# --------------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------------
def _json_path(loc: Sequence[Any]) -> str:
    return "$" + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in loc)


def schema_errors(data: Any, version: str = SCHEMA_VERSION) -> list[str]:
    """Structural JSON Schema errors of ``data`` against the packaged schema ``version`` (sorted, deterministic)."""
    validator = _validator(version)
    found = sorted(validator.iter_errors(data), key=lambda e: (e.json_path, e.message))
    return [f"[schema] {error.json_path}: {error.message}" for error in found]


def semantic_errors(data: Any) -> list[str]:
    """pydantic (normative model) errors of a v0.2 record dict."""
    try:
        MagneticRecord.model_validate(data)
    except ValidationError as error:
        items = sorted((_json_path(e["loc"]), e["msg"]) for e in error.errors(include_url=False))
        return [f"[semantic] {path}: {message}" for path, message in items]
    return []


# -- physical plausibility ---------------------------------------------------------------------------
#: property key -> sign rule for every SI quantity stored under that key anywhere inside ``properties``
#: (a bare ``{value, unit}`` object, the ``value`` of a ConditionedValue / TemperaturePoint, or every
#: item of a list of those). ``positive`` means ``value > 0``, ``nonnegative`` means ``value >= 0``.
#: The table is aligned with the normative model (records.py ``_positive`` / ``_nonnegative``): peaks and
#: rms values are ``>= 0`` there (a zero-excitation point is a valid statement), so they are here too.
#: Keys absent from the table (signed DC quantities such as ``field_dc``, ``flux_density_dc``,
#: ``current_dc``, ``dc_offset``, ``dc_bias_field``, the ``field`` condition; free-form ``dc_bias_points``)
#: carry no sign rule.
_SIGN_RULES: dict[str, str] = {
    # excitation and conditions
    "frequency": "positive", "sampling_rate": "positive", "temperature": "positive", "ambient_temperature": "positive",
    "flux_density_peak": "nonnegative", "field_peak": "nonnegative", "current_rms": "nonnegative", "voltage_rms": "nonnegative",
    "amplitude_peak": "nonnegative", "loss_density": "nonnegative", "loop_energy_density": "nonnegative",
    # material parameters (records.py: permeabilities, Bsat, Curie temperature, density, resistivity > 0; coercivity >= 0)
    "relative_permeability": "positive", "initial_permeability": "positive", "amplitude_permeability": "positive",
    "permeability_vs_temperature": "positive", "saturation_flux_density": "positive", "remanence": "nonnegative",
    "coercivity": "nonnegative", "curie_temperature": "positive", "density": "positive", "resistivity": "positive",
    "k": "positive",
    # core geometry (records.py: sizes > 0, air gaps >= 0)
    "effective_area": "positive", "effective_length": "positive", "effective_volume": "positive", "minimum_area": "positive",
    "core_constant_c1": "positive", "window_area": "positive", "bobbin_window_area": "positive", "mean_turn_length": "positive",
    "mass": "positive", "surface_area": "positive", "air_gap_options": "nonnegative",
    # windings (records.py: conductor sizes > 0, insulation >= 0)
    "turns": "positive", "conductor_area": "positive", "strand_diameter": "positive", "conductor_diameter": "positive",
    "foil_thickness": "positive", "foil_width": "positive", "insulation_thickness": "nonnegative",
    # components (records.py: inductances, turns ratio, ratings > 0; air gap >= 0)
    "inductance": "positive", "leakage_inductance": "positive", "magnetizing_inductance": "positive", "turns_ratio": "positive",
    "air_gap": "nonnegative", "rated_current_rms": "positive", "rated_voltage_rms": "positive", "rated_power": "positive",
}
_RULE_TEXT = {"positive": "must be positive", "nonnegative": "must be non-negative"}


def _quantity_value(node: Any) -> tuple[float, str] | None:
    """``(value, unit)`` of a ``{value, unit, ...}`` object, else ``None``."""
    if isinstance(node, dict) and isinstance(node.get("unit"), str):
        value = node.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), node["unit"]
    return None


def _apply_sign_rule(rule: str, node: Any, path: str, errors: list[str]) -> None:
    if isinstance(node, list):
        for index, item in enumerate(node):
            _apply_sign_rule(rule, item, f"{path}[{index}]", errors)
        return
    if not isinstance(node, dict):
        return
    found = _quantity_value(node)
    target = path
    if found is None:  # ConditionedValue / TemperaturePoint: the quantity sits under "value"
        found = _quantity_value(node.get("value"))
        target = f"{path}.value"
    if found is None:
        return  # not a quantity (structural problems are reported by the schema layer)
    value, unit = found
    if not math.isfinite(value):
        return  # reported by the finiteness walk
    if (rule == "positive" and value <= 0) or (rule == "nonnegative" and value < 0):
        errors.append(f"[physical] {target}.value: {_RULE_TEXT[rule]}, got {value!r} {unit}")


def _walk_sign_rules(node: Any, path: str, errors: list[str]) -> None:
    if isinstance(node, dict):
        for key, child in node.items():
            child_path = f"{path}.{key}"
            rule = _SIGN_RULES.get(key)
            if rule is not None:
                _apply_sign_rule(rule, child, child_path, errors)
            if key == "dimensions" and isinstance(child, dict) and isinstance(child.get("values"), dict):
                for name, quantity in child["values"].items():
                    _apply_sign_rule("positive", quantity, f"{child_path}.values.{name}", errors)
            _walk_sign_rules(child, child_path, errors)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _walk_sign_rules(item, f"{path}[{index}]", errors)


def _walk_finite(node: Any, path: str, errors: list[str]) -> None:
    if isinstance(node, bool):
        return
    if isinstance(node, float) and not math.isfinite(node):
        errors.append(f"[physical] {path}: value must be finite, got {node!r}")
    elif isinstance(node, dict):
        for key, child in node.items():
            _walk_finite(child, f"{path}.{key}", errors)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _walk_finite(item, f"{path}[{index}]", errors)


def physical_errors(data: Any) -> list[str]:
    """Plausibility errors (``[physical]``) of a v0.2 record dict: finiteness everywhere, :data:`_SIGN_RULES` in ``properties``.

    The rules are stated on the JSON form of the record, independently of the pydantic models, and
    complement (never replace) the schema and semantic layers.
    """
    if isinstance(data, MagneticRecord):
        data = data.to_json_dict()
    if not isinstance(data, dict):
        return []
    errors: list[str] = []
    _walk_finite(data, "$", errors)
    if isinstance(data.get("properties"), dict):
        _walk_sign_rules(data["properties"], "$.properties", errors)
    return sorted(set(errors))


def validate_record(data: Any) -> list[str]:
    """Validate one record dict, dispatching on ``schema_version``; returns error strings (empty = valid).

    v0.2 records receive the ``[schema]``, ``[semantic]`` and ``[physical]`` layers (see module docstring).
    """
    if isinstance(data, MagneticRecord):
        data = data.to_json_dict()
    if not isinstance(data, dict):
        return [f"[schema] $: record must be a JSON object, got {type(data).__name__}"]
    version = data.get("schema_version")
    if version is None:
        return [f"[schema] $.schema_version: missing; cannot select a schema (supported: {', '.join(SUPPORTED_SCHEMA_VERSIONS)})"]
    if version == SCHEMA_VERSION:
        errors = schema_errors(data, SCHEMA_VERSION)
        errors.extend(item for item in semantic_errors(data) if item not in errors)
        errors.extend(item for item in physical_errors(data) if item not in errors)
        return errors
    if version == LEGACY_SCHEMA_VERSION:
        return validate_legacy_records([data])
    return [f"[schema] $.schema_version: unsupported value {version!r} (supported: {', '.join(SUPPORTED_SCHEMA_VERSIONS)})"]


def validate_catalogue(records: Iterable[dict[str, Any] | MagneticRecord]) -> list[str]:
    """Per-record validation (version dispatch) plus cross-record checks of :func:`records.validate_catalogue`.

    v0.2 dicts and :class:`MagneticRecord` instances share one cross-reference check; v0.1 dicts in the
    same catalogue are validated as a group with the legacy rules (which include their own reference
    checks). Duplicate ids are reported across both groups.
    """
    errors: list[str] = []
    parsed: list[MagneticRecord] = []
    legacy: list[dict[str, Any]] = []
    for index, item in enumerate(records):
        if isinstance(item, MagneticRecord):
            parsed.append(item)
            continue
        if not isinstance(item, dict):
            errors.append(f"record[{index}]: not a JSON object ({type(item).__name__})")
            continue
        if item.get("schema_version") == LEGACY_SCHEMA_VERSION:
            legacy.append(item)
            continue
        item_errors = validate_record(item)
        if item_errors:
            errors.extend(f"record[{index}] {message}" for message in item_errors)
            continue
        parsed.append(MagneticRecord.model_validate(item))
    if legacy:
        errors.extend(f"legacy {message}" for message in validate_legacy_records(legacy))
    errors.extend(_records.validate_catalogue(parsed))
    v02_ids = {record.id for record in parsed}
    for item in legacy:
        identifier = item.get("id")
        if isinstance(identifier, str) and identifier in v02_ids:
            errors.append(f"Duplicate record ID across schema versions: {identifier}")
    return errors


# --------------------------------------------------------------------------------------------------
# Legacy (v0.1) rules
# --------------------------------------------------------------------------------------------------
_LEGACY_REFERENCE_KINDS: dict[str, set[str]] = {
    "material_ref": {"magnetic_material", "core_material"},
    "family_ref": {"core_family"},
    "core_ref": {"core_geometry"},
    "winding_refs": {"winding"},
    "component_ref": {"inductor", "transformer", "magnetic_component"},
    "operating_condition_ref": {"operating_condition"},
    "operating_condition_refs": {"operating_condition"},
}


@functools.lru_cache(maxsize=1)
def _legacy_validate_function() -> Any:
    """``scripts.validate.validate_records`` of the repository checkout when present, else ``None``.

    The module is loaded from its known path inside the detected repository root (never from an
    arbitrary ``scripts`` package found on ``sys.path``).
    """
    root = repository_root()
    if root is None:
        return None
    path = root / "scripts" / "validate.py"
    if not path.is_file():
        return None
    existing = sys.modules.get("scripts.validate")
    if existing is not None and getattr(existing, "__file__", None) and Path(existing.__file__).resolve() == path.resolve():
        return getattr(existing, "validate_records", None)
    spec = importlib.util.spec_from_file_location("aipe_magnetics_legacy_scripts_validate", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001 - fall back to the replica, never crash validation
        return None
    return getattr(module, "validate_records", None)


def _legacy_evidence_errors(record: dict[str, Any]) -> list[str]:
    """Evidence rules of ``scripts/validate.py`` for one schema-valid v0.1 record (same wording)."""
    errors: list[str] = []
    evidence = {item["id"]: item for item in record["evidence"]}
    if len(evidence) != len(record["evidence"]):
        errors.append(f"{record['id']}: duplicate evidence ID")
    for reference in record["evidence_refs"]:
        if reference not in evidence:
            errors.append(f"{record['id']}: unknown evidence {reference}")
        elif evidence[reference]["review_status"] == "rejected":
            errors.append(f"{record['id']}: rejected evidence {reference}")
    selected = [evidence[ref] for ref in record["evidence_refs"] if ref in evidence]
    if not record["synthetic"] and all(item["kind"] == "assumption" for item in selected):
        errors.append(f"{record['id']}: real records require non-assumption source evidence")
    if record["kind"] == "measurement" and not any(item["kind"] == "measurement" for item in selected):
        errors.append(f"{record['id']}: measurement requires measurement evidence")
    return errors


def _legacy_finite_errors(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def finite(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f"{record['id']}: quantities must be finite")
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)

    finite(record)
    return errors


def _legacy_reference_errors(record: dict[str, Any], identifiers: set[str], by_id: dict[str, dict[str, Any]]) -> list[str]:
    """Cross-record reference rules of ``scripts/validate.py`` for one record (unknown target, unknown field, wrong kind)."""
    errors: list[str] = []
    for key, value in record["properties"].items():
        references = [value] if key.endswith("_ref") else value if key.endswith("_refs") else []
        for reference in references:
            if reference not in identifiers:
                errors.append(f"{record['id']}: unknown record reference {reference}")
            elif key not in _LEGACY_REFERENCE_KINDS:
                errors.append(f"{record['id']}: unknown reference field {key}")
            elif by_id[reference]["kind"] not in _LEGACY_REFERENCE_KINDS[key]:
                errors.append(f"{record['id']}: {key} points to wrong record kind")
    return errors


def legacy_record_errors(record: dict[str, Any]) -> list[str]:
    """Per-record v0.1 rules (packaged v0.1 schema, evidence rules, finiteness) **without** cross-record references.

    This is what :func:`aipe_magnetics.data.migrate.migrate_record` applies before migrating, so a
    record the v0.1 validator rejects on its own (e.g. a ``measurement`` without measurement
    evidence) is never silently "healed" by the more permissive v0.2 rules.
    """
    if not isinstance(record, dict):
        return [f"record: not a JSON object ({type(record).__name__})"]
    structural = [f"record{error.json_path}: {error.message}"
                  for error in sorted(_validator(LEGACY_SCHEMA_VERSION).iter_errors(record), key=lambda e: (e.json_path, e.message))]
    if structural:
        return structural
    return _legacy_evidence_errors(record) + _legacy_finite_errors(record)


def _legacy_replica(records: list[dict[str, Any]]) -> list[str]:
    """Replica of ``scripts/validate.py`` (same rules, same message wording, same order) against the packaged v0.1 schema."""
    validator = _validator(LEGACY_SCHEMA_VERSION)
    errors: list[str] = []
    for index, record in enumerate(records):
        errors.extend(f"record[{index}]{error.json_path}: {error.message}"
                      for error in sorted(validator.iter_errors(record), key=lambda e: (e.json_path, e.message)))
    if errors:
        return errors
    identifiers: set[str] = set()
    by_id = {record["id"]: record for record in records}
    for record in records:
        if record["id"] in identifiers:
            errors.append(f"Duplicate record ID: {record['id']}")
        identifiers.add(record["id"])
    for record in records:
        errors.extend(_legacy_evidence_errors(record))
        errors.extend(_legacy_reference_errors(record, identifiers, by_id))
        errors.extend(_legacy_finite_errors(record))
    return errors


def validate_legacy_records(records: list[dict[str, Any]]) -> list[str]:
    """Validate v0.1 records as a group with the legacy rules (repository script if available, else replica)."""
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            return [f"record[{index}]: not a JSON object ({type(record).__name__})"]
    function = _legacy_validate_function()
    if function is not None:
        return [str(message) for message in function(records)]
    return _legacy_replica(records)


def legacy_validator_source() -> str:
    """``'scripts/validate.py'`` when the repository script is used for v0.1 validation, else ``'replica'``."""
    return "scripts/validate.py" if _legacy_validate_function() is not None else "replica"


# --------------------------------------------------------------------------------------------------
# File-set validation (command line and CLI consumers)
# --------------------------------------------------------------------------------------------------
_LEGACY_INDEX_RE = re.compile(r"^record\[(\d+)\]")
_DUPLICATE_ID_RE = re.compile(r"^Duplicate record ID(?: across schema versions)?: (.+)$")


def _bundled_context(exclude: set[Path]) -> list[MagneticRecord]:
    """Bundled catalogue records whose files are not among ``exclude`` (so a catalogue file can validate itself)."""
    from aipe_magnetics.data.catalog import Catalog  # local import: catalog imports this module

    catalog = Catalog.default()
    context: list[MagneticRecord] = []
    for record in catalog.records():
        source = catalog.source_of(record.id)
        if source is None or source.resolve() not in exclude:
            context.append(record)
    return context


def validate_files(paths: Sequence[str | Path], *, bundled_context: bool = True) -> tuple[dict[str, list[str]], list[str]]:
    """Validate record files **as one set** and attribute every message to the file it concerns.

    Returns ``(per_file_errors, set_errors)``: ``per_file_errors`` maps the given path (as a string) to
    its sorted messages (empty list = that file is fine); ``set_errors`` holds the messages that cannot
    be attributed to a single file. Per-file messages are the :func:`validate_record` layers (v0.2) or
    the legacy per-record rules (v0.1); cross-record messages come from the group check of
    :func:`validate_catalogue` over the files plus, when ``bundled_context`` is true, the bundled
    catalogue records that are not themselves among the files.
    """
    given = [Path(p) for p in paths]
    per_file: dict[str, list[str]] = {str(p): [] for p in given}
    set_errors: list[str] = []
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in given:
        try:
            data = load(path)
        except (OSError, ValueError) as error:
            per_file[str(path)].append(f"cannot load: {error}")
            continue
        if not isinstance(data, dict):
            per_file[str(path)].append(f"[schema] $: record must be a JSON object, got {type(data).__name__}")
            continue
        if data.get("schema_version") == LEGACY_SCHEMA_VERSION:
            per_file[str(path)].extend(legacy_record_errors(data))
        else:
            per_file[str(path)].extend(validate_record(data))
        loaded.append((path, data))
    files_by_id: dict[str, list[str]] = {}
    for path, data in loaded:
        identifier = data.get("id")
        if isinstance(identifier, str):
            files_by_id.setdefault(identifier, []).append(str(path))
    legacy_paths = [str(path) for path, data in loaded if data.get("schema_version") == LEGACY_SCHEMA_VERSION]
    group: list[dict[str, Any] | MagneticRecord] = [data for _path, data in loaded]
    if bundled_context:
        group.extend(_bundled_context({p.resolve() for p in given}))
    for message in validate_catalogue(group):
        text = message
        if text.startswith("legacy "):
            text = text[len("legacy "):]
            match = _LEGACY_INDEX_RE.match(text)
            if match and int(match.group(1)) < len(legacy_paths):  # same wording as legacy_record_errors
                per_file[legacy_paths[int(match.group(1))]].append("record" + text[match.end():])
                continue
        duplicate = _DUPLICATE_ID_RE.match(text)
        if duplicate and duplicate.group(1) in files_by_id:
            for file in files_by_id[duplicate.group(1)]:
                per_file[file].append(text)
            continue
        owner = text.split(": ", 1)[0]
        if owner in files_by_id:
            for file in files_by_id[owner]:
                per_file[file].append(text)
            continue
        if text.startswith("record["):  # per-record message of a v0.2 dict in the group (same order as ``loaded``)
            index_match = _LEGACY_INDEX_RE.match(text)
            if index_match and int(index_match.group(1)) < len(loaded):
                per_file[str(loaded[int(index_match.group(1))][0])].append(text.split("] ", 1)[-1])
                continue
        set_errors.append(text)
    return {file: sorted(set(messages)) for file, messages in per_file.items()}, sorted(set(set_errors))


def check_exported(directory: str | Path) -> int:
    """CLI-style wrapper of :func:`check_exported_schemas`: prints the differences, returns 0 (in sync) or 1 (stale)."""
    problems = check_exported_schemas(directory)
    if problems:
        print(f"STALE {directory}:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"OK {directory} matches a fresh export ({len(render_schema_files_v02())} files)")
    return 0


# --------------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aipe_magnetics.data.schema",
        description="Export, check and apply the AIPE magnetic record JSON Schemas (generated from the pydantic models).")
    parser.add_argument("--export", metavar="DIR", type=Path,
                        help="write a fresh export of all v0.2 schema files into DIR (existing files are overwritten)")
    parser.add_argument("--export-package", action="store_true",
                        help=f"refresh the canonical copies inside the package ({package_schema_dir()})")
    parser.add_argument("--check", metavar="DIR", type=Path, action="append", default=[],
                        help="compare the files in DIR with a fresh export; exit 1 when they differ (repeatable)")
    parser.add_argument("--validate", metavar="FILE", type=Path, nargs="+",
                        help="validate record files as one set (schema_version dispatch per file, cross-references "
                             "within the set plus the bundled catalogue); exit 1 on errors")
    parser.add_argument("--no-bundled-catalogue", action="store_true",
                        help="with --validate: references must resolve within the given files only")
    args = parser.parse_args(argv)
    if args.export is None and not args.export_package and not args.check and not args.validate:
        parser.print_help()
        return 2
    status = 0
    if args.export is not None:
        written = export_schemas(args.export)
        print(f"exported {len(written)} schema files to {args.export}")
    if args.export_package:
        written = export_schemas(package_schema_dir())
        print(f"exported {len(written)} schema files to {package_schema_dir()}")
    for directory in args.check:
        status |= check_exported(directory)
    if args.validate:
        per_file, set_errors = validate_files(args.validate, bundled_context=not args.no_bundled_catalogue)
        for path in args.validate:
            errors = per_file[str(path)]
            if errors:
                status = 1
                print(f"FAIL {path}")
                for message in errors:
                    print(f"  {message}")
            else:
                print(f"PASS {path}")
        if set_errors:
            status = 1
            print(f"FAIL set of {len(args.validate)} file(s)")
            for message in set_errors:
                print(f"  {message}")
    return status


if __name__ == "__main__":  # pragma: no cover - exercised through tests calling main()
    raise SystemExit(main())
