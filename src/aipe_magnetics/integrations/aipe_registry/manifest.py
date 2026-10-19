"""AIPE-Registry capability manifest (``aipe.yaml``) loading and validation, offline.

The normative contract is the vendored ``schema/capability-manifest.schema.json`` (AIPE-Registry
commit :data:`REGISTRY_COMMIT`, MIT; see ``schema/NOTICE.md``). The loader and the semantic
checks re-implement the essential behaviour of upstream ``scripts/registry.py`` (functions named
in the comments) so that a manifest accepted here is accepted by Registry's own validator:

* YAML loader (``UniqueLoader``): duplicate mapping keys and non-string keys are errors; only
  JSON-compatible scalars are accepted (quote dates); YAML 1.2 booleans only (``true``/``false``
  spellings; ``yes``/``no``/``on``/``off`` stay strings); recursive aliases, custom tags and
  non-finite numbers are rejected.
* ``validate``: JSON Schema 2020-12 with ``uri`` format checking; duplicate member ids in tools,
  connectors and dependencies; connectors must reference a declared tool; documentation and
  evidence references must stay inside the repository (and exist when ``repo_root`` is given);
  ``NOASSERTION`` licences, deprecated status, tools without automation interfaces and proprietary
  tools without open alternatives are warnings; dependency versions are ``<major>.<minor>.x``.

Registry-level checks that need the complete manifest set (unknown dependency ids, version
conflicts, dependency cycles, duplicate capabilities across repositories) are only performed when
``known_manifests`` is supplied; a single manifest cannot establish them.

The ``aipe.magnetics`` extension authored in this repository's ``aipe.yaml`` is validated against
:class:`MagneticsManifestExtension`. Nothing here fetches URLs or executes connectors.
"""

from __future__ import annotations

import hashlib
import math
import os
import posixpath
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

import yaml
from jsonschema import Draft202012Validator, FormatChecker
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, field_validator

from aipe_magnetics.common.errors import DataIntegrityError, DataNotFoundError, InvalidInputError, UnsupportedOperationError
from aipe_magnetics.common.jsonio import load as load_json

REGISTRY_REPOSITORY = "https://github.com/FulongLi/AIPE-Registry"
REGISTRY_COMMIT = "f419260bc604e5e79ecde9d998139a39a5e0ebdf"
MANIFEST_SCHEMA_VERSION = "0.1.0"
SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
SCHEMA_PATH = SCHEMA_DIR / "capability-manifest.schema.json"
VENDORED_SHA256 = {"capability-manifest.schema.json": "6d0818650937010c12f5b33efcb1adf9246118f80d989ee7d1079fa8e6e24c5a"}
#: Namespaced extension key this repository authors in ``aipe.yaml``.
MAGNETICS_EXTENSION_KEY = "aipe.magnetics"
#: ``id`` of this repository's own manifest (``aipe.yaml``); evidence about "the repository manifest"
#: is only produced for a manifest carrying this id and the ``aipe.magnetics`` extension.
REPOSITORY_MANIFEST_ID = "aipe.magnetics-database"
_DEPENDENCY_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.x$")
#: Registry ``confined``: drive letters and URL schemes are not repository-relative paths.
_UNSAFE_PREFIX = re.compile(r"^[A-Za-z]+:")
#: A verification level named in a free-text maturity statement is a claim (``L0``-``L3``).
_LEVEL_TOKEN = re.compile(r"\bL[0-3]\b")


# --------------------------------------------------------------------------------------------------
# YAML loader mirroring Registry's UniqueLoader
# --------------------------------------------------------------------------------------------------
class UniqueLoader(yaml.SafeLoader):
    """Registry ``UniqueLoader``: safe YAML with unique string keys and YAML 1.2 boolean words only."""


# Registry: YAML 1.2 treats words such as on/off/yes/no as strings, not booleans.
UniqueLoader.yaml_implicit_resolvers = {
    key: [(tag, pattern) for tag, pattern in values if tag != "tag:yaml.org,2002:bool"]
    for key, values in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
UniqueLoader.add_implicit_resolver("tag:yaml.org,2002:bool",
                                   re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))


def _mapping(loader: yaml.SafeLoader, node: yaml.Node, deep: bool = False) -> dict[str, Any]:
    # Registry ``_mapping``: reject duplicate and non-string keys instead of silently taking the last one.
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:  # type: ignore[union-attr]
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise InvalidInputError("YAML object keys must be strings")
        if key in result:
            raise InvalidInputError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _json_values(value: Any, seen: set[int] | None = None) -> None:
    # Registry ``_json_values``: JSON-compatible scalars, finite numbers, no recursive aliases.
    seen = set() if seen is None else seen
    if isinstance(value, (list, dict)):
        if id(value) in seen:
            raise InvalidInputError("recursive YAML aliases are unsupported")
        seen.add(id(value))
        for item in (value.values() if isinstance(value, dict) else value):
            _json_values(item, seen)
        seen.remove(id(value))
    elif value is not None and type(value) not in (str, bool, int, float):
        raise InvalidInputError(f"only JSON-compatible YAML values are supported; quote dates (got {type(value).__name__})")
    elif isinstance(value, float) and not math.isfinite(value):
        raise InvalidInputError("non-finite numbers are unsupported")


def parse_manifest_text(text: str, label: str = "manifest") -> Any:
    """Parse manifest YAML/JSON text with Registry's loader rules; raises :class:`InvalidInputError`."""
    try:
        value = yaml.load(text, Loader=UniqueLoader)  # noqa: S506 - UniqueLoader derives from SafeLoader
        _json_values(value)
        return value
    except yaml.YAMLError as error:
        raise InvalidInputError(f"{label}: {error}") from None
    except InvalidInputError as error:
        raise InvalidInputError(f"{label}: {error.message}", details=error.details) from None


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load ``aipe.yaml`` (UTF-8, BOM tolerated like Registry) and require a top-level mapping.

    Errors are typed: a missing path or a directory is :class:`DataNotFoundError`; an unreadable or
    non-UTF-8 file and any loader-rule violation are :class:`InvalidInputError` (path in ``details``).
    """
    path = Path(path)
    if not path.is_file():
        reason = "is a directory, not a file" if path.is_dir() else "not found"
        raise DataNotFoundError(f"manifest {reason}: {path}", details={"path": str(path)})
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise InvalidInputError(f"{path}: manifest is not UTF-8 text ({error.reason} at byte {error.start})",
                                details={"path": str(path)}) from None
    except OSError as error:
        raise InvalidInputError(f"{path}: manifest cannot be read: {error.strerror or error}",
                                details={"path": str(path), "errno": error.errno}) from None
    manifest = parse_manifest_text(text, str(path))
    if not isinstance(manifest, dict):
        raise InvalidInputError(f"{path}: manifest must be a mapping, got {type(manifest).__name__}",
                                details={"path": str(path)})
    return manifest


def repository_manifest_problems(manifest: Any) -> list[str]:
    """Why ``manifest`` is *not* this repository's own manifest (empty when it is).

    A manifest found by walking up the directory tree or named by ``AIPE_MAGNETICS_MANIFEST`` may
    belong to an unrelated tree; evidence about "the repository manifest" must not be produced for it.
    """
    problems: list[str] = []
    if not isinstance(manifest, dict):
        return [f"manifest must be a mapping, got {type(manifest).__name__}"]
    if manifest.get("id") != REPOSITORY_MANIFEST_ID:
        problems.append(f"manifest id {manifest.get('id')!r} is not {REPOSITORY_MANIFEST_ID!r}")
    extensions = manifest.get("extensions")
    if not isinstance(extensions, dict) or not isinstance(extensions.get(MAGNETICS_EXTENSION_KEY), dict):
        problems.append(f"manifest has no extensions[{MAGNETICS_EXTENSION_KEY!r}] mapping")
    return problems


# --------------------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------------------
def verify_vendored_integrity() -> dict[str, str]:
    digests: dict[str, str] = {}
    for name, expected in VENDORED_SHA256.items():
        path = SCHEMA_DIR / name
        if not path.is_file():
            raise DataIntegrityError(f"vendored AIPE-Registry file missing: {name}", details={"path": str(path)})
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise DataIntegrityError(f"vendored AIPE-Registry file {name} differs from the pinned upstream blob",
                                     details={"expected_sha256": expected, "actual_sha256": actual})
        digests[name] = actual
    return digests


@lru_cache(maxsize=1)
def manifest_schema() -> dict[str, Any]:
    schema = load_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    return schema


@lru_cache(maxsize=1)
def manifest_validator() -> Draft202012Validator:
    checker = FormatChecker()
    if "uri" not in checker.checkers:  # Registry ``validate``: fail closed without URI format validation
        raise UnsupportedOperationError("URI format validation is unavailable; install jsonschema[format]")
    return Draft202012Validator(manifest_schema(), format_checker=checker)


# --------------------------------------------------------------------------------------------------
# Magnetics extension model (authoring target for aipe.yaml extensions['aipe.magnetics'])
# --------------------------------------------------------------------------------------------------
VerificationLevel = Literal["L0", "L1", "L2", "L3"]
_LEVEL_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


class ModelMaturity(BaseModel):
    """Structured maturity statement for one model/capability id in ``model_maturity``."""

    model_config = ConfigDict(extra="forbid")
    maturity: str = Field(min_length=1, description="e.g. experimental, prototype, usable, stable, reference_only")
    family: str | None = Field(default=None, min_length=1)
    verification: VerificationLevel | None = Field(default=None,
                                                   description="highest verification level claimed (L0-L3)")
    notes: str | None = Field(default=None, min_length=1)


class MagneticsManifestExtension(BaseModel):
    """Shape of ``extensions['aipe.magnetics']`` in this repository's ``aipe.yaml``.

    Required keys (the authoring contract):

    * ``model_maturity``: model or capability id -> either a plain statement (``"L1 software-tested"``)
      or a :class:`ModelMaturity` record ``{maturity, family?, verification?, notes?}``. A level
      token ``L0``-``L3`` in a plain statement *is* a verification claim (a statement naming two
      different levels is ambiguous and rejected; use the structured form). Every claimed level,
      structured or plain, is cross-checked against the bundled evidence registry when one is
      supplied to :func:`validate_magnetics_extension`; never claim more than was executed.
    * ``verification_evidence``: repository-relative paths or https URLs of evidence (reports, tests);
      confined to the repository and, with ``repo_root``, required to exist.
    * ``platforms``: e.g. ``linux, windows, macos``.
    * ``interfaces``: a list of interface names (``python, cli, mcp``) or a mapping from interface
      kind to its entry point (``{python_api: aipe_magnetics.api, cli: aipe-magnetics, ...}``).
    * ``optional_integrations``: objects with at least an ``id`` (recommended: ``kind``, ``status``,
      ``license``, ``notes``); none may be required for standalone operation.
    * ``engineering_limitations``: explicit statements of what is not claimed.
    * ``runtime_requirements``: e.g. ``{"required": [...], "optional": {...}}``.

    Optional keys: ``package`` (distribution name and version), ``python`` (version specifier) and
    ``standalone_operation`` (``{requires_<x>: bool}``; every value must be ``false`` because the
    standalone expert may not require the orchestrator, a Registry service, other domain experts,
    commercial software or LLM credentials).
    """

    model_config = ConfigDict(extra="forbid")
    model_maturity: dict[str, str | ModelMaturity]
    verification_evidence: list[str]
    platforms: list[str]
    interfaces: list[str] | dict[str, Any]
    optional_integrations: list[dict[str, Any]]
    engineering_limitations: list[str]
    runtime_requirements: dict[str, Any]
    package: str | None = Field(default=None, min_length=1)
    python: str | None = Field(default=None, min_length=1)
    standalone_operation: dict[str, StrictBool] | None = None

    @field_validator("model_maturity")
    @classmethod
    def _maturity(cls, value: dict[str, str | ModelMaturity]) -> dict[str, str | ModelMaturity]:
        if not value:
            raise ValueError("model_maturity must name at least one model or capability")
        for key, statement in value.items():
            if not key.strip() or (isinstance(statement, str) and not statement.strip()):
                raise ValueError("model_maturity keys and statements must be non-empty")
            if isinstance(statement, str) and len(set(_LEVEL_TOKEN.findall(statement))) > 1:
                raise ValueError(f"model_maturity[{key!r}] names several verification levels "
                                 f"({', '.join(sorted(set(_LEVEL_TOKEN.findall(statement))))}); the claim is ambiguous, "
                                 "use the structured form {maturity, verification}")
        return value

    @field_validator("verification_evidence", "platforms", "engineering_limitations")
    @classmethod
    def _strings(cls, value: list[str], info: Any) -> list[str]:
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(f"{info.field_name} entries must be non-empty strings")
        if len(set(value)) != len(value):
            raise ValueError(f"{info.field_name} entries must be unique")
        if info.field_name in ("platforms", "engineering_limitations") and not value:
            raise ValueError(f"{info.field_name} must not be empty")
        return value

    @field_validator("interfaces")
    @classmethod
    def _interfaces(cls, value: list[str] | dict[str, Any]) -> list[str] | dict[str, Any]:
        if not value:
            raise ValueError("interfaces must not be empty")
        if isinstance(value, list):
            if any(not isinstance(item, str) or not item.strip() for item in value) or len(set(value)) != len(value):
                raise ValueError("interfaces entries must be unique non-empty strings")
        elif any(not key.strip() for key in value):
            raise ValueError("interfaces keys must be non-empty")
        return value

    @field_validator("optional_integrations")
    @classmethod
    def _integrations(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ids = []
        for item in value:
            if not isinstance(item.get("id"), str) or not item["id"].strip():
                raise ValueError("each optional integration needs a non-empty string 'id'")
            ids.append(item["id"])
        if len(set(ids)) != len(ids):
            raise ValueError("optional integration ids must be unique")
        return value

    @field_validator("standalone_operation")
    @classmethod
    def _standalone(cls, value: dict[str, StrictBool] | None) -> dict[str, StrictBool] | None:
        if value is not None and any(not key.strip() for key in value):
            raise ValueError("standalone_operation keys must be non-empty")
        return value

    def claimed_verification_levels(self) -> dict[str, str]:
        """``{model id: level}`` for every entry that claims a verification level.

        Structured entries claim ``verification``; a plain statement claims the single level token
        it contains (``"L2 verified against analytical references"`` claims ``L2``). Statements
        without a level token claim nothing.
        """
        claims: dict[str, str] = {}
        for key, entry in self.model_maturity.items():
            if isinstance(entry, ModelMaturity):
                if entry.verification is not None:
                    claims[key] = entry.verification
            else:
                tokens = set(_LEVEL_TOKEN.findall(entry))
                if tokens:
                    claims[key] = tokens.pop()  # exactly one; several are rejected by the validator
        return claims


def validate_magnetics_extension(manifest: dict[str, Any], repo_root: Path | str | None = None, *,
                                 evidence_registry: Any | None = None) -> list[dict[str, str]]:
    """Diagnostics for ``extensions['aipe.magnetics']`` (none when the key is absent).

    ``evidence_registry`` is a :class:`~aipe_magnetics.verification.evidence.VerificationRegistry`
    (anything with ``highest_level(capability) -> "L0".."L3"``); when given, every claimed
    ``verification`` level must be supported by passing evidence, otherwise the claim is an error.
    Without it the claims are reported as unchecked (warning).
    """
    extensions = manifest.get("extensions") if isinstance(manifest, dict) else None
    if not isinstance(extensions, dict) or MAGNETICS_EXTENSION_KEY not in extensions:
        return []
    prefix = f"extensions/{MAGNETICS_EXTENSION_KEY}"
    try:
        extension = MagneticsManifestExtension.model_validate(extensions[MAGNETICS_EXTENSION_KEY])
    except ValidationError as error:
        return [{"severity": "error", "code": "magnetics_extension",
                 "message": f"{prefix}/" + "/".join(str(p) for p in item["loc"]) + f": {item['msg']}"}
                for item in error.errors()]
    diagnostics: list[dict[str, str]] = []
    root = None if repo_root is None else Path(repo_root)
    for reference in extension.verification_evidence:
        if reference.startswith("https://"):
            continue
        try:
            path = confined_reference(reference, root)
        except InvalidInputError as error:
            diagnostics.append({"severity": "error", "code": "unsafe_reference",
                                "message": f"{prefix}/verification_evidence: {error.message}"})
            continue
        if path is not None and not path.is_file():
            diagnostics.append({"severity": "error", "code": "documentation_drift",
                                "message": f"{prefix}/verification_evidence: missing file: {reference}"})
    for key, required in sorted((extension.standalone_operation or {}).items()):
        if required:
            diagnostics.append({"severity": "error", "code": "standalone_operation",
                                "message": f"{prefix}/standalone_operation/{key} is true; the standalone expert "
                                           "must not require it"})
    claims = extension.claimed_verification_levels()
    if claims and evidence_registry is None:
        diagnostics.append({"severity": "warning", "code": "verification_claims_unchecked",
                            "message": f"{prefix}/model_maturity claims verification levels for "
                                       f"{len(claims)} model(s); no evidence registry supplied for cross-checking"})
    elif claims:
        for model_id, claimed in sorted(claims.items()):
            supported = evidence_registry.highest_level(model_id)
            if _LEVEL_ORDER[claimed] > _LEVEL_ORDER.get(supported, 0):
                diagnostics.append({"severity": "error", "code": "verification_claim",
                                    "message": f"{prefix}/model_maturity/{model_id} claims {claimed} but bundled "
                                               f"evidence supports {supported}"})
    return diagnostics


def find_repository_manifest(start: Path | str | None = None, *, max_levels: int = 8) -> Path | None:
    """Locate the repository ``aipe.yaml``: ``AIPE_MAGNETICS_MANIFEST`` env var, else walk up from ``start``.

    ``start`` defaults to this package's directory, so a source checkout (``src/`` layout) is found;
    an installed wheel without the repository yields ``None``.
    """
    override = os.environ.get("AIPE_MAGNETICS_MANIFEST")
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None
    current = Path(start).resolve() if start is not None else Path(__file__).resolve().parent
    for _ in range(max_levels):
        if (current / "aipe.yaml").is_file() and (current / "pyproject.toml").is_file():
            return current / "aipe.yaml"
        if current.parent == current:
            break
        current = current.parent
    return None


# --------------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------------
def confined_reference(reference: str, repo_root: Path | None = None) -> Path | None:
    """Registry ``confined``: a repository-relative reference may not escape the repository.

    Both modes apply the same lexical rule as Registry's ``(root / unquote(path)).resolve()``
    containment test, so a reference accepted here is accepted by Registry and vice versa:

    * drive letters, URL schemes and backslashes are rejected before *and after* percent-decoding
      (``%5C`` would otherwise behave differently on Windows and POSIX);
    * the decoded path is normalised (``docs/sub/../a.md`` stays inside and is accepted, exactly as
      Registry's ``resolve()`` accepts it); an absolute path, a path that climbs above the root or a
      reference to the root itself (``.``/empty, never a file) is rejected;
    * with ``repo_root`` the resolved path (symlinks followed) must additionally stay inside the root,
      and that path is returned; without it ``None`` is returned after the lexical check only.
    """
    if not isinstance(reference, str) or _UNSAFE_PREFIX.match(reference) or "\\" in reference:
        raise InvalidInputError(f"not a repository-relative path: {reference}")
    relative = unquote(urlsplit(reference).path)
    if _UNSAFE_PREFIX.match(relative) or "\\" in relative:
        raise InvalidInputError(f"not a repository-relative path: {reference} (percent-decoded form)")
    normalised = posixpath.normpath(relative) if relative else "."
    if normalised in (".", "..") or normalised.startswith(("/", "../")):
        raise InvalidInputError(f"path escapes repository: {reference}")
    if repo_root is None:
        return None
    root = Path(repo_root).resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise InvalidInputError(f"path escapes repository: {reference}")
    return candidate


def validate_manifest(manifest: dict[str, Any], repo_root: Path | str | None = None, *,
                      known_manifests: dict[str, dict[str, Any]] | None = None,
                      evidence_registry: Any | None = None) -> list[dict[str, str]]:
    """Return sorted diagnostics ``{severity, code, message}``; no errors means Registry-acceptable.

    ``repo_root`` enables existence checks of repository-relative references, ``known_manifests``
    (``{id: manifest}``) enables cross-manifest dependency checks and ``evidence_registry`` enables
    the cross-check of ``aipe.magnetics`` verification claims (see :func:`validate_magnetics_extension`).
    """
    if not isinstance(manifest, dict):
        return [{"severity": "error", "code": "schema", "message": f"/: manifest must be an object, got {type(manifest).__name__}"}]
    diagnostics: list[dict[str, str]] = []

    def report(severity: str, code: str, message: str) -> None:
        diagnostics.append({"severity": severity, "code": code, "message": message})

    # Registry ``validate``: JSON schema first; semantic checks only run on schema-valid manifests.
    schema_errors = list(manifest_validator().iter_errors(manifest))
    for error in schema_errors:
        pointer = "/" + "/".join(str(part) for part in error.absolute_path)
        report("error", "schema", f"{pointer}: {error.message}")
    if schema_errors:
        return _sorted(diagnostics)

    # duplicate member ids
    for kind in ("tools", "connectors", "dependencies"):
        values = [item["id"] for item in manifest[kind]]
        if len(values) != len(set(values)):
            report("error", "duplicate_member", f"duplicate IDs in {kind}")

    # tools and connectors
    tool_ids = {tool["id"] for tool in manifest["tools"]}
    open_capabilities = {capability for tool in manifest["tools"] if tool["licence_class"] == "open_source"
                         for capability in tool["capabilities"]}
    for tool in manifest["tools"]:
        if not tool["interfaces"]:
            report("warning", "unsupported_tool", f"{tool['id']} declares no automation interface")
        if tool["licence_class"] != "open_source":
            missing = sorted(set(tool["capabilities"]) - open_capabilities)
            if missing:
                report("warning", "missing_open_alternative",
                       f"{tool['id']}: {', '.join(missing)}; review fidelity and limitations")
    for connector in manifest["connectors"]:
        if connector["tool"] not in tool_ids:
            report("error", "unsupported_connector_tool", f"{connector['id']} references unknown tool {connector['tool']}")
        if connector["status"] == "supported" and not connector["supported_versions"]:
            report("warning", "unversioned_connector", connector["id"])

    # documentation and evidence references
    root = None if repo_root is None else Path(repo_root)
    for reference in manifest["documentation"] + manifest["validation"]["evidence"]:
        if reference.startswith("https://"):
            continue
        try:
            path = confined_reference(reference, root)
        except InvalidInputError as error:
            report("error", "unsafe_reference", error.message)
            continue
        if path is not None and not path.is_file():
            report("error", "documentation_drift", f"missing file: {reference}")

    # licence, status, dependency version format
    if manifest["license"] == "NOASSERTION":
        report("warning", "unresolved_license", "rights remain unconfirmed; no open-source licence inferred")
    if manifest["status"] == "deprecated":
        report("warning", "legacy_replacement", "review replacement/migration documentation before use")
    for dependency in manifest["dependencies"]:
        if not _DEPENDENCY_VERSION.match(str(dependency["version"])):
            report("error", "dependency_version", f"{dependency['id']}: version must look like 0.1.x")
        if dependency["id"] == manifest["id"]:
            report("error", "dependency_cycle", f"{manifest['id']} depends on itself")

    # cross-manifest checks (only with a known manifest set)
    if known_manifests is not None:
        for dependency in manifest["dependencies"]:
            target = known_manifests.get(dependency["id"])
            if target is None:
                report("error", "unknown_dependency", dependency["id"])
            elif not str(target.get("schema_version", "")).startswith(str(dependency["version"])[:-1]):
                report("error", "dependency_conflict", f"incompatible dependency: {dependency['id']}")
        for other_id, other in known_manifests.items():
            if other_id == manifest["id"] or not isinstance(other, dict):
                continue
            if other.get("repository") == manifest["repository"]:
                report("error", "duplicate_repository", f"one canonical manifest per repository is required ({other_id})")
            shared = sorted(set(manifest["capabilities"]) & set(other.get("capabilities", [])))
            if shared:
                report("warning", "duplicate_capability", f"{', '.join(shared)}: also declared by {other_id}; "
                                                          "may be intentional alternatives")

    diagnostics.extend(validate_magnetics_extension(manifest, root, evidence_registry=evidence_registry))
    return _sorted(diagnostics)


def _sorted(diagnostics: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(diagnostics, key=lambda item: (item["severity"], item["code"], item["message"]))


def manifest_errors(diagnostics: list[dict[str, str]]) -> list[dict[str, str]]:
    return [item for item in diagnostics if item["severity"] == "error"]


def describe_capabilities(manifest: dict[str, Any]) -> dict[str, Any]:
    """Compact, JSON-serialisable description of what a manifest declares (no validation performed)."""
    if not isinstance(manifest, dict):
        raise InvalidInputError(f"manifest must be a mapping, got {type(manifest).__name__}")
    extensions = manifest.get("extensions") if isinstance(manifest.get("extensions"), dict) else {}
    magnetics = extensions.get(MAGNETICS_EXTENSION_KEY) if isinstance(extensions, dict) else None
    compatibility = manifest.get("compatibility") if isinstance(manifest.get("compatibility"), dict) else {}
    validation = manifest.get("validation") if isinstance(manifest.get("validation"), dict) else {}
    return {
        "id": manifest.get("id"),
        "name": manifest.get("name"),
        "type": manifest.get("type"),
        "status": manifest.get("status"),
        "maturity": manifest.get("maturity"),
        "license": manifest.get("license"),
        "repository": manifest.get("repository"),
        "capabilities": list(manifest.get("capabilities", [])),
        "inputs": [port.get("name") for port in manifest.get("inputs", []) if isinstance(port, dict)],
        "outputs": [port.get("name") for port in manifest.get("outputs", []) if isinstance(port, dict)],
        "dependencies": [{"id": d.get("id"), "version": d.get("version"), "optional": d.get("optional")}
                         for d in manifest.get("dependencies", []) if isinstance(d, dict)],
        "compatibility": {"core": compatibility.get("core"), "engineering_state": compatibility.get("engineering_state"),
                          "integration": compatibility.get("integration")},
        "tools": [{"id": t.get("id"), "licence_class": t.get("licence_class"), "optional": t.get("optional"),
                   "interfaces": list(t.get("interfaces", [])), "platforms": list(t.get("platforms", []))}
                  for t in manifest.get("tools", []) if isinstance(t, dict)],
        "connectors": [{"id": c.get("id"), "tool": c.get("tool"), "type": c.get("type"), "status": c.get("status")}
                       for c in manifest.get("connectors", []) if isinstance(c, dict)],
        "validation_status": validation.get("status"),
        "validation_evidence": list(validation.get("evidence", [])),
        "limitations": list(validation.get("limitations", [])),
        "documentation": list(manifest.get("documentation", [])),
        "extensions": sorted(extensions) if isinstance(extensions, dict) else [],
        "magnetics_extension": magnetics if isinstance(magnetics, dict) else None,
    }


__all__ = [
    "MAGNETICS_EXTENSION_KEY", "MANIFEST_SCHEMA_VERSION", "REGISTRY_COMMIT", "REGISTRY_REPOSITORY",
    "REPOSITORY_MANIFEST_ID", "SCHEMA_PATH", "VENDORED_SHA256", "MagneticsManifestExtension", "ModelMaturity",
    "UniqueLoader", "confined_reference", "describe_capabilities", "find_repository_manifest", "load_manifest",
    "manifest_errors", "manifest_schema", "manifest_validator", "parse_manifest_text", "repository_manifest_problems",
    "validate_magnetics_extension", "validate_manifest", "verify_vendored_integrity",
]
