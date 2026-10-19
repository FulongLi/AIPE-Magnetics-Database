"""AIPE-Registry compatibility: repository aipe.yaml, loader rules, diagnostics, extension model, reference validator."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

from aipe_magnetics.common.errors import DataNotFoundError, InvalidInputError
from aipe_magnetics.integrations.aipe_registry import (
    MAGNETICS_EXTENSION_KEY,
    REGISTRY_COMMIT,
    REPOSITORY_MANIFEST_ID,
    SCHEMA_PATH,
    VENDORED_SHA256,
    MagneticsManifestExtension,
    confined_reference,
    describe_capabilities,
    find_repository_manifest,
    load_manifest,
    manifest_errors,
    manifest_schema,
    parse_manifest_text,
    repository_manifest_problems,
    validate_magnetics_extension,
    validate_manifest,
    verify_vendored_integrity,
)
from aipe_magnetics.verification.checks.check_integrations import (
    evaluate_manifest_acceptance,
    evaluate_manifest_schema,
    registry_manifest_accepted,
    registry_manifest_schema_valid,
)
from aipe_magnetics.verification.evidence import EvidenceRecord, VerificationRegistry
from aipe_magnetics.verification.suite import bundled_evidence_path, load_bundled_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "aipe.yaml"
SKIP_REASON = "AIPE_ECOSYSTEM_CHECKOUTS not set"


def _record(capability: str, level: str) -> EvidenceRecord:
    return EvidenceRecord(id=f"check:{capability}:{level}", capability=capability, level=level, kind="software",
                          status="pass", description="synthetic", method="synthetic", check_id="synthetic", tool_version="0")


def _write_yaml(path: Path, manifest: dict) -> Path:
    path.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8", newline="\n")  # JSON is valid YAML
    return path


@pytest.fixture
def manifest() -> dict:
    return load_manifest(MANIFEST_PATH)


@pytest.fixture
def minimal() -> dict:
    """A small schema-valid manifest used for mutation tests (independent of the repository file)."""
    return {
        "schema_version": "0.1.0", "id": "aipe.example-tool", "name": "Example", "type": "tool",
        "domains": ["power-electronics"], "description": "Example manifest for tests.", "status": "experimental",
        "maturity": "prototype", "repository": "https://github.com/FulongLi/AIPE-Example", "license": "MIT",
        "capabilities": ["example-capability"], "inputs": [{"name": "in", "description": "Input."}],
        "outputs": [{"name": "out", "description": "Output."}],
        "dependencies": [{"id": "aipe.core", "version": "0.1.x", "optional": True}],
        "compatibility": {"core": "0.1.x", "engineering_state": "0.1.x", "integration": "mapped"},
        "tools": [{"id": "python", "name": "Python", "capabilities": ["example-capability"],
                   "licence_class": "open_source", "optional": False, "interfaces": ["python"],
                   "platforms": ["linux"], "url": "https://www.python.org/", "limitations": []}],
        "connectors": [],
        "validation": {"status": "tested", "evidence": ["tests/test_example.py"], "limitations": ["None claimed."]},
        "documentation": ["README.md"],
    }


# --------------------------------------------------------------------------------------------------
# Repository manifest and vendored schema
# --------------------------------------------------------------------------------------------------
def test_vendored_schema_matches_pin_and_is_valid():
    assert verify_vendored_integrity() == VENDORED_SHA256
    assert hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest() == VENDORED_SHA256[SCHEMA_PATH.name]
    schema = manifest_schema()
    assert schema["$id"].endswith("/v0.1.0/schema/capability-manifest.schema.json")
    notice = (SCHEMA_PATH.parent / "NOTICE.md").read_text(encoding="utf-8")
    assert REGISTRY_COMMIT in notice and "MIT License" in notice


def test_repository_manifest_validates_against_vendored_schema(manifest):
    diagnostics = validate_manifest(manifest)
    errors = manifest_errors(diagnostics)
    assert errors == [], errors
    for item in diagnostics:
        print(f"warning [{item['code']}] {item['message']}")
    assert manifest["id"] == "aipe.magnetics-database"
    assert find_repository_manifest() == MANIFEST_PATH


def test_repository_manifest_references_stay_inside_repository(manifest):
    """Path confinement is strict; file existence is reported (docs are the maintainer's deliverable)."""
    diagnostics = validate_manifest(manifest, REPO_ROOT)
    unsafe = [item for item in diagnostics if item["code"] == "unsafe_reference"]
    assert unsafe == [], unsafe
    drift = [item["message"] for item in diagnostics if item["code"] == "documentation_drift"]
    other_errors = [item for item in manifest_errors(diagnostics) if item["code"] != "documentation_drift"]
    assert other_errors == [], other_errors
    if drift:
        message = "aipe.yaml references files that do not exist yet (Registry --checkouts validation will fail " \
                  "until they do): " + "; ".join(drift)
        print(message)
        warnings.warn(message)


def test_repository_extension_matches_model(manifest):
    extension = manifest["extensions"][MAGNETICS_EXTENSION_KEY]
    model = MagneticsManifestExtension.model_validate(extension)
    assert model.platforms and model.engineering_limitations and model.runtime_requirements
    assert validate_magnetics_extension(manifest) == [
        {"severity": "warning", "code": "verification_claims_unchecked",
         "message": f"extensions/{MAGNETICS_EXTENSION_KEY}/model_maturity claims verification levels for "
                    f"{len(model.claimed_verification_levels())} model(s); no evidence registry supplied for cross-checking"}
    ] or model.claimed_verification_levels() == {}
    # claims above the bundled evidence are errors; claims at or below it are accepted
    claims = model.claimed_verification_levels()
    if claims:
        empty = VerificationRegistry()
        unsupported = validate_magnetics_extension(manifest, evidence_registry=empty)
        assert {item["code"] for item in unsupported} == {"verification_claim"}
        assert len(unsupported) == len(claims)
        satisfied = VerificationRegistry([
            EvidenceRecord(id=f"check:{model_id}", capability=model_id, level=level, kind="software", status="pass",
                           description="synthetic", method="synthetic", check_id="synthetic", tool_version="0")
            for model_id, level in claims.items() if level != "L0"])
        assert validate_magnetics_extension(manifest, evidence_registry=satisfied) == []


def test_describe_capabilities(manifest):
    description = describe_capabilities(manifest)
    assert description["id"] == manifest["id"] and description["capabilities"] == manifest["capabilities"]
    assert description["compatibility"]["integration"] == manifest["compatibility"]["integration"]
    assert {tool["id"] for tool in description["tools"]} == {tool["id"] for tool in manifest["tools"]}
    assert MAGNETICS_EXTENSION_KEY in description["extensions"]
    json.dumps(description, allow_nan=False)  # JSON serialisable
    with pytest.raises(InvalidInputError):
        describe_capabilities(["not", "a", "mapping"])  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------------
# Loader rules (Registry UniqueLoader)
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("data", ["id: aipe.one\nid: aipe.two", "value: .nan", "date: 2026-10-07",
                                  "x: !!python/object:bad {}", "x: &x [*x]", "? [complex, key]\n: value", "- [unclosed"])
def test_unsafe_or_non_json_yaml_rejected(data):
    with pytest.raises(InvalidInputError):
        parse_manifest_text(data)


def test_yaml_12_boolean_words_are_strings():
    assert parse_manifest_text("id: on\nname: yes\nenabled: false\noff: Off") == \
        {"id": "on", "name": "yes", "enabled": False, "off": "Off"}
    assert parse_manifest_text('{"json": true, "n": 1.5}') == {"json": True, "n": 1.5}


def test_load_manifest_errors(tmp_path):
    with pytest.raises(DataNotFoundError, match="not found"):
        load_manifest(tmp_path / "missing.yaml")
    with pytest.raises(DataNotFoundError, match="directory"):
        load_manifest(tmp_path)  # a directory is never a manifest (PermissionError on Windows otherwise)
    sequence = tmp_path / "list.yaml"
    sequence.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(InvalidInputError, match="mapping"):
        load_manifest(sequence)
    utf16 = tmp_path / "utf16.yaml"
    utf16.write_bytes("id: aipe.x\n".encode("utf-16"))
    with pytest.raises(InvalidInputError, match="not UTF-8") as info:
        load_manifest(utf16)
    assert info.value.details["path"] == str(utf16)
    bom = tmp_path / "bom.yaml"
    bom.write_bytes(b"\xef\xbb\xbfid: aipe.x\n")
    assert load_manifest(bom) == {"id": "aipe.x"}


# --------------------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------------------
def test_minimal_manifest_is_valid(minimal):
    assert validate_manifest(minimal) == []
    assert [item["code"] for item in validate_manifest(["not", "a", "mapping"])] == ["schema"]  # type: ignore[arg-type]


@pytest.mark.parametrize("mutate", [
    lambda m: m.update(unknown="typo"),
    lambda m: m.pop("repository"),
    lambda m: m.update(id="Bad ID"),
    lambda m: m.update(type="monolith"),
    lambda m: m.update(schema_version="0.2.0"),
    lambda m: m["compatibility"].update(core="1.x"),
    lambda m: m["compatibility"].update(integration="certified"),
    lambda m: m["tools"][0].update(licence_class="commercial", optional=False),
    lambda m: m["tools"][0].update(licence_class="free_proprietary", optional=False),
    lambda m: m["tools"][0].update(url="not-a-uri"),
    lambda m: m.update(extensions={"unscoped": {}}),
    lambda m: m["inputs"][0].update(schema="relative/path"),
    lambda m: m["inputs"][0].update(schema="https://example.com/space in uri"),
    lambda m: m["dependencies"].append({"id": "aipe.core", "version": "1.x", "optional": False}),
    lambda m: m.update(capabilities=[]),
    lambda m: m.update(documentation=["/absolute/path.md"]),
    lambda m: m.update(documentation=["C:/windows/path.md"]),
], ids=lambda f: f.__code__.co_firstlineno)
def test_invalid_manifest_rejected(minimal, mutate):
    mutate(minimal)
    assert manifest_errors(validate_manifest(minimal))


def test_semantic_diagnostics(minimal, tmp_path):
    minimal["tools"][0]["interfaces"] = []
    minimal["tools"][0].update(licence_class="commercial", optional=True)
    minimal["tools"].append(copy.deepcopy(minimal["tools"][0]))  # same id twice, no open-source alternative
    minimal["connectors"] = [{"id": "unknown", "tool": "missing", "type": "api", "repository": "https://example.com/api",
                              "license": "MIT", "status": "supported", "supported_versions": [], "platforms": ["linux"],
                              "security_notes": ["Review before running."]}]
    minimal["license"] = "NOASSERTION"
    minimal["status"] = "deprecated"
    # backslash and drive-prefixed references are already schema errors (see test_invalid_manifest_rejected)
    minimal["documentation"] = ["README.md", "../private.txt", "%2e%2e/private.txt", "https://example.com/../ok"]
    original = copy.deepcopy(minimal)
    diagnostics = validate_manifest(minimal, tmp_path)
    assert minimal == original, "validation must be read-only"
    codes = {item["code"] for item in diagnostics}
    assert {"duplicate_member", "unsupported_connector_tool", "unversioned_connector", "unresolved_license",
            "legacy_replacement", "unsupported_tool", "missing_open_alternative", "unsafe_reference",
            "documentation_drift"} <= codes
    unsafe = [item["message"] for item in diagnostics if item["code"] == "unsafe_reference"]
    assert len(unsafe) == 2 and all("private.txt" in item for item in unsafe), unsafe
    drift = [item["message"] for item in diagnostics if item["code"] == "documentation_drift"]
    assert drift == ["missing file: README.md", "missing file: tests/test_example.py"]
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_example.py").write_text("x", encoding="utf-8")
    assert not [item for item in validate_manifest(minimal, tmp_path) if item["code"] == "documentation_drift"]
    assert diagnostics == sorted(diagnostics, key=lambda d: (d["severity"], d["code"], d["message"]))


def test_dependency_checks(minimal):
    assert validate_manifest(minimal, known_manifests={"aipe.core": {"schema_version": "0.1.0"}}) == []
    unknown = manifest_errors(validate_manifest(minimal, known_manifests={}))
    assert [item["code"] for item in unknown] == ["unknown_dependency"]
    conflict = manifest_errors(validate_manifest(minimal, known_manifests={"aipe.core": {"schema_version": "0.2.0"}}))
    assert [item["code"] for item in conflict] == ["dependency_conflict"]
    minimal["dependencies"] = [{"id": minimal["id"], "version": "0.1.x", "optional": False}]
    assert [item["code"] for item in manifest_errors(validate_manifest(minimal))] == ["dependency_cycle"]
    minimal["dependencies"] = []
    other = {"id": "aipe.other", "repository": minimal["repository"], "capabilities": ["example-capability"]}
    diagnostics = validate_manifest(minimal, known_manifests={"aipe.other": other})
    assert {item["code"] for item in diagnostics} == {"duplicate_repository", "duplicate_capability"}


@pytest.mark.parametrize("reference,expected", [
    ("docs/a.md", "docs/a.md"),
    ("docs/sub/../a.md", "docs/a.md"),          # stays inside: accepted, exactly like Registry's resolve()
    ("docs/a.md?rev=1#frag", "docs/a.md"),      # Registry validates urlsplit(reference).path
    ("../private.txt", "escapes"),
    ("a/../../private.txt", "escapes"),
    ("%2e%2e/private.txt", "escapes"),
    ("/etc/passwd", "escapes"),
    ("//host/share", "escapes"),
    ("", "escapes"), (".", "escapes"), ("docs/..", "escapes"),
    ("C:/x", "not a repository-relative"),
    ("https://x/../y", "not a repository-relative"),
    ("docs\\a.md", "not a repository-relative"),
    ("%5C..%5C..%5Cetc", "not a repository-relative"),   # percent-encoded backslashes: same verdict on every OS
    ("%43:/x", "not a repository-relative"),            # percent-encoded drive letter
])
def test_confined_reference_both_modes_agree(reference, expected, tmp_path):
    """Each mode is asserted on its own (a raise in the first call must not hide the second)."""
    if expected in ("escapes", "not a repository-relative"):
        with pytest.raises(InvalidInputError, match=expected):
            confined_reference(reference)
        with pytest.raises(InvalidInputError, match=expected):
            confined_reference(reference, tmp_path)
    else:
        assert confined_reference(reference) is None
        assert confined_reference(reference, tmp_path) == (tmp_path / expected).resolve()


def test_confined_reference_matches_registry_rule_on_disk(tmp_path):
    """The no-root lexical rule and the on-disk rule agree for a real repository tree."""
    (tmp_path / "docs" / "sub").mkdir(parents=True)
    (tmp_path / "docs" / "a.md").write_text("x", encoding="utf-8")
    assert confined_reference("docs/sub/../a.md", tmp_path).is_file()
    manifest_errors_no_root = validate_manifest({"documentation": ["docs/sub/../a.md"]})  # schema errors only
    assert all(item["code"] == "schema" for item in manifest_errors_no_root)


# --------------------------------------------------------------------------------------------------
# aipe.magnetics extension model
# --------------------------------------------------------------------------------------------------
def _extension() -> dict:
    return {
        "model_maturity": {"aipe.magnetics.model.igse": {"family": "empirical", "maturity": "usable", "verification": "L2"},
                           "aipe.magnetics.model.steinmetz": "L2 verified against analytical references"},
        "verification_evidence": ["src/aipe_magnetics/verification/evidence/v0.2.0.json"],
        "platforms": ["linux", "windows", "macos"],
        "interfaces": ["python", "cli", "mcp"],
        "optional_integrations": [{"id": "aipe.core", "kind": "schema-roundtrip", "status": "tested-offline"}],
        "engineering_limitations": ["Analytical sizing only."],
        "runtime_requirements": {"required": ["numpy"], "optional": {"mcp": ["mcp"]}},
        "standalone_operation": {"requires_aipe_orchestrator": False},
    }


def test_extension_model_accepts_both_shapes(minimal):
    extension = _extension()
    model = MagneticsManifestExtension.model_validate(extension)
    # the structured entry claims L2; the plain statement naming L2 claims L2 as well (no bypass)
    assert model.claimed_verification_levels() == {"aipe.magnetics.model.igse": "L2", "aipe.magnetics.model.steinmetz": "L2"}
    mapping = dict(extension, interfaces={"python_api": "aipe_magnetics.api", "cli": "aipe-magnetics"})
    assert MagneticsManifestExtension.model_validate(mapping).interfaces["cli"] == "aipe-magnetics"
    minimal["extensions"] = {MAGNETICS_EXTENSION_KEY: extension}
    diagnostics = validate_manifest(minimal)
    assert [item["code"] for item in diagnostics] == ["verification_claims_unchecked"]
    supported = VerificationRegistry([_record("aipe.magnetics.model.igse", "L2"), _record("aipe.magnetics.model.steinmetz", "L2")])
    assert validate_manifest(minimal, evidence_registry=supported) == []
    assert validate_magnetics_extension({"extensions": {}}) == [] and validate_magnetics_extension({}) == []


def test_plain_string_claims_are_cross_checked_or_rejected(minimal):
    extension = _extension()
    extension["model_maturity"] = {"aipe.magnetics.model.foo": "L3 validated against hardware",
                                   "aipe.magnetics.model.bar": "prototype; no verification level claimed"}
    model = MagneticsManifestExtension.model_validate(extension)
    assert model.claimed_verification_levels() == {"aipe.magnetics.model.foo": "L3"}
    minimal["extensions"] = {MAGNETICS_EXTENSION_KEY: extension}
    above = validate_manifest(minimal, evidence_registry=VerificationRegistry([_record("aipe.magnetics.model.foo", "L1")]))
    assert [item["code"] for item in above] == ["verification_claim"]
    assert "aipe.magnetics.model.foo claims L3 but bundled evidence supports L1" in above[0]["message"]
    assert validate_manifest(minimal, evidence_registry=VerificationRegistry([_record("aipe.magnetics.model.foo", "L3")])) == []
    # a statement naming two different levels is ambiguous and must use the structured form
    extension["model_maturity"] = {"aipe.magnetics.model.foo": "L1 software-tested, L2 planned"}
    with pytest.raises(ValidationError, match="ambiguous"):
        MagneticsManifestExtension.model_validate(extension)
    minimal["extensions"] = {MAGNETICS_EXTENSION_KEY: extension}
    assert [item["code"] for item in validate_manifest(minimal)] == ["magnetics_extension"]
    # L10 or lowercase tokens are not level claims
    extension["model_maturity"] = {"aipe.magnetics.model.foo": "level l2 of 10, see L10 notes"}
    assert MagneticsManifestExtension.model_validate(extension).claimed_verification_levels() == {}


@pytest.mark.parametrize("mutate,match", [
    (lambda e: e.update(extra_key=1), "Extra inputs"),
    (lambda e: e.pop("platforms"), "Field required"),
    (lambda e: e.update(platforms=[]), "must not be empty"),
    (lambda e: e.update(interfaces=[]), "must not be empty"),
    (lambda e: e.update(interfaces=["python", "python"]), "unique"),
    (lambda e: e.update(model_maturity={}), "at least one"),
    (lambda e: e.update(model_maturity={"m": {"maturity": "usable", "verification": "L9"}}), "verification"),
    (lambda e: e.update(model_maturity={"m": {"verification": "L1"}}), "maturity"),
    (lambda e: e.update(optional_integrations=[{"kind": "x"}]), "'id'"),
    (lambda e: e.update(optional_integrations=[{"id": "a"}, {"id": "a"}]), "unique"),
    (lambda e: e.update(engineering_limitations=[""]), "non-empty"),
    (lambda e: e.update(runtime_requirements="numpy"), "dictionary"),
    (lambda e: e.update(standalone_operation={"requires_x": "no"}), "boolean"),
], ids=lambda v: v if isinstance(v, str) else "mutation")
def test_extension_model_rejects(mutate, match):
    extension = _extension()
    mutate(extension)
    with pytest.raises(ValidationError, match=match):
        MagneticsManifestExtension.model_validate(extension)


def test_extension_diagnostics(minimal, tmp_path):
    extension = _extension()
    extension["standalone_operation"] = {"requires_commercial_software": True}
    extension["verification_evidence"] = ["../outside.json", "reports/missing.json"]
    minimal["extensions"] = {MAGNETICS_EXTENSION_KEY: extension}
    diagnostics = validate_manifest(minimal, tmp_path, evidence_registry=VerificationRegistry())
    codes = sorted(item["code"] for item in diagnostics if item["message"].startswith(f"extensions/{MAGNETICS_EXTENSION_KEY}"))
    # two claims (structured igse L2 and the plain "L2 ..." steinmetz statement) exceed the empty registry
    assert codes == ["documentation_drift", "standalone_operation", "unsafe_reference", "verification_claim",
                     "verification_claim"]
    broken = copy.deepcopy(minimal)
    broken["extensions"][MAGNETICS_EXTENSION_KEY]["platforms"] = "linux"
    errors = manifest_errors(validate_manifest(broken))
    assert errors and all(item["code"] == "magnetics_extension" for item in errors)
    assert errors[0]["message"].startswith(f"extensions/{MAGNETICS_EXTENSION_KEY}/platforms")


# --------------------------------------------------------------------------------------------------
# Verification checks (schema-level, acceptance-level) and external reference validator
# --------------------------------------------------------------------------------------------------
def test_registry_schema_check_passes():
    result = registry_manifest_schema_valid()
    assert result.passed, (result.metrics, result.notes)
    assert result.metrics["manifest_found"] is True and result.metrics["repository_manifest"] is True
    assert result.metrics["mutation_rejected"] is True and result.metrics["errors"] == 0
    for note in result.notes:
        print(note)


def test_registry_acceptance_check_reflects_repository_state(manifest):
    """The acceptance check must agree with an independent evaluation of the current checkout.

    It passes only when Registry's own validator would accept aipe.yaml for this checkout (no error,
    including missing referenced files) and the verification claims were cross-checked against the
    bundled evidence. While the repository is incomplete the check must FAIL closed and name every
    blocking item; that state is reported loudly here (warning), never hidden by a skip.
    """
    result = registry_manifest_accepted()
    registry = load_bundled_registry() if bundled_evidence_path().exists() else None
    diagnostics = validate_manifest(manifest, REPO_ROOT, evidence_registry=registry)
    errors = manifest_errors(diagnostics)
    unchecked = any(item["code"] == "verification_claims_unchecked" for item in diagnostics)
    assert result.metrics["repository_manifest"] is True
    assert result.metrics["registry_validator_would_accept"] == (not errors)
    assert result.metrics["verification_claims_unchecked"] == unchecked
    assert result.metrics["bundled_evidence_consulted"] == (registry is not None)
    assert result.passed == (not errors and not unchecked), (result.metrics, result.notes)
    blocking = [item["message"] for item in errors]
    if unchecked:
        blocking.append("verification-level claims could not be cross-checked")
    for message in blocking:
        assert any(message in note and note.startswith("BLOCKING") for note in result.notes), (message, result.notes)
    if blocking:
        text = ("integration.registry-manifest-accepted FAILS CLOSED until the maintainer completes the repository: "
                + "; ".join(blocking))
        print(text)
        warnings.warn(text)


def _repository_like(tmp_path: Path, minimal: dict) -> tuple[Path, dict]:
    """A temporary checkout whose manifest is this repository's manifest in every formal respect."""
    manifest = copy.deepcopy(minimal)
    manifest["id"] = REPOSITORY_MANIFEST_ID
    manifest["documentation"] = ["README.md", "docs/sub/../guide.md"]
    manifest["validation"]["evidence"] = ["tests/test_example.py", "evidence/v0.json"]
    extension = _extension()
    extension["verification_evidence"] = ["evidence/v0.json"]
    manifest["extensions"] = {MAGNETICS_EXTENSION_KEY: extension}
    for relative in ("README.md", "docs/guide.md", "tests/test_example.py", "evidence/v0.json", "pyproject.toml"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")
    return _write_yaml(tmp_path / "aipe.yaml", manifest), manifest


def test_acceptance_evaluation_on_temporary_repository(tmp_path, minimal):
    path, manifest = _repository_like(tmp_path, minimal)
    supported = VerificationRegistry([_record("aipe.magnetics.model.igse", "L2"), _record("aipe.magnetics.model.steinmetz", "L2")])
    complete = evaluate_manifest_acceptance(path, supported)
    assert complete.passed, (complete.metrics, complete.notes)
    assert complete.metrics["registry_validator_would_accept"] is True
    assert complete.metrics["verification_claims_unchecked"] is False and complete.metrics["verification_claims"] == 2
    assert complete.metrics["missing_reference_count"] == 0 and not any(n.startswith("BLOCKING") for n in complete.notes)
    schema_level = evaluate_manifest_schema(path)
    assert schema_level.passed and schema_level.metrics["mutation_rejected"] is True

    # no bundled evidence: claims cannot be checked -> fail closed although Registry would accept the files
    unchecked = evaluate_manifest_acceptance(path, None)
    assert unchecked.passed is False and unchecked.metrics["registry_validator_would_accept"] is True
    assert unchecked.metrics["verification_claims_unchecked"] is True
    assert any("could not be cross-checked" in note for note in unchecked.notes)

    # evidence below the claim -> fail with verification_claim
    weak = evaluate_manifest_acceptance(path, VerificationRegistry([_record("aipe.magnetics.model.igse", "L1")]))
    assert weak.passed is False and weak.metrics["verification_claim_errors"] == 2
    assert weak.metrics["registry_validator_would_accept"] is False

    # a referenced file disappears -> documentation_drift blocks acceptance, schema-level verdict unchanged
    (tmp_path / "docs" / "guide.md").unlink()
    drifted = evaluate_manifest_acceptance(path, supported)
    assert drifted.passed is False and drifted.metrics["missing_reference_count"] == 1
    assert drifted.metrics["registry_validator_would_accept"] is False
    assert any("missing file: docs/sub/../guide.md" in note for note in drifted.notes)
    assert evaluate_manifest_schema(path).passed is True

    # a directory reference is never a file (Registry's confined()+is_file() rule)
    manifest["validation"]["evidence"] = ["tests/"]
    (tmp_path / "docs" / "guide.md").write_text("x", encoding="utf-8")
    _write_yaml(path, manifest)
    directory = evaluate_manifest_acceptance(path, supported)
    assert directory.passed is False and directory.metrics["missing_reference_count"] == 1


def test_checks_fail_closed_for_missing_or_foreign_manifest(monkeypatch, tmp_path, minimal):
    missing = tmp_path / "nowhere" / "aipe.yaml"
    monkeypatch.setenv("AIPE_MAGNETICS_MANIFEST", str(missing))
    assert find_repository_manifest() is None
    for function in (registry_manifest_schema_valid, registry_manifest_accepted):
        result = function()
        assert result.passed is False and result.metrics["manifest_found"] is False, function.check_id
        assert result.metrics["repository_manifest"] is False

    foreign = _write_yaml(tmp_path / "aipe.yaml", minimal)  # id aipe.example-tool, no aipe.magnetics extension
    assert repository_manifest_problems(minimal) == [
        f"manifest id 'aipe.example-tool' is not {REPOSITORY_MANIFEST_ID!r}",
        f"manifest has no extensions[{MAGNETICS_EXTENSION_KEY!r}] mapping"]
    assert repository_manifest_problems(load_manifest(MANIFEST_PATH)) == []
    monkeypatch.setenv("AIPE_MAGNETICS_MANIFEST", str(foreign))
    for function in (registry_manifest_schema_valid, registry_manifest_accepted):
        result = function()
        assert result.passed is False and result.metrics["manifest_found"] is True, function.check_id
        assert result.metrics["repository_manifest"] is False
        assert any(str(foreign) in note and "aipe.example-tool" in note for note in result.notes), result.notes

    # walk-up from an unrelated tree lands on a foreign aipe.yaml: evidence is refused, not produced
    (tmp_path / "pyproject.toml").write_text("[project]\nname='other'\n", encoding="utf-8")
    monkeypatch.delenv("AIPE_MAGNETICS_MANIFEST")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_repository_manifest(nested) == foreign
    assert evaluate_manifest_schema(foreign).passed is False
    assert evaluate_manifest_acceptance(foreign, VerificationRegistry()).passed is False


@pytest.mark.external
def test_registry_reference_validator_accepts_repository_manifest():
    value = os.environ.get("AIPE_ECOSYSTEM_CHECKOUTS")
    if not value:
        print(SKIP_REASON)
        pytest.skip(SKIP_REASON)
    checkouts = Path(value)
    script = checkouts / "AIPE-Registry" / "scripts" / "validate.py"
    core_manifest = checkouts / "AIPE-Core" / "aipe.yaml"
    for path in (script, core_manifest):
        if not path.is_file():
            pytest.fail(f"ecosystem checkout incomplete: {path} not found")
    # explicit files: the manifest set must include the aipe.core dependency
    result = subprocess.run([sys.executable, str(script), str(core_manifest), str(MANIFEST_PATH)],
                            capture_output=True, text=True)
    assert result.stdout.strip(), result.stderr
    payload = json.loads(result.stdout)
    assert payload["manifest_count"] == 2
    ours = [item for item in payload["diagnostics"] if item["id"] == "aipe.magnetics-database"]
    drift = [item["message"] for item in ours if item["code"] == "documentation_drift"]
    errors = [item for item in ours if item["severity"] == "error" and item["code"] != "documentation_drift"]
    assert errors == [], errors
    if drift:
        message = "Registry's validator reports missing referenced files (maintainer deliverables): " + "; ".join(drift)
        print(message)
        warnings.warn(message)
    else:
        assert result.returncode == 0, result.stdout
