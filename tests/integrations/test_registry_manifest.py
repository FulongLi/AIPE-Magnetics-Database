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
    validate_magnetics_extension,
    validate_manifest,
    verify_vendored_integrity,
)
from aipe_magnetics.verification.checks.check_integrations import registry_manifest_valid
from aipe_magnetics.verification.evidence import EvidenceRecord, VerificationRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "aipe.yaml"
SKIP_REASON = "AIPE_ECOSYSTEM_CHECKOUTS not set"


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
    with pytest.raises(DataNotFoundError):
        load_manifest(tmp_path / "missing.yaml")
    sequence = tmp_path / "list.yaml"
    sequence.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(InvalidInputError, match="mapping"):
        load_manifest(sequence)
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


@pytest.mark.parametrize("reference,escapes", [
    ("docs/a.md", False), ("docs/sub/../a.md", True), ("../private.txt", True), ("%2e%2e/private.txt", True),
    ("/etc/passwd", True), ("C:/x", True), ("docs\\a.md", True), ("", True), (".", True), ("https://x/../y", True),
])
def test_confined_reference(reference, escapes, tmp_path):
    if escapes:
        with pytest.raises(InvalidInputError):
            confined_reference(reference)
            confined_reference(reference, tmp_path)
    else:
        assert confined_reference(reference) is None
        assert confined_reference(reference, tmp_path) == (tmp_path / reference).resolve()


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
    assert model.claimed_verification_levels() == {"aipe.magnetics.model.igse": "L2"}
    mapping = dict(extension, interfaces={"python_api": "aipe_magnetics.api", "cli": "aipe-magnetics"})
    assert MagneticsManifestExtension.model_validate(mapping).interfaces["cli"] == "aipe-magnetics"
    minimal["extensions"] = {MAGNETICS_EXTENSION_KEY: extension}
    diagnostics = validate_manifest(minimal)
    assert [item["code"] for item in diagnostics] == ["verification_claims_unchecked"]
    assert validate_manifest(minimal, evidence_registry=VerificationRegistry([EvidenceRecord(
        id="check:x", capability="aipe.magnetics.model.igse", level="L2", kind="software", status="pass",
        description="s", method="s", check_id="s", tool_version="0")])) == []
    assert validate_magnetics_extension({"extensions": {}}) == [] and validate_magnetics_extension({}) == []


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
    assert codes == ["documentation_drift", "standalone_operation", "unsafe_reference", "verification_claim"]
    broken = copy.deepcopy(minimal)
    broken["extensions"][MAGNETICS_EXTENSION_KEY]["platforms"] = "linux"
    errors = manifest_errors(validate_manifest(broken))
    assert errors and all(item["code"] == "magnetics_extension" for item in errors)
    assert errors[0]["message"].startswith(f"extensions/{MAGNETICS_EXTENSION_KEY}/platforms")


# --------------------------------------------------------------------------------------------------
# Verification check and external reference validator
# --------------------------------------------------------------------------------------------------
def test_registry_check_passes():
    result = registry_manifest_valid()
    assert result.passed, (result.metrics, result.notes)
    assert result.metrics["manifest_found"] is True and result.metrics["mutation_rejected"] is True
    for note in result.notes:
        print(note)


def test_registry_check_fails_closed_without_manifest(monkeypatch, tmp_path):
    missing = tmp_path / "nowhere" / "aipe.yaml"
    monkeypatch.setenv("AIPE_MAGNETICS_MANIFEST", str(missing))
    assert find_repository_manifest() is None
    result = registry_manifest_valid()
    assert result.passed is False and result.metrics["manifest_found"] is False


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
