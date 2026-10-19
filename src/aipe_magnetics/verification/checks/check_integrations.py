"""L1 verification checks for the AIPE-Core / AIPE-Registry compatibility adapters.

* ``integration.core-example-validates`` (schema): the vendored Core schemas are unmodified pinned
  copies, resolve offline, accept Core's own synthetic DAB example and reject mutated states.
* ``integration.core-roundtrip`` (software): an illustrative transformer summary with assumption,
  source and analytical evidence exports to a contract-valid Engineering State, imports back equal,
  preserves foreign extensions when extending a base state and re-exports idempotently (evidence,
  requirements and validation checks included).
* ``integration.registry-manifest-schema-valid`` (schema): the repository ``aipe.yaml`` (and only
  this repository's manifest, identified by its id and the ``aipe.magnetics`` extension) parses
  under Registry's loader rules, validates against the vendored Registry schema and Registry's
  manifest-level semantic rules that need no file system (member ids, connectors, path confinement,
  licence/status/dependency format, the extension model), and a mutated manifest is rejected.
* ``integration.registry-manifest-accepted`` (software): AIPE-Registry's own validator would accept
  the manifest for this checkout, i.e. no error at all including ``documentation_drift`` for
  referenced files that do not exist, *and* every verification level claimed in
  ``extensions.aipe.magnetics.model_maturity`` is supported by the bundled evidence registry. The
  check fails closed while referenced files are missing or while the claims cannot be cross-checked
  (no bundled evidence file yet); it never emits a passing record on the strength of a validator
  run that ignored those conditions. Registry-level checks that need the complete manifest set
  (dependency resolution against the ``aipe.core`` manifest) are outside this check.

The fixture values below are illustrative design inputs for a 10 kW, 800 V to 400 V, 100 kHz
dual-active-bridge transformer (turns ratio from the ideal transformer relation n = V1/V2 = 2);
they are labelled synthetic in the exported evidence and are not datasheet or measured data.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from referencing.exceptions import NoSuchResource

from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import SourceReference
from aipe_magnetics.integrations.aipe_core.export import (
    EXTENSION_KEY,
    CoreEvidence,
    CoreValidationCheck,
    DesignRequirements,
    MagneticComponentSummary,
    from_engineering_state,
    to_engineering_state,
    tool_attribution,
)
from aipe_magnetics.integrations.aipe_core.state import (
    CORE_COMMIT,
    CORE_REPOSITORY,
    CORE_SCHEMA_BASE,
    core_schema_ids,
    load_core_example,
    load_core_schemas,
    validate_engineering_state,
    verify_vendored_integrity,
)
from aipe_magnetics.integrations.aipe_registry.manifest import (
    MAGNETICS_EXTENSION_KEY,
    REGISTRY_COMMIT,
    REGISTRY_REPOSITORY,
    REPOSITORY_MANIFEST_ID,
    MagneticsManifestExtension,
    find_repository_manifest,
    load_manifest,
    manifest_schema,
    repository_manifest_problems,
    validate_manifest,
)
from aipe_magnetics.integrations.aipe_registry.manifest import verify_vendored_integrity as verify_registry_integrity
from aipe_magnetics.verification.evidence import OperatingRange
from aipe_magnetics.verification.suite import CheckResult, bundled_evidence_path, check, load_bundled_registry

CORE_CAPABILITY = "aipe.magnetics.integration.aipe-core"
REGISTRY_CAPABILITY = "aipe.magnetics.integration.aipe-registry"
FIXTURE_TIMESTAMP = "2026-10-19T00:00:00Z"  # authored fixture time, not build time

CORE_REFERENCE = SourceReference(
    id="src.aipe-core", kind="repository", reference="AIPE-Core: versioned SI Engineering State schemas, "
    "evidence conventions and offline contract validator (schemas/, scripts/validate.py, examples/dab-10kw)",
    url=CORE_REPOSITORY, version=CORE_COMMIT, license="MIT",
    note="Schemas and example vendored byte-exact; scripts/validate.py::validate_state reproduced under MIT "
         "attribution (src/aipe_magnetics/integrations/aipe_core/schemas/v0.1.0/NOTICE.md).")
REGISTRY_REFERENCE = SourceReference(
    id="src.aipe-registry", kind="repository", reference="AIPE-Registry: capability manifest schema v0.1 and "
    "manifest validation rules (schema/capability-manifest.schema.json, scripts/registry.py)",
    url=REGISTRY_REPOSITORY, version=REGISTRY_COMMIT, license="MIT",
    note="Schema vendored byte-exact; loader and semantic rules re-implemented under MIT attribution "
         "(src/aipe_magnetics/integrations/aipe_registry/schema/NOTICE.md).")
NO_PHYSICAL_RANGE = OperatingRange(note="Schema/software interoperability check; no physical operating range applies.")


# --------------------------------------------------------------------------------------------------
# Illustrative fixture shared with tests/integrations
# --------------------------------------------------------------------------------------------------
def illustrative_dab_transformer() -> dict[str, Any]:
    """Synthetic 10 kW DAB transformer summary, requirements and an analytical check (illustrative only)."""
    tool = tool_attribution()
    brief = CoreEvidence(
        id="evidence.magnetics-design-brief", kind="assumption",
        description="Illustrative DAB transformer design inputs: 10 kW, 800 V to 400 V, 100 kHz. Synthetic "
                    "fixture values; not a datasheet, simulation or measurement.",
        source={"type": "synthetic_example", "reference": "AIPE Magnetics v0.2 integration fixture (illustrative)"},
        provenance={"created_at": FIXTURE_TIMESTAMP, "activity": "Record illustrative design inputs for the "
                    "round-trip check; no engineering calculation claimed.", "inputs": []})
    inductances = CoreEvidence(
        id="evidence.inductance-assumptions", kind="assumption",
        description="Illustrative magnetising inductance 850 uH and leakage inductance 12 uH assumed for the "
                    "fixture; not derived from a core or winding design.",
        source={"type": "synthetic_example", "reference": "AIPE Magnetics v0.2 integration fixture (illustrative)"},
        provenance={"created_at": FIXTURE_TIMESTAMP, "activity": "State fixture assumptions", "inputs": []})
    material = CoreEvidence(
        id="evidence.material-reference", kind="source",
        description="Core material referenced by record id only; no material properties are reproduced here.",
        source={"type": "repository", "reference": "AIPE Magnetics material record example-synthetic-ferrite",
                "locator": "core_material_ref"},
        provenance={"created_at": FIXTURE_TIMESTAMP, "activity": "Cite the material record", "inputs": []})
    turns_ratio = CoreEvidence(
        id="evidence.turns-ratio-analytical", kind="analytical",
        description="Ideal transformer relation n = V1/V2 = 800 V / 400 V = 2 (unity voltage-transfer DAB "
                    "operating point); neglects leakage drop and dead time.",
        source={"type": "document", "reference": "Ideal transformer relation N1/N2 = V1/V2 (textbook)"},
        provenance={"created_at": FIXTURE_TIMESTAMP, "activity": "Analytical turns-ratio estimate",
                    "inputs": ["evidence.magnetics-design-brief"], "tool": tool})
    summary = MagneticComponentSummary(
        id="magnetics.dab-transformer", role="DAB isolation transformer", kind="transformer",
        design_status="candidate", inductance=Quantity.si(850, "uH"), leakage_inductance=Quantity.si(12, "uH"),
        turns_ratio=Quantity(value=2.0, unit="1"), core_material_ref="material:example-synthetic-ferrite",
        database_ref="records/example-dab-transformer.json",
        evidence=[brief, inductances, material, turns_ratio],
        limitations=["Inductances are illustrative assumptions, not design results or measurements.",
                     "No core, winding, loss, thermal or hardware verification is claimed by this fixture."],
        detail={"fixture": "illustrative", "turns": {"primary": 16, "secondary": 8},
                "flux_density_peak_assumed": {"value": 0.1, "unit": "T"}})
    requirements = DesignRequirements(
        nominal_power=Quantity.si(10, "kW"), input_voltage=Quantity(value=800, unit="V"),
        output_voltage=Quantity(value=400, unit="V"), switching_frequency=Quantity.si(100, "kHz"),
        evidence=[CoreEvidence(
            id="evidence.design-inputs", kind="assumption",
            description="Requested converter design inputs (synthetic fixture).",
            source={"type": "user_requirement", "reference": "AIPE Magnetics v0.2 integration fixture"},
            provenance={"created_at": FIXTURE_TIMESTAMP, "activity": "Record requested inputs", "inputs": []})])
    checks = [CoreValidationCheck(id="check.turns-ratio-consistency", status="pass",
                                  description="Turns ratio equals the requested voltage ratio (analytical).",
                                  evidence_refs=["evidence.turns-ratio-analytical"])]
    return {"summary": summary, "requirements": requirements, "checks": checks, "created_at": FIXTURE_TIMESTAMP,
            "project_id": "project.dab-10kw-fixture"}


def export_fixture_state(fixture: dict[str, Any] | None = None) -> dict[str, Any]:
    fixture = fixture or illustrative_dab_transformer()
    return to_engineering_state(fixture["summary"], created_at=fixture["created_at"], project_id=fixture["project_id"],
                                requirements=fixture["requirements"], checks=fixture["checks"],
                                project_name="Illustrative 10 kW DAB transformer fixture", project_purpose="example",
                                converter_topology="dual-active-bridge")


# --------------------------------------------------------------------------------------------------
# Core checks
# --------------------------------------------------------------------------------------------------
@check(
    id="integration.core-example-validates", capability=CORE_CAPABILITY, level="L1", kind="schema",
    description="Vendored AIPE-Core 0.1.0 schemas match their pinned upstream hashes, resolve fully offline, accept "
                "Core's synthetic DAB example without errors and reject representative contract violations.",
    method="SHA-256 of the 14 schemas and the example against pins; referencing.Registry retrieve must raise "
           "NoSuchResource for an unknown id; validate_engineering_state(example) == []; three mutations "
           "(wrong unit dimension, missing evidence, provenance cycle) must each produce errors.",
    range=NO_PHYSICAL_RANGE, reference=CORE_REFERENCE, tags=("integration", "core", "schema"),
)
def core_example_validates() -> CheckResult:
    digests = verify_vendored_integrity()
    schema_ids = core_schema_ids()
    example_errors = validate_engineering_state(load_core_example())
    offline = False
    try:
        load_core_schemas().get_or_retrieve(CORE_SCHEMA_BASE + "does-not-exist.schema.json")
    except NoSuchResource:
        offline = True

    def wrong_unit(state: dict[str, Any]) -> None:
        state["requirements"]["input_voltage"]["unit"] = "W"

    def missing_evidence(state: dict[str, Any]) -> None:
        state["requirements"]["evidence_refs"] = ["evidence.missing"]

    def cycle(state: dict[str, Any]) -> None:
        state["evidence"][0]["provenance"]["inputs"] = ["evidence.design-brief"]

    rejected: dict[str, int] = {}
    for mutate in (wrong_unit, missing_evidence, cycle):
        mutated = load_core_example()
        mutate(mutated)
        rejected[mutate.__name__] = len(validate_engineering_state(mutated))
    passed = (len(digests) == 15 and len(schema_ids) == 14 and not example_errors and offline
              and all(count > 0 for count in rejected.values()))
    notes = [f"Core commit {CORE_COMMIT}; example id aipe.example.dab-10kw."]
    if example_errors:
        notes.extend(f"example error: {error}" for error in example_errors)
    return CheckResult(passed=passed, metrics={"vendored_files": len(digests), "schema_ids": len(schema_ids),
                                               "example_errors": len(example_errors), "offline_registry": offline,
                                               "mutation_error_counts": rejected}, notes=notes)


@check(
    id="integration.core-roundtrip", capability=CORE_CAPABILITY, level="L1", kind="software",
    description="An illustrative transformer summary with assumption, source and analytical evidence exports to a "
                "contract-valid Core Engineering State 0.1.0, imports back field-for-field equal, keeps foreign "
                "extensions when extending Core's DAB example and re-exports idempotently (the identical summary, "
                "requirements and checks exported onto the state's own output leave it unchanged).",
    method="to_engineering_state -> validate_engineering_state == [] -> from_engineering_state == [summary]; "
           "validation.status is analytically_checked only with the supplied passing analytical check; the same "
           "summary extended onto the vendored example preserves extensions['example.custom']; the second export "
           "of summary + requirements + checks onto the first output (replace_existing=True) is equal to the first "
           "output, and the second export onto the extended example is unchanged too.",
    range=NO_PHYSICAL_RANGE, reference=CORE_REFERENCE, tags=("integration", "core", "roundtrip"),
)
def core_roundtrip() -> CheckResult:
    fixture = illustrative_dab_transformer()
    summary = fixture["summary"]
    state = export_fixture_state(fixture)
    errors = validate_engineering_state(state)
    imported = from_engineering_state(state)
    equal = imported == [summary]
    status_ok = state["validation"]["status"] == "analytically_checked"
    plain = to_engineering_state(summary, created_at=fixture["created_at"], project_id=fixture["project_id"],
                                 requirements=fixture["requirements"], project_purpose="example")
    plain_not_run = plain["validation"]["status"] == "not_run"
    extension_kept = state["magnetics"][0]["extensions"][EXTENSION_KEY]["detail"] == summary.detail
    again_state = to_engineering_state(summary, created_at=fixture["created_at"], base_state=state,
                                       requirements=fixture["requirements"], checks=fixture["checks"],
                                       replace_existing=True)
    idempotent_full = again_state == state

    base = load_core_example()
    base["extensions"] = {"example.custom": {"independent_domain": 3}}
    extended = to_engineering_state(summary, created_at=fixture["created_at"], base_state=base)
    base_errors = validate_engineering_state(extended)
    foreign_kept = extended["extensions"] == {"example.custom": {"independent_domain": 3}}
    base_untouched = base["magnetics"] == [] and len(base["evidence"]) == 1
    again = to_engineering_state(summary, created_at=fixture["created_at"], base_state=extended, replace_existing=True)
    idempotent_extended = again == extended

    passed = (not errors and equal and status_ok and plain_not_run and extension_kept and idempotent_full
              and not base_errors and foreign_kept and base_untouched and idempotent_extended)
    notes = [f"Exported state id {state['id']} with {len(state['evidence'])} evidence records; "
             f"validation.status={state['validation']['status']}."]
    notes.extend(f"export error: {error}" for error in errors + base_errors)
    return CheckResult(passed=passed, metrics={
        "export_errors": len(errors), "roundtrip_equal": equal, "status_analytically_checked": status_ok,
        "status_not_run_without_check": plain_not_run, "extension_detail_kept": extension_kept,
        "idempotent_reexport": idempotent_full, "base_state_errors": len(base_errors),
        "foreign_extension_kept": foreign_kept, "idempotent_reexport_on_example": idempotent_extended,
        "evidence_records": len(state["evidence"])}, notes=notes)


# --------------------------------------------------------------------------------------------------
# Registry checks
# --------------------------------------------------------------------------------------------------
def _repository_manifest() -> tuple[Path | None, dict[str, Any] | None, list[str]]:
    """Locate and load the repository manifest; the third element says why no evidence may be produced."""
    path = find_repository_manifest()
    if path is None:
        return None, None, ["aipe.yaml not found: run from a source checkout or set AIPE_MAGNETICS_MANIFEST."]
    manifest = load_manifest(path)
    problems = repository_manifest_problems(manifest)
    return path, manifest, [f"{path} is not this repository's manifest ({REPOSITORY_MANIFEST_ID}): {problem}"
                            for problem in problems]


def _claimed_levels(manifest: dict[str, Any]) -> dict[str, str]:
    extension = (manifest.get("extensions") or {}).get(MAGNETICS_EXTENSION_KEY)
    try:
        return MagneticsManifestExtension.model_validate(extension).claimed_verification_levels()
    except ValidationError:
        return {}


def evaluate_manifest_schema(manifest_path: Path, manifest: dict[str, Any] | None = None) -> CheckResult:
    """Schema-level verdict for ``manifest_path`` (no file-system or evidence dependence)."""
    manifest = load_manifest(manifest_path) if manifest is None else manifest
    problems = repository_manifest_problems(manifest)
    if problems:
        return CheckResult(passed=False, metrics={"manifest_found": True, "repository_manifest": False},
                           notes=[f"{manifest_path} is not this repository's manifest ({REPOSITORY_MANIFEST_ID}): "
                                  f"{problem}" for problem in problems])
    diagnostics = validate_manifest(manifest)
    errors = [item for item in diagnostics if item["severity"] == "error"]
    warnings = [item for item in diagnostics if item["severity"] == "warning"]
    mutated = copy.deepcopy(manifest)
    mutated.setdefault("compatibility", {})["integration"] = "certified"
    rejected = [item for item in validate_manifest(mutated) if item["severity"] == "error"]
    notes = [f"Registry commit {REGISTRY_COMMIT}; manifest {manifest_path} id {manifest.get('id')!r} with "
             f"{len(manifest.get('capabilities', []))} capabilities and {len(_claimed_levels(manifest))} "
             "verification-level claim(s) (claims are judged by integration.registry-manifest-accepted)."]
    notes.extend(f"error [{item['code']}] {item['message']}" for item in errors)
    notes.extend(f"warning [{item['code']}] {item['message']}" for item in warnings)
    return CheckResult(passed=not errors and bool(rejected), metrics={
        "manifest_found": True, "repository_manifest": True, "errors": len(errors), "warnings": len(warnings),
        "mutation_rejected": bool(rejected), "capabilities": len(manifest.get("capabilities", []))}, notes=notes)


def evaluate_manifest_acceptance(manifest_path: Path, evidence_registry: Any | None,
                                 manifest: dict[str, Any] | None = None) -> CheckResult:
    """Acceptance verdict: no Registry error for the checkout at ``manifest_path.parent`` and claims cross-checked.

    ``evidence_registry`` is the bundled :class:`~aipe_magnetics.verification.evidence.VerificationRegistry`
    or ``None`` when no bundled evidence exists; in the latter case the manifest's verification
    claims cannot be checked and the verdict is a fail (``verification_claims_unchecked``).
    """
    manifest = load_manifest(manifest_path) if manifest is None else manifest
    problems = repository_manifest_problems(manifest)
    if problems:
        return CheckResult(passed=False, metrics={"manifest_found": True, "repository_manifest": False},
                           notes=[f"{manifest_path} is not this repository's manifest ({REPOSITORY_MANIFEST_ID}): "
                                  f"{problem}" for problem in problems])
    repo_root = manifest_path.resolve().parent
    diagnostics = validate_manifest(manifest, repo_root, evidence_registry=evidence_registry)
    errors = [item for item in diagnostics if item["severity"] == "error"]
    warnings = [item for item in diagnostics if item["severity"] == "warning"]
    drift = [item for item in errors if item["code"] == "documentation_drift"]
    claim_errors = [item for item in errors if item["code"] == "verification_claim"]
    unchecked = any(item["code"] == "verification_claims_unchecked" for item in warnings)
    claims = _claimed_levels(manifest)
    would_accept = not errors
    passed = would_accept and not unchecked
    notes = [f"Registry commit {REGISTRY_COMMIT}; manifest {manifest_path} validated for repository root {repo_root}; "
             f"{len(claims)} verification-level claim(s) in extensions.{MAGNETICS_EXTENSION_KEY}.model_maturity."]
    if unchecked:
        notes.append("BLOCKING: verification-level claims could not be cross-checked because no bundled evidence "
                     "registry exists (run the verification suite with --update-bundled, then re-run this check).")
    notes.extend(f"BLOCKING error [{item['code']}] {item['message']}" for item in errors)
    notes.extend(f"warning [{item['code']}] {item['message']}" for item in warnings
                 if item["code"] != "verification_claims_unchecked")
    return CheckResult(passed=passed, metrics={
        "manifest_found": True, "repository_manifest": True, "registry_validator_would_accept": would_accept,
        "verification_claims_unchecked": unchecked, "verification_claims": len(claims),
        "verification_claim_errors": len(claim_errors), "missing_reference_count": len(drift),
        "errors": len(errors), "warnings": len(warnings), "bundled_evidence_consulted": evidence_registry is not None},
        notes=notes)


@check(
    id="integration.registry-manifest-schema-valid", capability=REGISTRY_CAPABILITY, level="L1", kind="schema",
    description="This repository's aipe.yaml (id aipe.magnetics-database with the aipe.magnetics extension) parses "
                "under Registry's loader rules, validates against the vendored AIPE-Registry capability-manifest "
                "schema v0.1 and Registry's manifest-level semantic rules that need no file system, and its "
                "extensions['aipe.magnetics'] matches MagneticsManifestExtension; a mutated manifest is rejected.",
    method="verify vendored schema hash; find_repository_manifest() must name a manifest whose id is "
           "aipe.magnetics-database and that carries extensions['aipe.magnetics'] (a foreign manifest fails closed); "
           "load_manifest with UniqueLoader; validate_manifest(manifest) without repo_root must yield no error; a "
           "manifest mutated to compatibility.integration='certified' must be rejected. File existence and "
           "verification claims are judged by integration.registry-manifest-accepted.",
    range=NO_PHYSICAL_RANGE, reference=REGISTRY_REFERENCE, tags=("integration", "registry", "schema"),
)
def registry_manifest_schema_valid() -> CheckResult:
    digests = verify_registry_integrity()
    manifest_schema()  # raises if the vendored schema is not a valid 2020-12 schema
    path, manifest, problems = _repository_manifest()
    if path is None or manifest is None or problems:
        return CheckResult(passed=False, metrics={"vendored_files": len(digests), "manifest_found": path is not None,
                                                  "repository_manifest": False}, notes=problems)
    result = evaluate_manifest_schema(path, manifest)
    result.metrics["vendored_files"] = len(digests)
    return result


@check(
    id="integration.registry-manifest-accepted", capability=REGISTRY_CAPABILITY, level="L1", kind="software",
    description="AIPE-Registry's own validator would accept this repository's aipe.yaml for this checkout (no error, "
                "including documentation_drift for referenced files that do not exist) and every verification level "
                "claimed in extensions.aipe.magnetics.model_maturity is supported by the bundled evidence registry. "
                "Fails closed while referenced files are missing or while the claims cannot be cross-checked.",
    method="validate_manifest(manifest, repo_root=<checkout>, evidence_registry=<bundled v<version>.json or None>) "
           "must report zero errors and must not report verification_claims_unchecked; metrics expose "
           "registry_validator_would_accept, verification_claims_unchecked and missing_reference_count. Cross-manifest "
           "Registry checks (dependency resolution against aipe.core) are outside this check.",
    range=NO_PHYSICAL_RANGE, reference=REGISTRY_REFERENCE, tags=("integration", "registry", "acceptance"),
)
def registry_manifest_accepted() -> CheckResult:
    verify_registry_integrity()
    path, manifest, problems = _repository_manifest()
    if path is None or manifest is None or problems:
        return CheckResult(passed=False, metrics={"manifest_found": path is not None, "repository_manifest": False},
                           notes=problems)
    evidence_registry = load_bundled_registry() if bundled_evidence_path().exists() else None
    return evaluate_manifest_acceptance(path, evidence_registry, manifest)


__all__ = ["core_example_validates", "core_roundtrip", "evaluate_manifest_acceptance", "evaluate_manifest_schema",
           "export_fixture_state", "illustrative_dab_transformer", "registry_manifest_accepted",
           "registry_manifest_schema_valid"]
