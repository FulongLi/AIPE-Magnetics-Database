"""AIPE-Registry capability manifest compatibility (vendored schema, loader rules, diagnostics)."""

from aipe_magnetics.integrations.aipe_registry.manifest import (
    MAGNETICS_EXTENSION_KEY,
    MANIFEST_SCHEMA_VERSION,
    REGISTRY_COMMIT,
    REGISTRY_REPOSITORY,
    REPOSITORY_MANIFEST_ID,
    SCHEMA_PATH,
    VENDORED_SHA256,
    MagneticsManifestExtension,
    ModelMaturity,
    UniqueLoader,
    confined_reference,
    describe_capabilities,
    find_repository_manifest,
    load_manifest,
    manifest_errors,
    manifest_schema,
    manifest_validator,
    parse_manifest_text,
    repository_manifest_problems,
    validate_magnetics_extension,
    validate_manifest,
    verify_vendored_integrity,
)

__all__ = [
    "MAGNETICS_EXTENSION_KEY", "MANIFEST_SCHEMA_VERSION", "REGISTRY_COMMIT", "REGISTRY_REPOSITORY",
    "REPOSITORY_MANIFEST_ID", "SCHEMA_PATH", "VENDORED_SHA256", "MagneticsManifestExtension", "ModelMaturity",
    "UniqueLoader", "confined_reference", "describe_capabilities", "find_repository_manifest", "load_manifest",
    "manifest_errors", "manifest_schema", "manifest_validator", "parse_manifest_text", "repository_manifest_problems",
    "validate_magnetics_extension", "validate_manifest", "verify_vendored_integrity",
]
