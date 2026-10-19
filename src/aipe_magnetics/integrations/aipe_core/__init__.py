"""AIPE-Core Engineering State 0.1.0 compatibility: offline schemas, contract validation, round-trip adapter.

* :mod:`~aipe_magnetics.integrations.aipe_core.state` - vendored schemas (pinned commit, MIT; see
  ``schemas/v0.1.0/NOTICE.md``), :func:`load_core_schemas`, :func:`validate_engineering_state` and
  the synthetic DAB example :func:`load_core_example`.
* :mod:`~aipe_magnetics.integrations.aipe_core.export` - :class:`MagneticComponentSummary` and the
  executable round trip :func:`to_engineering_state` / :func:`from_engineering_state` /
  :func:`extract_requirements`.

Command line: ``python -m aipe_magnetics.integrations.aipe_core validate <state.json> [...]``.
"""

from aipe_magnetics.integrations.aipe_core.export import (
    DEFAULT_LIMITATION,
    DETAIL_SCHEMA,
    EXTENSION_KEY,
    CoreArtifact,
    CoreAttribution,
    CoreEvidence,
    CoreProvenance,
    CoreSource,
    CoreValidationCheck,
    DesignRequirements,
    MagneticComponentSummary,
    extract_requirements,
    from_engineering_state,
    to_engineering_state,
    tool_attribution,
)
from aipe_magnetics.integrations.aipe_core.state import (
    CORE_COMMIT,
    CORE_REPOSITORY,
    CORE_SCHEMA_BASE,
    CORE_SCHEMA_VERSION,
    EXAMPLE_PATH,
    SCHEMA_DIR,
    STATE_SCHEMA_ID,
    VENDORED_SHA256,
    core_schema_ids,
    load_core_example,
    load_core_schemas,
    load_state,
    state_validator,
    validate_engineering_state,
    verify_vendored_integrity,
)

__all__ = [
    "CORE_COMMIT", "CORE_REPOSITORY", "CORE_SCHEMA_BASE", "CORE_SCHEMA_VERSION", "DEFAULT_LIMITATION", "DETAIL_SCHEMA",
    "EXAMPLE_PATH", "EXTENSION_KEY", "SCHEMA_DIR", "STATE_SCHEMA_ID", "VENDORED_SHA256",
    "CoreArtifact", "CoreAttribution", "CoreEvidence", "CoreProvenance", "CoreSource", "CoreValidationCheck",
    "DesignRequirements", "MagneticComponentSummary",
    "core_schema_ids", "extract_requirements", "from_engineering_state", "load_core_example", "load_core_schemas",
    "load_state", "state_validator", "to_engineering_state", "tool_attribution", "validate_engineering_state",
    "verify_vendored_integrity",
]
