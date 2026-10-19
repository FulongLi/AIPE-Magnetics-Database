"""Versioned migration of magnetic records: schema 0.1.0 -> 0.2.0.

The migration is **mechanical**. It never guesses a physical value and never changes the meaning of
a field. Rules:

* ``schema_version``: ``"0.1.0"`` -> ``"0.2.0"``.
* ``id``, ``kind``, ``name``, ``synthetic``, ``properties``, ``evidence``, ``evidence_refs`` and
  ``extensions`` are carried over unchanged. Every v0.1 kind is a v0.2 kind, every v0.1 evidence
  kind is a v0.2 evidence kind, and every v0.1 property field exists in the v0.2 property model of
  that kind, with one structural exception:
  ``measurement.properties.tool`` ``{name, version}`` becomes the v0.2 ``Attribution``
  ``{id: <name>, name: <name>, version}``. v0.2 requires a tool ``id`` and the v0.1 tool *name* is the
  only identity the source holds; a label is copied, no value is invented.
* record-level ``license`` = the single distinct ``license`` of the evidence items selected by
  ``evidence_refs`` when they all agree, otherwise ``"NOASSERTION"``.
* ``provenance`` = ``{activity: "migrated from schema 0.1.0 by aipe-magnetics <version>",
  tool: {id: "aipe.magnetics", name: "AIPE Magnetics", version: <version>}, inputs: [],
  normalization: "none"}``.
* ``extensions["aipe.migration"]`` = ``{"from": "0.1.0", "to": "0.2.0", "tool_version": <version>}``;
  :func:`migrate_file` adds ``source_filename`` and ``source_sha256`` (SHA-256 of the original bytes).
* A record that already carries ``extensions["aipe.migration"]`` is refused (never re-written).

The input is validated with the **per-record v0.1 rules** before migration (v0.1 JSON Schema,
evidence rules including "measurement requires measurement evidence", finiteness - see
:func:`aipe_magnetics.data.schema.legacy_record_errors`), so a record the v0.1 validator rejects is
never "healed" by the more permissive v0.2 rules; the output is validated as a v0.2 record (schema,
semantic and physical layers of :func:`aipe_magnetics.data.schema.validate_record`) before it is
returned, so a successful call guarantees a v0.2-valid record. Cross-record references (``*_ref``)
are not resolved here because they may live in other files; validate the migrated set with
:func:`aipe_magnetics.data.schema.validate_catalogue`.

No timestamps are inserted: migration output is deterministic.

Command line::

    python -m aipe_magnetics.data.migrate <in.json> <out.json> [--overwrite]
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

from aipe_magnetics import TOOL_ID, TOOL_NAME, __version__
from aipe_magnetics.common.errors import DataIntegrityError, InvalidInputError
from aipe_magnetics.common.jsonio import dump, load, sha256_bytes
from aipe_magnetics.data.records import SCHEMA_VERSION, MagneticRecord
from aipe_magnetics.data.schema import LEGACY_SCHEMA_VERSION, legacy_record_errors, validate_record

MIGRATION_EXTENSION = "aipe.migration"
_V01_KEYS = ("schema_version", "id", "kind", "name", "synthetic", "properties", "evidence_refs", "evidence", "extensions")


def _record_license(record: dict[str, Any]) -> str:
    by_id = {item.get("id"): item for item in record.get("evidence", []) if isinstance(item, dict)}
    licenses = {str(by_id[ref]["license"]) for ref in record.get("evidence_refs", []) if ref in by_id and "license" in by_id[ref]}
    return licenses.pop() if len(licenses) == 1 else "NOASSERTION"


def _migrate_properties(kind: str, properties: dict[str, Any]) -> dict[str, Any]:
    migrated = copy.deepcopy(properties)
    if kind == "measurement":
        tool = migrated.get("tool")
        if isinstance(tool, dict) and "id" not in tool:
            # v0.1: {name, version}; v0.2 Attribution: {id, version, name}. The name is the only identity available.
            migrated["tool"] = {"id": tool["name"], "name": tool["name"], "version": tool["version"]}
    return migrated


def migrate_record(record_v01: dict[str, Any]) -> dict[str, Any]:
    """Return the v0.2 form of a v0.1 record dict (the input is not modified)."""
    if not isinstance(record_v01, dict):
        raise InvalidInputError(f"record must be a JSON object, got {type(record_v01).__name__}")
    version = record_v01.get("schema_version")
    if version == SCHEMA_VERSION:
        raise InvalidInputError("record already uses schema_version 0.2.0; nothing to migrate",
                                details={"id": record_v01.get("id")})
    if version != LEGACY_SCHEMA_VERSION:
        raise InvalidInputError(f"migrate_record expects schema_version {LEGACY_SCHEMA_VERSION!r}, got {version!r}",
                                details={"schema_version": version})
    legacy_problems = legacy_record_errors(record_v01)
    if legacy_problems:
        raise DataIntegrityError(f"v0.1 record {record_v01.get('id')!r} is not valid under the schema 0.1.0 rules: "
                                 + "; ".join(legacy_problems), details={"errors": legacy_problems})
    extensions = copy.deepcopy(record_v01.get("extensions", {}))
    if MIGRATION_EXTENSION in extensions:
        raise InvalidInputError(f"record {record_v01['id']!r} already carries extensions[{MIGRATION_EXTENSION!r}]; "
                                "refusing to migrate twice", details={"id": record_v01["id"]})
    unknown = sorted(set(record_v01) - set(_V01_KEYS))
    if unknown:  # cannot happen for a schema-valid v0.1 record (additionalProperties: false), kept as a guard
        raise DataIntegrityError(f"unexpected top-level keys in v0.1 record: {unknown}", details={"keys": unknown})
    extensions[MIGRATION_EXTENSION] = {"from": LEGACY_SCHEMA_VERSION, "to": SCHEMA_VERSION, "tool_version": __version__}
    migrated: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": record_v01["id"],
        "kind": record_v01["kind"],
        "name": record_v01["name"],
        "synthetic": record_v01["synthetic"],
        "license": _record_license(record_v01),
        "properties": _migrate_properties(record_v01["kind"], record_v01["properties"]),
        "evidence_refs": list(record_v01["evidence_refs"]),
        "evidence": copy.deepcopy(record_v01["evidence"]),
        "provenance": {
            "activity": f"migrated from schema {LEGACY_SCHEMA_VERSION} by aipe-magnetics {__version__}",
            "tool": {"id": TOOL_ID, "name": TOOL_NAME, "version": __version__},
            "inputs": [],
            "normalization": "none",
        },
        "extensions": extensions,
    }
    problems = validate_record(migrated)  # schema + semantic + physical layers
    if problems:
        raise DataIntegrityError(f"migrated record {migrated['id']!r} does not validate under schema {SCHEMA_VERSION}: "
                                 + "; ".join(problems), details={"errors": problems})
    try:
        MagneticRecord.model_validate(migrated)
    except ValidationError as error:  # pragma: no cover - semantic_errors above already covers this
        raise DataIntegrityError(str(error)) from None
    return migrated


def migrate_file(src: str | Path, dst: str | Path, overwrite: bool = False) -> Path:
    """Migrate the v0.1 record file ``src`` to the v0.2 file ``dst`` (refuses to overwrite unless asked)."""
    src, dst = Path(src), Path(dst)
    raw = src.read_bytes()
    record = load(src)
    if src.resolve() == dst.resolve():
        raise InvalidInputError("source and destination are the same file; the original must be preserved",
                                details={"path": str(src)})
    migrated = migrate_record(record)
    migrated["extensions"][MIGRATION_EXTENSION]["source_filename"] = src.name
    migrated["extensions"][MIGRATION_EXTENSION]["source_sha256"] = sha256_bytes(raw)
    return dump(migrated, dst, overwrite=overwrite)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aipe_magnetics.data.migrate",
                                     description="Migrate a v0.1 magnetic record file to schema 0.2.0.")
    parser.add_argument("source", type=Path, help="v0.1 record file")
    parser.add_argument("destination", type=Path, help="v0.2 output file (not overwritten unless --overwrite)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        path = migrate_file(args.source, args.destination, overwrite=args.overwrite)
    except (OSError, ValueError) as error:
        print(f"ERROR {error}")
        return 1
    print(f"migrated {args.source} -> {path} (schema {LEGACY_SCHEMA_VERSION} -> {SCHEMA_VERSION})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
