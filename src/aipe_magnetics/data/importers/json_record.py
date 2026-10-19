"""Reviewable JSON record import (v0.2 port of ``importers/json_record.py``).

Accepts schema ``0.2.0`` records, and schema ``0.1.0`` records **only** when ``migrate_v01=True``
(the migration is the mechanical one of :mod:`aipe_magnetics.data.migrate`; the source file is left
untouched). The record is validated (JSON Schema + semantics + physical rules), checked against the
catalogue context (references must resolve to records of the right kind), stamped with
``extensions["aipe.import"]`` (source file name, SHA-256 of the original bytes, importer version,
``normalization: "none; source must already use explicit SI units"``) and, when a destination is
given, written without ever overwriting.

Explicit ``null`` values (anywhere outside the free-form ``extensions``) are **rejected**: the
standard omits missing values, and silently dropping or copying a ``null`` would modify or
propagate the input without a record of it.

:func:`import_record` is the path-oriented variant used by the ``aipe-magnetics import json``
command (extra catalogue directories / files instead of a :class:`Catalog`).

Command line::

    python -m aipe_magnetics.data.importers.json_record <source.json> <destination.json>
        [--catalogue DIR_OR_FILE ...] [--no-bundled-catalogue] [--migrate-v01]
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Iterable, Sequence

from aipe_magnetics.common.errors import DataIntegrityError, InvalidInputError
from aipe_magnetics.common.jsonio import loads
from aipe_magnetics.data.catalog import Catalog
from aipe_magnetics.data.importers.base import (
    IMPORT_EXTENSION,
    NO_NORMALIZATION,
    ImportResult,
    build_record,
    check_against_catalogue,
    ensure_no_import_extension,
    explicit_null_paths,
    import_extension,
    source_bytes_and_digest,
    tool_attribution,
    write_record,
)
from aipe_magnetics.data.migrate import migrate_record
from aipe_magnetics.data.records import SCHEMA_VERSION, MagneticRecord, Provenance
from aipe_magnetics.data.schema import LEGACY_SCHEMA_VERSION, validate_record

IMPORTER_ID = "aipe.magnetics.import.json_record"


def import_json_record(source: str | Path, destination: str | Path | None = None, *,
                       catalogue: Catalog | Iterable[MagneticRecord | dict[str, Any]] | None = None,
                       migrate_v01: bool = False) -> ImportResult:
    """Import one JSON record file. Nothing is written unless every validation passes."""
    source = Path(source)
    raw, digest = source_bytes_and_digest(source)
    try:
        data = loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise InvalidInputError(f"{source}: cannot parse JSON: {error}", details={"path": str(source)}) from None
    if not isinstance(data, dict):
        raise InvalidInputError(f"{source}: a record file must contain one JSON object", details={"path": str(source)})
    warnings: list[str] = []
    version = data.get("schema_version")
    if version == LEGACY_SCHEMA_VERSION:
        if not migrate_v01:
            raise InvalidInputError(f"{source}: schema {LEGACY_SCHEMA_VERSION} record; pass migrate_v01=True to migrate it "
                                    "during import (the source file is left untouched)", details={"schema_version": version})
        data = migrate_record(data)
        warnings.append(f"record migrated from schema {LEGACY_SCHEMA_VERSION} to {SCHEMA_VERSION} during import")
    elif version != SCHEMA_VERSION:
        raise InvalidInputError(f"{source}: unsupported schema_version {version!r} (supported: {LEGACY_SCHEMA_VERSION} with "
                                f"migrate_v01=True, {SCHEMA_VERSION})", details={"schema_version": version})
    nulls = explicit_null_paths(data)
    if nulls:
        raise InvalidInputError(f"{source}: explicit null values are not allowed (missing values are omitted, never null): "
                                + ", ".join(nulls[:10]) + (" ..." if len(nulls) > 10 else ""),
                                details={"paths": nulls, "path": str(source)})
    errors = validate_record(data)
    if errors:
        raise DataIntegrityError(f"{source}: " + "; ".join(errors), details={"errors": errors, "path": str(source)})
    ensure_no_import_extension(data)
    stamped = copy.deepcopy(data)
    stamped.setdefault("extensions", {})[IMPORT_EXTENSION] = import_extension(
        source_name=source.name, source_sha256=digest, importer=IMPORTER_ID, normalization=NO_NORMALIZATION)
    record = build_record(stamped)
    check_against_catalogue(record, catalogue)
    provenance = Provenance(activity=f"imported JSON record {record.id} from {source.name}", tool=tool_attribution(),
                            inputs=[record.id], normalization=NO_NORMALIZATION)
    written: list[str] = []
    if destination is not None:
        written.append(str(write_record(record, destination)))
    return ImportResult(importer=IMPORTER_ID, records=[record], artifacts=[], provenance=provenance, warnings=warnings,
                        written=written, source_sha256=digest)


def import_record(source: str | Path, destination: str | Path | None = None,
                  catalogue: Iterable[str | Path] | None = None, *, migrate: bool = False, bundled: bool = True) -> ImportResult:
    """Path-oriented variant of :func:`import_json_record` for command-line consumers (``aipe-magnetics import json``).

    ``catalogue`` lists extra catalogue directories / record files (what the ``--catalogue`` option
    gives); the bundled catalogue is the base context unless ``bundled=False``. ``migrate=True``
    accepts schema 0.1.0 records. Same guarantees as :func:`import_json_record` (nothing written
    unless every validation passes, destinations never overwritten).
    """
    paths = [] if catalogue is None else [Path(p) for p in catalogue]
    context = Catalog.load(paths, bundled=bundled)
    return import_json_record(source, destination, catalogue=context, migrate_v01=migrate)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aipe_magnetics.data.importers.json_record", description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--catalogue", nargs="*", type=Path, default=[], help="extra catalogue directories or record files")
    parser.add_argument("--no-bundled-catalogue", action="store_true", help="do not use the bundled catalogue as context")
    parser.add_argument("--migrate-v01", action="store_true", help="accept schema 0.1.0 records and migrate them")
    args = parser.parse_args(argv)
    try:
        catalogue = Catalog.load(args.catalogue, bundled=not args.no_bundled_catalogue)
        result = import_json_record(args.source, args.destination, catalogue=catalogue, migrate_v01=args.migrate_v01)
    except (OSError, ValueError, LookupError) as error:
        print(f"ERROR {error}")
        return 1
    print(f"imported {result.record.id} ({result.record.kind}) -> {result.written[0]}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    return 0


__all__ = ["IMPORTER_ID", "import_json_record", "import_record", "main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
