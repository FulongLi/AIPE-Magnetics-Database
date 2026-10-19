"""Shared importer machinery (v0.2 port of the conservative v0.1 import semantics).

The v0.1 importer (``importers/json_record.py`` at the repository root) established the rules that
every v0.2 importer keeps:

* the source file is never modified and its SHA-256 is recorded in
  ``extensions["aipe.import"]`` with the keys ``source_filename``, ``source_sha256``,
  ``importer_version`` and ``normalization`` (v0.2 adds ``importer`` and, where relevant, the
  explicit column / array mapping the caller supplied, so the normalisation is reproducible);
* an existing ``extensions["aipe.import"]`` is never replaced;
* a destination file is never overwritten (:class:`FileExistsError`) and nothing is written when
  validation fails;
* there is no unit inference - a unit is either stated in the record or given explicitly by the
  caller for every column / array.

Every importer returns an :class:`ImportResult` (records, artifacts, provenance, warnings, files
written). Records in the result are fully validated :class:`MagneticRecord` instances whose
references were resolved against the catalogue context supplied by the caller (``catalogue=None``
means the bundled default catalogue; pass ``()`` for no context).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aipe_magnetics import TOOL_ID, TOOL_NAME, __version__
from aipe_magnetics.common.errors import DataIntegrityError, InvalidInputError
from aipe_magnetics.common.jsonio import dump, sha256_bytes
from aipe_magnetics.data.catalog import Catalog
from aipe_magnetics.data.records import Artifact, Attribution, Evidence, MagneticRecord, Provenance
from aipe_magnetics.data.schema import validate_catalogue, validate_record

IMPORT_EXTENSION = "aipe.import"
NO_NORMALIZATION = "none; source must already use explicit SI units"


class ImportResult(BaseModel):
    """Outcome of one import: validated records, artifacts, provenance of the import itself, warnings."""

    model_config = ConfigDict(extra="forbid")

    importer: str = Field(description="importer id, e.g. aipe.magnetics.import.csv_loss_table")
    records: list[MagneticRecord] = Field(min_length=1)
    artifacts: list[Artifact] = Field(default_factory=list)
    provenance: Provenance
    warnings: list[str] = Field(default_factory=list)
    written: list[str] = Field(default_factory=list, description="files written by the import")
    source_sha256: str | None = None

    @property
    def record(self) -> MagneticRecord:
        return self.records[0]

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


def tool_attribution() -> Attribution:
    return Attribution(id=TOOL_ID, name=TOOL_NAME, version=__version__)


def source_bytes_and_digest(path: str | Path) -> tuple[bytes, str]:
    path = Path(path)
    if not path.is_file():
        raise InvalidInputError(f"source file not found: {path}", details={"path": str(path)})
    raw = path.read_bytes()
    return raw, sha256_bytes(raw)


def import_extension(*, source_name: str, source_sha256: str, importer: str, normalization: str,
                     extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The ``extensions["aipe.import"]`` payload (v0.1 keys plus importer id and optional mapping details)."""
    payload: dict[str, Any] = {
        "source_filename": source_name,
        "source_sha256": source_sha256,
        "importer_version": __version__,
        "importer": importer,
        "normalization": normalization,
    }
    for key, value in (extra or {}).items():
        payload[key] = copy.deepcopy(value)
    return payload


def ensure_no_import_extension(record: Mapping[str, Any] | MagneticRecord) -> None:
    extensions = record.extensions if isinstance(record, MagneticRecord) else record.get("extensions", {})
    if isinstance(extensions, Mapping) and IMPORT_EXTENSION in extensions:
        raise InvalidInputError("Existing import provenance must not be replaced "
                                f"(record already carries extensions[{IMPORT_EXTENSION!r}])")


def coerce_evidence(evidence: Evidence | Mapping[str, Any], *, label: str = "evidence") -> Evidence:
    """Validate caller-supplied evidence (never invented here)."""
    if isinstance(evidence, Evidence):
        return evidence
    if not isinstance(evidence, Mapping):
        raise InvalidInputError(f"{label}: supply an Evidence instance or a dict with id, kind, reference, description, "
                                "created_at, review_status and license")
    try:
        return Evidence.model_validate(dict(evidence))
    except ValidationError as error:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors(include_url=False))
        raise InvalidInputError(f"{label}: invalid evidence: {problems}") from None


def build_record(data: Mapping[str, Any]) -> MagneticRecord:
    """Validate a record dict built by an importer with **every** layer of :func:`validate_record`.

    JSON Schema (``[schema]``, e.g. RFC 3339 ``created_at``), normative semantics (``[semantic]``) and
    physical plausibility (``[physical]``) must all pass; failures are input problems, reported in full.
    """
    payload = dict(data)
    problems = validate_record(payload)
    if problems:
        raise InvalidInputError("import produced an invalid record: " + "; ".join(problems), details={"errors": problems})
    try:
        return MagneticRecord.model_validate(payload)
    except ValidationError as error:  # pragma: no cover - validate_record already reports these
        problems_text = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or '$'}: {e['msg']}" for e in error.errors(include_url=False))
        raise InvalidInputError(f"import produced an invalid record: {problems_text}", details={"errors": problems_text}) from None


def explicit_null_paths(node: Any, path: str = "$", *, skip_top_level: Iterable[str] = ("extensions",)) -> list[str]:
    """JSON paths of every explicit ``null`` in ``node`` (the standard omits missing values instead).

    Top-level keys in ``skip_top_level`` (default: the free-form ``extensions``) are not inspected.
    """
    found: list[str] = []
    skip = set(skip_top_level)
    if isinstance(node, Mapping):
        for key, value in node.items():
            if path == "$" and key in skip:
                continue
            child = f"{path}.{key}"
            if value is None:
                found.append(child)
            else:
                found.extend(explicit_null_paths(value, child, skip_top_level=()))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            child = f"{path}[{index}]"
            if item is None:
                found.append(child)
            else:
                found.extend(explicit_null_paths(item, child, skip_top_level=()))
    return found


def catalogue_context(catalogue: Catalog | Iterable[MagneticRecord | Mapping[str, Any]] | None) -> list[MagneticRecord | dict[str, Any]]:
    """Resolve the ``catalogue`` argument: ``None`` -> bundled default catalogue; iterable -> as given."""
    if catalogue is None:
        return list(Catalog.default().records())
    if isinstance(catalogue, Catalog):
        return list(catalogue.records())
    out: list[MagneticRecord | dict[str, Any]] = []
    for item in catalogue:
        if isinstance(item, MagneticRecord):
            out.append(item)
        elif isinstance(item, Mapping):
            out.append(dict(item))
        else:
            raise InvalidInputError(f"catalogue entries must be MagneticRecord or dict, got {type(item).__name__}")
    return out


def check_against_catalogue(record: MagneticRecord, catalogue: Catalog | Iterable[Any] | None) -> None:
    """Cross-record validation of ``record`` in its catalogue context; raises before anything is written."""
    context = catalogue_context(catalogue)
    errors = validate_catalogue([*context, record])
    if errors:
        raise DataIntegrityError("; ".join(errors), details={"errors": errors, "record": record.id})


def source_artifact(path: str | Path, *, media_type: str, sha256: str, role: str = "raw",
                    description: str | None = None, uri: str | None = None) -> Artifact:
    path = Path(path)
    return Artifact(uri=uri or path.name, media_type=media_type, role=role, sha256=sha256,  # type: ignore[arg-type]
                    size_bytes=path.stat().st_size if path.is_file() else None, description=description)


def relative_uri(target: str | Path, record_destination: str | Path | None) -> str:
    """POSIX path of ``target`` relative to the record file's directory (or just the file name).

    When no relative path exists (Windows: the two files are on different drives) the absolute
    ``file:`` URI of ``target`` is returned instead; callers report that in their warnings
    (:func:`is_file_uri`).
    """
    target = Path(target)
    if record_destination is None:
        return target.name
    base = Path(record_destination).resolve().parent
    resolved = target.resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError:
        pass
    try:
        return Path(os.path.relpath(resolved, base)).as_posix()
    except ValueError:  # different drives: no relative path exists
        return resolved.as_uri()


def is_file_uri(uri: str) -> bool:
    return uri.startswith("file:")


def write_record(record: MagneticRecord, destination: str | Path) -> Path:
    """Write a record file; refuses to overwrite (``FileExistsError``), like the v0.1 importer."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"{destination} exists; importers never overwrite")
    return dump(record.to_json_dict(), destination, overwrite=False)


__all__ = [
    "IMPORT_EXTENSION", "NO_NORMALIZATION", "ImportResult", "tool_attribution", "source_bytes_and_digest",
    "import_extension", "ensure_no_import_extension", "coerce_evidence", "build_record", "explicit_null_paths",
    "catalogue_context", "check_against_catalogue", "source_artifact", "relative_uri", "is_file_uri", "write_record",
]
