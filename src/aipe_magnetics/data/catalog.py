"""Bundled and user catalogues of v0.2 records.

:class:`Catalog` loads schema-0.2.0 record files from the packaged directory
``aipe_magnetics/data/catalog/**/*.json`` and from any extra directories or files the caller
names (user catalogues). Every file is loaded strictly (duplicate keys and non-finite numbers are
rejected), validated structurally (JSON Schema) and semantically (pydantic), and a malformed file
fails loudly with its path (:class:`DataIntegrityError`). Duplicate ids across files are refused at
load time. Nothing is inferred, repaired or clipped.

The bundled catalogue currently holds **one clearly synthetic** manufacturer / material / core
triple (``manufacturer.example-synthetic``, ``material.example-synthetic-ferrite``,
``core.example-synthetic-e42``) so that the loader, the cross-reference checks and the engineering
smoke tests have content. Every number in it is an illustrative round value with assumption evidence
and ``synthetic: true``. Real catalogue entries are added by the maintainer with datasheet or
publication evidence (see ``catalog/README.md``).

Look-ups never guess: :meth:`Catalog.get` raises :class:`DataNotFoundError`;
:meth:`Catalog.find_material` matches id, name or grade case-insensitively, raises
:class:`DataNotFoundError` with suggestions when nothing matches and :class:`InvalidInputError`
listing the candidates when the query is ambiguous.
"""

from __future__ import annotations

import difflib
import functools
from pathlib import Path
from typing import Any, Iterable, Mapping

from pydantic import ValidationError

from aipe_magnetics.common.errors import DataIntegrityError, DataNotFoundError, InvalidInputError
from aipe_magnetics.common.jsonio import load
from aipe_magnetics.data import records as _records
from aipe_magnetics.data.records import PROPERTY_MODELS, SCHEMA_VERSION, MagneticRecord
from aipe_magnetics.data.schema import LEGACY_SCHEMA_VERSION, validate_record

BUNDLED_CATALOG_DIR = Path(__file__).resolve().parent / "catalog"


def _record_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.rglob("*.json") if p.is_file())
    if path.is_file():
        return [path]
    raise DataNotFoundError(f"catalogue path does not exist: {path}", details={"path": str(path)})


def load_record_file(path: str | Path) -> MagneticRecord:
    """Strictly load and fully validate one v0.2 record file; errors always name the file."""
    path = Path(path)
    try:
        data = load(path)
    except (OSError, ValueError) as error:
        raise DataIntegrityError(f"{path}: cannot load record: {error}", details={"path": str(path)}) from None
    if not isinstance(data, dict):
        raise DataIntegrityError(f"{path}: a record file must contain one JSON object", details={"path": str(path)})
    version = data.get("schema_version")
    if version == LEGACY_SCHEMA_VERSION:
        raise DataIntegrityError(f"{path}: schema {LEGACY_SCHEMA_VERSION} record; migrate it first "
                                 f"(python -m aipe_magnetics.data.migrate {path.name} <out>)", details={"path": str(path)})
    if version != SCHEMA_VERSION:
        raise DataIntegrityError(f"{path}: unsupported schema_version {version!r} (expected {SCHEMA_VERSION})",
                                 details={"path": str(path), "schema_version": version})
    errors = validate_record(data)
    if errors:
        raise DataIntegrityError(f"{path}: invalid record: " + "; ".join(errors), details={"path": str(path), "errors": errors})
    try:
        return MagneticRecord.model_validate(data)
    except ValidationError as error:  # pragma: no cover - validate_record already reports these
        raise DataIntegrityError(f"{path}: {error}", details={"path": str(path)}) from None


class Catalog:
    """An immutable, validated collection of v0.2 records with typed look-ups."""

    def __init__(self, records: Iterable[MagneticRecord] = (), *, sources: Mapping[str, Path] | None = None):
        self._records: dict[str, MagneticRecord] = {}
        self._sources: dict[str, Path] = dict(sources or {})
        for record in records:
            if not isinstance(record, MagneticRecord):
                raise InvalidInputError(f"Catalog accepts MagneticRecord instances, got {type(record).__name__}")
            if record.id in self._records:
                where = self._sources.get(record.id)
                raise DataIntegrityError(f"duplicate record id {record.id!r}" + (f" (first seen in {where})" if where else ""),
                                         details={"id": record.id})
            self._records[record.id] = record

    # -- construction -----------------------------------------------------------------------------
    @classmethod
    def load(cls, paths: Iterable[str | Path] = (), *, bundled: bool = True) -> "Catalog":
        """Load the bundled catalogue (unless ``bundled=False``) plus the given directories / files."""
        roots: list[Path] = []
        if bundled:
            roots.append(BUNDLED_CATALOG_DIR)
        roots.extend(Path(p) for p in paths)
        records: list[MagneticRecord] = []
        sources: dict[str, Path] = {}
        for root in roots:
            for file in _record_files(root):
                record = load_record_file(file)
                if record.id in sources:
                    raise DataIntegrityError(f"{file}: duplicate record id {record.id!r} (also in {sources[record.id]})",
                                             details={"id": record.id, "paths": [str(sources[record.id]), str(file)]})
                sources[record.id] = file
                records.append(record)
        return cls(records, sources=sources)

    @classmethod
    def default(cls) -> "Catalog":
        """The bundled catalogue, loaded once and cached for the process."""
        return _default_catalog()

    @classmethod
    def clear_default_cache(cls) -> None:
        _default_catalog.cache_clear()

    def with_records(self, records: Iterable[MagneticRecord]) -> "Catalog":
        """A new catalogue containing this catalogue's records plus ``records`` (duplicates are refused)."""
        return Catalog(list(self._records.values()) + list(records), sources=self._sources)

    # -- access -----------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, record_id: object) -> bool:
        return record_id in self._records

    def __iter__(self):
        return iter(self._records.values())

    def ids(self) -> list[str]:
        return sorted(self._records)

    def records(self, kind: str | None = None) -> list[MagneticRecord]:
        """All records, or those of one ``kind`` (unknown kinds are an error, not an empty list)."""
        if kind is None:
            return [self._records[k] for k in sorted(self._records)]
        if kind not in PROPERTY_MODELS:
            raise InvalidInputError(f"unknown record kind {kind!r}; known kinds: {', '.join(PROPERTY_MODELS)}",
                                    details={"kind": kind})
        return [self._records[k] for k in sorted(self._records) if self._records[k].kind == kind]

    def get(self, record_id: str) -> MagneticRecord:
        try:
            return self._records[record_id]
        except KeyError:
            suggestions = difflib.get_close_matches(str(record_id), list(self._records), n=5, cutoff=0.5)
            hint = f"; did you mean {suggestions}?" if suggestions else ""
            raise DataNotFoundError(f"no record with id {record_id!r}{hint}",
                                    details={"id": record_id, "suggestions": suggestions}) from None

    def source_of(self, record_id: str) -> Path | None:
        """File the record was loaded from (``None`` for records added in memory)."""
        return self._sources.get(record_id)

    def materials(self) -> list[MagneticRecord]:
        return self.records("magnetic_material")

    def manufacturers(self) -> list[MagneticRecord]:
        return self.records("manufacturer")

    def cores(self, material_id: str | None = None) -> list[MagneticRecord]:
        """``core_geometry`` records, optionally only those offered in ``material_id`` (``material_refs``)."""
        cores = self.records("core_geometry")
        if material_id is None:
            return cores
        if material_id not in self._records:
            raise DataNotFoundError(f"no record with id {material_id!r}", details={"id": material_id})
        return [core for core in cores if material_id in core.properties.get("material_refs", [])]

    @staticmethod
    def _material_keys(record: MagneticRecord) -> list[str]:
        keys = [record.id, record.name]
        grade = record.properties.get("grade")
        if isinstance(grade, str) and grade:
            keys.append(grade)
        return keys

    def find_material(self, query: str) -> MagneticRecord:
        """Find one material by id, name or grade (case-insensitive exact match, then substring match)."""
        if not isinstance(query, str) or not query.strip():
            raise InvalidInputError("material query must be a non-empty string", details={"query": query})
        needle = query.strip().casefold()
        materials = self.materials()
        by_id = [m for m in materials if m.id.casefold() == needle]
        if len(by_id) == 1:
            return by_id[0]
        exact = [m for m in materials if any(k.casefold() == needle for k in self._material_keys(m))]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise InvalidInputError(f"material query {query!r} is ambiguous: {[m.id for m in exact]}",
                                    details={"query": query, "candidates": [m.id for m in exact]})
        partial = [m for m in materials if any(needle in k.casefold() for k in self._material_keys(m))]
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            raise InvalidInputError(f"material query {query!r} is ambiguous: {[m.id for m in partial]}",
                                    details={"query": query, "candidates": [m.id for m in partial]})
        vocabulary: dict[str, str] = {}
        for material in materials:
            for key in self._material_keys(material):
                vocabulary.setdefault(key.casefold(), material.id)
        close = difflib.get_close_matches(needle, list(vocabulary), n=5, cutoff=0.5)
        suggestions = sorted({vocabulary[c] for c in close})
        hint = f"; did you mean {suggestions}?" if suggestions else ""
        raise DataNotFoundError(f"no material matches {query!r} by id, name or grade{hint}",
                                details={"query": query, "suggestions": suggestions, "available": [m.id for m in materials]})

    # -- validation -------------------------------------------------------------------------------
    def validate(self) -> list[str]:
        """Cross-record checks (unique ids, resolvable references, reference kinds); empty = consistent."""
        return _records.validate_catalogue(self.records())

    def describe(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for record in self._records.values():
            counts[record.kind] = counts.get(record.kind, 0) + 1
        return {"records": len(self._records), "by_kind": dict(sorted(counts.items())),
                "synthetic": sum(r.synthetic for r in self._records.values()),
                "files": sorted(str(p) for p in set(self._sources.values()))}


@functools.lru_cache(maxsize=1)
def _default_catalog() -> Catalog:
    return Catalog.load()


__all__ = ["BUNDLED_CATALOG_DIR", "Catalog", "load_record_file"]
