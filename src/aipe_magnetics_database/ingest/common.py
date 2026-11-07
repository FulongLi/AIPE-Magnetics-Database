"""Shared building blocks of the reproducible source importers.

Conventions (normative for every importer, see docs/data-format.md):

* Records are AIPE Magnetic Data Schema v0.2 objects validated with :class:`MagneticRecord` before they are written.
* Scalar material properties use the typed v0.2 fields. Tabulated data (loss tables, complex permeability, B-H
  curves ...) are ``measurement`` records whose ``extensions["aipe.table"]`` holds the table in the
  ``aipe.magnetics.table/0.1`` layout: explicit columns with SI unit, original unit and definition, and the rows either
  inline (small tables) or in a JSONL sidecar file (one JSON array per line) referenced by an artifact with SHA-256.
* ``extensions["aipe.data"]["data_class"]`` separates experimental measurements, manufacturer datasheet data,
  manufacturer-tool exports, values derived from measurements and fitted parameters. Model predictions are never
  stored in the Database.
* Nothing is invented: missing values are omitted, rows with non-finite values are excluded and counted in the
  import summary, suspicious rows are kept verbatim and flagged.
* Output is deterministic (sorted keys inside tables, stable ordering, fixed timestamps taken from the source pin),
  so re-running an importer on the same pinned inputs rewrites byte-identical files (idempotent).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from aipe_magnetics_database import __version__
from aipe_magnetics_database.data.records import MagneticRecord

TABLE_LAYOUT = "aipe.magnetics.table/0.1"
INLINE_ROW_LIMIT = 2000
DataClass = Literal["experimental_measurement", "manufacturer_datasheet", "manufacturer_tool_export",
                    "derived_from_measurement", "fitted_parameters", "aggregated_literature", "identity"]
DATA_CLASSES = ("experimental_measurement", "manufacturer_datasheet", "manufacturer_tool_export", "derived_from_measurement",
                "fitted_parameters", "aggregated_literature", "identity")

# canonical SI unit per table column quantity (the table validator enforces it)
COLUMN_UNITS: dict[str, str] = {
    "frequency": "Hz", "temperature": "K", "flux_density_peak": "T", "flux_density": "T", "field": "A/m",
    "dc_bias_field": "A/m", "loss_density": "W/m^3", "mass_loss_density": "W/kg", "energy_density_per_cycle": "J/m^3",
    "mu_real": "1", "mu_imag": "1", "eps_real": "1", "eps_imag": "1", "relative_permeability": "1",
    "amplitude_permeability": "1", "loss_factor": "1", "duty": "1", "duty_p": "1", "duty_n": "1", "loss_tangent": "1",
    "uncertainty_relative": "1", "conductivity": "S/m", "electric_field_peak": "V/m", "polarization": "T", "flux_density_peak_to_peak": "T",
}


def importer_attribution(importer_id: str, version: str = __version__) -> dict[str, str]:
    return {"id": f"aipe.magnetics.database.ingest.{importer_id}", "version": version, "name": "AIPE Magnetics Database importer"}


def slug(text: str, *, transliterate: bool = True) -> str:
    """Lower-case id token: ``TDK N87`` -> ``tdk-n87``; ``µ`` and ``+`` are transliterated (or, with
    ``transliterate=False`` as in the first MAS import, dropped like every other symbol)."""
    if transliterate:
        text = text.replace("µ", "u").replace("μ", "u").replace("+", "plus")
    else:
        text = unicodedata.normalize("NFKD", text)
    token = re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")
    return token or "x"


def q(value: float, unit: str, original: tuple[float, str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"value": value, "unit": unit}
    if original is not None and (original[1] != unit or original[0] != value):
        out["original"] = {"value": original[0], "unit": original[1]}
    return out


def kelvin(celsius: float) -> dict[str, Any]:
    """Temperature quantity in K that retains the original degC value."""
    return q(celsius + 273.15, "K", (celsius, "degC"))


def finite(*values: Any) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def evidence(eid: str, kind: str, reference: str, description: str, created_at: str, license: str, *, url: str | None = None,
             version: str | None = None, locator: str | None = None, sha256: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"id": eid, "kind": kind, "reference": reference, "description": description, "created_at": created_at,
                           "review_status": "unreviewed", "license": license}
    for key, value in (("locator", locator), ("url", url), ("version", version), ("sha256", sha256)):
        if value is not None:
            out[key] = value
    return out


def table_extension(quantity: str, columns: Sequence[dict[str, Any]], rows: list[list[Any]], *, conditions: dict[str, Any] | None = None,
                    rows_file: str | None = None, rows_sha256: str | None = None, notes: Sequence[str] = ()) -> dict[str, Any]:
    """``aipe.magnetics.table/0.1`` block; ``rows`` are inlined when no ``rows_file`` is given."""
    for column in columns:
        expected = COLUMN_UNITS.get(column["name"])
        if expected is not None and column["unit"] != expected:
            raise ValueError(f"table column {column['name']} must be in {expected}, got {column['unit']}")
    block: dict[str, Any] = {"layout": TABLE_LAYOUT, "quantity": quantity, "columns": list(columns), "row_count": len(rows)}
    if conditions:
        block["conditions"] = conditions
    if rows_file is None:
        block["rows"] = rows
    else:
        block["rows_file"] = rows_file
        block["rows_sha256"] = rows_sha256
    if notes:
        block["notes"] = list(notes)
    return block


def column(name: str, definition: str, original_unit: str | None = None, unit: str | None = None) -> dict[str, Any]:
    out = {"name": name, "unit": unit or COLUMN_UNITS[name], "definition": definition}
    if original_unit is not None:
        out["original_unit"] = original_unit
    return out


def rows_jsonl(rows: Iterable[Sequence[Any]]) -> bytes:
    """Deterministic JSONL bytes of table rows (``repr``-exact floats, no spaces)."""
    return b"".join(json.dumps(list(r), separators=(",", ":"), allow_nan=False).encode() + b"\n" for r in rows)


def _compact(obj: Any, *, keep_empty: bool = False) -> Any:
    if isinstance(obj, dict):
        return {k: _compact(v, keep_empty=keep_empty) for k, v in obj.items() if v is not None and (keep_empty or (v != [] and v != {}))}
    if isinstance(obj, list):
        return [_compact(v, keep_empty=keep_empty) for v in obj]
    return obj


def validated(record: dict[str, Any]) -> dict[str, Any]:
    """Validate against schema v0.2 (raises) and return the JSON form without ``None`` / empty envelope fields.

    Empty lists and ``None`` are dropped from the envelope, ``quality``, ``provenance`` and namespaced extensions
    (they equal the schema defaults); ``properties`` and table rows are kept exactly (a ``null`` table cell means
    "not reported"). The compacted form is validated again.
    """
    out = MagneticRecord.model_validate(record).to_json_dict()
    for key in [k for k, v in out.items() if v == [] or v == {} and k != "properties"]:
        del out[key]
    for key in ("quality", "provenance"):
        if isinstance(out.get(key), dict):
            out[key] = _compact(out[key])
    if isinstance(out.get("extensions"), dict):
        # extensions: only None is dropped (an empty mapping such as dimensions_range_m = {} is information)
        out["extensions"] = {ns: (_compact(v, keep_empty=True) if ns != "aipe.table" else v) for ns, v in out["extensions"].items()}
    MagneticRecord.model_validate(out)
    return out


def dumps_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def dumps_pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def write_if_changed(path: Path, data: bytes) -> bool:
    """Write ``data`` unless the file already holds exactly these bytes; returns True when written."""
    if path.is_file() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return True


def write_shards(records: Sequence[dict[str, Any]], directory: Path, prefix: str, per_shard: int) -> list[str]:
    """JSONL shards ``<prefix>-001.jsonl`` ... of ``per_shard`` records; stale shards beyond the new count are removed."""
    directory.mkdir(parents=True, exist_ok=True)
    names = []
    for index in range(0, len(records), per_shard):
        name = f"{prefix}-{index // per_shard + 1:03d}.jsonl"
        write_if_changed(directory / name, "".join(dumps_line(r) + "\n" for r in records[index:index + per_shard]).encode("utf-8"))
        names.append(name)
    for stale in directory.glob(f"{prefix}-*.jsonl"):
        if stale.name not in names:
            stale.unlink()
    return names


@dataclass
class ImportSummary:
    """Counts, exclusions and warnings of one importer run (written to ``data/reports/import-<source>.json``)."""

    source_id: str
    version: str
    license: str
    redistribution: str
    counts: dict[str, int] = field(default_factory=dict)
    table_rows: dict[str, int] = field(default_factory=dict)
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    flags: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    inputs: list[dict[str, Any]] = field(default_factory=list)
    files_written: int = 0
    files_unchanged: int = 0

    def count(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def rows(self, quantity: str, n: int) -> None:
        self.table_rows[quantity] = self.table_rows.get(quantity, 0) + n

    def flag(self, name: str, n: int = 1) -> None:
        self.flags[name] = self.flags.get(name, 0) + n

    def exclude(self, what: str, reason: str, n: int = 1, **details: Any) -> None:
        self.exclusions.append({"what": what, "reason": reason, "count": n, **details})

    def to_json(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "version": self.version, "license": self.license, "redistribution": self.redistribution,
                "importer_version": __version__, "inputs": self.inputs, "counts": dict(sorted(self.counts.items())),
                "table_rows": dict(sorted(self.table_rows.items())), "flags": dict(sorted(self.flags.items())),
                "exclusions": self.exclusions, "notes": self.notes}


class SourceWriter:
    """Writes one source's normalised output directory deterministically and tracks what changed."""

    def __init__(self, root: Path, summary: ImportSummary):
        self.root = root
        self.summary = summary

    def file(self, relative: str, data: bytes) -> str:
        if write_if_changed(self.root / relative, data):
            self.summary.files_written += 1
        else:
            self.summary.files_unchanged += 1
        return sha256_bytes(data)

    def json(self, relative: str, obj: Any) -> str:
        return self.file(relative, dumps_pretty(obj).encode("utf-8"))

    def shards(self, records: Sequence[dict[str, Any]], sub: str, prefix: str, per_shard: int) -> list[str]:
        before = {p.name: p.read_bytes() for p in (self.root / sub).glob(f"{prefix}-*.jsonl")} if (self.root / sub).is_dir() else {}
        names = write_shards(records, self.root / sub, prefix, per_shard)
        for name in names:
            if before.get(name) == (self.root / sub / name).read_bytes():
                self.summary.files_unchanged += 1
            else:
                self.summary.files_written += 1
        return [f"{sub}/{n}" for n in names]


def table_sidecar(writer: SourceWriter, record_id: str, rows: list[list[Any]], folder: str = "tables") -> tuple[str, str, int]:
    """Write ``<folder>/<record-id>.jsonl``; returns ``(relative path, sha256, size)``."""
    data = rows_jsonl(rows)
    relative = f"{folder}/{record_id.split('.', 1)[1]}.jsonl"
    digest = writer.file(relative, data)
    return relative, digest, len(data)


def measurement_record(*, rid: str, name: str, material_ref: str, license: str, evidence_items: list[dict[str, Any]], data_class: str,
                       source_id: str, quantity: str, columns: list[dict[str, Any]], rows: list[list[Any]], writer: SourceWriter | None,
                       provenance: dict[str, Any], method: str | None = None, specimen: str | None = None, instrument: str | None = None,
                       description: str | None = None, conditions: dict[str, Any] | None = None, flags: Sequence[str] = (),
                       quality_notes: Sequence[str] = (), dataset_ref: str | None = None, underlying_source: str | None = None,
                       derived_from: Sequence[str] = (), extra: dict[str, Any] | None = None, table_notes: Sequence[str] = (),
                       commit_sidecar: bool = True) -> dict[str, Any]:
    """A ``measurement`` record carrying one ``aipe.magnetics.table/0.1`` table (inline or sidecar).

    ``commit_sidecar=False`` writes a large sidecar into ``tables-local/`` (not committed to Git; regenerated by
    ``aipe-magnetics-db refresh``); the record still carries the row count and the SHA-256 of the exact rows.
    """
    if data_class not in DATA_CLASSES:
        raise ValueError(data_class)
    artifacts = []
    if len(rows) > INLINE_ROW_LIMIT and writer is not None:
        relative, digest, size = table_sidecar(writer, rid, rows, "tables" if commit_sidecar else "tables-local")
        # artifact uri is relative to the source directory root (records live one level below)
        artifacts.append({"uri": f"../{relative}", "media_type": "application/x-ndjson", "role": "normalized", "sha256": digest,
                          "size_bytes": size, "description": f"{len(rows)} table rows, one JSON array per line, columns as in aipe.table"})
        table = table_extension(quantity, columns, rows, conditions=conditions, rows_file=relative, rows_sha256=digest, notes=table_notes)
    else:
        table = table_extension(quantity, columns, rows, conditions=conditions, notes=table_notes)
    props: dict[str, Any] = {"material_ref": material_ref}
    for key, value in (("method", method), ("specimen", specimen), ("instrument", instrument), ("description", description)):
        if value:
            props[key] = value
    data_ext: dict[str, Any] = {"data_class": data_class, "source_id": source_id}
    if underlying_source:
        data_ext["underlying_source"] = underlying_source
    if derived_from:
        data_ext["derived_from"] = list(derived_from)
    if extra:
        data_ext.update(extra)
    record = {"schema_version": "0.2.0", "id": rid, "kind": "measurement", "name": name, "synthetic": False, "license": license,
              "properties": props, "evidence_refs": [e["id"] for e in evidence_items], "evidence": evidence_items,
              "quality": {"status": "normalized", "flags": sorted(set(flags) | {"third_party_unreviewed"}), "notes": list(quality_notes)},
              "provenance": provenance, "extensions": {"aipe.table": table, "aipe.data": data_ext}}
    if not record["quality"]["notes"]:
        del record["quality"]["notes"]
    if artifacts:
        record["artifacts"] = artifacts
    if dataset_ref:
        record["dataset_ref"] = dataset_ref
    return validated(record)


def dataset_record(*, rid: str, name: str, source_id: str, title: str, license: str, url: str, version: str, evidence_items: list[dict[str, Any]],
                   provenance: dict[str, Any], redistribution: bool | None, record_count: int, publisher: str | None = None,
                   citation: str | None = None, source_commit: str | None = None, checksums: dict[str, str] | None = None,
                   materials: Sequence[str] = (), description: str | None = None, flags: Sequence[str] = (),
                   extensions: dict[str, Any] | None = None) -> dict[str, Any]:
    props: dict[str, Any] = {"source_id": source_id, "title": title, "url": url, "version": version, "license": license,
                             "redistribution_permitted": redistribution, "record_count": record_count}
    for key, value in (("publisher", publisher), ("citation", citation), ("source_commit", source_commit), ("description", description)):
        if value:
            props[key] = value
    if checksums:
        props["checksums"] = dict(sorted(checksums.items()))
    if materials:
        props["materials"] = sorted(materials)
    record = {"schema_version": "0.2.0", "id": rid, "kind": "dataset", "name": name, "synthetic": False, "license": license,
              "properties": props, "evidence_refs": [e["id"] for e in evidence_items], "evidence": evidence_items,
              "quality": {"status": "normalized", "flags": sorted(set(flags) | {"third_party_unreviewed"})}, "provenance": provenance,
              "extensions": extensions or {}}
    return validated(record)


def finish_source(writer: SourceWriter, summary: ImportSummary, shard_files: dict[str, list[str]], dataset_files: list[str]) -> ImportSummary:
    """Write the per-source ``index.json`` (shards, datasets, tables, counts)."""
    counts: dict[str, int] = {}
    for files in list(shard_files.values()):
        for rel in files:
            for line in (writer.root / rel).read_text(encoding="utf-8").splitlines():
                kind = json.loads(line)["kind"]
                counts[kind] = counts.get(kind, 0) + 1
    counts["dataset"] = len(dataset_files)
    summary.counts.update({f"records.{k}": v for k, v in counts.items()})
    index = {"schema_version": "0.2.0", "source_id": summary.source_id, "version": summary.version, "license": summary.license,
             "shards": {k: v for k, v in shard_files.items() if v}, "datasets": dataset_files,
             "tables": sorted(p.relative_to(writer.root).as_posix() for p in (writer.root / "tables").glob("*.jsonl")),
             "tables_local": sorted(p.relative_to(writer.root).as_posix() for p in (writer.root / "tables-local").glob("*.jsonl")),
             **({"waveforms_local": sorted(p.relative_to(writer.root).as_posix() for p in (writer.root / "waveforms-local").glob("*.h5"))}
                if (writer.root / "waveforms-local").is_dir() else {}),
             "record_counts": dict(sorted(counts.items())), "table_rows": dict(sorted(summary.table_rows.items())),
             "review_state": "third_party_unreviewed"}
    writer.json("index.json", index)
    return summary


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


__all__ = [
    "COLUMN_UNITS", "DATA_CLASSES", "INLINE_ROW_LIMIT", "TABLE_LAYOUT", "ImportSummary", "SourceWriter", "column", "dataset_record",
    "dumps_line", "dumps_pretty", "evidence", "finish_source", "finite", "git_blob_sha1", "importer_attribution", "kelvin", "measurement_record", "q",
    "read_jsonl", "rows_jsonl", "sha256_bytes", "sha256_path", "slug", "table_extension", "table_sidecar", "validated",
    "write_if_changed", "write_shards",
]
