"""Read-only access to the repository data store (normalised source records + canonical catalogue).

Data location, in this order: the ``root`` argument, the environment variable ``AIPE_MAGNETICS_DATA`` (a directory
containing ``normalized/`` and ``canonical/``), the copy bundled into an installed wheel
(``aipe_magnetics_database/_bundled``), or the ``data/`` directory of a source checkout.

Physical conditions in queries always carry units - ``"100 kHz"``, ``("100", "degC")``, ``{"value": 0.1, "unit": "T"}``
or a range ``("50 kHz", "200 kHz")`` / ``"50 kHz..200 kHz"`` - bare numbers are rejected (no unit inference). Results
are plain JSON-serialisable dictionaries; table rows are streamed from the JSONL sidecars.

This module serves data only. It performs no loss prediction, fitting, interpolation or design.
"""

from __future__ import annotations

import functools
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from aipe_magnetics_database.common.errors import DataNotFoundError, InvalidInputError
from aipe_magnetics_database.common.quantity import Quantity

DATA_ENV_VAR = "AIPE_MAGNETICS_DATA"
SI = {"frequency": "Hz", "temperature": "K", "flux_density_peak": "T", "dc_bias_field": "A/m", "duty": "1"}
_QTY_RE = re.compile(r"^\s*([-+0-9.eE]+)\s*(\S.*)$")


def data_root(root: str | Path | None = None) -> Path:
    candidates = [Path(root)] if root else []
    if os.environ.get(DATA_ENV_VAR):
        candidates.append(Path(os.environ[DATA_ENV_VAR]))
    candidates.append(Path(__file__).resolve().parent / "_bundled")
    here = Path(__file__).resolve()
    candidates += [p / "data" for p in here.parents]
    for c in candidates:
        if (c / "normalized").is_dir() and (c / "canonical" / "index.json").is_file():
            return c
    raise DataNotFoundError("AIPE Magnetics data store not found: install the package from a wheel, run from a checkout, or set "
                            f"{DATA_ENV_VAR} to a directory with normalized/ and canonical/")


def quantity(value: Any, unit: str, name: str) -> float:
    """SI magnitude of a unit-bearing value (string ``"100 kHz"``, tuple, dict or Quantity)."""
    if isinstance(value, str):
        m = _QTY_RE.match(value)
        if not m:
            raise InvalidInputError(f"{name}: '{value}' is not '<number> <unit>'")
        value = (float(m.group(1)), m.group(2).strip())
    if isinstance(value, (int, float)):
        raise InvalidInputError(f"{name}: give a unit, e.g. ({value}, '{unit}') or '{value} {unit}' (bare numbers are not accepted)")
    return Quantity.parse(value, name=name).to(unit).value


def interval(value: Any, unit: str, name: str) -> tuple[float, float] | None:
    """``(lo, hi)`` in SI from a single value (degenerate interval) or a range; ``None`` passes through."""
    if value is None:
        return None
    if isinstance(value, str) and ".." in value:
        lo, hi = value.split("..", 1)
        value = (lo, hi)
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(isinstance(v, (str, tuple, list, dict, Quantity)) for v in value) \
            and not (isinstance(value[1], str) and not _QTY_RE.match(value[1])):
        lo, hi = quantity(value[0], unit, name), quantity(value[1], unit, name)
    else:
        lo = hi = quantity(value, unit, name)
    if lo > hi:
        raise InvalidInputError(f"{name}: reversed range {lo} > {hi} {unit}")
    return lo, hi


@dataclass(frozen=True)
class Table:
    """A measurement table: column descriptors and an iterator of rows (lists aligned with ``columns``)."""

    record_id: str
    columns: list[dict[str, Any]]
    rows: list[list[Any]]
    total_rows: int
    conditions: dict[str, Any]

    def column(self, name: str) -> list[Any]:
        i = [c["name"] for c in self.columns].index(name)
        return [r[i] for r in self.rows]

    def to_json(self) -> dict[str, Any]:
        return {"record_id": self.record_id, "columns": self.columns, "rows": self.rows, "returned_rows": len(self.rows),
                "total_rows": self.total_rows, "conditions": self.conditions}


class MagneticDatabase:
    """Lazy, read-only view of the data store."""

    def __init__(self, root: str | Path | None = None):
        self.root = data_root(root)
        self.normalized = self.root / "normalized"
        self.canonical_dir = self.root / "canonical"

    # -- raw loading ---------------------------------------------------------------------------------------
    @functools.cached_property
    def source_indexes(self) -> dict[str, dict[str, Any]]:
        out = {}
        for path in sorted(self.normalized.glob("*/index.json")):
            index = json.loads(path.read_text(encoding="utf-8"))
            index["_directory"] = path.parent.name
            out[index["source_id"]] = index
        return out

    @functools.cached_property
    def sources(self) -> list[dict[str, Any]]:
        path = self.root / "sources.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []

    def _shard_records(self, groups: Sequence[str]) -> Iterator[tuple[str, dict[str, Any]]]:
        for source_id, index in self.source_indexes.items():
            base = self.normalized / index["_directory"]
            for group in groups:
                for rel in index["shards"].get(group, []):
                    with open(base / rel, encoding="utf-8") as handle:
                        for line in handle:
                            yield source_id, json.loads(line)

    @functools.cached_property
    def _canonical_materials(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (self.canonical_dir / "materials.jsonl").read_text(encoding="utf-8").splitlines()]

    @functools.cached_property
    def _measurement_index(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (self.canonical_dir / "measurements.jsonl").read_text(encoding="utf-8").splitlines()]

    @functools.cached_property
    def _source_materials(self) -> list[dict[str, Any]]:
        return [r | {"_source": s} for s, r in self._shard_records(["materials"])]

    @functools.cached_property
    def _cores(self) -> list[dict[str, Any]]:
        return [r | {"_source": s} for s, r in self._shard_records(["shapes", "cores"])]

    @functools.cached_property
    def _loss_models(self) -> list[dict[str, Any]]:
        return [r | {"_source": s} for s, r in self._shard_records(["loss_models"])]

    @functools.cached_property
    def _measurement_records(self) -> dict[str, tuple[str, dict[str, Any]]]:
        return {r["id"]: (s, r) for s, r in self._shard_records(["measurements"])}

    # -- summary -------------------------------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        canonical = json.loads((self.canonical_dir / "index.json").read_text(encoding="utf-8"))
        per_source = {sid: {"records": idx["record_counts"], "table_rows": idx["table_rows"], "version": idx["version"], "license": idx["license"]}
                      for sid, idx in self.source_indexes.items()}
        return {"data_root": str(self.root), "schema_version": "0.2.0", "canonical": canonical["counts"], "sources": per_source}

    # -- materials -----------------------------------------------------------------------------------------
    def materials(self, *, query: str | None = None, manufacturer: str | None = None, material_class: str | None = None,
                  source: str | None = None, canonical: bool = True) -> list[dict[str, Any]]:
        """Canonical materials (default) or source material records, filtered by text, manufacturer, class and source."""
        items = self._canonical_materials if canonical else self._source_materials
        needle = (query or "").casefold().strip()
        out = []
        for r in items:
            props = r["properties"]
            members = r.get("extensions", {}).get("aipe.canonical", {}).get("members", [])
            sources = {m["source_id"] for m in members} if canonical else {r["_source"]}
            if source and source not in sources:
                continue
            if material_class and props.get("material_class") != material_class:
                continue
            if manufacturer and manufacturer.casefold() not in r["name"].casefold():
                continue
            if needle and needle not in " ".join([r["id"], r["name"], props.get("grade") or ""]).casefold():
                continue
            out.append(r)
        return out

    def material(self, query: str) -> dict[str, Any]:
        """Resolve a canonical material by id, exact name (``TDK N87``), member record id or unique grade (``N87``)."""
        q = query.strip().casefold()
        mats = self._canonical_materials
        for r in mats:
            if r["id"].casefold() == q or r["name"].casefold() == q:
                return r
        for r in mats:
            if any(m["record"].casefold() == q for m in r["extensions"]["aipe.canonical"]["members"]):
                return r
        by_grade = [r for r in mats if (r["properties"].get("grade") or "").casefold() == q]
        if len(by_grade) == 1:
            return by_grade[0]
        if by_grade:
            raise InvalidInputError(f"material '{query}' is ambiguous: {[r['name'] for r in by_grade][:10]}; use the full name or id",
                                    details={"candidates": [r["id"] for r in by_grade]})
        raise DataNotFoundError(f"no material matches '{query}'")

    def material_members(self, query: str) -> list[dict[str, Any]]:
        """The source records consolidated into a canonical material."""
        canon = self.material(query)
        ids = {m["record"] for m in canon["extensions"]["aipe.canonical"]["members"]}
        found = [r for r in self._source_materials if r["id"] in ids]
        from aipe_magnetics_database.ingest.pipeline import curated_records
        found += [r | {"_source": "aipe_curated"} for r in curated_records() if r["id"] in ids]
        return found

    def loss_models(self, material: str | None = None) -> list[dict[str, Any]]:
        """Fitted loss-model coefficient records (``fitted_parameters``; never measurements)."""
        if material is None:
            return self._loss_models
        ids = {m["record"] for m in self.material(material)["extensions"]["aipe.canonical"]["members"]}
        return [r for r in self._loss_models if set(r["properties"].get("material_refs", [])) & ids]

    # -- cores ---------------------------------------------------------------------------------------------
    def cores(self, *, query: str | None = None, family: str | None = None, manufacturer: str | None = None, material: str | None = None,
              kind: str = "all", limit: int | None = None) -> list[dict[str, Any]]:
        """Core shapes (``kind='shapes'``), manufacturer parts (``'parts'``) or both; curated cores via the ``database`` layer."""
        needle = (query or "").casefold()
        mat_ids = None
        if material:
            mat_ids = {m["record"] for m in self.material(material)["extensions"]["aipe.canonical"]["members"]}
        out = []
        for r in self._cores:
            mas = r.get("extensions", {}).get("aipe.mas", {})
            is_part = r["id"].startswith("core.mas-part-")
            if (kind == "shapes" and is_part) or (kind == "parts" and not is_part):
                continue
            if family and (mas.get("family") or "").casefold() != family.casefold() and not (mas.get("shape_name") or "").casefold().startswith(family.casefold()):
                continue
            if manufacturer and manufacturer.casefold() != (mas.get("manufacturer") or "").casefold():
                continue
            if mat_ids is not None and not set(r["properties"].get("material_refs", [])) & mat_ids:
                continue
            if needle and needle not in " ".join([r["name"], r["properties"].get("part_number") or "", *mas.get("aliases", [])]).casefold():
                continue
            out.append(r)
            if limit and len(out) >= limit:
                break
        return out

    def core(self, query: str) -> dict[str, Any]:
        q = query.strip().casefold()
        for r in self._cores:
            mas = r.get("extensions", {}).get("aipe.mas", {})
            if q in (r["id"].casefold(), r["name"].casefold(), (r["properties"].get("part_number") or "").casefold()) or \
                    q in [a.casefold() for a in mas.get("aliases", [])]:
                return r
        raise DataNotFoundError(f"no core shape or part matches '{query}'")

    # -- measurements --------------------------------------------------------------------------------------
    def measurements(self, *, material: str | None = None, quantity: str | None = None, data_class: str | None = None, source: str | None = None,
                     frequency: Any = None, temperature: Any = None, flux_density: Any = None, dc_bias: Any = None, waveform: str | None = None,
                     include_duplicates: bool = False) -> list[dict[str, Any]]:
        """Canonical measurement entries whose condition ranges overlap the requested values / ranges."""
        wanted = {"frequency": interval(frequency, "Hz", "frequency"), "temperature": interval(temperature, "K", "temperature"),
                  "flux_density_peak": interval(flux_density, "T", "flux_density"), "dc_bias_field": interval(dc_bias, "A/m", "dc_bias")}
        canon = self.material(material)["id"] if material else None
        out = []
        for e in self._measurement_index:
            if not include_duplicates and e["status"] != "primary":
                continue
            if canon and e["material"] != canon:
                continue
            if quantity and e["quantity"] != quantity:
                continue
            if data_class and e["data_class"] != data_class:
                continue
            if source and e["source_id"] != source:
                continue
            if waveform and waveform not in e["conditions"].get("waveforms", []):
                continue
            ok = True
            for key, rng in wanted.items():
                if rng is None:
                    continue
                c = e["conditions"].get(key)
                if c is None or c["max"] < rng[0] * (1 - 1e-9) or c["min"] > rng[1] * (1 + 1e-9):
                    ok = False
                    break
            if ok:
                out.append(e)
        return out

    def measurement(self, record_id: str) -> dict[str, Any]:
        try:
            return self._measurement_records[record_id][1]
        except KeyError:
            raise DataNotFoundError(f"no measurement record '{record_id}'") from None

    def _rows(self, record_id: str) -> Iterator[list[Any]]:
        source_id, rec = self._measurement_records[record_id]
        table = rec["extensions"]["aipe.table"]
        if "rows" in table:
            yield from table["rows"]
            return
        path = self.normalized / self.source_indexes[source_id]["_directory"] / table["rows_file"]
        if not path.is_file():
            raise DataNotFoundError(f"{record_id}: table sidecar {table['rows_file']} is not in this data store (large regenerated table); "
                                    f"run 'aipe-magnetics-db refresh --source {source_id} --allow-download' in a checkout",
                                    details={"rows_sha256": table.get("rows_sha256"), "row_count": table["row_count"]})
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)

    def table(self, record_id: str, *, frequency: Any = None, temperature: Any = None, flux_density: Any = None, dc_bias: Any = None,
              waveform: str | None = None, limit: int | None = None) -> Table:
        """Rows of a measurement table, optionally filtered row by row (values within the requested ranges)."""
        rec = self.measurement(record_id)
        table = rec["extensions"]["aipe.table"]
        names = [c["name"] for c in table["columns"]]
        filters = []
        for key, val, unit in (("frequency", frequency, "Hz"), ("temperature", temperature, "K"), ("flux_density_peak", flux_density, "T"),
                               ("dc_bias_field", dc_bias, "A/m")):
            rng = interval(val, unit, key)
            if rng is not None:
                if key not in names:
                    raise InvalidInputError(f"{record_id} has no '{key}' column; columns: {names}")
                filters.append((names.index(key), rng))
        wf = names.index("waveform") if waveform and "waveform" in names else None
        if waveform and wf is None:
            raise InvalidInputError(f"{record_id} has no waveform column")
        rows = []
        for row in self._rows(record_id):
            if wf is not None and row[wf] != waveform:
                continue
            if all(row[i] is not None and lo * (1 - 1e-9) <= row[i] <= hi * (1 + 1e-9) for i, (lo, hi) in filters):
                rows.append(row)
                if limit and len(rows) >= limit:
                    break
        return Table(record_id, table["columns"], rows, table["row_count"], table.get("conditions", {}))

    # -- provenance and export ---------------------------------------------------------------------------------
    def record(self, record_id: str) -> dict[str, Any]:
        if record_id in self._measurement_records:
            return self._measurement_records[record_id][1]
        for pool in (self._canonical_materials, self._source_materials, self._cores, self._loss_models):
            for r in pool:
                if r["id"] == record_id:
                    return {k: v for k, v in r.items() if k != "_source"}
        raise DataNotFoundError(f"no record '{record_id}'")

    def provenance(self, record_id: str) -> dict[str, Any]:
        rec = self.record(record_id)
        entry = next((e for e in self._measurement_index if e["id"] == record_id), None)
        dataset = None
        if rec.get("dataset_ref"):
            for sid, idx in self.source_indexes.items():
                for rel in idx["datasets"]:
                    d = json.loads((self.normalized / idx["_directory"] / rel).read_text(encoding="utf-8"))
                    if d["id"] == rec["dataset_ref"]:
                        dataset = d["properties"]
        return {"id": rec["id"], "kind": rec["kind"], "license": rec["license"], "evidence": rec.get("evidence", []),
                "provenance": rec.get("provenance"), "quality": rec.get("quality"), "data": rec.get("extensions", {}).get("aipe.data"),
                "canonical": entry and {k: entry.get(k) for k in ("status", "duplicate_of", "derived_from", "duplicate_match", "material")},
                "dataset": dataset}

    def export(self, record_ids: Iterable[str], path: str | Path, *, include_rows: bool = True) -> dict[str, Any]:
        """Write the records (tables with rows inlined when ``include_rows``) to a JSONL file; returns a manifest."""
        path = Path(path)
        n = 0
        with open(path, "w", encoding="utf-8") as handle:
            for rid in record_ids:
                rec = json.loads(json.dumps(self.record(rid)))
                table = rec.get("extensions", {}).get("aipe.table")
                if include_rows and table and "rows" not in table:
                    table["rows"] = list(self._rows(rid))
                    table.pop("rows_file", None)
                handle.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                n += 1
        return {"path": str(path), "records": n, "format": "jsonl", "schema_version": "0.2.0"}


@functools.lru_cache(maxsize=4)
def open_database(root: str | None = None) -> MagneticDatabase:
    return MagneticDatabase(root)


__all__ = ["DATA_ENV_VAR", "MagneticDatabase", "Table", "data_root", "interval", "open_database", "quantity"]
