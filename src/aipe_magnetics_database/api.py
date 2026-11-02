"""Stable Python API of the AIPE Magnetics Database (data retrieval only).

>>> import aipe_magnetics_database.api as db
>>> db.get_material("TDK N87")["id"]
'material.canonical-tdk-n87'
>>> [m["id"] for m in db.find_measurements(material="TDK N87", quantity="core_loss", data_class="experimental_measurement")]
['measurement.princeton-magnet-n87-core-loss']

Every function returns JSON-serialisable dictionaries / lists. Conditions carry units (``"100 kHz"``,
``("90", "degC")`` is *not* accepted - use ``"90 degC"`` or ``(90, "degC")``); ranges are ``("50 kHz", "200 kHz")`` or
``"50 kHz..200 kHz"``. Loss prediction, fitting, interpolation and design live in AIPE-Magnetics-Design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from aipe_magnetics_database.store import MagneticDatabase, open_database


def _db(root: str | Path | None = None) -> MagneticDatabase:
    return open_database(str(root) if root else None)


def summary(*, root: str | Path | None = None) -> dict[str, Any]:
    """Record counts per source and of the canonical catalogue."""
    return _db(root).summary()


def list_sources(*, status: str | None = None, root: str | Path | None = None) -> list[dict[str, Any]]:
    """Every investigated source with integration status, version, licence and the reason for anything not integrated."""
    return [s for s in _db(root).sources if status is None or s["status"] == status]


def list_materials(*, query: str | None = None, manufacturer: str | None = None, material_class: str | None = None, source: str | None = None,
                   canonical: bool = True, root: str | Path | None = None) -> list[dict[str, Any]]:
    """Compact listing: id, name, grade, class and contributing sources."""
    out = []
    for r in _db(root).materials(query=query, manufacturer=manufacturer, material_class=material_class, source=source, canonical=canonical):
        members = r.get("extensions", {}).get("aipe.canonical", {}).get("members", [])
        out.append({"id": r["id"], "name": r["name"], "grade": r["properties"].get("grade"), "material_class": r["properties"].get("material_class"),
                    "sources": sorted({m["source_id"] for m in members}) if canonical else [r["_source"]], "license": r["license"]})
    return out


def get_material(query: str, *, root: str | Path | None = None) -> dict[str, Any]:
    """The canonical material record (v0.2) for an id, name or unique grade."""
    return _db(root).material(query)


def material_properties(query: str, *, root: str | Path | None = None) -> dict[str, Any]:
    """Typed properties of the canonical record plus every contributing source record and the fitted loss-model records."""
    db = _db(root)
    canon = db.material(query)
    members = db.material_members(query)
    return {"material": canon["id"], "name": canon["name"], "properties": canon["properties"],
            "properties_from": canon["extensions"]["aipe.canonical"]["properties_from"],
            "source_records": [{"id": m["id"], "source_id": m["_source"], "license": m["license"], "properties": m["properties"]} for m in members],
            "fitted_loss_models": [{"id": r["id"], "architecture": r["properties"]["architecture"],
                                    "coefficients": r["properties"]["inference_config"]["coefficients"],
                                    "equation": r["properties"]["inference_config"]["equation"], "data_class": "fitted_parameters"}
                                   for r in db.loss_models(query)],
            "measurements": canon["extensions"]["aipe.canonical"]["measurements"]}


def list_cores(*, query: str | None = None, family: str | None = None, manufacturer: str | None = None, material: str | None = None,
               kind: str = "all", limit: int = 50, root: str | Path | None = None) -> list[dict[str, Any]]:
    """Core shapes and manufacturer parts (MAS); effective parameters are only present where a source publishes them."""
    out = []
    for r in _db(root).cores(query=query, family=family, manufacturer=manufacturer, material=material, kind=kind, limit=limit):
        mas = r.get("extensions", {}).get("aipe.mas", {})
        out.append({"id": r["id"], "name": r["name"], "part_number": r["properties"].get("part_number"), "family": mas.get("family"),
                    "manufacturer": mas.get("manufacturer"), "material_refs": r["properties"].get("material_refs", []),
                    "flags": r.get("quality", {}).get("flags", [])})
    return out


def get_core(query: str, *, root: str | Path | None = None) -> dict[str, Any]:
    """A core shape or part record by id, name, part number or alias."""
    return _db(root).core(query)


def find_measurements(*, material: str | None = None, quantity: str | None = None, data_class: str | None = None, source: str | None = None,
                      frequency: Any = None, temperature: Any = None, flux_density: Any = None, dc_bias: Any = None, waveform: str | None = None,
                      include_duplicates: bool = False, root: str | Path | None = None) -> list[dict[str, Any]]:
    """Measurement tables whose condition ranges overlap the given values; duplicates of other sources are hidden by default."""
    return _db(root).measurements(material=material, quantity=quantity, data_class=data_class, source=source, frequency=frequency,
                                  temperature=temperature, flux_density=flux_density, dc_bias=dc_bias, waveform=waveform,
                                  include_duplicates=include_duplicates)


def get_table(record_id: str, *, frequency: Any = None, temperature: Any = None, flux_density: Any = None, dc_bias: Any = None,
              waveform: str | None = None, limit: int | None = None, root: str | Path | None = None) -> dict[str, Any]:
    """The rows of a measurement table (original measured values in SI units), optionally filtered row by row."""
    return _db(root).table(record_id, frequency=frequency, temperature=temperature, flux_density=flux_density, dc_bias=dc_bias,
                           waveform=waveform, limit=limit).to_json()


def get_record(record_id: str, *, root: str | Path | None = None) -> dict[str, Any]:
    return _db(root).record(record_id)


def provenance(record_id: str, *, root: str | Path | None = None) -> dict[str, Any]:
    """Evidence, licence, transformation history, quality flags, dataset and duplicate status of a record."""
    return _db(root).provenance(record_id)


def export(record_ids: Iterable[str], path: str | Path, *, include_rows: bool = True, root: str | Path | None = None) -> dict[str, Any]:
    """Write records as v0.2 JSONL (tables with their rows inlined)."""
    return _db(root).export(record_ids, path, include_rows=include_rows)


__all__ = ["export", "find_measurements", "get_core", "get_material", "get_record", "get_table", "list_cores", "list_materials", "list_sources",
           "material_properties", "provenance", "summary"]
