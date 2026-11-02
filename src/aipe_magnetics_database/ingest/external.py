"""User-supplied-file adapters for sources whose redistribution is not permitted or not established.

The user downloads the original file under the provider's terms; the adapter converts it into v0.2 records in a
directory of the user's choice (default ``data/user/<source>/``, ignored by Git). Nothing is fetched by these
adapters and their output is never part of the committed catalogue.

* :func:`asu_mhz_csv` - ASU MAPEL ``MHzCoreLoss.csv`` (aggregate of 13 published studies; each row keeps its study
  label as ``underlying_source`` so aggregated rows are never counted as independent of the originals);
* :func:`psma_coreloss_db` - PSMA / Dartmouth Phase II ``coreloss.db`` (SQLite, opened read-only): per-run period,
  whole-core energy per cycle and peak flux linkage; per-volume values need the user's core geometry and are not derived;
* :func:`user_table` - any CSV with an explicit column -> (AIPE column, unit) mapping (e.g. manufacturer files the
  user is entitled to use).
"""

from __future__ import annotations

import csv
import io
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from aipe_magnetics_database.common.units import units
from aipe_magnetics_database.ingest.common import (
    COLUMN_UNITS,
    ImportSummary,
    SourceWriter,
    column,
    evidence,
    finish_source,
    importer_attribution,
    measurement_record,
    sha256_path,
    slug,
    validated,
)

NOW = "1970-01-01T00:00:00Z"  # user files carry no publication time; the SHA-256 identifies the input


def _identity(mid: str, name: str, grade: str, source_id: str, ev: list[dict[str, Any]], license: str) -> dict[str, Any]:
    return validated({"schema_version": "0.2.0", "id": mid, "kind": "magnetic_material", "name": name, "synthetic": False, "license": license,
                      "properties": {"grade": grade, "description": f"{name} as named in a user-supplied {source_id} file."},
                      "evidence_refs": [e["id"] for e in ev], "evidence": ev,
                      "quality": {"status": "normalized", "flags": ["identity_only", "user_supplied_source", "third_party_unreviewed"]},
                      "provenance": {"activity": f"user_{source_id}", "tool": importer_attribution(source_id)},
                      "extensions": {"aipe.data": {"data_class": "identity", "source_id": source_id}}})


def asu_mhz_csv(csv_path: str | Path, output_dir: str | Path) -> ImportSummary:
    """Normalise a user copy of ``MHzCoreLoss.csv`` (columns as at commit 8cf2378d)."""
    path = Path(csv_path)
    digest = sha256_path(path)
    summary = ImportSummary("asu_mhz_core_loss_aggregate", f"user file sha256 {digest}", "NOASSERTION", "not redistributable - local use only")
    writer = SourceWriter(Path(output_dir), summary)
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    groups: dict[tuple[str, str, str], list[list[Any]]] = defaultdict(list)
    for r in rows:
        try:
            b, p, f = float(r["FluxDensity_mT"]) / 1000.0, float(r["PowerLossDensity_mW_per_cm3"]) * 1000.0, float(r["frequency_Hz"])
        except (KeyError, ValueError):
            summary.exclude("row", "missing B / P / f")
            continue
        t = float(r["temperature_C"]) + 273.15 if r.get("temperature_C") else None
        unc = float(r["PowerLossDensity_stdunc_mW_per_cm3"]) * 1000.0 if r.get("PowerLossDensity_stdunc_mW_per_cm3") else None
        groups[(r["dataset"], r.get("manufacturer") or "", r.get("material_code") or "")].append(
            [f, b, t, p, unc, f"R{r.get('OD_mm')}x{r.get('ID_mm')}x{r.get('H_mm')} mm", r.get("method") or None])
    ev = [evidence("evidence.asu-user-file", "source", "asu-mapel/MHzCoreLossAggregateDataset (user copy)", "User-supplied file; no licence upstream.",
                   NOW, "NOASSERTION", locator=path.name, sha256=digest)]
    materials, measurements = {}, []
    for (study, manufacturer, code), table in sorted(groups.items()):
        mid = f"material.user-asu-{slug(manufacturer + ' ' + code)}"
        materials.setdefault(mid, _identity(mid, f"{manufacturer} {code}".strip(), code, summary.source_id, ev, "NOASSERTION"))
        mev = ev + [evidence("evidence.asu-study", "measurement", study, f"Measurement of the original study '{study}' as aggregated by ASU MAPEL.",
                             NOW, "NOASSERTION", locator=f"{path.name}: dataset == {study!r}")]
        measurements.append(measurement_record(
            rid=f"measurement.user-asu-{slug(study)}-{slug(manufacturer + ' ' + code)}", name=f"{study}: {manufacturer} {code}", material_ref=mid,
            license="NOASSERTION", evidence_items=mev, data_class="aggregated_literature", source_id=summary.source_id, quantity="core_loss",
            columns=[column("frequency", "excitation frequency"), column("flux_density_peak", "peak flux density as aggregated", "mT"),
                     column("temperature", "temperature (null = not stated)", "degC"), column("loss_density", "core loss density", "mW/cm^3"),
                     column("loss_density", "standard uncertainty of the loss density (null = not stated)", "mW/cm^3") | {"name": "loss_density_stdunc", "unit": "W/m^3"},
                     {"name": "specimen", "unit": "", "definition": "toroid OD x ID x H as aggregated"},
                     {"name": "method", "unit": "", "definition": "measurement method label as aggregated"}],
            rows=table, writer=writer, provenance={"activity": "user_asu_mhz_csv", "tool": importer_attribution("asu")},
            flags=("user_supplied_source", "aggregated_from_literature"), underlying_source=f"literature:{slug(study)}"))
        summary.rows("core_loss", len(table))
    shards = {"materials": writer.shards(list(materials.values()), "materials", "materials", 100),
              "measurements": writer.shards(measurements, "measurements", "measurements", 100)}
    return finish_source(writer, summary, shards, [])


def psma_coreloss_db(db_path: str | Path, output_dir: str | Path) -> ImportSummary:
    """Normalise the ``data`` table of a user copy of the PSMA Phase II ``coreloss.db`` (opened read-only)."""
    path = Path(db_path)
    digest = sha256_path(path)
    summary = ImportSummary("psma_core_loss_studies", f"user file sha256 {digest}", "proprietary", "not redistributable - local use only")
    writer = SourceWriter(Path(output_dir), summary)
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cols = {r[1] for r in con.execute("PRAGMA table_info(data)")}
    needed = {"setId", "runId", "coreId", "period", "Ecyc", "dPhi", "N"}
    if not needed <= cols:
        raise ValueError(f"coreloss.db 'data' table lacks columns {sorted(needed - cols)}")
    groups: dict[tuple[str, str], list[list[Any]]] = defaultdict(list)
    for r in con.execute("SELECT * FROM data ORDER BY setId, runId"):
        if not r["period"] or r["Ecyc"] is None:
            summary.exclude("run", "missing period or Ecyc")
            continue
        groups[(r["setId"], str(r["coreId"]))].append([1.0 / r["period"], r["Ecyc"], r["dPhi"], r["N"],
                                                       r["TSet"] + 273.15 if "TSet" in cols and r["TSet"] is not None else None, r["runId"]])
    ev = [evidence("evidence.psma-user-file", "measurement", "PSMA / Dartmouth Core Loss Studies Phase II (user copy)",
                   "User-supplied coreloss.db; PSMA copyright, reproduction requires written consent.", NOW, "proprietary", locator=path.name, sha256=digest)]
    materials, measurements = {}, []
    for (set_id, core_id), table in sorted(groups.items()):
        mid = f"material.user-psma-core-{slug(core_id)}"
        materials.setdefault(mid, _identity(mid, f"PSMA core {core_id}", core_id, summary.source_id, ev, "proprietary"))
        measurements.append(measurement_record(
            rid=f"measurement.user-psma-{slug(set_id)}-{slug(core_id)}", name=f"PSMA run set {set_id}, core {core_id}", material_ref=mid,
            license="proprietary", evidence_items=ev, data_class="experimental_measurement", source_id=summary.source_id, quantity="core_energy_loss",
            columns=[column("frequency", "1 / period of the rectangular-voltage excitation"),
                     {"name": "energy_per_cycle_core", "unit": "J", "definition": "Ecyc: energy per period dissipated in the whole core"},
                     {"name": "flux_linkage_peak", "unit": "Wb", "definition": "dPhi: peak flux (documented as T upstream, values are Vps*t1/(2N) in Wb)"},
                     {"name": "turns", "unit": "1", "definition": "N"}, column("temperature", "set temperature TSet (null = absent)", "degC"),
                     {"name": "run_id", "unit": "", "definition": "upstream runId"}],
            rows=table, writer=writer, provenance={"activity": "user_psma_coreloss_db", "tool": importer_attribution("psma")},
            flags=("user_supplied_source", "per_volume_values_need_core_geometry")))
        summary.rows("core_energy_loss", len(table))
    shards = {"materials": writer.shards(list(materials.values()), "materials", "materials", 100),
              "measurements": writer.shards(measurements, "measurements", "measurements", 100)}
    return finish_source(writer, summary, shards, [])


def user_table(csv_path: str | Path, output_dir: str | Path, *, material: str, mapping: Mapping[str, tuple[str, str]], quantity: str,
               data_class: str = "manufacturer_datasheet", license: str = "proprietary", reference: str | None = None) -> ImportSummary:
    """Convert a CSV with an explicit ``{csv column: (AIPE column, unit as written)}`` mapping into one measurement record.

    Units are converted with the AIPE unit registry (``degC`` -> K, ``mT`` -> T, ``kW/m^3`` -> W/m^3 ...); unknown AIPE
    column names or units raise. Empty cells become null.
    """
    path = Path(csv_path)
    digest = sha256_path(path)
    summary = ImportSummary("user_table", f"user file sha256 {digest}", license, "user-supplied - local use only")
    writer = SourceWriter(Path(output_dir), summary)
    rows_in = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")), skipinitialspace=True))
    columns, rows = [], []
    for src, (name, unit) in mapping.items():
        if name not in COLUMN_UNITS:
            raise ValueError(f"unknown AIPE column {name!r}; known: {sorted(COLUMN_UNITS)}")
        columns.append(column(name, f"user column {src!r}", unit))
    for r in rows_in:
        row = []
        for src, (name, unit) in mapping.items():
            cell = (r.get(src) or "").strip()
            row.append(None if not cell else units.convert(float(cell), unit, COLUMN_UNITS[name]))
        rows.append(row)
    ev = [evidence("evidence.user-file", "measurement", reference or path.name, "User-supplied table.", NOW, license, locator=path.name, sha256=digest)]
    mid = f"material.user-{slug(material)}"
    mat = _identity(mid, material, material, "user_table", ev, license)
    rec = measurement_record(rid=f"measurement.user-{slug(material)}-{slug(path.stem)}", name=f"{material} {quantity} (user file {path.name})",
                             material_ref=mid, license=license, evidence_items=ev, data_class=data_class, source_id="user_table", quantity=quantity,
                             columns=columns, rows=rows, writer=writer, provenance={"activity": "user_table", "tool": importer_attribution("user_table")},
                             flags=("user_supplied_source",))
    summary.rows(quantity, len(rows))
    shards = {"materials": writer.shards([mat], "materials", "materials", 100), "measurements": writer.shards([rec], "measurements", "measurements", 100)}
    return finish_source(writer, summary, shards, [])


__all__ = ["asu_mhz_csv", "psma_coreloss_db", "user_table"]
