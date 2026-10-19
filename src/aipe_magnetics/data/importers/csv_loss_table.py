"""Import a user-supplied CSV core-loss table into a ``core_loss_measurement`` record.

The CSV holds one scalar loss point per row: frequency, peak flux density, loss density and
optionally temperature and DC bias field. **The caller maps every column explicitly** to
``(quantity, unit)``; nothing is inferred from header names::

    columns = {"f_kHz": ("frequency", "kHz"), "B_mT": ("flux_density_peak", "mT"),
               "T_degC": ("temperature", "degC"), "Pv_kW_m3": ("loss_density", "kW/m^3")}

Normalisation: every value becomes an SI :class:`Quantity` (``Hz``, ``T``, ``K``, ``W/m^3``,
``A/m``) through ``Quantity.si`` with the original ``{value, unit}`` retained, so the conversion is
reproducible from the record alone. Rows are rejected (not skipped) when they do not have exactly
one cell per header column, when a required cell is empty, when a cell is not a plain decimal
numeral (``[+-]digits[.digits][e[+-]digits]``; digit separators such as ``1_00``, ``inf``/``nan``
and padded tokens are rejected), when the SI conversion overflows float64, or when a value is
physically impossible (``frequency <= 0``, ``flux_density_peak <= 0``, ``loss_density < 0``,
``temperature <= 0 K``). Empty optional cells are omitted (never zero-filled). A UTF-8 byte-order
mark in front of the header is removed and reported as a warning.

The evidence (reference, licence, locator, ...) is **supplied by the caller** and must be of kind
``measurement`` unless ``synthetic=True``; the record's ``source_ref`` of every point cites it. The
source CSV is recorded as a raw artifact with its SHA-256, and ``extensions["aipe.import"]`` keeps
the column mapping.
"""

from __future__ import annotations

import csv
import io
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from aipe_magnetics.common.errors import InvalidInputError, UnitError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.units import units
from aipe_magnetics.data.catalog import Catalog
from aipe_magnetics.data.importers.base import (
    IMPORT_EXTENSION,
    ImportResult,
    build_record,
    check_against_catalogue,
    coerce_evidence,
    import_extension,
    source_artifact,
    source_bytes_and_digest,
    tool_attribution,
    write_record,
)
from aipe_magnetics.data.records import SCHEMA_VERSION, Evidence, MagneticRecord, Provenance, Quality

IMPORTER_ID = "aipe.magnetics.import.csv_loss_table"
CSV_MEDIA_TYPE = "text/csv"

#: quantity -> canonical SI unit
LOSS_TABLE_QUANTITIES: dict[str, str] = {
    "frequency": "Hz", "flux_density_peak": "T", "temperature": "K", "loss_density": "W/m^3", "dc_bias_field": "A/m",
}
REQUIRED_QUANTITIES: tuple[str, ...] = ("frequency", "flux_density_peak", "loss_density")
Excitation = str  # LossReferencePoint.excitation literal; validated by the record model


def _validate_columns(columns: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    if not isinstance(columns, Mapping) or not columns:
        raise InvalidInputError("columns must be a non-empty mapping CSV header -> (quantity, unit)")
    mapping: dict[str, tuple[str, str]] = {}
    seen: dict[str, str] = {}
    for header, spec in columns.items():
        if not isinstance(header, str) or not header:
            raise InvalidInputError(f"column header must be a non-empty string, got {header!r}")
        if not isinstance(spec, (tuple, list)) or len(spec) != 2 or not all(isinstance(s, str) for s in spec):
            raise InvalidInputError(f"column {header!r}: mapping must be a (quantity, unit) pair of strings, got {spec!r}")
        quantity, unit = spec
        if quantity not in LOSS_TABLE_QUANTITIES:
            raise InvalidInputError(f"column {header!r}: unknown quantity {quantity!r}; allowed: {sorted(LOSS_TABLE_QUANTITIES)}")
        if quantity in seen:
            raise InvalidInputError(f"quantity {quantity!r} is mapped twice ({seen[quantity]!r} and {header!r})")
        si_unit = LOSS_TABLE_QUANTITIES[quantity]
        if not units.compatible(unit, si_unit):  # raises UnitError for unknown units
            raise UnitError(f"column {header!r}: unit {unit!r} is not compatible with {si_unit!r} ({quantity})",
                            details={"header": header, "unit": unit, "quantity": quantity})
        seen[quantity] = header
        mapping[header] = (quantity, unit)
    missing = [q for q in REQUIRED_QUANTITIES if q not in seen]
    if missing:
        raise InvalidInputError(f"column mapping lacks required quantities {missing}", details={"missing": missing})
    return mapping


#: plain decimal numeral: optional sign, digits with optional fraction (or a leading-dot fraction), optional exponent
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _parse_cell(row_number: int, header: str, text: str) -> float:
    if _NUMBER_RE.match(text):
        return float(text)
    try:
        value = float(text)
    except ValueError:
        raise InvalidInputError(f"row {row_number}, column {header!r}: {text!r} is not a number") from None
    if not math.isfinite(value):
        raise InvalidInputError(f"row {row_number}, column {header!r}: non-finite value {text!r} is rejected")
    raise InvalidInputError(f"row {row_number}, column {header!r}: {text!r} is not a plain decimal numeral "
                            "(digit separators, padding and special tokens are rejected)")


def parse_loss_table(text: str, columns: Mapping[str, Any], *, delimiter: str = ",") -> tuple[list[dict[str, Quantity]], list[str]]:
    """Parse CSV text into per-row SI quantities (with originals) using the explicit column mapping."""
    mapping = _validate_columns(columns)
    warnings: list[str] = []
    if text.startswith("﻿"):
        text = text[1:]
        warnings.append("UTF-8 byte-order mark removed in front of the CSV header")
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration:
        raise InvalidInputError("CSV is empty (no header row)") from None
    header = [h.strip() for h in header]
    duplicates = sorted({h for h in header if header.count(h) > 1})
    if duplicates:
        raise InvalidInputError(f"CSV header has duplicate columns {duplicates}", details={"duplicates": duplicates})
    missing = [h for h in mapping if h not in header]
    if missing:
        raise InvalidInputError(f"CSV header lacks mapped columns {missing}; header is {header}",
                                details={"missing": missing, "header": header})
    unmapped = [h for h in header if h not in mapping]
    if unmapped:
        warnings.append(f"unmapped CSV columns ignored: {unmapped}")
    index = {h: header.index(h) for h in mapping}
    rows: list[dict[str, Quantity]] = []
    for number, cells in enumerate(reader, start=2):
        if not cells or all(not c.strip() for c in cells):
            continue  # blank line
        if len(cells) != len(header):
            raise InvalidInputError(f"row {number}: {len(cells)} cells but {len(header)} header columns (rows must match the header exactly)")
        point: dict[str, Quantity] = {}
        for header_name, (quantity, unit) in mapping.items():
            cell = cells[index[header_name]].strip()
            if cell == "":
                if quantity in REQUIRED_QUANTITIES:
                    raise InvalidInputError(f"row {number}, column {header_name!r}: required value is missing")
                continue  # optional missing value: omitted, never zero-filled
            value = _parse_cell(number, header_name, cell)
            si_unit = LOSS_TABLE_QUANTITIES[quantity]
            if not math.isfinite(units.convert(value, unit, si_unit)):
                raise InvalidInputError(f"row {number}, column {header_name!r}: {value!r} {unit} overflows float64 when converted to {si_unit}")
            q = Quantity.si(value, unit)
            if quantity in ("frequency", "flux_density_peak", "temperature") and q.value <= 0:
                raise InvalidInputError(f"row {number}, column {header_name!r}: {quantity} must be positive, got {value} {unit}")
            if quantity == "loss_density" and q.value < 0:
                raise InvalidInputError(f"row {number}, column {header_name!r}: loss_density must be non-negative, got {value} {unit}")
            point[quantity] = q
        rows.append(point)
    if not rows:
        raise InvalidInputError("CSV has a header but no data rows")
    zero_loss = sum(1 for p in rows if p["loss_density"].value == 0)
    if zero_loss:
        warnings.append(f"{zero_loss} row(s) report exactly zero loss density")
    return rows, warnings


def import_csv_loss_table(source: str | Path, *, columns: Mapping[str, Any], record_id: str, name: str, material_ref: str,
                          evidence: Evidence | Mapping[str, Any], destination: str | Path | None = None,
                          catalogue: Catalog | Iterable[Any] | None = None, excitation: Excitation = "sinusoidal",
                          duty: float | None = None, synthetic: bool = False, specimen_core_ref: str | None = None,
                          specimen_description: str | None = None, method: str | None = None, instrument: str | None = None,
                          description: str | None = None, license: str | None = None, tags: Iterable[str] = (),
                          delimiter: str = ",", encoding: str = "utf-8") -> ImportResult:
    """Import a CSV loss table as one ``core_loss_measurement`` record (see module docstring)."""
    source = Path(source)
    if license is not None and not license.strip():
        raise InvalidInputError("license must be a non-empty SPDX expression or 'NOASSERTION'; omit it to inherit the evidence licence")
    raw, digest = source_bytes_and_digest(source)
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError as error:
        raise InvalidInputError(f"{source}: cannot decode as {encoding}: {error}") from None
    rows, warnings = parse_loss_table(text, columns, delimiter=delimiter)
    item = coerce_evidence(evidence)
    if not synthetic and item.kind != "measurement":
        raise InvalidInputError("core_loss_measurement records need evidence of kind 'measurement' (reference, licence, "
                                f"locator supplied by the caller); got kind {item.kind!r}", details={"evidence": item.id})
    if item.locator is None:
        warnings.append("evidence has no locator (file / table / column reference); consider adding one")
    basis = "synthetic" if synthetic else "measurement"
    points = []
    for point in rows:
        entry: dict[str, Any] = {k: q.model_dump(mode="json", exclude_none=True) for k, q in point.items()}
        entry.update({"excitation": excitation, "source_ref": item.id, "basis": basis})
        if duty is not None:
            entry["duty"] = duty
        points.append(entry)
    si_targets = ", ".join(f"{q} -> {u}" for q, u in LOSS_TABLE_QUANTITIES.items() if any(q in r for r in rows))
    normalization = (f"values converted to SI ({si_targets}) with Quantity.si; the original value and unit of every "
                     "converted quantity are retained in its 'original' field")
    column_record = {header: {"quantity": spec[0], "unit": spec[1]} for header, spec in columns.items()}
    properties: dict[str, Any] = {"material_ref": material_ref, "points": points}
    for key, value in (("specimen_core_ref", specimen_core_ref), ("specimen_description", specimen_description),
                       ("method", method), ("instrument", instrument), ("description", description)):
        if value is not None:
            properties[key] = value
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "id": record_id, "kind": "core_loss_measurement", "name": name,
        "synthetic": synthetic, "license": license or item.license, "properties": properties,
        "evidence_refs": [item.id], "evidence": [item.model_dump(mode="json", exclude_none=True)],
        "provenance": Provenance(activity=f"imported core-loss table from {source.name} ({len(points)} points)",
                                 tool=tool_attribution(), inputs=[item.id, material_ref],
                                 normalization=normalization).model_dump(mode="json", exclude_none=True),
        "quality": Quality(status="normalized", flags=(["zero_loss_rows"] if any("zero loss" in w for w in warnings) else []),
                           notes=[]).model_dump(mode="json", exclude_none=True),
        "artifacts": [source_artifact(source, media_type=CSV_MEDIA_TYPE, sha256=digest, role="raw",
                                      description="original loss table as supplied").model_dump(mode="json", exclude_none=True)],
        "tags": list(tags),
        "extensions": {IMPORT_EXTENSION: import_extension(source_name=source.name, source_sha256=digest, importer=IMPORTER_ID,
                                                          normalization=normalization,
                                                          extra={"columns": column_record, "delimiter": delimiter,
                                                                 "rows": len(points)})},
    }
    if description is not None:
        data["description"] = description
    record: MagneticRecord = build_record(data)
    check_against_catalogue(record, catalogue)
    written: list[str] = []
    if destination is not None:
        written.append(str(write_record(record, destination)))
    return ImportResult(importer=IMPORTER_ID, records=[record], artifacts=list(record.artifacts),
                        provenance=record.provenance, warnings=warnings, written=written, source_sha256=digest)  # type: ignore[arg-type]


__all__ = ["IMPORTER_ID", "LOSS_TABLE_QUANTITIES", "REQUIRED_QUANTITIES", "parse_loss_table", "import_csv_loss_table"]
