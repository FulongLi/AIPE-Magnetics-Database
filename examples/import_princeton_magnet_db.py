#!/usr/bin/env python
"""Complete, executable import of one Princeton MagNet scalar core-loss database (genuine external data).

Usage::

    python examples/import_princeton_magnet_db.py <checkout_or_data_dir> <out_dir> [--material N87] [--max-json-points 200]
                                                  [--temperature-degC 25] [--excitation sinusoidal]

``<checkout_or_data_dir>`` is a clone of https://github.com/PrincetonUniversity/magnet (pinned commit
8a6892e91ac622e42f8b9182a6dfe9094ceea97a) or any directory holding ``<material>_database.h5``. Nothing
is downloaded (pass a checkout; on-demand download exists in the adapter and needs an explicit opt-in).

What it does: discover -> inspect (checksum against the pinned file) -> extract (explicit column units)
-> validate -> write a ``dataset`` record, a ``magnetic_material`` identity record, a ``measurement``
record, the full table as an HDF5 artifact (``aipe.magnetics.tables/0.2``) and a ``core_loss_measurement``
record with at most ``--max-json-points`` points, then re-validates everything that was written and
prints row counts, units, checksums, licence and attribution. A file that differs from the pinned
upstream file is refused unless ``--allow-unpinned`` is given (records are then flagged
``source_differs_from_pinned_commit`` and versioned by the file's SHA-256).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from aipe_magnetics.common.errors import AIPEMagneticsError
from aipe_magnetics.common.jsonio import load
from aipe_magnetics.data.records import MagneticRecord
from aipe_magnetics.data.schema import validate_catalogue, validate_record
from aipe_magnetics.data.sources import LocalCache, PrincetonMagnetAdapter, TableStore


def run(source: Path, out: Path, *, material: str, max_json_points: int, temperature_degc: float | None = None,
        excitation: str | None = None, quiet: bool = False, allow_unpinned: bool = False) -> dict:
    adapter = PrincetonMagnetAdapter(LocalCache(out / ".aipe-cache"))
    items = [i for i in adapter.discover(source, materials=[material]) if i.available]
    if not items:
        raise FileNotFoundError(f"{material}_database.h5 not found under {source}")
    item = items[0]
    info = adapter.inspect(item)
    extracted = adapter.extract(item, temperature_degC=temperature_degc, excitation=excitation)
    problems = adapter.validate(extracted)
    result = adapter.to_records(extracted, destination_dir=out, max_json_points=max_json_points, allow_unpinned=allow_unpinned)

    # independent re-validation of what was written
    records = [MagneticRecord.model_validate(load(p)) for p in result.written if p.endswith(".json")]
    errors = [e for r in records for e in validate_record(r.to_json_dict())] + validate_catalogue(records)
    table = [p for p in result.written if p.endswith(".h5")][0]
    mismatches = TableStore().verify_checksums(table)
    if errors or mismatches:
        raise RuntimeError("post-write validation failed: " + "; ".join(errors + mismatches))
    loss = next(r for r in result.records if r.kind == "core_loss_measurement")
    dataset = next(r for r in result.records if r.kind == "dataset")
    raw_artifact = next(a for a in dataset.artifacts if a.role == "raw")
    summary = {
        "source_file": info["path"], "source_sha256": info["sha256"], "matches_pinned_commit": info["matches_pinned_commit"],
        "pinned": dataset.extensions["aipe.sources"]["pinned"], "version": dataset.properties["version"], "raw_artifact_uri": raw_artifact.uri,
        "n_points_file": info["n_points"], "n_points_extracted": extracted.n_rows, "excitation_counts": extracted.metadata["excitation_counts"],
        "units": extracted.unit_mapping(), "validation_problems": problems, "records": [(r.kind, r.id) for r in result.records],
        "table_artifact": table, "json_points": len(loss.properties["points"]), "license": dataset.license,
        "citation": dataset.properties.get("citation"), "written": result.written, "warnings": result.warnings,
    }
    if not quiet:
        print(f"Princeton MagNet import of {material} from {info['path']}")
        print(f"  source sha256 {info['sha256']} (matches pinned commit {adapter.descriptor.pinned_commit[:12]}: {info['matches_pinned_commit']})")
        print(f"  records versioned as: {summary['version']} (pinned: {summary['pinned']}); raw artifact URI: {raw_artifact.uri}")
        print(f"  rows: {info['n_points']} in file, {extracted.n_rows} extracted {extracted.metadata['excitation_counts']}")
        print(f"  units (as stored upstream -> SI in the artifact): {extracted.unit_mapping()} (temperature degC -> K)")
        print(f"  validation problems: {problems or 'none'}")
        print(f"  HDF5 table artifact: {table} (per-array SHA-256 verified)")
        for kind, record_id in summary["records"]:
            print(f"  record: {kind:22s} {record_id}")
        print(f"  core_loss_measurement JSON holds {summary['json_points']} points; the full table stays in the HDF5 artifact")
        print(f"  licence: {dataset.license}; redistribution permitted: {dataset.properties.get('redistribution_permitted')}")
        print(f"  citation: {summary['citation']}")
        for ref in adapter.attribution():
            print(f"  attribution: {ref.reference} ({ref.url} @ {ref.version}, {ref.license})")
        if result.warnings:
            print(f"  warnings: {result.warnings}")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import a Princeton MagNet scalar core-loss database into AIPE v0.2 records.")
    parser.add_argument("source", type=Path, help="magnet checkout or directory with <material>_database.h5 files")
    parser.add_argument("out", type=Path, help="output directory (new files only; never overwrites)")
    parser.add_argument("--material", default="N87")
    parser.add_argument("--max-json-points", type=int, default=200)
    parser.add_argument("--temperature-degC", type=float, default=None, dest="temperature_degc", help="keep only this set-point temperature")
    parser.add_argument("--excitation", choices=["sinusoidal", "triangular", "trapezoidal"], default=None)
    parser.add_argument("--allow-unpinned", action="store_true", dest="allow_unpinned",
                        help="import a file that differs from the pinned upstream file (records are flagged and versioned by SHA-256)")
    args = parser.parse_args(argv)
    try:
        run(args.source, args.out, material=args.material, max_json_points=args.max_json_points, temperature_degc=args.temperature_degc,
            excitation=args.excitation, allow_unpinned=args.allow_unpinned)
    except (AIPEMagneticsError, FileExistsError, FileNotFoundError, RuntimeError) as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
