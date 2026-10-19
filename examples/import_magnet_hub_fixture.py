#!/usr/bin/env python
"""Complete, executable import of the MagNet-Hub measured loss fixture (genuine external data, MIT).

Usage::

    python examples/import_magnet_hub_fixture.py <checkout_root> <out_dir> [--materials N87 3C90]

``<checkout_root>`` is a clone of https://github.com/upb-lea/mag-net-hub at the pinned commit
1ed635d7bfac3c2aa024471af363beadc05f8932. The fixture ``tests/test_files/unit_test_data_ploss_at_450kWpm3.csv``
holds 90 measured single-period flux-density waveforms (1024 samples, T) with frequency (Hz),
temperature (degC) and measured loss density (W/m^3) for 15 materials.

The adapter writes one waveform file per material (layout ``aipe.magnetics.waveforms/0.2``, SI units,
per-array SHA-256), one ``waveform_dataset`` record per material, identity ``magnetic_material`` records
and a ``dataset`` record with checksums, licence and attribution. Nothing is downloaded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from aipe_magnetics.common.errors import AIPEMagneticsError
from aipe_magnetics.common.jsonio import load
from aipe_magnetics.data.hdf5 import WaveformStore
from aipe_magnetics.data.records import MagneticRecord
from aipe_magnetics.data.schema import validate_catalogue, validate_record
from aipe_magnetics.data.sources import LocalCache, MagnetHubAdapter


def run(checkout: Path, out: Path, *, materials: Sequence[str] | None = None, quiet: bool = False, allow_unpinned: bool = False) -> dict:
    adapter = MagnetHubAdapter(LocalCache(out / ".aipe-cache"))
    fixture = [i for i in adapter.discover(checkout) if i.extra.get("fixture") == "ploss"]
    if not fixture:
        raise FileNotFoundError(f"fixture CSV not found under {checkout}")
    info = adapter.inspect(fixture[0])
    result = adapter.import_test_fixture(checkout, out, materials=materials, allow_unpinned=allow_unpinned)
    records = [MagneticRecord.model_validate(load(p)) for p in result.written if p.endswith(".json")]
    errors = [e for r in records for e in validate_record(r.to_json_dict())] + validate_catalogue(records)
    store = WaveformStore()
    mismatches = [m for p in result.written if p.endswith(".h5") for m in store.verify_checksums(p)]
    if errors or mismatches:
        raise RuntimeError("post-write validation failed: " + "; ".join(errors + mismatches))
    dataset = next(r for r in result.records if r.kind == "dataset")
    waveforms = [r for r in result.records if r.kind == "waveform_dataset"]
    raw_artifact = next(a for a in dataset.artifacts if a.role == "raw")
    summary = {
        "fixture": info["path"], "sha256": info["sha256"], "matches_pinned_commit": info["matches_pinned_commit"], "rows_in_file": info["rows"],
        "pinned": dataset.extensions["aipe.sources"]["pinned"], "version": dataset.properties["version"], "raw_artifact_uri": raw_artifact.uri,
        "materials_in_file": info["materials"], "imported_rows": dataset.properties["record_count"], "materials_imported": dataset.properties["materials"],
        "column_units": info["column_units"], "written": result.written, "records": [(r.kind, r.id) for r in result.records],
        "license": dataset.license, "citation": dataset.properties.get("citation"), "warnings": result.warnings,
    }
    if not quiet:
        print(f"MagNet-Hub fixture import from {info['path']}")
        print(f"  sha256 {info['sha256']} (matches pinned commit {adapter.descriptor.pinned_commit[:12]}: {info['matches_pinned_commit']})")
        print(f"  records versioned as: {summary['version']} (pinned: {summary['pinned']}); raw artifact URI: {raw_artifact.uri}")
        print(f"  rows in file: {info['rows']} ({len(info['materials'])} materials); imported: {summary['imported_rows']} rows of {summary['materials_imported']}")
        print(f"  column units (upstream): {info['column_units']} -> stored as flux_density [T], frequency [Hz], temperature [K], loss_density [W/m^3]")
        for record in waveforms:
            props = record.typed_properties()
            print(f"  waveform file: {record.artifacts[0].uri}: {props.n_sequences} x {props.samples_per_sequence} samples, "
                  f"conditions {sorted(props.conditions)}, sha256 {record.artifacts[0].sha256[:16]}...")
        for kind, record_id in summary["records"]:
            print(f"  record: {kind:18s} {record_id}")
        print(f"  licence: {dataset.license}; redistribution permitted: {dataset.properties.get('redistribution_permitted')}")
        print(f"  citation: {summary['citation']}")
        for ref in adapter.attribution():
            print(f"  attribution: {ref.reference} ({ref.url} @ {ref.version}, {ref.license})")
        if result.warnings:
            print(f"  warnings: {result.warnings}")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import the MagNet-Hub measured loss fixture into AIPE v0.2 records.")
    parser.add_argument("checkout", type=Path, help="mag-net-hub checkout root")
    parser.add_argument("out", type=Path, help="output directory (new files only; never overwrites)")
    parser.add_argument("--materials", nargs="*", default=None, help="subset of materials (default: all 15)")
    parser.add_argument("--allow-unpinned", action="store_true", dest="allow_unpinned",
                        help="import a fixture that differs from the pinned upstream file (records are flagged and versioned by SHA-256)")
    args = parser.parse_args(argv)
    try:
        run(args.checkout, args.out, materials=args.materials or None, allow_unpinned=args.allow_unpinned)
    except (AIPEMagneticsError, FileExistsError, FileNotFoundError, RuntimeError) as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
