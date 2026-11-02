"""``aipe-magnetics-db`` command line interface (read-only data access, validation and the refresh pipeline).

Output is JSON (``--json``) or a short human-readable table. Exit codes: 0 success, 1 error or invalid record (one
``ERROR <code>: <message>`` line on stderr, never a traceback for user errors), 2 validation failures.

The v0.2 data commands ``schema validate|migrate|export`` and ``import json`` are kept unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from aipe_magnetics_database import __version__
from aipe_magnetics_database.common.errors import AIPEMagneticsError


def _range(value: str | None) -> Any:
    return value  # store.interval parses "100 kHz" and "50 kHz..200 kHz"


def _print(payload: Any, args: argparse.Namespace, table: list[str] | None = None) -> None:
    if getattr(args, "json", False) or table is None or not isinstance(payload, list):
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return
    for item in payload:
        print("  ".join(str(item.get(k, "")) for k in table))
    print(f"({len(payload)} item(s))", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aipe-magnetics-db", description="AIPE Magnetics Database: query, validate and refresh magnetic data")
    parser.add_argument("--version", action="version", version=f"aipe-magnetics-database {__version__}")
    parser.add_argument("--data", help="data directory (default: bundled data or the checkout's data/)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="record counts per source and of the canonical catalogue"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("sources", help="source inventory with status and licence"); p.add_argument("--status"); p.add_argument("--json", action="store_true")

    m = sub.add_parser("materials", help="materials").add_subparsers(dest="action", required=True)
    p = m.add_parser("list"); p.add_argument("--query"); p.add_argument("--manufacturer"); p.add_argument("--class", dest="material_class")
    p.add_argument("--source"); p.add_argument("--source-records", action="store_true", help="list source records instead of canonical materials")
    p.add_argument("--json", action="store_true")
    p = m.add_parser("get"); p.add_argument("material"); p.add_argument("--json", action="store_true")
    p = m.add_parser("properties", help="properties of every contributing source and fitted loss models"); p.add_argument("material")
    p.add_argument("--json", action="store_true")

    c = sub.add_parser("cores", help="core shapes and parts").add_subparsers(dest="action", required=True)
    p = c.add_parser("list"); p.add_argument("--query"); p.add_argument("--family"); p.add_argument("--manufacturer"); p.add_argument("--material")
    p.add_argument("--kind", choices=["all", "shapes", "parts"], default="all"); p.add_argument("--limit", type=int, default=50)
    p.add_argument("--json", action="store_true")
    p = c.add_parser("get"); p.add_argument("core"); p.add_argument("--json", action="store_true")

    ms = sub.add_parser("measurements", help="measurement tables").add_subparsers(dest="action", required=True)
    p = ms.add_parser("find"); p.add_argument("--material"); p.add_argument("--quantity"); p.add_argument("--data-class"); p.add_argument("--source")
    p.add_argument("--frequency", "-f", help="'100 kHz' or '50 kHz..200 kHz'"); p.add_argument("--temperature", "-t", help="'90 degC' or range")
    p.add_argument("--flux-density", "-b", help="peak flux density, e.g. '0.1 T..0.2 T'"); p.add_argument("--dc-bias", help="e.g. '0 A/m..50 A/m'")
    p.add_argument("--waveform"); p.add_argument("--include-duplicates", action="store_true"); p.add_argument("--json", action="store_true")
    for name in ("table",):
        p = ms.add_parser(name, help="rows of one measurement table"); p.add_argument("record_id")
        p.add_argument("--frequency", "-f"); p.add_argument("--temperature", "-t"); p.add_argument("--flux-density", "-b"); p.add_argument("--dc-bias")
        p.add_argument("--waveform"); p.add_argument("--limit", type=int, default=20); p.add_argument("--json", action="store_true")
        p.add_argument("--csv", type=Path, help="write the selected rows to a CSV file")

    p = sub.add_parser("provenance", help="evidence, licence, quality and duplicate status of a record"); p.add_argument("record_id")
    p = sub.add_parser("record", help="print one v0.2 record"); p.add_argument("record_id")
    p = sub.add_parser("export", help="export records (tables inlined) to JSONL"); p.add_argument("output", type=Path)
    p.add_argument("record_ids", nargs="+"); p.add_argument("--no-rows", action="store_true")

    p = sub.add_parser("validate", help="schema-validate the data store (all source records, references, canonical layer)")
    p = sub.add_parser("refresh", help="re-run the pinned importers, deduplication and reports (checkout only)")
    p.add_argument("--source", action="append", help="source id (repeatable; default: all integrated sources)")
    p.add_argument("--local", action="append", default=[], metavar="SOURCE=DIR", help="use verified local copies of the pinned inputs")
    p.add_argument("--allow-download", action="store_true", help="download missing pinned inputs into the cache (SHA-256 verified)")
    p.add_argument("--cache", help="cache directory (default AIPE_MAGNETICS_CACHE or .aipe-cache)")

    s = sub.add_parser("schema", help="record schema tools").add_subparsers(dest="action", required=True)
    p = s.add_parser("validate"); p.add_argument("records", nargs="+", type=Path)
    p = s.add_parser("migrate"); p.add_argument("source", type=Path); p.add_argument("destination", type=Path); p.add_argument("--overwrite", action="store_true")
    p = s.add_parser("export"); p.add_argument("directory", type=Path); p.add_argument("--check", action="store_true")
    i = sub.add_parser("import", help="data import").add_subparsers(dest="action", required=True)
    p = i.add_parser("json"); p.add_argument("source", type=Path); p.add_argument("destination", type=Path); p.add_argument("--catalogue", nargs="*", type=Path, default=[])
    p.add_argument("--migrate", action="store_true", help="migrate a v0.1 record to v0.2 first")
    u = sub.add_parser("user-import", help="convert a user-supplied file of a non-redistributable source (output stays local)").add_subparsers(dest="action", required=True)
    for name in ("asu-mhz-csv", "psma-coreloss-db"):
        p = u.add_parser(name); p.add_argument("path", type=Path); p.add_argument("output", type=Path)
    p = sub.add_parser("mcp", help="MCP server").add_subparsers(dest="action", required=True)
    p.add_parser("serve"); p.add_parser("self-test")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except AIPEMagneticsError as error:
        print(f"ERROR {error.code}: {error.message}", file=sys.stderr)
        return 1
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"ERROR {type(error).__name__}: {error}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:  # noqa: C901 - flat command table
    import aipe_magnetics_database.api as api

    root = args.data
    if args.command == "info":
        _print(api.summary(root=root), args)
        return 0
    if args.command == "sources":
        _print(api.list_sources(status=args.status, root=root), args, ["id", "status", "license", "version"])
        return 0
    if args.command == "materials":
        if args.action == "list":
            _print(api.list_materials(query=args.query, manufacturer=args.manufacturer, material_class=args.material_class, source=args.source,
                                      canonical=not args.source_records, root=root), args, ["id", "name", "material_class", "sources"])
        elif args.action == "get":
            _print(api.get_material(args.material, root=root), args)
        else:
            _print(api.material_properties(args.material, root=root), args)
        return 0
    if args.command == "cores":
        if args.action == "list":
            _print(api.list_cores(query=args.query, family=args.family, manufacturer=args.manufacturer, material=args.material, kind=args.kind,
                                  limit=args.limit, root=root), args, ["id", "name", "part_number", "manufacturer"])
        else:
            _print(api.get_core(args.core, root=root), args)
        return 0
    if args.command == "measurements":
        if args.action == "find":
            items = api.find_measurements(material=args.material, quantity=args.quantity, data_class=args.data_class, source=args.source,
                                          frequency=args.frequency, temperature=args.temperature, flux_density=args.flux_density,
                                          dc_bias=args.dc_bias, waveform=args.waveform, include_duplicates=args.include_duplicates, root=root)
            _print(items, args, ["id", "quantity", "data_class", "row_count", "status"])
            return 0
        table = api.get_table(args.record_id, frequency=args.frequency, temperature=args.temperature, flux_density=args.flux_density,
                              dc_bias=args.dc_bias, waveform=args.waveform, limit=None if args.csv else args.limit, root=root)
        if args.csv:
            import csv
            with open(args.csv, "w", newline="", encoding="utf-8") as handle:
                w = csv.writer(handle)
                w.writerow([f"{c['name']} [{c['unit']}]" if c["unit"] else c["name"] for c in table["columns"]])
                w.writerows(table["rows"])
            print(f"wrote {len(table['rows'])} row(s) to {args.csv}")
            return 0
        if args.json:
            _print(table, args)
        else:
            print("  ".join(f"{c['name']}[{c['unit']}]" if c["unit"] else c["name"] for c in table["columns"]))
            for row in table["rows"]:
                print("  ".join("" if v is None else str(v) for v in row))
            print(f"({table['returned_rows']} of {table['total_rows']} row(s))", file=sys.stderr)
        return 0
    if args.command == "provenance":
        _print(api.provenance(args.record_id, root=root), args)
        return 0
    if args.command == "record":
        _print(api.get_record(args.record_id, root=root), args)
        return 0
    if args.command == "export":
        _print(api.export(args.record_ids, args.output, include_rows=not args.no_rows, root=root), args)
        return 0
    if args.command == "validate":
        return _validate(root)
    if args.command == "refresh":
        from aipe_magnetics_database.ingest.pipeline import refresh
        local = dict(item.split("=", 1) for item in args.local)
        report = refresh(args.source, local=local, allow_download=args.allow_download, cache_dir=args.cache)
        print(json.dumps(report["canonical"], indent=2))
        return 0
    if args.command == "schema":
        from aipe_magnetics_database.data import schema as sc
        if args.action == "validate":
            from aipe_magnetics_database.common.jsonio import load
            failed = False
            for path in args.records:
                errors = sc.validate_record(load(path))
                print(("FAIL " if errors else "PASS ") + str(path))
                for e in errors:
                    print("  " + e)
                failed |= bool(errors)
            return 1 if failed else 0  # v0.2 exit code kept
        if args.action == "migrate":
            from aipe_magnetics_database.data import migrate as mg
            mg.migrate_file(args.source, args.destination, overwrite=args.overwrite)
            print(f"migrated {args.source} -> {args.destination}")
            return 0
        if args.check:
            problems = sc.check_exported_schemas(args.directory)
            print("\n".join(problems) if problems else f"schemas in {args.directory} match a fresh export")
            return 1 if problems else 0
        written = sc.export_schemas(args.directory)
        print(f"exported {len(written)} schema file(s) to {args.directory}")
        return 0
    if args.command == "import":
        from aipe_magnetics_database.common.jsonio import load
        from aipe_magnetics_database.data.importers.json_record import import_json_record
        catalogue = [load(p) for p in args.catalogue] or None
        result = import_json_record(args.source, args.destination, catalogue=catalogue, migrate_v01=args.migrate)
        print(f"imported {args.source} -> {args.destination} ({len(result.records)} record(s), source sha256 {result.source_sha256})")
        for warning in result.warnings:
            print(f"  warning: {warning}")
        return 0
    if args.command == "user-import":
        from aipe_magnetics_database.ingest import external
        fn = external.asu_mhz_csv if args.action == "asu-mhz-csv" else external.psma_coreloss_db
        summary = fn(args.path, args.output)
        print(json.dumps(summary.to_json() | {"note": "local output only - do not redistribute"}, indent=2))
        return 0
    if args.command == "mcp":
        from aipe_magnetics_database.mcp import server
        return server.main(["--self-test"] if args.action == "self-test" else [])
    raise AssertionError(args.command)  # pragma: no cover


def _validate(root: str | None) -> int:
    from aipe_magnetics_database.ingest.pipeline import validate_tree
    from aipe_magnetics_database.store import MagneticDatabase
    db = MagneticDatabase(root)
    problems = validate_tree(db.normalized, [idx["_directory"] for idx in db.source_indexes.values()])
    from aipe_magnetics_database.data.records import MagneticRecord
    for line in (db.canonical_dir / "materials.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            MagneticRecord.model_validate_json(line)
        except Exception as error:  # noqa: BLE001
            problems.append(f"canonical: {error}")
    print(json.dumps({"data_root": str(db.root), "problems": problems[:50], "problem_count": len(problems),
                      "note": "schema and reference validation only - not an independent experimental verification"}, indent=2))
    return 2 if problems else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
