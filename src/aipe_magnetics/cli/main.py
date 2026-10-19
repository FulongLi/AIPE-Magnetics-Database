"""``aipe-magnetics`` command line interface.

Every subcommand prints an engineering result envelope (JSON) or a human summary; exit code 0 for
``status == ok``, 2 for ``rejected``, 1 for errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from aipe_magnetics import __version__
from aipe_magnetics.common.results import EngineeringResult


def _qty(text: str, name: str) -> dict[str, Any]:
    """Parse ``'100 kHz'`` / ``'0.2 T'`` / ``'100degC'`` into a quantity dict."""
    text = text.strip()
    for index, char in enumerate(text):
        if char.isalpha() and not (char in "eE" and index > 0 and text[index - 1].isdigit() and index + 1 < len(text)
                                   and (text[index + 1].isdigit() or text[index + 1] in "+-")):
            value, unit = text[:index].strip(), text[index:].strip()
            break
    else:
        raise SystemExit(f"{name}: give a value with a unit, e.g. '100 kHz' (got {text!r})")
    try:
        return {"value": float(value), "unit": unit}
    except ValueError:
        raise SystemExit(f"{name}: cannot parse number in {text!r}") from None


def _emit(result: EngineeringResult, args: argparse.Namespace) -> int:
    if getattr(args, "json", False):
        print(json.dumps(result.to_json_dict(), indent=2, ensure_ascii=False))
    else:
        print(_summary(result))
    return 0 if result.status == "ok" else 2


def _summary(result: EngineeringResult) -> str:
    lines = [f"[{result.status}] {result.capability}" + (f"  model={result.model.id}@{result.model.version}" if result.model else "")]
    for key, v in result.values.items():
        lines.append(f"  {v.label} ({key}): {v.quantity.value:.6g} {v.quantity.unit}  [{v.status}]")
    for e in result.errors:
        lines.append(f"  ERROR {e.code}: {e.message}")
    for w in result.warnings:
        lines.append(f"  {w.severity}: {w.code}: {w.message}")
    if result.data.get("materials") is not None:
        for item in result.data["materials"]:
            lines.append(f"  - {item['id']}  grade={item.get('grade')}  class={item.get('material_class')}  sets={item.get('steinmetz_sets')}"
                         f"  synthetic={item.get('synthetic')}")
    if result.data.get("comparison") is not None:
        for row in result.data["comparison"]:
            pv = row.get("loss_density_W_per_m3")
            margin = row.get("saturation_margin")
            pv_text = "n/a" if pv is None else "%.1f kW/m^3" % (pv / 1e3)
            margin_text = "n/a" if margin is None else "%.1f%%" % (margin * 100)
            lines.append("  - %s: Pv=%s  margin=%s  %s" % (row["id"], pv_text, margin_text, "; ".join(row.get("notes", []))))
    if result.data.get("candidates") is not None:
        for c in result.data["candidates"]:
            total = c.get("total_loss_W")
            bpk = c.get("flux_density_peak_T")
            bpk_text = "n/a" if bpk is None else "%.3f T" % bpk
            total_text = "n/a" if total is None else "%.1f W" % total
            lines.append("  - %s: feasible=%s Bpk=%s total=%s %s" % (c.get("core_name", c.get("core")), c.get("feasible"), bpk_text,
                                                                     total_text, "; ".join(c.get("reasons", []))))
    lines.append(f"  verification: {result.verification.level} {result.verification.state} - {result.verification.statement}")
    for lim in result.limitations:
        lines.append(f"  limitation: {lim.statement}")
    return "\n".join(lines)


def _add_json_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="print the full result envelope as JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aipe-magnetics", description="AIPE Magnetics engineering CLI (no AI agent required)")
    parser.add_argument("--version", action="version", version=f"aipe-magnetics {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="package, catalogue and verification summary")

    m = sub.add_parser("materials", help="material lookup and comparison").add_subparsers(dest="action", required=True)
    p = m.add_parser("list"); p.add_argument("--query"); p.add_argument("--class", dest="material_class"); _add_json_flag(p)
    p = m.add_parser("get"); p.add_argument("material"); _add_json_flag(p)
    p = m.add_parser("compare"); p.add_argument("materials", nargs="+"); p.add_argument("--frequency", "-f"); p.add_argument("--flux", "-b")
    p.add_argument("--temperature", "-t"); _add_json_flag(p)

    l = sub.add_parser("loss", help="core-loss estimation").add_subparsers(dest="action", required=True)
    p = l.add_parser("estimate"); p.add_argument("material"); p.add_argument("--frequency", "-f", required=True)
    p.add_argument("--flux", "-b", required=True, help="peak flux density, e.g. '200 mT'"); p.add_argument("--temperature", "-t")
    p.add_argument("--waveform", default="sinusoidal", choices=["sinusoidal", "triangular", "trapezoidal"]); p.add_argument("--duty", type=float)
    p.add_argument("--duty-rise", type=float); p.add_argument("--duty-flat", type=float)
    p.add_argument("--model", default="auto", choices=["auto", "steinmetz", "igse", "bertotti"]); p.add_argument("--core")
    p.add_argument("--parameter-set"); p.add_argument("--policy", default="reject", choices=["reject", "warn"]); _add_json_flag(p)

    h = sub.add_parser("hysteresis", help="Jiles-Atherton loop evaluation").add_subparsers(dest="action", required=True)
    p = h.add_parser("evaluate"); p.add_argument("--field-peak", required=True, help="e.g. '500 A/m'"); p.add_argument("--frequency", default="1 Hz")
    p.add_argument("--field-dc"); p.add_argument("--periods", type=int, default=10, help="maximum periods to reach steady state")
    p.add_argument("--waveform", default="sinusoidal", choices=["sinusoidal", "triangular"])
    p.add_argument("--param", action="append", default=[], help="override parameter, e.g. --param Ms=1.6e6 (repeatable)"); _add_json_flag(p)

    d = sub.add_parser("design", help="inductor / transformer design").add_subparsers(dest="action", required=True)
    p = d.add_parser("inductor"); p.add_argument("--inductance", "-L", required=True); p.add_argument("--current-peak", required=True)
    p.add_argument("--current-rms", required=True); p.add_argument("--ripple"); p.add_argument("--frequency", "-f", required=True)
    p.add_argument("--material", required=True); p.add_argument("--core"); p.add_argument("--temperature", "-t")
    p.add_argument("--conductor", default="round", choices=["round", "litz", "foil"]); p.add_argument("--flux-fraction", type=float); _add_json_flag(p)
    p = d.add_parser("transformer"); p.add_argument("--power", "-P", required=True); p.add_argument("--frequency", "-f", required=True)
    p.add_argument("--primary-voltage", required=True); p.add_argument("--secondary-voltage", required=True, action="append")
    p.add_argument("--material", required=True); p.add_argument("--core", action="append"); p.add_argument("--temperature", "-t")
    p.add_argument("--waveform", default="square", choices=["square", "sinusoidal", "trapezoidal"]); p.add_argument("--duty", type=float, default=0.5)
    p.add_argument("--conductor", default="litz", choices=["round", "litz", "foil"]); p.add_argument("--flux-fraction", type=float)
    p.add_argument("--loss-target", help="core-loss density ceiling, e.g. '300 kW/m^3'"); _add_json_flag(p)

    p = sub.add_parser("validate", help="validate an existing design"); p.add_argument("--material", required=True); p.add_argument("--core", required=True)
    p.add_argument("--turns", type=int, required=True); p.add_argument("--frequency", "-f", required=True); p.add_argument("--temperature", "-t")
    p.add_argument("--voltage-peak"); p.add_argument("--flux"); p.add_argument("--current-peak"); p.add_argument("--air-gap")
    p.add_argument("--waveform", default="square", choices=["square", "sinusoidal", "trapezoidal"]); _add_json_flag(p)

    p = sub.add_parser("report", help="render result JSON files into a report"); p.add_argument("results", nargs="+", type=Path)
    p.add_argument("--format", default="markdown", choices=["markdown", "json"]); p.add_argument("--output", type=Path); p.add_argument("--overwrite", action="store_true")
    p.add_argument("--title", default="AIPE Magnetics engineering report")

    p = sub.add_parser("verify", help="run the verification suite and write reports"); p.add_argument("--output", type=Path, default=Path("reports/verification"))
    p.add_argument("--capability"); p.add_argument("--check-bundled", action="store_true", help="fail if the bundled evidence differs from this run")
    p.add_argument("--update-bundled", action="store_true", help="rewrite the bundled evidence file from this run (maintainers)")

    p = sub.add_parser("demo", help="executable demonstrations"); p.add_argument("name", choices=["dab-10kw"]); p.add_argument("--output", type=Path)
    p.add_argument("--temperature", "-t", default="100 degC"); p.add_argument("--json", action="store_true")

    s = sub.add_parser("schema", help="record schema tools").add_subparsers(dest="action", required=True)
    p = s.add_parser("validate"); p.add_argument("records", nargs="+", type=Path)
    p = s.add_parser("migrate"); p.add_argument("source", type=Path); p.add_argument("destination", type=Path); p.add_argument("--overwrite", action="store_true")
    p = s.add_parser("export"); p.add_argument("directory", type=Path); p.add_argument("--check", action="store_true")

    i = sub.add_parser("import", help="data import").add_subparsers(dest="action", required=True)
    p = i.add_parser("json"); p.add_argument("source", type=Path); p.add_argument("destination", type=Path); p.add_argument("--catalogue", nargs="*", type=Path, default=[])
    p.add_argument("--migrate", action="store_true", help="accept 0.1.0 records and migrate them")

    p = sub.add_parser("mcp", help="MCP server").add_subparsers(dest="action", required=True)
    q = p.add_parser("serve"); q.add_argument("--allow-write", action="store_true"); q.add_argument("--output-dir")
    q = p.add_parser("self-test")
    return parser


def _winding(conductor: str) -> dict[str, Any]:
    return {"conductor_type": conductor}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    import aipe_magnetics.api as api

    if args.command == "info":
        from aipe_magnetics.data.catalog import Catalog
        from aipe_magnetics.verification.suite import bundled_evidence_path, load_bundled_registry
        cat = Catalog.default()
        reg = load_bundled_registry()
        print(f"aipe-magnetics {__version__}")
        print(f"catalogue: {len(cat.materials())} materials, {len(cat.cores())} cores, {len(cat.manufacturers())} manufacturers")
        print(f"verification evidence: {bundled_evidence_path()} ({len(reg.records)} records, {len(reg.capabilities())} capabilities)")
        for cap, info in reg.summary().items():
            print(f"  {cap}: highest passing level {info['highest_passing_level']} (pass {info['pass']}, fail {info['fail']})")
        return 0
    if args.command == "materials":
        if args.action == "list":
            return _emit(api.list_materials(args.query, args.material_class), args)
        if args.action == "get":
            return _emit(api.get_material(args.material), args)
        return _emit(api.compare_materials(args.materials, _qty(args.frequency, "frequency") if args.frequency else None,
                                           _qty(args.flux, "flux") if args.flux else None,
                                           _qty(args.temperature, "temperature") if args.temperature else None), args)
    if args.command == "loss":
        waveform: dict[str, Any] = {"kind": args.waveform}
        if args.duty is not None:
            waveform["duty"] = args.duty
        if args.duty_rise is not None:
            waveform.update(duty_rise=args.duty_rise, duty_flat=args.duty_flat if args.duty_flat is not None else 0.0)
        op: dict[str, Any] = {"frequency": _qty(args.frequency, "frequency"), "flux_density_peak": _qty(args.flux, "flux"), "waveform": waveform}
        if args.temperature:
            op["temperature"] = _qty(args.temperature, "temperature")
        return _emit(api.estimate_core_loss({"material": args.material, "operating_point": op, "model": args.model, "core": args.core,
                                             "parameter_set": args.parameter_set, "policy": args.policy}), args)
    if args.command == "hysteresis":
        params = {k: float(v) for k, v in (item.split("=", 1) for item in args.param)} or None
        payload: dict[str, Any] = {"field_peak": _qty(args.field_peak, "field-peak"), "frequency": _qty(args.frequency, "frequency"),
                                   "periods": args.periods, "waveform": args.waveform, "parameters": params}
        if args.field_dc:
            payload["field_dc"] = _qty(args.field_dc, "field-dc")
        return _emit(api.evaluate_hysteresis(payload), args)
    if args.command == "design" and args.action == "inductor":
        payload = {"inductance": _qty(args.inductance, "inductance"), "current_peak": _qty(args.current_peak, "current-peak"),
                   "current_rms": _qty(args.current_rms, "current-rms"), "frequency": _qty(args.frequency, "frequency"), "material": args.material,
                   "core": args.core, "winding": _winding(args.conductor)}
        if args.ripple:
            payload["current_ripple_peak_to_peak"] = _qty(args.ripple, "ripple")
        if args.temperature:
            payload["temperature"] = _qty(args.temperature, "temperature")
        if args.flux_fraction:
            payload["flux_limit"] = {"mode": "fraction_of_saturation", "fraction": args.flux_fraction}
        return _emit(api.design_inductor(payload), args)
    if args.command == "design" and args.action == "transformer":
        payload = {"power": _qty(args.power, "power"), "frequency": _qty(args.frequency, "frequency"),
                   "primary": {"name": "primary", "voltage": _qty(args.primary_voltage, "primary-voltage"), "winding": _winding(args.conductor)},
                   "secondaries": [{"name": f"secondary{i + 1 if len(args.secondary_voltage) > 1 else ''}", "voltage": _qty(v, "secondary-voltage"),
                                    "winding": _winding(args.conductor)} for i, v in enumerate(args.secondary_voltage)],
                   "material": args.material, "cores": args.core, "voltage_waveform": args.waveform, "duty": args.duty}
        if args.temperature:
            payload["temperature"] = _qty(args.temperature, "temperature")
        if args.flux_fraction:
            payload["flux_limit"] = {"mode": "fraction_of_saturation", "fraction": args.flux_fraction}
        if args.loss_target:
            payload["loss_density_target"] = _qty(args.loss_target, "loss-target")
        return _emit(api.design_transformer(payload), args)
    if args.command == "validate":
        payload = {"material": args.material, "core": args.core, "turns": args.turns, "frequency": _qty(args.frequency, "frequency"),
                   "voltage_waveform": args.waveform}
        for key, value in (("temperature", args.temperature), ("voltage_peak", args.voltage_peak), ("flux_density_peak", args.flux),
                           ("current_peak", args.current_peak), ("air_gap", args.air_gap)):
            if value:
                payload[key] = _qty(value, key)
        return _emit(api.validate_design(payload), args)
    if args.command == "report":
        from aipe_magnetics.common.jsonio import load
        results = [EngineeringResult.model_validate(load(p)) for p in args.results]
        res = api.generate_report(results, {"title": args.title, "format": args.format, "output_path": str(args.output) if args.output else None,
                                            "allow_write": bool(args.output), "overwrite": args.overwrite})
        if res.status != "ok":
            print(_summary(res)); return 2
        if args.output:
            print(f"wrote {res.data['written_to']}")
        else:
            print(res.data["content"])
        return 0
    if args.command == "verify":
        from aipe_magnetics.verification import release as rel
        return rel.main_cli(output=args.output, capability=args.capability, check_bundled=args.check_bundled, update_bundled=args.update_bundled)
    if args.command == "demo":
        from aipe_magnetics.engineering.demo_dab import run_dab_demo
        return run_dab_demo(output=args.output, temperature=_qty(args.temperature, "temperature"), as_json=args.json)
    if args.command == "schema":
        from aipe_magnetics.data import schema as sc
        if args.action == "validate":
            from aipe_magnetics.common.jsonio import load
            failed = False
            for path in args.records:
                errors = sc.validate_record(load(path))
                print(("FAIL " if errors else "PASS ") + str(path))
                for e in errors:
                    print("  " + e)
                failed |= bool(errors)
            return 1 if failed else 0
        if args.action == "migrate":
            from aipe_magnetics.data import migrate as mg
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
        from aipe_magnetics.common.jsonio import load
        from aipe_magnetics.data.importers.json_record import import_json_record
        catalogue = [load(p) for p in args.catalogue] or None
        result = import_json_record(args.source, args.destination, catalogue=catalogue, migrate_v01=args.migrate)
        print(f"imported {args.source} -> {args.destination} ({len(result.records)} record(s), source sha256 {result.source_sha256})")
        for warning in result.warnings:
            print(f"  warning: {warning}")
        return 0
    if args.command == "mcp":
        from aipe_magnetics.mcp import server as srv
        if args.action == "self-test":
            return srv.main(["--self-test"])
        argv2 = []
        if args.allow_write:
            argv2 += ["--allow-write", "--output-dir", args.output_dir or "."]
        return srv.main(argv2)
    parser.error("unknown command")  # pragma: no cover
    return 1


if __name__ == "__main__":
    sys.exit(main())
