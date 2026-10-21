"""``aipe-magnetics`` command line interface.

Every subcommand prints an engineering result envelope (JSON) or a human summary; exit code 0 for
``status == ok``, 2 for ``rejected`` (including a design that fails validation checks), 1 for errors.
Ordinary user errors (unreadable files, malformed options, non-envelope inputs) are reported as one
``ERROR <code>: <message>`` line on stderr with exit code 1, never as a traceback. The data layer's
never-overwrite contract is kept as an exception: ``import json`` / ``schema migrate`` into an existing
destination report ``destination_exists`` and exit 1 without touching the file; ``report --output`` into an
existing file is a rejected result (``destination_exists``, exit 2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aipe_magnetics import __version__
from aipe_magnetics.common.errors import AIPEMagneticsError, InvalidInputError
from aipe_magnetics.common.format import summary as _summary
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


def _parameters(items: list[str]) -> dict[str, float] | None:
    """Parse repeated ``--param NAME=VALUE`` options with explicit errors."""
    params: dict[str, float] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name.strip():
            raise InvalidInputError(f"--param expects NAME=VALUE, got {item!r}")
        try:
            params[name.strip()] = float(value)
        except ValueError:
            raise InvalidInputError(f"--param {name.strip()}: {value!r} is not a number") from None
    return params or None


def _load_envelopes(path: Path) -> list[EngineeringResult]:
    """Result envelopes from a file: one envelope, a JSON report (``results``) or the demo wrapper (``steps[].result``)."""
    from aipe_magnetics.common.jsonio import load
    data = load(path)
    if isinstance(data, dict) and "steps" in data and isinstance(data["steps"], list):
        items = [step.get("result") for step in data["steps"] if isinstance(step, dict)]
    elif isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        items = data["results"]
    elif isinstance(data, list):
        items = data
    else:
        items = [data]
    try:
        return [EngineeringResult.model_validate(item) for item in items]
    except ValidationError as error:
        raise InvalidInputError(f"{path} does not contain aipe.magnetics.result/0.2 envelopes: {error.errors()[0]['msg']}") from None


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
    p.add_argument("--current-rms", required=True); p.add_argument("--ripple"); p.add_argument("--ripple-duty", type=float)
    p.add_argument("--frequency", "-f", required=True)
    p.add_argument("--material", required=True); p.add_argument("--core"); p.add_argument("--temperature", "-t")
    p.add_argument("--conductor", default="round", choices=["round", "litz", "foil"]); p.add_argument("--flux-fraction", type=float); _add_json_flag(p)
    p = d.add_parser("transformer"); p.add_argument("--power", "-P", required=True); p.add_argument("--frequency", "-f", required=True)
    p.add_argument("--primary-voltage", required=True); p.add_argument("--secondary-voltage", required=True, action="append")
    p.add_argument("--material", required=True); p.add_argument("--core", action="append"); p.add_argument("--temperature", "-t")
    p.add_argument("--waveform", default="square", choices=["square", "sinusoidal", "trapezoidal"]); p.add_argument("--duty", type=float, default=0.5)
    p.add_argument("--conductor", default="litz", choices=["round", "litz", "foil"]); p.add_argument("--flux-fraction", type=float)
    p.add_argument("--loss-target", help="core-loss density ceiling, e.g. '300 kW/m^3'"); _add_json_flag(p)
    p = d.add_parser("waveform-transformer", help="transformer design from the converter's actual waveforms (HDF5 / CSV / JSON) or SPS parameters")
    p.add_argument("--source", help="waveform file (HDF5 / CSV / JSON); omit with --sps-power to synthesise SPS waveforms")
    p.add_argument("--kind", choices=["hdf5", "csv", "json"], help="container kind (default: from the file suffix)")
    p.add_argument("--channel", action="append", default=[], metavar="ROLE=NAME",
                   help="role -> channel: primary_current (required), secondary_current, primary_voltage, magnetising_current (repeatable)")
    p.add_argument("--unit", action="append", default=[], metavar="NAME=UNIT", help="csv: unit of a mapped column (repeatable)")
    p.add_argument("--time-column", help="csv: time column header"); p.add_argument("--time-unit", help="csv: time unit, e.g. us")
    p.add_argument("--window", nargs=2, metavar=("T0", "T1"), help="steady-state window, e.g. '200 us' '240 us'")
    p.add_argument("--frequency", "-f", required=True); p.add_argument("--turns-ratio", type=float, required=True, help="N1/N2 target")
    p.add_argument("--primary-voltage", help="square-wave amplitude, needed when the source has no voltage channel")
    p.add_argument("--series-inductance", help="total series inductance referred to the primary, e.g. '35 uH'")
    p.add_argument("--material", action="append", help="candidate material (repeatable; default: every material with sufficient data)")
    p.add_argument("--core", action="append", help="candidate core id (repeatable)")
    p.add_argument("--offer-policy", default="catalogue", choices=["catalogue", "standard_size_equivalent"])
    p.add_argument("--temperature", "-t", help="design (core) temperature, default 100 degC")
    p.add_argument("--flux-fraction", type=float); p.add_argument("--loss-target", help="core-loss density ceiling, e.g. '300 kW/m^3'")
    p.add_argument("--conductor", action="append", choices=["litz", "foil"], help="conductor kinds explored (repeatable; default litz)")
    p.add_argument("--arrangement", action="append", help="winding arrangements explored (repeatable; default P-S and P-S-P)")
    p.add_argument("--cooling", choices=["natural", "forced"]); p.add_argument("--ambient"); p.add_argument("--heat-transfer-coefficient")
    p.add_argument("--turn-steps", type=int); p.add_argument("--max-candidates", type=int)
    p.add_argument("--sps-power"); p.add_argument("--sps-primary-voltage"); p.add_argument("--sps-secondary-voltage")
    p.add_argument("--sps-phase-shift", help="e.g. '22.5 deg'"); p.add_argument("--sps-series-inductance", help="e.g. '35 uH'")
    p.add_argument("--state-output", type=Path, help="write the AIPE-Core Engineering State of the selected design to this file")
    p.add_argument("--created-at", help="RFC 3339 timestamp recorded in the result and the state (default: now, UTC)")
    p.add_argument("--overwrite", action="store_true"); _add_json_flag(p)

    p = sub.add_parser("measurement-requests", help="magnetic-material measurement requests for an excitation (structured JSON)")
    p.add_argument("--material", action="append", help="material (repeatable; default: every non-synthetic catalogue material)")
    p.add_argument("--frequency", "-f", required=True); p.add_argument("--flux", "-b", help="design peak flux density, e.g. '178 mT'")
    p.add_argument("--temperature", "-t", help="design temperature, default 100 degC")
    p.add_argument("--temperature-range", nargs=2, metavar=("TMIN", "TMAX"), help="operating temperature range, e.g. '25 degC' '120 degC'")
    p.add_argument("--excitation", default="triangular", choices=["sinusoidal", "triangular", "trapezoidal", "piecewise_linear", "sampled"])
    p.add_argument("--dc-bias-max", help="largest DC bias field, e.g. '100 A/m'"); p.add_argument("--harmonics", type=int, default=5)
    p.add_argument("--output", type=Path, help="write the JSON envelope to this file"); p.add_argument("--overwrite", action="store_true"); _add_json_flag(p)

    p = sub.add_parser("validate", help="validate an existing design"); p.add_argument("--material", required=True); p.add_argument("--core", required=True)
    p.add_argument("--turns", type=int, required=True); p.add_argument("--frequency", "-f", required=True); p.add_argument("--temperature", "-t")
    p.add_argument("--voltage-peak"); p.add_argument("--flux"); p.add_argument("--current-peak"); p.add_argument("--air-gap")
    p.add_argument("--waveform", default="square", choices=["square", "sinusoidal", "trapezoidal"]); _add_json_flag(p)

    p = sub.add_parser("report", help="render result JSON files (envelopes, JSON reports or demo output) into a report")
    p.add_argument("results", nargs="+", type=Path)
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
    q = p.add_parser("serve"); q.add_argument("--allow-write", action="store_true", help="requires --output-dir")
    q.add_argument("--output-dir", help="directory that written reports must stay inside")
    q = p.add_parser("self-test")
    return parser


def _winding(conductor: str) -> dict[str, Any]:
    return {"conductor_type": conductor}


def _pairs(items: list[str], option: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise InvalidInputError(f"{option} expects KEY=VALUE, got {item!r}")
        out[key.strip()] = value.strip()
    return out


def _write_json(path: Path, payload: Any, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def _waveform_transformer(args: argparse.Namespace) -> int:
    import aipe_magnetics.api as api

    roles = ("primary_current", "secondary_current", "primary_voltage", "magnetising_current")
    if args.sps_power:
        for name in ("sps_primary_voltage", "sps_secondary_voltage"):
            if not getattr(args, name):
                raise InvalidInputError(f"--{name.replace('_', '-')} is required with --sps-power")
        if bool(args.sps_phase_shift) == bool(args.sps_series_inductance):
            raise InvalidInputError("give exactly one of --sps-phase-shift and --sps-series-inductance")
        sps: dict[str, Any] = {"power": _qty(args.sps_power, "sps-power"), "primary_voltage": _qty(args.sps_primary_voltage, "sps-primary-voltage"),
                               "secondary_voltage": _qty(args.sps_secondary_voltage, "sps-secondary-voltage")}
        if args.sps_phase_shift:
            sps["phase_shift"] = _qty(args.sps_phase_shift, "sps-phase-shift")
        else:
            sps["series_inductance"] = _qty(args.sps_series_inductance, "sps-series-inductance")
        waveforms: dict[str, Any] = {"kind": "sps", "sps": sps}
    else:
        if not args.source:
            raise InvalidInputError("--source (waveform file) or --sps-power is required")
        channels = _pairs(args.channel, "--channel")
        unknown = sorted(set(channels) - set(roles))
        if unknown:
            raise InvalidInputError(f"--channel roles must be one of {list(roles)}; unknown: {unknown}")
        if "primary_current" not in channels:
            raise InvalidInputError("--channel primary_current=<name> is required")
        suffix = Path(args.source).suffix.lower()
        kind = args.kind or {".h5": "hdf5", ".hdf5": "hdf5", ".csv": "csv", ".json": "json"}.get(suffix)
        if kind is None:
            raise InvalidInputError(f"cannot tell the container kind of {args.source!r}; pass --kind")
        waveforms = {"kind": kind, "path": args.source, "channels": channels}
        if args.window:
            waveforms["window"] = {"t0": _qty(args.window[0], "window"), "t1": _qty(args.window[1], "window")}
        if kind == "csv":
            waveforms.update({"time_column": args.time_column, "time_unit": args.time_unit, "units": _pairs(args.unit, "--unit")})
    payload: dict[str, Any] = {"frequency": _qty(args.frequency, "frequency"), "turns_ratio_target": args.turns_ratio, "waveforms": waveforms,
                               "material_offer_policy": args.offer_policy}
    if args.primary_voltage:
        payload["primary_voltage"] = _qty(args.primary_voltage, "primary-voltage")
    if args.series_inductance:
        payload["series_inductance_total"] = _qty(args.series_inductance, "series-inductance")
    if args.material:
        payload["materials"] = args.material
    if args.core:
        payload["cores"] = args.core
    if args.temperature:
        payload["temperature"] = _qty(args.temperature, "temperature")
    if args.flux_fraction is not None:
        payload["flux_limit"] = {"mode": "fraction_of_saturation", "fraction": args.flux_fraction}
    if args.loss_target:
        payload["loss_density_target"] = _qty(args.loss_target, "loss-target")
    windings: dict[str, Any] = {}
    if args.conductor:
        windings["conductors"] = args.conductor
    if args.arrangement:
        windings["arrangements"] = args.arrangement
    if windings:
        payload["windings"] = windings
    thermal: dict[str, Any] = {}
    if args.cooling:
        thermal["cooling"] = args.cooling
    if args.ambient:
        thermal["ambient"] = _qty(args.ambient, "ambient")
    if args.heat_transfer_coefficient:
        thermal["heat_transfer_coefficient"] = _qty(args.heat_transfer_coefficient, "heat-transfer-coefficient")
    if thermal:
        payload["thermal"] = thermal
    if args.turn_steps is not None:
        payload["turn_steps"] = args.turn_steps
    if args.max_candidates is not None:
        payload["max_candidates"] = args.max_candidates
    created_at = args.created_at
    if created_at is None:
        from datetime import datetime, timezone
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = api.design_transformer_from_waveforms(payload, created_at=created_at, export_state=bool(args.state_output))
    if args.state_output and result.status == "ok" and result.data.get("engineering_state") is not None:
        _write_json(args.state_output, result.data["engineering_state"], args.overwrite)
        print(f"wrote {args.state_output}")
    return _emit(result, args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(parser, args)
    except AIPEMagneticsError as error:
        print(f"ERROR {error.code}: {error.message}", file=sys.stderr)
        return 1
    except ValidationError as error:
        first = error.errors()[0]
        print(f"ERROR invalid_input: {'.'.join(str(p) for p in first['loc'])}: {first['msg']}", file=sys.stderr)
        return 1
    except FileExistsError as error:
        # never-overwrite contract of importers/migrations: report, leave the destination untouched, exit 1
        print(f"ERROR destination_exists: {error} (pass --overwrite where supported or choose another destination)", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"ERROR {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"ERROR invalid_input: {error}", file=sys.stderr)
        return 1


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    import aipe_magnetics.api as api

    if args.command == "info":
        from aipe_magnetics.data.catalog import Catalog
        from aipe_magnetics.verification.suite import bundled_evidence_path, load_bundled_registry
        cat = Catalog.default()
        reg = load_bundled_registry()
        print(f"aipe-magnetics {__version__}")
        print(f"catalogue: {len(cat.materials())} materials, {len(cat.cores())} cores, {len(cat.manufacturers())} manufacturers")
        path = bundled_evidence_path()
        if not path.exists():
            print(f"verification evidence: MISSING - {path} does not exist; every result reports L0 unverified until a maintainer "
                  "runs 'aipe-magnetics verify --update-bundled'")
        else:
            print(f"verification evidence: {path} ({len(reg.records)} records, {len(reg.capabilities())} capabilities)")
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
            waveform["duty_rise"] = args.duty_rise
            if args.duty_flat is not None:
                waveform["duty_flat"] = args.duty_flat  # otherwise derived as 0.5 - duty_rise (symmetric trapezoid)
        elif args.duty_flat is not None:
            raise InvalidInputError("--duty-flat needs --duty-rise")
        op: dict[str, Any] = {"frequency": _qty(args.frequency, "frequency"), "flux_density_peak": _qty(args.flux, "flux"), "waveform": waveform}
        if args.temperature:
            op["temperature"] = _qty(args.temperature, "temperature")
        return _emit(api.estimate_core_loss({"material": args.material, "operating_point": op, "model": args.model, "core": args.core,
                                             "parameter_set": args.parameter_set, "policy": args.policy}), args)
    if args.command == "hysteresis":
        params = _parameters(args.param)
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
        if args.ripple_duty is not None:
            payload["ripple_duty"] = args.ripple_duty
        if args.temperature:
            payload["temperature"] = _qty(args.temperature, "temperature")
        if args.flux_fraction is not None:  # an explicit 0 must reach validation and be rejected there, never dropped
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
        if args.flux_fraction is not None:
            payload["flux_limit"] = {"mode": "fraction_of_saturation", "fraction": args.flux_fraction}
        if args.loss_target:
            payload["loss_density_target"] = _qty(args.loss_target, "loss-target")
        return _emit(api.design_transformer(payload), args)
    if args.command == "design" and args.action == "waveform-transformer":
        return _waveform_transformer(args)
    if args.command == "measurement-requests":
        payload = {"frequency": _qty(args.frequency, "frequency"), "excitation": args.excitation, "harmonics": args.harmonics}
        if args.material:
            payload["materials"] = args.material
        if args.flux:
            payload["flux_density_peak"] = _qty(args.flux, "flux")
        if args.temperature:
            payload["temperature"] = _qty(args.temperature, "temperature")
        if args.temperature_range:
            payload["temperature_range"] = [_qty(args.temperature_range[0], "temperature-range"), _qty(args.temperature_range[1], "temperature-range")]
        if args.dc_bias_max:
            payload["dc_bias_field_max"] = _qty(args.dc_bias_max, "dc-bias-max")
        res = api.measurement_requests(payload)
        if args.output:
            _write_json(args.output, res.to_json_dict(), args.overwrite)
            print(f"wrote {args.output}")
        return _emit(res, args)
    if args.command == "validate":
        payload = {"material": args.material, "core": args.core, "turns": args.turns, "frequency": _qty(args.frequency, "frequency"),
                   "voltage_waveform": args.waveform}
        for key, value in (("temperature", args.temperature), ("voltage_peak", args.voltage_peak), ("flux_density_peak", args.flux),
                           ("current_peak", args.current_peak), ("air_gap", args.air_gap)):
            if value:
                payload[key] = _qty(value, key)
        return _emit(api.validate_design(payload), args)
    if args.command == "report":
        results = [r for p in args.results for r in _load_envelopes(p)]
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
            if not args.output_dir:
                parser.error("mcp serve --allow-write requires --output-dir (the directory written reports must stay inside)")
            argv2 += ["--allow-write", "--output-dir", args.output_dir]
        elif args.output_dir:
            argv2 += ["--output-dir", args.output_dir]
        return srv.main(argv2)
    parser.error("unknown command")  # pragma: no cover
    return 1


if __name__ == "__main__":
    sys.exit(main())
