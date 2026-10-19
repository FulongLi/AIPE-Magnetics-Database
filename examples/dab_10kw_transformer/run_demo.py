#!/usr/bin/env python3
"""10 kW / 100 kHz dual-active-bridge transformer demonstration (same as ``aipe-magnetics demo dab-10kw``).

Usage::

    python examples/dab_10kw_transformer/run_demo.py [--output reports/demo] [--temperature "100 degC"] [--json]

The script only calls the public API (material lookup, candidate screening, turns, flux density,
saturation screening, iGSE core loss, Dowell winding loss, candidate comparison, report, AIPE-Core
export). See docs/demo-dab-10kw.md for the specification and assumptions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aipe_magnetics.cli.main import _qty
from aipe_magnetics.engineering.demo_dab import run_dab_demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=None, help="directory for the Markdown/JSON report and the Core state")
    parser.add_argument("--temperature", "-t", default="100 degC", help="core temperature, e.g. '100 degC' or '90 degC'")
    parser.add_argument("--json", action="store_true", help="print all result envelopes as JSON instead of the summary")
    args = parser.parse_args(argv)
    return run_dab_demo(output=args.output, temperature=_qty(args.temperature, "temperature"), as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
