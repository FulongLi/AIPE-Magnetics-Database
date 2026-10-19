"""Command line: offline AIPE-Core Engineering State contract validation.

    python -m aipe_magnetics.integrations.aipe_core validate <state.json> [<state.json> ...] [--json]
    python -m aipe_magnetics.integrations.aipe_core info

Exit status 1 when any state fails. Output mirrors AIPE-Core ``scripts/validate.py``: a passing
state establishes interchange structure and reference consistency only, never physical validity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from aipe_magnetics.common.jsonio import dumps
from aipe_magnetics.integrations.aipe_core.state import (
    CORE_COMMIT,
    CORE_REPOSITORY,
    CORE_SCHEMA_VERSION,
    EXAMPLE_PATH,
    SCHEMA_DIR,
    core_schema_ids,
    load_state,
    validate_engineering_state,
)

PASS_NOTE = "(structure and references; no physical validation)"


def validate_files(paths: Sequence[Path]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for path in paths:
        try:
            errors = validate_engineering_state(load_state(path))
        except (OSError, ValueError) as error:  # unreadable file, duplicate keys, NaN/Infinity, bad JSON
            errors = [str(error)]
        results.append({"path": str(path), "passed": not errors, "errors": errors})
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aipe_magnetics.integrations.aipe_core",
                                     description="Offline AIPE-Core Engineering State 0.1.0 contract checks.")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate one or more Engineering State JSON files")
    validate.add_argument("states", nargs="+", type=Path)
    validate.add_argument("--json", action="store_true", help="machine-readable output")
    commands.add_parser("info", help="show the pinned Core commit, schema directory and schema ids")
    args = parser.parse_args(argv)

    if args.command == "info":
        print(dumps({"core_repository": CORE_REPOSITORY, "core_commit": CORE_COMMIT,
                     "core_schema_version": CORE_SCHEMA_VERSION, "schema_dir": str(SCHEMA_DIR),
                     "example": str(EXAMPLE_PATH), "schema_ids": core_schema_ids()}), end="")
        return 0

    results = validate_files(args.states)
    if args.json:
        print(dumps({"results": results}), end="")
    else:
        for result in results:
            if result["passed"]:
                print(f"PASS {result['path']} {PASS_NOTE}")
            else:
                print(f"FAIL {result['path']}")
                for error in result["errors"]:  # type: ignore[union-attr]
                    print(f"  {error}")
    return int(any(not result["passed"] for result in results))


if __name__ == "__main__":
    sys.exit(main())
