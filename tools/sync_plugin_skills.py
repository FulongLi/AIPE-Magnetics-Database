#!/usr/bin/env python3
"""Copy the canonical skills from ``skills/`` into both plugin packages (or check that they are in sync).

Both the Claude Code plugin and the Codex plugin share one engineering engine and one set of skill
documents. The canonical sources live in ``skills/<name>/SKILL.md``; this script mirrors them into
``plugins/claude/aipe-magnetics/skills`` and ``plugins/codex/aipe-magnetics/skills``.

Usage::

    python tools/sync_plugin_skills.py          # copy
    python tools/sync_plugin_skills.py --check  # exit 1 if any mirror differs
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "skills"
TARGETS = [ROOT / "plugins" / "claude" / "aipe-magnetics" / "skills",
           ROOT / "plugins" / "codex" / "aipe-magnetics" / "skills"]


def skill_files() -> list[Path]:
    return sorted(p for p in SOURCE.rglob("*") if p.is_file())


def sync(check: bool) -> int:
    problems: list[str] = []
    files = skill_files()
    if not files:
        print("no skills found under skills/", file=sys.stderr)
        return 1
    for target in TARGETS:
        for source in files:
            relative = source.relative_to(SOURCE)
            destination = target / relative
            if check:
                if not destination.exists():
                    problems.append(f"missing: {destination.relative_to(ROOT)}")
                elif destination.read_bytes() != source.read_bytes():
                    problems.append(f"differs: {destination.relative_to(ROOT)}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        if target.exists():
            for extra in target.rglob("*"):
                if extra.is_file() and not (SOURCE / extra.relative_to(target)).exists():
                    if check:
                        problems.append(f"stale: {extra.relative_to(ROOT)}")
                    else:
                        extra.unlink()
    if check:
        for problem in problems:
            print(problem)
        print("skills in sync" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    print(f"synced {len(files)} file(s) into {len(TARGETS)} plugin(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    return sync(args.check)


if __name__ == "__main__":
    raise SystemExit(main())
