"""Optional-dependency extras named in installation hints are declared in pyproject.toml."""

from __future__ import annotations

import re
from pathlib import Path

import tomllib_shim

ROOT = Path(__file__).resolve().parents[2]


def test_every_install_hint_names_a_declared_extra():
    extras = set(tomllib_shim.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"])
    assert {"hdf5", "mcp", "dev"} <= extras
    hinted: set[str] = set()
    for path in (ROOT / "src").rglob("*.py"):
        for match in re.finditer(r"aipe-magnetics-database\[([a-z0-9_,]+)\]", path.read_text(encoding="utf-8")):
            hinted.update(match.group(1).split(","))
    assert hinted <= extras, sorted(hinted - extras)
