"""Optional-dependency extras named in installation hints are declared in pyproject.toml."""

from __future__ import annotations

import re
import tomllib_shim as shim  # noqa: F401 (tomllib on Python 3.10)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_every_install_hint_names_a_declared_extra():
    import tomllib_shim
    extras = set(tomllib_shim.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"])
    assert {"hdf5", "mcp", "torch", "viz", "mesh", "cad", "dev"} <= extras
    hinted: set[str] = set()
    for path in (ROOT / "src").rglob("*.py"):
        for match in re.finditer(r"aipe-magnetics\[([a-z0-9_,]+)\]", path.read_text(encoding="utf-8")):
            hinted.update(match.group(1).split(","))
    assert hinted and hinted <= extras, sorted(hinted - extras)
