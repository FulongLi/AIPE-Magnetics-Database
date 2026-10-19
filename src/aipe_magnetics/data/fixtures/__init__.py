"""Packaged, clearly synthetic fixtures used by tests and verification checks.

``v01/`` holds one **synthetic** schema-0.1.0 record per v0.1 record kind (eleven kinds). They
exercise the v0.1 -> v0.2 migration and the legacy validator; none of them describes a real
product, material or measurement (``synthetic: true``, assumption evidence, placeholder artifact
paths). Numerical values inside them are round illustrative numbers, not data.

The fixture set is a self-consistent v0.1 catalogue: every ``*_ref`` resolves inside the set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aipe_magnetics.common.jsonio import load

FIXTURE_DIR = Path(__file__).resolve().parent
V01_FIXTURE_DIR = FIXTURE_DIR / "v01"

V01_KINDS: tuple[str, ...] = (
    "magnetic_material", "core_material", "core_family", "core_geometry", "winding", "inductor",
    "transformer", "magnetic_component", "loss_model", "measurement", "operating_condition",
)


def v01_fixture_paths() -> list[Path]:
    """Sorted paths of the packaged synthetic v0.1 records (one per v0.1 kind)."""
    return sorted(V01_FIXTURE_DIR.glob("*.json"))


def v01_fixture_records() -> dict[str, dict[str, Any]]:
    """``kind -> record dict`` for every packaged synthetic v0.1 record (strict JSON load)."""
    records: dict[str, dict[str, Any]] = {}
    for path in v01_fixture_paths():
        record = load(path)
        records[record["kind"]] = record
    return records


__all__ = ["FIXTURE_DIR", "V01_FIXTURE_DIR", "V01_KINDS", "v01_fixture_paths", "v01_fixture_records"]
