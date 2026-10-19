"""Shared fixtures for the data-layer tests.

Optional dependencies never cause silent skips: ``require_h5py`` fails with an explicit message.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from aipe_magnetics.common.jsonio import load

REPO_ROOT = Path(__file__).resolve().parents[2]
CREATED_AT = "2026-10-19T00:00:00Z"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def v01_example_path(repo_root: Path) -> Path:
    return repo_root / "examples" / "synthetic-transformer.json"


@pytest.fixture
def v01_example(v01_example_path: Path) -> dict:
    return copy.deepcopy(load(v01_example_path))


@pytest.fixture
def evidence_measurement() -> dict:
    return {
        "id": "evidence.test-measurement", "kind": "measurement",
        "reference": "test fixture: user-supplied loss table (not a real measurement)",
        "description": "synthetic table used only to exercise the importer", "created_at": CREATED_AT,
        "review_status": "unreviewed", "license": "NOASSERTION", "locator": "losses.csv",
    }


@pytest.fixture
def evidence_assumption() -> dict:
    return {
        "id": "evidence.test-assumption", "kind": "assumption", "reference": "test fixture",
        "description": "synthetic structure fixture", "created_at": CREATED_AT,
        "review_status": "unreviewed", "license": "NOASSERTION",
    }


@pytest.fixture
def evidence_source() -> dict:
    return {
        "id": "evidence.test-source", "kind": "source", "reference": "test fixture: user-provided waveform export",
        "description": "synthetic arrays used only to exercise the importer", "created_at": CREATED_AT,
        "review_status": "unreviewed", "license": "NOASSERTION", "locator": "src.npz",
    }


@pytest.fixture
def require_h5py() -> None:
    from aipe_magnetics.data.hdf5 import h5py_available

    if not h5py_available():
        pytest.fail("h5py is required for the data-layer tests: install the 'hdf5' extra (pip install -e .[dev])")
