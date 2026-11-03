"""Imports are reproducible and idempotent: re-running an importer on the pinned, committed raw files rewrites
byte-identical output; changed raw bytes are refused."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from aipe_magnetics_database.common.errors import DataIntegrityError
from aipe_magnetics_database.ingest.sources import paderborn, zenodo

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    (zenodo.run_hefmag, "hefmag"),
    (zenodo.run_triangular, "zenodo-smc-ferrite-triangular"),
    (paderborn.run, "paderborn-lea"),
]


@pytest.mark.parametrize("fn, directory", CASES)
def test_rerun_from_committed_raw_is_byte_identical(tmp_path: Path, fn, directory):
    summary = fn(tmp_path, local_dir=ROOT / "data" / "raw" / directory / "files")
    committed = ROOT / "data" / "normalized" / directory
    produced = sorted(p.relative_to(tmp_path / directory) for p in (tmp_path / directory).rglob("*") if p.is_file())
    expected = sorted(p.relative_to(committed) for p in committed.rglob("*") if p.is_file())
    assert produced == expected
    for rel in produced:
        assert (tmp_path / directory / rel).read_bytes() == (committed / rel).read_bytes(), rel
    again = fn(tmp_path, local_dir=ROOT / "data" / "raw" / directory / "files")
    assert again.files_written == 0 and again.files_unchanged == summary.files_written + summary.files_unchanged


def test_modified_raw_file_is_refused(tmp_path: Path):
    raw = tmp_path / "raw"
    shutil.copytree(ROOT / "data" / "raw" / "zenodo-smc-ferrite-triangular" / "files", raw)
    (raw / "SMC_sin.txt").write_text("100\t1.0\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="sha256"):
        zenodo.run_triangular(tmp_path / "out", local_dir=raw)


def test_missing_input_without_download_permission_raises(tmp_path: Path):
    from aipe_magnetics_database.data.sources.cache import LocalCache
    with pytest.raises(Exception, match="download"):
        zenodo.run_triangular(tmp_path / "out", local_dir=tmp_path / "empty", cache=LocalCache(tmp_path / "cache"))
