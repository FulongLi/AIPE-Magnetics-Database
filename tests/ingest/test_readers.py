"""Format readers: magic-byte detection and faithful reading of CSV/TSV/text, XLSX, MAT v5/v7.3, HDF5, Parquet, JSON, ZIP."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from aipe_magnetics_database.ingest import readers


def test_text_formats_and_decimal_comma():
    csv_sheet = readers.read_text(b"# measured 2024\nf [Hz],B [mT],P [kW/m3]\n100000,100,250.5\n200000,50,\n", "a.csv")
    assert csv_sheet.header == ["f [Hz]", "B [mT]", "P [kW/m3]"] and csv_sheet.rows == [[100000, 100, 250.5], [200000, 50, None]]
    assert csv_sheet.notes == ["# measured 2024"] and csv_sheet.numeric_rows() == [[100000.0, 100.0, 250.5]]
    tsv = readers.read_text(b"freq\tmu'\tmu''\n1e5\t2000,5\t10\n", "b.txt")
    assert tsv.rows == [[100000.0, 2000.5, 10]]
    ws = readers.read_text(b"1.0  2.0   3.0\n4 5 6\n", "c.txt")
    assert ws.header == ["col0", "col1", "col2"] and ws.rows[1] == [4, 5, 6]
    semi = readers.read_text("T;Bs\n25;0,49\n100;0,39\n".encode(), "d.csv")
    assert semi.rows == [[25, 0.49], [100, 0.39]]


def test_detect_by_magic_bytes_not_extension(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "misnamed.csv"
    with h5py.File(p, "w") as f:
        f["x"] = np.arange(3)
    assert readers.detect(p) == "hdf5"
    assert readers.detect(b'{"a": 1}', "x.json") == "json"
    assert readers.detect(b'{"a": 1}\n{"a": 2}\n', "x.ndjson") == "jsonl"
    assert readers.detect(b"\x00\x01\x02" * 100, "x.bin") == "binary"


def test_xlsx(tmp_path: Path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "N87"
    ws.append(["Core loss of N87 at 100 degC"])
    ws.append([])
    ws.append(["f (kHz)", "B (mT)", "Pv (kW/m3)"])
    ws.append([100, 200, 375.25])
    ws.append([25, 200, 90.0])
    p = tmp_path / "loss.xlsx"
    wb.save(p)
    sheets = readers.read_tables(p)
    assert len(sheets) == 1 and sheets[0].name == "N87" and sheets[0].header == ["f (kHz)", "B (mT)", "Pv (kW/m3)"]
    assert sheets[0].rows == [[100, 200, 375.25], [25, 200, 90.0]] and sheets[0].notes == ["Core loss of N87 at 100 degC"]


def test_mat_v5_and_v73(tmp_path: Path):
    sio = pytest.importorskip("scipy.io")
    h5py = pytest.importorskip("h5py")
    p5 = tmp_path / "w5.mat"
    sio.savemat(p5, {"B": np.linspace(-0.1, 0.1, 5), "meta": {"f": 1e5, "T": 25.0}})
    arrays = readers.read_arrays(p5)
    assert np.allclose(arrays["B"], np.linspace(-0.1, 0.1, 5)) and float(arrays["meta/f"]) == 1e5
    p73 = tmp_path / "w73.mat"
    with h5py.File(p73, "w", userblock_size=512) as f:
        f["H"] = np.arange(6.0).reshape(2, 3)
    with open(p73, "r+b") as f:
        f.write(b"MATLAB 7.3 MAT-file, Platform: GLNXA64".ljust(116, b" "))
    assert readers.detect(p73) == "mat73"
    assert readers.read_arrays(p73)["H"].shape == (3, 2)  # MATLAB column-major order restored


def test_hdf5_tables_and_parquet(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    p = tmp_path / "x.h5"
    with h5py.File(p, "w") as f:
        f["grp/table"] = np.array([[1.0, 2.0], [3.0, 4.0]])
        f["grp/vec"] = np.arange(4)
    sheets = readers.read_tables(p)
    assert [s.name for s in sheets] == ["grp/table"] and sheets[0].rows == [[1.0, 2.0], [3.0, 4.0]]
    pq_path = tmp_path / "x.parquet"
    pq.write_table(pa.table({"f": [1e5, 2e5], "Pv": [10.5, None]}), pq_path)
    sheet = readers.read_tables(pq_path)[0]
    assert sheet.header == ["f", "Pv"] and sheet.rows == [[1e5, 10.5], [2e5, None]]


def test_zip_read_in_memory_and_skips_metadata(tmp_path: Path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("data/a.txt", "x y\n1 2\n")
        z.writestr("__MACOSX/data/._a.txt", "junk")
        z.writestr("data/b.json", json.dumps([{"f": 1, "P": 2}]))
    p = tmp_path / "bundle.zip"
    p.write_bytes(buf.getvalue())
    sheets = readers.read_tables(p)
    assert [s.source.split("!")[1] for s in sheets] == ["data/a.txt", "data/b.json"]
    assert sheets[0].rows == [[1, 2]] and sheets[1].header == ["P", "f"]
    assert not list(tmp_path.glob("data"))  # nothing extracted


def test_reading_is_deterministic_and_non_destructive(tmp_path: Path):
    p = tmp_path / "a.csv"
    p.write_bytes(b"f,P\n1,2\n")
    before = p.read_bytes()
    assert readers.read_tables(p)[0].rows == readers.read_tables(p)[0].rows
    assert p.read_bytes() == before
