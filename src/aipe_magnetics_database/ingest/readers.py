"""Read heterogeneous original files without modifying them (importer building block).

:func:`detect` identifies a file by its magic bytes (not its extension); :func:`read_tables` returns every table
found in it as :class:`Sheet` objects (``name``, ``header``, ``rows``; cells are kept as read: numbers as float/int,
text as str, empty cells as ``None``), and :func:`read_arrays` returns the numeric arrays of MATLAB / HDF5 files.
ZIP archives are read member by member in memory - nothing is extracted to disk and nothing inside is executed.

Supported: CSV / TSV / whitespace-delimited text (delimiter sniffed; ``#``/``%`` comment lines kept as notes), Excel
XLSX (``openpyxl``) and XLS (``xlrd``), MATLAB MAT v4/v5 (``scipy.io``) and v7.3 (HDF5, ``h5py``), HDF5 (``h5py``),
Parquet (``pyarrow``), JSON / JSONL / NDJSON, ZIP (recursively). The optional readers are imported only when a file
of that type is read (``pip install 'aipe-magnetics-database[importers]'``); the runtime package stays data-only.
No value is converted, interpolated or rounded here: unit handling belongs to the importer that maps the columns.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

MAGIC = [(b"\x89HDF\r\n\x1a\n", "hdf5"), (b"PAR1", "parquet"), (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "xls"), (b"MATLAB 5.0 MAT-file", "mat5"),
         (b"PK\x03\x04", "zip")]
_NUM = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$")


@dataclass
class Sheet:
    name: str
    header: list[str]
    rows: list[list[Any]]
    notes: list[str] = field(default_factory=list)
    source: str = ""  # file (and member / sheet) the table came from

    def column(self, name: str) -> list[Any]:
        return [r[self.header.index(name)] if self.header.index(name) < len(r) else None for r in self.rows]

    def numeric_rows(self) -> list[list[float]]:
        """Rows in which every cell is a finite number (others are left to the importer to count and exclude)."""
        out = []
        for r in self.rows:
            if r and all(isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf") for v in r):
                out.append([float(v) for v in r])
        return out


def detect(data: bytes | str | Path, name: str = "") -> str:
    """File type from magic bytes: hdf5, mat5, mat73, xls, xlsx, zip, parquet, json, jsonl, text, binary."""
    if not isinstance(data, bytes):
        name = name or str(data)
        with open(data, "rb") as handle:
            data = handle.read(4096)
    head = data[:4096]
    if head.startswith(b"MATLAB 7.3 MAT-file") or (len(head) > 512 and head[512:520] == b"\x89HDF\r\n\x1a\n"):
        return "mat73"
    for magic, kind in MAGIC:
        if head.startswith(magic):
            if kind == "zip" and (b"[Content_Types].xml" in head or b"xl/" in head):
                return "xlsx"
            return kind
    if name.lower().endswith(".mat") and len(head) >= 128 and head[124:126] in (b"\x00\x01", b"\x01\x00"):
        return "mat5"
    if name.lower().endswith(".mat"):
        return "mat4"
    try:
        text = head.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = head.decode("latin-1")
        except UnicodeDecodeError:  # pragma: no cover - latin-1 decodes everything
            return "binary"
    if sum(ch < " " and ch not in "\t\r\n\f" for ch in text) > len(text) // 20:
        return "binary"
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        first = stripped.splitlines()[0] if stripped else ""
        if name.lower().endswith((".jsonl", ".ndjson")) or (first.startswith("{") and first.rstrip().endswith("}") and len(stripped.splitlines()) > 1):
            return "jsonl"
        return "json"
    return "text"


def _cell(token: str, decimal_comma: bool) -> Any:
    token = token.strip().strip('"')
    if token == "":
        return None
    t = token.replace(",", ".") if decimal_comma else token
    if _NUM.match(t):
        try:
            return int(t) if re.fullmatch(r"[+-]?\d+", t) else float(t)
        except ValueError:  # pragma: no cover
            return token
    if t.lower() in ("nan", "inf", "-inf", "+inf"):
        return float(t)
    return token


def read_text(data: bytes, name: str = "", *, delimiter: str | None = None, header_row: int | None = None) -> Sheet:
    """Delimited text: delimiter sniffed (tab, comma, semicolon, whitespace); a decimal comma is recognised when ``;`` or
    tab delimits and numbers contain ``,``. ``header_row`` (0-based among non-comment lines) defaults to the last
    non-numeric line before the first numeric one."""
    text = data.decode("utf-8-sig") if not data.startswith(b"\xff\xfe") else data.decode("utf-16")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    notes = [ln for ln in lines if ln.lstrip().startswith(("#", "%", "//"))]
    body = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith(("#", "%", "//"))]
    if delimiter is None:
        sample = "\n".join(body[:50])
        if "\t" in sample:
            delimiter = "\t"
        elif sample.count(";") >= max(1, len(body[:50])):
            delimiter = ";"
        elif sample.count(",") >= max(1, len(body[:50])) and not re.search(r"\d,\d+\s+\d", sample):
            delimiter = ","
        else:
            delimiter = " "
    decimal_comma = delimiter in (";", "\t", " ") and bool(re.search(r"\d,\d", "\n".join(body[:50])))
    if delimiter == " ":
        split = [re.split(r"\s+", ln.strip()) for ln in body]
    else:
        split = list(csv.reader(body, delimiter=delimiter))
    cells = [[_cell(t, decimal_comma) for t in row] for row in split]
    if header_row is None:
        header_row = -1
        for i, row in enumerate(cells):
            if row and all(isinstance(v, (int, float)) or v is None for v in row):
                break
            header_row = i
    header = [str(v) if v is not None else "" for v in cells[header_row]] if header_row >= 0 else []
    pre = [" ".join(str(v) for v in r if v is not None) for r in cells[:max(header_row, 0)]]
    rows = cells[header_row + 1:]
    width = max((len(r) for r in rows), default=len(header))
    if not header:
        header = [f"col{i}" for i in range(width)]
    return Sheet(Path(name).stem or "table", header, rows, notes + pre, name)


def read_excel(path_or_bytes: bytes | str | Path, name: str = "") -> list[Sheet]:
    kind = detect(path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes), name)
    data = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
    sheets = []
    if kind == "xlsx":
        try:
            import openpyxl
        except ImportError as error:
            raise ImportError("reading .xlsx needs openpyxl: pip install 'aipe-magnetics-database[importers]'") from error
        book = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        for ws in book.worksheets:
            grid = [[(None if v == "" else v) for v in row] for row in ws.iter_rows(values_only=True)]
            sheets.append(_grid_sheet(grid, ws.title, f"{name}#{ws.title}"))
        book.close()
    elif kind == "xls":
        try:
            import xlrd
        except ImportError as error:
            raise ImportError("reading .xls needs xlrd: pip install 'aipe-magnetics-database[importers]'") from error
        book = xlrd.open_workbook(file_contents=data)
        for ws in book.sheets():
            grid = [[(None if c.ctype in (0, 6) else c.value) for c in ws.row(i)] for i in range(ws.nrows)]
            sheets.append(_grid_sheet(grid, ws.name, f"{name}#{ws.name}"))
    else:
        raise ValueError(f"{name or path_or_bytes}: not an Excel workbook ({kind})")
    return sheets


def _grid_sheet(grid: list[list[Any]], title: str, source: str) -> Sheet:
    while grid and all(v is None for v in grid[-1]):
        grid.pop()
    header_row = -1
    for i, row in enumerate(grid):
        filled = [v for v in row if v is not None]
        if filled and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in filled):
            break
        if filled:
            header_row = i
    header = [("" if v is None else str(v)) for v in grid[header_row]] if header_row >= 0 else []
    notes = [" ".join(str(v) for v in r if v is not None) for r in grid[:max(header_row, 0)] if any(v is not None for v in r)]
    return Sheet(title, header, [list(r) for r in grid[header_row + 1:]], notes, source)


def read_arrays(path_or_bytes: bytes | str | Path, name: str = "") -> dict[str, Any]:
    """``{variable or dataset path: numpy array}`` of a MATLAB (v4/v5/v7.3) or HDF5 file; structs are flattened with
    ``/`` separators. Object / pickled content is never deserialised (HDF5 object references are skipped)."""
    import numpy as np
    data = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
    kind = detect(data, name)
    out: dict[str, Any] = {}
    if kind in ("hdf5", "mat73"):
        try:
            import h5py
        except ImportError as error:
            raise ImportError("reading HDF5 / MAT v7.3 needs h5py: pip install 'aipe-magnetics-database[hdf5]'") from error
        with h5py.File(io.BytesIO(data), "r") as handle:
            def visit(path: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "biufcSU" and not path.startswith("#refs#"):
                    value = obj[()]
                    out[path] = value.T if kind == "mat73" and getattr(value, "ndim", 0) >= 2 else value
            handle.visititems(visit)
        return out
    if kind in ("mat5", "mat4"):
        try:
            from scipy.io import loadmat
        except ImportError as error:
            raise ImportError("reading MAT v4/v5 needs scipy: pip install 'aipe-magnetics-database[importers]'") from error
        mat = loadmat(io.BytesIO(data), squeeze_me=True, struct_as_record=False, mat_dtype=False)

        def walk(prefix: str, value: Any) -> None:
            if hasattr(value, "_fieldnames"):
                for f in value._fieldnames:
                    walk(f"{prefix}/{f}", getattr(value, f))
            elif isinstance(value, np.ndarray) and value.dtype.kind in "biufcSU":
                out[prefix] = value
            elif isinstance(value, (int, float, complex, str)):
                out[prefix] = np.asarray(value)
        for key, value in mat.items():
            if not key.startswith("__"):
                walk(key, value)
        return out
    raise ValueError(f"{name}: not a MATLAB or HDF5 file ({kind})")


def read_parquet(path_or_bytes: bytes | str | Path, name: str = "") -> Sheet:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("reading Parquet needs pyarrow: pip install 'aipe-magnetics-database[importers]'") from error
    data = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
    table = pq.read_table(io.BytesIO(data))
    cols = [table.column(i).to_pylist() for i in range(table.num_columns)]
    return Sheet(Path(name).stem or "table", list(table.column_names), [list(r) for r in zip(*cols)], [], name)


def read_tables(path: str | Path | bytes, name: str = "") -> list[Sheet]:
    """Every table of one file (or, for a ZIP, of every member)."""
    data = path if isinstance(path, bytes) else Path(path).read_bytes()
    name = name or (str(path) if not isinstance(path, bytes) else "")
    kind = detect(data, name)
    if kind in ("xlsx", "xls"):
        return read_excel(data, name)
    if kind == "parquet":
        return [read_parquet(data, name)]
    if kind == "zip":
        return [sheet for member, blob in iter_zip(data) for sheet in read_tables(blob, f"{name}!{member}") if sheet.rows]
    if kind == "json":
        obj = json.loads(data.decode("utf-8-sig"))
        return [_json_sheet(obj, name)]
    if kind == "jsonl":
        objs = [json.loads(ln) for ln in data.decode("utf-8-sig").splitlines() if ln.strip()]
        return [_json_sheet(objs, name)]
    if kind == "text":
        return [read_text(data, name)]
    if kind in ("hdf5", "mat73", "mat5", "mat4"):
        sheets = []
        for key, arr in read_arrays(data, name).items():
            if getattr(arr, "ndim", 0) == 2 and arr.dtype.kind in "biuf":
                sheets.append(Sheet(key, [f"col{i}" for i in range(arr.shape[1])], arr.tolist(), [], f"{name}#{key}"))
        return sheets
    return []


def _json_sheet(obj: Any, name: str) -> Sheet:
    if isinstance(obj, list) and obj and all(isinstance(o, dict) for o in obj):
        header = sorted({k for o in obj for k in o})
        return Sheet(Path(name).stem or "table", header, [[o.get(k) for k in header] for o in obj], [], name)
    if isinstance(obj, list) and obj and all(isinstance(o, list) for o in obj):
        return Sheet(Path(name).stem or "table", [f"col{i}" for i in range(max(len(o) for o in obj))], obj, [], name)
    return Sheet(Path(name).stem or "table", [], [], [json.dumps(obj)[:200]], name)


def iter_zip(data: bytes, *, max_member_bytes: int = 1 << 31) -> Iterator[tuple[str, bytes]]:
    """``(member, bytes)`` of the regular files of a ZIP, in name order, macOS metadata skipped; read in memory."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in sorted(archive.infolist(), key=lambda i: i.filename):
            n = info.filename
            if info.is_dir() or n.startswith("__MACOSX/") or n.endswith(".DS_Store") or "/._" in f"/{n}":
                continue
            if info.file_size > max_member_bytes:
                raise ValueError(f"zip member {n} is larger than {max_member_bytes} bytes")
            yield n, archive.read(info)


__all__ = ["Sheet", "detect", "iter_zip", "read_arrays", "read_excel", "read_parquet", "read_tables", "read_text"]
