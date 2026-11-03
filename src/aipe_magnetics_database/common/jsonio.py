"""Strict, deterministic JSON I/O shared by every module.

* Duplicate object keys and non-finite numbers are rejected on load (as in v0.1).
* Dumps are sorted, UTF-8, LF terminated, with ``allow_nan=False`` so output is reproducible.
* SHA-256 helpers for original-byte checksums.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def loads(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)


def load(path: str | Path) -> Any:
    return loads(Path(path).read_text(encoding="utf-8"))


def dumps(value: Any, *, indent: int | None = 2, sort_keys: bool = True) -> str:
    return json.dumps(value, indent=indent, sort_keys=sort_keys, ensure_ascii=False, allow_nan=False) + "\n"


def dump(value: Any, path: str | Path, *, overwrite: bool = False, indent: int | None = 2, sort_keys: bool = True) -> Path:
    """Write JSON; refuses to overwrite an existing file unless ``overwrite=True``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        stream.write(dumps(value, indent=indent, sort_keys=sort_keys))
    return path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """SHA-256 of the canonical (sorted, compact) JSON serialisation of ``value``."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return sha256_bytes(text.encode("utf-8"))
