"""JSON-safe conversion of model diagnostics (numpy arrays, pydantic models, dataclasses) for result envelopes."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

MAX_ARRAY_POINTS = 2000


def jsonable(value: Any, *, max_points: int = MAX_ARRAY_POINTS) -> Any:
    """Convert ``value`` recursively into JSON-serialisable Python objects.

    Long arrays are decimated to at most ``max_points`` samples (the decimation factor is reported in
    a sibling key ``<name>_decimated_by`` when converting dicts); non-finite floats become ``None``.
    """
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, np.ndarray) and item.ndim == 1 and item.size > max_points:
                step = int(np.ceil(item.size / max_points))
                out[str(key)] = jsonable(item[::step], max_points=max_points)
                out[f"{key}_decimated_by"] = step
            else:
                out[str(key)] = jsonable(item, max_points=max_points)
        return out
    if isinstance(value, (list, tuple)):
        return [jsonable(v, max_points=max_points) for v in value]
    if isinstance(value, np.ndarray):
        if value.size > max_points and value.ndim == 1:
            step = int(np.ceil(value.size / max_points))
            value = value[::step]
        elif value.size > max_points:
            return {"array": True, "shape": list(value.shape), "dtype": str(value.dtype), "note": "omitted (too large)"}
        return [jsonable(v, max_points=max_points) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json", exclude_none=True), max_points=max_points)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable({f.name: getattr(value, f.name) for f in dataclasses.fields(value)}, max_points=max_points)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)
