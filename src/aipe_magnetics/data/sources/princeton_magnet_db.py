"""Reader for the Princeton MagNet packaged scalar core-loss database (``<material>_database.h5``).

Source: https://github.com/PrincetonUniversity/magnet (MIT licence, Copyright (c) 2021 Minjie Chen),
files ``src/magnet/data/<material>_database.h5`` written with ``pandas.HDFStore`` in *fixed* format
(PyTables layout). This module reads that layout with ``h5py`` only (no pandas/pytables), never
unpickles the embedded metadata (it is parsed as text), and never modifies the file.

Layout (verified on commit 8a6892e91ac622e42f8b9182a6dfe9094ceea97a, N87 file sha256
263c0087f6f43ccfd1b338d105abb6b9432be2cb85c59528539317f99b25bd63):

```
/dataframe                 attrs: pandas_type='frame', metadata=<pickle protocol-0 text>
/dataframe/axis0           |S12 [7]      column names
/dataframe/block0_items    |S12 [7]      same order as block0_values columns
/dataframe/block0_values   float64 [n, 7]
```

Columns and units (from the MagNet processing description embedded in ``metadata`` and from
``app/ui_predict.py`` unit conversions in the upstream repository):

| column | unit | meaning |
| --- | --- | --- |
| ``Frequency`` | Hz | fundamental frequency (Welch estimate) |
| ``Flux_Density`` | T | flux density **amplitude** (half peak-to-peak) |
| ``DC_Bias`` | A/m | DC magnetic field |
| ``Duty_P``, ``Duty_N`` | 1 | duty cycles of the rising / falling voltage intervals; ``-1`` = sinusoidal |
| ``Temperature`` | degC | core temperature (oil bath) |
| ``Power_Loss`` | W/m^3 | volumetric loss density |

Nothing here is redistributed: the caller points at a local checkout of the upstream repository
(or the installed ``magnet`` package data directory).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aipe_magnetics.common.errors import DataIntegrityError, DataNotFoundError, UnsupportedOperationError

UPSTREAM_REPOSITORY = "https://github.com/PrincetonUniversity/magnet"
UPSTREAM_COMMIT = "8a6892e91ac622e42f8b9182a6dfe9094ceea97a"
UPSTREAM_LICENSE = "MIT"
EXPECTED_COLUMNS = ("Frequency", "Flux_Density", "DC_Bias", "Duty_P", "Duty_N", "Temperature", "Power_Loss")
COLUMN_UNITS = {"Frequency": "Hz", "Flux_Density": "T", "DC_Bias": "A/m", "Duty_P": "1", "Duty_N": "1",
                "Temperature": "degC", "Power_Loss": "W/m^3"}
KNOWN_MATERIALS = ("3C90", "3C94", "3E6", "3F4", "77", "78", "N27", "N30", "N49", "N87")


@dataclass
class MagnetDatabase:
    """In-memory view of one ``<material>_database.h5`` file."""

    material: str
    path: Path
    sha256: str
    columns: tuple[str, ...]
    values: np.ndarray  # float64 [n, 7]
    metadata: dict[str, str] = field(default_factory=dict)
    core_info: dict[str, Any] = field(default_factory=dict)

    def column(self, name: str) -> np.ndarray:
        try:
            return self.values[:, self.columns.index(name)]
        except ValueError:
            raise DataNotFoundError(f"column {name!r} not in {self.path.name}") from None

    @property
    def n_points(self) -> int:
        return int(self.values.shape[0])

    def sinusoidal_zero_bias(self) -> np.ndarray:
        """Row mask: sinusoidal excitation (``Duty_P == -1``) and zero DC bias."""
        return (self.column("Duty_P") == -1.0) & (self.column("Duty_N") == -1.0) & (self.column("DC_Bias") == 0.0)

    def temperatures(self) -> list[float]:
        return sorted(set(np.round(self.column("Temperature"), 1).tolist()))

    def summary(self) -> dict[str, Any]:
        f, b, t, p = (self.column("Frequency"), self.column("Flux_Density"), self.column("Temperature"), self.column("Power_Loss"))
        return {
            "material": self.material, "file": self.path.name, "sha256": self.sha256, "n_points": self.n_points,
            "frequency_range_Hz": [float(f.min()), float(f.max())],
            "flux_density_range_T": [float(b.min()), float(b.max())],
            "temperatures_degC": self.temperatures(),
            "loss_density_range_W_per_m3": [float(p.min()), float(p.max())],
            "n_sinusoidal_zero_bias": int(self.sinusoidal_zero_bias().sum()),
            "core": self.core_info,
            "units": dict(COLUMN_UNITS),
        }


def _parse_protocol0_text(text: str) -> dict[str, str]:
    """Parse the pandas ``metadata`` attribute (a *pickle protocol 0* dict of unicode strings) as text.

    The attribute is never unpickled. Protocol 0 writes ``V<text>\\n`` for unicode strings and
    ``p<n>\\n`` memo markers; dictionary items appear as key, value pairs followed by ``s``.
    """
    items = re.findall(r"V(.*?)\np\d+\n", text, flags=re.DOTALL)
    result: dict[str, str] = {}
    for key, value in zip(items[0::2], items[1::2]):
        result[key.strip()] = value.encode("latin-1", "backslashreplace").decode("raw_unicode_escape").strip()
    return result


def _parse_core_info(info_core: str) -> dict[str, Any]:
    """``'N87 --- R34.0X20.5X12.5 --- Ae 8.26e-05 --- Ve 6.778e-06 --- le 0.08206 --- CoreN 1 --- N1 5 --- N2 5 --- Dataset 5'``."""
    parts = [p.strip() for p in info_core.split("---")]
    out: dict[str, Any] = {}
    if parts:
        out["material"] = parts[0]
    if len(parts) > 1:
        out["core"] = parts[1]
    for part in parts[2:]:
        tokens = part.split()
        if len(tokens) == 2:
            key, value = tokens
            try:
                out[key] = float(value) if "." in value or "e" in value.lower() else int(value)
            except ValueError:
                out[key] = value
    units = {"Ae": "m^2", "Ve": "m^3", "le": "m"}
    out["units"] = {k: units[k] for k in units if k in out}
    return out


def read_magnet_database(path: str | Path, *, material: str | None = None) -> MagnetDatabase:
    """Read one ``<material>_database.h5`` file. Raises :class:`DataIntegrityError` on an unexpected layout."""
    try:
        import h5py  # noqa: WPS433 - optional dependency
    except ImportError as error:  # pragma: no cover - depends on environment
        raise UnsupportedOperationError("reading MagNet HDF5 databases requires h5py (pip install 'aipe-magnetics[hdf5]')") from error
    path = Path(path)
    if not path.is_file():
        raise DataNotFoundError(f"MagNet database file not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with h5py.File(path, "r") as handle:
        if "dataframe" not in handle:
            raise DataIntegrityError(f"{path.name}: missing /dataframe group (not a pandas fixed-format store)")
        group = handle["dataframe"]
        if group.attrs.get("pandas_type", b"") not in (b"frame", "frame"):
            raise DataIntegrityError(f"{path.name}: /dataframe is not a pandas frame")
        if int(group.attrs.get("nblocks", 1)) != 1 or "block0_values" not in group:
            raise DataIntegrityError(f"{path.name}: expected a single float block (nblocks=1, block0_values)")
        items = tuple(x.decode() if isinstance(x, bytes) else str(x) for x in group["block0_items"][()])
        values = np.asarray(group["block0_values"][()], dtype=np.float64)
        raw_meta = group.attrs.get("metadata", b"")
        if isinstance(raw_meta, bytes):
            raw_meta = raw_meta.decode("utf-8", "replace")
    if set(items) != set(EXPECTED_COLUMNS):
        raise DataIntegrityError(f"{path.name}: unexpected columns {items}")
    if values.ndim != 2 or values.shape[1] != len(items):
        raise DataIntegrityError(f"{path.name}: block shape {values.shape} does not match {len(items)} columns")
    if not np.all(np.isfinite(values)):
        raise DataIntegrityError(f"{path.name}: non-finite values present")
    metadata = _parse_protocol0_text(raw_meta) if raw_meta else {}
    core_info = _parse_core_info(metadata.get("info_core", "")) if metadata.get("info_core") else {}
    name = material or core_info.get("material") or path.name.split("_database")[0]
    return MagnetDatabase(material=str(name), path=path, sha256=digest, columns=items, values=values,
                          metadata=metadata, core_info=core_info)


def find_database_files(root: str | Path) -> dict[str, Path]:
    """Locate ``<material>_database.h5`` files under a repository checkout or package data directory."""
    root = Path(root)
    candidates = [root, root / "src" / "magnet" / "data", root / "magnet" / "data", root / "data"]
    for base in candidates:
        if base.is_dir():
            files = sorted(base.glob("*_database.h5"))
            if files:
                return {p.name.split("_database")[0]: p for p in files}
    raise DataNotFoundError(f"no <material>_database.h5 files found under {root}")


@dataclass
class SteinmetzFit:
    material: str
    temperature_degC: float
    k: float
    alpha: float
    beta: float
    n_points: int
    frequency_range_Hz: tuple[float, float]
    flux_density_range_T: tuple[float, float]
    max_rel_error: float
    mean_abs_rel_error: float
    rms_log10_residual: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "material": self.material, "temperature_degC": self.temperature_degC,
            "k_W_per_m3": self.k, "alpha": self.alpha, "beta": self.beta, "n_points": self.n_points,
            "frequency_range_Hz": list(self.frequency_range_Hz), "flux_density_range_T": list(self.flux_density_range_T),
            "fit_metrics": {"max_rel_error": self.max_rel_error, "mean_abs_rel_error": self.mean_abs_rel_error,
                            "rms_log10_residual": self.rms_log10_residual},
        }


def fit_steinmetz(frequency: np.ndarray, flux_density: np.ndarray, loss_density: np.ndarray) -> tuple[float, float, float, dict[str, float]]:
    """Ordinary least squares in log space: ``log Pv = log k + alpha log f + beta log B``.

    Returns ``(k, alpha, beta, metrics)`` with k in W/m^3 for f in Hz and B in T. Requires at least 10
    points with positive values; raises :class:`DataIntegrityError` otherwise.
    """
    f, b, p = (np.asarray(frequency, float), np.asarray(flux_density, float), np.asarray(loss_density, float))
    mask = np.isfinite(f) & np.isfinite(b) & np.isfinite(p) & (f > 0) & (b > 0) & (p > 0)
    if mask.sum() < 10:
        raise DataIntegrityError(f"not enough valid points for a Steinmetz fit ({int(mask.sum())})")
    f, b, p = f[mask], b[mask], p[mask]
    design = np.column_stack([np.ones_like(f), np.log(f), np.log(b)])
    coefficients, *_ = np.linalg.lstsq(design, np.log(p), rcond=None)
    log_k, alpha, beta = (float(x) for x in coefficients)
    prediction = np.exp(design @ coefficients)
    rel = prediction / p - 1.0
    metrics = {"max_rel_error": float(np.max(np.abs(rel))), "mean_abs_rel_error": float(np.mean(np.abs(rel))),
               "rms_log10_residual": float(np.sqrt(np.mean((np.log10(prediction) - np.log10(p)) ** 2))), "n_points": int(mask.sum())}
    return float(np.exp(log_k)), alpha, beta, metrics


def fit_steinmetz_by_temperature(db: MagnetDatabase, *, frequency_range: tuple[float, float] | None = None,
                                 flux_density_range: tuple[float, float] | None = None,
                                 min_points: int = 30) -> list[SteinmetzFit]:
    """Fit one classical Steinmetz set per measured temperature from sinusoidal, zero-bias points."""
    mask = db.sinusoidal_zero_bias()
    f, b, t, p = db.column("Frequency"), db.column("Flux_Density"), db.column("Temperature"), db.column("Power_Loss")
    if frequency_range:
        mask &= (f >= frequency_range[0]) & (f <= frequency_range[1])
    if flux_density_range:
        mask &= (b >= flux_density_range[0]) & (b <= flux_density_range[1])
    fits: list[SteinmetzFit] = []
    for temperature in sorted(set(np.round(t[mask], 0).tolist())):
        sel = mask & (np.round(t, 0) == temperature)
        if sel.sum() < min_points:
            continue
        k, alpha, beta, metrics = fit_steinmetz(f[sel], b[sel], p[sel])
        fits.append(SteinmetzFit(material=db.material, temperature_degC=float(temperature), k=k, alpha=alpha, beta=beta,
                                 n_points=metrics["n_points"], frequency_range_Hz=(float(f[sel].min()), float(f[sel].max())),
                                 flux_density_range_T=(float(b[sel].min()), float(b[sel].max())),
                                 max_rel_error=metrics["max_rel_error"], mean_abs_rel_error=metrics["mean_abs_rel_error"],
                                 rms_log10_residual=metrics["rms_log10_residual"]))
    if not fits:
        raise DataNotFoundError(f"{db.material}: no temperature group with at least {min_points} sinusoidal zero-bias points")
    return fits
