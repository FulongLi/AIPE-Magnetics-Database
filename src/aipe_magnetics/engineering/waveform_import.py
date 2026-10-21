"""Import sampled converter waveforms (HDF5, CSV, JSON) into periodic waveform objects with explicit units.

Sources
-------
* **AIPE DAB HDF5 layout** (optional ``h5py``): ``/time`` [s] plus ``/channels/<name>`` datasets whose attributes
  carry ``unit`` (required), ``description`` and ``sign_convention``; the root attribute ``metadata`` is a JSON
  string (producer, solver, specification with ``f_sw``, ``n_ratio``, ``l_ext``, ``l_m`` ...). Nothing is inferred
  from dataset names: the caller maps roles (primary current, secondary current, primary voltage ...) to channel
  names. The file's SHA-256 is recorded with every imported set.
* **CSV**: one time column and value columns; every unit is given explicitly by the caller (never inferred from
  headers); cells must be plain decimal numerals; time must increase strictly.
* **JSON**: ``{"time": {"unit": "us", "values": [...]}, "channels": {"<name>": {"unit": "A", "values": [...],
  "description": ...}}, "metadata": {...}}`` or one uniformly sampled period
  ``{"frequency": {"value": 100, "unit": "kHz"}, "channels": {...}}`` (samples span ``[0, T)``).

Processing
----------
window selection (``t0``, ``t1``) -> period detection (normalised autocorrelation, checked against a frequency
hint when one is given) -> extraction of the **last complete period** of the window (steady state; the
cycle-to-cycle deviation over the window is reported) -> periodic closure (the end value is replaced by the start
value and the residual is reported, never hidden) -> :class:`~aipe_magnetics.engineering.periodic.PiecewiseLinear`
(exact RMS, mean, Fourier coefficients on the original, non-uniform sample times) or flux linkage by trapezoidal
integration of a voltage with the per-period drift removed -> flux-density waveform for given turns and core area.

Everything is SI inside; original units are kept on every channel. Nothing is smoothed or filtered: measured noise
reaches the loss integrals unchanged (as the iGSE documentation requires) and must be removed by the caller.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from aipe_magnetics.common.errors import DataNotFoundError, InvalidInputError, UnsupportedOperationError
from aipe_magnetics.common.jsonio import sha256_file
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.units import units
from aipe_magnetics.engineering.periodic import PiecewiseLinear

#: Closure residual (|v(T) - v(0)| / peak-to-peak) above which a period extraction is flagged.
DEFAULT_CLOSURE_TOLERANCE = 0.02
#: Autocorrelation value at the detected lag below which the record is not considered periodic.
MIN_PERIODICITY = 0.8
_NUMERAL = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def h5py_available() -> bool:
    try:
        import h5py  # noqa: F401
    except ImportError:
        return False
    return True


# -------------------------------------------------------------------------------------------------
# data containers
# -------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SampledChannel:
    """One sampled quantity in SI with its original unit and the attributes the source stated."""

    name: str
    values: np.ndarray
    unit: str
    original_unit: str
    description: str | None = None
    sign_convention: str | None = None

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "unit": self.unit, "original_unit": self.original_unit, "n_samples": int(self.values.size),
                "min": float(self.values.min()), "max": float(self.values.max()), "description": self.description,
                "sign_convention": self.sign_convention}


@dataclass(frozen=True)
class SampledWaveformSet:
    """Time base (s, strictly increasing) plus named channels from one source file or document."""

    time: np.ndarray
    channels: dict[str, SampledChannel]
    source: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    window: tuple[float, float] | None = None

    @property
    def n_samples(self) -> int:
        return int(self.time.size)

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0])

    def channel(self, name: str) -> SampledChannel:
        try:
            return self.channels[name]
        except KeyError:
            raise DataNotFoundError(f"channel {name!r} not in the waveform set; available: {sorted(self.channels)}",
                                    details={"channel": name, "available": sorted(self.channels)}) from None

    def select_window(self, t0: float | None, t1: float | None) -> "SampledWaveformSet":
        """Samples with ``t0 <= t <= t1`` (seconds); the window is recorded. At least two samples must remain."""
        lo = self.time[0] if t0 is None else float(t0)
        hi = self.time[-1] if t1 is None else float(t1)
        if not (hi > lo):
            raise InvalidInputError(f"window end {hi:g} s must exceed the start {lo:g} s")
        tol = 1e-9 * (hi - lo)  # floating-point edges such as 240 us -> 0.00023999999999999998 s keep the boundary samples
        mask = (self.time >= lo - tol) & (self.time <= hi + tol)
        if int(mask.sum()) < 2:
            raise InvalidInputError(f"window [{lo:g}, {hi:g}] s contains fewer than two samples (record spans "
                                    f"{self.time[0]:g}..{self.time[-1]:g} s)")
        channels = {k: SampledChannel(c.name, c.values[mask], c.unit, c.original_unit, c.describe().get("description"), c.sign_convention)
                    for k, c in self.channels.items()}
        return SampledWaveformSet(time=self.time[mask], channels=channels, source=dict(self.source), metadata=dict(self.metadata),
                                  window=(lo, hi))

    def describe(self) -> dict[str, Any]:
        return {"source": dict(self.source), "n_samples": self.n_samples, "t_start_s": float(self.time[0]), "t_end_s": float(self.time[-1]),
                "window_s": None if self.window is None else list(self.window), "channels": {k: c.describe() for k, c in self.channels.items()}}


@dataclass(frozen=True)
class PeriodEstimate:
    period: float
    frequency: float
    method: str
    periodicity: float | None = None          # normalised autocorrelation at the period lag (1 = perfectly periodic)
    hint_frequency: float | None = None
    hint_mismatch: float | None = None        # (f_detected - f_hint) / f_hint
    periods_in_record: float | None = None
    notes: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {"period_s": self.period, "frequency_Hz": self.frequency, "method": self.method, "periodicity": self.periodicity,
                "hint_frequency_Hz": self.hint_frequency, "hint_mismatch": self.hint_mismatch, "periods_in_record": self.periods_in_record,
                "notes": list(self.notes)}


@dataclass(frozen=True)
class PeriodSlice:
    """One period of a sampled quantity on the normalised time axis ``x = (t - t_start) / T`` in ``[0, 1]``."""

    x: np.ndarray
    values: np.ndarray
    t_start: float
    t_end: float
    closure_residual: float
    unit: str
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def period(self) -> float:
        return self.t_end - self.t_start

    def to_piecewise_linear(self, label: str = "") -> PiecewiseLinear:
        return PiecewiseLinear.from_lists(self.x.tolist(), self.values.tolist(), unit=self.unit, label=label)


# -------------------------------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------------------------------
def _si_array(values: Any, unit: str, name: str) -> tuple[np.ndarray, str]:
    """Finite float64 array converted to the canonical SI unit of ``unit`` (unknown units are rejected)."""
    si_unit = units.si_unit_of(unit)
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InvalidInputError(f"{name}: values are not numeric: {error}") from None
    if array.ndim != 1 or array.size < 2:
        raise InvalidInputError(f"{name}: expected a one-dimensional array with at least two samples, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise InvalidInputError(f"{name}: values must be finite (no NaN / inf)")
    if unit != si_unit:
        from aipe_magnetics.data.hdf5 import convert_array
        array = convert_array(array, unit, si_unit)
    return array, si_unit


def _time_axis(values: Any, unit: str) -> np.ndarray:
    t, si = _si_array(values, unit, "time")
    if si != "s":
        raise InvalidInputError(f"time unit {unit!r} does not have the dimension of seconds")
    if not np.all(np.diff(t) > 0):
        raise InvalidInputError("time samples must increase strictly")
    return t


def _attr_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        value = value.tolist()
    text = str(value)
    return text if text else None


def _window_tuple(window: Any) -> tuple[float | None, float | None] | None:
    if window is None:
        return None
    if isinstance(window, Mapping):
        t0, t1 = window.get("t0"), window.get("t1")
    elif isinstance(window, (tuple, list)) and len(window) == 2:
        t0, t1 = window
    else:
        raise InvalidInputError("window must be (t0, t1) in seconds or {'t0': ..., 't1': ...} (quantities or seconds)")

    def sec(v: Any) -> float | None:
        if v is None:
            return None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        q = Quantity.parse(v, name="window")
        return q.magnitude("s")

    return sec(t0), sec(t1)


# -------------------------------------------------------------------------------------------------
# loaders
# -------------------------------------------------------------------------------------------------
def load_waveforms_hdf5(path: str | Path, *, channels: list[str] | None = None, time_dataset: str = "time",
                        channel_group: str = "channels", window: Any = None) -> SampledWaveformSet:
    """Read the AIPE DAB HDF5 layout; ``channels`` limits the datasets read (``None`` = every dataset in the group)."""
    if not h5py_available():
        raise UnsupportedOperationError("reading HDF5 waveform files needs h5py: install aipe-magnetics[hdf5]", details={"extra": "hdf5"})
    import h5py  # type: ignore

    file = Path(path)
    if not file.is_file():
        raise DataNotFoundError(f"waveform file not found: {file}", details={"path": str(file)})
    digest = sha256_file(file)
    with h5py.File(file, "r") as handle:
        if time_dataset not in handle:
            raise InvalidInputError(f"{file.name}: time dataset {time_dataset!r} not found")
        t_ds = handle[time_dataset]
        t_unit = _attr_text(t_ds.attrs.get("unit")) or "s"
        time = _time_axis(t_ds[()], t_unit)
        if channel_group not in handle:
            raise InvalidInputError(f"{file.name}: channel group {channel_group!r} not found")
        group = handle[channel_group]
        available = sorted(name for name in group if hasattr(group[name], "shape"))
        wanted = available if channels is None else list(channels)
        loaded: dict[str, SampledChannel] = {}
        for name in wanted:
            if name not in group or not hasattr(group[name], "shape"):
                raise DataNotFoundError(f"{file.name}: channel {name!r} not found in /{channel_group}; available: {available}",
                                        details={"channel": name, "available": available})
            ds = group[name]
            unit = _attr_text(ds.attrs.get("unit"))
            if not unit:
                raise InvalidInputError(f"{file.name}: channel {name!r} has no 'unit' attribute; units are never inferred")
            values, si_unit = _si_array(ds[()], unit, f"channel {name}")
            if values.size != time.size:
                raise InvalidInputError(f"{file.name}: channel {name!r} has {values.size} samples, time has {time.size}")
            loaded[name] = SampledChannel(name=name, values=values, unit=si_unit, original_unit=unit,
                                          description=_attr_text(ds.attrs.get("description")),
                                          sign_convention=_attr_text(ds.attrs.get("sign_convention")))
        metadata: dict[str, Any] = {}
        raw_meta = _attr_text(handle.attrs.get("metadata"))
        if raw_meta:
            try:
                parsed = json.loads(raw_meta)
                metadata = parsed if isinstance(parsed, dict) else {"metadata": parsed}
            except json.JSONDecodeError:
                metadata = {"metadata_text": raw_meta}
        root_attrs = {k: _attr_text(v) for k, v in handle.attrs.items() if k != "metadata"}
    source = {"kind": "hdf5", "path": str(file.resolve()), "uri": file.resolve().as_uri(), "sha256": digest, "size_bytes": file.stat().st_size,
              "media_type": "application/x-hdf5", "time_dataset": time_dataset, "channel_group": channel_group, "channels_read": wanted,
              "root_attributes": root_attrs}
    result = SampledWaveformSet(time=time, channels=loaded, source=source, metadata=metadata)
    w = _window_tuple(window)
    return result if w is None else result.select_window(*w)


def load_waveforms_csv(path: str | Path, *, time_column: str, time_unit: str, columns: Mapping[str, tuple[str, str] | list],
                       delimiter: str = ",", window: Any = None) -> SampledWaveformSet:
    """Read a CSV with one time column; ``columns`` maps channel name -> (CSV header, unit). Units are never inferred."""
    file = Path(path)
    if not file.is_file():
        raise DataNotFoundError(f"waveform file not found: {file}", details={"path": str(file)})
    if not isinstance(columns, Mapping) or not columns:
        raise InvalidInputError("columns must map channel name -> (CSV header, unit)")
    mapping: dict[str, tuple[str, str]] = {}
    for name, spec in columns.items():
        if not isinstance(spec, (tuple, list)) or len(spec) != 2 or not all(isinstance(s, str) and s for s in spec):
            raise InvalidInputError(f"columns[{name!r}] must be a (CSV header, unit) pair of strings, got {spec!r}")
        mapping[str(name)] = (spec[0], spec[1])
    text = file.read_text(encoding="utf-8-sig")
    reader = csv.reader(text.splitlines(), delimiter=delimiter)
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if len(rows) < 3:
        raise InvalidInputError(f"{file.name}: a CSV waveform needs a header and at least two data rows")
    header = [h.strip() for h in rows[0]]
    needed = [time_column] + [h for h, _ in mapping.values()]
    missing = [h for h in needed if h not in header]
    if missing:
        raise InvalidInputError(f"{file.name}: columns {missing} not in the header {header}")
    index = {h: header.index(h) for h in needed}

    def parse(cell: str, line: int, name: str) -> float:
        token = cell.strip()
        if not _NUMERAL.match(token):
            raise InvalidInputError(f"{file.name} line {line}: column {name!r} cell {cell!r} is not a plain decimal numeral")
        return float(token)

    data: dict[str, list[float]] = {h: [] for h in needed}
    for line_no, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise InvalidInputError(f"{file.name} line {line_no}: expected {len(header)} cells, got {len(row)}")
        for h in needed:
            data[h].append(parse(row[index[h]], line_no, h))
    time = _time_axis(data[time_column], time_unit)
    channels: dict[str, SampledChannel] = {}
    for name, (col, unit) in mapping.items():
        values, si_unit = _si_array(data[col], unit, f"column {col}")
        channels[name] = SampledChannel(name=name, values=values, unit=si_unit, original_unit=unit, description=f"CSV column {col!r}")
    source = {"kind": "csv", "path": str(file.resolve()), "uri": file.resolve().as_uri(), "sha256": sha256_file(file),
              "size_bytes": file.stat().st_size, "media_type": "text/csv", "time_column": time_column, "time_unit": time_unit,
              "columns": {k: {"header": v[0], "unit": v[1]} for k, v in mapping.items()}}
    result = SampledWaveformSet(time=time, channels=channels, source=source, metadata={})
    w = _window_tuple(window)
    return result if w is None else result.select_window(*w)


def load_waveforms_json(source: str | Path | Mapping[str, Any], *, window: Any = None) -> SampledWaveformSet:
    """Read the JSON layout described in the module docstring from a file or an in-memory mapping."""
    info: dict[str, Any]
    if isinstance(source, Mapping):
        document = dict(source)
        info = {"kind": "json", "path": None, "sha256": None, "inline": True}
    else:
        file = Path(source)
        if not file.is_file():
            raise DataNotFoundError(f"waveform file not found: {file}", details={"path": str(file)})
        from aipe_magnetics.common.jsonio import load
        try:
            document = load(file)
        except ValueError as error:
            raise InvalidInputError(f"{file.name}: invalid JSON: {error}") from None
        if not isinstance(document, dict):
            raise InvalidInputError(f"{file.name}: the JSON document must be an object")
        info = {"kind": "json", "path": str(file.resolve()), "uri": file.resolve().as_uri(), "sha256": sha256_file(file),
                "size_bytes": file.stat().st_size, "media_type": "application/json", "inline": False}
    raw_channels = document.get("channels")
    if not isinstance(raw_channels, dict) or not raw_channels:
        raise InvalidInputError("JSON waveform document needs a non-empty 'channels' object")
    channels: dict[str, SampledChannel] = {}
    n = None
    for name, item in raw_channels.items():
        if not isinstance(item, dict) or "values" not in item or "unit" not in item:
            raise InvalidInputError(f"channel {name!r} must be an object with 'unit' and 'values'")
        values, si_unit = _si_array(item["values"], str(item["unit"]), f"channel {name}")
        if n is None:
            n = values.size
        elif values.size != n:
            raise InvalidInputError(f"channel {name!r} has {values.size} samples, previous channels have {n}")
        channels[str(name)] = SampledChannel(name=str(name), values=values, unit=si_unit, original_unit=str(item["unit"]),
                                             description=item.get("description"), sign_convention=item.get("sign_convention"))
    assert n is not None
    if "time" in document:
        t_item = document["time"]
        if not isinstance(t_item, dict) or "values" not in t_item or "unit" not in t_item:
            raise InvalidInputError("'time' must be an object with 'unit' and 'values'")
        time = _time_axis(t_item["values"], str(t_item["unit"]))
        if time.size != n:
            raise InvalidInputError(f"time has {time.size} samples, channels have {n}")
        info["timebase"] = "explicit"
    elif "frequency" in document:
        f = Quantity.parse(document["frequency"], name="frequency").magnitude("Hz")
        if not f > 0:
            raise InvalidInputError("frequency must be positive")
        # one period sampled on [0, T): the period is closed by appending the first sample at t = T (periodicity)
        time = np.arange(n + 1) / (f * n)
        channels = {k: SampledChannel(c.name, np.append(c.values, c.values[0]), c.unit, c.original_unit, c.description, c.sign_convention)
                    for k, c in channels.items()}
        info["timebase"] = "uniform_one_period"
        info["frequency_Hz"] = f
        info["closed_by_periodicity"] = True
    else:
        raise InvalidInputError("JSON waveform document needs 'time' (explicit samples) or 'frequency' (one uniform period)")
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    result = SampledWaveformSet(time=time, channels=channels, source=info, metadata=dict(metadata))
    w = _window_tuple(window)
    return result if w is None else result.select_window(*w)


# -------------------------------------------------------------------------------------------------
# period detection and extraction
# -------------------------------------------------------------------------------------------------
def _uniform(time: np.ndarray, values: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    tu = np.linspace(time[0], time[-1], n)
    return tu, np.interp(tu, time, values)


def _acf(x: np.ndarray) -> np.ndarray:
    """Unbiased normalised autocorrelation (lag ``k`` divided by the ``n - k`` overlapping samples); 1 at lag 0."""
    n = x.size
    spec = np.fft.rfft(x - x.mean(), 2 * n)
    acf = np.fft.irfft(spec * np.conj(spec))[:n]
    acf = acf * (n / (n - np.arange(n, dtype=float)))
    return acf / acf[0] if acf[0] > 0 else acf


def _refine_peak(acf: np.ndarray, lag: int) -> float:
    """Parabolic interpolation of the autocorrelation maximum around integer ``lag``."""
    if lag <= 0 or lag >= acf.size - 1:
        return float(lag)
    y0, y1, y2 = acf[lag - 1], acf[lag], acf[lag + 1]
    denominator = y0 - 2 * y1 + y2
    if abs(denominator) < 1e-15:
        return float(lag)
    return float(lag + 0.5 * (y0 - y2) / denominator)


def _zero_crossing_refine(t: np.ndarray, x: np.ndarray, period0: float) -> tuple[float, int] | None:
    """Least-squares period from the positive-going zero crossings of the mean-removed record (sub-sample accuracy).

    Crossings are assigned to period indices with the coarse estimate ``period0``; crossings that do not fall near an
    integer period offset (harmonic-induced extra crossings) are discarded. Returns ``(period, crossings_used)`` or
    ``None`` when fewer than two consistent crossings exist.
    """
    y = x - x.mean()
    idx = np.where((y[:-1] < 0.0) & (y[1:] >= 0.0))[0]
    if idx.size < 2:
        return None
    frac = -y[idx] / (y[idx + 1] - y[idx])
    crossings = t[idx] + frac * (t[idx + 1] - t[idx])
    offsets = (crossings - crossings[0]) / period0
    k = np.round(offsets)
    good = np.abs(offsets - k) < 0.25
    if int(good.sum()) < 2 or np.ptp(k[good]) < 1:
        return None
    kk, cc = k[good], crossings[good]
    slope = float(np.polyfit(kk, cc, 1)[0])
    return (slope, int(good.sum())) if slope > 0 else None


def detect_period(time: Any, values: Any, *, frequency_hint: float | None = None, min_periods: float = 2.0) -> PeriodEstimate:
    """Fundamental period of a (multi-period) record by normalised autocorrelation.

    With ``frequency_hint`` the hint period is checked (autocorrelation at that lag, refined locally) and the
    detected/hint mismatch is reported; without it the dominant spectral line selects the search region. The record
    must hold at least ``min_periods`` periods for an autonomous detection; with a hint a single period is accepted
    (the hint is then used unchecked and noted).
    """
    t = np.asarray(time, dtype=float)
    x = np.asarray(values, dtype=float)
    if t.ndim != 1 or t.size < 8 or x.shape != t.shape:
        raise InvalidInputError("period detection needs one-dimensional time and value arrays of equal length (>= 8 samples)")
    if not np.all(np.diff(t) > 0):
        raise InvalidInputError("time samples must increase strictly")
    duration = float(t[-1] - t[0])
    if frequency_hint is not None and not (frequency_hint > 0 and math.isfinite(frequency_hint)):
        raise InvalidInputError("frequency_hint must be a positive finite number (Hz)")
    n = int(min(1 << 17, max(4096, 1 << int(math.ceil(math.log2(max(8, 8 * t.size)))))))
    tu, xu = _uniform(t, x, n)
    dt = float(tu[1] - tu[0])
    if float(np.ptp(xu)) <= 0.0:
        raise InvalidInputError("the record is constant; no period can be detected")
    acf = _acf(xu)
    notes: list[str] = []
    if frequency_hint is not None:
        t_hint = 1.0 / frequency_hint
        periods = duration / t_hint
        if periods < 1.0 - 1e-9:
            raise InvalidInputError(f"the record ({duration:g} s) is shorter than one period of the hint frequency ({t_hint:g} s)")
        if periods < min_periods:
            notes.append(f"record holds {periods:.2f} period(s) of the hint frequency; periodicity not checked (needs {min_periods:g})")
            return PeriodEstimate(period=t_hint, frequency=frequency_hint, method="frequency_hint_unchecked", hint_frequency=frequency_hint,
                                  periods_in_record=periods, notes=tuple(notes))
        lag0 = t_hint / dt
        lo, hi = int(max(2, math.floor(lag0 * 0.9))), int(min(acf.size - 2, math.ceil(lag0 * 1.1)))
        if hi <= lo:
            raise InvalidInputError("the record is too short to check the hint frequency")
        k = lo + int(np.argmax(acf[lo:hi + 1]))
        lag = _refine_peak(acf, k)
        period = lag * dt
        periodicity = float(min(acf[k], 1.0))
        refined = _zero_crossing_refine(t, x, period)
        if refined is not None:
            period = refined[0]
        mismatch = (1.0 / period - frequency_hint) / frequency_hint
        if periodicity < MIN_PERIODICITY:
            notes.append(f"autocorrelation at the hint period is only {periodicity:.3f} (< {MIN_PERIODICITY:g}); the record may not be periodic "
                         "or not in steady state")
        if abs(mismatch) > 0.01:
            notes.append(f"detected frequency deviates {mismatch:+.2%} from the hint; the hint period is used")
        return PeriodEstimate(period=t_hint, frequency=frequency_hint, method="frequency_hint_checked_by_autocorrelation",
                              periodicity=periodicity, hint_frequency=frequency_hint, hint_mismatch=mismatch, periods_in_record=periods,
                              notes=tuple(notes))
    # autonomous: dominant spectral line (excluding DC and lines slower than min_periods per record), refined by the ACF peak
    spectrum = np.abs(np.fft.rfft(xu - xu.mean()))
    freqs = np.fft.rfftfreq(n, dt)
    valid = freqs >= min_periods / duration
    if not np.any(valid):
        raise InvalidInputError(f"record too short for autonomous period detection (needs >= {min_periods:g} periods)")
    f_dom = float(freqs[valid][int(np.argmax(spectrum[valid]))])
    lag0 = 1.0 / (f_dom * dt)
    lo, hi = int(max(2, math.floor(lag0 * 0.7))), int(min(int(0.6 * acf.size), math.ceil(lag0 * 1.3)))
    if hi <= lo:
        raise InvalidInputError(f"record too short for autonomous period detection (needs >= {min_periods:g} periods)")
    k = lo + int(np.argmax(acf[lo:hi + 1]))
    lag = _refine_peak(acf, k)
    period = lag * dt
    periodicity = float(min(acf[k], 1.0))
    method = "autocorrelation_peak_near_dominant_spectral_line"
    refined = _zero_crossing_refine(t, x, period)
    if refined is not None:
        period = refined[0]
        method += "_refined_by_zero_crossings"
        notes.append(f"period refined with {refined[1]} positive-going zero crossings")
    if periodicity < MIN_PERIODICITY:
        raise InvalidInputError(f"no periodic structure detected: the autocorrelation at the best candidate period ({period:g} s) is only "
                                f"{periodicity:.3f} (< {MIN_PERIODICITY:g}); the record may hold less than {min_periods:g} periods or not be "
                                "in steady state - pass frequency_hint to use a known period", details={"periodicity": periodicity, "period": period})
    if duration / period < min_periods * (1 - 1e-9):
        raise InvalidInputError(f"the record holds only {duration / period:.2f} period(s) of the detected {period:g} s; autonomous detection needs "
                                f">= {min_periods:g}", details={"period": period, "periods_in_record": duration / period})
    return PeriodEstimate(period=period, frequency=1.0 / period, method=method, periodicity=periodicity,
                          periods_in_record=duration / period, notes=tuple(notes))


def extract_period(time: Any, values: Any, period: float, *, t_end: float | None = None, unit: str = "1",
                   closure_tolerance: float = DEFAULT_CLOSURE_TOLERANCE) -> PeriodSlice:
    """The last complete period ending at ``t_end`` (default: the last sample) on the normalised axis ``x in [0, 1]``.

    Original sample times inside the period are kept (no resampling); the two end points are interpolated. The end
    value is replaced by the start value so that the slice is exactly periodic; the relative closure residual
    ``|v(t_end) - v(t_start)| / peak_to_peak`` is reported (``info['closure_flagged']`` when above the tolerance).
    """
    t = np.asarray(time, dtype=float)
    x = np.asarray(values, dtype=float)
    if not (period > 0 and math.isfinite(period)):
        raise InvalidInputError("period must be a positive finite number (s)")
    end = float(t[-1]) if t_end is None else float(t_end)
    start = end - period
    if start < t[0] - 1e-12 * period or end > t[-1] + 1e-12 * period:
        raise InvalidInputError(f"period [{start:g}, {end:g}] s is not inside the record {t[0]:g}..{t[-1]:g} s")
    v_start, v_end = float(np.interp(start, t, x)), float(np.interp(end, t, x))
    inside = (t > start) & (t < end)
    tt = np.concatenate([[start], t[inside], [end]])
    vv = np.concatenate([[v_start], x[inside], [v_end]])
    keep = np.concatenate([[True], np.diff(tt) > 0])
    tt, vv = tt[keep], vv[keep]
    xn = (tt - start) / period
    xn[0], xn[-1] = 0.0, 1.0
    pp = float(vv.max() - vv.min())
    residual = abs(v_end - v_start) / pp if pp > 0 else 0.0
    vv[-1] = vv[0]
    info = {"closure_residual": residual, "closure_flagged": residual > closure_tolerance, "closure_tolerance": closure_tolerance,
            "n_breakpoints": int(xn.size), "t_start_s": start, "t_end_s": end}
    return PeriodSlice(x=xn, values=vv, t_start=start, t_end=end, closure_residual=residual, unit=unit, info=info)


def steady_state_metrics(time: Any, values: Any, period: float, *, t_end: float | None = None, max_cycles: int = 8) -> dict[str, Any]:
    """Cycle-to-cycle variation of RMS and peak over the complete periods ending at ``t_end`` (steady-state check)."""
    t = np.asarray(time, dtype=float)
    x = np.asarray(values, dtype=float)
    end = float(t[-1]) if t_end is None else float(t_end)
    rms: list[float] = []
    peaks: list[float] = []
    cycle_end = end
    while cycle_end - period >= t[0] - 1e-12 * period and len(rms) < max_cycles:
        s = extract_period(t, x, period, t_end=cycle_end)
        pl = s.to_piecewise_linear()
        rms.append(pl.rms())
        peaks.append(pl.peak())
        cycle_end -= period
    if not rms:
        return {"cycles": 0}
    ref = rms[0] if rms[0] > 0 else 1.0
    return {"cycles": len(rms), "rms_by_cycle": rms, "peak_by_cycle": peaks,
            "rms_relative_spread": (max(rms) - min(rms)) / ref, "peak_relative_spread": (max(peaks) - min(peaks)) / (peaks[0] or 1.0)}


def flux_linkage_from_voltage(time: Any, voltage: Any, period: float, *, t_end: float | None = None) -> PeriodSlice:
    """Flux linkage ``lambda(t) = integral v dt`` over the last period: trapezoidal on the original samples, the net
    volt-seconds per period removed as a linear drift (reported), zero time average. Unit ``V*s``."""
    t = np.asarray(time, dtype=float)
    v = np.asarray(voltage, dtype=float)
    end = float(t[-1]) if t_end is None else float(t_end)
    start = end - period
    if start < t[0] - 1e-12 * period:
        raise InvalidInputError("the voltage record is shorter than one period")
    inside = (t > start) & (t < end)
    tt = np.concatenate([[start], t[inside], [end]])
    vv = np.concatenate([[float(np.interp(start, t, v))], v[inside], [float(np.interp(end, t, v))]])
    keep = np.concatenate([[True], np.diff(tt) > 0])
    tt, vv = tt[keep], vv[keep]
    lam = np.concatenate([[0.0], np.cumsum(0.5 * (vv[1:] + vv[:-1]) * np.diff(tt))])
    net = float(lam[-1] - lam[0])
    xn = (tt - start) / period
    xn[0], xn[-1] = 0.0, 1.0
    lam = lam - net * xn                       # remove the per-period drift so that lambda is periodic
    mean = float(np.sum(0.5 * (lam[1:] + lam[:-1]) * np.diff(xn)))
    lam = lam - mean
    lam[-1] = lam[0]
    pp = float(lam.max() - lam.min())
    info = {"flux_linkage_peak_to_peak_Vs": pp, "net_volt_seconds_per_period_Vs": net,
            "volt_second_relative_imbalance": abs(net) / pp if pp > 0 else 0.0, "voltage_rms_V": float(math.sqrt(np.sum(
                (vv[:-1] ** 2 + vv[:-1] * vv[1:] + vv[1:] ** 2) / 3.0 * np.diff(tt)) / period)),
            "n_breakpoints": int(xn.size), "t_start_s": start, "t_end_s": end, "drift_removed": net != 0.0}
    return PeriodSlice(x=xn, values=lam, t_start=start, t_end=end, closure_residual=0.0, unit="V*s", info=info)


def flux_density_piecewise(flux_linkage: PiecewiseLinear, turns: int, area: float, label: str = "flux density") -> PiecewiseLinear:
    """Flux density ``B(t) = lambda(t) / (N A_e)`` in tesla as a periodic piecewise-linear waveform."""
    if isinstance(turns, bool) or not isinstance(turns, int) or turns < 1:
        raise InvalidInputError("turns must be a positive integer")
    if not (area > 0 and math.isfinite(area)):
        raise InvalidInputError("area must be positive (m^2)")
    return flux_linkage.scaled(1.0 / (turns * area), unit="T", label=label)


def flux_density_breakpoints(flux_linkage: PiecewiseLinear, turns: int, area: float) -> list[list[float]]:
    """Normalised ``[[x, B / B_peak], ...]`` breakpoints for ``WaveformSpec(kind='piecewise_linear')`` plus ``B_peak``."""
    if isinstance(turns, bool) or not isinstance(turns, int) or turns < 1:
        raise InvalidInputError("turns must be a positive integer")
    if not (area > 0 and math.isfinite(area)):
        raise InvalidInputError("area must be positive (m^2)")
    b = np.array([p[1] for p in flux_linkage.points]) / (turns * area)
    peak = 0.5 * float(b.max() - b.min())
    if peak <= 0:
        raise InvalidInputError("flux linkage has zero excursion")
    norm = b / peak
    norm[-1] = norm[0]
    return [[float(p[0]), float(v)] for p, v in zip(flux_linkage.points, norm)]


def square_piecewise(amplitude: float, duty: float = 0.5, unit: str = "A", label: str = "", edge: float = 1e-4) -> PiecewiseLinear:
    """Bipolar square wave ``+-amplitude`` (``+`` for ``duty`` of the period) with linear edges of fraction ``edge``."""
    if not (0 < duty < 1):
        raise InvalidInputError("duty must be in (0, 1)")
    e = edge
    pts = [(0.0, -amplitude), (e, amplitude), (duty, amplitude), (duty + e, -amplitude), (1.0, -amplitude)]
    return PiecewiseLinear(points=tuple(pts), unit=unit, label=label)


__all__ = [
    "DEFAULT_CLOSURE_TOLERANCE", "MIN_PERIODICITY", "PeriodEstimate", "PeriodSlice", "SampledChannel", "SampledWaveformSet",
    "detect_period", "extract_period", "flux_density_breakpoints", "flux_density_piecewise", "flux_linkage_from_voltage", "h5py_available",
    "load_waveforms_csv", "load_waveforms_hdf5", "load_waveforms_json", "square_piecewise", "steady_state_metrics",
]
