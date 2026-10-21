"""Waveform import: HDF5 / CSV / JSON loaders with explicit units, window selection, period detection, closure, flux integration."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from aipe_magnetics.common.errors import DataNotFoundError, InvalidInputError
from aipe_magnetics.common.jsonio import sha256_file
from aipe_magnetics.engineering import waveform_import as wi

REAL_FILE = Path("D:/Cursor Workspace/AIPE-Full-Ecosystem-Integration/runs/dab-l1-nominal-ic/waveforms.h5")
F_SW = 100e3


def _sps_current(t: np.ndarray, *, v1: float = 800.0, v2n: float = 800.0, inductance: float = 35e-6, phi: float = math.radians(22.5)) -> np.ndarray:
    """Exact SPS trapezoidal inductor current evaluated on arbitrary times (through the analytical model)."""
    from aipe_magnetics.design.transformers.dab import inductor_current
    pl = inductor_current(v1, v2n, 1.0, F_SW, inductance, phi)
    return pl.value(t * F_SW)


def _square(t: np.ndarray, amplitude: float) -> np.ndarray:
    return amplitude * np.where(np.mod(t * F_SW, 1.0) < 0.5, 1.0, -1.0)


@pytest.fixture
def synthetic_hdf5(tmp_path: Path) -> Path:
    h5py = pytest.importorskip("h5py", reason="the HDF5 waveform loader needs h5py (aipe-magnetics[hdf5])")
    t = np.linspace(0.0, 60e-6, 6001)  # 6 periods, 1000 samples per period, 10 ns step
    path = tmp_path / "waveforms.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["metadata"] = json.dumps({"spec": {"f_sw": F_SW, "n_ratio": 2.0, "l_ext": 35e-6, "l_lk_p": 0.0, "l_lk_s": 0.0, "l_m": 1e-3}})
        ds = handle.create_dataset("time", data=t * 1e6)
        ds.attrs["unit"] = "us"
        group = handle.create_group("channels")
        i = group.create_dataset("i_L", data=_sps_current(t) * 1e3)   # stored in mA
        i.attrs["unit"] = "mA"
        i.attrs["description"] = "series inductor current"
        i.attrs["sign_convention"] = "from node a towards the transformer"
        v = group.create_dataset("v_tp", data=_square(t, 800.0))
        v.attrs["unit"] = "V"
        group.create_dataset("no_unit", data=t)
    return path


def test_hdf5_loader_converts_units_reads_metadata_and_records_sha256(synthetic_hdf5: Path):
    ws = wi.load_waveforms_hdf5(synthetic_hdf5, channels=["i_L", "v_tp"])
    assert ws.n_samples == 6001 and ws.time[0] == 0.0 and abs(ws.time[-1] - 60e-6) < 1e-15
    ch = ws.channel("i_L")
    assert ch.unit == "A" and ch.original_unit == "mA" and ch.sign_convention.startswith("from node a")
    assert abs(ch.values.max() - 14.3) < 0.2  # kA/mA conversion applied: amperes, not milliamperes
    assert ws.metadata["spec"]["f_sw"] == F_SW
    assert ws.source["sha256"] == sha256_file(synthetic_hdf5) and ws.source["uri"].startswith("file:///")
    with pytest.raises(DataNotFoundError):
        ws.channel("i_s")
    with pytest.raises(DataNotFoundError):
        wi.load_waveforms_hdf5(synthetic_hdf5, channels=["missing"])
    with pytest.raises(InvalidInputError, match="no 'unit' attribute"):
        wi.load_waveforms_hdf5(synthetic_hdf5, channels=["no_unit"])


def test_window_selection_keeps_boundary_samples_despite_floating_point_edges(synthetic_hdf5: Path):
    ws = wi.load_waveforms_hdf5(synthetic_hdf5, channels=["i_L"], window={"t0": {"value": 20, "unit": "us"}, "t1": {"value": 60, "unit": "us"}})
    assert ws.window[0] == pytest.approx(20e-6) and ws.n_samples == 4001  # 20.000 us and 60.000 us samples are both kept
    assert abs(ws.time[-1] - 60e-6) < 1e-15
    with pytest.raises(InvalidInputError):
        wi.load_waveforms_hdf5(synthetic_hdf5, channels=["i_L"], window=(50e-6, 40e-6))


def test_csv_loader_requires_explicit_units_and_plain_numerals(tmp_path: Path):
    t = np.linspace(0.0, 20e-6, 401)
    lines = ["t_us,i_p_A,v_V"] + [f"{a * 1e6:.6f},{b:.6f},{c:.3f}" for a, b, c in zip(t, _sps_current(t), _square(t, 800.0))]
    path = tmp_path / "wave.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ws = wi.load_waveforms_csv(path, time_column="t_us", time_unit="us", columns={"i_p": ("i_p_A", "A"), "v_p": ("v_V", "V")})
    assert ws.channel("i_p").unit == "A" and ws.channel("v_p").original_unit == "V" and ws.n_samples == 401
    assert abs(ws.time[-1] - 20e-6) < 1e-12 and ws.source["kind"] == "csv" and ws.source["sha256"] == sha256_file(path)
    with pytest.raises(InvalidInputError, match="not in the header"):
        wi.load_waveforms_csv(path, time_column="t_us", time_unit="us", columns={"i_p": ("missing", "A")})
    bad = tmp_path / "bad.csv"
    bad.write_text("t,i\n0,1\n1e-6,nan\n2e-6,2\n", encoding="utf-8")
    with pytest.raises(InvalidInputError, match="plain decimal numeral"):
        wi.load_waveforms_csv(bad, time_column="t", time_unit="s", columns={"i": ("i", "A")})
    with pytest.raises(InvalidInputError):
        wi.load_waveforms_csv(path, time_column="t_us", time_unit="A", columns={"i_p": ("i_p_A", "A")})  # time unit must be seconds


def test_json_loader_explicit_time_and_uniform_period(tmp_path: Path):
    t = np.linspace(0.0, 30e-6, 3001)
    doc = {"time": {"unit": "s", "values": t.tolist()}, "channels": {"i_p": {"unit": "A", "values": _sps_current(t).tolist()}}, "metadata": {"source": "test"}}
    path = tmp_path / "wave.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    ws = wi.load_waveforms_json(path)
    assert ws.n_samples == 3001 and ws.metadata == {"source": "test"} and ws.source["sha256"] == sha256_file(path)
    inline = wi.load_waveforms_json({"frequency": {"value": 100, "unit": "kHz"}, "channels": {"i": {"unit": "A", "values": [0, 1, 0, -1] * 8}}})
    assert inline.n_samples == 33 and inline.source["inline"] is True and inline.source["timebase"] == "uniform_one_period"
    assert abs(inline.time[-1] - 10e-6) < 1e-15 and inline.channel("i").values[-1] == inline.channel("i").values[0]  # [0, T) closed at T
    with pytest.raises(InvalidInputError):
        wi.load_waveforms_json({"channels": {"i": {"unit": "A", "values": [0, 1]}}})  # neither time nor frequency
    with pytest.raises(InvalidInputError):
        wi.load_waveforms_json({"frequency": {"value": 1, "unit": "kHz"}, "channels": {"i": {"values": [0, 1, 0, 1]}}})  # unit missing


def test_detect_period_autonomously_and_against_a_hint():
    t = np.linspace(0.0, 50e-6, 5001)
    x = _sps_current(t)
    auto = wi.detect_period(t, x)
    assert auto.frequency == pytest.approx(F_SW, rel=1e-6) and auto.periodicity > 0.999 and auto.periods_in_record == pytest.approx(5.0, abs=1e-3)
    hinted = wi.detect_period(t, x, frequency_hint=F_SW)
    assert hinted.period == 1.0 / F_SW and abs(hinted.hint_mismatch) < 1e-6 and hinted.notes == ()
    off = wi.detect_period(t, x, frequency_hint=0.9 * F_SW)
    assert any("deviates" in n for n in off.notes) and off.period == pytest.approx(1 / (0.9 * F_SW))
    with pytest.raises(InvalidInputError):
        wi.detect_period(t, np.ones_like(t))  # constant record has no period
    short = np.linspace(0.0, 8e-6, 801)
    with pytest.raises(InvalidInputError):
        wi.detect_period(short, _sps_current(short))  # less than two periods without a hint


def test_extract_period_closes_the_waveform_and_reports_the_residual():
    t = np.linspace(0.0, 25e-6, 2501)
    x = _sps_current(t)
    period = wi.extract_period(t, x, 1 / F_SW, unit="A")
    pl = period.to_piecewise_linear("i")
    assert period.x[0] == 0.0 and period.x[-1] == 1.0 and period.closure_residual < 1e-6 and pl.unit == "A"
    assert pl.rms() == pytest.approx(13.68, rel=0.005) and pl.peak() == pytest.approx(14.3, rel=0.01) and abs(pl.mean()) < 0.02
    drifting = x + 0.5 * t / t[-1] * 14.0  # a ramp makes the period non-periodic
    slice_ = wi.extract_period(t, drifting, 1 / F_SW, unit="A")
    assert slice_.closure_residual > 0.05 and slice_.info["closure_flagged"] is True and slice_.values[-1] == slice_.values[0]
    steady = wi.steady_state_metrics(t, x, 1 / F_SW)
    assert steady["cycles"] == 2 and steady["rms_relative_spread"] < 1e-6
    with pytest.raises(InvalidInputError):
        wi.extract_period(t, x, 30e-6)  # longer than the record


def test_flux_linkage_from_square_voltage_matches_faraday():
    t = np.linspace(0.0, 30e-6, 30001)
    v = _square(t, 800.0)
    lam = wi.flux_linkage_from_voltage(t, v, 1 / F_SW)
    assert lam.info["flux_linkage_peak_to_peak_Vs"] == pytest.approx(800.0 / (2 * F_SW), rel=2e-3)  # V / (2 f)
    assert abs(lam.info["net_volt_seconds_per_period_Vs"]) < 1e-9 and lam.unit == "V*s"
    pl = lam.to_piecewise_linear("lambda")
    assert abs(pl.mean()) < 1e-9 * pl.peak_to_peak()
    bps = wi.flux_density_breakpoints(pl, 21, 535e-6)
    peaks = [b for _, b in bps]
    assert max(peaks) == pytest.approx(1.0, abs=1e-9) and min(peaks) == pytest.approx(-1.0, abs=1e-9) and bps[0][1] == bps[-1][1]
    b = wi.flux_density_piecewise(pl, 21, 535e-6)
    assert b.unit == "T" and b.peak() == pytest.approx(800.0 / (4 * F_SW * 21 * 535e-6), rel=2e-3)  # V / (4 f N Ae)
    with pytest.raises(InvalidInputError):
        wi.flux_density_breakpoints(pl, 0, 535e-6)


@pytest.mark.skipif(not REAL_FILE.exists(), reason=f"executed DAB waveform file not present: {REAL_FILE}")
def test_real_dab_waveform_file_reproduces_the_published_metrics():
    pytest.importorskip("h5py", reason="the HDF5 waveform loader needs h5py (aipe-magnetics[hdf5])")
    ws = wi.load_waveforms_hdf5(REAL_FILE, channels=["i_L", "i_s", "v_tp"], window=(200e-6, 240e-6))
    est = wi.detect_period(ws.time, ws.channel("i_L").values, frequency_hint=ws.metadata["spec"]["f_sw"])
    assert est.period == 1e-5 and est.periodicity > 0.999 and abs(est.hint_mismatch) < 1e-4
    i_l = wi.extract_period(ws.time, ws.channel("i_L").values, est.period, unit="A").to_piecewise_linear()
    i_s = wi.extract_period(ws.time, ws.channel("i_s").values, est.period, unit="A").to_piecewise_linear()
    assert i_l.rms() == pytest.approx(13.68, rel=2e-3) and i_l.peak() == pytest.approx(14.3, rel=5e-3)
    assert i_s.rms() == pytest.approx(27.9, rel=3e-3)
    lam = wi.flux_linkage_from_voltage(ws.time, ws.channel("v_tp").values, est.period)
    assert lam.info["flux_linkage_peak_to_peak_Vs"] == pytest.approx(4.0e-3, rel=1e-3)
    assert ws.source["sha256"] == sha256_file(REAL_FILE)
