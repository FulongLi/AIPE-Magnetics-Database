"""WaveformStore: layout, SI normalisation with originals, NaN missing values, checksums, determinism, strict errors."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from aipe_magnetics.common.errors import DataIntegrityError, InvalidInputError, UnitError, UnsupportedOperationError
from aipe_magnetics.common.jsonio import sha256_file
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.units import convert
from aipe_magnetics.data import hdf5 as hdf5_module
from aipe_magnetics.data.hdf5 import (
    CONDITION_QUANTITIES,
    LAYOUT_VERSION,
    SEQUENCE_QUANTITIES,
    PreparedWaveforms,
    WaveformMeta,
    WaveformStore,
    array_sha256,
    canonicalize_nan,
    convert_array,
    missing_mask,
)
from aipe_magnetics.data.records import MagneticRecord

pytestmark = pytest.mark.usefixtures("require_h5py")

N_SEQ, N_SAMP = 6, 128


@pytest.fixture
def sample():
    t = np.arange(N_SAMP) / N_SAMP
    b_mt = np.stack([100.0 * (k + 1) * np.sin(2 * np.pi * t) for k in range(N_SEQ)])
    b_mt[2, 5] = np.nan
    return {
        "arrays": {"flux_density": (b_mt, "mT")},
        "conditions": {"frequency": (np.array([50.0, 100.0, 200.0, 50.0, 100.0, 200.0]), "kHz"),
                       "temperature": (np.array([25.0, 25.0, 25.0, 100.0, 100.0, np.nan]), "degC")},
        "meta": WaveformMeta(source_id="aipe.test", dataset_version="t1", license="NOASSERTION", description="test"),
    }


def test_roundtrip_units_missing_values_and_metadata(tmp_path: Path, sample):
    store = WaveformStore()
    path = tmp_path / "wf.h5"
    props = store.write(path, sample["arrays"], sample["conditions"], meta=sample["meta"])
    assert props.layout_version == LAYOUT_VERSION and props.format == "hdf5"
    assert props.n_sequences == N_SEQ and props.samples_per_sequence == N_SAMP
    assert [(a.path, a.unit, a.original_unit, a.shape, a.missing_value) for a in props.arrays] == [
        ("/sequences/flux_density", "T", "mT", [N_SEQ, N_SAMP], "NaN")]
    assert props.conditions == {"frequency": "/conditions/frequency", "temperature": "/conditions/temperature"}
    data = store.read(path)
    b_mt = sample["arrays"]["flux_density"][0]
    expected = np.array([[convert(v, "mT", "T") if not math.isnan(v) else math.nan for v in row] for row in b_mt])
    assert np.array_equal(data.arrays["flux_density"], expected, equal_nan=True), "bit-identical to the scalar registry"
    assert data.array_units == {"flux_density": "T"} and data.array_original_units == {"flux_density": "mT"}
    assert data.conditions["frequency"].tolist() == [50000.0, 100000.0, 200000.0, 50000.0, 100000.0, 200000.0]
    assert data.conditions["temperature"][:5].tolist() == [298.15, 298.15, 298.15, 373.15, 373.15]
    assert math.isnan(data.conditions["temperature"][5]) and data.mask("temperature").tolist()[-1] is True
    assert int(missing_mask(data.arrays["flux_density"]).sum()) == 1 and data.mask("flux_density")[2, 5]
    assert data.condition_original_units == {"frequency": "kHz", "temperature": "degC"}
    assert data.attrs["layout_version"] == LAYOUT_VERSION and data.attrs["schema_version"] == "0.2.0"
    assert data.attrs["created_by_tool"] == "aipe.magnetics" and data.attrs["source_id"] == "aipe.test"
    assert data.attrs["timebase"] == "uniform_per_period" and data.attrs["license"] == "NOASSERTION"
    assert "sign_convention" in data.attrs and "coordinate_definition" in data.attrs
    assert data.n_sequences == N_SEQ and data.n_samples == N_SAMP
    assert store.verify_checksums(path) == []
    assert store.describe(path) == props
    with pytest.raises(InvalidInputError):
        data.mask("nothing")


def test_checksums_are_canonical_and_detect_tampering(tmp_path: Path, sample):
    import h5py

    store = WaveformStore()
    path = tmp_path / "wf.h5"
    store.write(path, sample["arrays"], sample["conditions"], meta=sample["meta"])
    data = store.read(path)
    assert data.checksums["sequences/flux_density"] == array_sha256(data.arrays["flux_density"])
    assert set(data.checksums) == {"sequences/flux_density", "conditions/frequency", "conditions/temperature"}
    with h5py.File(path, "r+") as handle:
        handle["conditions/frequency"][1] = 999.0
    mismatches = store.verify_checksums(path)
    assert len(mismatches) == 1 and mismatches[0].startswith("conditions/frequency: expected")
    with pytest.raises(DataIntegrityError):
        store.read(path)
    assert store.read(path, verify=False).conditions["frequency"][1] == 999.0
    with h5py.File(path, "r+") as handle:
        del handle["checksums"].attrs["conditions/temperature"]
    assert any("no checksum recorded" in m for m in store.verify_checksums(path))


def test_identical_inputs_give_identical_files(tmp_path: Path, sample):
    store = WaveformStore()
    store.write(tmp_path / "a.h5", sample["arrays"], sample["conditions"], meta=sample["meta"])
    store.write(tmp_path / "b.h5", sample["arrays"], sample["conditions"], meta=sample["meta"])
    assert sha256_file(tmp_path / "a.h5") == sha256_file(tmp_path / "b.h5")
    artifact = store.artifact(tmp_path / "a.h5", uri="data/a.h5")
    assert artifact.sha256 == sha256_file(tmp_path / "a.h5") and artifact.role == "waveform_data"
    assert artifact.media_type == "application/x-hdf5" and artifact.size_bytes == (tmp_path / "a.h5").stat().st_size
    with pytest.raises(FileExistsError):
        store.write(tmp_path / "a.h5", sample["arrays"], sample["conditions"], meta=sample["meta"])
    store.write(tmp_path / "a.h5", sample["arrays"], sample["conditions"], meta=sample["meta"], overwrite=True)


def test_properties_make_a_valid_waveform_dataset_record(tmp_path: Path, sample, evidence_assumption):
    store = WaveformStore()
    props = store.write(tmp_path / "wf.h5", sample["arrays"], sample["conditions"], meta=sample["meta"])
    record = MagneticRecord(id="waveform-dataset.test", kind="waveform_dataset", name="test", synthetic=True,
                            properties=props.model_dump(mode="json", exclude_none=True), evidence_refs=[evidence_assumption["id"]],
                            evidence=[evidence_assumption], artifacts=[store.artifact(tmp_path / "wf.h5")])
    assert record.typed_properties().n_sequences == N_SEQ


@pytest.mark.parametrize("label, arrays, conditions, meta_kwargs, error", [
    ("shape mismatch", {"flux_density": ("B", "mT"), "field": ("B[:, :64]", "A/m")}, None, {}, InvalidInputError),
    ("wrong dimension", {"flux_density": ("B", "A/m")}, None, {}, UnitError),
    ("unknown unit", {"flux_density": ("B", "tesla")}, None, {}, UnitError),
    ("infinite", {"flux_density": ("Binf", "mT")}, None, {}, InvalidInputError),
    ("one dimensional", {"flux_density": ("B[0]", "mT")}, None, {}, InvalidInputError),
    ("too few samples", {"flux_density": ("B[:, :1]", "mT")}, None, {}, InvalidInputError),
    ("boolean dtype", {"flux_density": ("B > 0", "mT")}, None, {}, InvalidInputError),
    ("unknown quantity", {"flux_density": ("B", "mT"), "humidity": ("B", "1")}, None, {}, InvalidInputError),
    ("time only", {"time": ("T", "s")}, None, {"timebase": "explicit_time_array"}, InvalidInputError),
    ("no unit tuple", {"flux_density": "B"}, None, {}, InvalidInputError),
    ("empty arrays", {}, None, {}, InvalidInputError),
    ("condition length", {"flux_density": ("B", "mT")}, {"frequency": ("f[:3]", "kHz")}, {}, InvalidInputError),
    ("negative frequency", {"flux_density": ("B", "mT")}, {"frequency": ("-f", "kHz")}, {}, InvalidInputError),
    ("zero temperature", {"flux_density": ("B", "mT")}, {"frequency": ("f", "kHz"), "temperature": ("0*f", "K")}, {}, InvalidInputError),
    ("negative loss", {"flux_density": ("B", "mT")}, {"frequency": ("f", "kHz"), "loss_density": ("-f", "W/m^3")}, {}, InvalidInputError),
    ("duty out of range", {"flux_density": ("B", "mT")}, {"frequency": ("f", "kHz"), "duty": ("f", "1")}, {}, InvalidInputError),
    ("unknown condition", {"flux_density": ("B", "mT")}, {"frequency": ("f", "kHz"), "humidity": ("f", "1")}, {}, InvalidInputError),
    ("condition unit", {"flux_density": ("B", "mT")}, {"frequency": ("f", "mT")}, {}, UnitError),
    ("no timebase info", {"flux_density": ("B", "mT")}, None, {}, InvalidInputError),
    ("time without explicit timebase", {"flux_density": ("B", "mT"), "time": ("T", "s")}, {"frequency": ("f", "kHz")}, {}, InvalidInputError),
    ("explicit timebase without time", {"flux_density": ("B", "mT")}, None, {"timebase": "explicit_time_array"}, InvalidInputError),
    ("absolute without rate", {"flux_density": ("B", "mT")}, None, {"timebase": "uniform_absolute"}, InvalidInputError),
    ("non-monotonic time", {"flux_density": ("B", "mT"), "time": ("Tbad", "s")}, None, {"timebase": "explicit_time_array"}, InvalidInputError),
    ("nan in time", {"flux_density": ("B", "mT"), "time": ("Tnan", "s")}, None, {"timebase": "explicit_time_array"}, InvalidInputError),
])
def test_invalid_inputs_are_rejected_before_writing(tmp_path: Path, label, arrays, conditions, meta_kwargs, error):
    t = np.arange(N_SAMP) / N_SAMP
    env = {
        "B": np.stack([100.0 * (k + 1) * np.sin(2 * np.pi * t) for k in range(N_SEQ)]),
        "f": np.array([50.0, 100.0, 200.0, 50.0, 100.0, 200.0]),
        "T": np.tile(t, (N_SEQ, 1)),
    }
    env["Binf"] = np.where(np.isnan(env["B"]), 0.0, env["B"]).copy()
    env["Binf"][0, 0] = np.inf
    env["Tbad"] = env["T"].copy()
    env["Tbad"][0, 3] = env["Tbad"][0, 2]
    env["Tnan"] = env["T"].copy()
    env["Tnan"][1, 4] = np.nan

    def resolve(spec):
        if isinstance(spec, str):
            return spec  # malformed on purpose
        return (eval(spec[0], {"np": np}, env), spec[1])  # noqa: S307 - test-local expressions

    arrays = {k: resolve(v) for k, v in arrays.items()}
    conditions = None if conditions is None else {k: resolve(v) for k, v in conditions.items()}
    meta = WaveformMeta(source_id="aipe.test", dataset_version="t1", **meta_kwargs)
    path = tmp_path / "never.h5"
    with pytest.raises(error):
        WaveformStore().write(path, arrays, conditions, meta=meta)
    assert not path.exists(), f"{label}: nothing may be written when inputs are rejected"


def test_explicit_time_array_and_uniform_absolute(tmp_path: Path):
    t = np.arange(N_SAMP) / N_SAMP
    b = np.stack([np.sin(2 * np.pi * t) for _ in range(2)])
    store = WaveformStore()
    meta = WaveformMeta(source_id="aipe.test", dataset_version="t1", timebase="explicit_time_array")
    props = store.write(tmp_path / "t.h5", {"flux_density": (b, "T"), "time": (np.tile(t, (2, 1)) * 1e3, "ms")}, None, meta=meta)
    assert [(a.quantity, a.unit, a.original_unit) for a in props.arrays] == [("flux_density", "T", "T"), ("time", "s", "ms")]
    data = store.read(tmp_path / "t.h5")
    assert np.array_equal(data.arrays["time"], convert_array(np.tile(t, (2, 1)) * 1e3, "ms", "s"))
    meta = WaveformMeta(source_id="aipe.test", dataset_version="t1", timebase="uniform_absolute", sampling_rate=Quantity.si(10, "MHz"))
    props = store.write(tmp_path / "u.h5", {"current": (b, "mA")}, None, meta=meta)
    assert props.sampling_rate is not None and props.sampling_rate.unit == "Hz" and props.sampling_rate.value == 1e7
    assert props.sampling_rate.original is not None and props.sampling_rate.original.unit == "MHz"
    described = store.describe(tmp_path / "u.h5")
    assert described == props, "describe() reproduces the write result, original sampling-rate unit included"
    assert described.sampling_rate == props.sampling_rate and described.sampling_rate.original.value == 10.0
    attrs = store.read(tmp_path / "u.h5").attrs
    assert attrs["sampling_rate_hz"] == 1e7 and attrs["sampling_rate_original_unit"] == "MHz" and attrs["sampling_rate_original_value"] == 10.0
    plain = WaveformMeta(source_id="aipe.test", dataset_version="t1", timebase="uniform_absolute", sampling_rate=Quantity(value=2e6, unit="Hz"))
    props = store.write(tmp_path / "v.h5", {"current": (b, "mA")}, None, meta=plain)
    assert store.describe(tmp_path / "v.h5") == props and "sampling_rate_original_unit" not in store.read(tmp_path / "v.h5").attrs


def test_conditions_accept_column_vectors_and_all_quantities(tmp_path: Path):
    t = np.arange(16) / 16
    b = np.stack([np.sin(2 * np.pi * t) for _ in range(3)])
    conditions = {name: (np.array([[1.0], [2.0], [np.nan]]) * (0.25 if name == "duty" else 1.0), unit)
                  for name, unit in CONDITION_QUANTITIES.items()}
    conditions["dc_bias_field"] = (np.array([-10.0, 0.0, 10.0]), "Oe")
    # uniform_absolute: the frequency condition is an excitation condition, not the timebase, so a NaN entry is allowed
    meta = WaveformMeta(source_id="aipe.test", dataset_version="t1", timebase="uniform_absolute", sampling_rate=Quantity(value=16.0, unit="Hz"))
    props = WaveformStore().write(tmp_path / "c.h5", {q: (b, u) for q, u in SEQUENCE_QUANTITIES.items() if q != "time"}, conditions, meta=meta)
    assert set(props.conditions) == set(CONDITION_QUANTITIES)
    data = WaveformStore().read(tmp_path / "c.h5")
    assert data.conditions["dc_bias_field"][2] == convert(10.0, "Oe", "A/m")
    assert set(data.arrays) == set(SEQUENCE_QUANTITIES) - {"time"}


def test_meta_validation():
    with pytest.raises(ValueError):
        WaveformMeta(source_id="aipe.test", dataset_version="t1", sampling_rate=Quantity(value=1.0, unit="T"))
    with pytest.raises(ValueError):
        WaveformMeta(source_id="aipe.test", dataset_version="t1", sampling_rate=Quantity(value=-1.0, unit="Hz"))
    with pytest.raises(ValueError, match="uncertainty"):
        WaveformMeta(source_id="aipe.test", dataset_version="t1",
                     sampling_rate=Quantity(value=1.0, unit="Hz", uncertainty={"kind": "standard", "value": 0.1}))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        WaveformMeta(source_id="", dataset_version="t1")
    with pytest.raises(InvalidInputError):
        WaveformStore(compression_level=12)
    with pytest.raises(InvalidInputError):
        WaveformStore(compression_level=True)  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError, match="compression must be one of"):
        WaveformStore(compression="zip")
    with pytest.raises(InvalidInputError):
        WaveformStore(max_chunk_bytes=0)
    with pytest.raises(InvalidInputError):
        WaveformStore().write("x.h5", {"flux_density": (np.zeros((2, 4)), "T")}, None, meta={"source_id": "x"})  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError):
        WaveformStore().prepare({"flux_density": (np.zeros((2, 4)), "T")}, None, meta={"source_id": "x"})  # type: ignore[arg-type]


def test_compression_options_all_produce_valid_files(tmp_path: Path, sample):
    for compression in ("gzip", "lzf", None):
        store = WaveformStore(compression=compression)
        path = tmp_path / f"{compression}.h5"
        props = store.write(path, sample["arrays"], sample["conditions"], meta=sample["meta"])
        assert WaveformStore().verify_checksums(path) == [] and WaveformStore().describe(path) == props


def test_overflow_during_unit_conversion_is_rejected_not_stored(tmp_path: Path):
    import warnings

    meta = WaveformMeta(source_id="aipe.test", dataset_version="t1")
    freq = (np.array([1e3, 2e3, 3e3]), "Hz")
    big = np.zeros((3, 8))
    big[0, 0] = 1e306
    with pytest.raises(InvalidInputError, match="overflow float64 when converted from 'kA/m' to 'A/m'"):
        WaveformStore().write(tmp_path / "x.h5", {"field": (big, "kA/m")}, {"frequency": freq}, meta=meta)
    with pytest.raises(InvalidInputError, match="dc_bias_field: 1 value\\(s\\) overflow"):
        WaveformStore().write(tmp_path / "x.h5", {"field": (np.zeros((3, 8)), "A/m")},
                              {"dc_bias_field": (np.array([1e306, 1.0, 1.0]), "kA/m"), "frequency": freq}, meta=meta)
    assert not (tmp_path / "x.h5").exists()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the helper reports the overflow as inf without a numpy RuntimeWarning
        assert np.isinf(convert_array([1e306], "kA/m", "A/m")).all()


def test_integers_beyond_2_53_and_ragged_inputs_are_rejected(tmp_path: Path):
    meta = WaveformMeta(source_id="aipe.test", dataset_version="t1", timebase="uniform_absolute", sampling_rate=Quantity(value=1.0, unit="Hz"))
    store = WaveformStore()
    big = np.array([[2 ** 53 + 1, 2 ** 53 + 3], [1, 2]], dtype=np.int64)
    with pytest.raises(InvalidInputError, match=r"exceed 2\*\*53"):
        store.write(tmp_path / "i.h5", {"current": (big, "A")}, None, meta=meta)
    exact = np.array([[2 ** 53, -(2 ** 53)], [1, 2]], dtype=np.int64)
    store.write(tmp_path / "i.h5", {"current": (exact, "A")}, None, meta=meta)
    assert store.read(tmp_path / "i.h5").arrays["current"][0].tolist() == [float(2 ** 53), -float(2 ** 53)]
    with pytest.raises(InvalidInputError, match="rectangular numeric array"):
        store.write(tmp_path / "r.h5", {"flux_density": ([[1.0, 2.0, 3.0], [1.0, 2.0]], "T")}, None, meta=meta)
    with pytest.raises(InvalidInputError, match="numeric array required"):
        store.write(tmp_path / "r.h5", {"flux_density": ([[1.0, "x"], [1.0, 2.0]], "T")}, None, meta=meta)
    assert not (tmp_path / "r.h5").exists()


def test_timebase_consistency_rules(tmp_path: Path):
    b = np.stack([np.sin(2 * np.pi * np.arange(64) / 64) for _ in range(3)])
    store = WaveformStore()
    per_period = WaveformMeta(source_id="aipe.test", dataset_version="t1")
    with pytest.raises(InvalidInputError, match="only timebase information"):
        store.write(tmp_path / "n.h5", {"flux_density": (b, "T")}, {"frequency": (np.array([1e3, np.nan, 3e3]), "Hz")}, meta=per_period)
    both = WaveformMeta(source_id="aipe.test", dataset_version="t1", sampling_rate=Quantity(value=64e3, unit="Hz"))
    with pytest.raises(InvalidInputError, match=r"frequency \* n_samples == sampling_rate"):
        store.write(tmp_path / "n.h5", {"flux_density": (b, "T")}, {"frequency": (np.array([1e3, 2e3, 3e3]), "Hz")}, meta=both)
    assert not (tmp_path / "n.h5").exists()
    props = store.write(tmp_path / "ok.h5", {"flux_density": (b, "T")}, {"frequency": (np.array([1e3, 1e3, np.nan]), "Hz")}, meta=both)
    assert props.sampling_rate.value == 64e3, "1 kHz * 64 samples == 64 kHz is consistent; a missing row frequency is fine when fs is given"
    absolute = WaveformMeta(source_id="aipe.test", dataset_version="t1", timebase="uniform_absolute", sampling_rate=Quantity(value=1.0, unit="Hz"))
    store.write(tmp_path / "abs.h5", {"flux_density": (b, "T")}, {"frequency": (np.array([1e3, 2e3, np.nan]), "Hz")}, meta=absolute)
    assert store.describe(tmp_path / "abs.h5").timebase == "uniform_absolute", "no f/fs coupling in absolute time"


def test_malformed_layout_files_raise_data_integrity_errors(tmp_path: Path):
    import h5py

    store = WaveformStore()

    def make(name, builder):
        path = tmp_path / name
        with h5py.File(path, "w") as handle:
            handle.attrs["layout_version"] = LAYOUT_VERSION
            builder(handle)
        return path

    def no_unit(h):
        h.create_dataset("sequences/flux_density", data=np.zeros((2, 4)))

    def one_dimensional(h):
        h.create_dataset("sequences/flux_density", data=np.zeros(4)).attrs["unit"] = "T"

    def wrong_condition_length(h):
        h.create_dataset("sequences/flux_density", data=np.zeros((2, 4))).attrs["unit"] = "T"
        h.create_dataset("conditions/frequency", data=np.zeros(3)).attrs["unit"] = "Hz"

    def sequences_not_a_group(h):
        h.create_dataset("sequences", data=np.zeros((2, 4)))

    cases = [(make("a.h5", lambda h: None), "sequences"), (make("b.h5", no_unit), "unit"), (make("c.h5", one_dimensional), "2-D"),
             (make("d.h5", wrong_condition_length), "entries"), (make("e.h5", sequences_not_a_group), "sequences")]
    for path, fragment in cases:
        with pytest.raises(DataIntegrityError, match=fragment):
            store.read(path, verify=False)
        with pytest.raises(DataIntegrityError, match=fragment):
            store.describe(path)
        with pytest.raises(DataIntegrityError):
            store.verify_checksums(path)


def test_prepare_writes_nothing_and_write_prepared_reuses_it(tmp_path: Path, sample):
    store = WaveformStore()
    prepared = store.prepare(sample["arrays"], sample["conditions"], meta=sample["meta"])
    assert isinstance(prepared, PreparedWaveforms) and prepared.n_sequences == N_SEQ and prepared.n_samples == N_SAMP
    assert list(tmp_path.iterdir()) == [], "prepare() touches no files"
    props = store.write(tmp_path / "w.h5", sample["arrays"], sample["conditions"], meta=sample["meta"])
    assert prepared.properties == props
    assert store.write_prepared(tmp_path / "p.h5", prepared) == props
    assert sha256_file(tmp_path / "w.h5") == sha256_file(tmp_path / "p.h5")
    with pytest.raises(FileExistsError):
        store.write_prepared(tmp_path / "p.h5", prepared)
    with pytest.raises(InvalidInputError):
        store.write_prepared(tmp_path / "q.h5", {"not": "prepared"})  # type: ignore[arg-type]
    assert not (tmp_path / "q.h5").exists()


def test_reading_foreign_or_missing_files_fails_loudly(tmp_path: Path):
    import h5py

    foreign = tmp_path / "foreign.h5"
    with h5py.File(foreign, "w") as handle:
        handle.create_dataset("B_seq_1", data=np.zeros((3, 8)))
    with pytest.raises(DataIntegrityError):
        WaveformStore().read(foreign)
    with pytest.raises(DataIntegrityError):
        WaveformStore().describe(foreign)
    with pytest.raises(DataIntegrityError):
        WaveformStore().read(tmp_path / "missing.h5")
    with pytest.raises(DataIntegrityError):
        WaveformStore().artifact(tmp_path / "missing.h5")


def test_array_helpers_match_scalar_registry():
    values = np.array([0.0, 25.0, 100.0, np.nan, -40.0])
    out = convert_array(values, "degC", "K")
    assert out[:3].tolist() == [convert(0.0, "degC", "K"), convert(25.0, "degC", "K"), convert(100.0, "degC", "K")]
    assert math.isnan(out[3]) and out[4] == convert(-40.0, "degC", "K")
    assert convert_array([[100.0, 200.0]], "mT", "T").tolist() == [[0.1, 0.2]]
    assert convert_array([1.0], "Oe", "A/m")[0] == 1000.0 / (4.0 * math.pi)
    assert convert_array([180.0], "deg", "rad")[0] == pytest.approx(math.pi)
    with pytest.raises(UnitError):
        convert_array([1.0], "mT", "A/m")
    with pytest.raises(UnitError):
        convert_array([1.0], "deg", "1")
    assert array_sha256(np.array([np.nan, 1.0])) == array_sha256(np.array([-np.nan, 1.0])), "NaN payload canonicalised"
    assert array_sha256(np.array([1.0, 2.0])) != array_sha256(np.array([2.0, 1.0]))
    assert array_sha256(np.array([1, 2])) == array_sha256(np.array([1.0, 2.0]))
    canonical = canonicalize_nan(np.array([np.nan, 1.0]))
    assert np.isnan(canonical[0]) and canonical[1] == 1.0
    assert missing_mask([1.0, float("nan")]).tolist() == [False, True]


def test_missing_h5py_raises_unsupported_operation(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(hdf5_module, "h5py", None)
    assert hdf5_module.h5py_available() is False
    store = WaveformStore()
    meta = WaveformMeta(source_id="aipe.test", dataset_version="t1", timebase="uniform_absolute", sampling_rate=Quantity(value=1.0, unit="Hz"))
    with pytest.raises(UnsupportedOperationError, match=r"aipe-magnetics\[hdf5\]"):
        store.write(tmp_path / "x.h5", {"flux_density": (np.zeros((2, 4)), "T")}, None, meta=meta)
    with pytest.raises(UnsupportedOperationError):
        store.read(tmp_path / "x.h5")
    with pytest.raises(UnsupportedOperationError):
        store.describe(tmp_path / "x.h5")
    assert convert_array([1.0], "mT", "T").tolist() == [0.001], "array helpers work without h5py"
