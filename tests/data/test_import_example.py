"""The executable synthetic data-import example runs offline, validates, and the committed fixture set is consistent."""

from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from aipe_magnetics.common.jsonio import load, sha256_file
from aipe_magnetics.data.hdf5 import WaveformStore
from aipe_magnetics.data.records import MagneticRecord
from aipe_magnetics.data.schema import validate_catalogue, validate_record

pytestmark = pytest.mark.usefixtures("require_h5py")

EXPECTED_FILES = {
    "waveforms-synthetic-source.npz", "waveforms-synthetic.h5", "dataset.aipe-synthetic-waveforms.json",
    "waveform-dataset.aipe-synthetic-flux-density.json", "excitation.aipe-synthetic-sinusoidal.json",
    "excitation.aipe-synthetic-triangular.json",
}


@pytest.fixture(scope="module")
def example_module(repo_root: Path):
    path = repo_root / "examples" / "import_synthetic_waveforms.py"
    assert path.is_file(), path
    spec = importlib.util.spec_from_file_location("aipe_example_import_synthetic_waveforms", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _check_output_set(directory: Path) -> list[dict]:
    assert {p.name for p in directory.iterdir()} >= EXPECTED_FILES
    h5 = directory / "waveforms-synthetic.h5"
    assert h5.stat().st_size < 200_000, "committed fixture must stay small"
    assert WaveformStore().verify_checksums(h5) == []
    records = [load(directory / name) for name in sorted(EXPECTED_FILES) if name.endswith(".json")]
    for record in records:
        assert record["synthetic"] is True
        assert validate_record(record) == [], record["id"]
    assert validate_catalogue(records) == []
    by_kind = {r["kind"] for r in records}
    assert by_kind == {"dataset", "waveform_dataset", "excitation_waveform"}
    waveform = next(r for r in records if r["kind"] == "waveform_dataset")
    artifact = next(a for a in waveform["artifacts"] if a["role"] == "waveform_data")
    assert artifact["sha256"] == sha256_file(h5) and artifact["uri"] == "waveforms-synthetic.h5"
    dataset = next(r for r in records if r["kind"] == "dataset")
    assert dataset["properties"]["checksums"]["waveforms-synthetic.h5"] == sha256_file(h5)
    assert dataset["properties"]["checksums"]["waveforms-synthetic-source.npz"] == sha256_file(directory / "waveforms-synthetic-source.npz")
    props = MagneticRecord.model_validate(waveform).typed_properties()
    assert props.n_sequences == 18 and set(props.conditions) == {"frequency", "temperature"}, "no loss values are claimed"
    assert [a.quantity for a in props.arrays] == ["flux_density"] and props.arrays[0].original_unit == "mT"
    data = WaveformStore().read(h5)
    assert "loss_density" not in data.conditions and set(data.arrays) == {"flux_density"}
    assert data.conditions["temperature"].tolist() == [298.15] * 18
    assert sorted(set(data.conditions["frequency"].tolist())) == [50000.0, 100000.0, 200000.0]
    assert float(abs(data.arrays["flux_density"]).max()) == pytest.approx(0.2)
    shapes = waveform["extensions"]["aipe.synthetic"]["shapes_by_row"]
    assert shapes == ["sinusoidal"] * 9 + ["triangular"] * 9
    return records


def test_example_runs_offline_into_tmp_path(tmp_path: Path, example_module, capsys):
    out = tmp_path / "synthetic"
    with patch.object(socket, "socket", side_effect=AssertionError("network forbidden")):
        assert example_module.main(["--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "validation: 4 records valid" in printed and "checksums verified" in printed
    _check_output_set(out)
    first = sha256_file(out / "waveforms-synthetic.h5")
    assert example_module.main(["--out", str(out)]) == 1, "refuses to overwrite"
    assert "ERROR" in capsys.readouterr().err
    assert example_module.main(["--out", str(out), "--overwrite"]) == 0
    assert sha256_file(out / "waveforms-synthetic.h5") == first, "generation is deterministic"
    small = tmp_path / "small"
    summary = example_module.run(small, n_samples=32, quiet=True)
    assert summary["n_sequences"] == 18 and summary["n_samples"] == 32 and summary["validation_errors"] == 0
    with pytest.raises(ValueError):
        example_module.generate_sequences(4)


def test_committed_fixture_set_is_consistent(repo_root: Path):
    directory = repo_root / "examples" / "synthetic"
    assert directory.is_dir(), "examples/synthetic must be committed (run python examples/import_synthetic_waveforms.py)"
    _check_output_set(directory)


def test_generated_waveforms_follow_the_documented_equations(example_module):
    import numpy as np

    generated = example_module.generate_sequences(64)
    phase = 2 * np.pi * np.arange(64) / 64
    assert np.allclose(generated["flux_density_mT"][0], 50.0 * np.sin(phase))
    tri = generated["flux_density_mT"][9] / 50.0
    assert tri[0] == pytest.approx(0.0) and tri[16] == pytest.approx(1.0) and tri[48] == pytest.approx(-1.0)
    assert np.allclose(np.diff(tri[:16]), np.diff(tri[:16])[0]), "triangle rises linearly"
    assert generated["frequency_kHz"].tolist() == [50.0, 50.0, 50.0, 100.0, 100.0, 100.0, 200.0, 200.0, 200.0] * 2
