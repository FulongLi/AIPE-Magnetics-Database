"""Every verification check of the data layer is executed here and must pass (no silent skips)."""

from __future__ import annotations

import time

import pytest

from aipe_magnetics.data.hdf5 import h5py_available
from aipe_magnetics.verification.checks import check_data
from aipe_magnetics.verification.suite import CheckResult, registered_checks, run_check

EXPECTED_IDS = {
    "data.schema-v02-is-valid-draft2020",
    "data.schema-mirror-in-sync",
    "data.v01-examples-migrate-and-validate",
    "data.hdf5-roundtrip-checksums",
    "data.unit-normalisation-reproducible",
    "data.catalog-loads-and-cross-references",
    "data.importers-provenance-and-no-overwrite",
}


def _data_specs():
    return {spec_id: spec for spec_id, spec in registered_checks().items() if spec.module == check_data.__name__}


def test_all_data_checks_are_registered_with_capability_ids():
    specs = _data_specs()
    assert set(specs) == EXPECTED_IDS
    for spec in specs.values():
        assert spec.capability.startswith("aipe.magnetics.data."), spec.id
        assert spec.level in ("L1", "L2") and spec.description and spec.method
    assert specs["data.unit-normalisation-reproducible"].level == "L2"
    assert specs["data.unit-normalisation-reproducible"].reference is not None


@pytest.mark.parametrize("check_id", sorted(EXPECTED_IDS))
def test_each_data_check_passes(check_id):
    if check_id == "data.hdf5-roundtrip-checksums" and not h5py_available():
        pytest.fail("h5py is not installed: the HDF5 round-trip check cannot produce evidence (install aipe-magnetics[hdf5])")
    spec = _data_specs()[check_id]
    start = time.perf_counter()
    result = spec.func()
    duration = time.perf_counter() - start
    assert isinstance(result, CheckResult)
    assert result.passed, f"{check_id} failed: notes={result.notes} metrics={result.metrics}"
    assert duration < 2.0, f"{check_id} took {duration:.2f} s (checks must stay fast)"


def test_outcomes_become_evidence_records():
    for check_id, spec in sorted(_data_specs().items()):
        outcome = run_check(spec)
        assert outcome.status == "pass", (check_id, outcome.error, outcome.result and outcome.result.notes)
        record = outcome.to_record()
        assert record.id == f"check:{check_id}" and record.status == "pass" and record.capability == spec.capability


def test_hdf5_check_fails_explicitly_without_h5py(monkeypatch):
    monkeypatch.setattr(check_data, "h5py_available", lambda: False)
    result = check_data.hdf5_roundtrip_checksums()
    assert result.passed is False
    assert any("h5py is not installed" in note for note in result.notes)


def test_checks_are_deterministic():
    first = check_data.unit_normalisation_reproducible()
    second = check_data.unit_normalisation_reproducible()
    assert first == second and first.metrics["max_rel_error"] == 0.0
    assert check_data.v01_examples_migrate_and_validate().metrics == check_data.v01_examples_migrate_and_validate().metrics
