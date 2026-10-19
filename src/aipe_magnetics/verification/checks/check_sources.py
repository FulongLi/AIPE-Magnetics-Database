"""Verification checks of the external source adapters (descriptors, cache integrity, licence policy, explicit units).

All checks are L1 (software correctness), deterministic, offline (the cache check injects an in-memory
transport; no socket is opened) and touch only temporary directories. They never read the upstream
checkouts, so they run identically on every machine.
"""

from __future__ import annotations

import hashlib
import io
import tempfile
from pathlib import Path

import numpy as np

from aipe_magnetics.common.errors import DataIntegrityError, InvalidInputError, LicenseRestrictionError, UnitError
from aipe_magnetics.data.hdf5 import h5py_available
from aipe_magnetics.data.sources import (
    ADAPTERS,
    DownloadNotAllowedError,
    LocalCache,
    MagnetChallenge2Adapter,
    MagnetHubAdapter,
    PaderbornMaterialDatabaseAdapter,
    PrincetonMagnetAdapter,
    SourceDescriptor,
    SourceItem,
    TableMeta,
    TableStore,
    descriptors,
)
from aipe_magnetics.data.sources.base import LICENSE_ACTIONS
from aipe_magnetics.data.sources.paderborn_materialdatabase import SOURCE_ID as MDB_SOURCE_ID
from aipe_magnetics.verification.suite import CheckResult, check

CAPABILITY = "aipe.magnetics.data.sources"
#: audited licence facts the descriptors must reproduce
EXPECTED_POLICY = {
    "princeton_magnet": ("MIT", True), "magnet_challenge2": ("NOASSERTION", False),
    "paderborn_materialdatabase": ("GPL-3.0-only", False), "paderborn_magnethub": ("MIT", True),
}
EXPECTED_COMMITS = {
    "princeton_magnet": "8a6892e91ac622e42f8b9182a6dfe9094ceea97a", "magnet_challenge2": "ab4913ca63ebda1f8eab429c64d3e93f96204713",
    "paderborn_materialdatabase": "04fff8e98ae3ddb72809223c42e8c939ed1073bb", "paderborn_magnethub": "1ed635d7bfac3c2aa024471af363beadc05f8932",
}


@check(
    id="sources.descriptors-valid", capability=CAPABILITY, level="L1", kind="schema",
    description="Every registered source adapter carries a valid descriptor: 40-hex pinned commit equal to the audited commit, "
                "https repository URL, SPDX licence (MIT / GPL-3.0-only / NOASSERTION) with the audited redistribution flag, citation, "
                "documented data locations and a consistent record-id token.",
    method="Validate SourceDescriptor models, compare against the audited (licence, redistribution, commit) table, run descriptor.problems(), "
           "and reject a malformed descriptor (short commit, http URL, free-text licence).")
def descriptors_valid() -> CheckResult:
    problems: list[str] = []
    seen = descriptors()
    if sorted(d.source_id for d in seen) != sorted(EXPECTED_POLICY):
        problems.append(f"registered adapters {sorted(ADAPTERS)} differ from the audited set {sorted(EXPECTED_POLICY)}")
    for d in seen:
        problems.extend(d.problems())
        expected = EXPECTED_POLICY.get(d.source_id)
        if expected and (d.license, d.redistribution_permitted) != expected:
            problems.append(f"{d.source_id}: licence policy {(d.license, d.redistribution_permitted)} != audited {expected}")
        if EXPECTED_COMMITS.get(d.source_id) != d.pinned_commit:
            problems.append(f"{d.source_id}: pinned commit {d.pinned_commit} != audited {EXPECTED_COMMITS.get(d.source_id)}")
        if not d.pinned_commit_date:
            problems.append(f"{d.source_id}: no pinned commit date (evidence created_at would be undefined)")
        refs = ADAPTERS[d.source_id].__new__(ADAPTERS[d.source_id]).attribution()  # attribution needs no state
        if not refs or refs[0].version != d.pinned_commit or refs[0].license != d.license:
            problems.append(f"{d.source_id}: attribution() does not carry the pinned commit and licence")
    base = dict(source_id="x", title="x", repository_url="https://example.org/x", pinned_commit="0" * 40, license="MIT",
                redistribution_permitted=True, citation="c", data_locations=["https://example.org"])
    for bad in ({"pinned_commit": "abc123"}, {"repository_url": "http://example.org/x"}, {"license": "free for all"}):
        try:
            SourceDescriptor(**{**base, **bad})
            problems.append(f"malformed descriptor accepted: {bad}")
        except ValueError:
            pass
    return CheckResult(passed=not problems, metrics={"adapters": len(seen), "problems": len(problems)},
                       notes=problems or [f"{len(seen)} descriptors valid and consistent with the audit"])


@check(
    id="sources.cache-tamper-detected", capability=CAPABILITY, level="L1", kind="software",
    description="The local cache never downloads without allow_download=True, verifies SHA-256 after a download (mismatch deletes the file), "
                "records a manifest with url / sha256 / size / licence, serves cached files without re-downloading, and verify() detects "
                "tampering and missing files.",
    method="LocalCache in a temporary directory with an in-memory transport (no network): fetch without permission -> DownloadNotAllowedError; "
           "wrong expected digest -> DataIntegrityError and no file left; correct digest -> cached path; second fetch served from cache "
           "(transport not called); modify the cached bytes -> verify() reports size and sha256 mismatches; delete -> reported missing.")
def cache_tamper_detected() -> CheckResult:
    problems: list[str] = []
    payload = b"synthetic cache payload\n" * 64
    digest = hashlib.sha256(payload).hexdigest()
    calls = {"n": 0}

    def transport(url: str, timeout: float) -> io.BytesIO:
        calls["n"] += 1
        return io.BytesIO(payload)

    url = "https://example.invalid/pinned/file.bin"
    with tempfile.TemporaryDirectory() as tmp:
        cache = LocalCache(Path(tmp) / "cache", transport=transport)
        try:
            cache.fetch(url, digest)
            problems.append("fetch downloaded without allow_download=True")
        except DownloadNotAllowedError:
            pass
        if calls["n"] != 0:
            problems.append("transport was called although downloading was not authorised")
        try:
            cache.fetch(url, "0" * 64, allow_download=True)
            problems.append("sha256 mismatch was accepted")
        except DataIntegrityError:
            pass
        leftovers = [p for p in (Path(tmp) / "cache").rglob("*") if p.is_file() and p.name != "manifest.json"]
        if leftovers:
            problems.append(f"files left behind after a rejected download: {[p.name for p in leftovers]}")
        path = cache.fetch(url, digest, allow_download=True, license="MIT", commit="0" * 40, source_id="check")
        if not path.is_file() or path.read_bytes() != payload:
            problems.append("downloaded file missing or altered")
        entry = cache.lookup(url)
        if entry is None or entry.sha256 != digest or entry.size != len(payload) or entry.license != "MIT" or entry.retrieved_at is not None:
            problems.append(f"manifest entry incomplete or non-deterministic: {entry}")
        before = calls["n"]
        if cache.fetch(url, digest) != path or calls["n"] != before:
            problems.append("cached file was not served from the cache")
        if cache.verify():
            problems.append(f"fresh cache reports problems: {cache.verify()}")
        path.write_bytes(payload[:-1] + b"X")
        report = cache.verify()
        if not any("sha256" in r for r in report):
            problems.append("tampered cached file not detected by verify()")
        try:
            cache.fetch(url, digest)
            problems.append("fetch served a tampered cached file")
        except DataIntegrityError:
            pass
        path.unlink()
        if not any("missing" in r for r in cache.verify()):
            problems.append("missing cached file not detected by verify()")
        try:
            cache.clone_pinned("https://example.invalid/repo", "1" * 40)
            problems.append("clone_pinned cloned without allow_download=True")
        except DownloadNotAllowedError:
            pass
    return CheckResult(passed=not problems, metrics={"transport_calls": calls["n"], "problems": len(problems)},
                       notes=problems or ["download opt-in, digest verification, manifest, cache hits and tamper detection verified offline"])


@check(
    id="sources.license-policy-enforced", capability=CAPABILITY, level="L1", kind="invalid_input_rejection",
    description="Licence policy is enforced per source: MagNet Challenge 2 (NOASSERTION) refuses any download, caching, bundling or "
                "redistribution; the GPL-3.0 materialdatabase allows read and cache but refuses bundle and redistribute; the MIT sources "
                "allow all four actions; unknown actions are rejected.",
    method="Call check_license for every action on every adapter and fetch()/download()/discover(allow_download=True) on the MagNet "
           "Challenge 2 adapter; compare with the audited policy table; the raised errors must be LicenseRestrictionError.")
def license_policy_enforced() -> CheckResult:
    problems: list[str] = []
    expected = {
        "princeton_magnet": {"read": True, "cache": True, "bundle": True, "redistribute": True},
        "paderborn_magnethub": {"read": True, "cache": True, "bundle": True, "redistribute": True},
        "paderborn_materialdatabase": {"read": True, "cache": True, "bundle": False, "redistribute": False},
        "magnet_challenge2": {"read": True, "cache": False, "bundle": False, "redistribute": False},
    }
    with tempfile.TemporaryDirectory() as tmp:
        cache = LocalCache(Path(tmp))
        adapters = {"princeton_magnet": PrincetonMagnetAdapter(cache), "paderborn_magnethub": MagnetHubAdapter(cache),
                    "paderborn_materialdatabase": PaderbornMaterialDatabaseAdapter(cache), "magnet_challenge2": MagnetChallenge2Adapter()}
        for source_id, adapter in adapters.items():
            for action in LICENSE_ACTIONS:
                try:
                    adapter.check_license(action)
                    allowed = True
                except LicenseRestrictionError:
                    allowed = False
                if allowed != expected[source_id][action]:
                    problems.append(f"{source_id}: {action} {'allowed' if allowed else 'refused'}, expected {'allowed' if expected[source_id][action] else 'refused'}")
            try:
                adapter.check_license("sell")
                problems.append(f"{source_id}: unknown action accepted")
            except InvalidInputError:
                pass
        mc2 = adapters["magnet_challenge2"]
        for label, call in (("fetch", lambda: mc2.fetch("x")), ("download", lambda: mc2.download("x")),
                            ("discover(allow_download=True)", lambda: mc2.discover(allow_download=True))):
            try:
                call()
                problems.append(f"magnet_challenge2.{label} did not refuse")
            except LicenseRestrictionError as error:
                if "manual" not in error.message.lower() and "manual_path" not in error.details:
                    problems.append(f"magnet_challenge2.{label}: refusal does not name the manual path")
        mdb = adapters["paderborn_materialdatabase"]
        try:
            mdb.resolve_data_dir(None, allow_download=False)
            problems.append("materialdatabase resolved a data dir without a checkout or permission to clone")
        except DownloadNotAllowedError:
            pass
        if MDB_SOURCE_ID != mdb.descriptor.source_id:
            problems.append("materialdatabase source id mismatch")
    return CheckResult(passed=not problems, metrics={"adapters": 4, "actions": len(LICENSE_ACTIONS), "problems": len(problems)},
                       notes=problems or ["licence policy table enforced; MagNet Challenge 2 downloads refused with the manual path"])


@check(
    id="sources.unit-mapping-explicit", capability=CAPABILITY, level="L1", kind="invalid_input_rejection",
    description="No adapter infers units: MagNet Challenge 2 imports require the column_units mapping and reject wrong dimensions; the "
                "materialdatabase adapter rejects millitesla-valued flux-density columns unless the caller overrides the unit, requires "
                "confirmation of the unit-less p_v column, and the TableStore refuses columns without a unit; conversions keep the original unit.",
    method="Synthetic in-memory CSV files in a temporary directory: import_user_files without column_units -> InvalidInputError, with a "
           "frequency unit for B -> UnitError; a B-H file with b up to 450 -> DataIntegrityError, accepted with unit_overrides={'b': 'mT'} "
           "and stored in tesla with original_unit mT; a p_v file -> InvalidInputError until confirm_units is given; "
           "TableStore.write with a bare array -> InvalidInputError.")
def unit_mapping_explicit() -> CheckResult:
    if not h5py_available():
        return CheckResult(passed=False, notes=["h5py is not installed; the unit-mapping check needs the HDF5 table store and was NOT executed"])
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = 4
        b = np.stack([0.1 * np.sin(2 * np.pi * np.arange(128) / 128) for _ in range(rows)])
        h = 20.0 * b
        for name, array in (("S_1_B.csv", b), ("S_1_H.csv", h), ("S_1_T.csv", np.full((rows, 1), 25.0))):
            np.savetxt(root / name, array, delimiter=",")
        mc2 = MagnetChallenge2Adapter()
        try:
            mc2.load_user_files(root / "S_1_B.csv", root / "S_1_H.csv", root / "S_1_T.csv", material="S", frequency_or_sampling_time=(100e3, "Hz"),
                                column_units=None)
            problems.append("MagNet Challenge 2 import accepted missing column_units")
        except InvalidInputError:
            pass
        try:
            mc2.load_user_files(root / "S_1_B.csv", root / "S_1_H.csv", root / "S_1_T.csv", material="S", frequency_or_sampling_time=(100e3, "Hz"),
                                column_units={"B": "kHz", "H": "A/m", "T": "degC"})
            problems.append("MagNet Challenge 2 import accepted a frequency unit for B")
        except UnitError:
            pass
        extracted = mc2.load_user_files(root / "S_1_B.csv", root / "S_1_H.csv", root / "S_1_T.csv", material="S",
                                        frequency_or_sampling_time=(100e3, "Hz"), column_units={"B": "mT", "H": "A/m", "T": "degC"})
        if extracted.unit_mapping() != {"flux_density": "mT", "field": "A/m", "temperature": "degC", "frequency": "Hz"}:
            problems.append(f"unit mapping not recorded as given: {extracted.unit_mapping()}")

        data_dir = root / "materialdatabase" / "data"
        curve_dir = data_dir / "datasheet_curves" / "SYN"
        curve_dir.mkdir(parents=True)
        (data_dir / "complex_permeability").mkdir()
        (curve_dir / "b_over_h_at_f_T.csv").write_text("h,b,T,f,branch\n10,100,25,10000,r\n20,300,25,10000,r\n30,450,25,10000,f\n", encoding="utf-8")
        (curve_dir / "p_v_over_b_at_f_T.csv").write_text("b,p_v,f,T\n0.05,5000,100000,25\n0.1,30000,100000,25\n", encoding="utf-8")
        mdb = PaderbornMaterialDatabaseAdapter(LocalCache(root / "cache"))
        items = {i.extra["file_type"]: i for i in mdb.discover(root)}
        try:
            mdb.extract(items["b_over_h_at_f_T"])
            problems.append("millitesla-valued B-H column accepted as tesla")
        except DataIntegrityError:
            pass
        bh = mdb.extract(items["b_over_h_at_f_T"], unit_overrides={"b": "mT"})
        if bh.columns["flux_density"][1] != "mT" or not np.allclose(bh.columns["flux_density"][0], [100, 300, 450]):
            problems.append("unit override not applied as given")
        try:
            mdb.extract(items["p_v_over_b_at_f_T"])
            problems.append("ambiguous p_v unit accepted without confirmation")
        except InvalidInputError:
            pass
        pv = mdb.extract(items["p_v_over_b_at_f_T"], confirm_units={"p_v": "kW/m^3"})
        if pv.columns["loss_density"][1] != "kW/m^3":
            problems.append("confirmed unit not recorded")
        store = TableStore()
        meta = TableMeta(source_id="aipe.verification", dataset_version="check")
        try:
            store.write(root / "bare.h5", {"frequency": np.array([1.0, 2.0])}, meta=meta)  # type: ignore[dict-item]
            problems.append("TableStore accepted a column without a unit")
        except InvalidInputError:
            pass
        description = store.write(root / "ok.h5", {"flux_density": (np.array([100.0, 300.0]), "mT"), "temperature": (np.array([25.0, 100.0]), "degC")}, meta=meta)
        data = store.read(root / "ok.h5")
        if data.units != {"flux_density": "T", "temperature": "K"} or data.original_units != {"flux_density": "mT", "temperature": "degC"}:
            problems.append(f"TableStore units not normalised with originals retained: {data.units} {data.original_units}")
        if not np.allclose(data.columns["flux_density"], [0.1, 0.3]) or not np.allclose(data.columns["temperature"], [298.15, 373.15]):
            problems.append("TableStore conversion values wrong")
        if description.n_rows != 2 or store.verify_checksums(root / "ok.h5"):
            problems.append("TableStore description or checksums inconsistent")
        item = SourceItem(source_id="princeton_magnet", item_id="x", kind="file", available=True)
        if item.license != "NOASSERTION":
            problems.append("SourceItem default licence is not NOASSERTION")
    return CheckResult(passed=not problems, metrics={"problems": len(problems)},
                       notes=problems or ["explicit unit mappings enforced; overrides and confirmations recorded; originals retained"])


__all__ = ["descriptors_valid", "cache_tamper_detected", "license_policy_enforced", "unit_mapping_explicit"]
