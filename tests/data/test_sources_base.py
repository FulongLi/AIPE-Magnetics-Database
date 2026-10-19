"""Source-adapter contract: descriptors, TableStore layout, local cache (offline), record helpers, generic validation."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

from aipe_magnetics.common.errors import DataIntegrityError, InvalidInputError, LicenseRestrictionError, UnitError
from aipe_magnetics.common.jsonio import sha256_file
from aipe_magnetics.data.sources import (
    CACHE_ENV_VAR,
    DEFAULT_CACHE_DIRNAME,
    ADAPTERS,
    DownloadNotAllowedError,
    ExtractedData,
    LocalCache,
    SourceDescriptor,
    SourceItem,
    TableMeta,
    TableStore,
    descriptors,
    get_adapter,
)
from aipe_magnetics.data.sources.base import (
    TABLE_LAYOUT_VERSION,
    checksum_problems,
    dataset_token,
    finite_problems,
    monotone_time_problems,
    range_problems,
    slug,
    write_records,
)
from aipe_magnetics.data.sources.cache import MANIFEST_VERSION, urllib_transport
from aipe_magnetics.data.sources.princeton_magnet import PrincetonMagnetAdapter

PAYLOAD = b"synthetic payload for the cache tests\n" * 100
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
URL = "https://example.invalid/data/file.bin"


def fake_transport(calls: dict | None = None):
    def transport(url: str, timeout: float) -> io.BytesIO:
        if calls is not None:
            calls["n"] = calls.get("n", 0) + 1
            calls["timeout"] = timeout
        return io.BytesIO(PAYLOAD)

    return transport


# -- descriptors ----------------------------------------------------------------------------------------
def test_registered_descriptors_are_valid_and_pinned():
    seen = descriptors()
    assert sorted(d.source_id for d in seen) == ["magnet_challenge2", "paderborn_magnethub", "paderborn_materialdatabase", "princeton_magnet"]
    for d in seen:
        assert len(d.pinned_commit) == 40 and int(d.pinned_commit, 16) >= 0
        assert d.repository_url.startswith("https://github.com/")
        assert d.license in ("MIT", "GPL-3.0-only", "NOASSERTION")
        assert d.problems() == []
        assert d.pinned_commit_date and d.data_locations
    by_id = {d.source_id: d for d in seen}
    assert by_id["princeton_magnet"].redistribution_permitted is True
    assert by_id["paderborn_magnethub"].redistribution_permitted is True
    assert by_id["magnet_challenge2"].redistribution_permitted is False and by_id["magnet_challenge2"].license == "NOASSERTION"
    assert by_id["paderborn_materialdatabase"].redistribution_permitted is False and by_id["paderborn_materialdatabase"].license == "GPL-3.0-only"
    assert get_adapter("princeton_magnet").descriptor is by_id["princeton_magnet"]
    with pytest.raises(LookupError):
        get_adapter("does_not_exist")
    assert set(ADAPTERS) == set(by_id)


@pytest.mark.parametrize("bad", [{"pinned_commit": "abc"}, {"pinned_commit": "G" * 40}, {"repository_url": "http://x.org"},
                                 {"license": "free for all"}, {"source_id": "Bad-Id"}])
def test_descriptor_rejects_malformed_fields(bad):
    base = dict(source_id="x", title="x", repository_url="https://example.org/x", pinned_commit="0" * 40, license="MIT",
                redistribution_permitted=True, citation="c", data_locations=["https://example.org"])
    with pytest.raises(ValueError):
        SourceDescriptor(**{**base, **bad})
    d = SourceDescriptor(**{**base, "license": "NOASSERTION", "redistribution_permitted": True, "data_locations": [], "citation": None})
    assert any("cannot be permitted" in p for p in d.problems()) and any("no data locations" in p for p in d.problems())


def test_attribution_and_license_actions():
    adapter = PrincetonMagnetAdapter(LocalCache(Path("unused")))
    refs = adapter.attribution()
    assert refs[0].kind == "repository" and refs[0].version == adapter.descriptor.pinned_commit and refs[0].license == "MIT"
    for action in ("read", "cache", "bundle", "redistribute"):
        adapter.check_license(action)
    with pytest.raises(InvalidInputError):
        adapter.check_license("sell")
    mc2 = get_adapter("magnet_challenge2")
    with pytest.raises(LicenseRestrictionError):
        mc2.check_license("cache")


# -- helpers --------------------------------------------------------------------------------------------
def test_slug_dataset_token_and_problem_helpers():
    assert slug("R34.0X20.5X12.5") == "r34-0x20-5x12-5" and slug("3C90") == "3c90" and slug("ML95S") == "ml95s"
    with pytest.raises(InvalidInputError):
        slug("---")
    assert dataset_token("dataset.my-import") == "my-import"
    with pytest.raises(InvalidInputError):
        dataset_token("material.x")
    assert finite_problems("x", [1.0, np.inf]) == ["x: 1 infinite value(s)"]
    assert finite_problems("x", [1.0, np.nan]) and finite_problems("x", [1.0, np.nan], allow_nan=True) == []
    assert range_problems("f", [1.0, -1.0, np.nan], "Hz", minimum=0.0, strict_min=True)[0].startswith("f: 1 value(s) <= 0.0 Hz")
    assert range_problems("b", [0.1, 3.0], "T", maximum=2.0) and range_problems("b", [0.1], "T", maximum=2.0) == []
    assert monotone_time_problems(np.array([[0.0, 1.0, 1.0]])) and monotone_time_problems(np.array([[0.0, 1.0, 2.0]])) == []


def test_checksum_problems_detects_modified_source(tmp_path: Path):
    source = tmp_path / "s.csv"
    source.write_text("a\n", encoding="utf-8")
    digest = sha256_file(source)
    assert checksum_problems({"s.csv": digest}, {"s.csv": str(source)}) == []
    source.write_text("b\n", encoding="utf-8")
    assert checksum_problems({"s.csv": digest}, {"s.csv": str(source)})[0].startswith("s.csv: source bytes changed")
    assert checksum_problems({"gone.csv": digest}, {"gone.csv": str(tmp_path / "gone.csv")}) == []


# -- TableStore -----------------------------------------------------------------------------------------
@pytest.mark.usefixtures("require_h5py")
def test_table_store_roundtrip_units_checksums_and_tamper(tmp_path: Path):
    import h5py

    store = TableStore()
    meta = TableMeta(source_id="aipe.test", dataset_version="t", license="NOASSERTION", description="synthetic table", material_ref="material.x")
    columns = {"frequency": (np.array([50.0, 100.0, 200.0]), "kHz"), "temperature": (np.array([25.0, np.nan, 100.0]), "degC"),
               "loss_density": (np.array([[10.0], [20.0], [30.0]]), "kW/m^3")}
    categorical = {"excitation": (np.array([0, 1, -1]), ["sinusoidal", "triangular"])}
    description = store.write(tmp_path / "t.h5", columns, meta=meta, categorical=categorical, source_columns={"frequency": "Frequency"})
    assert description.layout_version == TABLE_LAYOUT_VERSION and description.n_rows == 3
    assert {c.name: (c.unit, c.original_unit, c.kind) for c in description.columns} == {
        "frequency": ("Hz", "kHz", "numeric"), "temperature": ("K", "degC", "numeric"), "loss_density": ("W/m^3", "kW/m^3", "numeric"),
        "excitation": (None, None, "categorical")}
    assert next(c for c in description.columns if c.name == "frequency").source_column == "Frequency"
    data = store.read(tmp_path / "t.h5")
    assert data.n_rows == 3 and data.units["temperature"] == "K" and data.original_units["temperature"] == "degC"
    assert np.array_equal(data.columns["frequency"], [50e3, 100e3, 200e3])
    assert np.array_equal(data.columns["temperature"], [298.15, np.nan, 373.15], equal_nan=True), "missing stays NaN"
    assert np.array_equal(data.columns["loss_density"], [1e4, 2e4, 3e4])
    assert data.labels("excitation") == ["sinusoidal", "triangular", None]
    assert store.verify_checksums(tmp_path / "t.h5") == [] and store.describe(tmp_path / "t.h5") == description
    artifact = store.artifact(tmp_path / "t.h5")
    assert artifact.role == "normalized" and artifact.sha256 == sha256_file(tmp_path / "t.h5")
    store.write(tmp_path / "t2.h5", columns, meta=meta, categorical=categorical, source_columns={"frequency": "Frequency"})
    assert sha256_file(tmp_path / "t.h5") == sha256_file(tmp_path / "t2.h5"), "deterministic bytes"
    with pytest.raises(FileExistsError):
        store.write(tmp_path / "t.h5", columns, meta=meta)
    with h5py.File(tmp_path / "t.h5", "r+") as handle:
        handle["conditions/frequency"][0] = 1.0
    assert any(m.startswith("conditions/frequency") for m in store.verify_checksums(tmp_path / "t.h5"))
    with pytest.raises(DataIntegrityError):
        store.read(tmp_path / "t.h5")
    with h5py.File(tmp_path / "t2.h5", "r+") as handle:
        handle["categories/excitation"][0] = 1
    assert any(m.startswith("categories/excitation") for m in store.verify_checksums(tmp_path / "t2.h5"))


@pytest.mark.usefixtures("require_h5py")
def test_table_store_rejects_bad_input_before_writing(tmp_path: Path):
    store = TableStore()
    meta = TableMeta(source_id="aipe.test", dataset_version="t")
    target = tmp_path / "x.h5"
    with pytest.raises(InvalidInputError):
        store.write(target, {"frequency": np.array([1.0])}, meta=meta)  # type: ignore[dict-item]
    with pytest.raises(UnitError):
        store.write(target, {"frequency": (np.array([1.0]), "furlong")}, meta=meta)
    with pytest.raises(InvalidInputError):
        store.write(target, {"frequency": (np.array([1.0, np.inf]), "Hz")}, meta=meta)
    with pytest.raises(InvalidInputError):
        store.write(target, {"frequency": (np.array([1.0, 2.0]), "Hz"), "temperature": (np.array([1.0]), "K")}, meta=meta)
    with pytest.raises(InvalidInputError):
        store.write(target, {"Bad Name": (np.array([1.0]), "Hz")}, meta=meta)
    with pytest.raises(InvalidInputError):
        store.write(target, {"frequency": (np.array([1.0]), "Hz")}, meta=meta, categorical={"c": (np.array([5]), ["a"])})
    with pytest.raises(InvalidInputError):
        store.write(target, {}, meta=meta)
    assert not target.exists()


# -- LocalCache -----------------------------------------------------------------------------------------
def test_cache_refuses_download_unless_allowed_and_verifies_sha256(tmp_path: Path):
    calls: dict = {}
    cache = LocalCache(tmp_path / "cache", transport=fake_transport(calls))
    with pytest.raises(DownloadNotAllowedError) as info:
        cache.fetch(URL, DIGEST)
    assert "allow_download=True" in str(info.value) and info.value.code == "download_not_allowed" and calls == {}
    with pytest.raises(DataIntegrityError, match="was expected; the file was discarded"):
        cache.fetch(URL, "0" * 64, allow_download=True)
    assert [p for p in (tmp_path / "cache").rglob("*") if p.is_file()] == [], "nothing is kept after a digest mismatch"
    path = cache.fetch(URL, DIGEST, allow_download=True, filename="file.bin", commit="1" * 40, license="MIT", source_id="t", timeout=12.5)
    assert path.is_file() and path.read_bytes() == PAYLOAD and calls["timeout"] == 12.5 and calls["n"] == 2
    manifest = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == MANIFEST_VERSION
    entry = manifest["files"][URL]
    assert entry["sha256"] == DIGEST and entry["size"] == len(PAYLOAD) and entry["commit"] == "1" * 40 and entry["license"] == "MIT"
    assert "retrieved_at" not in entry, "no wall-clock timestamp unless the caller supplies one"
    assert cache.fetch(URL, DIGEST) == path and cache.fetch(URL) == path and calls["n"] == 2, "served from the cache"
    assert cache.verify() == [] and cache.lookup(URL).filename == "file.bin"
    with pytest.raises(DataIntegrityError, match="different upstream version"):
        cache.fetch(URL, "f" * 64)
    with pytest.raises(InvalidInputError):
        cache.fetch(URL, "not-a-digest", allow_download=True)


def test_cache_detects_tampering_and_missing_files(tmp_path: Path):
    cache = LocalCache(tmp_path / "cache", transport=fake_transport())
    path = cache.fetch(URL, DIGEST, allow_download=True)
    path.write_bytes(PAYLOAD[:-3] + b"xyz")
    problems = cache.verify()
    assert any("sha256" in p for p in problems)
    with pytest.raises(DataIntegrityError, match="no longer matches its manifest digest"):
        cache.fetch(URL)
    path.unlink()
    assert any("missing" in p for p in cache.verify())
    with pytest.raises(DownloadNotAllowedError):
        cache.fetch(URL, DIGEST)


def test_cache_root_resolution_and_manifest_integrity(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
    assert LocalCache().root == Path(DEFAULT_CACHE_DIRNAME)
    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "env-cache"))
    cache = LocalCache(transport=fake_transport())
    assert cache.root == tmp_path / "env-cache"
    cache.fetch(URL, DIGEST, allow_download=True)
    cache.manifest_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        cache.lookup(URL)
    cache.manifest_path.write_text(json.dumps({"manifest_version": "other/9"}), encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        cache.verify()


def test_transport_rejects_unsafe_urls_and_oversized_downloads(tmp_path: Path):
    with pytest.raises(InvalidInputError):
        urllib_transport("file:///etc/passwd", 1.0)
    with pytest.raises(InvalidInputError):
        urllib_transport("https://user:secret@example.invalid/x", 1.0)
    cache = LocalCache(tmp_path / "cache", transport=fake_transport(), max_bytes=10)
    with pytest.raises(DataIntegrityError, match="exceeds the cache limit"):
        cache.fetch(URL, DIGEST, allow_download=True)
    assert [p for p in (tmp_path / "cache").rglob("*") if p.is_file()] == []


def test_clone_pinned_requires_permission_and_valid_inputs(tmp_path: Path):
    cache = LocalCache(tmp_path / "cache")
    with pytest.raises(DownloadNotAllowedError):
        cache.clone_pinned("https://example.invalid/repo.git", "a" * 40)
    with pytest.raises(InvalidInputError):
        cache.clone_pinned("https://example.invalid/repo.git", "short", allow_download=True)
    with pytest.raises(InvalidInputError):
        cache.clone_pinned("git://example.invalid/repo.git", "a" * 40, allow_download=True)
    assert cache.find_checkout("https://example.invalid/repo.git", "a" * 40) is None
    assert not (tmp_path / "cache" / "checkouts").exists()


# -- record writing -------------------------------------------------------------------------------------
def test_write_records_never_overwrites_but_reuses_identical_identity_records(tmp_path: Path):
    adapter = PrincetonMagnetAdapter(LocalCache(tmp_path / "cache"))
    evidence = adapter._repository_evidence()
    identity = adapter._material_identity_record(material="N87", evidence=[evidence], manufacturer="TDK")
    assert identity.id == "material.princeton-magnet-n87" and identity.properties == {"grade": "N87", "description": identity.description}
    written = write_records([identity], tmp_path)
    assert written == [str(tmp_path / "material.princeton-magnet-n87.json")]
    assert write_records([identity], tmp_path, reusable_ids=[identity.id]) == [], "identical identity record reused"
    with pytest.raises(FileExistsError):
        write_records([identity], tmp_path)
    other = adapter._material_identity_record(material="N87", evidence=[evidence], manufacturer="Other")
    with pytest.raises(FileExistsError):
        write_records([other], tmp_path, reusable_ids=[other.id])


def test_generic_validate_reports_shape_unit_and_checksum_problems(tmp_path: Path):
    adapter = PrincetonMagnetAdapter(LocalCache(tmp_path / "cache"))
    item = SourceItem(source_id="princeton_magnet", item_id="x", kind="file", available=True)
    extracted = ExtractedData(source_id="princeton_magnet", item=item, kind="sequences",
                              arrays={"flux_density": (np.array([[0.0, 1.0, np.inf]]), "T"), "time": (np.array([[0.0, 2.0, 1.0]]), "s")},
                              conditions={"frequency": (np.array([-1.0]), "Hz"), "temperature": (np.array([1.0, 2.0]), "parsec")})
    problems = adapter.validate(extracted)
    assert any("infinite" in p for p in problems) and any("strictly increasing" in p for p in problems)
    assert any(p.startswith("frequency: 1 value(s) <= 0.0") for p in problems) and any("unknown unit 'parsec'" in p for p in problems)
    assert any("expected (1,)" in p for p in problems)
    wrong_source = ExtractedData(source_id="magnet_challenge2", item=item, kind="table", columns={"frequency": (np.array([1.0]), "Hz")})
    assert adapter.validate(wrong_source)[0].startswith("extracted data belongs to")
    source = tmp_path / "s.bin"
    source.write_bytes(b"1")
    checked = ExtractedData(source_id="princeton_magnet", item=item, kind="table", columns={"frequency": (np.array([1.0e5]), "Hz")},
                            source_files={"s.bin": "0" * 64}, source_paths={"s.bin": str(source)})
    assert adapter.validate(checked) == ["s.bin: source bytes changed since extraction (recorded 000000000000..., now "
                                         f"{sha256_file(source)[:12]}...)"]
