"""Catalog: bundled records, user catalogues, typed look-ups, loud failures on malformed files.

The bundled catalogue starts with one clearly synthetic manufacturer / material / core triple and
grows as the maintainer adds real entries. Assertions about the *bundled* catalogue are therefore
per-record invariants and membership tests; exact-list and not-found behaviour is asserted on an
in-memory catalogue that holds only the synthetic triple.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from aipe_magnetics.common.errors import DataIntegrityError, DataNotFoundError, InvalidInputError
from aipe_magnetics.common.jsonio import dump
from aipe_magnetics.data.catalog import BUNDLED_CATALOG_DIR, Catalog, load_record_file
from aipe_magnetics.data.fixtures import v01_fixture_records
from aipe_magnetics.data.migrate import migrate_record
from aipe_magnetics.data.records import MagneticRecord

SYNTHETIC_IDS = {"manufacturer.example-synthetic", "material.example-synthetic-ferrite", "core.example-synthetic-e42"}
REAL_EVIDENCE_KINDS = {"source", "measurement", "analytical", "simulation"}


def _clone(record: MagneticRecord, **changes) -> MagneticRecord:
    data = record.to_json_dict()
    data.update(changes)
    return MagneticRecord.model_validate(data)


@pytest.fixture
def triple() -> Catalog:
    """In-memory catalogue holding only the synthetic triple (exact-list assertions live here)."""
    return Catalog([r for r in Catalog.default().records() if r.id in SYNTHETIC_IDS])


def test_bundled_catalogue_is_consistent_cached_and_every_record_keeps_its_evidence_invariants():
    catalog = Catalog.default()
    assert catalog is Catalog.default()
    assert SYNTHETIC_IDS <= set(catalog.ids())
    assert catalog.validate() == []
    assert (BUNDLED_CATALOG_DIR / "README.md").is_file()
    for record in catalog:
        assert catalog.source_of(record.id) is not None and catalog.source_of(record.id).is_relative_to(BUNDLED_CATALOG_DIR)
        selected = [e for e in record.evidence if e.id in record.evidence_refs]
        if record.synthetic:
            assert "synthetic" in record.tags, record.id
        else:
            assert any(e.kind in REAL_EVIDENCE_KINDS for e in selected), f"{record.id}: real record without source/measurement evidence"
            for e in selected:
                if e.kind == "source":
                    assert e.locator or e.url, f"{record.id}: source evidence {e.id} needs a locator or url"
        assert all(e.license for e in record.evidence), record.id
    for record_id in SYNTHETIC_IDS:
        record = catalog.get(record_id)
        assert record.synthetic and all(e.kind == "assumption" for e in record.evidence), "the synthetic triple rests on assumption evidence"
    description = catalog.describe()
    assert description["records"] == len(catalog) >= 3 and description["synthetic"] >= 3
    assert {"core_geometry", "magnetic_material", "manufacturer"} <= set(description["by_kind"])
    Catalog.clear_default_cache()
    assert Catalog.default() is not catalog and SYNTHETIC_IDS <= set(Catalog.default().ids())


def test_typed_lookups_on_the_bundled_catalogue():
    catalog = Catalog.default()
    assert "material.example-synthetic-ferrite" in [m.id for m in catalog.materials()]
    assert "manufacturer.example-synthetic" in [m.id for m in catalog.manufacturers()]
    assert "core.example-synthetic-e42" in [c.id for c in catalog.cores()]
    assert "core.example-synthetic-e42" in [c.id for c in catalog.cores("material.example-synthetic-ferrite")]
    assert catalog.cores("manufacturer.example-synthetic") == [], "a manufacturer id never appears in material_refs"
    assert "core.example-synthetic-e42" in [r.id for r in catalog.records("core_geometry")]
    assert all(r.kind == "core_geometry" for r in catalog.records("core_geometry"))
    assert catalog.records() == sorted(catalog.records(), key=lambda r: r.id)
    assert "core.example-synthetic-e42" in catalog and "core.nope" not in catalog
    material = catalog.get("material.example-synthetic-ferrite")
    assert material.typed_properties().grade == "EXAMPLE-SYN-1"
    with pytest.raises(InvalidInputError):
        catalog.records("not_a_kind")
    with pytest.raises(DataNotFoundError):
        catalog.cores("material.unknown")
    with pytest.raises(DataNotFoundError) as info:
        catalog.get("material.example-synthetic-ferite")
    assert info.value.details["suggestions"][0] == "material.example-synthetic-ferrite", "best match first"
    with pytest.raises(DataNotFoundError) as info:
        catalog.get("zzz")
    assert info.value.details["suggestions"] == [] and "did you mean" not in str(info.value)


def test_typed_lookups_exact_lists_on_the_synthetic_triple(triple: Catalog):
    assert len(triple) == 3 and triple.validate() == []
    assert [m.id for m in triple.materials()] == ["material.example-synthetic-ferrite"]
    assert [m.id for m in triple.manufacturers()] == ["manufacturer.example-synthetic"]
    assert [c.id for c in triple.cores()] == ["core.example-synthetic-e42"]
    assert [c.id for c in triple.cores("material.example-synthetic-ferrite")] == ["core.example-synthetic-e42"]
    assert triple.records("bh_loop") == []
    assert triple.describe()["by_kind"] == {"core_geometry": 1, "magnetic_material": 1, "manufacturer": 1}
    assert triple.source_of("core.example-synthetic-e42") is None, "in-memory records have no source file"


@pytest.mark.parametrize("query", ["material.example-synthetic-ferrite", "MATERIAL.EXAMPLE-SYNTHETIC-FERRITE", "EXAMPLE-SYN-1", "example-syn-1"])
def test_find_material_by_id_and_grade_on_the_bundled_catalogue(query):
    assert Catalog.default().find_material(query).id == "material.example-synthetic-ferrite"


@pytest.mark.parametrize("query", ["Example synthetic MnZn-type ferrite (illustrative)", "synthetic mnzn", "SYN-1"])
def test_find_material_by_name_and_substring_on_the_synthetic_triple(triple: Catalog, query):
    assert triple.find_material(query).id == "material.example-synthetic-ferrite"


def test_find_material_errors(triple: Catalog):
    with pytest.raises(DataNotFoundError) as info:
        triple.find_material("N87")
    assert info.value.details["available"] == ["material.example-synthetic-ferrite"]
    with pytest.raises(DataNotFoundError) as info:
        triple.find_material("example-synthetic-ferite")
    assert info.value.details["suggestions"] == ["material.example-synthetic-ferrite"]
    for bad in ("", "   ", None, 5):
        with pytest.raises(InvalidInputError):
            triple.find_material(bad)  # type: ignore[arg-type]
    base = triple.get("material.example-synthetic-ferrite")
    twin = _clone(base, id="material.example-synthetic-ferrite-b", name="Example synthetic twin (illustrative)")
    twin = _clone(twin, properties=dict(twin.properties, grade="EXAMPLE-SYN-1"))
    extended = triple.with_records([twin])
    with pytest.raises(InvalidInputError) as info:
        extended.find_material("EXAMPLE-SYN-1")
    assert set(info.value.details["candidates"]) == {base.id, twin.id}
    with pytest.raises(InvalidInputError):
        extended.find_material("synthetic")
    assert extended.find_material("material.example-synthetic-ferrite").id == base.id, "exact id wins over shared grade"
    assert extended.find_material("twin").id == twin.id


def test_with_records_refuses_duplicates_and_keeps_sources():
    catalog = Catalog.default()
    with pytest.raises(DataIntegrityError):
        catalog.with_records([catalog.get("core.example-synthetic-e42")])
    with pytest.raises(InvalidInputError):
        Catalog([{"id": "not.a.model"}])  # type: ignore[list-item]
    migrated = [MagneticRecord.model_validate(migrate_record(r)) for r in v01_fixture_records().values()]
    extended = catalog.with_records(migrated)
    assert len(extended) == len(catalog) + len(migrated) and extended.validate() == []
    assert extended.source_of(migrated[0].id) is None
    assert extended.source_of("core.example-synthetic-e42") == catalog.source_of("core.example-synthetic-e42")


def test_user_catalogue_directories_and_files_are_merged(tmp_path: Path):
    user_dir = tmp_path / "user"
    records = [migrate_record(r) for r in v01_fixture_records().values()]
    for record in records:
        dump(record, user_dir / "nested" / f"{record['id']}.json")
    single = tmp_path / "single.json"
    base = Catalog.default().get("core.example-synthetic-e42").to_json_dict()
    base["id"] = "core.example-synthetic-second"
    dump(base, single)
    catalog = Catalog.load([user_dir, single])
    assert len(catalog) == len(Catalog.default()) + len(records) + 1 and catalog.validate() == []
    assert catalog.source_of("core.example-synthetic-second") == single
    only_user = Catalog.load([user_dir], bundled=False)
    assert len(only_user) == len(records)
    with pytest.raises(DataNotFoundError):
        Catalog.load([tmp_path / "missing"])


@pytest.mark.parametrize("label, content", [
    ("broken json", "{not json"),
    ("duplicate keys", '{"schema_version": "0.2.0", "schema_version": "0.2.0"}'),
    ("array instead of object", "[]"),
])
def test_malformed_files_fail_loudly_with_path(tmp_path: Path, label, content):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DataIntegrityError) as info:
        Catalog.load([tmp_path], bundled=False)
    assert "bad.json" in str(info.value)
    with pytest.raises(DataIntegrityError):
        load_record_file(path)


def test_invalid_legacy_duplicate_and_physically_impossible_records_fail_loudly(tmp_path: Path, v01_example):
    legacy = tmp_path / "legacy.json"
    dump(v01_example, legacy)
    with pytest.raises(DataIntegrityError, match="migrate it first"):
        Catalog.load([tmp_path], bundled=False)
    legacy.unlink()
    invalid = Catalog.default().get("manufacturer.example-synthetic").to_json_dict()
    invalid["properties"]["unknown_field"] = 1
    dump(invalid, tmp_path / "invalid.json")
    with pytest.raises(DataIntegrityError) as info:
        Catalog.load([tmp_path], bundled=False)
    assert "invalid.json" in str(info.value) and "unknown_field" in str(info.value)
    (tmp_path / "invalid.json").unlink()
    negative = Catalog.default().get("core.example-synthetic-e42").to_json_dict()
    negative["id"] = "core.example-negative"
    negative["properties"]["effective_area"] = {"value": -2e-4, "unit": "m^2"}
    dump(negative, tmp_path / "negative.json")
    with pytest.raises(DataIntegrityError, match=r"\[physical\] \$\.properties\.effective_area\.value: must be positive"):
        Catalog.load([tmp_path], bundled=False)
    (tmp_path / "negative.json").unlink()
    duplicate = Catalog.default().get("manufacturer.example-synthetic").to_json_dict()
    dump(duplicate, tmp_path / "dup.json")
    with pytest.raises(DataIntegrityError, match="duplicate record id"):
        Catalog.load([tmp_path])
    unsupported = copy.deepcopy(duplicate)
    unsupported["schema_version"] = "0.3.0"
    (tmp_path / "dup.json").unlink()
    dump(unsupported, tmp_path / "future.json")
    with pytest.raises(DataIntegrityError, match="unsupported schema_version"):
        Catalog.load([tmp_path], bundled=False)
