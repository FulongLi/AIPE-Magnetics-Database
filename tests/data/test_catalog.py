"""Catalog: bundled synthetic triple, user catalogues, typed look-ups, loud failures on malformed files."""

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


def _clone(record: MagneticRecord, **changes) -> MagneticRecord:
    data = record.to_json_dict()
    data.update(changes)
    return MagneticRecord.model_validate(data)


def test_bundled_catalogue_is_synthetic_consistent_and_cached():
    catalog = Catalog.default()
    assert catalog is Catalog.default()
    assert set(catalog.ids()) == SYNTHETIC_IDS and len(catalog) == 3
    assert catalog.validate() == []
    assert all(record.synthetic for record in catalog) and all("synthetic" in record.tags for record in catalog)
    assert all(e.kind == "assumption" for record in catalog for e in record.evidence)
    assert (BUNDLED_CATALOG_DIR / "README.md").is_file()
    for record in catalog:
        assert catalog.source_of(record.id) is not None and catalog.source_of(record.id).is_relative_to(BUNDLED_CATALOG_DIR)
    description = catalog.describe()
    assert description["by_kind"] == {"core_geometry": 1, "magnetic_material": 1, "manufacturer": 1} and description["synthetic"] == 3
    Catalog.clear_default_cache()
    assert Catalog.default() is not catalog and set(Catalog.default().ids()) == SYNTHETIC_IDS


def test_typed_lookups():
    catalog = Catalog.default()
    assert [m.id for m in catalog.materials()] == ["material.example-synthetic-ferrite"]
    assert [m.id for m in catalog.manufacturers()] == ["manufacturer.example-synthetic"]
    assert [c.id for c in catalog.cores()] == ["core.example-synthetic-e42"]
    assert [c.id for c in catalog.cores("material.example-synthetic-ferrite")] == ["core.example-synthetic-e42"]
    assert catalog.cores("manufacturer.example-synthetic") == []
    assert [r.id for r in catalog.records("core_geometry")] == ["core.example-synthetic-e42"]
    assert catalog.records("bh_loop") == []
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


@pytest.mark.parametrize("query", ["material.example-synthetic-ferrite", "MATERIAL.EXAMPLE-SYNTHETIC-FERRITE", "EXAMPLE-SYN-1",
                                   "example-syn-1", "Example synthetic MnZn-type ferrite (illustrative)", "synthetic mnzn", "SYN-1"])
def test_find_material_by_id_grade_and_name(query):
    assert Catalog.default().find_material(query).id == "material.example-synthetic-ferrite"


def test_find_material_errors():
    catalog = Catalog.default()
    with pytest.raises(DataNotFoundError) as info:
        catalog.find_material("N87")
    assert info.value.details["available"] == ["material.example-synthetic-ferrite"]
    with pytest.raises(DataNotFoundError) as info:
        catalog.find_material("example-synthetic-ferite")
    assert info.value.details["suggestions"] == ["material.example-synthetic-ferrite"]
    for bad in ("", "   ", None, 5):
        with pytest.raises(InvalidInputError):
            catalog.find_material(bad)  # type: ignore[arg-type]
    base = catalog.get("material.example-synthetic-ferrite")
    twin = _clone(base, id="material.example-synthetic-ferrite-b", name="Example synthetic twin (illustrative)")
    twin_props = dict(twin.properties, grade="EXAMPLE-SYN-1")
    twin = _clone(twin, properties=twin_props)
    extended = catalog.with_records([twin])
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
    assert len(extended) == 3 + len(migrated) and extended.validate() == []
    assert extended.source_of(migrated[0].id) is None


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
    assert len(catalog) == 3 + len(records) + 1 and catalog.validate() == []
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


def test_invalid_legacy_and_duplicate_records_fail_loudly(tmp_path: Path, v01_example):
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
