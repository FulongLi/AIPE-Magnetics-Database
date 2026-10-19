"""v0.1 -> v0.2 migration: every v0.1 kind, meaning preserved, licence rule, refusals, file CLI."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from aipe_magnetics import __version__
from aipe_magnetics.common.errors import DataIntegrityError, InvalidInputError
from aipe_magnetics.common.jsonio import load, sha256_bytes
from aipe_magnetics.data import migrate as migrate_module
from aipe_magnetics.data.fixtures import V01_KINDS, v01_fixture_paths, v01_fixture_records
from aipe_magnetics.data.migrate import MIGRATION_EXTENSION, migrate_file, migrate_record
from aipe_magnetics.data.records import MagneticRecord
from aipe_magnetics.data.schema import validate_catalogue, validate_legacy_records, validate_record


def test_example_migrates_and_keeps_meaning(v01_example):
    original = copy.deepcopy(v01_example)
    migrated = migrate_record(v01_example)
    assert v01_example == original, "input must not be modified"
    assert validate_record(migrated) == []
    MagneticRecord.model_validate(migrated)
    for key in ("id", "kind", "name", "synthetic", "properties", "evidence", "evidence_refs"):
        assert migrated[key] == original[key]
    assert migrated["schema_version"] == "0.2.0"
    assert migrated["license"] == "NOASSERTION"
    assert migrated["provenance"] == {
        "activity": f"migrated from schema 0.1.0 by aipe-magnetics {__version__}",
        "tool": {"id": "aipe.magnetics", "name": "AIPE Magnetics", "version": __version__},
        "inputs": [], "normalization": "none",
    }
    assert migrated["extensions"] == {MIGRATION_EXTENSION: {"from": "0.1.0", "to": "0.2.0", "tool_version": __version__}}
    assert migrate_record(original) == migrated, "migration is deterministic"


def test_fixture_set_covers_every_v01_kind_and_is_legacy_valid():
    fixtures = v01_fixture_records()
    assert sorted(fixtures) == sorted(V01_KINDS)
    assert len(v01_fixture_paths()) == len(V01_KINDS)
    assert validate_legacy_records(list(fixtures.values())) == []
    assert all(record["synthetic"] is True for record in fixtures.values())


@pytest.mark.parametrize("kind", sorted(V01_KINDS))
def test_every_v01_kind_migrates_and_validates(kind):
    record = v01_fixture_records()[kind]
    migrated = migrate_record(record)
    assert validate_record(migrated) == [], kind
    typed = MagneticRecord.model_validate(migrated)
    assert typed.kind == kind
    typed.typed_properties()
    if kind != "measurement":
        assert migrated["properties"] == record["properties"]


def test_migrated_fixture_catalogue_cross_validates():
    migrated = [migrate_record(record) for record in v01_fixture_records().values()]
    assert validate_catalogue(migrated) == []


def test_measurement_tool_becomes_attribution_without_inventing_values():
    record = v01_fixture_records()["measurement"]
    migrated = migrate_record(record)
    tool = migrated["properties"]["tool"]
    assert tool == {"id": record["properties"]["tool"]["name"], "name": record["properties"]["tool"]["name"],
                    "version": record["properties"]["tool"]["version"]}
    rest = {k: v for k, v in migrated["properties"].items() if k != "tool"}
    assert rest == {k: v for k, v in record["properties"].items() if k != "tool"}


def test_license_is_taken_from_referenced_evidence_only_when_unanimous(v01_example):
    record = copy.deepcopy(v01_example)
    record["evidence"][0]["license"] = "CC-BY-4.0"
    assert migrate_record(record)["license"] == "CC-BY-4.0"
    second = dict(record["evidence"][0], id="evidence.second", license="MIT")
    record["evidence"].append(second)
    assert migrate_record(record)["license"] == "CC-BY-4.0", "unreferenced evidence does not count"
    record["evidence_refs"].append("evidence.second")
    assert migrate_record(record)["license"] == "NOASSERTION"


@pytest.mark.parametrize("mutate, error", [
    (lambda r: r.__setitem__("schema_version", "0.2.0"), InvalidInputError),
    (lambda r: r.__setitem__("schema_version", "9.0.0"), InvalidInputError),
    (lambda r: r.pop("schema_version"), InvalidInputError),
    (lambda r: r["properties"].__setitem__("permeablity", 1), DataIntegrityError),
    (lambda r: r["evidence"][0].__setitem__("created_at", "not a time"), DataIntegrityError),
    (lambda r: r.setdefault("extensions", {}).__setitem__(MIGRATION_EXTENSION, {"from": "0.1.0"}), InvalidInputError),
])
def test_migration_refuses_invalid_or_already_migrated_input(v01_example, mutate, error):
    mutate(v01_example)
    with pytest.raises(error):
        migrate_record(v01_example)


def test_migration_rejects_non_dict():
    with pytest.raises(InvalidInputError):
        migrate_record(["not", "a", "record"])  # type: ignore[arg-type]


def test_migrate_file_records_source_digest_and_never_overwrites(tmp_path: Path, v01_example_path: Path):
    destination = tmp_path / "out" / "migrated.json"
    written = migrate_file(v01_example_path, destination)
    assert written == destination
    migrated = load(destination)
    assert validate_record(migrated) == []
    extension = migrated["extensions"][MIGRATION_EXTENSION]
    assert extension["source_filename"] == v01_example_path.name
    assert extension["source_sha256"] == sha256_bytes(v01_example_path.read_bytes())
    assert destination.read_bytes().endswith(b"\n") and b"\r\n" not in destination.read_bytes()
    with pytest.raises(FileExistsError):
        migrate_file(v01_example_path, destination)
    assert migrate_file(v01_example_path, destination, overwrite=True) == destination
    with pytest.raises(InvalidInputError):
        migrate_file(v01_example_path, v01_example_path)
    with pytest.raises(InvalidInputError):  # the output is already 0.2.0
        migrate_file(destination, tmp_path / "twice.json")
    assert not (tmp_path / "twice.json").exists()


def test_cli(tmp_path: Path, v01_example_path: Path, capsys):
    out = tmp_path / "cli.json"
    assert migrate_module.main([str(v01_example_path), str(out)]) == 0
    assert "migrated" in capsys.readouterr().out
    assert migrate_module.main([str(v01_example_path), str(out)]) == 1
    assert migrate_module.main([str(v01_example_path), str(out), "--overwrite"]) == 0
    assert migrate_module.main([str(tmp_path / "missing.json"), str(tmp_path / "x.json")]) == 1
