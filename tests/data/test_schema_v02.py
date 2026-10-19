"""JSON Schema v0.2 generation, export/mirror policy, version-dispatching validation and the CLI."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from aipe_magnetics.common.errors import DataNotFoundError, InvalidInputError
from aipe_magnetics.common.jsonio import dump, load
from aipe_magnetics.data import schema as schema_module
from aipe_magnetics.data.catalog import Catalog
from aipe_magnetics.data.migrate import migrate_record
from aipe_magnetics.data.records import PROPERTY_MODELS
from aipe_magnetics.data.schema import (
    SCHEMA_ID_V02,
    SCHEMA_TITLE_V02,
    check_exported_schemas,
    compare_directories,
    export_schemas,
    generate_property_schema_v02,
    generate_quantity_schema_v02,
    generate_schema_v02,
    legacy_validator_source,
    load_schema,
    mirror_schema_dir,
    package_schema_dir,
    render_schema_files_v02,
    repository_root,
    schema_files_v02,
    validate_catalogue,
    validate_record,
)


def _refs(node, out):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref":
                out.append(value)
            else:
                _refs(value, out)
    elif isinstance(node, list):
        for item in node:
            _refs(item, out)
    return out


# -- generation -------------------------------------------------------------------------------------
def test_generated_schema_is_valid_draft_2020_12_and_self_contained():
    schema = generate_schema_v02()
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == SCHEMA_ID_V02
    assert schema["title"] == SCHEMA_TITLE_V02
    assert schema["required"][0] == "schema_version"
    assert schema["additionalProperties"] is False
    refs = _refs(schema, [])
    assert refs and all(r.startswith("#/$defs/") for r in refs)
    assert all(r.split("/")[-1] in schema["$defs"] for r in refs)
    for name in ("Quantity", "Evidence", "Provenance", "Artifact", "Quality", "MaterialProperties", "WaveformDatasetProperties"):
        assert name in schema["$defs"]
    assert schema["$defs"]["Evidence"]["properties"]["created_at"]["format"] == "date-time"


def test_if_then_blocks_bind_every_kind_in_normative_order():
    schema = generate_schema_v02()
    kinds = [block["if"]["properties"]["kind"]["const"] for block in schema["allOf"]]
    assert kinds == list(PROPERTY_MODELS)
    for block, (kind, model) in zip(schema["allOf"], PROPERTY_MODELS.items()):
        assert block["if"]["required"] == ["kind"]
        assert block["then"]["properties"]["properties"] == {"$ref": f"#/$defs/{model.__name__}"}


def test_generation_is_deterministic():
    first, second = render_schema_files_v02(), render_schema_files_v02()
    assert first == second
    assert all(content.endswith(b"\n") and b"\r\n" not in content for content in first.values())
    assert set(first) == {"magnetic-record.schema.json", "quantity.schema.json"} | {f"properties/{k}.schema.json" for k in PROPERTY_MODELS}


@pytest.mark.parametrize("kind", sorted(PROPERTY_MODELS))
def test_per_kind_property_schema_is_standalone(kind):
    document = generate_property_schema_v02(kind)
    Draft202012Validator.check_schema(document)
    assert document["$id"].endswith(f"/schemas/v0.2/properties/{kind}.schema.json")
    assert all(r.startswith("#/$defs/") for r in _refs(document, []))
    validator = Draft202012Validator(document, format_checker=FormatChecker())
    assert list(validator.iter_errors({})) == [] or kind in ("core_loss_measurement", "bh_loop", "waveform_dataset", "dataset",
                                                             "excitation_waveform", "loss_model", "model_metadata")
    assert list(validator.iter_errors({"definitely_not_a_field": 1}))


def test_property_schema_accepts_bundled_material_and_rejects_corruption():
    material = Catalog.default().get("material.example-synthetic-ferrite").properties
    validator = Draft202012Validator(generate_property_schema_v02("magnetic_material"))
    assert list(validator.iter_errors(material)) == []
    broken = copy.deepcopy(material)
    broken["steinmetz_parameter_sets"][0]["alpha"] = "1.5"
    assert list(validator.iter_errors(broken))
    with pytest.raises(InvalidInputError):
        generate_property_schema_v02("not_a_kind")


def test_quantity_schema():
    document = generate_quantity_schema_v02()
    Draft202012Validator.check_schema(document)
    validator = Draft202012Validator(document)
    assert list(validator.iter_errors({"value": 1.0, "unit": "T", "original": {"value": 1000, "unit": "mT"}})) == []
    assert list(validator.iter_errors({"value": "1", "unit": "T"}))
    assert list(validator.iter_errors({"value": 1, "unit": "T", "extra": 1}))


# -- packaged copies and mirror -------------------------------------------------------------------
def test_packaged_v02_schema_matches_fresh_export():
    assert check_exported_schemas(package_schema_dir()) == []
    assert load_schema("0.2.0") == generate_schema_v02()


def test_repository_mirror_is_byte_identical():
    mirror = mirror_schema_dir()
    assert mirror is not None and mirror.is_dir(), "tests run from a source checkout"
    assert compare_directories(package_schema_dir(), mirror) == []
    assert check_exported_schemas(mirror) == []


def test_legacy_schema_packaged_copy_is_byte_identical(repo_root: Path):
    legacy = repo_root / "schemas" / "magnetic-record.schema.json"
    packaged = package_schema_dir("0.1.0") / "magnetic-record.schema.json"
    assert packaged.read_bytes() == legacy.read_bytes(), (
        "src/aipe_magnetics/schemas/v0.1/magnetic-record.schema.json must be an exact byte copy of schemas/magnetic-record.schema.json")
    assert load_schema("0.1.0")["$id"].endswith("/v0.1.0/schemas/magnetic-record.schema.json")


def test_load_schema_returns_copies_and_rejects_unknown_versions():
    first = load_schema("0.2.0")
    first["title"] = "mutated"
    assert load_schema("0.2.0")["title"] == SCHEMA_TITLE_V02
    with pytest.raises(InvalidInputError):
        load_schema("0.3.0")
    with pytest.raises(InvalidInputError):
        package_schema_dir("9.9.9")


def test_missing_packaged_file_is_reported(monkeypatch, tmp_path: Path):
    schema_module._load_schema_cached.cache_clear()
    monkeypatch.setattr(schema_module, "package_schema_dir", lambda version="0.2.0": tmp_path / "nowhere")
    try:
        with pytest.raises(DataNotFoundError):
            schema_module._load_schema_cached("0.2.0")
    finally:
        schema_module._load_schema_cached.cache_clear()
    monkeypatch.undo()
    schema_module._load_schema_cached.cache_clear()
    assert load_schema("0.2.0")["title"] == SCHEMA_TITLE_V02


# -- validation -------------------------------------------------------------------------------------
def test_validate_record_dispatches_on_schema_version(v01_example):
    assert validate_record(v01_example) == []
    assert legacy_validator_source() in ("scripts/validate.py", "replica")
    migrated = migrate_record(v01_example)
    assert validate_record(migrated) == []
    assert validate_record({"id": "x"}) == ["[schema] $.schema_version: missing; cannot select a schema (supported: 0.1.0, 0.2.0)"]
    assert validate_record(dict(migrated, schema_version="9.0.0"))[0].startswith("[schema] $.schema_version: unsupported")
    assert validate_record(["not", "a", "dict"]) == ["[schema] $: record must be a JSON object, got list"]
    assert validate_record("x")[0].startswith("[schema] $: record must be a JSON object")


@pytest.mark.parametrize("mutate, fragment", [
    (lambda r: r["properties"].__setitem__("leakage_inductance", {"value": 1, "unit": "uH"}), "SI unit 'H'"),
    (lambda r: r["properties"].__setitem__("leakage_inductance", 1e-6), "[schema]"),
    (lambda r: r["properties"].__setitem__("permeablity", 23), "permeablity"),
    (lambda r: r["evidence"][0].__setitem__("created_at", "2026-99-10"), "date-time"),
    (lambda r: r.__setitem__("synthetic", False), "non-assumption"),
    (lambda r: r.__setitem__("evidence_refs", ["evidence.absent"]), "evidence.absent"),
    (lambda r: r["evidence"].append(dict(r["evidence"][0])), "duplicate"),
    (lambda r: r["evidence"][0].__setitem__("review_status", "rejected"), "rejected"),
    (lambda r: r.__setitem__("extensions", {"noprefix": {}}), "namespaced"),
    (lambda r: r.__setitem__("id", "Bad Id"), "must match"),
    # v0.2 applies the measurement-evidence rule to real (non-synthetic) records only
    (lambda r: (r.__setitem__("kind", "measurement"), r.__setitem__("synthetic", False),
                r["evidence"][0].__setitem__("kind", "source")), "measurement evidence"),
    (lambda r: r.__setitem__("properties", {"core_ref": 5}), "core_ref"),
    (lambda r: r.__setitem__("license", ""), "license"),
    (lambda r: r.__setitem__("artifacts", [{"uri": "x.h5", "media_type": "application/x-hdf5", "sha256": "zz"}]), "sha256"),
])
def test_validate_record_rejects_invalid_v02(v01_example, mutate, fragment):
    record = migrate_record(v01_example)
    mutate(record)
    errors = validate_record(record)
    assert errors, "expected validation errors"
    assert any(fragment in error for error in errors), errors


def test_legacy_validation_rules_still_apply_to_v01(v01_example):
    bad = copy.deepcopy(v01_example)
    bad["properties"]["leakage_inductance"] = {"value": -1, "unit": "H"}
    assert validate_record(bad)
    bad = copy.deepcopy(v01_example)
    bad["kind"] = "measurement"  # v0.1 schema requires component_ref/operating_condition_ref/artifact/tool
    errors = validate_record(bad)
    assert errors and any("required property" in e for e in errors), errors
    bad = copy.deepcopy(v01_example)
    bad["synthetic"] = False
    assert any("non-assumption" in e for e in validate_record(bad))


def test_legacy_replica_matches_repository_script(v01_example):
    """The packaged replica must agree with scripts/validate.py on the example and on corruptions."""
    cases = [copy.deepcopy(v01_example)]
    broken = copy.deepcopy(v01_example)
    broken["evidence"][0]["review_status"] = "rejected"
    cases.append(broken)
    broken = copy.deepcopy(v01_example)
    broken["properties"]["core_ref"] = "core.missing"
    cases.append(broken)
    for case in cases:
        via_replica = schema_module._legacy_replica([case])
        via_dispatch = validate_record(case)
        assert bool(via_replica) == bool(via_dispatch)


def test_validate_catalogue_mixed_versions(v01_example):
    migrated = migrate_record(v01_example)
    bundled = Catalog.default().records()
    assert validate_catalogue([*bundled, migrated]) == []
    assert validate_catalogue([v01_example, migrated]) == ["Duplicate record ID across schema versions: magnetic.example-transformer"]
    other = dict(migrated, id="magnetic.other-transformer")
    other["properties"] = dict(other["properties"], core_ref="core.example-synthetic-e42")
    assert validate_catalogue([*bundled, other]) == []
    assert any("unknown record reference core.example-synthetic-e42" in e for e in validate_catalogue([other]))
    assert validate_catalogue([42])[0].startswith("record[0]: not a JSON object")
    assert validate_catalogue([dict(migrated, name="")])[0].startswith("record[0] [")


# -- command line -----------------------------------------------------------------------------------
def test_cli_export_check_and_validate(tmp_path: Path, v01_example_path: Path, capsys):
    out = tmp_path / "export"
    assert schema_module.main(["--export", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["magnetic-record.schema.json", "properties", "quantity.schema.json"]
    assert schema_module.main(["--check", str(out)]) == 0
    (out / "quantity.schema.json").write_bytes(b"{}\n")
    assert schema_module.main(["--check", str(out)]) == 1
    assert "content differs" in capsys.readouterr().out
    (out / "quantity.schema.json").unlink()
    assert "quantity.schema.json: missing" in check_exported_schemas(out)
    (out / "stray.schema.json").write_bytes(b"{}\n")
    assert any("unexpected file" in p for p in check_exported_schemas(out))
    assert check_exported_schemas(tmp_path / "nope") == [f"{tmp_path / 'nope'}: directory does not exist"]
    crlf = tmp_path / "crlf"
    export_schemas(crlf)
    path = crlf / "quantity.schema.json"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert any("line endings" in p for p in check_exported_schemas(crlf))
    assert schema_module.main(["--validate", str(v01_example_path)]) == 0
    bad = tmp_path / "bad.json"
    dump({"schema_version": "0.2.0", "id": "x"}, bad)
    assert schema_module.main(["--validate", str(bad)]) == 1
    assert schema_module.main([]) == 2


def test_repository_root_detection(repo_root: Path):
    assert repository_root() == repo_root
    files = schema_files_v02()
    assert files["magnetic-record.schema.json"]["$id"] == SCHEMA_ID_V02
