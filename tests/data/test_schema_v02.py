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
from aipe_magnetics.data.fixtures import v01_fixture_paths, v01_fixture_records
from aipe_magnetics.data.migrate import migrate_record
from aipe_magnetics.data.records import PROPERTY_MODELS, MagneticRecord
from aipe_magnetics.data.schema import (
    SCHEMA_ID_V02,
    SCHEMA_TITLE_V02,
    check_exported,
    check_exported_schemas,
    compare_directories,
    export_schemas,
    generate_property_schema_v02,
    generate_quantity_schema_v02,
    generate_schema_v02,
    generator_versions,
    legacy_record_errors,
    legacy_validator_source,
    load_schema,
    mirror_schema_dir,
    package_schema_dir,
    physical_errors,
    render_schema_files_v02,
    repository_root,
    schema_errors,
    schema_files_v02,
    semantic_errors,
    validate_catalogue,
    validate_files,
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
    # v0.1 parity: measurement.artifact and loss_model.model_artifact are uri-references; v0.2 extends it to the URI fields
    assert schema["$defs"]["LossModelProperties"]["properties"]["model_artifact"]["format"] == "uri-reference"
    assert schema["$defs"]["Artifact"]["properties"]["uri"]["format"] == "uri-reference"
    for definition, prop in (("MeasurementProperties", "artifact"), ("Evidence", "url"), ("DatasetProperties", "url"),
                             ("ManufacturerProperties", "website"), ("ModelMetadataProperties", "weights_artifact")):
        branches = schema["$defs"][definition]["properties"][prop]["anyOf"]
        assert [b for b in branches if b.get("type") == "string"][0]["format"] == "uri-reference", (definition, prop)
        assert {"type": "null"} in branches, "optional fields keep their null branch"


def test_if_then_blocks_bind_every_kind_in_normative_order_and_require_properties_when_mandatory():
    schema = generate_schema_v02()
    kinds = [block["if"]["properties"]["kind"]["const"] for block in schema["allOf"]]
    assert kinds == list(PROPERTY_MODELS)
    for block, (kind, model) in zip(schema["allOf"], PROPERTY_MODELS.items()):
        assert block["if"]["required"] == ["kind"]
        assert block["then"]["properties"]["properties"] == {"$ref": f"#/$defs/{model.__name__}"}
        mandatory = any(f.is_required() for f in model.model_fields.values())
        assert block["then"].get("required") == (["properties"] if mandatory else None), kind
    assert {k for k, m in PROPERTY_MODELS.items() if any(f.is_required() for f in m.model_fields.values())} == {
        "core_loss_measurement", "bh_loop", "waveform_dataset", "dataset", "excitation_waveform", "loss_model", "model_metadata"}


def test_schema_layer_alone_rejects_a_mandatory_kind_without_properties(evidence_assumption):
    record = {"schema_version": "0.2.0", "id": "measurement.noprops", "kind": "core_loss_measurement", "name": "x", "synthetic": True,
              "evidence_refs": [evidence_assumption["id"]], "evidence": [evidence_assumption]}
    assert any("'properties' is a required property" in e for e in schema_errors(record))
    manufacturer = dict(record, id="manufacturer.noprops", kind="manufacturer")
    assert schema_errors(manufacturer) == [] and validate_record(manufacturer) == []


def test_generator_versions_are_reported():
    versions = generator_versions()
    assert set(versions) == {"pydantic", "jsonschema"} and all(v[0].isdigit() for v in versions.values())


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


def _corruptions(example: dict) -> list[tuple[str, dict]]:
    cases: list[tuple[str, dict]] = [("valid", copy.deepcopy(example))]
    broken = copy.deepcopy(example)
    broken["evidence"][0]["review_status"] = "rejected"
    cases.append(("rejected evidence", broken))
    broken = copy.deepcopy(example)
    broken["properties"]["core_ref"] = "core.missing"
    cases.append(("unknown reference", broken))
    broken = copy.deepcopy(example)
    broken["properties"]["leakage_inductance"] = {"value": float("inf"), "unit": "H"}
    cases.append(("non-finite", broken))
    broken = copy.deepcopy(example)
    broken["synthetic"] = False
    cases.append(("assumption-only real record", broken))
    broken = copy.deepcopy(example)
    broken["evidence_refs"] = ["evidence.absent"]
    cases.append(("unknown evidence", broken))
    broken = copy.deepcopy(example)
    broken["evidence"].append(copy.deepcopy(broken["evidence"][0]))
    cases.append(("duplicate evidence", broken))
    broken = copy.deepcopy(example)
    broken["properties"]["permeablity"] = 23
    cases.append(("unknown field", broken))
    return cases


def test_legacy_replica_matches_repository_script_message_for_message(v01_example):
    """The packaged replica must agree with scripts/validate.py on the example and on corruptions - same messages."""
    assert legacy_validator_source() == "scripts/validate.py", "tests run from a source checkout, so the real script is used"
    script = schema_module._legacy_validate_function()
    for label, case in _corruptions(v01_example):
        via_replica = schema_module._legacy_replica([case])
        via_script = [str(m) for m in script([case])]
        assert via_replica and via_script if label != "valid" else via_replica == via_script == []
        assert sorted(via_replica) == sorted(via_script), (label, via_replica, via_script)
        assert validate_record(case) == via_script, label
    pair = [copy.deepcopy(v01_example), copy.deepcopy(v01_example)]
    assert schema_module._legacy_replica(pair) == [str(m) for m in script(pair)] == ["Duplicate record ID: magnetic.example-transformer"]


def test_legacy_record_errors_apply_the_per_record_rules_without_references(v01_example):
    assert legacy_record_errors(v01_example) == []
    referencing = copy.deepcopy(v01_example)
    referencing["properties"]["core_ref"] = "core.elsewhere"
    assert legacy_record_errors(referencing) == [], "cross-record references are not a per-record rule"
    assert any("unknown record reference" in e for e in validate_record(referencing))
    measurement = v01_fixture_records()["measurement"]
    weakened = copy.deepcopy(measurement)
    for item in weakened["evidence"]:
        item["kind"] = "assumption"
    assert legacy_record_errors(weakened) == [f"{measurement['id']}: measurement requires measurement evidence"]
    rejected = copy.deepcopy(v01_example)
    rejected["evidence"][0]["review_status"] = "rejected"
    assert legacy_record_errors(rejected) == [f"{v01_example['id']}: rejected evidence {v01_example['evidence'][0]['id']}"]
    structural = copy.deepcopy(v01_example)
    structural["properties"]["leakage_inductance"] = {"value": -1, "unit": "H"}
    assert legacy_record_errors(structural) and all(e.startswith("record$.properties.leakage_inductance") for e in legacy_record_errors(structural))
    assert legacy_record_errors("nope")[0].startswith("record: not a JSON object")  # type: ignore[arg-type]


# -- physical plausibility layer --------------------------------------------------------------------
def _fixture_v02(kind: str) -> dict:
    return migrate_record(v01_fixture_records()[kind])


@pytest.mark.parametrize("kind, mutate, path, in_model", [
    # rules the normative model also enforces (records.py _positive/_nonnegative): both layers report, [physical] carries the path
    ("core_geometry", lambda p: p.__setitem__("effective_area", {"value": -1e-4, "unit": "m^2"}), "$.properties.effective_area.value", True),
    ("core_geometry", lambda p: p.__setitem__("effective_area", {"value": 0.0, "unit": "m^2"}), "$.properties.effective_area.value", True),
    ("core_geometry", lambda p: p.__setitem__("effective_volume", {"value": -1e-6, "unit": "m^3"}), "$.properties.effective_volume.value", True),
    ("core_geometry", lambda p: p.__setitem__("air_gap_options", [{"value": 0.0, "unit": "m"}, {"value": -1e-3, "unit": "m"}]),
     "$.properties.air_gap_options[1].value", True),
    ("core_geometry", lambda p: p.__setitem__("dimensions", {"values": {"A": {"value": -0.042, "unit": "m"}}}), "$.properties.dimensions.values.A.value", True),
    ("transformer", lambda p: p.__setitem__("leakage_inductance", {"value": -1e-6, "unit": "H"}), "$.properties.leakage_inductance.value", True),
    ("transformer", lambda p: p.__setitem__("turns_ratio", {"value": 0.0, "unit": "1"}), "$.properties.turns_ratio.value", True),
    ("inductor", lambda p: p.__setitem__("inductance", {"value": -1e-3, "unit": "H"}), "$.properties.inductance.value", True),
    ("operating_condition", lambda p: p.__setitem__("frequency", {"value": -100.0, "unit": "Hz"}), "$.properties.frequency.value", True),
    ("operating_condition", lambda p: p.__setitem__("temperature", {"value": -5.0, "unit": "K"}), "$.properties.temperature.value", True),
    ("operating_condition", lambda p: p.__setitem__("flux_density_peak", {"value": -0.1, "unit": "T"}), "$.properties.flux_density_peak.value", True),
    ("operating_condition", lambda p: p.__setitem__("current_rms", {"value": -1.0, "unit": "A"}), "$.properties.current_rms.value", True),
    ("winding", lambda p: p.__setitem__("conductor_area", {"value": -1e-6, "unit": "m^2"}), "$.properties.conductor_area.value", True),
    ("core_material", lambda p: p.__setitem__("relative_permeability", {"value": 0.0, "unit": "1"}), "$.properties.relative_permeability.value", True),
    # rules only the data layer states (the normative model leaves these quantities unconstrained)
    ("magnetic_material", lambda p: p.__setitem__("steinmetz_parameter_sets", [{"id": "set.neg", "k": {"value": -1.0, "unit": "W/m^3"}, "alpha": 1.5,
                                                                                "beta": 2.5, "basis": "synthetic"}]),
     "$.properties.steinmetz_parameter_sets[0].k.value", False),
    ("magnetic_material", lambda p: p.__setitem__("curie_temperature", {"value": {"value": 483.15, "unit": "K"}, "frequency": {"value": 0.0, "unit": "Hz"}}),
     "$.properties.curie_temperature.frequency.value", False),
    ("magnetic_material", lambda p: p.__setitem__("permeability_vs_temperature", [{"temperature": {"value": -1.0, "unit": "K"}, "value": {"value": 2000.0, "unit": "1"}}]),
     "$.properties.permeability_vs_temperature[0].temperature.value", False),
])
def test_physical_layer_rejects_impossible_values_with_exact_paths(kind, mutate, path, in_model):
    record = _fixture_v02(kind)
    mutate(record["properties"])
    errors = validate_record(record)
    assert any(e.startswith(f"[physical] {path}: must be") for e in errors), errors
    assert schema_errors(record) == [], "the JSON Schema cannot express sign rules on nested quantities"
    if in_model:
        assert semantic_errors(record), "records.py enforces this rule as well (v0.1 parity, commit 63ea968)"
    else:
        assert semantic_errors(record) == [], "gap in the normative model - reported to the records owner; the data layer still rejects"
        MagneticRecord.model_validate(record)  # the model alone accepts it; validate_record (all file-level paths) does not
    assert physical_errors(record) == [e for e in errors if e.startswith("[physical]")]


def test_physical_layer_covers_nested_material_points_and_steinmetz_sets():
    material = Catalog.default().get("material.example-synthetic-ferrite").to_json_dict()
    broken = copy.deepcopy(material)
    broken["properties"]["steinmetz_parameter_sets"][0]["k"] = {"value": -1.0, "unit": "W/m^3"}
    broken["properties"]["loss_reference_points"] = [{"frequency": {"value": 1e5, "unit": "Hz"}, "flux_density_peak": {"value": 0.1, "unit": "T"},
                                                      "loss_density": {"value": -5.0, "unit": "W/m^3"}, "temperature": {"value": 0.0, "unit": "K"}}]
    broken["properties"]["saturation_flux_density"] = [{"value": {"value": -0.4, "unit": "T"}, "temperature": {"value": 298.15, "unit": "K"},
                                                        "field": {"value": -1200.0, "unit": "A/m"}}]
    errors = physical_errors(broken)
    assert "[physical] $.properties.steinmetz_parameter_sets[0].k.value: must be positive, got -1.0 W/m^3" in errors
    assert "[physical] $.properties.loss_reference_points[0].loss_density.value: must be non-negative, got -5.0 W/m^3" in errors
    assert "[physical] $.properties.loss_reference_points[0].temperature.value: must be positive, got 0.0 K" in errors
    assert "[physical] $.properties.saturation_flux_density[0].value.value: must be positive, got -0.4 T" in errors
    assert not any("field" in e for e in errors), "a signed DC field condition carries no sign rule"
    assert validate_record(material) == [] and physical_errors(material) == []


def test_physical_layer_allows_signed_dc_quantities_and_zero_where_physical():
    condition = _fixture_v02("operating_condition")
    condition["properties"].update({"field_dc": {"value": -10.0, "unit": "A/m"}, "flux_density_dc": {"value": -0.05, "unit": "T"},
                                    "current_dc": {"value": -2.0, "unit": "A"}})
    assert validate_record(condition) == []
    core = _fixture_v02("core_geometry")
    core["properties"]["air_gap_options"] = [{"value": 0.0, "unit": "m"}]
    assert validate_record(core) == []
    excitation = {"schema_version": "0.2.0", "id": "excitation.dc-test", "kind": "excitation_waveform", "name": "dc", "synthetic": True,
                  "properties": {"waveform_type": "dc", "controlled_quantity": "field", "amplitude_peak": {"value": 0.0, "unit": "A/m"},
                                 "dc_offset": {"value": -300.0, "unit": "A/m"}},
                  "evidence_refs": [condition["evidence_refs"][0]], "evidence": condition["evidence"]}
    assert validate_record(excitation) == []
    excitation["properties"]["amplitude_peak"] = {"value": -1.0, "unit": "A/m"}
    assert any("amplitude_peak.value: must be non-negative" in e for e in validate_record(excitation))


def test_physical_layer_rejects_non_finite_numbers_anywhere(v01_example):
    material = Catalog.default().get("material.example-synthetic-ferrite").to_json_dict()
    for mutate, path in (
        (lambda r: r["properties"].__setitem__("recommended_frequency_range", {"min": float("nan"), "max": 1e6, "unit": "Hz"}),
         "$.properties.recommended_frequency_range.min"),
        (lambda r: r["properties"]["steinmetz_parameter_sets"][0].__setitem__("alpha", float("inf")), "$.properties.steinmetz_parameter_sets[0].alpha"),
        (lambda r: r["properties"]["steinmetz_parameter_sets"][0].__setitem__("fit_metrics", {"max_rel_error": float("nan")}),
         "$.properties.steinmetz_parameter_sets[0].fit_metrics.max_rel_error"),
        (lambda r: r.__setitem__("extensions", {"aipe.test": {"x": float("-inf")}}), "$.extensions.aipe.test.x"),
    ):
        broken = copy.deepcopy(material)
        mutate(broken)
        errors = validate_record(broken)
        assert any(e.startswith(f"[physical] {path}: value must be finite") for e in errors), (path, errors)
        assert any(f"[physical] {path}: value must be finite" in e for e in validate_catalogue([broken])), "catalogue validation applies the same layer"
        assert semantic_errors(broken) == [] or path.endswith("leakage_inductance.value"), "the normative model does not check non-Quantity floats"
    migrated = migrate_record(v01_example)
    migrated["properties"]["leakage_inductance"] = {"value": float("nan"), "unit": "H"}
    assert any("must be finite" in e for e in validate_record(migrated))
    assert physical_errors(["not", "a", "dict"]) == [] and physical_errors(MagneticRecord.model_validate(material)) == []


# -- file sets (--validate semantics) ---------------------------------------------------------------
def test_validate_files_treats_the_command_line_as_one_set(tmp_path: Path, v01_example):
    paths = v01_fixture_paths()
    per_file, set_errors = validate_files(paths)
    assert set_errors == [] and all(errors == [] for errors in per_file.values()) and len(per_file) == len(paths)
    measurement = next(p for p in paths if p.name == "measurement.json")
    per_file, set_errors = validate_files([measurement])
    assert set_errors == []
    assert per_file[str(measurement)] == [
        "measurement.fixture-v01-synthetic: unknown record reference inductor.fixture-v01-synthetic",
        "measurement.fixture-v01-synthetic: unknown record reference operating-condition.fixture-v01-synthetic",
    ], "a v0.1 file that references other files needs them on the command line; the message names the file"
    # a v0.2 record referencing the bundled catalogue passes with the default context and fails without it
    transformer = migrate_record(v01_example)
    transformer["properties"] = dict(transformer["properties"], core_ref="core.example-synthetic-e42")
    path = tmp_path / "transformer.json"
    dump(transformer, path)
    assert validate_files([path]) == ({str(path): []}, [])
    per_file, set_errors = validate_files([path], bundled_context=False)
    assert per_file[str(path)] == ["magnetic.example-transformer: unknown record reference core.example-synthetic-e42 (core_ref)"] and set_errors == []
    # bundled catalogue files validate themselves without a duplicate-id complaint; a copy under a new path does not
    bundled = [Catalog.default().source_of(i) for i in ("material.example-synthetic-ferrite", "core.example-synthetic-e42")]
    per_file, set_errors = validate_files(bundled)
    assert set_errors == [] and all(errors == [] for errors in per_file.values())
    copied = tmp_path / "copy.json"
    copied.write_bytes(bundled[0].read_bytes())
    per_file, _ = validate_files([copied])
    assert per_file[str(copied)] == ["Duplicate record ID: material.example-synthetic-ferrite"]
    # per-file layers, attribution of a v0.2 record's own errors, duplicate ids across files and unreadable files
    bad = tmp_path / "bad.json"
    dump(dict(transformer, id="magnetic.bad", name=""), bad)
    missing = tmp_path / "missing.json"
    per_file, set_errors = validate_files([path, bad, missing, path])
    assert any("[semantic] $.name" in e for e in per_file[str(bad)])
    assert per_file[str(missing)][0].startswith("cannot load")
    assert "Duplicate record ID: magnetic.example-transformer" in per_file[str(path)] and set_errors == []
    # mixed v0.1 / v0.2 set: the legacy group is checked with the legacy rules, duplicates across versions are reported
    legacy = tmp_path / "legacy.json"
    dump(v01_example, legacy)
    per_file, set_errors = validate_files([legacy, path], bundled_context=False)
    assert "Duplicate record ID across schema versions: magnetic.example-transformer" in per_file[str(legacy)]
    assert "Duplicate record ID across schema versions: magnetic.example-transformer" in per_file[str(path)]


def test_check_exported_shim_prints_and_returns_exit_codes(tmp_path: Path, capsys):
    out = tmp_path / "export"
    export_schemas(out)
    assert check_exported(out) == 0 and "OK" in capsys.readouterr().out
    (out / "quantity.schema.json").write_bytes(b"{}\n")
    assert check_exported(out) == 1
    printed = capsys.readouterr().out
    assert printed.startswith("STALE") and "quantity.schema.json: content differs" in printed
    assert check_exported(tmp_path / "nope") == 1


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
    capsys.readouterr()
    fixtures = [str(p) for p in v01_fixture_paths()]
    assert schema_module.main(["--validate", *fixtures]) == 0, "the fixture files form one consistent v0.1 set"
    printed = capsys.readouterr().out
    assert printed.count("PASS ") == len(fixtures) and "FAIL" not in printed
    measurement = next(f for f in fixtures if f.endswith("measurement.json"))
    assert schema_module.main(["--validate", measurement]) == 1
    printed = capsys.readouterr().out
    assert printed.startswith(f"FAIL {measurement}") and "unknown record reference inductor.fixture-v01-synthetic" in printed
    referencing = tmp_path / "transformer.json"
    record = migrate_record(load(v01_example_path))
    record["properties"] = dict(record["properties"], core_ref="core.example-synthetic-e42")
    dump(record, referencing)
    assert schema_module.main(["--validate", str(referencing)]) == 0
    assert schema_module.main(["--validate", str(referencing), "--no-bundled-catalogue"]) == 1
    assert "unknown record reference core.example-synthetic-e42" in capsys.readouterr().out


def test_repository_root_detection(repo_root: Path):
    assert repository_root() == repo_root
    files = schema_files_v02()
    assert files["magnetic-record.schema.json"]["$id"] == SCHEMA_ID_V02
