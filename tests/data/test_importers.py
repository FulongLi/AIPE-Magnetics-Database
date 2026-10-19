"""Importers: JSON records, CSV loss tables (explicit units), waveform sets; provenance, no overwrite, strict rejection."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from aipe_magnetics import __version__
from aipe_magnetics.common.errors import DataIntegrityError, InvalidInputError, UnitError, UnsupportedOperationError
from aipe_magnetics.common.jsonio import dump, load, sha256_file
from aipe_magnetics.data.catalog import Catalog
from aipe_magnetics.data.fixtures import v01_fixture_records
from aipe_magnetics.data.hdf5 import WaveformMeta, WaveformStore
from aipe_magnetics.data.importers import (
    IMPORT_EXTENSION,
    ImportResult,
    import_csv_loss_table,
    import_json_record,
    import_record,
    import_waveform_sequences,
    parse_loss_table,
)
from aipe_magnetics.data.importers import hdf5_waveforms as hdf5_waveforms_module
from aipe_magnetics.data.importers import json_record as json_record_module
from aipe_magnetics.data.importers.base import coerce_evidence, explicit_null_paths, relative_uri
from aipe_magnetics.data.migrate import migrate_record
from aipe_magnetics.data.records import MagneticRecord
from aipe_magnetics.data.schema import validate_catalogue, validate_record

COLUMNS = {"f_kHz": ("frequency", "kHz"), "B_mT": ("flux_density_peak", "mT"), "T_C": ("temperature", "degC"),
           "Pv": ("loss_density", "kW/m^3")}
CSV_TEXT = "f_kHz,B_mT,T_C,Pv,comment\n100,100,100,100,a\n200,50,,20,b\n50,200,100,250.5,c\n"


# -- JSON records -----------------------------------------------------------------------------------
def test_json_import_preserves_raw_bytes_records_digest_and_refuses_overwrite(tmp_path: Path, v01_example_path: Path):
    raw = v01_example_path.read_bytes()
    destination = tmp_path / "candidate.json"
    with pytest.raises(InvalidInputError, match="migrate_v01=True"):
        import_json_record(v01_example_path, destination)
    assert not destination.exists()
    result = import_json_record(v01_example_path, destination, migrate_v01=True)
    assert isinstance(result, ImportResult) and result.importer == "aipe.magnetics.import.json_record"
    extension = result.record.extensions[IMPORT_EXTENSION]
    assert extension["source_sha256"] == hashlib.sha256(raw).hexdigest() == result.source_sha256
    assert extension["source_filename"] == v01_example_path.name and extension["importer_version"] == __version__
    assert extension["normalization"] == "none; source must already use explicit SI units"
    assert result.warnings == ["record migrated from schema 0.1.0 to 0.2.0 during import"]
    assert result.written == [str(destination)]
    assert validate_record(load(destination)) == []
    assert raw == v01_example_path.read_bytes()
    with pytest.raises(FileExistsError):
        import_json_record(v01_example_path, destination, migrate_v01=True)
    with pytest.raises(InvalidInputError, match="Existing import provenance"):
        import_json_record(destination, tmp_path / "second.json")
    assert not (tmp_path / "second.json").exists()


def test_json_import_rejects_ambiguous_or_invalid_data_without_writing(tmp_path: Path):
    source, destination = tmp_path / "raw.json", tmp_path / "candidate.json"
    source.write_text('{"schema_version":"9.0.0"}', encoding="utf-8")
    with pytest.raises(InvalidInputError):
        import_json_record(source, destination)
    source.write_text('{"id":1,"id":2}', encoding="utf-8")
    with pytest.raises(InvalidInputError):
        import_json_record(source, destination)
    source.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(InvalidInputError):
        import_json_record(source, destination)
    record = migrate_record(v01_fixture_records()["winding"])
    record["properties"]["turns"] = {"value": 2.5, "unit": "1"}
    dump(record, source, overwrite=True)
    with pytest.raises(DataIntegrityError):
        import_json_record(source, destination)
    assert not destination.exists()
    with pytest.raises(InvalidInputError):
        import_json_record(tmp_path / "missing.json", destination)


def test_json_import_resolves_references_in_catalogue_context(tmp_path: Path, evidence_assumption):
    transformer = {
        "schema_version": "0.2.0", "id": "transformer.test-context", "kind": "transformer", "name": "context test", "synthetic": True,
        "properties": {"core_ref": "core.example-synthetic-e42", "material_ref": "material.example-synthetic-ferrite"},
        "evidence_refs": [evidence_assumption["id"]], "evidence": [evidence_assumption],
    }
    source = tmp_path / "transformer.json"
    dump(transformer, source)
    result = import_json_record(source)  # default context = bundled catalogue
    assert result.written == [] and result.record.kind == "transformer"
    with pytest.raises(DataIntegrityError, match="unknown record reference core.example-synthetic-e42"):
        import_json_record(source, catalogue=())
    extra = Catalog.default().with_records([])
    assert import_json_record(source, catalogue=extra).record.id == "transformer.test-context"
    assert import_json_record(source, catalogue=Catalog.default().records()).record.id == "transformer.test-context"
    with pytest.raises(InvalidInputError):
        import_json_record(source, catalogue=[42])  # type: ignore[list-item]


def test_json_import_cli(tmp_path: Path, v01_example_path: Path, capsys):
    out = tmp_path / "cli.json"
    assert json_record_module.main([str(v01_example_path), str(out)]) == 1
    assert "ERROR" in capsys.readouterr().out
    assert json_record_module.main([str(v01_example_path), str(out), "--migrate-v01"]) == 0
    assert "imported magnetic.example-transformer" in capsys.readouterr().out
    assert json_record_module.main([str(v01_example_path), str(out), "--migrate-v01"]) == 1


def test_json_import_rejects_explicit_nulls_without_writing(tmp_path: Path):
    record = Catalog.default().get("manufacturer.example-synthetic").to_json_dict()
    record["id"] = "manufacturer.test-null"
    record["properties"]["country"] = None
    record["description"] = None
    record["extensions"] = {"aipe.test": {"free_form": None}}
    source, destination = tmp_path / "manufacturer.json", tmp_path / "out.json"
    dump(record, source)
    assert set(explicit_null_paths(record)) == {"$.description", "$.properties.country"}, "extensions are free-form and not inspected"
    with pytest.raises(InvalidInputError, match=r"explicit null values are not allowed.*\$\.properties\.country"):
        import_json_record(source, destination)
    assert not destination.exists()
    del record["properties"]["country"], record["description"]
    dump(record, source, overwrite=True)
    assert import_json_record(source, destination).written == [str(destination)]
    written = load(destination)
    assert explicit_null_paths(written) == [], "no null outside the free-form extensions"
    assert written["extensions"]["aipe.test"] == {"free_form": None}, "extensions are copied verbatim (opaque, namespaced)"


def test_import_record_shim_accepts_path_catalogues_for_cli_consumers(tmp_path: Path, v01_example_path: Path, evidence_assumption):
    result = import_record(v01_example_path, tmp_path / "r.json", catalogue=[], migrate=True)
    assert result.written == [str(tmp_path / "r.json")] and validate_record(load(tmp_path / "r.json")) == []
    with pytest.raises(InvalidInputError, match="migrate_v01=True"):
        import_record(v01_example_path, tmp_path / "again.json")
    assert not (tmp_path / "again.json").exists()
    user_dir = tmp_path / "user"
    for kind, record in v01_fixture_records().items():  # the whole set, so the fixture core's family_ref resolves
        dump(migrate_record(record), user_dir / f"{kind}.json")
    transformer = {
        "schema_version": "0.2.0", "id": "transformer.test-shim", "kind": "transformer", "name": "shim", "synthetic": True,
        "properties": {"core_ref": "core.fixture-v01-synthetic", "material_ref": "material.example-synthetic-ferrite"},
        "evidence_refs": [evidence_assumption["id"]], "evidence": [evidence_assumption],
    }
    source = tmp_path / "transformer.json"
    dump(transformer, source)
    with pytest.raises(DataIntegrityError, match="unknown record reference core.fixture-v01-synthetic"):
        import_record(source)
    assert import_record(source, catalogue=[user_dir]).record.id == "transformer.test-shim", "path catalogue + bundled context"
    with pytest.raises(DataIntegrityError, match="unknown record reference material.example-synthetic-ferrite"):
        import_record(source, catalogue=[user_dir], bundled=False)


# -- CSV loss tables ---------------------------------------------------------------------------------
def test_csv_import_converts_explicit_units_and_keeps_originals(tmp_path: Path, evidence_measurement):
    csv = tmp_path / "losses.csv"
    csv.write_text(CSV_TEXT, encoding="utf-8")
    result = import_csv_loss_table(csv, columns=COLUMNS, record_id="measurement.test-losses", name="test losses",
                                   material_ref="material.example-synthetic-ferrite", evidence=evidence_measurement,
                                   destination=tmp_path / "out" / "losses.json", method="two-winding wattmeter",
                                   specimen_description="toroid (fixture)", tags=["test"])
    record = result.record
    assert record.kind == "core_loss_measurement" and record.synthetic is False and record.license == "NOASSERTION"
    points = record.properties["points"]
    assert len(points) == 3 and result.warnings == ["unmapped CSV columns ignored: ['comment']"]
    assert points[0]["frequency"] == {"value": 100000.0, "unit": "Hz", "original": {"value": 100.0, "unit": "kHz"}}
    assert points[0]["flux_density_peak"] == {"value": 0.1, "unit": "T", "original": {"value": 100.0, "unit": "mT"}}
    assert points[0]["temperature"] == {"value": 373.15, "unit": "K", "original": {"value": 100.0, "unit": "degC"}}
    assert points[0]["loss_density"] == {"value": 100000.0, "unit": "W/m^3", "original": {"value": 100.0, "unit": "kW/m^3"}}
    assert points[0]["basis"] == "measurement" and points[0]["source_ref"] == evidence_measurement["id"]
    assert "temperature" not in points[1], "empty optional cell is omitted, never zero-filled"
    assert points[2]["loss_density"]["value"] == 250500.0
    assert record.artifacts[0].media_type == "text/csv" and record.artifacts[0].sha256 == sha256_file(csv)
    extension = record.extensions[IMPORT_EXTENSION]
    assert extension["source_sha256"] == sha256_file(csv) and extension["rows"] == 3
    assert extension["columns"] == {h: {"quantity": q, "unit": u} for h, (q, u) in COLUMNS.items()}
    assert record.provenance is not None and "Quantity.si" in record.provenance.normalization
    assert record.properties["method"] == "two-winding wattmeter"
    written = load(tmp_path / "out" / "losses.json")
    assert validate_record(written) == [] and validate_catalogue([*Catalog.default().records(), written]) == []
    with pytest.raises(FileExistsError):
        import_csv_loss_table(csv, columns=COLUMNS, record_id="measurement.test-losses", name="x",
                              material_ref="material.example-synthetic-ferrite", evidence=evidence_measurement,
                              destination=tmp_path / "out" / "losses.json")


def test_csv_import_synthetic_tables_accept_assumption_evidence(tmp_path: Path, evidence_assumption):
    csv = tmp_path / "synthetic.csv"
    csv.write_text("f;B;P\n100000;0.1;100000\n", encoding="utf-8")
    result = import_csv_loss_table(csv, columns={"f": ("frequency", "Hz"), "B": ("flux_density_peak", "T"), "P": ("loss_density", "W/m^3")},
                                   record_id="measurement.test-synthetic", name="synthetic", material_ref="material.example-synthetic-ferrite",
                                   evidence=evidence_assumption, synthetic=True, delimiter=";", excitation="triangular", duty=0.5)
    point = result.record.properties["points"][0]
    assert point["basis"] == "synthetic" and point["excitation"] == "triangular" and point["duty"] == 0.5
    assert "original" not in point["frequency"], "already SI: no original recorded"
    assert any("locator" in w for w in result.warnings)


@pytest.mark.parametrize("label, text, columns, kwargs, error, fragment", [
    ("wrong dimension", CSV_TEXT, dict(COLUMNS, f_kHz=("frequency", "mT")), {}, UnitError, "not compatible"),
    ("unknown unit", CSV_TEXT, dict(COLUMNS, f_kHz=("frequency", "kilohertz")), {}, UnitError, "unknown unit"),
    ("unknown quantity", CSV_TEXT, dict(COLUMNS, comment=("humidity", "1")), {}, InvalidInputError, "unknown quantity"),
    ("missing required", CSV_TEXT, {"f_kHz": ("frequency", "kHz"), "B_mT": ("flux_density_peak", "mT")}, {}, InvalidInputError, "required quantities"),
    ("quantity twice", CSV_TEXT, dict(COLUMNS, comment=("frequency", "Hz")), {}, InvalidInputError, "mapped twice"),
    ("malformed mapping", CSV_TEXT, dict(COLUMNS, f_kHz="frequency"), {}, InvalidInputError, "(quantity, unit)"),
    ("empty mapping", CSV_TEXT, {}, {}, InvalidInputError, "non-empty"),
    ("header lacks column", CSV_TEXT.replace("Pv", "Loss"), COLUMNS, {}, InvalidInputError, "lacks mapped columns"),
    ("duplicate header", "f_kHz,B_mT,T_C,Pv,Pv\n1,1,1,1,1\n", COLUMNS, {}, InvalidInputError, "duplicate columns"),
    ("no rows", "f_kHz,B_mT,T_C,Pv\n", COLUMNS, {}, InvalidInputError, "no data rows"),
    ("empty file", "", COLUMNS, {}, InvalidInputError, "empty"),
    ("missing required cell", "f_kHz,B_mT,T_C,Pv\n100,100,100,\n", COLUMNS, {}, InvalidInputError, "required value is missing"),
    ("non numeric", "f_kHz,B_mT,T_C,Pv\n100,abc,100,1\n", COLUMNS, {}, InvalidInputError, "not a number"),
    ("non finite", "f_kHz,B_mT,T_C,Pv\n100,100,100,inf\n", COLUMNS, {}, InvalidInputError, "non-finite"),
    ("nan", "f_kHz,B_mT,T_C,Pv\n100,nan,100,1\n", COLUMNS, {}, InvalidInputError, "non-finite"),
    ("negative loss", "f_kHz,B_mT,T_C,Pv\n100,100,100,-1\n", COLUMNS, {}, InvalidInputError, "non-negative"),
    ("zero frequency", "f_kHz,B_mT,T_C,Pv\n0,100,100,1\n", COLUMNS, {}, InvalidInputError, "must be positive"),
    ("negative temperature", "f_kHz,B_mT,T_C,Pv\n100,100,-300,1\n", COLUMNS, {}, InvalidInputError, "must be positive"),
    ("short row", "f_kHz,B_mT,T_C,Pv\n100,100\n", COLUMNS, {}, InvalidInputError, "cells but"),
    ("long row", "f_kHz,B_mT,T_C,Pv\n100,100,100,100,999,888\n", COLUMNS, {}, InvalidInputError, "6 cells but 4 header columns"),
    ("trailing delimiter", "f_kHz,B_mT,T_C,Pv\n100,100,100,100,\n", COLUMNS, {}, InvalidInputError, "5 cells but 4 header columns"),
    ("digit separator", "f_kHz,B_mT,T_C,Pv\n1_00,100,100,100\n", COLUMNS, {}, InvalidInputError, "not a plain decimal numeral"),
    ("hex token", "f_kHz,B_mT,T_C,Pv\n0x10,100,100,100\n", COLUMNS, {}, InvalidInputError, "not a number"),
    ("overflow after conversion", "f_kHz,B_mT,T_C,Pv\n100,100,100,1e307\n", COLUMNS, {}, InvalidInputError, "overflows float64 when converted to W/m^3"),
    ("empty licence", CSV_TEXT, COLUMNS, {"license": ""}, InvalidInputError, "license must be a non-empty"),
    ("blank licence", CSV_TEXT, COLUMNS, {"license": "   "}, InvalidInputError, "license must be a non-empty"),
    ("source evidence", CSV_TEXT, COLUMNS, {"evidence": "SOURCE"}, InvalidInputError, "kind 'measurement'"),
    ("invalid evidence", CSV_TEXT, COLUMNS, {"evidence": {"id": "evidence.x"}}, InvalidInputError, "invalid evidence"),
    # rejected by the normative Evidence model (RFC 3339 validator) and, for records built from dicts, by the JSON Schema format
    ("non-RFC3339 created_at", CSV_TEXT, COLUMNS, {"evidence": "BAD_TIME"}, InvalidInputError, "date-time"),
    ("unknown material", CSV_TEXT, COLUMNS, {"material_ref": "material.nope"}, DataIntegrityError, "unknown record reference"),
    ("wrong material kind", CSV_TEXT, COLUMNS, {"material_ref": "core.example-synthetic-e42"}, DataIntegrityError, "wrong record kind"),
    ("bad record id", CSV_TEXT, COLUMNS, {"record_id": "Bad Id"}, InvalidInputError, "invalid record"),
])
def test_csv_import_rejects_bad_tables_and_arguments(tmp_path: Path, evidence_measurement, evidence_source, label, text, columns, kwargs, error, fragment):
    csv = tmp_path / "table.csv"
    csv.write_text(text, encoding="utf-8")
    args = dict(columns=columns, record_id="measurement.test-reject", name="reject", material_ref="material.example-synthetic-ferrite",
                evidence=evidence_measurement, destination=tmp_path / "never.json")
    if kwargs.get("evidence") == "SOURCE":
        kwargs = dict(kwargs, evidence=evidence_source)
    if kwargs.get("evidence") == "BAD_TIME":
        kwargs = dict(kwargs, evidence=dict(evidence_measurement, created_at="yesterday afternoon"))
    args.update(kwargs)
    with pytest.raises(error) as info:
        import_csv_loss_table(csv, **args)
    assert fragment in str(info.value), (label, str(info.value))
    assert not (tmp_path / "never.json").exists()


def test_csv_import_output_passes_every_validation_layer_and_loads_as_catalogue(tmp_path: Path, evidence_measurement):
    csv = tmp_path / "losses.csv"
    csv.write_text("﻿" + CSV_TEXT, encoding="utf-8")  # a UTF-8 BOM in front of the header is removed and reported
    result = import_csv_loss_table(csv, columns=COLUMNS, record_id="measurement.test-layers", name="layers",
                                   material_ref="material.example-synthetic-ferrite", evidence=evidence_measurement,
                                   destination=tmp_path / "out" / "losses.json")
    assert any("byte-order mark" in w for w in result.warnings)
    written = load(tmp_path / "out" / "losses.json")
    assert validate_record(written) == [], "schema + semantic + physical layers"
    assert Catalog.load([tmp_path / "out"]).get("measurement.test-layers").kind == "core_loss_measurement"


def test_parse_loss_table_direct():
    rows, warnings = parse_loss_table("f,B,P\n1000,0.2,50\n\n2000,0.1,0\n", {"f": ("frequency", "Hz"), "B": ("flux_density_peak", "T"), "P": ("loss_density", "W/m^3")})
    assert len(rows) == 2 and rows[1]["loss_density"].value == 0.0
    assert any("zero loss" in w for w in warnings)
    evidence = coerce_evidence({"id": "evidence.x", "kind": "source", "reference": "r", "description": "d",
                                "created_at": "2026-10-19T00:00:00Z", "review_status": "unreviewed", "license": "MIT"})
    assert coerce_evidence(evidence) is evidence
    with pytest.raises(InvalidInputError):
        coerce_evidence("not evidence")  # type: ignore[arg-type]


# -- waveform sets ----------------------------------------------------------------------------------
@pytest.fixture
def waveform_source(tmp_path: Path):
    n_seq, n_samp = 4, 64
    t = np.arange(n_samp) / n_samp
    b = np.stack([(k + 1) * 50.0 * np.sin(2 * np.pi * t) for k in range(n_seq)])
    f = np.array([50.0, 100.0, 50.0, 100.0]).reshape(-1, 1)
    np.savez(tmp_path / "src.npz", B_mT=b, f_kHz=f)
    csv_dir = tmp_path / "csvdir"
    csv_dir.mkdir()
    np.savetxt(csv_dir / "B.csv", b, delimiter=",")
    np.savetxt(csv_dir / "f.csv", f.ravel(), delimiter=",")
    return {"b": b, "f": f, "npz": tmp_path / "src.npz", "csv_dir": csv_dir,
            "meta": WaveformMeta(source_id="user.upload", dataset_version="2026-10-19", license="NOASSERTION")}


@pytest.mark.usefixtures("require_h5py")
def test_waveform_import_from_npz_writes_layout_and_record(tmp_path: Path, waveform_source, evidence_source):
    out = tmp_path / "out"
    result = import_waveform_sequences(waveform_source["npz"], out / "wf.h5", arrays={"flux_density": ("B_mT", "mT")},
                                       conditions={"frequency": ("f_kHz", "kHz")}, meta=waveform_source["meta"],
                                       record_id="waveform-dataset.test", name="user waveforms", evidence=evidence_source,
                                       destination_record=out / "wf.json", catalogue=())
    record = result.record
    assert record.kind == "waveform_dataset" and record.synthetic is False
    assert result.warnings == ["condition 'frequency': column array flattened to one entry per sequence"]
    assert [(a.uri, a.role, a.media_type) for a in record.artifacts] == [
        ("wf.h5", "waveform_data", "application/x-hdf5"), ("../src.npz", "raw", "application/octet-stream")]
    assert record.artifacts[0].sha256 == sha256_file(out / "wf.h5") and record.artifacts[1].sha256 == sha256_file(waveform_source["npz"])
    extension = record.extensions[IMPORT_EXTENSION]
    assert extension["source_sha256"] == sha256_file(waveform_source["npz"]) and extension["source_format"] == "npz"
    assert extension["arrays"] == {"flux_density": {"key": "B_mT", "unit": "mT"}}
    assert extension["conditions"] == {"frequency": {"key": "f_kHz", "unit": "kHz"}}
    data = WaveformStore().read(out / "wf.h5")
    assert data.array_original_units["flux_density"] == "mT" and data.conditions["frequency"].tolist() == [50000.0, 100000.0, 50000.0, 100000.0]
    assert np.allclose(data.arrays["flux_density"], waveform_source["b"] * 1e-3)
    written = load(out / "wf.json")
    assert validate_record(written) == [] and MagneticRecord.model_validate(written).typed_properties().n_sequences == 4
    assert sorted(result.written) == sorted([str(out / "wf.h5"), str(out / "wf.json")])
    with pytest.raises(FileExistsError):
        import_waveform_sequences(waveform_source["npz"], out / "wf.h5", arrays={"flux_density": ("B_mT", "mT")},
                                  conditions={"frequency": ("f_kHz", "kHz")}, meta=waveform_source["meta"],
                                  record_id="waveform-dataset.test", name="x", evidence=evidence_source, catalogue=())


@pytest.mark.usefixtures("require_h5py")
def test_waveform_import_from_csv_directory_and_hdf5_source(tmp_path: Path, waveform_source, evidence_source):
    import h5py

    result = import_waveform_sequences(waveform_source["csv_dir"], tmp_path / "csv.h5", arrays={"flux_density": ("B.csv", "mT")},
                                       conditions={"frequency": ("f.csv", "kHz")}, meta=waveform_source["meta"],
                                       record_id="waveform-dataset.csv", name="csv", evidence=evidence_source, catalogue=())
    extension = result.record.extensions[IMPORT_EXTENSION]
    assert set(extension["source_files"]) == {"B.csv", "f.csv"} and extension["source_format"] == "csv"
    assert extension["source_sha256_method"].startswith("sha256 of the canonical JSON")
    assert {a.uri for a in result.record.artifacts if a.role == "raw"} == {"B.csv", "f.csv"}
    foreign = tmp_path / "foreign.h5"
    with h5py.File(foreign, "w") as handle:
        handle.create_dataset("grp/B_seq", data=waveform_source["b"])
        handle.create_dataset("grp/T_C", data=np.full((4, 1), 25.0))
    result = import_waveform_sequences(foreign, tmp_path / "h5.h5", arrays={"flux_density": ("grp/B_seq", "mT")},
                                       conditions={"temperature": ("grp/T_C", "degC"), "frequency": ("grp/T_C", "kHz")},
                                       meta=waveform_source["meta"], record_id="waveform-dataset.h5", name="h5",
                                       evidence=evidence_source, catalogue=(), source_format="hdf5")
    data = WaveformStore().read(tmp_path / "h5.h5")
    assert data.conditions["temperature"].tolist() == [298.15] * 4
    single = tmp_path / "single.csv"
    np.savetxt(single, waveform_source["b"], delimiter=",")
    result = import_waveform_sequences(single, tmp_path / "single.h5", arrays={"flux_density": ("single.csv", "T")},
                                       meta=WaveformMeta(source_id="u", dataset_version="1", timebase="uniform_absolute",
                                                         sampling_rate={"value": 1.0, "unit": "MHz"}),  # type: ignore[arg-type]
                                       record_id="waveform-dataset.single", name="single", evidence=evidence_source, catalogue=())
    assert result.record.typed_properties().sampling_rate.value == 1e6


@pytest.mark.usefixtures("require_h5py")
def test_waveform_import_rejections_leave_nothing_behind(tmp_path: Path, waveform_source, evidence_source):
    common = dict(meta=waveform_source["meta"], record_id="waveform-dataset.x", name="x", evidence=evidence_source, catalogue=())
    target = tmp_path / "never.h5"
    with pytest.raises(InvalidInputError, match="not found in src.npz"):
        import_waveform_sequences(waveform_source["npz"], target, arrays={"flux_density": ("nope", "mT")}, **common)
    with pytest.raises(InvalidInputError, match="container format"):
        import_waveform_sequences(tmp_path / "data.bin", target, arrays={"flux_density": ("B", "mT")}, **common)
    with pytest.raises(InvalidInputError, match="source_format"):
        import_waveform_sequences(waveform_source["npz"], target, arrays={"flux_density": ("B_mT", "mT")}, source_format="mat", **common)
    with pytest.raises(InvalidInputError):
        import_waveform_sequences(waveform_source["npz"], target, arrays={}, **common)
    with pytest.raises(InvalidInputError):
        import_waveform_sequences(waveform_source["npz"], target, arrays={"flux_density": "B_mT"}, **common)
    with pytest.raises(UnitError):
        import_waveform_sequences(waveform_source["npz"], target, arrays={"flux_density": ("B_mT", "A/m")}, **common)
    with pytest.raises(InvalidInputError, match="CSV file 'nope.csv' not found"):
        import_waveform_sequences(waveform_source["csv_dir"], target, arrays={"flux_density": ("nope.csv", "mT")}, **common)
    with pytest.raises(InvalidInputError, match="source not found"):
        import_waveform_sequences(tmp_path / "missing_dir", target, arrays={"flux_density": ("B.csv", "mT")}, source_format="csv", **common)
    assert not target.exists()
    # cross-reference failure: detected before the HDF5 file is written
    with pytest.raises(DataIntegrityError, match="unknown dataset_ref"):
        import_waveform_sequences(waveform_source["npz"], target, arrays={"flux_density": ("B_mT", "mT")},
                                  conditions={"frequency": ("f_kHz", "kHz")}, dataset_ref="dataset.nope", **common)
    assert not target.exists()


@pytest.mark.usefixtures("require_h5py")
@pytest.mark.parametrize("label, overrides, error, fragment", [
    ("bad record id", {"record_id": "Bad Id"}, InvalidInputError, "must match"),
    ("empty name", {"name": ""}, InvalidInputError, "name"),
    ("assumption evidence on a real record", {"evidence": "ASSUMPTION"}, InvalidInputError, "non-assumption source evidence"),
    ("non-RFC3339 created_at", {"evidence": "BAD_TIME"}, InvalidInputError, "date-time"),
    ("unknown material_ref in meta", {"meta": "BAD_MATERIAL"}, DataIntegrityError, "unknown record reference material.nope"),
])
def test_waveform_import_validates_the_record_before_writing_anything(tmp_path: Path, waveform_source, evidence_source, evidence_assumption,
                                                                      label, overrides, error, fragment):
    out = tmp_path / "out"
    args = dict(arrays={"flux_density": ("B_mT", "mT")}, conditions={"frequency": ("f_kHz", "kHz")}, meta=waveform_source["meta"],
                record_id="waveform-dataset.late", name="late failure", evidence=evidence_source, destination_record=out / "late.json",
                catalogue=())
    if overrides.get("evidence") == "ASSUMPTION":
        overrides = dict(overrides, evidence=evidence_assumption)
    if overrides.get("evidence") == "BAD_TIME":
        overrides = dict(overrides, evidence=dict(evidence_source, created_at="nope"))
    if overrides.get("meta") == "BAD_MATERIAL":
        overrides = dict(overrides, meta=WaveformMeta(source_id="user.upload", dataset_version="1", material_ref="material.nope"))
    args.update(overrides)
    with pytest.raises(error, match=fragment):
        import_waveform_sequences(waveform_source["npz"], out / "late.h5", **args)
    assert not (out / "late.h5").exists(), f"{label}: no orphan waveform file"
    assert not (out / "late.json").exists(), f"{label}: no record written"


@pytest.mark.usefixtures("require_h5py")
def test_waveform_import_refuses_taken_destinations_before_doing_any_work(tmp_path: Path, waveform_source, evidence_source):
    out = tmp_path / "out"
    out.mkdir()
    (out / "taken.json").write_text("{}", encoding="utf-8")
    common = dict(arrays={"flux_density": ("B_mT", "mT")}, conditions={"frequency": ("f_kHz", "kHz")}, meta=waveform_source["meta"],
                  record_id="waveform-dataset.taken", name="taken", evidence=evidence_source, catalogue=())
    with pytest.raises(FileExistsError, match="taken.json"):
        import_waveform_sequences(waveform_source["npz"], out / "new.h5", destination_record=out / "taken.json", **common)
    assert not (out / "new.h5").exists()
    (out / "taken.h5").write_bytes(b"not hdf5")
    with pytest.raises(FileExistsError, match="overwrite_h5=True"):
        import_waveform_sequences(waveform_source["npz"], out / "taken.h5", destination_record=out / "fresh.json", **common)
    assert (out / "taken.h5").read_bytes() == b"not hdf5" and not (out / "fresh.json").exists()
    result = import_waveform_sequences(waveform_source["npz"], out / "taken.h5", destination_record=out / "fresh.json", overwrite_h5=True, **common)
    assert WaveformStore().verify_checksums(out / "taken.h5") == [] and validate_record(load(out / "fresh.json")) == []
    assert result.record.artifacts[0].sha256 == sha256_file(out / "taken.h5")


@pytest.mark.usefixtures("require_h5py")
def test_waveform_import_rolls_back_the_waveform_file_when_the_record_cannot_be_written(tmp_path: Path, waveform_source, evidence_source, monkeypatch):
    out = tmp_path / "out"
    common = dict(arrays={"flux_density": ("B_mT", "mT")}, conditions={"frequency": ("f_kHz", "kHz")}, meta=waveform_source["meta"],
                  record_id="waveform-dataset.rollback", name="rollback", evidence=evidence_source, catalogue=())

    def failing_write_record(record, destination):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(hdf5_waveforms_module, "write_record", failing_write_record)
    with pytest.raises(OSError, match="disk full"):
        import_waveform_sequences(waveform_source["npz"], out / "wf.h5", destination_record=out / "wf.json", **common)
    assert not (out / "wf.h5").exists() and not (out / "wf.json").exists()


def test_relative_uri_falls_back_to_a_file_uri_across_drives(tmp_path: Path):
    import os

    assert relative_uri(tmp_path / "data" / "wf.h5", tmp_path / "records" / "r.json") == "../data/wf.h5"
    assert relative_uri(tmp_path / "wf.h5", None) == "wf.h5"
    uri = relative_uri(Path("Z:/data/x.h5"), Path("D:/records/r.json"))
    if os.name == "nt":
        assert uri == "file:///Z:/data/x.h5", "no relative path exists between drives: absolute file URI instead of a ValueError"
    else:
        assert uri.endswith("Z:/data/x.h5") and not uri.startswith("file:")


def test_waveform_import_without_h5py_is_an_explicit_unsupported_operation(monkeypatch, tmp_path: Path, waveform_source, evidence_source):
    from aipe_magnetics.data import hdf5 as hdf5_module

    monkeypatch.setattr(hdf5_module, "h5py", None)
    monkeypatch.setattr(hdf5_waveforms_module, "h5py_available", lambda: False)
    with pytest.raises(UnsupportedOperationError):
        import_waveform_sequences(tmp_path / "foreign.h5", tmp_path / "x.h5", arrays={"flux_density": ("B", "mT")},
                                  meta=waveform_source["meta"], record_id="waveform-dataset.x", name="x", evidence=evidence_source, catalogue=())
    with pytest.raises(UnsupportedOperationError):
        import_waveform_sequences(waveform_source["npz"], tmp_path / "x.h5", arrays={"flux_density": ("B_mT", "mT")},
                                  conditions={"frequency": ("f_kHz", "kHz")}, meta=waveform_source["meta"],
                                  record_id="waveform-dataset.x", name="x", evidence=evidence_source, catalogue=())
    assert not (tmp_path / "x.h5").exists()
