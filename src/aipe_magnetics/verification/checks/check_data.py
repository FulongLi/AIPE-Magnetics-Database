"""Verification checks of the data layer (schema, migration, HDF5 store, unit normalisation, catalogue, importers).

Levels: L1 = software / schema correctness; the unit-normalisation check is L2 because its expected
values are analytical definitions (SI prefixes, ``T/K = t/degC + 273.15``, ``1 Oe = 1000/(4 pi) A/m``).
Every check is deterministic, offline, fast, and touches only temporary directories.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from aipe_magnetics.common.errors import DataIntegrityError, DataNotFoundError, InvalidInputError
from aipe_magnetics.common.jsonio import dump, load, sha256_file
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import SourceReference
from aipe_magnetics.common.units import units
from aipe_magnetics.data.catalog import Catalog
from aipe_magnetics.data.fixtures import V01_KINDS, v01_fixture_records
from aipe_magnetics.data.hdf5 import WaveformMeta, WaveformStore, array_sha256, convert_array, h5py_available
from aipe_magnetics.data.migrate import migrate_record
from aipe_magnetics.data.records import PROPERTY_MODELS, MagneticRecord
from aipe_magnetics.data.schema import (
    SCHEMA_ID_V02,
    SCHEMA_TITLE_V02,
    check_exported_schemas,
    compare_directories,
    generate_schema_v02,
    generator_versions,
    mirror_schema_dir,
    package_schema_dir,
    repository_root,
    schema_files_v02,
    validate_catalogue,
    validate_legacy_records,
    validate_record,
)
from aipe_magnetics.verification.suite import CheckResult, check

_SI_BROCHURE = SourceReference(
    id="ref.bipm-si-brochure-9", kind="standard",
    reference="BIPM, The International System of Units (SI), 9th edition (2019): SI prefixes; kelvin / degree Celsius relation",
    locator="Chapter 3 (prefixes) and Section 2.3.1 (degree Celsius)", url="https://www.bipm.org/en/publications/si-brochure",
    license="NOASSERTION")
_NIST_SP811 = SourceReference(
    id="ref.nist-sp811-2008", kind="standard",
    reference="A. Thompson, B. N. Taylor, Guide for the Use of the International System of Units (SI), NIST Special Publication 811, 2008 edition",
    locator="Appendix B.9: oersted -> ampere per meter, factor 7.957 747 E+01 (= 1000/(4 pi))",
    url="https://www.nist.gov/pml/special-publication-811", license="NOASSERTION")


def _collect_refs(node: Any, out: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                out.append(value)
            else:
                _collect_refs(value, out)
    elif isinstance(node, list):
        for item in node:
            _collect_refs(item, out)


@check(
    id="data.schema-v02-is-valid-draft2020", capability="aipe.magnetics.data.schema", level="L1", kind="schema",
    description="The generated v0.2 JSON Schema (and every per-kind / quantity schema) is valid draft 2020-12, "
                "self-contained, binds every record kind, and agrees with the packaged copy.",
    method="Draft202012Validator.check_schema on every generated document; structural assertions on $id, title, "
           "required keys, allOf/if/then per PROPERTY_MODELS kind, local-only $ref resolution; byte comparison of the "
           "packaged files with a fresh export; a bundled record validates and a corrupted one is rejected.")
def schema_v02_is_valid_draft2020() -> CheckResult:
    notes: list[str] = []
    problems: list[str] = []
    files = schema_files_v02()
    for relative, document in files.items():
        try:
            Draft202012Validator.check_schema(document)
        except Exception as error:  # noqa: BLE001 - reported as a failing check
            problems.append(f"{relative}: not a valid draft 2020-12 schema: {error}")
        refs: list[str] = []
        _collect_refs(document, refs)
        remote = [r for r in refs if not r.startswith("#/")]
        if remote:
            problems.append(f"{relative}: non-local $ref {remote}")
        defs = document.get("$defs", {})
        dangling = sorted({r for r in refs if r.startswith("#/$defs/") and r.split("/")[-1] not in defs})
        if dangling:
            problems.append(f"{relative}: dangling $ref {dangling}")
    schema = files["magnetic-record.schema.json"]
    if schema.get("$id") != SCHEMA_ID_V02:
        problems.append(f"$id is {schema.get('$id')!r}")
    if schema.get("title") != SCHEMA_TITLE_V02:
        problems.append(f"title is {schema.get('title')!r}")
    if "schema_version" not in schema.get("required", []):
        problems.append("schema_version is not required")
    bound = [block["if"]["properties"]["kind"]["const"] for block in schema.get("allOf", [])]
    if bound != list(PROPERTY_MODELS):
        problems.append(f"allOf binds {bound}, expected {list(PROPERTY_MODELS)}")
    stale = check_exported_schemas(package_schema_dir())
    if stale:
        problems.append("packaged schema files differ from a fresh export: " + "; ".join(stale))
    sample = Catalog.default().get("material.example-synthetic-ferrite").to_json_dict()
    if validate_record(sample):
        problems.append("bundled material record does not validate")
    broken = dict(sample)
    broken["properties"] = dict(sample["properties"], not_a_field=1)
    if not validate_record(broken):
        problems.append("unknown property field was not rejected")
    negative = dict(sample)
    negative["properties"] = dict(sample["properties"], relative_permeability={"value": -1.0, "unit": "1"})
    if not any(error.startswith("[physical]") for error in validate_record(negative)):
        problems.append("negative permeability was not rejected by the physical layer")
    versions = generator_versions()
    notes.append(f"{len(files)} schema documents checked; {len(schema['$defs'])} shared definitions; generated with "
                 + ", ".join(f"{k} {v}" for k, v in versions.items()))
    return CheckResult(passed=not problems, metrics={"documents": len(files), "definitions": len(schema["$defs"]),
                                                     "kinds_bound": len(bound), "problems": len(problems),
                                                     "generator_versions": versions},
                       notes=notes + problems)


@check(
    id="data.schema-mirror-in-sync", capability="aipe.magnetics.data.schema", level="L1", kind="software",
    description="The packaged schema files equal a fresh export and the repository mirror schemas/v0.2 is byte-identical "
                "to the packaged canonical copy (the mirror comparison runs only in a source checkout).",
    method="check_exported_schemas(package dir) and compare_directories(package dir, repository schemas/v0.2).")
def schema_mirror_in_sync() -> CheckResult:
    problems = list(check_exported_schemas(package_schema_dir()))
    mirror = mirror_schema_dir()
    mirror_compared = False
    notes: list[str] = []
    if mirror is None:
        notes.append("no repository checkout detected: mirror comparison not applicable (packaged files compared to a fresh export only)")
    else:
        mirror_compared = True
        problems.extend(compare_directories(package_schema_dir(), mirror))
        if problems:
            notes.append("refresh with: python -m aipe_magnetics.data.schema --export schemas/v0.2 --export-package")
    return CheckResult(passed=not problems, metrics={"mirror_compared": mirror_compared, "differences": len(problems)},
                       notes=notes + problems)


@check(
    id="data.v01-examples-migrate-and-validate", capability="aipe.magnetics.data.migration", level="L1", kind="software",
    description="Every v0.1 record (repository examples and the packaged synthetic fixture of every v0.1 kind) validates "
                "under the legacy rules, migrates mechanically to schema 0.2.0, validates under v0.2 and keeps its meaning.",
    method="validate_legacy_records on the inputs; migrate_record; validate_record + validate_catalogue on the outputs; "
           "field-by-field equality of id, kind, name, synthetic, evidence, evidence_refs and (except measurement.tool) properties.")
def v01_examples_migrate_and_validate() -> CheckResult:
    problems: list[str] = []
    sources: dict[str, dict[str, Any]] = {}
    fixtures = v01_fixture_records()
    missing_kinds = sorted(set(V01_KINDS) - set(fixtures))
    if missing_kinds:
        problems.append(f"no packaged v0.1 fixture for kinds {missing_kinds}")
    for kind, record in fixtures.items():
        sources[f"fixture:{kind}"] = record
    root = repository_root()
    if root is not None:
        for path in sorted((root / "examples").glob("*.json")):
            data = load(path)
            if isinstance(data, dict) and data.get("schema_version") == "0.1.0":
                sources[f"example:{path.name}"] = data
    legacy_errors = validate_legacy_records(list(fixtures.values()))
    if legacy_errors:
        problems.append("fixture set fails legacy validation: " + "; ".join(legacy_errors))
    migrated: list[dict[str, Any]] = []
    for label, record in sources.items():
        try:
            out = migrate_record(record)
        except Exception as error:  # noqa: BLE001
            problems.append(f"{label}: migration raised {type(error).__name__}: {error}")
            continue
        errors = validate_record(out)
        if errors:
            problems.append(f"{label}: migrated record invalid: {'; '.join(errors)}")
        for key in ("id", "kind", "name", "synthetic", "evidence", "evidence_refs"):
            if out.get(key) != record.get(key):
                problems.append(f"{label}: {key} changed during migration")
        if record["kind"] != "measurement" and out["properties"] != record["properties"]:
            problems.append(f"{label}: properties changed during migration")
        if out.get("extensions", {}).get("aipe.migration", {}).get("from") != "0.1.0":
            problems.append(f"{label}: aipe.migration extension missing")
        if label.startswith("fixture:"):
            migrated.append(out)
    cross = validate_catalogue(migrated)
    if cross:
        problems.append("migrated fixture catalogue has cross-reference errors: " + "; ".join(cross))
    return CheckResult(passed=not problems, metrics={"records_migrated": len(sources) - sum("raised" in p for p in problems),
                                                     "v01_kinds": len(fixtures), "repository_examples": sum(k.startswith("example:") for k in sources)},
                       notes=problems or [f"{len(sources)} v0.1 records migrated and validated"])


@check(
    id="data.hdf5-roundtrip-checksums", capability="aipe.magnetics.data.hdf5", level="L1", kind="software",
    description="WaveformStore writes the aipe.magnetics.waveforms/0.2 layout with SI normalisation, NaN missing values and "
                "per-array SHA-256; reading reproduces the arrays bit-for-bit, checksums verify, array-content tampering is "
                "detected (attributes are covered only by the file-level artifact SHA-256), identical inputs give identical "
                "files, describe() reproduces the write result including a converted sampling rate, and values that overflow "
                "during unit conversion are rejected.",
    method="Write a 6x128 sequence set (mT, kHz, degC with one NaN) twice into a temporary directory; compare file "
           "hashes; read back; compare arrays (equal_nan) with scalar conversions; verify_checksums; corrupt one sample and "
           "expect a reported mismatch; describe() equals the write result (also for a 1 MHz uniform_absolute file); "
           "a 1e306 kA/m field and a shape mismatch raise InvalidInputError and write nothing.")
def hdf5_roundtrip_checksums() -> CheckResult:
    if not h5py_available():
        return CheckResult(passed=False, notes=["h5py is not installed (install aipe-magnetics[hdf5]); the HDF5 round-trip "
                                                "was NOT executed, so no evidence exists for this capability"])
    import h5py  # type: ignore

    problems: list[str] = []
    n_seq, n_samp = 6, 128
    t = np.arange(n_samp) / n_samp
    b_mt = np.stack([100.0 * (k + 1) * np.sin(2 * np.pi * t) for k in range(n_seq)])
    b_mt[2, 5] = np.nan
    f_khz = np.array([50.0, 100.0, 200.0, 50.0, 100.0, 200.0])
    t_degc = np.array([25.0, 25.0, 25.0, 100.0, 100.0, np.nan])
    store = WaveformStore()
    meta = WaveformMeta(source_id="aipe.verification", dataset_version="check", license="NOASSERTION")
    with tempfile.TemporaryDirectory() as tmp:
        p1, p2 = Path(tmp) / "a.h5", Path(tmp) / "b.h5"
        props = store.write(p1, {"flux_density": (b_mt, "mT")}, {"frequency": (f_khz, "kHz"), "temperature": (t_degc, "degC")}, meta=meta)
        store.write(p2, {"flux_density": (b_mt, "mT")}, {"frequency": (f_khz, "kHz"), "temperature": (t_degc, "degC")}, meta=meta)
        deterministic = sha256_file(p1) == sha256_file(p2)
        if not deterministic:
            problems.append("two writes of identical inputs produced different file bytes")
        data = store.read(p1)
        expected = convert_array(b_mt, "mT", "T")
        if not np.array_equal(data.arrays["flux_density"], expected, equal_nan=True):
            problems.append("flux_density read back differs from the SI conversion")
        scalar = np.array([units.convert(v, "degC", "K") if not math.isnan(v) else math.nan for v in t_degc])
        if not np.array_equal(data.conditions["temperature"], scalar, equal_nan=True):
            problems.append("temperature condition differs from scalar degC->K conversion")
        if data.array_original_units.get("flux_density") != "mT" or data.array_units.get("flux_density") != "T":
            problems.append("original / SI unit attributes not retained")
        if int(np.isnan(data.arrays["flux_density"]).sum()) != 1 or not bool(np.isnan(data.conditions["temperature"][5])):
            problems.append("missing values were not preserved as NaN")
        mismatches = store.verify_checksums(p1)
        if mismatches:
            problems.append(f"fresh file reports checksum mismatches: {mismatches}")
        if store.describe(p1) != props:
            problems.append("describe() differs from the properties returned by write()")
        if data.checksums.get("sequences/flux_density") != array_sha256(expected):
            problems.append("stored checksum differs from array_sha256 of the SI array")
        with h5py.File(p1, "r+") as handle:
            handle["sequences/flux_density"][0, 0] = 12345.0
        tampered = store.verify_checksums(p1)
        if not any(m.startswith("sequences/flux_density") for m in tampered):
            problems.append("tampering was not detected")
        try:
            store.read(p1)
            problems.append("read() accepted a tampered file")
        except DataIntegrityError:
            pass
        try:
            store.write(Path(tmp) / "c.h5", {"flux_density": (b_mt, "mT"), "field": (b_mt[:, :64], "A/m")}, None, meta=meta)
            problems.append("shape mismatch was not rejected")
        except InvalidInputError:
            pass
        overflow = np.where(np.isnan(b_mt), 0.0, b_mt)
        overflow[0, 0] = 1e306
        try:
            store.write(Path(tmp) / "d.h5", {"field": (overflow, "kA/m")}, {"frequency": (f_khz, "kHz")}, meta=meta)
            problems.append("overflow to inf during unit conversion was not rejected")
        except InvalidInputError:
            pass
        if (Path(tmp) / "c.h5").exists() or (Path(tmp) / "d.h5").exists():
            problems.append("a rejected write left a file behind")
        absolute = WaveformMeta(source_id="aipe.verification", dataset_version="check", timebase="uniform_absolute",
                                sampling_rate=Quantity.si(1.0, "MHz"))
        props_abs = store.write(Path(tmp) / "e.h5", {"current": (np.where(np.isnan(b_mt), 0.0, b_mt), "mA")}, None, meta=absolute)
        described = store.describe(Path(tmp) / "e.h5")
        if described != props_abs or described.sampling_rate is None or described.sampling_rate.original is None \
                or described.sampling_rate.original.unit != "MHz":
            problems.append("describe() does not reproduce the converted sampling rate with its original")
    return CheckResult(passed=not problems, metrics={"n_sequences": n_seq, "n_samples": n_samp, "deterministic_bytes": deterministic,
                                                     "tamper_detected": not any("tamper" in p for p in problems)},
                       notes=problems or ["round trip, checksums, array-content tamper detection, determinism, describe() "
                                          "reproduction and overflow rejection verified"])


_UNIT_CASES: tuple[tuple[float, str, str, float], ...] = (
    (25.0, "degC", "K", 298.15),
    (100.0, "degC", "K", 373.15),
    (-273.15, "degC", "K", 0.0),
    (100.0, "mT", "T", 0.1),
    (200.0, "mT", "T", 0.2),
    (10.0, "kG", "T", 1.0),
    (100.0, "kW/m^3", "W/m^3", 1.0e5),
    (100.0, "mW/cm^3", "W/m^3", 1.0e5),
    (1.0, "Oe", "A/m", 1000.0 / (4.0 * math.pi)),
    (100.0, "kHz", "Hz", 1.0e5),
    (1.0, "kA/m", "A/m", 1000.0),
    (2.0, "mm^2", "m^2", 2.0e-6),
)


@check(
    id="data.unit-normalisation-reproducible", capability="aipe.magnetics.data.units", level="L2", kind="analytical_reference",
    description="Unit normalisation reproduces analytically defined conversion factors exactly (degC->K offset 273.15, "
                "SI prefixes, 1 Oe = 1000/(4 pi) A/m) in the scalar registry, in Quantity.si (original retained) and in the "
                "vectorised array path, which is bit-identical to the scalar path.",
    method="Compare units.convert, Quantity.si and convert_array against expected values computed from the defining "
           "relations; exact equality is required for every case; dimensionally wrong conversions must raise.",
    reference=_NIST_SP811)
def unit_normalisation_reproducible() -> CheckResult:
    problems: list[str] = []
    max_rel = 0.0
    for value, src, dst, expected in _UNIT_CASES:
        got = units.convert(value, src, dst)
        rel = abs(got - expected) / abs(expected) if expected != 0 else abs(got - expected)
        max_rel = max(max_rel, rel)
        if got != expected:
            problems.append(f"{value} {src} -> {dst}: got {got!r}, expected {expected!r}")
        q = Quantity.si(value, src)
        if q.unit != dst or q.value != expected or q.original is None or q.original.value != value or q.original.unit != src:
            problems.append(f"Quantity.si({value}, {src!r}) -> {q}")
        vec = convert_array(np.array([value, value, math.nan]), src, dst)
        if not (vec[0] == got and vec[1] == got and math.isnan(vec[2])):
            problems.append(f"convert_array differs from scalar path for {src}->{dst}: {vec.tolist()}")
    for value, src, dst in ((1.0, "mT", "A/m"), (1.0, "degC", "Hz"), (1.0, "W/m^3", "W/kg")):
        try:
            units.convert(value, src, dst)
            problems.append(f"{src}->{dst} should have raised")
        except Exception:  # noqa: BLE001 - UnitError expected
            pass
    return CheckResult(passed=not problems, metrics={"cases": len(_UNIT_CASES), "max_rel_error": max_rel},
                       notes=problems or ["all conversions exact; array path bit-identical to scalar path"],
                       reference=_SI_BROCHURE if not problems else None)


_SYNTHETIC_TRIPLE = {"manufacturer.example-synthetic", "material.example-synthetic-ferrite", "core.example-synthetic-e42"}
_REAL_EVIDENCE_KINDS = {"source", "measurement", "analytical", "simulation"}


@check(
    id="data.catalog-loads-and-cross-references", capability="aipe.magnetics.data.catalog", level="L1", kind="software",
    description="The bundled catalogue loads, every record validates, all cross-references resolve to records of the right "
                "kind, every record satisfies the per-record evidence invariants (synthetic records are tagged and may rest on "
                "assumption evidence; real records cite source/measurement evidence with a locator or url), the synthetic "
                "manufacturer/material/core triple is present, look-ups behave (id / grade, not-found with suggestions) and "
                "malformed files fail loudly with their path.",
    method="Catalog.default(); validate(); per-record invariants independent of how many real entries the maintainer adds; "
           "resolve manufacturer_ref / material_refs; find_material by id and grade on the bundled catalogue and by name substring "
           "on the synthetic triple alone; cores(material) contains the synthetic core; get() of an unknown id raises "
           "DataNotFoundError; a temporary directory with a corrupted record raises DataIntegrityError naming the file.")
def catalog_loads_and_cross_references() -> CheckResult:
    problems: list[str] = []
    catalog = Catalog.default()
    if catalog.validate():
        problems.append(f"catalogue cross-check errors: {catalog.validate()}")
    if not _SYNTHETIC_TRIPLE.issubset(set(catalog.ids())):
        problems.append(f"bundled synthetic records missing: {sorted(_SYNTHETIC_TRIPLE - set(catalog.ids()))}")
    real = 0
    for record in catalog.records():
        selected = [e for e in record.evidence if e.id in record.evidence_refs]
        if record.synthetic:
            if "synthetic" not in record.tags:
                problems.append(f"{record.id}: synthetic record lacks the 'synthetic' tag")
        else:
            real += 1
            cited = [e for e in selected if e.kind in _REAL_EVIDENCE_KINDS]
            if not cited:
                problems.append(f"{record.id}: real record cites no source/measurement evidence")
            for e in cited:
                if e.kind == "source" and not (e.locator or e.url):
                    problems.append(f"{record.id}: source evidence {e.id} has neither locator nor url")
        for e in record.evidence:
            if not e.license:
                problems.append(f"{record.id}: evidence {e.id} has no licence statement")
        for field, target in record.references():
            if target not in catalog:
                problems.append(f"{record.id}: {field} -> {target} unresolved")
    try:
        material = catalog.find_material("EXAMPLE-SYN-1")
        if material.id != "material.example-synthetic-ferrite":
            problems.append("find_material by grade returned the wrong record")
        if catalog.find_material("material.example-synthetic-ferrite").id != material.id:
            problems.append("find_material by id failed")
        triple = Catalog([r for r in catalog.records() if r.id in _SYNTHETIC_TRIPLE])
        if triple.find_material("synthetic mnzn").id != material.id:
            problems.append("find_material by name substring failed on the synthetic triple")
        if "core.example-synthetic-e42" not in [c.id for c in catalog.cores(material.id)]:
            problems.append("cores(material_id) does not contain the synthetic core")
        if triple.find_material("material.example-synthetic-ferrite").id != material.id or len(triple) != 3:
            problems.append("in-memory synthetic triple is inconsistent")
    except (DataNotFoundError, InvalidInputError) as error:
        problems.append(f"look-up raised {error}")
    try:
        catalog.get("material.does-not-exist")
        problems.append("get() of an unknown id did not raise")
    except DataNotFoundError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.json"
        record = catalog.get("manufacturer.example-synthetic").to_json_dict()
        record["properties"]["unknown_field"] = 1
        dump(record, bad)
        try:
            Catalog.load([Path(tmp)], bundled=False)
            problems.append("malformed catalogue file was accepted")
        except DataIntegrityError as error:
            if bad.name not in str(error):
                problems.append("DataIntegrityError does not name the offending file")
    synthetic = sum(r.synthetic for r in catalog.records())
    return CheckResult(passed=not problems, metrics={"records": len(catalog), "synthetic": synthetic, "real": real},
                       notes=problems or [f"bundled catalogue consistent ({synthetic} synthetic, {real} real records); "
                                          "per-record evidence invariants hold and look-ups behave"])


@check(
    id="data.importers-provenance-and-no-overwrite", capability="aipe.magnetics.data.import", level="L1", kind="software",
    description="Importers record the SHA-256 of the original bytes in extensions['aipe.import'], convert explicitly mapped "
                "units to SI with originals retained, never overwrite destinations, refuse to replace existing import provenance "
                "and apply every validation layer (a non-RFC 3339 evidence timestamp is rejected and nothing is written).",
    method="Import the v0.1 fixture winding (migrated) and a 3-row CSV loss table (kHz, mT, degC, kW/m^3) into a temporary "
           "directory; compare digests, converted values and refusal behaviour; repeat the CSV import with an invalid "
           "created_at and expect InvalidInputError without a written file.")
def importers_provenance_and_no_overwrite() -> CheckResult:
    from aipe_magnetics.common.jsonio import sha256_bytes
    from aipe_magnetics.data.importers import import_csv_loss_table, import_json_record

    problems: list[str] = []
    fixtures = v01_fixture_records()
    evidence = {"id": "evidence.check-measurement", "kind": "measurement", "reference": "verification check input (not a real measurement)",
                "description": "synthetic three-row table used only to exercise the importer", "created_at": "2026-10-19T00:00:00Z",
                "review_status": "unreviewed", "license": "NOASSERTION", "locator": "losses.csv"}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "winding.json"
        dump(fixtures["winding"], source)
        raw_digest = sha256_bytes(source.read_bytes())
        result = import_json_record(source, tmp_path / "out" / "winding.json", migrate_v01=True, catalogue=())
        ext = result.record.extensions.get("aipe.import", {})
        if ext.get("source_sha256") != raw_digest or ext.get("source_filename") != "winding.json":
            problems.append("json import did not record the source digest / name")
        try:
            import_json_record(source, tmp_path / "out" / "winding.json", migrate_v01=True, catalogue=())
            problems.append("json import overwrote an existing destination")
        except FileExistsError:
            pass
        try:
            import_json_record(tmp_path / "out" / "winding.json", tmp_path / "out" / "again.json", catalogue=())
            problems.append("existing import provenance was replaced")
        except InvalidInputError:
            pass
        csv = tmp_path / "losses.csv"
        csv.write_text("f_kHz,B_mT,T_C,Pv\n100,100,100,100\n200,50,25,20\n50,200,100,250.5\n", encoding="utf-8")
        csv_result = import_csv_loss_table(
            csv, columns={"f_kHz": ("frequency", "kHz"), "B_mT": ("flux_density_peak", "mT"), "T_C": ("temperature", "degC"),
                          "Pv": ("loss_density", "kW/m^3")},
            record_id="measurement.check-losses", name="check", material_ref="material.example-synthetic-ferrite",
            evidence=evidence, synthetic=True, destination=tmp_path / "out" / "losses.json")
        point = csv_result.record.properties["points"][0]
        expected = {"frequency": (1.0e5, "Hz", 100.0, "kHz"), "flux_density_peak": (0.1, "T", 100.0, "mT"),
                    "temperature": (373.15, "K", 100.0, "degC"), "loss_density": (1.0e5, "W/m^3", 100.0, "kW/m^3")}
        for key, (value, unit, original_value, original_unit) in expected.items():
            q = point[key]
            if q["value"] != value or q["unit"] != unit or q["original"] != {"value": original_value, "unit": original_unit}:
                problems.append(f"csv import: {key} = {q}")
        if csv_result.record.extensions["aipe.import"]["source_sha256"] != sha256_bytes(csv.read_bytes()):
            problems.append("csv import did not record the source digest")
        if MagneticRecord.model_validate(load(tmp_path / "out" / "losses.json")).id != "measurement.check-losses":
            problems.append("written csv record does not load")
        try:
            import_csv_loss_table(
                csv, columns={"f_kHz": ("frequency", "kHz"), "B_mT": ("flux_density_peak", "mT"), "T_C": ("temperature", "degC"),
                              "Pv": ("loss_density", "kW/m^3")},
                record_id="measurement.check-bad-timestamp", name="check", material_ref="material.example-synthetic-ferrite",
                evidence=dict(evidence, created_at="yesterday afternoon"), synthetic=True, destination=tmp_path / "out" / "bad.json")
            problems.append("non-RFC 3339 created_at was accepted by the csv importer")
        except InvalidInputError as error:
            if "date-time" not in str(error):
                problems.append(f"created_at rejection does not name the schema format: {error}")
        if (tmp_path / "out" / "bad.json").exists():
            problems.append("a rejected import wrote its destination")
    return CheckResult(passed=not problems, metrics={"importers_exercised": 2},
                       notes=problems or ["digests, SI conversion with originals, non-overwrite, provenance protection and "
                                          "full-layer validation verified"])


__all__ = [
    "schema_v02_is_valid_draft2020", "schema_mirror_in_sync", "v01_examples_migrate_and_validate", "hdf5_roundtrip_checksums",
    "unit_normalisation_reproducible", "catalog_loads_and_cross_references", "importers_provenance_and_no_overwrite",
]
