"""Paderborn University LEA ``materialdatabase`` importer (GPL-3.0-only), commit ``04fff8e9``.

The 79 CSV files are classified by provenance (upstream README, data loaders and changelog):

* ``complex_permeability/LEA_MTB`` and ``complex_permittivity/LEA_MTB`` - **independent LEA measurements**
  (LEA magnetic test bench, toroid / cuboid specimens) -> ``experimental_measurement``;
* ``complex_permeability/MagNet`` - mu', mu'' **recomputed by LEA from Princeton MagNet** sine records (elliptic-loop
  assumption) -> ``derived_from_measurement`` with ``underlying_source = princeton-magnet:<grade>``;
* ``complex_permeability/TDK_MDT`` - computed from TDK Magnetic Design Tool exports -> ``manufacturer_tool_export``;
  ``complex_permeability/datasheet`` is numerically identical to ``TDK_MDT`` and is **not** imported (duplicate);
* ``datasheet_curves`` - digitised manufacturer datasheet curves -> ``manufacturer_datasheet``.

Licence handling (conservative GPL-3.0 reading): the normalised subtree ``data/normalized/paderborn-lea`` is released
under GPL-3.0-only, carries the verbatim upstream LICENSE and a NOTICE that states the modification and date and
offers the corresponding source (upstream URL + commit + this importer). The GPL grant cannot clear third-party rights
in the underlying MagNet / TDK / datasheet content; those records carry an explicit flag.

Upstream unit anomalies (documented in docs/provenance-and-limitations.md) are handled explicitly: B-H files whose ``b``
column is evidently mT (|b| > 3, files N27/N30/T37) are converted with ``original_unit = mT`` and flagged
``unit_reinterpreted_mT``; isolated mT rows inside T-valued files are excluded and counted; the T37 B-H curve is flagged
as a probable copy of N30.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from aipe_magnetics_database.common.errors import DataIntegrityError
from aipe_magnetics_database.ingest.acquire import RawFile, obtain
from aipe_magnetics_database.ingest.common import (
    ImportSummary,
    SourceWriter,
    column,
    dataset_record,
    evidence,
    finish_source,
    importer_attribution,
    kelvin,
    measurement_record,
    q,
    slug,
    validated,
)

SOURCE_ID = "paderborn_lea_materialdatabase"
DIRECTORY = "paderborn-lea"
PINS = json.loads((Path(__file__).resolve().parents[1] / "pins" / "paderborn-lea.json").read_text(encoding="utf-8"))
COMMIT = PINS["commit"]
COMMIT_TIME = PINS["commit_time"]
LICENSE = "GPL-3.0-only"
REPOSITORY = PINS["repository"]
MANUFACTURER = {"N27": "TDK", "N30": "TDK", "N49": "TDK", "N87": "TDK", "N95": "TDK", "PC200": "TDK", "T37": "TDK", "3C90": "Ferroxcube",
                "3C92": "Ferroxcube", "3C94": "Ferroxcube", "3C95": "Ferroxcube", "3E6": "Ferroxcube", "3F4": "Ferroxcube", "3F46": "Ferroxcube",
                "DMR96A": "DMEGC"}
PREFIX = "materialdatabase/data/"


def raw_files() -> list[RawFile]:
    return [RawFile(f["path"], f"https://raw.githubusercontent.com/upb-lea/materialdatabase/{COMMIT}/{f['path']}", f["sha256"], f["size"], LICENSE)
            for f in PINS["files"]]


def _rows(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text), skipinitialspace=True))


def _f(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def _k(value: str | None) -> float | None:
    v = _f(value)
    return None if v is None else v + 273.15


def _provenance(activity: str, inputs: list[str] | None = None) -> dict[str, Any]:
    out = {"activity": activity, "source_commit": COMMIT, "tool": importer_attribution("paderborn"),
           "normalization": "CSV columns mapped with the units documented by the upstream loaders (f Hz, T degC -> K, b T peak, h A/m, "
                            "p_v W/m^3, mu and eps relative with mu = mu' - j mu''); empty cells -> null; rows kept in file order."}
    if inputs:
        out["inputs"] = inputs
    return out


FAMILIES = {
    "complex_permeability/LEA_MTB": ("experimental_measurement", "complex_permeability", "LEA magnetic test bench, capacitively compensated "
                                     "two-winding method on toroid specimens"),
    "complex_permeability/MagNet": ("derived_from_measurement", "complex_permeability", "mu', mu'' recomputed by LEA from Princeton MagNet "
                                    "sinusoidal records assuming elliptical B-H loops"),
    "complex_permeability/TDK_MDT": ("manufacturer_tool_export", "complex_permeability", "mu', mu'' computed by LEA from TDK Magnetic Design "
                                     "Tool loss and amplitude-permeability exports"),
    "complex_permittivity/LEA_MTB": ("experimental_measurement", "complex_permittivity", "LEA impedance measurement (Wayne Kerr 6515B) on "
                                     "silver-plated cuboid specimens"),
}


def run(output_root: Path, *, local_dir: str | Path | None = None, allow_download: bool = False, cache: Any = None) -> ImportSummary:
    files = raw_files()
    paths = obtain(SOURCE_ID, files, local_dir=local_dir, allow_download=allow_download, cache=cache, commit=COMMIT)
    summary = ImportSummary(SOURCE_ID, COMMIT, LICENSE, "permitted under GPL-3.0-only (copyleft subtree with LICENSE, NOTICE and source offer)")
    summary.inputs = [{"path": f.name, "sha256": f.sha256} for f in files]
    summary.notes.append("pyproject.toml upstream declares an MIT classifier; the LICENSE file (GPL-3.0) is treated as authoritative.")
    writer = SourceWriter(output_root / DIRECTORY, summary)
    materials: dict[str, dict[str, Any]] = {}
    measurements: list[dict[str, Any]] = []
    general: dict[str, dict[str, str]] = {}
    for raw in files:
        if raw.name.endswith("general_information.csv"):
            grade = raw.name.split("/")[-2]
            rows = _rows(paths[raw.name])
            if rows:
                general[grade] = rows[0]
    for grade in sorted({f.name.split("/")[-1].removesuffix(".csv") for f in files if "/complex_" in f.name}
                        | {f.name.split("/")[-2] for f in files if "/datasheet_curves/" in f.name}):
        materials[grade] = material_record(grade, general.get(grade), summary)

    def ev(path: str, data_class: str, what: str) -> list[dict[str, Any]]:
        url = f"{REPOSITORY}/blob/{COMMIT}/{path}"
        sha = next(f.sha256 for f in files if f.name == path)
        items = [evidence("evidence.paderborn-lea-source", "source", "upb-lea/materialdatabase", "GPL-3.0 Paderborn University LEA material database.",
                          COMMIT_TIME, LICENSE, url=url, version=COMMIT, locator=path, sha256=sha)]
        description = {"experimental_measurement": "Independent LEA measurement: " + what,
                       "derived_from_measurement": "Derived by LEA from Princeton MagNet measurements: " + what,
                       "manufacturer_tool_export": "Derived by LEA from TDK Magnetic Design Tool output: " + what,
                       "manufacturer_datasheet": "Digitised manufacturer datasheet curve: " + what}[data_class]
        items.append(evidence("evidence.paderborn-lea-measurement", "measurement", "Paderborn University LEA" if data_class == "experimental_measurement"
                              else ("Princeton MagNet via LEA" if data_class == "derived_from_measurement" else "Manufacturer via LEA"),
                              description, COMMIT_TIME, LICENSE, url=url, version=COMMIT, locator=path))
        return items

    for raw in files:
        rel = raw.name.removeprefix(PREFIX)
        if not raw.name.endswith(".csv") or rel.endswith("general_information.csv"):
            continue
        if rel.startswith("complex_permeability/datasheet/"):
            summary.exclude(rel, "numerically identical to complex_permeability/TDK_MDT (max relative difference 2.5e-16); imported once")
            continue
        rows = _rows(paths[raw.name])
        family = "/".join(rel.split("/")[:2])
        if family in FAMILIES:
            data_class, quantity, method = FAMILIES[family]
            grade = rel.split("/")[-1].removesuffix(".csv")
            measurements.extend(_complex_table(rel, raw.name, rows, grade, data_class, quantity, method, materials[grade], writer, summary, ev))
        elif rel.startswith("datasheet_curves/"):
            grade = rel.split("/")[1]
            rec = _datasheet_table(rel, raw.name, rows, grade, materials[grade], writer, summary, ev)
            if rec:
                measurements.append(rec)
        else:  # pragma: no cover - upstream layout change
            raise DataIntegrityError(f"unclassified Paderborn file {rel}")
    summary.count("magnetic_material", len(materials))
    summary.count("measurement", len(measurements))
    dataset = dataset_record(
        rid="dataset.paderborn-lea-materialdatabase", name="Paderborn LEA materialdatabase", source_id=SOURCE_ID,
        title="Paderborn University LEA materialdatabase (upb-lea/materialdatabase)", license=LICENSE, url=f"{REPOSITORY}/tree/{COMMIT}/materialdatabase/data",
        version=COMMIT, source_commit=COMMIT, publisher="Paderborn University, Department of Power Electronics and Electrical Drives (LEA)",
        redistribution=True, record_count=len(measurements), checksums={f.name: f.sha256 for f in files}, materials=sorted(materials),
        description="Mixed-provenance collection: independent LEA measurements, MagNet-derived permeability, TDK MDT-derived permeability and "
                    "digitised datasheet curves (see extensions aipe.data.data_class of every record). Redistributed under GPL-3.0-only.",
        evidence_items=[evidence("evidence.paderborn-lea-source", "source", "upb-lea/materialdatabase", "GPL-3.0 licence (LICENSE at the pinned commit).",
                                 COMMIT_TIME, LICENSE, url=f"{REPOSITORY}/blob/{COMMIT}/LICENSE", version=COMMIT)],
        provenance=_provenance("paderborn_lea_dataset"),
        flags=["third_party_rights_in_underlying_content"])
    mats = [materials[g] for g in sorted(materials)]
    shard_files = {"materials": writer.shards(mats, "materials", "materials", 100),
                   "measurements": writer.shards(measurements, "measurements", "measurements", 100)}
    writer.json("datasets/paderborn-lea-materialdatabase.json", dataset)
    license_path = paths.get("LICENSE")
    if license_path is not None:
        writer.file("LICENSE", license_path.read_bytes())
    writer.file("NOTICE.md", NOTICE.encode("utf-8"))
    return finish_source(writer, summary, shard_files, ["datasets/paderborn-lea-materialdatabase.json"])


NOTICE = f"""# Notice - modified GPL-3.0 material

The records in this directory are a **modified version** of data from the Paderborn University LEA
`materialdatabase` (https://github.com/upb-lea/materialdatabase), copyright UPB-LEA, commit
`{COMMIT}`, licensed under the GNU General Public License v3.0 (see `LICENSE`, verbatim upstream text).

Modification (2026-11-02, AIPE Magnetics Database 0.3.0): the CSV files were converted into AIPE Magnetic Data
Schema v0.2 JSON/JSONL records with SI units (degC -> K), explicit column definitions and provenance; anomalous
rows were excluded or unit-flagged as documented in `docs/provenance-and-limitations.md`. Values were not otherwise changed.

This directory as a whole is released under **GPL-3.0-only**. Corresponding source: the original CSV files at the
pinned commit above (SHA-256 of each file in `src/aipe_magnetics_database/ingest/pins/paderborn-lea.json`) together with
the conversion program `src/aipe_magnetics_database/ingest/sources/paderborn.py` of this repository.

The GPL grant covers UPB-LEA's work only. Records classified `derived_from_measurement` rest on Princeton MagNet
measurements (MIT), `manufacturer_tool_export` on TDK Magnetic Design Tool output and `manufacturer_datasheet` on
digitised manufacturer datasheets; the rights of those third parties are not cleared by this licence.
"""


def material_record(grade: str, general: dict[str, str] | None, summary: ImportSummary) -> dict[str, Any]:
    manufacturer = MANUFACTURER.get(grade, "unknown")
    props: dict[str, Any] = {"grade": grade, "description": f"{manufacturer} {grade} as named by the Paderborn LEA materialdatabase."}
    flags = ["third_party_unreviewed"]
    if general:
        props["material_class"] = "ferrite_mnzn" if general.get("material") == "MnZn" else "other"
        ref = "evidence.paderborn-lea-datasheet"
        if _f(general.get("initial permeability")):
            props["initial_permeability"] = {"value": q(_f(general["initial permeability"]), "1"), "source_ref": ref}
        if _f(general.get("curie temperature in C")) is not None:
            props["curie_temperature"] = {"value": kelvin(_f(general["curie temperature in C"])), "source_ref": ref,
                                          "note": "datasheet value as transcribed upstream; datasheets often state a lower bound"}
            flags.append("curie_temperature_may_be_bound")
        if _f(general.get("density in kg/m^3")):
            props["density"] = {"value": q(_f(general["density in kg/m^3"]), "kg/m^3"), "source_ref": ref}
        if _f(general.get("resistivity in Ohm m")):
            props["resistivity"] = {"value": q(_f(general["resistivity in Ohm m"]), "Ohm*m"), "source_ref": ref}
        sat = []
        for t in (25, 100):
            v = _f(general.get(f"saturation flux density in T at {t} C"))
            if v:
                sat.append({"value": q(v, "T"), "temperature": kelvin(t), "source_ref": ref})
        if sat:
            props["saturation_flux_density"] = sat
    items = [evidence("evidence.paderborn-lea-source", "source", "upb-lea/materialdatabase", "Material named by the LEA materialdatabase.",
                      COMMIT_TIME, LICENSE, url=f"{REPOSITORY}/tree/{COMMIT}/materialdatabase/data", version=COMMIT)]
    if general:
        path = f"{PREFIX}datasheet_curves/{grade}/general_information.csv"
        items.append(evidence("evidence.paderborn-lea-datasheet", "source", f"{manufacturer} {grade} datasheet via LEA",
                              "Datasheet scalars transcribed by LEA (general_information.csv).", COMMIT_TIME, LICENSE,
                              url=f"{REPOSITORY}/blob/{COMMIT}/{path}", version=COMMIT, locator=path))
    return validated({"schema_version": "0.2.0", "id": f"material.paderborn-lea-{slug(grade)}", "kind": "magnetic_material",
                      "name": f"{manufacturer} {grade}", "synthetic": False, "license": LICENSE, "properties": props,
                      "evidence_refs": [e["id"] for e in items], "evidence": items, "dataset_ref": "dataset.paderborn-lea-materialdatabase",
                      "quality": {"status": "normalized", "flags": sorted(flags)}, "provenance": _provenance("paderborn_lea_material"),
                      "extensions": {"aipe.data": {"data_class": "manufacturer_datasheet" if general else "identity", "source_id": SOURCE_ID},
                                     "aipe.paderborn": {"manufacturer": manufacturer}}})


def _complex_table(rel, raw_name, rows, grade, data_class, quantity, method, material, writer, summary, ev) -> list[dict[str, Any]]:
    perm = quantity == "complex_permeability"
    out_rows, negative = [], 0
    for r in rows:
        f, t = _f(r.get("f")), _k(r.get("T"))
        a, b = (_f(r.get("mu_real")), _f(r.get("mu_imag"))) if perm else (_f(r.get("eps_real")), _f(r.get("eps_imag")))
        if f is None or a is None or b is None:
            summary.exclude(rel, "row without f / real / imaginary value")
            continue
        if b < 0:
            negative += 1
        row = [r.get("probe") or None, f, t] + ([_f(r.get("b")), _f(r.get("h_offset"))] if perm else []) + [a, b]
        out_rows.append(row)
    cols = [{"name": "specimen", "unit": "", "definition": "upstream probe / specimen code (e.g. R29.5x19x14.9 toroid OD x ID x H in mm); null when not given"},
            column("frequency", "excitation frequency" + (" (commanded, rounded)" if data_class == "derived_from_measurement" else "")),
            column("temperature", "specimen temperature" + (" (commanded oil-bath set point)" if data_class == "derived_from_measurement" else ""), "degC")]
    if perm:
        cols += [column("flux_density_peak", "peak AC flux density"), column("dc_bias_field", "DC bias field H_dc (null = column absent)"),
                 column("mu_real", "mu' of mu = mu' - j mu''"), column("mu_imag", "mu'' of mu = mu' - j mu'' (positive = loss)")]
    else:
        cols += [column("eps_real", "eps' of eps = eps' - j eps''"), column("eps_imag", "eps'' of eps = eps' - j eps''")]
    flags = []
    if negative:
        flags.append("negative_values_present")
        summary.flag(f"{quantity}_negative_imaginary_rows", negative)
    if data_class != "experimental_measurement":
        flags.append("third_party_rights_in_underlying_content")
    family = rel.split("/")[1].lower().replace("_", "-")
    rid = f"measurement.paderborn-lea-{slug(grade)}-{quantity.replace('_', '-')}-{family}"
    rec = measurement_record(
        rid=rid, name=f"LEA {material['name']} {quantity.replace('_', ' ')} ({rel.split('/')[1]})", material_ref=material["id"], license=LICENSE,
        evidence_items=ev(raw_name, data_class, method), data_class=data_class, source_id=SOURCE_ID, quantity=quantity, columns=cols,
        rows=out_rows, writer=writer, provenance=_provenance("normalize_paderborn_lea", [material["id"]]), method=method,
        description=f"{rel} at commit {COMMIT[:8]}", flags=flags, dataset_ref="dataset.paderborn-lea-materialdatabase",
        underlying_source=f"princeton-magnet:{slug(grade)}" if data_class == "derived_from_measurement" else None,
        derived_from=[f"princeton-magnet:{slug(grade)}"] if data_class == "derived_from_measurement" else (),
        extra={"upstream_file": raw_name})
    summary.rows(quantity, len(out_rows))
    return [rec]


DATASHEET = {
    "b_over_h_at_f_T.csv": ("bh_curve", ["h", "b", "T", "f", "branch"]),
    "mu_amplitude_over_b_at_T.csv": ("amplitude_permeability", ["b", "mu", "T"]),
    "p_v_over_b_at_f_T.csv": ("core_loss", ["b", "p_v", "f", "T"]),
    "p_v_over_f_at_b_T.csv": ("core_loss", ["f", "p_v", "T", "b"]),
    "p_v_over_T_at_f_b.csv": ("core_loss", ["T", "p_v", "f", "b"]),
    "small_signal_mu_real_over_f_at_T.csv": ("complex_permeability", ["f", "mu_real"]),
    "small_signal_mu_imag_over_f_at_T.csv": ("complex_permeability", ["f", "mu_imag"]),
    "small_signal_mu_initial_over_T.csv": ("initial_permeability", ["T", "mu"]),
}


def _datasheet_table(rel, raw_name, rows, grade, material, writer, summary, ev) -> dict[str, Any] | None:
    name = rel.split("/")[-1]
    quantity, _ = DATASHEET[name]
    if not rows:
        summary.exclude(rel, "header only, no data rows")
        return None
    flags, notes = ["third_party_rights_in_underlying_content"], []
    b_unit_mT = False
    if quantity == "bh_curve":
        if max(abs(_f(r["b"]) or 0.0) for r in rows) > 3.0:
            b_unit_mT = True
            flags.append("unit_reinterpreted_mT")
            notes.append("upstream b column is evidently mT (|b| up to several hundred); converted to T with original_unit mT")
            summary.flag("paderborn_bh_files_in_mT")
        if grade == "T37":
            flags.append("suspected_copy_of_other_material")
            notes.append("this curve matches N30/b_over_h_at_f_T.csv except one row (probable upstream copy)")
        out, cols = [], [column("temperature", "core temperature", "degC"), column("frequency", "excitation frequency (null/0 as given upstream)"),
                         column("field", "magnetic field strength H"), column("flux_density", "flux density B", "mT" if b_unit_mT else None),
                         {"name": "branch", "unit": "", "definition": "r = rising, f = falling loop branch (null = not given)"}]
        for r in rows:
            b = _f(r.get("b"))
            if b is None or _f(r.get("h")) is None:
                summary.exclude(rel, "row without h or b")
                continue
            out.append([_k(r.get("T")), _f(r.get("f")), _f(r["h"]), b / 1000.0 if b_unit_mT else b, (r.get("branch") or None)])
    elif quantity == "core_loss":
        cols = [column("frequency", "excitation frequency"), column("flux_density_peak", "peak AC flux density (sinusoidal datasheet curve)"),
                column("temperature", "core temperature", "degC"), column("loss_density", "core loss per unit volume")]
        out = []
        for r in rows:
            f, b, t, p = _f(r.get("f")), _f(r.get("b")), _k(r.get("T")), _f(r.get("p_v"))
            if None in (f, b, t, p):
                summary.exclude(rel, "row without f / b / T / p_v")
                continue
            if b > 3.0:
                summary.exclude(rel, "b > 3 T: isolated mT row inside a T-valued file (not reinterpreted)")
                continue
            out.append([f, b, t, p])
    elif quantity == "amplitude_permeability":
        cols = [column("flux_density_peak", "peak AC flux density"), column("temperature", "core temperature", "degC"),
                column("amplitude_permeability", "amplitude permeability")]
        out = []
        for r in rows:
            b, mu = _f(r.get("b")), _f(r.get("mu"))
            if b is None or mu is None:
                summary.exclude(rel, "row without b or mu")
                continue
            if b > 3.0:
                summary.exclude(rel, "b > 3 T: isolated mT row inside a T-valued file (not reinterpreted)")
                continue
            out.append([b, _k(r.get("T")), mu])
    elif quantity == "complex_permeability":
        comp = "mu_real" if "mu_real" in name else "mu_imag"
        cols = [column("frequency", "small-signal frequency"), column(comp, ("mu'" if comp == "mu_real" else "mu''") + " (small signal, datasheet curve; "
                                                                      "temperature not stated in the file)")]
        out = [[_f(r["f"]), _f(r[comp])] for r in rows if _f(r.get("f")) is not None and _f(r.get(comp)) is not None]
        quantity = "complex_permeability"
    else:  # initial permeability vs T
        cols = [column("temperature", "core temperature", "degC"), column("relative_permeability", "initial permeability")]
        out = [[_k(r["T"]), _f(r["mu"])] for r in rows if _f(r.get("T")) is not None and _f(r.get("mu")) is not None]
    if not out:
        summary.exclude(rel, "no usable rows")
        return None
    rid = f"measurement.paderborn-lea-{slug(grade)}-{slug(name.removesuffix('.csv'))}"
    rec = measurement_record(
        rid=rid, name=f"LEA {material['name']} datasheet {name.removesuffix('.csv')}", material_ref=material["id"], license=LICENSE,
        evidence_items=ev(raw_name, "manufacturer_datasheet", name), data_class="manufacturer_datasheet", source_id=SOURCE_ID, quantity=quantity,
        columns=cols, rows=out, writer=writer, provenance=_provenance("normalize_paderborn_lea", [material["id"]]),
        method="digitised manufacturer datasheet curve (digitising tool and datasheet revision not recorded upstream)",
        description=f"{rel} at commit {COMMIT[:8]}", flags=flags, quality_notes=notes, dataset_ref="dataset.paderborn-lea-materialdatabase",
        extra={"upstream_file": raw_name})
    summary.rows(quantity, len(out))
    return rec


__all__ = ["COMMIT", "LICENSE", "SOURCE_ID", "raw_files", "run"]
