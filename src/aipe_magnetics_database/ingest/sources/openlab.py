"""Importers of four openly licensed laboratory datasets found in the 2026-11 source discovery.

* ``nabih_magneto`` (MIT) - A. Nabih (Virginia Tech CPES), *Core characterization data set: loss, permeability and
  permittivity*, github.com/asalahnabih/magneto @ ``0a3a0f1e``. MS Access database: sinusoidal core loss P_v(B_m) at
  0.1-3.5 MHz and 25 degC for eight MHz ferrites on toroids, complex permeability and permittivity vs frequency. The
  ``.accdb`` is read through a Jackcess 4.0.5 TSV export (``data/raw/nabih-magneto/derived``, reproduced by
  ``tools/accdb_export``) whose recorded input SHA-256 must equal the committed original.
* ``guillod_n87_dielectric`` (MIT) - T. Guillod, W. Roberts, C. Sullivan (Dartmouth), *Characterization and Impact of
  Large-Signal Dielectric Properties in MnZn Ferrites*, APEC 2024, github.com/otvam/large_signal_ferrite_apec24 @
  ``dd068898``. Small-signal, large-signal (vs peak electric field) and temperature-dependent real permittivity and
  conductivity of TDK N87, read from the MATLAB ``data.mat`` files (full float64 precision; the CSV copies are rounded).
  The two permeability folders are *not* imported: ``N87_mu_small`` is a TDK datasheet curve and ``N87_mu_large`` is
  computed from MagNet data (both stated in their README files).
* ``zenodo_7368936_cwh_n87`` (CC-BY-4.0) - Guillod et al., *Composite Waveform Hypothesis* reproducibility dataset,
  DOI 10.5281/zenodo.7368936: MagNet measurements of TDK N87 under piecewise-linear (triangular / trapezoidal)
  excitation on two toroids. The R34 variable-temperature set repeats measurements of the packaged Princeton MagNet
  N87 database and is linked to it as a duplicate; the R22 ambient set is not in that database.
* ``zenodo_10725890_inrim`` (CC-BY-4.0) - M. Pasquale (INRIM), *Examples of first magnetization curves of ferromagnetic
  samples*, DOI 10.5281/zenodo.10725890: DC first-magnetisation curves of 24 materials (steels, FeSiAl, SMC, cast
  iron). The file states no units; values are consistent with A/m and T and are stored so, flagged ``units_not_stated``.

Units are scaled with decimal arithmetic from the published text (``1.1 MHz`` -> ``1100000.0`` exactly); nothing is
interpolated, smoothed or averaged; rows that cannot be attributed are excluded and counted in the import report.
"""

from __future__ import annotations

import csv
import io
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

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
    measurement_record,
    sha256_path,
    slug,
    validated,
)


def dec(text: str | float | int, factor: str = "1", offset: str = "0") -> float:
    """``float(Decimal(text) * factor + offset)``: exact decimal scaling of a published value."""
    return float(Decimal(str(text).strip()) * Decimal(factor) + Decimal(offset))


def _prov(tool: str, version: str, normalization: str, inputs: list[str] | None = None) -> dict[str, Any]:
    out = {"activity": f"normalize_{tool}", "dataset_version": version, "tool": importer_attribution(tool), "normalization": normalization}
    if inputs:
        out["inputs"] = inputs
    return out


def _identity(mid: str, name: str, grade: str, license: str, ev: list[dict[str, Any]], dataset_ref: str, prov: dict[str, Any], source_id: str,
              description: str, flags: tuple[str, ...] = (), ext: dict[str, Any] | None = None) -> dict[str, Any]:
    extensions: dict[str, Any] = {"aipe.data": {"data_class": "identity", "source_id": source_id}}
    if ext:
        extensions.update(ext)
    return validated({
        "schema_version": "0.2.0", "id": mid, "kind": "magnetic_material", "name": name, "synthetic": False, "license": license,
        "properties": {"grade": grade, "description": description}, "evidence_refs": [e["id"] for e in ev], "evidence": ev,
        "quality": {"status": "normalized", "flags": sorted({"identity_only", "third_party_unreviewed", *flags})},
        "dataset_ref": dataset_ref, "provenance": prov, "extensions": extensions})


# =================================================================================================
# Nabih "magneto" (MIT)
# =================================================================================================
NABIH_ID = "nabih_magneto"
NABIH_DIR = "nabih-magneto"
NABIH_COMMIT = "0a3a0f1e10356b294ae6782eaa46f4d478e5bff9"
NABIH_TIME = "2025-01-22T17:33:14Z"
NABIH_REPO = "https://github.com/asalahnabih/magneto"
NABIH_ACCDB = "Core loss database_v2.accdb"
NABIH_LICENSE = "MIT"
NABIH_CITATION = ("A. Nabih, 'Core characterization data set: loss, permeability and permittivity', GitHub asalahnabih/magneto "
                  "(MIT); measurements described in A. Nabih et al., IEEE Xplore document 10149511")
_NABIH_RAW = {NABIH_ACCDB: "cc2938a9b3a9c79fd72ecd8de3d49aa5a06b625a21ff28be455756b27f2b01c2"}
#: grade -> manufacturer; the database names only the grade, the manufacturer is taken from the identical grade name
#: in the MAS catalogue (ACME P53/P63, Proterial ML-series, Ferroxcube 3F36, DMEGC DMR-series)
NABIH_MANUFACTURERS = {"ML91S": "Proterial", "ML95S": "Proterial", "ML27D": "Proterial", "P53": "ACME", "P63": "ACME", "3F36": "Ferroxcube",
                       "DMR51W": "DMEGC", "DMR96A": "DMEGC"}


def nabih_raw_files() -> list[RawFile]:
    return [RawFile(name, f"https://raw.githubusercontent.com/asalahnabih/magneto/{NABIH_COMMIT}/{name.replace(' ', '%20')}", sha, None, NABIH_LICENSE)
            for name, sha in _NABIH_RAW.items()]


def _tsv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")), delimiter="\t"))


def _clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "").replace("&nbsp;", " ").replace("・", "-")
    return re.sub(r"\s+", " ", text).strip()


def run_nabih(output_root: Path, *, local_dir: str | Path | None = None, allow_download: bool = False, cache: Any = None,
              export_dir: Path | None = None) -> ImportSummary:
    files = nabih_raw_files()
    paths = obtain(NABIH_ID, files, local_dir=local_dir, allow_download=allow_download, cache=cache, commit=NABIH_COMMIT)
    export_dir = export_dir or Path(paths[NABIH_ACCDB]).parent.parent / "derived"
    manifest = json.loads((export_dir / "EXPORT.json").read_text(encoding="utf-8"))
    if manifest["input_sha256"] != _NABIH_RAW[NABIH_ACCDB]:
        raise DataIntegrityError(f"{export_dir}/EXPORT.json was made from another database (sha256 {manifest['input_sha256']})")
    for name, digest in manifest["outputs"].items():
        if sha256_path(export_dir / name) != digest:
            raise DataIntegrityError(f"{export_dir / name} differs from the SHA-256 recorded in EXPORT.json")
    summary = ImportSummary(NABIH_ID, NABIH_COMMIT, NABIH_LICENSE, "permitted (MIT; cite per CITATION.cff)")
    summary.inputs = [{"path": f.name, "url": f.url, "sha256": f.sha256} for f in files] + [
        {"path": f"derived/{n}", "sha256": d, "derived_with": manifest["tool"]} for n, d in sorted(manifest["outputs"].items())]
    writer = SourceWriter(output_root / NABIH_DIR, summary)
    tests = _tsv(export_dir / "Core_loss_database.tsv")
    points = _tsv(export_dir / "Bm_vs_Pv_table.tsv")
    chars = {r["CID"]: r for r in _tsv(export_dir / "Characteristics_database.tsv")}
    mu = _tsv(export_dir / "permeability.tsv")
    eps = _tsv(export_dir / "permittivity.tsv")
    summary.notes.append("The Access table 'Paste Errors' (201 rows, an import-error log of MS Access) is not data and is not imported.")
    by_test = {r["ID"]: r for r in tests}

    def ev(locator: str, measured: str | None = None) -> list[dict[str, Any]]:
        items = [evidence("evidence.nabih-source", "source", "asalahnabih/magneto", "MIT-licensed core characterisation database.", NABIH_TIME,
                          NABIH_LICENSE, url=f"{NABIH_REPO}/tree/{NABIH_COMMIT}", version=NABIH_COMMIT, locator=locator,
                          sha256=_NABIH_RAW[NABIH_ACCDB])]
        if measured:
            items.append(evidence("evidence.nabih-measurement", "measurement", "Virginia Tech CPES (A. Nabih)", measured, NABIH_TIME, NABIH_LICENSE,
                                  url=NABIH_REPO, version=NABIH_COMMIT, locator=locator))
        return items

    norm = ("Jackcess TSV export of the Access tables (derived/EXPORT.json); MHz -> Hz, mT -> T, kW/m^3 -> W/m^3, degC -> K and mm -> m "
            "scaled exactly from the exported decimal text; points are joined to their test by ID and permeability / permittivity rows to "
            "the characteristics table by CID; rows without an ID / CID are excluded and counted.")
    materials: dict[str, dict[str, Any]] = {}

    def material(grade: str) -> str:
        if grade not in materials:
            maker = NABIH_MANUFACTURERS.get(grade)
            name = f"{maker} {grade}" if maker else grade
            materials[grade] = _identity(
                f"material.nabih-{slug(grade)}", name, grade, NABIH_LICENSE, ev(f"{NABIH_ACCDB}: Material = {grade}"), "dataset.nabih-magneto",
                _prov("nabih", NABIH_COMMIT, norm), NABIH_ID, f"{name} as named by the Nabih core characterisation database (grade only; "
                                                                "manufacturer from the identical grade name in the MAS catalogue).",
                flags=("manufacturer_from_grade_name",), ext={"aipe.nabih": {"material_label": grade, "manufacturer": maker}})
        return materials[grade]["id"]

    measurements = []
    # --- core loss -----------------------------------------------------------------------------------------------
    loss_rows: dict[str, list[list[Any]]] = {}
    for p in points:
        test = by_test.get(p["ID"])
        if test is None:
            summary.exclude("Bm vs Pv table", "point without a test ID (cannot be attributed to a material or frequency)")
            continue
        if test["Excitation"].strip().lower() != "sinusoidal":
            summary.exclude("Bm vs Pv table", f"excitation {test['Excitation']!r} not mapped")
            continue
        loss_rows.setdefault(test["Material"], []).append([
            dec(test["Frequency (MHz)"], "1e6"), dec(p["Bm (mT)"], "0.001"), dec(p["Pv (kW/m3)"], "1000"), dec(test["Temp (deg C)"], "1", "273.15"),
            0.0 if dec(test["DC Bias (A-turn)"]) == 0 else None, "sinusoidal", int(test["ID"]),
            dec(test["OD (mm)"], "0.001"), dec(test["ID (mm)"], "0.001"), dec(test["H (mm)"], "0.001")])
    nonzero_bias = sum(1 for t in tests if dec(t["DC Bias (A-turn)"]) != 0)
    if nonzero_bias:
        summary.notes.append(f"{nonzero_bias} test(s) with non-zero DC bias in ampere-turns: the field is not computed (no path length).")
    for grade in sorted(loss_rows):
        rows = loss_rows[grade]
        mid = material(grade)
        n_tests = len({r[6] for r in rows})
        measurements.append(measurement_record(
            rid=f"measurement.nabih-{slug(grade)}-core-loss", name=f"Nabih {grade} sinusoidal core loss (MHz)", material_ref=mid,
            license=NABIH_LICENSE, evidence_items=ev(f"{NABIH_ACCDB}: tables 'Core loss database' + 'Bm vs Pv table'",
                                                     "Sinusoidal large-signal core-loss measurement of ferrite toroids (Virginia Tech CPES)."),
            data_class="experimental_measurement", source_id=NABIH_ID, quantity="core_loss",
            columns=[column("frequency", "excitation frequency of the test", "MHz"),
                     column("flux_density_peak", "Bm: peak flux density of the sinusoidal excitation", "mT"),
                     column("loss_density", "Pv: core loss per unit volume", "kW/m3"),
                     column("temperature", "test temperature", "degC"),
                     column("dc_bias_field", "DC bias; the database states 0 ampere-turns for every test, i.e. 0 A/m", "A-turn"),
                     {"name": "waveform", "unit": "", "definition": "excitation waveform (database field 'Excitation')"},
                     {"name": "test_id", "unit": "", "definition": "ID of the test in the database (links the points of one frequency and core)"},
                     {"name": "core_outer_diameter", "unit": "m", "original_unit": "mm", "definition": "toroid outer diameter"},
                     {"name": "core_inner_diameter", "unit": "m", "original_unit": "mm", "definition": "toroid inner diameter"},
                     {"name": "core_height", "unit": "m", "original_unit": "mm", "definition": "toroid height"}],
            rows=rows, writer=writer, provenance=_prov("nabih", NABIH_COMMIT, norm, [mid]), method="sinusoidal excitation, large signal",
            specimen=f"{grade} toroid(s); {n_tests} test(s); geometry per row", description=f"{len(rows)} P_v(B_m) points of {n_tests} tests.",
            dataset_ref="dataset.nabih-magneto", underlying_source=f"nabih:{slug(grade)}", extra={"manufacturer_attribution": "grade name"}))
        summary.rows("core_loss", len(rows))
    # --- permeability / permittivity --------------------------------------------------------------------------------
    for table_name, rows_in, quantity, real, imag in (("permeability", mu, "complex_permeability", "real permeability", "Imaginary permeability"),
                                                      ("permittivity", eps, "complex_permittivity", "real permittivity", "Imaginary permittivity")):
        grouped: dict[str, list[list[float]]] = {}
        for r in rows_in:
            if not r["CID"]:
                summary.exclude(table_name, "row without a characteristics ID (CID): cannot be attributed to a material")
                continue
            grouped.setdefault(r["CID"], []).append([dec(r["f (MHz)"], "1e6"), dec(r[real]), dec(r[imag])])
        for cid in sorted(grouped, key=int):
            ch = chars[cid]
            grade = ch["Material"]
            notes = _clean_html(ch["Notes"])
            from_datasheet = table_name == "permeability" and "from datasheet" in notes.lower()
            mid = material(grade)
            prefix = "mu" if quantity == "complex_permeability" else "eps"
            measured = (f"Small-signal {table_name} measurement at {ch['Temp (deg C)']} degC; {notes}" if not from_datasheet else
                        f"Manufacturer datasheet curve entered in the database ('{notes}'); measured by the manufacturer, not by the source.")
            measurements.append(measurement_record(
                rid=f"measurement.nabih-{slug(grade)}-{quantity.replace('_', '-')}", name=f"Nabih {grade} {quantity.replace('_', ' ')}",
                material_ref=mid, license=NABIH_LICENSE, evidence_items=ev(f"{NABIH_ACCDB}: table '{table_name}', CID {cid}", measured),
                data_class="manufacturer_datasheet" if from_datasheet else "experimental_measurement", source_id=NABIH_ID, quantity=quantity,
                columns=[column("frequency", "measurement frequency", "MHz"), column(f"{prefix}_real", f"real part of the relative {table_name}"),
                         column(f"{prefix}_imag", f"imaginary part of the relative {table_name} (as published)")],
                rows=grouped[cid], writer=writer, provenance=_prov("nabih", NABIH_COMMIT, norm, [mid]),
                method=notes or None, specimen=f"OD {ch['OD (mm)']} mm, ID {ch['ID (mm)']} mm, H {ch['H (mm)']} mm toroid; plate "
                                               f"{ch['W (mm)']} x {ch['L (mm)']} x {ch['He (mm)']} mm; measured by {ch['Author']}",
                conditions={"temperature": {"min": dec(ch["Temp (deg C)"], "1", "273.15"), "max": dec(ch["Temp (deg C)"], "1", "273.15"), "unit": "K"}},
                flags=("datasheet_curve_in_lab_database",) if from_datasheet else (), dataset_ref="dataset.nabih-magneto",
                underlying_source=f"nabih:{slug(grade)}-{prefix}", extra={"characteristics_id": int(cid)}))
            summary.rows(quantity, len(grouped[cid]))
    summary.count("magnetic_material", len(materials))
    summary.count("measurement", len(measurements))
    dataset = dataset_record(
        rid="dataset.nabih-magneto", name="Nabih core characterisation database", source_id=NABIH_ID,
        title="Core characterization data set: loss, permeability and permittivity", license=NABIH_LICENSE, url=NABIH_REPO,
        version=NABIH_COMMIT, source_commit=NABIH_COMMIT, publisher="Ahmed Nabih, Virginia Tech (GitHub asalahnabih)", redistribution=True,
        record_count=len(measurements), citation=NABIH_CITATION, checksums=_NABIH_RAW, materials=sorted(materials),
        description="MHz ferrite core loss (sinusoidal, 25 degC, no DC bias) and small-signal permeability / permittivity; independent "
                    "laboratory data except the DMR96A permeability, which the database marks as a datasheet curve.",
        evidence_items=ev("LICENSE"), provenance=_prov("nabih", NABIH_COMMIT, norm))
    shard_files = {"materials": writer.shards([materials[g] for g in sorted(materials)], "materials", "materials", 100),
                   "measurements": writer.shards(measurements, "measurements", "measurements", 100)}
    writer.json("datasets/nabih-magneto.json", dataset)
    return finish_source(writer, summary, shard_files, ["datasets/nabih-magneto.json"])


# =================================================================================================
# Guillod N87 dielectric (MIT)
# =================================================================================================
GUILLOD_ID = "guillod_n87_dielectric"
GUILLOD_DIR = "guillod-n87-dielectric"
GUILLOD_COMMIT = "dd06889876ca7d360ae3c4cf6da6c507ba5fef2c"
GUILLOD_TIME = "2025-11-05T14:17:19Z"
GUILLOD_REPO = "https://github.com/otvam/large_signal_ferrite_apec24"
GUILLOD_LICENSE = "MIT"
GUILLOD_CITATION = ("T. Guillod, W. V. R. Roberts, C. R. Sullivan, 'Characterization and Impact of Large-Signal Dielectric Properties in MnZn "
                    "Ferrites', IEEE APEC 2024, doi:10.1109/APEC48139.2024.10509366; data: github.com/otvam/large_signal_ferrite_apec24 (MIT)")
_GUILLOD_MATS = {"dataset/N87_eps_large/data.mat": "ef18d977124ef6d6eb182c44509e39fcee1639de3e49813d039fa3f4d0a6e3fc",
                 "dataset/N87_eps_small/data.mat": "cb5dd86a816a0ca3336804ac1e7350242846360c99da7a541d1d9bd9c6731fc4",
                 "dataset/N87_eps_temperature/data.mat": "3bdf594d1e1c772d68dc5d40f034b790cf8ad0bdba1f5cdb67375d3627e44235"}


def guillod_raw_files() -> list[RawFile]:
    return [RawFile(name, f"https://raw.githubusercontent.com/otvam/large_signal_ferrite_apec24/{GUILLOD_COMMIT}/{name}", sha, None, GUILLOD_LICENSE)
            for name, sha in _GUILLOD_MATS.items()]


def run_guillod(output_root: Path, *, local_dir: str | Path | None = None, allow_download: bool = False, cache: Any = None) -> ImportSummary:
    from aipe_magnetics_database.ingest.readers import read_arrays
    files = guillod_raw_files()
    paths = obtain(GUILLOD_ID, files, local_dir=local_dir, allow_download=allow_download, cache=cache, commit=GUILLOD_COMMIT)
    summary = ImportSummary(GUILLOD_ID, GUILLOD_COMMIT, GUILLOD_LICENSE, "permitted (MIT)")
    summary.inputs = [{"path": f.name, "url": f.url, "sha256": f.sha256} for f in files]
    summary.notes.append("N87_mu_small (TDK datasheet curve) and N87_mu_large (computed from MagNet data) are not imported: neither is a "
                         "measurement of this source.")
    writer = SourceWriter(output_root / GUILLOD_DIR, summary)
    norm = ("MATLAB data.mat variables read with scipy.io (float64, unmodified); degC -> K; electric field and conductivity in the SI units "
            "used by the authors' solver (code prints 1e-3*E as V/mm; sigma enters the complex permittivity with epsilon_0).")

    def ev(locator: str, what: str | None = None) -> list[dict[str, Any]]:
        items = [evidence("evidence.guillod-source", "source", "otvam/large_signal_ferrite_apec24", "MIT-licensed dataset of the APEC 2024 paper.",
                          GUILLOD_TIME, GUILLOD_LICENSE, url=f"{GUILLOD_REPO}/tree/{GUILLOD_COMMIT}", version=GUILLOD_COMMIT, locator=locator,
                          sha256=_GUILLOD_MATS.get(locator))]
        if what:
            items.append(evidence("evidence.guillod-measurement", "measurement", "Dartmouth College (Sullivan group)", what, GUILLOD_TIME,
                                  GUILLOD_LICENSE, url=GUILLOD_REPO, version=GUILLOD_COMMIT, locator=locator))
        return items

    mid = "material.guillod-tdk-n87"
    mat = _identity(mid, "TDK N87", "N87", GUILLOD_LICENSE, ev("README.md"), "dataset.guillod-n87-dielectric", _prov("guillod", GUILLOD_COMMIT, norm),
                    GUILLOD_ID, "EPCOS TDK N87 specimen material of the APEC 2024 dielectric characterisation.")
    cols_small = [column("frequency", "measurement frequency"), column("temperature", "specimen temperature (README)", "degC"),
                  column("eps_real", "real part of the relative permittivity"),
                  {"name": "conductivity", "unit": "S/m", "definition": "real conductivity (loss term of the complex permittivity)"}]
    flags = ("units_from_authors_code",)
    small = read_arrays(paths["dataset/N87_eps_small/data.mat"])
    temp = read_arrays(paths["dataset/N87_eps_temperature/data.mat"])
    large = read_arrays(paths["dataset/N87_eps_large/data.mat"])
    measurements = []
    rows = [[float(f), 298.15, float(e), float(s)] for f, e, s in zip(small["f_vec"], small["eps_vec"], small["sigma_vec"])]
    measurements.append(measurement_record(
        rid="measurement.guillod-n87-small-signal-permittivity", name="N87 small-signal permittivity and conductivity (Guillod, APEC 2024)",
        material_ref=mid, license=GUILLOD_LICENSE, evidence_items=ev("dataset/N87_eps_small/data.mat", "Impedance measurement of an I14/1.5/5 "
                                                                       "core at 25 degC, peak electric field below 0.05 V/mm."),
        data_class="experimental_measurement", source_id=GUILLOD_ID, quantity="dielectric_properties", columns=cols_small, rows=rows,
        writer=writer, provenance=_prov("guillod", GUILLOD_COMMIT, norm, [mid]), specimen="I14/1.5/5 core", flags=flags,
        method="small signal (peak electric field < 0.05 V/mm)", dataset_ref="dataset.guillod-n87-dielectric", underlying_source="guillod:n87-eps-small"))
    summary.rows("dielectric_properties", len(rows))
    rows = []
    for t in (25, 55, 85):
        rows += [[float(f), t + 273.15, float(e), float(s)] for f, e, s in zip(temp["f_vec"], temp[f"eps_{t}_vec"], temp[f"sigma_{t}_vec"])]
    measurements.append(measurement_record(
        rid="measurement.guillod-n87-permittivity-vs-temperature", name="N87 permittivity and conductivity at 25 / 55 / 85 degC (Guillod)",
        material_ref=mid, license=GUILLOD_LICENSE, evidence_items=ev("dataset/N87_eps_temperature/data.mat", "Small-signal impedance measurement "
                                                                       "of an I14/1.5/5 core at 25, 55 and 85 degC."),
        data_class="experimental_measurement", source_id=GUILLOD_ID, quantity="dielectric_properties", columns=cols_small, rows=rows,
        writer=writer, provenance=_prov("guillod", GUILLOD_COMMIT, norm, [mid]), specimen="I14/1.5/5 core", flags=flags,
        method="small signal (peak electric field < 0.05 V/mm)", dataset_ref="dataset.guillod-n87-dielectric", underlying_source="guillod:n87-eps-temperature"))
    summary.rows("dielectric_properties", len(rows))
    rows = []
    for i, f in enumerate(large["f_vec"]):
        rows += [[float(f), 298.15, float(large["E_mat"][i][j]), float(large["eps_mat"][i][j]), float(large["sigma_mat"][i][j])]
                 for j in range(large["E_mat"].shape[1])]
    measurements.append(measurement_record(
        rid="measurement.guillod-n87-large-signal-permittivity", name="N87 large-signal permittivity and conductivity vs electric field (Guillod)",
        material_ref=mid, license=GUILLOD_LICENSE, evidence_items=ev("dataset/N87_eps_large/data.mat", "Large-signal measurement of an I14/1.5/5 "
                                                                       "core at 25 degC for increasing peak electric field."),
        data_class="experimental_measurement", source_id=GUILLOD_ID, quantity="dielectric_properties",
        columns=[cols_small[0], cols_small[1], {"name": "electric_field_peak", "unit": "V/m", "definition": "peak electric field of the measurement"},
                 cols_small[2], cols_small[3]], rows=rows, writer=writer, provenance=_prov("guillod", GUILLOD_COMMIT, norm, [mid]),
        specimen="I14/1.5/5 core", flags=flags, method="large signal", dataset_ref="dataset.guillod-n87-dielectric",
        underlying_source="guillod:n87-eps-large"))
    summary.rows("dielectric_properties", len(rows))
    summary.count("magnetic_material", 1)
    summary.count("measurement", len(measurements))
    dataset = dataset_record(
        rid="dataset.guillod-n87-dielectric", name="Guillod N87 large-signal dielectric properties", source_id=GUILLOD_ID,
        title="Large-Signal Properties in MnZn Ferrites: Code and Data (datasets)", license=GUILLOD_LICENSE, url=GUILLOD_REPO, version=GUILLOD_COMMIT,
        source_commit=GUILLOD_COMMIT, publisher="Thomas Guillod, Dartmouth College", redistribution=True, record_count=len(measurements),
        citation=GUILLOD_CITATION, checksums={Path(k).parent.name + "/data.mat": v for k, v in _GUILLOD_MATS.items()}, materials=["TDK N87"],
        description="Real permittivity and conductivity of TDK N87 (small signal, vs temperature, vs peak electric field); independent "
                    "laboratory measurements.", evidence_items=ev("LICENSE.md"), provenance=_prov("guillod", GUILLOD_COMMIT, norm))
    shard_files = {"materials": writer.shards([mat], "materials", "materials", 100),
                   "measurements": writer.shards(measurements, "measurements", "measurements", 100)}
    writer.json("datasets/guillod-n87-dielectric.json", dataset)
    return finish_source(writer, summary, shard_files, ["datasets/guillod-n87-dielectric.json"])


# =================================================================================================
# Zenodo 7368936 - composite waveform hypothesis, MagNet N87 (CC-BY-4.0)
# =================================================================================================
CWH_ID = "zenodo_7368936_cwh_n87"
CWH_DIR = "zenodo-7368936-cwh-n87"
CWH_DOI = "10.5281/zenodo.7368936"
CWH_TIME = "2022-11-29T00:00:00Z"
CWH_LICENSE = "CC-BY-4.0"
CWH_CITATION = ("T. Guillod, J. S. Lee, H. Li, S. Wang, M. Chen, C. R. Sullivan, 'Calculation of Ferrite Core Losses with Arbitrary Waveforms "
                "using the Composite Waveform Hypothesis: Reproducibility Dataset', Zenodo, 2022, doi:10.5281/zenodo.7368936 (CC BY 4.0); "
                "measurements from MagNet (Princeton / Dartmouth / Plexim)")
_CWH = {"N87_ambient_temperature.csv": ("fcc76d7f9c5cf3ca3a33d147e6a5f6a29d55df4f5f43665b06219db0535f0eb7", 3218508),
        "N87_variable_temperature.csv": ("83609c3341b5e02d258e238a6114a3b24c69c9de42f5bcd80c0b4ffb49ea9455", 7761548),
        "dataset_metadata.csv": ("6844cbf62228260de93f62efc98bc2d036d0a2f8c8a0f5bcce84b48c024614cf", 550),
        "readme.txt": ("38516c005e4e7d8193f900c2859d8a4330d754d4a6d9e6cc3f522239f59c0b6b", 2745),
        "N87_ambient_temperature.mat": ("d74d390bbe6f728b13dfe2898319ae80be9fc409c1eecc1206fde6b80cb8a263", 986472),
        "N87_variable_temperature.mat": ("a5cd58ef9bce4c251559aa82a125946cb7c69be40fd8259e668e0b2d2a3b094d", 2524824),
        "N87_ambient_temperature.h5": ("edcc57d3cd76cd9fd09c5449b254491e07d657d5e531ef78598b1a3b83a7d296", 2585557),
        "N87_variable_temperature.h5": ("5c22e9c880934be977c93d8d3ec5989b178b6b092fb6cba91958f232c0c1c871", 4725789),
        "test_python.py": ("8b84877935b30844f18418597f0c03c07b74bb010fab6f5468050b6e0596d729", 1033),
        "test_matlab.m": ("1ee1c87f35fdf1edf207243dcff51354a88ebd0d5f226aade003c6d38c712a68", 880)}
CWH_SETS = {"N87_ambient_temperature.csv": ("ambient", "R22.1X13.7X7.9 toroid, measured 2022-02-01", None),
            "N87_variable_temperature.csv": ("variable-temperature", "R34.0X20.5X12.5 toroid, measured 2022-07-14", "princeton-magnet:n87")}


def cwh_raw_files() -> list[RawFile]:
    return [RawFile(name, f"https://zenodo.org/api/records/7368936/files/{name}/content", sha, size, CWH_LICENSE) for name, (sha, size) in _CWH.items()]


def run_cwh(output_root: Path, *, local_dir: str | Path | None = None, allow_download: bool = False, cache: Any = None) -> ImportSummary:
    files = cwh_raw_files()
    paths = obtain(CWH_ID, files, local_dir=local_dir, allow_download=allow_download, cache=cache)
    summary = ImportSummary(CWH_ID, f"doi:{CWH_DOI}", CWH_LICENSE, "permitted (CC-BY-4.0, attribution required)")
    summary.inputs = [{"path": f.name, "url": f.url, "sha256": f.sha256} for f in files]
    summary.notes.append("CSV used as input; the .mat (MATLAB table objects) and .h5 (pandas) copies are archived as raw files only.")
    writer = SourceWriter(output_root / CWH_DIR, summary)
    norm = ("CSV columns mapped one to one (dataset_metadata.csv); T degC -> K; flux_density_peak = B_pkpk / 2, i.e. (max B - min B) / 2 "
            "(the MagNet amplitude definition; B_pkpk equals max - min of B_vec to 1e-15 T); the published B_pkpk and the five-point "
            "piecewise-linear waveform (t_vec, B_vec) are kept verbatim.")

    def ev(locator: str, measured: bool = False) -> list[dict[str, Any]]:
        items = [evidence("evidence.cwh-source", "source", f"Zenodo {CWH_DOI}", "CC-BY-4.0 reproducibility dataset.", CWH_TIME, CWH_LICENSE,
                          url="https://zenodo.org/records/7368936", version=CWH_DOI, locator=locator, sha256=_CWH.get(locator, (None,))[0])]
        if measured:
            items.append(evidence("evidence.cwh-measurement", "measurement", "MagNet (Princeton University / Dartmouth College)",
                                  "Large-signal core-loss measurement of a toroid under piecewise-linear excitation.", CWH_TIME, CWH_LICENSE,
                                  url="https://mag-net.princeton.edu/", version=CWH_DOI, locator=locator))
        return items

    mid = "material.zenodo-7368936-tdk-n87"
    mat = _identity(mid, "TDK N87", "N87", CWH_LICENSE, ev("readme.txt"), "dataset.zenodo-7368936", _prov("cwh", CWH_DOI, norm), CWH_ID,
                    "EPCOS TDK N87 measured by MagNet for the composite waveform hypothesis paper.")
    measurements = []
    for name, (tag, specimen, underlying) in CWH_SETS.items():
        rows = []
        with open(paths[name], encoding="utf-8", newline="") as handle:
            for r in csv.DictReader(handle):
                b_pkpk = float(r["B_pkpk"])
                rows.append([float(r["f"]), b_pkpk / 2.0, b_pkpk, dec(r["T"], "1", "273.15"), float(r["P"]), r["type"], r["paper"] == "1",
                             *[float(r[f"t_vec_{i}"]) for i in range(1, 6)], *[float(r[f"B_vec_{i}"]) for i in range(1, 6)]])
        columns = [column("frequency", "frequency of the waveform"),
                   column("flux_density_peak", "(max B - min B) / 2 = B_pkpk / 2 (derived from the published peak-to-peak value)"),
                   {"name": "flux_density_peak_to_peak", "unit": "T", "definition": "B_pkpk: peak-to-peak flux density (as published)"},
                   column("temperature", "temperature of the core material", "degC"),
                   column("loss_density", "measured loss density for the waveform"),
                   {"name": "waveform", "unit": "", "definition": "waveform type as published (symmetric / asymmetric triangular / trapezoidal)"},
                   {"name": "used_in_paper", "unit": "", "definition": "true when the iGCC could be evaluated without extrapolation (paper subset)"},
                   *[{"name": f"waveform_t{i}", "unit": "s", "definition": f"time of corner {i} of the piecewise-linear B waveform"} for i in range(1, 6)],
                   *[{"name": f"waveform_b{i}", "unit": "T", "definition": f"flux density at corner {i} of the piecewise-linear B waveform"}
                     for i in range(1, 6)]]
        flags = ("repeats_princeton_magnet_measurements",) if underlying else ()
        measurements.append(measurement_record(
            rid=f"measurement.zenodo-7368936-n87-{tag}-core-loss", name=f"MagNet N87 piecewise-linear core loss, {tag} ({specimen.split(',')[0]})",
            material_ref=mid, license=CWH_LICENSE, evidence_items=ev(name, True), data_class="experimental_measurement", source_id=CWH_ID,
            quantity="core_loss", columns=columns, rows=rows, writer=writer, provenance=_prov("cwh", CWH_DOI, norm, [mid]),
            method="MagNet large-signal measurement, piecewise-linear (triangular / trapezoidal) excitation", specimen=specimen, flags=flags,
            dataset_ref="dataset.zenodo-7368936", underlying_source=underlying or f"zenodo-7368936:n87-{tag}",
            quality_notes=["Same MagNet measurement campaign as the packaged Princeton MagNet N87 database, processed separately: in a "
                           "3,000-row sample about two thirds of the rows match a packaged row within 0.1 % in f, 1 % in B and 2 % in P_v, "
                           "while the canonical exact-match rule (0.1 % in f, B and P_v) matches about 1 %. Linked as a duplicate so it is "
                           "not counted as an independent measurement."] if underlying else (),
            commit_sidecar=underlying is None))  # the repeat of MagNet stays out of Git (tables-local/, published on Hugging Face)
        summary.rows("core_loss", len(rows))
    summary.count("magnetic_material", 1)
    summary.count("measurement", len(measurements))
    dataset = dataset_record(
        rid="dataset.zenodo-7368936", name="Composite waveform hypothesis N87 dataset", source_id=CWH_ID,
        title="Calculation of Ferrite Core Losses with Arbitrary Waveforms using the Composite Waveform Hypothesis: Reproducibility Dataset",
        license=CWH_LICENSE, url="https://zenodo.org/records/7368936", version=CWH_DOI, publisher="Zenodo", redistribution=True,
        record_count=len(measurements), citation=CWH_CITATION, checksums={k: v[0] for k, v in _CWH.items()}, materials=["TDK N87"],
        description="MagNet N87 loss under triangular / trapezoidal excitation (50-500 kHz, 25-90 degC) with the exact piecewise-linear "
                    "waveforms; the variable-temperature set repeats packaged MagNet measurements.", evidence_items=ev("readme.txt"),
        provenance=_prov("cwh", CWH_DOI, norm))
    shard_files = {"materials": writer.shards([mat], "materials", "materials", 100),
                   "measurements": writer.shards(measurements, "measurements", "measurements", 100)}
    writer.json("datasets/zenodo-7368936.json", dataset)
    return finish_source(writer, summary, shard_files, ["datasets/zenodo-7368936.json"])


# =================================================================================================
# Zenodo 10725890 - INRIM first magnetisation curves (CC-BY-4.0)
# =================================================================================================
INRIM_ID = "zenodo_10725890_inrim"
INRIM_DIR = "zenodo-10725890-inrim"
INRIM_DOI = "10.5281/zenodo.10725890"
INRIM_TIME = "2024-02-29T00:00:00Z"
INRIM_LICENSE = "CC-BY-4.0"
INRIM_FILE = "HBJ data_file_res_Zenodo.dat"
INRIM_PINNED = "1fe4d6b98d5309940a8c2db5a94aa3ac3b1a31c69f8c6ddcc4d9f1505791607a"
INRIM_CITATION = ("M. Pasquale, 'Examples of first magnetization curves of ferromagnetic samples', Zenodo, v1.0.0, 2024, "
                  "doi:10.5281/zenodo.10725890 (CC BY 4.0)")
_INRIM_COLS = {"H": ("field", "A/m", "magnetic field strength H"), "B": ("flux_density", "T", "magnetic flux density B"),
               "J": ("polarization", "T", "magnetic polarization J")}


def inrim_raw_files() -> list[RawFile]:
    return [RawFile(INRIM_FILE, "https://zenodo.org/api/records/10725890/files/HBJ%20data_file_res_Zenodo.dat/content", INRIM_PINNED, 16705, INRIM_LICENSE)]


def parse_inrim(text: str) -> list[tuple[str, list[str], list[list[str]]]]:
    """Blocks of ``(material label, column letters, rows of decimal text)``; a block starts with a label line followed by a line
    of column letters (``H B``, ``J H`` ...)."""
    lines = [ln.strip() for ln in text.replace("\r", "").split("\n")]
    blocks: list[tuple[str, list[str], list[list[str]]]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if re.fullmatch(r"[HBJ](\s+[HBJ])+", nxt):
            label, cols, rows = line, nxt.split(), []
            i += 2
            while i < len(lines) and lines[i] and re.fullmatch(r"[-+0-9.eE]+(\s+[-+0-9.eE]+)*", lines[i]):
                rows.append(lines[i].split())
                i += 1
            blocks.append((label, cols, rows))
        elif re.fullmatch(r"[HBJ](\s+[HBJ])+", line):  # column header without a label: belongs to the previous material label
            label = blocks[-1][0] + " (second curve)" if blocks else "unlabelled"
            cols, rows = line.split(), []
            i += 1
            while i < len(lines) and lines[i] and re.fullmatch(r"[-+0-9.eE]+(\s+[-+0-9.eE]+)*", lines[i]):
                rows.append(lines[i].split())
                i += 1
            blocks.append((label, cols, rows))
        else:
            i += 1
    return blocks


def run_inrim(output_root: Path, *, local_dir: str | Path | None = None, allow_download: bool = False, cache: Any = None) -> ImportSummary:
    files = inrim_raw_files()
    paths = obtain(INRIM_ID, files, local_dir=local_dir, allow_download=allow_download, cache=cache)
    summary = ImportSummary(INRIM_ID, f"doi:{INRIM_DOI} v1.0.0", INRIM_LICENSE, "permitted (CC-BY-4.0, attribution required)")
    summary.inputs = [{"path": f.name, "url": f.url, "sha256": f.sha256} for f in files]
    writer = SourceWriter(output_root / INRIM_DIR, summary)
    norm = ("Whitespace-separated blocks (label line, column-letter line, numeric rows); columns reordered to a fixed layout, values kept "
            "verbatim. The file states no units: H is stored as A/m and B / J as T, consistent with the magnitudes (records flagged "
            "units_not_stated).")
    blocks = parse_inrim(paths[INRIM_FILE].read_text(encoding="utf-8", errors="replace"))

    def ev(locator: str, measured: bool = False) -> list[dict[str, Any]]:
        items = [evidence("evidence.inrim-source", "source", f"Zenodo {INRIM_DOI}", "CC-BY-4.0 INRIM data file.", INRIM_TIME, INRIM_LICENSE,
                          url="https://zenodo.org/records/10725890", version="1.0.0", locator=locator, sha256=INRIM_PINNED)]
        if measured:
            items.append(evidence("evidence.inrim-measurement", "measurement", "INRIM (Istituto Nazionale di Ricerca Metrologica)",
                                  "DC first magnetisation curve measured on a ferromagnetic sample.", INRIM_TIME, INRIM_LICENSE,
                                  url="https://zenodo.org/records/10725890", version="1.0.0", locator=locator))
        return items

    materials, measurements, seen = [], [], set()
    for n, (label, cols, rows) in enumerate(blocks, start=1):
        key = slug(label)
        while key in seen:
            key += "-b"
        seen.add(key)
        mid = f"material.inrim-{key}"
        ext = {"aipe.inrim": {"label": label, "block": n}}
        rec = _identity(mid, f"INRIM sample {label}", label, INRIM_LICENSE, ev(f"{INRIM_FILE}: block {n} '{label}'"), "dataset.zenodo-10725890",
                        _prov("inrim", INRIM_DOI, norm), INRIM_ID, f"Sample '{label}' of the INRIM first magnetisation curve examples "
                                                                   "(label as published; no manufacturer or composition beyond the label).",
                        flags=("laboratory_specimen",), ext=ext)
        materials.append(rec)
        order = [c for c in ("H", "B", "J") if c in cols]
        bad = [r for r in rows if len(r) != len(cols)]
        if bad:
            summary.exclude(f"{INRIM_FILE} block {n}", "row with a missing value", len(bad))
        table = [[float(r[cols.index(c)]) for c in order] for r in rows if len(r) == len(cols)]
        measurements.append(measurement_record(
            rid=f"measurement.inrim-{key}-first-magnetization", name=f"INRIM {label} first magnetisation curve", material_ref=mid,
            license=INRIM_LICENSE, evidence_items=ev(f"{INRIM_FILE}: block {n} '{label}'", True), data_class="experimental_measurement",
            source_id=INRIM_ID, quantity="bh_curve",
            columns=[{"name": _INRIM_COLS[c][0], "unit": _INRIM_COLS[c][1], "definition": _INRIM_COLS[c][2] + " (unit not stated in the file)"}
                     for c in order],
            rows=table, writer=writer, provenance=_prov("inrim", INRIM_DOI, norm, [mid]), method="DC first magnetisation (normal) curve",
            specimen=f"INRIM sample '{label}'", flags=("units_not_stated", "temperature_not_stated"), dataset_ref="dataset.zenodo-10725890",
            underlying_source=f"inrim:{key}", extra={"published_column_order": cols}))
        summary.rows("bh_curve", len(table))
    summary.count("magnetic_material", len(materials))
    summary.count("measurement", len(measurements))
    dataset = dataset_record(
        rid="dataset.zenodo-10725890", name="INRIM first magnetisation curves", source_id=INRIM_ID,
        title="Examples of first magnetization curves of ferromagnetic samples", license=INRIM_LICENSE, url="https://zenodo.org/records/10725890",
        version="1.0.0", publisher="Zenodo (M. Pasquale, INRIM)", redistribution=True, record_count=len(measurements), citation=INRIM_CITATION,
        checksums={INRIM_FILE: INRIM_PINNED}, materials=[m["properties"]["grade"] for m in materials],
        description="DC first magnetisation curves (H, B, J) of steels, irons, FeSiAl and an SMC; units not stated in the file.",
        evidence_items=ev(INRIM_FILE), provenance=_prov("inrim", INRIM_DOI, norm))
    shard_files = {"materials": writer.shards(materials, "materials", "materials", 100),
                   "measurements": writer.shards(measurements, "measurements", "measurements", 100)}
    writer.json("datasets/zenodo-10725890.json", dataset)
    return finish_source(writer, summary, shard_files, ["datasets/zenodo-10725890.json"])


IMPORTERS: dict[str, tuple[Callable[..., ImportSummary], str]] = {
    NABIH_ID: (run_nabih, NABIH_DIR), GUILLOD_ID: (run_guillod, GUILLOD_DIR), CWH_ID: (run_cwh, CWH_DIR), INRIM_ID: (run_inrim, INRIM_DIR)}

__all__ = ["CWH_ID", "GUILLOD_ID", "IMPORTERS", "INRIM_ID", "NABIH_ID", "dec", "parse_inrim", "run_cwh", "run_guillod", "run_inrim", "run_nabih"]
