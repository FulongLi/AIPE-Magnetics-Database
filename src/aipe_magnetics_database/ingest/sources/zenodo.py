"""Importers of two CC-BY-4.0 Zenodo measurement records.

* HEFMAG (EMPIR 19ENG06), DOI 10.5281/zenodo.8292861 v1.0.0 - fluxmetric loss and complex permeability (2 Hz - 10 MHz,
  0.5 - 500 mT peak induction, sinusoidal) and VNA transmission-line permeability (100 kHz - 1 GHz) of four tape-wound
  rings: two Co67Fe4B14.5Si14.5 amorphous and two Fe73Nb3Cu1Si16B7 nanocrystalline ribbons (INRiM / PoliTo / UPJS).
* Dobak et al., DOI 10.5281/zenodo.21626116 - loss of a Magnetics XFlux Ultra soft magnetic composite and a TDK N30
  Mn-Zn ferrite core at 100 mT peak induction under sinusoidal polarisation (vs frequency) and triangular polarisation
  (vs duty cycle at 1, 10 and 100 kHz).

Loss is stored exactly as published, as energy per cycle and unit volume (J/m^3); the time-averaged loss density
P_v = E * f is *not* precomputed (a consumer may derive it and must record the derivation). Neither record states the
specimen temperature or a DC bias: rows carry no temperature and records are flagged ``temperature_not_stated``.
The HEFMAG zip is read in memory (``__MACOSX`` / ``.DS_Store`` members ignored); nothing is extracted or executed.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

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
    q,
    validated,
)

LICENSE = "CC-BY-4.0"

# -------------------------------------------------------------------------------------------------
# HEFMAG
# -------------------------------------------------------------------------------------------------
HEFMAG_ID = "zenodo_8292861_hefmag"
HEFMAG_DIR = "hefmag"
HEFMAG_DOI = "10.5281/zenodo.8292861"
HEFMAG_TIME = "2023-08-28T00:00:00Z"
HEFMAG_ZIP = "EMPIR 19ENG06 HEFMAG dataset Co67 & Fmet.zip"
HEFMAG_FILES = [RawFile(HEFMAG_ZIP, "https://zenodo.org/records/8292861/files/EMPIR%2019ENG06%20HEFMAG%20dataset%20Co67%20%26%20Fmet.zip?download=1",
                        "95c457de0ad4e28a1055772678cfcbbd5791d03efa7c18f73e80966a3004758a", 120075, LICENSE)]
HEFMAG_CITATION = ("C. S. Ragusa, S. Dobak, C. Beatrice, F. Fiorillo, '(EMPIR 19ENG06 HEFMAG) Data sets of measurements of magnetic loss and "
                   "complex permeability on amorphous and nanocrystalline samples up to the MHz range', Zenodo, v1.0.0, 2023, "
                   "doi:10.5281/zenodo.8292861 (CC BY 4.0)")
SAMPLES = {
    "Co67_HT_13.65um": ("Co67Fe4B14.5Si14.5 amorphous ribbon (heat treated), 13.65 um", "amorphous", 141e-8, 7730.0, 13.65e-6, 0.65),
    "Co67_HT_d25um": ("Co67Fe4B14.5Si14.5 amorphous ribbon (heat treated), 25 um", "amorphous", 141e-8, 7730.0, 25e-6, 0.65),
    "FMET_HT_d12.90um": ("Fe73Nb3Cu1Si16B7 nanocrystalline ribbon (transverse anisotropy), 12.90 um", "nanocrystalline", 118e-8, 7200.0, 12.9e-6, 1.25),
    "FMET_HT_d20.3um": ("Fe73Nb3Cu1Si16B7 nanocrystalline ribbon (transverse anisotropy), 20.3 um", "nanocrystalline", 118e-8, 7200.0, 20.3e-6, 1.25),
}


def _numbers(text: str) -> list[list[float]]:
    rows = []
    for line in text.replace("\r", "").split("\n"):
        parts = [p for p in re.split(r"[\t ]+", line.strip()) if p]
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue  # header or stray label line
    return rows


def _peak_mT(name: str) -> float:
    token = re.match(r"Jp_(\d+)mT", name).group(1)
    return float("0." + token[1:]) if len(token) > 1 and token.startswith("0") else float(token)


def _sample_key(member: str) -> str | None:
    for key in SAMPLES:
        if f"dataset {key}/" in member:
            return key
    return None


def _hefmag_evidence(locator: str, measurement: str | None = None) -> list[dict[str, Any]]:
    items = [evidence("evidence.hefmag-source", "source", f"Zenodo {HEFMAG_DOI}", "CC-BY-4.0 HEFMAG dataset v1.0.0.", HEFMAG_TIME, LICENSE,
                      url="https://zenodo.org/records/8292861", version="1.0.0", locator=locator, sha256=HEFMAG_FILES[0].sha256)]
    if measurement:
        items.append(evidence("evidence.hefmag-measurement", "measurement", "INRiM / Politecnico di Torino / UPJS (EMPIR 19ENG06 HEFMAG)",
                              measurement, HEFMAG_TIME, LICENSE, url="https://zenodo.org/records/8292861", version="1.0.0", locator=locator))
    return items


def _prov(tool: str, doi: str, inputs: list[str] | None = None) -> dict[str, Any]:
    out = {"activity": f"normalize_{tool}", "dataset_version": doi, "tool": importer_attribution(tool),
           "normalization": "Tab-separated columns mapped as documented in the record readme; header and stray label lines skipped; "
                            "rows with empty cells excluded; mT -> T for the peak-induction condition; loss kept as J/m^3 per cycle."}
    if inputs:
        out["inputs"] = inputs
    return out


def run_hefmag(output_root: Path, *, local_dir: str | Path | None = None, allow_download: bool = False, cache: Any = None) -> ImportSummary:
    paths = obtain(HEFMAG_ID, HEFMAG_FILES, local_dir=local_dir, allow_download=allow_download, cache=cache)
    summary = ImportSummary(HEFMAG_ID, f"doi:{HEFMAG_DOI} v1.0.0", LICENSE, "permitted (CC-BY-4.0, attribution required)")
    summary.inputs = [{"path": f.name, "url": f.url, "sha256": f.sha256} for f in HEFMAG_FILES]
    writer = SourceWriter(output_root / HEFMAG_DIR, summary)
    archive = zipfile.ZipFile(paths[HEFMAG_ZIP])
    members = sorted(m for m in archive.namelist() if not m.startswith("__MACOSX") and not m.endswith(".DS_Store") and not m.endswith("/"))
    materials, measurements = [], []
    for key, (label, cls, rho, density, thickness, js) in SAMPLES.items():
        mid = f"material.hefmag-{key.lower().replace('_', '-').replace('.', 'p')}"
        materials.append(validated({
            "schema_version": "0.2.0", "id": mid, "kind": "magnetic_material", "name": f"HEFMAG sample {key}", "synthetic": False, "license": LICENSE,
            "properties": {"grade": key, "material_class": cls, "description": label,
                           "resistivity": {"value": q(rho, "Ohm*m"), "source_ref": "evidence.hefmag-source"},
                           "density": {"value": q(density, "kg/m^3"), "source_ref": "evidence.hefmag-source"}},
            "evidence_refs": ["evidence.hefmag-source"], "evidence": _hefmag_evidence(f"{key}/readme.txt"),
            "quality": {"status": "normalized", "flags": ["laboratory_specimen", "third_party_unreviewed"]}, "dataset_ref": "dataset.hefmag",
            "provenance": _prov("hefmag", HEFMAG_DOI),
            "extensions": {"aipe.data": {"data_class": "experimental_measurement", "source_id": HEFMAG_ID},
                           "aipe.hefmag": {"ribbon_thickness_m": thickness, "saturation_polarization_T": js,
                                           "specimen": "tape-wound ring, 18 mm diameter",
                                           "note": "saturation polarization Js (= mu0 Ms) as stated in the readme; not converted into Bsat"}}}))
        flux_rows: list[list[Any]] = []
        vna_rows: list[list[Any]] = []
        for member in members:
            if _sample_key(member) != key:
                continue
            base = member.rsplit("/", 1)[-1]
            text = archive.read(member).decode("utf-8", errors="replace")
            if re.match(r"Jp_\d+mT\.(txt|TXT)$", base):
                bpk = _peak_mT(base) / 1000.0
                for r in _numbers(text):
                    if len(r) != 4:
                        summary.exclude(f"{key}/{base}", "row with empty value cells")
                        continue
                    flux_rows.append([bpk, r[0], r[1], r[2], r[3]])
            elif base.lower() == "analyzer_hf.txt":
                vna_rows.extend([r[0], r[1], r[2]] for r in _numbers(text) if len(r) == 3)
        flux_rows.sort(key=lambda r: (r[0], r[1]))
        common_flags = ("temperature_not_stated",)
        measurements.append(measurement_record(
            rid=f"measurement.{mid.split('.', 1)[1]}-fluxmetric", name=f"HEFMAG {key} fluxmetric loss and permeability", material_ref=mid,
            license=LICENSE, evidence_items=_hefmag_evidence(f"{key}/Jp_*mT.txt", "Fluxmetric measurement under sinusoidal induction."),
            data_class="experimental_measurement", source_id=HEFMAG_ID, quantity="energy_loss_and_permeability",
            columns=[column("flux_density_peak", "peak induction of the measurement series (file name Jp_<X>mT)", "mT"),
                     column("frequency", "excitation frequency (sinusoidal)"), column("mu_real", "real part of the relative permeability"),
                     column("mu_imag", "imaginary part of the relative permeability"),
                     column("energy_density_per_cycle", "magnetic loss per cycle and unit volume (as published; P_v = E * f)")],
            rows=flux_rows, writer=writer, provenance=_prov("hefmag", HEFMAG_DOI, [mid]), method="fluxmetric method, sinusoidal induction",
            specimen=f"{label}; tape-wound ring, 18 mm diameter", description=f"All Jp_<X>mT files of sample {key}.", flags=common_flags,
            dataset_ref="dataset.hefmag", underlying_source=f"hefmag:{key.lower()}"))
        summary.rows("energy_loss_and_permeability", len(flux_rows))
        if vna_rows:
            measurements.append(measurement_record(
                rid=f"measurement.{mid.split('.', 1)[1]}-vna", name=f"HEFMAG {key} VNA complex permeability", material_ref=mid, license=LICENSE,
                evidence_items=_hefmag_evidence(f"{key}/Analyzer_HF.TXT", "Transmission-line (VNA, shorted 50-ohm coaxial line) measurement."),
                data_class="experimental_measurement", source_id=HEFMAG_ID, quantity="complex_permeability",
                columns=[column("frequency", "VNA sweep frequency"), column("mu_real", "real part of the relative permeability"),
                         column("mu_imag", "imaginary part of the relative permeability")],
                rows=vna_rows, writer=writer, provenance=_prov("hefmag", HEFMAG_DOI, [mid]),
                method="transmission-line method with a vector network analyser (shorted 50-ohm coaxial cable)",
                specimen=label, description=f"Analyzer_HF.TXT of sample {key} (small signal).", flags=common_flags, dataset_ref="dataset.hefmag",
                underlying_source=f"hefmag:{key.lower()}"))
            summary.rows("complex_permeability", len(vna_rows))
    summary.count("magnetic_material", len(materials))
    summary.count("measurement", len(measurements))
    dataset = dataset_record(
        rid="dataset.hefmag", name="EMPIR 19ENG06 HEFMAG amorphous and nanocrystalline data", source_id=HEFMAG_ID,
        title="(EMPIR 19ENG06 HEFMAG) Data sets of measurements of magnetic loss and complex permeability on amorphous and nanocrystalline "
              "samples up to the MHz range", license=LICENSE, url="https://zenodo.org/records/8292861", version="1.0.0",
        publisher="Zenodo", redistribution=True, record_count=len(measurements), citation=HEFMAG_CITATION,
        checksums={HEFMAG_ZIP: HEFMAG_FILES[0].sha256}, materials=list(SAMPLES), evidence_items=_hefmag_evidence("record"),
        provenance=_prov("hefmag", HEFMAG_DOI), description="Independent laboratory measurements; temperature and DC bias are not stated.")
    shard_files = {"materials": writer.shards(materials, "materials", "materials", 100),
                   "measurements": writer.shards(measurements, "measurements", "measurements", 100)}
    writer.json("datasets/hefmag.json", dataset)
    return finish_source(writer, summary, shard_files, ["datasets/hefmag.json"])


# -------------------------------------------------------------------------------------------------
# SMC / ferrite triangular (Zenodo 21626116)
# -------------------------------------------------------------------------------------------------
TRI_ID = "zenodo_21626116_smc_ferrite_triangular"
TRI_DIR = "zenodo-smc-ferrite-triangular"
TRI_DOI = "10.5281/zenodo.21626116"
TRI_TIME = "2026-07-27T00:00:00Z"
_TRI = {"SMC_sin.txt": "06edbedd4d8777cef53c3ee506c88e704bb94f5d2d04944ceeac0dbe9c806b32",
        "Ferrite_sin.txt": "018b0267ac4ab1ba56a68c6e508a863b8864ae5103cd511b7d2bdc2c91c97f52",
        "SMC_tri_1kHz.txt": "7e71c73658311487d2aa441fc8354deac61aaaed41ba30213957eaf18f9ebd71",
        "SMC_tri_10kHz.txt": "eb445188b30badc17257d8fa2d95610845286f0c2f85b15aca7f7d8096658a2c",
        "SMC_tri_100kHz.txt": "f4ee79a2e3e65c3ac5b4c92a9ff96bcd9ee95b69781a6cf856a75f3fb2f9b447",
        "Ferrite_tri_1kHz.txt": "32ce229d715433f7bb222404790781b0feaf96aa59e68b26c6568e0f176a0027",
        "Ferrite_tri_10kHz.txt": "08bc89e4739a7637937eff407e91ffae853fe7d06c8e6c22db96e31477471d0c",
        "Ferrite_tri_100kHz.txt": "7f273218aacbbd1da417e82addc6b8de88e206b95200e37c9431550001eb2af1",
        "readme.txt": "40a8251dfd8b76ab64b6050b7a9053089edff3f0cb13fce4e72223fabdddac27"}
TRI_FILES = [RawFile(name, f"https://zenodo.org/api/records/21626116/files/{name}/content", sha, None, LICENSE) for name, sha in _TRI.items()]
TRI_CITATION = ("S. Dobak, F. Onderko, J. Fuzer, P. Kollar, 'Data Sets of Magnetic Loss Measurements on Soft Magnetic Composite and Mn-Zn "
                "Ferrite under Symmetric and Asymmetric Triangular Magnetic Polarization Profiles', Zenodo, doi:10.5281/zenodo.21626116 (CC BY 4.0)")
TRI_MATERIALS = {"SMC": ("material.zenodo-21626116-xflux-ultra", "Magnetics XFlux Ultra", "XFlux Ultra", "soft magnetic composite core"),
                 "Ferrite": ("material.zenodo-21626116-tdk-n30", "TDK N30", "N30", "Mn-Zn ferrite core")}


def run_triangular(output_root: Path, *, local_dir: str | Path | None = None, allow_download: bool = False, cache: Any = None) -> ImportSummary:
    paths = obtain(TRI_ID, TRI_FILES, local_dir=local_dir, allow_download=allow_download, cache=cache)
    summary = ImportSummary(TRI_ID, f"doi:{TRI_DOI}", LICENSE, "permitted (CC-BY-4.0, attribution required)")
    summary.inputs = [{"path": f.name, "url": f.url, "sha256": f.sha256} for f in TRI_FILES]
    summary.notes.append("Zenodo metadata gives publication_date 2025-07-27 but the record was created 2026-07-27 (PDF dated 2026-06-27).")
    writer = SourceWriter(output_root / TRI_DIR, summary)

    def ev(locator: str, measured: bool) -> list[dict[str, Any]]:
        items = [evidence("evidence.zenodo-21626116-source", "source", f"Zenodo {TRI_DOI}", "CC-BY-4.0 data set record.", TRI_TIME, LICENSE,
                          url="https://zenodo.org/records/21626116", version=TRI_DOI, locator=locator, sha256=_TRI.get(locator))]
        if measured:
            items.append(evidence("evidence.zenodo-21626116-measurement", "measurement", "P. J. Safarik University in Kosice",
                                  "Digital-feedback hysteresisgraph-wattmeter measurement at 100 mT peak induction.", TRI_TIME, LICENSE,
                                  url="https://zenodo.org/records/21626116", version=TRI_DOI, locator=locator))
        return items

    materials, measurements = [], []
    for prefix, (mid, name, grade, description) in TRI_MATERIALS.items():
        materials.append(validated({
            "schema_version": "0.2.0", "id": mid, "kind": "magnetic_material", "name": name, "synthetic": False, "license": LICENSE,
            "properties": {"grade": grade, "description": f"{name} {description} measured in Zenodo record 21626116 (core geometry not given)."},
            "evidence_refs": ["evidence.zenodo-21626116-source"], "evidence": ev("readme.txt", False), "dataset_ref": "dataset.zenodo-21626116",
            "quality": {"status": "normalized", "flags": ["identity_only", "third_party_unreviewed"]}, "provenance": _prov("zenodo_triangular", TRI_DOI),
            "extensions": {"aipe.data": {"data_class": "identity", "source_id": TRI_ID}}}))
        sin_rows = [[0.1, r[0], r[1]] for r in _numbers(paths[f"{prefix}_sin.txt"].read_text(encoding="utf-8")) if len(r) == 2]
        measurements.append(measurement_record(
            rid=f"measurement.{mid.split('.', 1)[1]}-sinusoidal", name=f"{name} loss vs frequency, sinusoidal, 100 mT", material_ref=mid,
            license=LICENSE, evidence_items=ev(f"{prefix}_sin.txt", True), data_class="experimental_measurement", source_id=TRI_ID,
            quantity="energy_loss", columns=[column("flux_density_peak", "peak induction (all measurements at 100 mT)"),
                                             column("frequency", "excitation frequency"),
                                             column("energy_density_per_cycle", "magnetic loss per cycle and unit volume (as published)")],
            rows=sin_rows, writer=writer, provenance=_prov("zenodo_triangular", TRI_DOI, [mid]), method="hysteresisgraph-wattmeter, sinusoidal polarization",
            flags=("temperature_not_stated",), dataset_ref="dataset.zenodo-21626116", underlying_source=f"zenodo-21626116:{prefix.lower()}"))
        tri_rows = []
        for f_khz in (1, 10, 100):
            tri_rows.extend([0.1, f_khz * 1000.0, r[0], r[1]] for r in _numbers(paths[f"{prefix}_tri_{f_khz}kHz.txt"].read_text(encoding="utf-8"))
                            if len(r) == 2)
        measurements.append(measurement_record(
            rid=f"measurement.{mid.split('.', 1)[1]}-triangular", name=f"{name} loss vs duty cycle, triangular, 100 mT", material_ref=mid,
            license=LICENSE, evidence_items=ev(f"{prefix}_tri_*kHz.txt", True), data_class="experimental_measurement", source_id=TRI_ID,
            quantity="energy_loss", columns=[column("flux_density_peak", "peak induction (all measurements at 100 mT)"),
                                             column("frequency", "fundamental frequency (from the file name)"),
                                             column("duty", "duty cycle of the triangular polarization (0.5 = symmetric triangle)"),
                                             column("energy_density_per_cycle", "magnetic loss per cycle and unit volume (as published)")],
            rows=tri_rows, writer=writer, provenance=_prov("zenodo_triangular", TRI_DOI, [mid]),
            method="hysteresisgraph-wattmeter with rectangular induced voltage (triangular polarization)", flags=("temperature_not_stated",),
            dataset_ref="dataset.zenodo-21626116", underlying_source=f"zenodo-21626116:{prefix.lower()}"))
        summary.rows("energy_loss", len(sin_rows) + len(tri_rows))
    summary.count("magnetic_material", len(materials))
    summary.count("measurement", len(measurements))
    dataset = dataset_record(
        rid="dataset.zenodo-21626116", name="SMC and Mn-Zn ferrite loss under triangular polarization", source_id=TRI_ID,
        title="Data Sets of Magnetic Loss Measurements on Soft Magnetic Composite and Mn-Zn Ferrite under Symmetric and Asymmetric "
              "Triangular Magnetic Polarization Profiles", license=LICENSE, url="https://zenodo.org/records/21626116", version=TRI_DOI,
        publisher="Zenodo", redistribution=True, record_count=len(measurements), citation=TRI_CITATION, checksums=_TRI,
        materials=[m[1] for m in TRI_MATERIALS.values()], evidence_items=ev("readme.txt", False), provenance=_prov("zenodo_triangular", TRI_DOI),
        description="Independent laboratory measurements at 100 mT peak; temperature, DC bias and core geometry not stated.")
    shard_files = {"materials": writer.shards(materials, "materials", "materials", 100),
                   "measurements": writer.shards(measurements, "measurements", "measurements", 100)}
    writer.json("datasets/zenodo-21626116.json", dataset)
    return finish_source(writer, summary, shard_files, ["datasets/zenodo-21626116.json"])


__all__ = ["HEFMAG_FILES", "HEFMAG_ID", "TRI_FILES", "TRI_ID", "run_hefmag", "run_triangular"]
