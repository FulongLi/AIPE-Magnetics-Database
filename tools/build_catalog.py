#!/usr/bin/env python3
"""Generate the bundled catalogue records (manufacturers, materials, cores) with explicit provenance.

Every number below is transcribed from a cited source (datasheet PDF, repository file at a pinned
commit, peer-reviewed table) or derived by a documented script. The only hand-derived value is marked
as such inside its record (the UU 93/152/30 window height, 2 x the drawn 48 mm single-U window height,
stored under ``extensions['aipe.datasheet']['derived']``); nothing is estimated.
Records are written one per file under ``src/aipe_magnetics/data/catalog/<kind>/<id>.json`` and
validated against the v0.2 record standard (``aipe_magnetics.data.records``) before writing.

Re-run after changing data::

    python tools/build_catalog.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aipe_magnetics import __version__  # noqa: E402
from aipe_magnetics.common.jsonio import dump, load  # noqa: E402
from aipe_magnetics.common.quantity import Quantity  # noqa: E402
from aipe_magnetics.data.records import MagneticRecord, validate_catalogue  # noqa: E402

CATALOG_DIR = ROOT / "src" / "aipe_magnetics" / "data" / "catalog"
FITS_FILE = ROOT / "src" / "aipe_magnetics" / "data" / "derived" / "princeton-magnet-steinmetz-fits.json"
CREATED = "2026-10-19T00:00:00Z"  # authoring date of these transcriptions (not a measurement date)


def q(value: float, unit: str) -> dict[str, Any]:
    return Quantity.si(value, unit).model_dump(mode="json", exclude_none=True)


def evidence(id: str, kind: str, reference: str, description: str, *, license: str = "NOASSERTION", locator: str | None = None,
             url: str | None = None, version: str | None = None, sha256: str | None = None) -> dict[str, Any]:
    item = {"id": id, "kind": kind, "reference": reference, "description": description, "created_at": CREATED,
            "review_status": "unreviewed", "license": license}
    for key, value in (("locator", locator), ("url", url), ("version", version), ("sha256", sha256)):
        if value:
            item[key] = value
    return item


# -------------------------------------------------------------------------------------------------
# shared evidence
# -------------------------------------------------------------------------------------------------
MAGNET_COMMIT = "8a6892e91ac622e42f8b9182a6dfe9094ceea97a"
EV_MAGNET_CONSTANTS = evidence(
    "evidence.princeton-magnet-constants", "source",
    "PrincetonUniversity/magnet, src/magnet/constants.py (material_steinmetz_param, material_extra, material_manufacturers)",
    "Per-material constants published in the Princeton MagNet repository (MIT): manufacturer, initial permeability mu_r_0, "
    "recommended frequency range and the one-line application statement (material_applications); the repository's own "
    "Steinmetz/iGSE coefficients are documented there as (k_i, alpha, beta).",
    license="MIT", url="https://github.com/PrincetonUniversity/magnet/blob/" + MAGNET_COMMIT + "/src/magnet/constants.py",
    version=MAGNET_COMMIT, locator="material_steinmetz_param, material_extra, material_manufacturers, material_applications")
EV_AIPE_FIT = evidence(
    "evidence.aipe-steinmetz-fit-magnet", "analytical",
    "benchmarks/fit_steinmetz_from_magnet.py (aipe-magnetics " + __version__ + ")",
    "Classical Steinmetz parameter sets fitted by ordinary least squares in log space on sinusoidal, zero-DC-bias rows of the "
    "Princeton MagNet scalar databases, per measured temperature, 50-500 kHz, 10-300 mT. Fit metrics are stored with each set. "
    "These are AIPE-derived values, not manufacturer data.", license="NOASSERTION",
    locator="src/aipe_magnetics/data/derived/princeton-magnet-steinmetz-fits.json")

TDK_N87_URL = "https://www.tdk-electronics.tdk.com/download/528882/71e02c7b9384de1331b3f625ce4b2123/pdf-n87.pdf"
EV_TDK_N87 = evidence(
    "evidence.tdk-n87-datasheet-2025-06", "source",
    "TDK Electronics, 'Ferrites and accessories - SIFERRIT material N87', datasheet, Date: June 2025 (release 2024-02)",
    "Manufacturer datasheet; values transcribed from the material table and the 'Relative core losses PV (typical values)' table. "
    "Typical values measured by TDK on R34 toroids; not AIPE measurements.", url=TDK_N87_URL, version="June 2025",
    locator="material table p.1-2; PV typical values table")
EV_TDK_N87_FIT = evidence(
    "evidence.aipe-datasheet-fit-n87-100c", "analytical",
    "Least-squares Steinmetz fit of the four TDK N87 datasheet loss points at 100 degC (aipe-magnetics " + __version__ + ")",
    "log(Pv) = log(k) + alpha log(f) + beta log(B) fitted to (25 kHz, 200 mT, 57 kW/m3), (100 kHz, 200 mT, 375 kW/m3), "
    "(300 kHz, 100 mT, 390 kW/m3), (500 kHz, 50 mT, 215 kW/m3); residuals +1.0 %, -3.2 %, +4.7 %, -2.3 %. Derived data, valid only "
    "as a 100 degC interpolation over 25-500 kHz and 50-200 mT.", locator="see fit note")
EV_MUHLETHALER = evidence(
    "evidence.muhlethaler-2012-tpel", "source",
    "J. Muhlethaler, J. Biela, J. W. Kolar, A. Ecklebe, 'Improved core-loss calculation for magnetic components employed in power "
    "electronic systems', IEEE Trans. Power Electron., vol. 27, no. 2, pp. 964-973, Feb. 2012, doi:10.1109/TPEL.2011.2162252",
    "Table I: iGSE parameters of EPCOS N87 (R42 toroid, 25 degC, symmetric triangular flux) fitted on three operating points "
    "(dB 0.1 T at 20 kHz, 0.1 T at 50 kHz, 0.2 T at 50 kHz; dB peak-to-peak): k_i = 8.41, alpha = 1.09, beta = 2.16 (SI: W/m3, Hz, T). "
    "Sec. IX reports triangular-waveform measurements over 20-100 kHz, which is wider than the fit span. Converted to the classical k "
    "by the exact k_i relation for storage.",
    url="https://doi.org/10.1109/TPEL.2011.2162252", locator="Table I, p. 970; Sec. IX validity")


# Ferroxcube material specifications (edition 2008 Sep 01), transcribed in aipe-magnetics 0.3.0 from the
# manufacturer PDFs (SHA-256 of the retrieved files recorded); values are the specification-table entries.
FERROXCUBE_DATASHEETS: dict[str, dict[str, Any]] = {
    "3C90": {"sha256": "803f0da1bbc44a82a52f323ce7c0fcf631431d823334f0396ab68715e87fd9d6",
             "bs_mT": {25: 470, 100: 380}, "mu_i": 2300, "tc_C": 220, "density": 4800, "rho": 5.0,
             "pv": [(100, 200, 100, 450)], "fmax_note": "up to 0.2 MHz"},
    "3C94": {"sha256": "85baa0ebc78e351bff15026df6c3d7be8c8b7a02e808fa005a09b43665ae0fa9",
             "bs_mT": {25: 470, 100: 380}, "mu_i": 2300, "tc_C": 220, "density": 4800, "rho": 5.0,
             "pv": [(100, 100, 100, 50), (100, 200, 100, 350)], "fmax_note": "up to 0.3 MHz"},
}


def ferroxcube_evidence(grade: str) -> dict[str, Any]:
    data = FERROXCUBE_DATASHEETS[grade]
    return evidence(
        f"evidence.ferroxcube-{grade.lower()}-datasheet-2008-09", "source",
        f"Ferroxcube, '{grade} Material specification', data sheet, 2008 Sep 01 (supersedes data of September 2004)",
        "Manufacturer specification table: initial permeability (25 degC, <= 10 kHz, 0.25 mT), B at 1200 A/m and 10 kHz (25 and 100 degC, "
        "'approximately' values), PV typical values at 100 degC, DC resistivity, Curie temperature (lower bound) and density. "
        "Transcribed by aipe-magnetics 0.3.0; not AIPE measurements.",
        url=f"https://www.ferroxcube.com/upload/media/product/file/MDS/{grade.lower()}.pdf", version="2008 Sep 01",
        locator="'SPECIFICATIONS' table, page 2", sha256=data["sha256"])


def tdk_core_evidence(core: str, slug: str, *, title: str | None = None, coil_former_locator: str | None = None) -> dict[str, Any]:
    """Evidence item for one TDK core datasheet (October 2024 editions).

    ``title`` is the document title when it differs from the core name (the U-core sheet is titled
    'U 93/76/30 with I 93/28/30'); ``coil_former_locator`` names the coil-former table row used for AN/lN/AR,
    or ``None`` when the datasheet lists no coil former.
    """
    description = ("Manufacturer datasheet: 'Magnetic characteristics (per set)' table (sum l/A, le, Ae, Amin, Ve, approx. weight) "
                   "and outline drawing" + ("; coil-former table (AN winding cross section, lN average length of turn, AR resistance "
                                            "factor) for the ordering code named in the record." if coil_former_locator else
                                            "; the datasheet lists no coil former (no AN/lN)."))
    locator = "Magnetic characteristics (per set) table; outline drawing" + (f"; {coil_former_locator}" if coil_former_locator else "")
    return evidence(
        f"evidence.tdk-{slug.replace('_', '-')}-datasheet-2024-10", "source",
        f"TDK Electronics, 'Ferrites and accessories - {title or core}', core and accessories datasheet, Date: October 2024",
        description, url=f"https://www.tdk-electronics.tdk.com/inf/80/db/fer/{slug}.pdf", version="October 2024", locator=locator)


# -------------------------------------------------------------------------------------------------
# manufacturers
# -------------------------------------------------------------------------------------------------
def manufacturers() -> list[dict[str, Any]]:
    def rec(id: str, name: str, legal: str, brands: list[str], country: str, website: str) -> dict[str, Any]:
        ev = evidence(f"evidence.{id.split('.')[-1]}-website", "source", website, "Manufacturer identity from the public website.",
                      url=website)
        return {"schema_version": "0.2.0", "id": id, "kind": "manufacturer", "name": name, "synthetic": False,
                "license": "NOASSERTION", "properties": {"legal_name": legal, "brand_names": brands, "country": country, "website": website},
                "evidence_refs": [ev["id"]], "evidence": [ev]}
    return [
        rec("manufacturer.tdk", "TDK Electronics", "TDK Electronics AG", ["EPCOS", "SIFERRIT"], "DE", "https://www.tdk-electronics.tdk.com/"),
        rec("manufacturer.ferroxcube", "Ferroxcube", "Ferroxcube International Holding B.V.", ["Ferroxcube"], "NL", "https://www.ferroxcube.com/"),
        rec("manufacturer.fair-rite", "Fair-Rite", "Fair-Rite Products Corp.", ["Fair-Rite"], "US", "https://www.fair-rite.com/"),
    ]


# -------------------------------------------------------------------------------------------------
# materials
# -------------------------------------------------------------------------------------------------
MAGNET_MATERIALS: dict[str, dict[str, Any]] = {
    # material -> (manufacturer id, mu_r_0, f_min, f_max) from constants.py material_extra / material_manufacturers
    "3C90": {"manufacturer": "manufacturer.ferroxcube", "mu": 2300, "fmin": 25e3, "fmax": 200e3},
    "3C94": {"manufacturer": "manufacturer.ferroxcube", "mu": 2300, "fmin": 25e3, "fmax": 300e3},
    "3E6": {"manufacturer": "manufacturer.ferroxcube", "mu": 10000, "fmin": None, "fmax": None},
    "3F4": {"manufacturer": "manufacturer.ferroxcube", "mu": 900, "fmin": 25e3, "fmax": 2e6},
    "77": {"manufacturer": "manufacturer.fair-rite", "mu": 2000, "fmin": 10e3, "fmax": 100e3},
    "78": {"manufacturer": "manufacturer.fair-rite", "mu": 2300, "fmin": 25e3, "fmax": 500e3},
    "N27": {"manufacturer": "manufacturer.tdk", "mu": 2000, "fmin": 25e3, "fmax": 150e3},
    "N30": {"manufacturer": "manufacturer.tdk", "mu": 4300, "fmin": 10e3, "fmax": 400e3},
    "N49": {"manufacturer": "manufacturer.tdk", "mu": 1500, "fmin": 300e3, "fmax": 1e6},
    "N87": {"manufacturer": "manufacturer.tdk", "mu": 2200, "fmin": 25e3, "fmax": 500e3},
}
MANUFACTURER_SLUG = {"manufacturer.tdk": "tdk", "manufacturer.ferroxcube": "ferroxcube", "manufacturer.fair-rite": "fair-rite"}
# material_applications from constants.py at MAGNET_COMMIT (verbatim); the 'power' tag is derived from these statements
MAGNET_APPLICATIONS: dict[str, str] = {
    "3C90": "Power and general purpose transformers",
    "3C94": "Power and general purpose transformers",
    "3E6": "Wideband transformers and EMI-suppression filters",
    "3F4": "Power and general purpose transformers",
    "77": "High and low flux density inductive designs",
    "78": "Power applications and low loss inductive applications",
    "N27": "Power transformers",
    "N30": "Broadband transformers",
    "N49": "Power transformers",
    "N87": "Power transformers",
}


def magnet_fit_sets(material: str, fits: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    sets = []
    file_info = fits["source"]["files"].get(material)
    db_evidence = None
    if file_info:
        db_evidence = evidence(
            f"evidence.princeton-magnet-database-{material.lower()}", "measurement",
            f"PrincetonUniversity/magnet, src/magnet/data/{file_info['file']} (scalar core-loss database, {file_info['n_points']} points)",
            f"Measured sinusoidal/triangular/trapezoidal core-loss points on core {file_info['core'].get('core', '?')} "
            f"(Ae {file_info['core'].get('Ae')} m^2, le {file_info['core'].get('le')} m, Ve {file_info['core'].get('Ve')} m^3, "
            f"N1=N2={file_info['core'].get('N1')}), temperatures {file_info['temperatures_degC']} degC; Princeton Power Electronics "
            "Research Lab 2022 setup (see embedded metadata).", license="MIT", version=MAGNET_COMMIT, sha256=file_info["sha256"],
            url=f"https://github.com/PrincetonUniversity/magnet/blob/{MAGNET_COMMIT}/src/magnet/data/{file_info['file']}")
    for fit in fits["fits"]:
        if fit["material"] != material:
            continue
        t = fit["temperature_degC"]
        sets.append({
            "id": f"set.magnet-fit-{material.lower()}-{int(t)}c",
            "k": q(fit["k_W_per_m3"], "W/m^3"), "alpha": fit["alpha"], "beta": fit["beta"],
            "temperature": q(t, "degC"),
            "frequency_range": {"min": fit["frequency_range_Hz"][0], "max": fit["frequency_range_Hz"][1], "unit": "Hz"},
            "flux_density_range": {"min": fit["flux_density_range_T"][0], "max": fit["flux_density_range_T"][1], "unit": "T"},
            "excitation": "sinusoidal", "basis": "measurement_fit",
            "source_ref": db_evidence["id"] if db_evidence else None,
            "fit_metrics": {**fit["fit_metrics"], "n_points": fit["n_points"]},
            "note": (f"AIPE least-squares fit on Princeton MagNet sinusoidal zero-bias points at {t:.0f} degC; mean |rel. error| "
                     f"{fit['fit_metrics']['mean_abs_rel_error']:.1%}, max {fit['fit_metrics']['max_rel_error']:.1%}. "
                     "Validity window: the measured temperature +-5 K (selection policy), stated f and B ranges."),
        })
    return sets, db_evidence


def material_records(fits: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for grade, info in MAGNET_MATERIALS.items():
        slug = MANUFACTURER_SLUG[info["manufacturer"]]
        sets, db_ev = magnet_fit_sets(grade, fits)
        ev = [EV_MAGNET_CONSTANTS, EV_AIPE_FIT]
        if db_ev:
            ev.append(db_ev)
        props: dict[str, Any] = {
            "manufacturer_ref": info["manufacturer"], "grade": grade, "material_class": "ferrite_mnzn",
            "steinmetz_parameter_sets": sets,
        }
        if grade in FERROXCUBE_DATASHEETS:
            ds = FERROXCUBE_DATASHEETS[grade]
            ev_ds = ferroxcube_evidence(grade)
            ev.append(ev_ds)
            sid = ev_ds["id"]
            props.update({
                "initial_permeability": {"value": q(ds["mu_i"], "1"), "temperature": q(25, "degC"), "source_ref": sid,
                                         "note": f"datasheet: mu_i = {ds['mu_i']} +-20 % at 25 degC, <= 10 kHz, 0.25 mT"},
                "saturation_flux_density": [
                    {"value": q(b, "mT"), "temperature": q(t, "degC"), "field": q(1200, "A/m"), "frequency": q(10, "kHz"), "source_ref": sid,
                     "note": f"B ({t} degC) at H = 1200 A/m, f = 10 kHz ('approximately' value)"} for t, b in sorted(ds["bs_mT"].items())],
                "curie_temperature": {"value": q(ds["tc_C"], "degC"), "source_ref": sid,
                                      "note": f"datasheet states TC >= {ds['tc_C']} degC (lower bound); stored as the bound"},
                "density": {"value": q(ds["density"], "kg/m^3"), "source_ref": sid, "note": "datasheet 'approximately' value"},
                "resistivity": {"value": q(ds["rho"], "Ohm*m"), "source_ref": sid, "note": "DC, 25 degC, 'approximately' value"},
                "loss_reference_points": [
                    {"frequency": q(f, "kHz"), "flux_density_peak": q(b, "mT"), "temperature": q(t, "degC"), "loss_density": q(pv, "kW/m^3"),
                     "source_ref": sid, "basis": "datasheet_typical"} for f, b, t, pv in ds["pv"]],
            })
            if info["fmin"] is not None:
                props["recommended_frequency_range"] = {"min": info["fmin"], "max": info["fmax"], "unit": "Hz"}
            props["description"] = (f"Ferroxcube {grade} MnZn ferrite (datasheet: 'a low frequency power material for use in power and general "
                                    f"purpose transformers at frequencies {ds['fmax_note']}'). Loss parameters: Princeton MagNet "
                                    "derived Steinmetz sets (25-90 degC); datasheet values transcribed in aipe-magnetics 0.3.0. Application tag per "
                                    f"Princeton MagNet constants.py (material_applications): '{MAGNET_APPLICATIONS[grade]}'.")
        elif grade != "N87":
            props["initial_permeability"] = {"value": q(info["mu"], "1"), "source_ref": EV_MAGNET_CONSTANTS["id"],
                                             "note": "mu_r_0 as listed in Princeton MagNet constants.py (material_extra); conditions not stated there"}
            if info["fmin"] is not None:
                props["recommended_frequency_range"] = {"min": info["fmin"], "max": info["fmax"], "unit": "Hz"}
            props["description"] = (f"MnZn ferrite {grade}; application per Princeton MagNet constants.py (material_applications): "
                                    f"'{MAGNET_APPLICATIONS[grade]}'. Datasheet parameters (Bsat, Curie temperature, density) are not yet "
                                    "transcribed for this grade; only Princeton MagNet derived loss parameters are available.")
        else:
            ev += [EV_TDK_N87, EV_TDK_N87_FIT, EV_MUHLETHALER]
            props.update({
                "initial_permeability": {"value": q(2200, "1"), "temperature": q(25, "degC"), "source_ref": EV_TDK_N87["id"],
                                         "note": "datasheet: mu_i = 2200 +-25 % at 25 degC"},
                "saturation_flux_density": [
                    {"value": q(490, "mT"), "temperature": q(25, "degC"), "field": q(1200, "A/m"), "frequency": q(10, "kHz"),
                     "source_ref": EV_TDK_N87["id"], "note": "BS (25 degC) at H = 1200 A/m, f = 10 kHz"},
                    {"value": q(390, "mT"), "temperature": q(100, "degC"), "field": q(1200, "A/m"), "frequency": q(10, "kHz"),
                     "source_ref": EV_TDK_N87["id"], "note": "BS (100 degC) at H = 1200 A/m, f = 10 kHz"},
                ],
                "coercivity": [
                    {"value": q(21, "A/m"), "temperature": q(25, "degC"), "frequency": q(10, "kHz"), "source_ref": EV_TDK_N87["id"]},
                    {"value": q(13, "A/m"), "temperature": q(100, "degC"), "frequency": q(10, "kHz"), "source_ref": EV_TDK_N87["id"]},
                ],
                "curie_temperature": {"value": q(210, "degC"), "source_ref": EV_TDK_N87["id"],
                                      "note": "datasheet states TC > 210 degC (lower bound); stored as the bound"},
                "density": {"value": q(4850, "kg/m^3"), "source_ref": EV_TDK_N87["id"], "note": "typical value"},
                "resistivity": {"value": q(10, "Ohm*m"), "source_ref": EV_TDK_N87["id"]},
                "recommended_frequency_range": {"min": 25e3, "max": 500e3, "unit": "Hz"},
                "loss_reference_points": [
                    {"frequency": q(25, "kHz"), "flux_density_peak": q(200, "mT"), "temperature": q(100, "degC"),
                     "loss_density": q(57, "kW/m^3"), "source_ref": EV_TDK_N87["id"], "basis": "datasheet_typical"},
                    {"frequency": q(100, "kHz"), "flux_density_peak": q(200, "mT"), "temperature": q(100, "degC"),
                     "loss_density": q(375, "kW/m^3"), "source_ref": EV_TDK_N87["id"], "basis": "datasheet_typical"},
                    {"frequency": q(300, "kHz"), "flux_density_peak": q(100, "mT"), "temperature": q(100, "degC"),
                     "loss_density": q(390, "kW/m^3"), "source_ref": EV_TDK_N87["id"], "basis": "datasheet_typical"},
                    {"frequency": q(500, "kHz"), "flux_density_peak": q(50, "mT"), "temperature": q(100, "degC"),
                     "loss_density": q(215, "kW/m^3"), "source_ref": EV_TDK_N87["id"], "basis": "datasheet_typical"},
                ],
                "description": ("TDK/EPCOS SIFERRIT N87, MnZn power ferrite for power transformers (RM, P, PM, ETD, EFD, E, ER, EP, EQ, "
                                "ELP, U, toroid shapes)."),
            })
            # datasheet-derived 100 degC set (TDK typical values)
            props["steinmetz_parameter_sets"].insert(0, {
                "id": "set.tdk-datasheet-fit-n87-100c", "k": q(1.865, "W/m^3"), "alpha": 1.3286, "beta": 1.9368,
                "temperature": q(100, "degC"), "temperature_range": {"min": 368.15, "max": 378.15, "unit": "K"},
                "frequency_range": {"min": 25e3, "max": 500e3, "unit": "Hz"},
                "flux_density_range": {"min": 0.05, "max": 0.2, "unit": "T"},
                "excitation": "sinusoidal", "basis": "datasheet_fit", "source_ref": EV_TDK_N87_FIT["id"],
                "fit_metrics": {"max_rel_error": 0.047, "mean_abs_rel_error": 0.028, "n_points": 4},
                "note": "Four TDK datasheet typical points at 100 degC; validity window 95-105 degC (policy), 25-500 kHz, 50-200 mT.",
            })
            # Muhlethaler 2012 (publication), converted from k_i to k
            props["steinmetz_parameter_sets"].append({
                "id": "set.muhlethaler-2012-n87-25c", "k": q(81.12, "W/m^3"), "alpha": 1.09, "beta": 2.16,
                "temperature": q(25, "degC"), "frequency_range": {"min": 20e3, "max": 50e3, "unit": "Hz"},
                "flux_density_range": {"min": 0.05, "max": 0.1, "unit": "T"},
                "excitation": "sinusoidal", "basis": "publication", "source_ref": EV_MUHLETHALER["id"],
                "fit_metrics": {"n_points": 3},
                "note": ("Published iGSE k_i = 8.41 (W/m3, Hz, T) converted to classical k = k_i (2 pi)^(alpha-1) 2^(beta-alpha) "
                         "Int_0^2pi |cos|^alpha = 81.12 W/m3; fitted on an R42 toroid at 25 degC with symmetric triangular flux "
                         "(D = 0.5; source excitation triangular, stored as the sinusoidal-equivalent classical k), dB = 0.1-0.2 T "
                         "peak-to-peak. Validity ranges are the span of the three Table I fit points (20-50 kHz, 0.05-0.1 T peak); "
                         "the paper's 20-100 kHz triangular measurement range (Sec. IX) is wider than the fit span."),
            })
        out.append({
            "schema_version": "0.2.0", "id": f"material.{slug}-{grade.lower()}", "kind": "magnetic_material", "name": grade,
            "synthetic": False, "license": "NOASSERTION", "properties": props,
            "evidence_refs": [e["id"] for e in ev], "evidence": ev,
            "provenance": {"created_at": CREATED, "activity": "transcribed from cited sources and derived fits by tools/build_catalog.py",
                           "tool": {"id": "aipe.magnetics", "version": __version__}, "inputs": [e["id"] for e in ev],
                           "normalization": "all quantities converted to SI by aipe_magnetics.common.units with original units retained"},
            "quality": {"status": "normalized", "flags": [] if grade == "N87" or grade in FERROXCUBE_DATASHEETS else ["datasheet_parameters_missing"],
                        "notes": ["review_status unreviewed: awaiting owner review of the transcriptions"]},
            # 'power' only where a transcribed application statement says so (N87 datasheet 'Preferred application: Power
            # transformers'; MagNet constants.py material_applications for the other grades)
            "tags": ["ferrite", "mnzn"] + (["power"] if "power" in MAGNET_APPLICATIONS[grade].lower() else []),
        })
    return out


# -------------------------------------------------------------------------------------------------
# cores (TDK, October 2024 datasheets)
# -------------------------------------------------------------------------------------------------
def core_records() -> list[dict[str, Any]]:
    cores = []

    def rec(id: str, name: str, family: str, slug: str, part: str, *, c1_per_mm: float, le_mm: float, ae_mm2: float, amin_mm2: float,
            ve_mm3: float, mass_g: float, an_mm2: float | None, ln_mm: float | None, ar_uohm: float | None, dims_mm: dict[str, float],
            gaps_mm: list[float], set_desc: str, extra: dict[str, Any], coil_former: str | None, coil_former_locator: str | None = None,
            title: str | None = None, derived: dict[str, Any] | None = None) -> dict[str, Any]:
        ev = tdk_core_evidence(name, slug, title=title, coil_former_locator=coil_former_locator)
        props: dict[str, Any] = {
            "family_ref": family, "manufacturer_ref": "manufacturer.tdk", "part_number": part,
            "material_refs": ["material.tdk-n27", "material.tdk-n87"], "set_description": set_desc,
            "effective_area": q(ae_mm2, "mm^2"), "effective_length": q(le_mm, "mm"), "effective_volume": q(ve_mm3, "mm^3"),
            "minimum_area": q(amin_mm2, "mm^2"), "mass": q(mass_g, "g"),
            "dimensions": {"values": {k: q(v, "mm") for k, v in dims_mm.items()}},
            "air_gap_options": [q(g, "mm") for g in gaps_mm],
            "description": f"TDK {name} ferrite core set; magnetic characteristics per set from the October 2024 datasheet. "
                           f"Core constant sum(l/A) = {c1_per_mm} 1/mm (datasheet).",
        }
        if an_mm2 is not None:
            props["bobbin_window_area"] = q(an_mm2, "mm^2")
        if ln_mm is not None:
            props["mean_turn_length"] = q(ln_mm, "mm")
        return {
            "schema_version": "0.2.0", "id": id, "kind": "core_geometry", "name": name, "synthetic": False, "license": "NOASSERTION",
            "properties": props, "evidence_refs": [ev["id"]], "evidence": [ev],
            "provenance": {"created_at": CREATED, "activity": "transcribed from the TDK datasheet by tools/build_catalog.py",
                           "tool": {"id": "aipe.magnetics", "version": __version__}, "inputs": [ev["id"]],
                           "normalization": "mm -> m, mm^2 -> m^2, mm^3 -> m^3, g -> kg with original units retained"},
            "quality": {"status": "normalized", "notes": ["review_status unreviewed: awaiting owner review of the transcriptions"]},
            "extensions": {"aipe.datasheet": {"core_constant_sum_l_over_A_per_mm": c1_per_mm, "coil_former": coil_former,
                                              "coil_former_AR_uOhm": ar_uohm, **({"derived": derived} if derived else {}), **extra}},
            "tags": ["tdk", "ferrite-core"],
        }

    cores.append(rec("core.tdk-e65-32-27", "E 65/32/27", "core-family.tdk-e", "e_65_32_27", "B66387", c1_per_mm=0.27, le_mm=147, ae_mm2=535,
                     amin_mm2=529, ve_mm3=78650, mass_g=394, an_mm2=426, ln_mm=188, ar_uohm=15.2,
                     dims_mm={"length_a": 65.0, "height_b_half": 32.8, "depth_c": 27.4, "centre_leg_width": 20.0,
                              "window_width_total": 44.2, "window_height_half": 22.2},
                     gaps_mm=[0.5, 1.0, 1.5, 2.5, 4.0], set_desc="core set of 2 E halves (per set values)",
                     extra={"ungapped_N87": {"AL_nH": 7900, "mu_e": 1700, "BS_mT_at_250A_per_m_100C": 320,
                                             "PV_W_per_set_max": 6.7, "PV_condition": "100 mT, 100 kHz, 100 degC"}},
                     coil_former="B66388B1000T001 (one core set): AN 426 mm2, lN 188 mm, AR 15.2 uOhm",
                     coil_former_locator="coil former table p.4, row B66388B1000T001 (requires 1 core set)"))
    cores.append(rec("core.tdk-e70-33-32", "E 70/33/32", "core-family.tdk-e", "e_70_33_32", "B66371", c1_per_mm=0.22, le_mm=149, ae_mm2=683,
                     amin_mm2=676, ve_mm3=102000, mass_g=514, an_mm2=456, ln_mm=209, ar_uohm=15.8,
                     dims_mm={"length_a": 70.5, "height_b_half": 33.2, "depth_c": 32.0, "centre_leg_width": 22.0,
                              "window_width_total": 48.0, "window_height_half": 21.9},
                     gaps_mm=[2.0, 2.5, 4.0], set_desc="core set of 2 E halves (per set values)",
                     extra={"ungapped_N87": {"AL_nH": 9700, "mu_e": 1700, "BS_mT_at_250A_per_m_100C": 320,
                                             "PV_W_per_set_max": 9.5, "PV_condition": "100 mT, 100 kHz, 100 degC"}},
                     coil_former="B66372B1000T001 (one core set): AN 456 mm2, lN 209 mm, AR 15.8 uOhm",
                     coil_former_locator="coil former table p.4, row B66372B1000T001 (requires 1 core set; the 230.5 mm / 20 uOhm rows "
                                         "are the two-core-set formers B66372A2000T001 and B66372B2000T001)"))
    cores.append(rec("core.tdk-etd59-31-22", "ETD 59/31/22", "core-family.tdk-etd", "etd_59_31_22", "B66397", c1_per_mm=0.38, le_mm=139,
                     ae_mm2=368, amin_mm2=368, ve_mm3=51200, mass_g=260, an_mm2=365.6, ln_mm=106.1, ar_uohm=10.0,
                     dims_mm={"length_a": 59.8, "height_b_half": 31.2, "depth_c": 22.1, "centre_leg_diameter": 22.1,
                              "window_width_total": 43.6, "window_height_half": 22.0},
                     gaps_mm=[0.2, 1.0, 1.5, 2.0], set_desc="core set of 2 ETD halves (per set values), IEC 63093-6",
                     extra={"ungapped_N87": {"AL_nH": 5300, "mu_e": 1590, "BS_mT_at_250A_per_m_100C": 320,
                                             "PV_W_per_set_max": 5.2, "PV_condition": "100 mT, 100 kHz, 100 degC"}},
                     coil_former="B66398W1024T001: AN 365.6 mm2, lN 106.1 mm, AR 10.0 uOhm",
                     coil_former_locator="coil former table, row B66398W1024T001"))
    cores.append(rec("core.tdk-pm87-70", "PM 87/70", "core-family.tdk-pm", "pm_87_70", "B65713", c1_per_mm=0.16, le_mm=146, ae_mm2=910,
                     amin_mm2=700, ve_mm3=133000, mass_g=770, an_mm2=657, ln_mm=158, ar_uohm=8.27,
                     dims_mm={"diameter": 87.0, "height_total": 70.0, "centre_post_diameter": 31.7, "centre_hole_diameter": 8.5,
                              "window_height": 48.0},
                     gaps_mm=[], set_desc="core set of 2 PM halves (per set values), IEC 63093-10",
                     extra={"ungapped_N87": {"AL_nH": 14000, "mu_e": 1790, "PV_W_per_set_max": 8.0, "PV_condition": "100 mT, 100 kHz, 100 degC"}},
                     coil_former="B65714K1020T001: AN 657 mm2, lN 158 mm, AR 8.27 uOhm",
                     coil_former_locator="coil former table, row B65714K1020T001"))
    cores.append(rec("core.tdk-uu93-152-30", "UU 93/152/30", "core-family.tdk-u", "u_93_76_30", "B67345", c1_per_mm=0.42, le_mm=354, ae_mm2=840,
                     amin_mm2=840, ve_mm3=297000, mass_g=1500, an_mm2=None, ln_mm=None, ar_uohm=None,
                     dims_mm={"length_a": 93.0, "height_b": 76.0, "depth_c": 30.0, "leg_width": 28.0, "window_width": 34.6,
                              "window_height_single_u": 48.0},
                     gaps_mm=[], set_desc="two U 93/76/30 cores (UU combination, per set values); no coil former listed",
                     extra={"ungapped_N87": {"AL_nH": 5700, "mu_e": 1900, "PV_W_per_set_max": 5.5, "PV_condition": "100 mT, 25 kHz, 100 degC"}},
                     coil_former=None, title="U 93/76/30 with I 93/28/30",
                     derived={"window_height_uu_mm": 96.0,
                              "note": "UU 93/152/30 window height = 2 x 48 mm (single-U window height from the outline drawing, p.2); "
                                      "derived by hand, not printed in the datasheet. window_width is the drawing's '34.6 min.'"}))
    return cores


def family_records() -> list[dict[str, Any]]:
    # a dimensional standard is recorded only where the core datasheet itself prints it ('To IEC 63093-6', 'To IEC 63093-10');
    # the E and U datasheets mention IEC 63093 only for visual inspection, so no standard is asserted for them
    out = []
    for fid, name, shape, std, where in (("core-family.tdk-e", "TDK E cores", "E", None, None),
                                         ("core-family.tdk-etd", "TDK ETD cores", "ETD", "IEC 63093-6", "ETD 59/31/22 datasheet header 'To IEC 63093-6'"),
                                         ("core-family.tdk-pm", "TDK PM cores", "PM", "IEC 63093-10", "PM 87/70 datasheet header 'To IEC 63093-10'"),
                                         ("core-family.tdk-u", "TDK U cores", "U", None, None)):
        description = "Core family identity as used in the TDK datasheets." + (f" Dimensional standard: {where}." if where else
                                                                               " No dimensional standard is printed on the datasheet.")
        ev = evidence(f"evidence.{fid.split('.')[-1]}-family", "source", "TDK Electronics ferrite core datasheets (October 2024)",
                      description, url="https://www.tdk-electronics.tdk.com/")
        properties: dict[str, Any] = {"manufacturer_ref": "manufacturer.tdk", "shape": shape}
        if std:
            properties["standard"] = std
        out.append({"schema_version": "0.2.0", "id": fid, "kind": "core_family", "name": name, "synthetic": False, "license": "NOASSERTION",
                    "properties": properties, "evidence_refs": [ev["id"]], "evidence": [ev]})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--catalog-dir", type=Path, default=CATALOG_DIR)
    args = parser.parse_args(argv)
    fits = load(FITS_FILE)
    groups = {"manufacturers": manufacturers(), "families": family_records(), "materials": material_records(fits), "cores": core_records()}
    records = [MagneticRecord.model_validate(r) for group in groups.values() for r in group]
    # include the synthetic examples already present so cross references are checked against the complete catalogue
    existing = [MagneticRecord.model_validate(load(p)) for p in args.catalog_dir.rglob("*.json")
                if not any(p.parent.name == g for g in groups)]
    errors = validate_catalogue(records + existing)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    written = 0
    for group, items in groups.items():
        for item in items:
            path = args.catalog_dir / group / (item["id"] + ".json")
            dump(MagneticRecord.model_validate(item).to_json_dict(), path, overwrite=args.overwrite)
            written += 1
    print(f"wrote {written} records into {args.catalog_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
