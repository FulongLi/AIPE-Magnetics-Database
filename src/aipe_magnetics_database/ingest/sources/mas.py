"""OpenMagnetics / PSMA MAS importer (Apache-2.0), pinned at commit ``c33ccfec``.

Inputs (verified by SHA-256 from the upstream ``data/MANIFEST.sha256``): ``core_materials.ndjson`` (1137 rows),
``core_shapes.ndjson`` (2294), ``cores.ndjson`` (18,943 manufacturer parts) and the Git-LFS object
``advanced_core_materials.ndjson`` (380 name-keyed patches: measured / datasheet loss points, B-H cycles,
amplitude permeability, mass losses).

Outputs under ``data/normalized/mas/``:

* ``materials/`` - 1137 ``magnetic_material`` records. Identifiers and every property of the first import
  (v0.2 catalogue batch) are preserved; added typed fields: ``density``, ``resistivity`` (single-temperature
  values only), ``curie_temperature`` (with a note that datasheets often state a bound).
* ``loss_models/`` - ``model_metadata`` records holding the upstream *fitted* loss-model coefficients exactly as
  published (Steinmetz with MAS temperature polynomial, Magnetics, Micrometals, TDG, POCO, Roshen). They are
  ``fitted_parameters``, never measurements, and are not converted into AIPE Steinmetz sets (MAS sets have no
  single fit temperature, so a conversion would invent one).
* ``measurements/`` (+ ``tables/`` sidecars) - ``measurement`` records with ``aipe.magnetics.table/0.1`` tables:
  core loss points split by upstream ``origin`` (manufacturer datasheet / MagNet / measurement), mass loss,
  complex permeability, complex permittivity, initial-permeability curves, amplitude permeability, loss factor and
  B-H cycles.
* ``shapes/`` - 2294 standard shapes (tolerance ranges only, unchanged), ``cores/`` - 18,943 part records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aipe_magnetics_database.ingest.acquire import RawFile, obtain
from aipe_magnetics_database.ingest.common import (
    ImportSummary,
    SourceWriter,
    column,
    dataset_record,
    finish_source,
    evidence,
    finite,
    importer_attribution,
    kelvin,
    measurement_record,
    q,
    read_jsonl,
    slug,
    validated,
)

SOURCE_ID = "openmagnetics_mas"
DIRECTORY = "mas"
REPOSITORY = "https://github.com/Power-Supply-Manufacturers-Association/MAS"
COMMIT = "c33ccfec8e00e156d632c9a52b02bc1cfdf714dc"
COMMIT_TIME = "2026-11-02T20:52:36Z"
LICENSE = "Apache-2.0"
BLOB = {"data/core_materials.ndjson": "fc49e78c22bff9350f2f3649dcc9017662bfd1c8",
        "data/core_shapes.ndjson": "3c75852eac6f38245e4c4dfc035bfec7bb10c9a3"}
RAW_FILES = [
    RawFile("data/core_materials.ndjson", f"https://raw.githubusercontent.com/Power-Supply-Manufacturers-Association/MAS/{COMMIT}/data/core_materials.ndjson",
            "10a6b3f3e72f5d963d05a4f9d11ed3595d25c14634a739e8ade263f3edff44b2", 11010297, LICENSE),
    RawFile("data/core_shapes.ndjson", f"https://raw.githubusercontent.com/Power-Supply-Manufacturers-Association/MAS/{COMMIT}/data/core_shapes.ndjson",
            "84eeed999eb06c69c09c0fb38e8f693dac79c284e32cf372b5a6c08a80bc50f2", 639486, LICENSE),
    RawFile("data/cores.ndjson", f"https://raw.githubusercontent.com/Power-Supply-Manufacturers-Association/MAS/{COMMIT}/data/cores.ndjson",
            "6fb6142385c5541477bbf99bab4d2c6ade4a87bbb9470002bdcdad72ff1cd81e", 8196923, LICENSE),
    # Git LFS object (the repository stores a pointer); served by GitHub's LFS media endpoint at the pinned commit
    RawFile("data/advanced_core_materials.ndjson",
            f"https://media.githubusercontent.com/media/Power-Supply-Manufacturers-Association/MAS/{COMMIT}/data/advanced_core_materials.ndjson",
            "0157ea8bbb361acc06093a3955928a436b136d7e10d948dbc27259c48f20d620", 145044312, LICENSE),
]
PUBLISHER = "OpenMagnetics / Power Supply Manufacturers Association"
CLASS_MAP = {("ferrite", "MnZn"): "ferrite_mnzn", ("ferrite", "NiZn"): "ferrite_nizn"}
FAMILY_CLASS = {"powder": "powder_alloy", "nanocrystalline": "nanocrystalline", "amorphous": "amorphous", "electricalSteel": "silicon_steel"}
MAPPED_MATERIAL_FIELDS = {"name", "material", "materialComposition", "manufacturerInfo", "permeability", "saturation", "remanence",
                          "coerciveForce", "recommendations"}
LOSS_METHODS = ("steinmetz", "magnetics", "micrometals", "tdg", "poco", "roshen")
ORIGIN_CLASS = {"manufacturer": "manufacturer_datasheet", "datasheet": "manufacturer_datasheet", "MagNet": "experimental_measurement",
                "measurement": "experimental_measurement"}


def _blob_url(path: str, line: int | None = None) -> str:
    return f"{REPOSITORY}/blob/{COMMIT}/{path}" + (f"#L{line}" if line else "")


def _mas_evidence(path: str, line: int, description: str) -> dict[str, Any]:
    return evidence("evidence.mas-source", "source", "Power-Supply-Manufacturers-Association/MAS", description, COMMIT_TIME, LICENSE,
                    locator=f"{path}:L{line}", url=_blob_url(path, line), version=COMMIT)


def _provenance(activity: str, normalization: str, inputs: list[str] | None = None) -> dict[str, Any]:
    out = {"activity": activity, "source_commit": COMMIT, "tool": importer_attribution("mas"), "normalization": normalization}
    if inputs:
        out["inputs"] = inputs
    return out


def _cv(value: float, unit: str, *, temperature: Any = None, field: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {"value": q(value, unit), "source_ref": "evidence.mas-source"}
    if temperature is not None:
        out["temperature"] = kelvin(temperature)
    if field is not None:
        out["field"] = q(field, "A/m")
    return out


# -------------------------------------------------------------------------------------------------
# materials
# -------------------------------------------------------------------------------------------------
def material_id(row: dict[str, Any], line: int) -> str:
    return f"material.mas-{slug(row['manufacturerInfo']['name'] + ' ' + row['name'], transliterate=False)}-{line:04d}"


def material_class(row: dict[str, Any]) -> str:
    family = row.get("material")
    if family == "ferrite":
        return CLASS_MAP.get((family, row.get("materialComposition")), "other")
    return FAMILY_CLASS.get(family, "other")


def material_record(row: dict[str, Any], line: int, summary: ImportSummary) -> dict[str, Any]:
    manufacturer = row["manufacturerInfo"]["name"]
    name = f"{manufacturer} {row['name']}"
    props: dict[str, Any] = {"grade": row["name"], "material_class": material_class(row),
                             "description": f"MAS source: {name}; classification is based on upstream material and composition fields."}
    perm = row.get("permeability") or {}
    initial = perm.get("initial")
    if isinstance(initial, dict) and finite(initial.get("value")) and initial["value"] > 0:
        props["initial_permeability"] = {"value": q(initial["value"], "1"), "source_ref": "evidence.mas-source"}
        if finite(initial.get("temperature")):
            props["initial_permeability"]["temperature"] = kelvin(initial["temperature"])
        if finite(initial.get("frequency")):
            props["initial_permeability"]["frequency"] = q(initial["frequency"], "Hz")
        if finite(initial.get("magneticFluxDensityPeak")):
            props["initial_permeability"]["flux_density_peak"] = q(initial["magneticFluxDensityPeak"], "T")
    elif isinstance(initial, list):
        # first-import rule: every point that states a temperature (curves vs f or H_dc keep their full form in the table record)
        # (non-positive values - upstream marks mu = 0 above the Curie point - are not a permeability and are left out)
        points = [p for p in initial if finite(p.get("value"), p.get("temperature")) and p["value"] > 0]
        if len(points) < sum(1 for p in initial if finite(p.get("value"), p.get("temperature"))):
            summary.flag("permeability_nonpositive_points_not_typed")
        if points:
            props["permeability_vs_temperature"] = [{"temperature": kelvin(p["temperature"]), "value": q(p["value"], "1"),
                                                     "source_ref": "evidence.mas-source"} for p in points]
    props["saturation_flux_density"] = [_cv(p["magneticFluxDensity"], "T", temperature=p.get("temperature"), field=p.get("magneticField"))
                                        for p in row.get("saturation") or [] if finite(p.get("magneticFluxDensity"))]
    if row.get("remanence"):
        props["remanence"] = [_cv(p["magneticFluxDensity"], "T", temperature=p.get("temperature"), field=p.get("magneticField"))
                              for p in row["remanence"] if finite(p.get("magneticFluxDensity"))]
    if row.get("coerciveForce"):
        props["coercivity"] = [_cv(p["magneticField"], "A/m", temperature=p.get("temperature")) for p in row["coerciveForce"]
                               if finite(p.get("magneticField"))]
    rec = row.get("recommendations") or {}
    if finite(rec.get("minimumFrequency")) or finite(rec.get("maximumFrequency")):
        rng: dict[str, Any] = {"unit": "Hz"}
        if finite(rec.get("minimumFrequency")):
            rng["min"] = rec["minimumFrequency"]
        if finite(rec.get("maximumFrequency")):
            rng["max"] = rec["maximumFrequency"]
        props["recommended_frequency_range"] = rng
    # --- typed scalar fields added in 0.3 ---------------------------------------------------------
    flags = ["third_party_unreviewed"]
    added = []
    if finite(row.get("density")):
        props["density"] = {"value": q(row["density"], "kg/m^3"), "source_ref": "evidence.mas-source"}
        added.append("density")
    resistivity = [p for p in row.get("resistivity") or [] if finite(p.get("value"))]
    if len(resistivity) == 1:
        p = resistivity[0]
        props["resistivity"] = {"value": q(p["value"], "Ohm*m"), "source_ref": "evidence.mas-source"}
        if finite(p.get("temperature")):
            props["resistivity"]["temperature"] = kelvin(p["temperature"])
        added.append("resistivity")
    elif len(resistivity) > 1:
        summary.flag("resistivity_multiple_points_not_typed")
    if finite(row.get("curieTemperature")):
        props["curie_temperature"] = {"value": kelvin(row["curieTemperature"]), "source_ref": "evidence.mas-source",
                                      "note": "MAS value; manufacturer datasheets often state the Curie temperature as a bound "
                                              "(e.g. '>= 210 degC') - treat it as the stated threshold, not an exact measurement."}
        flags.append("curie_temperature_may_be_bound")
        added.append("curieTemperature")
    unmapped = sorted(set(row) - MAPPED_MATERIAL_FIELDS - {"density", "curieTemperature"} - ({"resistivity"} if "resistivity" in added else set()))
    ext = {"source_file": "data/core_materials.ndjson", "source_line": line, "source_blob_sha1": BLOB["data/core_materials.ndjson"],
           "manufacturer": manufacturer, "material_composition": row.get("materialComposition"),
           "source_datasheet": row["manufacturerInfo"].get("datasheetUrl"), "raw_available_from_upstream": True,
           "unmapped_fields": unmapped}
    if row.get("provenance"):
        ext["upstream_provenance"] = row["provenance"]
    record = {"schema_version": "0.2.0", "id": material_id(row, line), "kind": "magnetic_material", "name": name, "synthetic": False,
              "license": LICENSE, "properties": props, "evidence_refs": ["evidence.mas-source"],
              "evidence": [_mas_evidence("data/core_materials.ndjson", line, "Third-party material catalogue, not independently tested by AIPE.")],
              "dataset_ref": "dataset.mas-core-materials",
              "quality": {"status": "normalized", "flags": sorted(flags),
                          "notes": ["Unmapped upstream fields and non-scalar curves remain in the pinned original file or in the "
                                    "linked measurement / loss-model records; uncertain thresholds have not been converted into exact values."]},
              "provenance": _provenance("normalize_mas_core_materials",
                                        "Material grades/classes, selected permeability and magnetic values, density, single-point "
                                        "resistivity and Curie temperature; degC converted to K, other SI units preserved."),
              "extensions": {"aipe.mas": ext, "aipe.data": {"data_class": "manufacturer_datasheet", "source_id": SOURCE_ID}}}
    return validated(record)


# -------------------------------------------------------------------------------------------------
# fitted loss models
# -------------------------------------------------------------------------------------------------
LOSS_EQUATIONS = {
    "steinmetz": "P_v [W/m^3] = k * f^alpha * B_peak^beta * (ct0 - ct1*T + ct2*T^2), f in Hz, B_peak in T, T in degC (MAS docs/units.md); "
                 "ranges apply between minimumFrequency and maximumFrequency",
    "magnetics": "Magnetics Inc. core-loss curve fit (a, b, c) as published by the manufacturer; see MAS documentation for the equation form",
    "micrometals": "Micrometals core-loss curve fit (a, b, c, d) as published by the manufacturer",
    "tdg": "TDG core-loss curve fit (a, b, c, d) as published by the manufacturer",
    "poco": "POCO core-loss curve fit (a, b, c) as published by the manufacturer",
    "roshen": "Roshen core-loss model coefficients as stored in MAS",
}


def loss_model_records(row: dict[str, Any], line: int, mat_id: str, summary: ImportSummary) -> list[dict[str, Any]]:
    out = []
    for shape_key, items in sorted((row.get("volumetricLosses") or {}).items()):
        for index, item in enumerate(items):
            if not isinstance(item, dict) or item.get("method") not in LOSS_METHODS:
                continue
            method = item["method"]
            coefficients = {k: v for k, v in item.items() if k != "method"}
            if not coefficients:
                summary.exclude("loss_model", "method named without coefficients", material=mat_id, method=method)
                continue
            if method == "roshen" and all(v == 1 for v in (coefficients.get("coefficients") or {}).values()):
                summary.exclude("loss_model", "placeholder coefficients (all 1)", material=mat_id, method=method)
                continue
            rid = f"loss-model.{mat_id.split('.', 1)[1]}-{slug(shape_key)}-{index}-{method}"
            validity: dict[str, Any] = {}
            if method == "steinmetz":
                lows = [r.get("minimumFrequency") for r in item.get("ranges", []) if finite(r.get("minimumFrequency"))]
                highs = [r.get("maximumFrequency") for r in item.get("ranges", []) if finite(r.get("maximumFrequency"))]
                if lows and highs:
                    validity = {"frequency_range": {"min": min(lows), "max": max(highs), "unit": "Hz"}}
            record = {"schema_version": "0.2.0", "id": rid, "kind": "model_metadata", "name": f"{row['manufacturerInfo']['name']} {row['name']} {method} loss fit",
                      "synthetic": False, "license": LICENSE,
                      "properties": {"model_id": f"mas.loss.{method}", "family": "empirical", "architecture": method, "material_refs": [mat_id],
                                     "inference_config": {"equation": LOSS_EQUATIONS[method], "shape_key": shape_key, "coefficients": coefficients},
                                     "validity": validity,
                                     "description": "Fitted loss-model coefficients as published in MAS (manufacturer or OpenMagnetics fit); "
                                                    "fitted parameters, not measurements; AIPE has not re-fitted or verified them."},
                      "evidence_refs": ["evidence.mas-source"],
                      "evidence": [_mas_evidence("data/core_materials.ndjson", line, "Third-party fitted loss-model coefficients, not independently tested by AIPE.")],
                      "dataset_ref": "dataset.mas-core-materials",
                      "quality": {"status": "normalized", "flags": ["third_party_unreviewed"]},
                      "provenance": _provenance("normalize_mas_loss_models", "Coefficients copied verbatim; equation recorded from MAS documentation.", [mat_id]),
                      "extensions": {"aipe.data": {"data_class": "fitted_parameters", "source_id": SOURCE_ID},
                                     "aipe.mas": {"source_file": "data/core_materials.ndjson", "source_line": line,
                                                  "json_path": f"volumetricLosses.{shape_key}[{index}]"}}}
            out.append(validated(record))
            summary.count("loss_model")
    return out


# -------------------------------------------------------------------------------------------------
# tables
# -------------------------------------------------------------------------------------------------
def _num(v: Any) -> float | None:
    return v if finite(v) else None


def _k(celsius: Any) -> float | None:
    return celsius + 273.15 if finite(celsius) else None


def _table_records(row: dict[str, Any], adv: dict[str, Any] | None, line: int, mat_id: str, writer: SourceWriter,
                   summary: ImportSummary) -> list[dict[str, Any]]:
    token = mat_id.split(".", 1)[1]
    mname = f"{row['manufacturerInfo']['name']} {row['name']}"
    out: list[dict[str, Any]] = []

    def emit(kind: str, quantity: str, columns: list[dict[str, Any]], rows: list[list[Any]], data_class: str, *, path: str, file: str,
             flags: tuple[str, ...] = (), underlying: str | None = None, notes: tuple[str, ...] = (), method: str | None = None,
             commit_sidecar: bool = True) -> None:
        if not rows:
            return
        src_line = line if file == "data/core_materials.ndjson" else adv_line
        description = {"experimental_measurement": "Measured data re-published in MAS",
                       "manufacturer_datasheet": "Manufacturer datasheet curve (digitised or extracted by OpenMagnetics)"}[data_class]
        ev_kind_desc = ("Measurement as re-published by MAS (origin: " + (underlying or "unspecified") + ")"
                        if data_class == "experimental_measurement" else "Manufacturer-reported typical curve via MAS (datasheet digitisation)")
        items = [_mas_evidence(file, src_line, "Third-party data, not independently tested by AIPE."),
                 evidence("evidence.mas-measurement", "measurement", mname, ev_kind_desc, COMMIT_TIME, LICENSE, locator=f"{file}:{path}",
                          url=_blob_url(file, src_line if file == "data/core_materials.ndjson" else None), version=COMMIT)]
        rid = f"measurement.{token}-{kind}"
        rec = measurement_record(rid=rid, name=f"{mname} {quantity.replace('_', ' ')}" + (f" ({kind.rsplit('-', 1)[-1]})" if kind.count("-") else ""),
                                 material_ref=mat_id, license=LICENSE, evidence_items=items, data_class=data_class, source_id=SOURCE_ID,
                                 quantity=quantity, columns=columns, rows=rows, writer=writer, method=method,
                                 description=f"{description}; MAS path {path}.",
                                 provenance=_provenance("normalize_mas_tables", "Upstream SI values copied; degC converted to K; "
                                                        "missing values are null; no interpolation or resampling.", [mat_id]),
                                 flags=flags, dataset_ref="dataset.mas-core-materials" if file.endswith("core_materials.ndjson") else "dataset.mas-advanced-core-materials",
                                 underlying_source=underlying, table_notes=notes,
                                 extra={"mas_path": path, "mas_file": file}, commit_sidecar=commit_sidecar)
        out.append(rec)
        summary.count("measurement")
        summary.rows(quantity, len(rows))

    adv_line = adv.get("_line") if adv else None
    perm = row.get("permeability") or {}
    # complex permeability (real and imaginary lists merged on identical frequency; missing component = null)
    cplx = perm.get("complex") or {}
    merged: dict[float, list[Any]] = {}
    for comp, idx in (("real", 1), ("imaginary", 2)):
        for p in cplx.get(comp) or []:
            if finite(p.get("frequency"), p.get("value")):
                merged.setdefault(p["frequency"], [p["frequency"], None, None])[idx] = p["value"]
    rows = [merged[f] for f in sorted(merged)]
    neg = sum(1 for r in rows if (r[1] is not None and r[1] < 0) or (r[2] is not None and r[2] < 0))
    emit("complex-permeability", "complex_permeability",
         [column("frequency", "excitation frequency"), column("mu_real", "mu' of mu = mu' - j mu'' (relative)"),
          column("mu_imag", "mu'' of mu = mu' - j mu'' (relative, positive = loss)")], rows, "manufacturer_datasheet",
         path="permeability.complex", file="data/core_materials.ndjson", flags=("negative_values_present",) if neg else (),
         notes=("small-signal values; null = component not reported at this frequency",))
    if neg:
        summary.flag("complex_permeability_negative_rows", neg)
    # initial permeability curves (list form): vs temperature, frequency, DC bias
    initial = perm.get("initial")
    if isinstance(initial, list):
        rows = [[_k(p.get("temperature")), _num(p.get("frequency")), _num(p.get("magneticFieldDcBias")),
                 _num(p.get("magneticFluxDensityPeak")), p["value"]] for p in initial if finite(p.get("value"))]
        emit("initial-permeability", "initial_permeability",
             [column("temperature", "core temperature", "degC"), column("frequency", "measurement frequency (null = not stated)"),
              column("dc_bias_field", "DC bias field H_dc (null = not stated)"),
              column("flux_density_peak", "AC peak flux density (null = not stated)"),
              column("relative_permeability", "initial (small-signal) relative permeability")], rows, "manufacturer_datasheet",
             path="permeability.initial", file="data/core_materials.ndjson")
    # amplitude permeability (inline + advanced)
    for source, file, prefix in ((perm.get("amplitude"), "data/core_materials.ndjson", "amplitude-permeability"),
                                 (((adv or {}).get("permeability") or {}).get("amplitude"), "data/advanced_core_materials.ndjson",
                                  "amplitude-permeability-advanced")):
        if source:
            rows = [[_k(p.get("temperature")), _num(p.get("frequency")), _num(p.get("magneticFluxDensityPeak")),
                     _num(p.get("magneticFieldPeak")), _num(p.get("magneticFieldDcBias")), p["value"]] for p in source if finite(p.get("value"))]
            emit(prefix, "amplitude_permeability",
                 [column("temperature", "core temperature", "degC"), column("frequency", "excitation frequency (null = not stated)"),
                  column("flux_density_peak", "AC peak flux density B_peak (null = not stated)"),
                  column("field", "AC peak field H_peak (null = not stated)"), column("dc_bias_field", "DC bias field (null = not stated)"),
                  column("amplitude_permeability", "amplitude permeability mu_a = B_peak / (mu0 H_peak)")], rows, "manufacturer_datasheet",
                 path="permeability.amplitude", file=file)
    # complex permittivity
    cp = (row.get("permittivity") or {}).get("complex") or {}
    merged = {}
    for comp, idx in (("real", 1), ("imaginary", 2)):
        for p in cp.get(comp) or []:
            if finite(p.get("frequency"), p.get("value")):
                merged.setdefault(p["frequency"], [p["frequency"], None, None])[idx] = p["value"]
    emit("complex-permittivity", "complex_permittivity",
         [column("frequency", "excitation frequency"), column("eps_real", "eps' (relative)"), column("eps_imag", "eps'' (relative)")],
         [merged[f] for f in sorted(merged)], "manufacturer_datasheet", path="permittivity.complex", file="data/core_materials.ndjson")
    # loss factor tan(delta)/mu_i
    for shape_key, items in sorted((row.get("volumetricLosses") or {}).items()):
        for index, item in enumerate(items):
            if isinstance(item, dict) and item.get("method") == "lossFactor":
                rows = [[p["frequency"], _k(p.get("temperature")), p["value"]]
                        for p in item.get("factors") or [] if finite(p.get("frequency"), p.get("value"))]
                emit(f"loss-factor-{slug(shape_key)}-{index}", "loss_factor",
                     [column("frequency", "excitation frequency"), column("temperature", "core temperature (null = not stated)", "degC"),
                      column("loss_factor", "relative loss factor tan(delta)/mu_i")], rows, "manufacturer_datasheet",
                     path=f"volumetricLosses.{shape_key}[{index}]", file="data/core_materials.ndjson")
    # core-loss points: inline lists and advanced patches, split by origin
    def loss_points(container: dict[str, Any] | None) -> list[tuple[str, int, dict[str, Any]]]:
        pts = []
        for shape_key, items in sorted((container or {}).items()):
            for index, item in enumerate(items):
                if isinstance(item, list):
                    pts.extend((shape_key, index, p) for p in item)
        return pts

    for container, file, unit_name, quantity, prefix in (
            (row.get("volumetricLosses"), "data/core_materials.ndjson", "loss_density", "core_loss", "core-loss"),
            ((adv or {}).get("volumetricLosses"), "data/advanced_core_materials.ndjson", "loss_density", "core_loss", "core-loss-advanced"),
            ((adv or {}).get("massLosses"), "data/advanced_core_materials.ndjson", "mass_loss_density", "mass_core_loss", "mass-loss-advanced")):
        by_origin: dict[str, list[list[Any]]] = {}
        for shape_key, index, p in loss_points(container):
            proc = ((p.get("magneticFluxDensity") or {}).get("magneticFluxDensity") or {}).get("processed") or {}
            f = (p.get("magneticFluxDensity") or {}).get("frequency")
            if not finite(f, proc.get("peak"), p.get("value"), p.get("temperature")):
                summary.exclude(quantity + "_point", "non-finite or missing frequency / B_peak / temperature / value", material=mat_id)
                continue
            by_origin.setdefault(p.get("origin") or "unspecified", []).append(
                [f, proc["peak"], _num(proc.get("offset")), _num(proc.get("dutyCycle")), proc.get("label"), p["temperature"] + 273.15, p["value"]])
        for origin, rows in sorted(by_origin.items()):
            rows.sort(key=lambda r: (r[5], r[4] or "", r[0], r[1], r[3] if r[3] is not None else -1))
            data_class = ORIGIN_CLASS.get(origin, "manufacturer_datasheet")
            underlying = f"princeton-magnet:{slug(row['name'])}" if origin == "MagNet" else None
            flags: tuple[str, ...] = ()
            if origin == "measurement":
                flags = ("measurement_laboratory_unspecified",)
            if origin not in ORIGIN_CLASS:
                flags = ("origin_unspecified",)
            emit(f"{prefix}-{slug(origin)}", quantity,
                 [column("frequency", "fundamental excitation frequency"), column("flux_density_peak", "peak AC flux density (amplitude, (Bmax-Bmin)/2)"),
                  column("flux_density", "DC flux density offset (null = not stated)"),
                  column("duty", "duty cycle of the triangular flux waveform (rising fraction; null = sinusoidal / not stated)"),
                  {"name": "waveform", "unit": "", "definition": "upstream waveform label: sinusoidal | triangular"},
                  column("temperature", "core temperature", "degC"), column(unit_name, "time-averaged core loss per unit volume" if unit_name == "loss_density"
                                                                           else "time-averaged core loss per unit mass")],
                 rows, data_class, path=f"{'massLosses' if 'mass' in prefix else 'volumetricLosses'} points, origin={origin}", file=file,
                 flags=flags, underlying=underlying, method=f"upstream origin label '{origin}'")
    # B-H cycles (advanced)
    bh = (adv or {}).get("bhCycle") or []
    if bh:
        rows = [[p["temperature"] + 273.15, p["magneticField"], p["magneticFluxDensity"]] for p in bh
                if finite(p.get("temperature"), p.get("magneticField"), p.get("magneticFluxDensity"))]
        if len(rows) != len(bh):
            summary.exclude("bh_point", "non-finite value", len(bh) - len(rows), material=mat_id)
        flags = ()
        if any(abs(r[2]) > 3.0 for r in rows):
            flags = ("suspected_unit_error_flux_density",)
            summary.flag("bh_cycle_flux_above_3T_records")
        emit("bh-cycle", "bh_curve", [column("temperature", "core temperature", "degC"), column("field", "magnetic field strength H"),
                                       column("flux_density", "flux density B (same winding reference direction as H)")],
             rows, "manufacturer_datasheet", path="bhCycle", file="data/advanced_core_materials.ndjson", flags=flags,
             notes=("points in upstream order (traversal order of the digitised loop); |B| > 3 T indicates a unit error upstream",),
             commit_sidecar=False)  # 1.5 M digitised datasheet points: regenerated locally from the pinned LFS object
    return out


# -------------------------------------------------------------------------------------------------
# shapes and cores
# -------------------------------------------------------------------------------------------------
def shape_record(row: dict[str, Any], line: int, summary: ImportSummary) -> dict[str, Any]:
    ranges: dict[str, dict[str, float]] = {}
    inverted: dict[str, dict[str, float]] = {}
    nominal: dict[str, float] = {}
    for key, dim in (row.get("dimensions") or {}).items():
        lo, hi = dim.get("minimum"), dim.get("maximum")
        if finite(dim.get("nominal")):
            nominal[key] = dim["nominal"]
        entry = {}
        if finite(lo):
            entry["minimum_m"] = lo
        if finite(hi):
            entry["maximum_m"] = hi
        if "minimum_m" in entry and "maximum_m" in entry and lo > hi:
            inverted[key] = entry
            continue
        if entry:
            ranges[key] = entry
    flags = ["dimension_ranges_not_nominal", "third_party_unreviewed"]
    ext: dict[str, Any] = {"source_file": "data/core_shapes.ndjson", "source_line": line, "source_blob_sha1": BLOB["data/core_shapes.ndjson"],
                           "family": row.get("family"), "aliases": row.get("aliases") or [], "magnetic_circuit": row.get("magneticCircuit"),
                           "shape_type": row.get("type"), "dimensions_range_m": ranges}
    if nominal:
        ext["dimensions_nominal_m"] = nominal
    if inverted:
        ext["inconsistent_dimension_ranges_m"] = inverted
        flags.append("upstream_dimension_range_inverted")
        summary.flag("shape_dimension_range_inverted")
    unmapped = sorted(set(row) - {"name", "family", "aliases", "magneticCircuit", "type", "dimensions"})
    ext["unmapped_fields"] = unmapped
    record = {"schema_version": "0.2.0", "id": f"core.mas-shape-{line:05d}", "kind": "core_geometry", "name": row["name"], "synthetic": False,
              "license": LICENSE,
              "properties": {"part_number": row["name"],
                             "description": "MAS core shape: dimension tolerance ranges, not nominal sizes; see aipe.mas dimensions_range_m."},
              "evidence_refs": ["evidence.mas-source"],
              "evidence": [evidence("evidence.mas-source", "source", "Power-Supply-Manufacturers-Association/MAS",
                                    "Third-party core dimensions, not independently verified.", COMMIT_TIME, LICENSE,
                                    locator=f"data/core_shapes.ndjson:L{line}", url=_blob_url("data/core_shapes.ndjson", line), version=COMMIT)],
              "quality": {"status": "normalized", "flags": sorted(flags)}, "dataset_ref": "dataset.mas-core-shapes",
              "provenance": {"activity": "normalize_mas_core_shapes", "source_commit": COMMIT, "tool": importer_attribution("mas"),
                             "normalization": "Original tolerance bounds in metres represented as dimension ranges; no midpoints inferred; "
                                              "upstream nominal values kept separately; no effective parameters derived."},
              "extensions": {"aipe.mas": ext, "aipe.data": {"data_class": "manufacturer_datasheet", "source_id": SOURCE_ID}}}
    return validated(record)


def core_record(row: dict[str, Any], line: int, materials: dict[tuple[str, str], str], materials_by_name: dict[str, list[str]],
                shapes: dict[str, str], summary: ImportSummary) -> dict[str, Any]:
    info = row.get("manufacturerInfo") or {}
    fd = row.get("functionalDescription") or {}
    manufacturer = info.get("name") or ""
    mat_name = fd.get("material") if isinstance(fd.get("material"), str) else None
    mat_ref = materials.get((manufacturer.casefold(), (mat_name or "").casefold()))
    flags = ["no_effective_parameters", "third_party_unreviewed"]
    if mat_ref is None and mat_name:
        candidates = materials_by_name.get(mat_name.casefold(), [])
        if len(candidates) == 1:
            mat_ref = candidates[0]
            flags.append("material_resolved_by_name_only")
        else:
            flags.append("material_reference_unresolved")
            summary.flag("core_material_unresolved")
    shape_name = fd.get("shape") if isinstance(fd.get("shape"), str) else None
    shape_ref = shapes.get((shape_name or "").casefold())
    if shape_name and shape_ref is None:
        flags.append("shape_reference_unresolved")
        summary.flag("core_shape_unresolved")
    gaps = [g for g in fd.get("gapping") or [] if isinstance(g, dict)]
    props: dict[str, Any] = {"part_number": info.get("reference") or row["name"], "set_description": fd.get("type")}
    if mat_ref:
        props["material_refs"] = [mat_ref]
    record = {"schema_version": "0.2.0", "id": f"core.mas-part-{line:05d}", "kind": "core_geometry", "name": row["name"], "synthetic": False,
              "license": LICENSE, "properties": {k: v for k, v in props.items() if v},
              "evidence_refs": ["evidence.mas-source"],
              "evidence": [evidence("evidence.mas-source", "source", "Power-Supply-Manufacturers-Association/MAS",
                                    "Third-party core part catalogue, not independently verified.", COMMIT_TIME, LICENSE,
                                    locator=f"data/cores.ndjson:L{line}", url=_blob_url("data/cores.ndjson", line), version=COMMIT)],
              "quality": {"status": "normalized", "flags": sorted(flags)}, "dataset_ref": "dataset.mas-cores",
              "provenance": {"activity": "normalize_mas_cores", "source_commit": COMMIT, "tool": importer_attribution("mas"),
                             "normalization": "Part identity, manufacturer, material and shape references and gapping copied; "
                                              "gap lengths in m as published; distributor prices are not imported."},
              "extensions": {"aipe.mas": {"source_file": "data/cores.ndjson", "source_line": line, "manufacturer": manufacturer,
                                          "manufacturer_reference": info.get("reference"), "datasheet_url": info.get("datasheetUrl"),
                                          "status": info.get("status"), "material_name": mat_name, "shape_name": shape_name,
                                          "shape_ref": shape_ref, "gapping": gaps, "number_stacks": fd.get("numberStacks"),
                                          "coating": fd.get("coating")},
                             "aipe.data": {"data_class": "manufacturer_datasheet", "source_id": SOURCE_ID}}}
    if not props.get("material_refs"):
        record["properties"].pop("material_refs", None)
    return validated(record)


# -------------------------------------------------------------------------------------------------
# entry point
# -------------------------------------------------------------------------------------------------
def run(output_root: Path, *, local_dir: str | Path | None = None, allow_download: bool = False, cache: Any = None) -> ImportSummary:
    paths = obtain(SOURCE_ID, RAW_FILES, local_dir=local_dir, allow_download=allow_download, cache=cache, commit=COMMIT)
    summary = ImportSummary(SOURCE_ID, COMMIT, LICENSE, "permitted (Apache-2.0; upstream values originate from manufacturer datasheets)")
    summary.inputs = [{"path": f.name, "url": f.url, "sha256": f.sha256, "size": f.size} for f in RAW_FILES]
    writer = SourceWriter(output_root / DIRECTORY, summary)
    raw = read_jsonl(paths["data/core_materials.ndjson"])
    advanced = {}
    for n, row in enumerate(read_jsonl(paths["data/advanced_core_materials.ndjson"]), 1):
        row["_line"] = n
        advanced[(row["manufacturerInfo"]["name"].casefold(), row["name"])] = row
    materials, models, tables = [], [], []
    by_key: dict[tuple[str, str], str] = {}
    by_name: dict[str, list[str]] = {}
    used_adv = set()
    for line, row in enumerate(raw, 1):
        rec = material_record(row, line, summary)
        materials.append(rec)
        by_key[(row["manufacturerInfo"]["name"].casefold(), row["name"].casefold())] = rec["id"]
        by_name.setdefault(row["name"].casefold(), []).append(rec["id"])
        adv_key = (row["manufacturerInfo"]["name"].casefold(), row["name"])
        adv = advanced.get(adv_key)
        if adv:
            used_adv.add(adv_key)
        models.extend(loss_model_records(row, line, rec["id"], summary))
        tables.extend(_table_records(row, adv, line, rec["id"], writer, summary))
    for key in sorted(set(advanced) - used_adv):
        summary.exclude("advanced_patch", "no matching core material (manufacturer + name)", name=key[1], manufacturer=key[0])
    summary.count("magnetic_material", len(materials))
    shapes_raw = read_jsonl(paths["data/core_shapes.ndjson"])
    shapes = [shape_record(row, line, summary) for line, row in enumerate(shapes_raw, 1)]
    summary.count("core_shape", len(shapes))
    shape_lookup: dict[str, str] = {}
    for rec, row in zip(shapes, shapes_raw):
        for name in [row["name"], *(row.get("aliases") or [])]:
            shape_lookup.setdefault(name.casefold(), rec["id"])
    cores = [core_record(row, line, by_key, by_name, shape_lookup, summary) for line, row in enumerate(read_jsonl(paths["data/cores.ndjson"]), 1)]
    summary.count("core_part", len(cores))
    # dataset records (kept identical in identity to the first import)
    datasets = []
    for rid, title, path, count, extra_desc in (
            ("dataset.mas-core-materials", "MAS core-materials catalogue", "data/core_materials.ndjson", len(raw), ""),
            ("dataset.mas-core-shapes", "MAS core-shapes catalogue", "data/core_shapes.ndjson", len(shapes), ""),
            ("dataset.mas-advanced-core-materials", "MAS advanced core-material curves (Git LFS)", "data/advanced_core_materials.ndjson",
             len(advanced), " Contains MagNet-origin loss points that duplicate Princeton MagNet measurements (see canonical duplicates)."),
            ("dataset.mas-cores", "MAS manufacturer core parts", "data/cores.ndjson", len(cores), "")):
        raw_file = next(f for f in RAW_FILES if f.name == path)
        datasets.append(dataset_record(
            rid=rid, name=title.replace("catalogue", "catalogue").replace("MAS core-", "MAS core "), source_id=SOURCE_ID, title=title, license=LICENSE,
            url=_blob_url(path), version=COMMIT, source_commit=COMMIT, publisher=PUBLISHER, redistribution=True, record_count=count,
            checksums={path: raw_file.sha256},
            description="Unreviewed third-party records; no independent AIPE physical verification. Values largely originate from "
                        "manufacturer datasheets, whose own terms are not covered by the MAS licence." + extra_desc,
            evidence_items=[evidence("evidence.mas-source", "source", "Power-Supply-Manufacturers-Association/MAS", "Apache-2.0 MAS catalogue source.",
                                     COMMIT_TIME, LICENSE, url=f"{REPOSITORY}/tree/{COMMIT}/data", version=COMMIT)],
            provenance=_provenance("MAS catalogue ingestion", "See docs/provenance-and-limitations.md."),
            extensions={"aipe.mas": {"source_file": path, "sha256": raw_file.sha256, "git_blob_sha1": BLOB.get(path)}}))
    shard_files = {
        "materials": writer.shards(materials, "materials", "materials", 100),
        "shapes": writer.shards(shapes, "shapes", "shapes", 500),
        "cores": writer.shards(cores, "cores", "cores", 1000),
        "loss_models": writer.shards(models, "loss_models", "loss-models", 500),
        "measurements": writer.shards(tables, "measurements", "measurements", 100),
    }
    names = {"dataset.mas-core-materials": "core-materials", "dataset.mas-core-shapes": "core-shapes",
             "dataset.mas-advanced-core-materials": "advanced-core-materials", "dataset.mas-cores": "cores"}
    dataset_files = []
    for rec in datasets:
        dataset_files.append(f"datasets/{names[rec['id']]}.json")
        writer.json(dataset_files[-1], rec)
    return finish_source(writer, summary, shard_files, dataset_files)


__all__ = ["COMMIT", "LICENSE", "RAW_FILES", "SOURCE_ID", "material_id", "run"]
