"""Princeton MagNet importer (MIT), packaged scalar databases at commit ``8a6892e9``.

Reads the ten ``src/magnet/data/<material>_database.h5`` files through the existing
:class:`~aipe_magnetics_database.data.sources.princeton_magnet.PrincetonMagnetAdapter` (exact column/unit mapping,
excitation classification, SHA-256 pin check, embedded metadata read as text - never unpickled) and writes, per
material, one ``measurement`` record whose complete table (every row, upstream order) is a committed JSONL sidecar.

Measurement semantics (upstream ``info_processing``): ``Flux_Density`` is the AC amplitude ``(max B - min B)/2`` from
the integral of the voltage; ``DC_Bias`` is ``round(mean H)`` (small negative values of -1/-2 A/m occur and are kept
verbatim); ``Frequency`` is a Welch estimate quantised to 10 Hz; ``Temperature`` is the oil-bath set point; loss
density is the mean v*i product divided by the effective volume of the toroid specimen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aipe_magnetics_database.data.sources import LocalCache, PrincetonMagnetAdapter
from aipe_magnetics_database.data.sources.princeton_magnet import CITATION, UPSTREAM_COMMIT, known_file_checksums
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
    slug,
    validated,
)

SOURCE_ID = "princeton_magnet"
DIRECTORY = "princeton-magnet"
COMMIT = UPSTREAM_COMMIT
COMMIT_TIME = "2026-05-12T00:00:00Z"
LICENSE = "MIT"
REPOSITORY = "https://github.com/PrincetonUniversity/magnet"
MANUFACTURERS = {"N27": "TDK", "N30": "TDK", "N49": "TDK", "N87": "TDK", "3C90": "Ferroxcube", "3C94": "Ferroxcube", "3E6": "Ferroxcube",
                 "3F4": "Ferroxcube", "77": "Fair-Rite", "78": "Fair-Rite"}


def raw_files() -> list[RawFile]:
    sums = known_file_checksums()
    return [RawFile(f"src/magnet/data/{m}_database.h5", f"https://raw.githubusercontent.com/PrincetonUniversity/magnet/{COMMIT}/src/magnet/data/{m}_database.h5",
                    sums[f"{m}_database.h5"], None, LICENSE) for m in sorted(MANUFACTURERS)]


def _provenance(activity: str, inputs: list[str] | None = None) -> dict[str, Any]:
    out = {"activity": activity, "source_commit": COMMIT, "tool": importer_attribution("princeton"),
           "normalization": "Packaged MagNet scalar table mapped column by column (docs/sources-adapters.md); degC -> K; Duty_P/Duty_N "
                            "sentinel -1 (sinusoidal) -> null; every upstream row kept in upstream order."}
    if inputs:
        out["inputs"] = inputs
    return out


def material_identity(material: str) -> dict[str, Any]:
    manufacturer = MANUFACTURERS[material]
    return validated({
        "schema_version": "0.2.0", "id": f"material.princeton-magnet-{slug(material)}", "kind": "magnetic_material",
        "name": f"{manufacturer} {material}", "synthetic": False, "license": LICENSE,
        "properties": {"grade": material, "description": f"Identity of the MagNet specimen material {manufacturer} {material}; "
                                                         "no datasheet values are asserted by this record."},
        "evidence_refs": ["evidence.princeton-magnet-source"],
        "evidence": [evidence("evidence.princeton-magnet-source", "source", "PrincetonUniversity/magnet", "Material named by the MagNet database.",
                              COMMIT_TIME, LICENSE, url=f"{REPOSITORY}/tree/{COMMIT}/src/magnet/data", version=COMMIT,
                              locator=f"src/magnet/data/{material}_database.h5")],
        "quality": {"status": "normalized", "flags": ["identity_only", "third_party_unreviewed"]},
        "dataset_ref": "dataset.princeton-magnet", "provenance": _provenance("princeton_magnet_material_identity"),
        "extensions": {"aipe.data": {"data_class": "identity", "source_id": SOURCE_ID}, "aipe.princeton": {"manufacturer": manufacturer}}})


def run(output_root: Path, *, local_dir: str | Path | None = None, allow_download: bool = False, cache: Any = None) -> ImportSummary:
    files = raw_files()
    paths = obtain(SOURCE_ID, files, local_dir=local_dir, allow_download=allow_download, cache=cache, commit=COMMIT)
    summary = ImportSummary(SOURCE_ID, COMMIT, LICENSE, "permitted (MIT; cite the three IEEE TPEL MagNet papers)")
    summary.inputs = [{"path": f.name, "url": f.url, "sha256": f.sha256} for f in files]
    writer = SourceWriter(output_root / DIRECTORY, summary)
    adapter = PrincetonMagnetAdapter(cache if isinstance(cache, LocalCache) else LocalCache())
    materials, measurements = [], []
    for raw in files:
        material = Path(raw.name).name.split("_database.h5")[0]
        items = [i for i in adapter.discover(str(paths[raw.name].parent)) if i.material == material and i.available]
        data = adapter.extract(items[0])
        problems = adapter.validate(data)
        if problems:
            summary.notes.append(f"{material}: adapter validation reported {problems}")
        cols = data.columns
        codes, labels = data.categorical["excitation"]
        n = len(cols["frequency"][0])
        rows = []
        for i in range(n):
            dp, dn = cols["duty_p"][0][i], cols["duty_n"][0][i]
            rows.append([float(cols["frequency"][0][i]), float(cols["flux_density_peak"][0][i]), float(cols["dc_bias_field"][0][i]),
                         None if np.isnan(dp) else float(dp), None if np.isnan(dn) else float(dn), labels[int(codes[i])],
                         float(cols["temperature"][0][i]) + 273.15, float(cols["loss_density"][0][i])])
        negative_bias = int(np.sum(cols["dc_bias_field"][0] < 0))
        flags = ["dc_bias_small_negative_values"] if negative_bias else []
        if negative_bias:
            summary.flag("rows_with_negative_dc_bias", negative_bias)
        mat = material_identity(material)
        materials.append(mat)
        meta = data.metadata
        core = meta.get("core") or {}
        rid = f"measurement.princeton-magnet-{slug(material)}-core-loss"
        items_ev = [evidence("evidence.princeton-magnet-source", "source", "PrincetonUniversity/magnet", "MIT-licensed packaged MagNet database.",
                             COMMIT_TIME, LICENSE, url=raw.url, version=COMMIT, locator=raw.name, sha256=raw.sha256),
                    evidence("evidence.princeton-magnet-measurement", "measurement", "Princeton Power Electronics Research Lab (MagNet)",
                             "Large-signal core-loss measurement of a toroid specimen (two-winding method, oil bath).", COMMIT_TIME, LICENSE,
                             url="https://mag-net.princeton.edu/", version=COMMIT, locator=f"{raw.name} /dataframe")]
        rec = measurement_record(
            rid=rid, name=f"MagNet {MANUFACTURERS[material]} {material} core loss", material_ref=mat["id"], license=LICENSE,
            evidence_items=items_ev, data_class="experimental_measurement", source_id=SOURCE_ID, quantity="core_loss",
            columns=[column("frequency", "fundamental frequency: Welch PSD estimate of the voltage, 10 Hz resolution"),
                     column("flux_density_peak", "AC amplitude (max B - min B)/2 of the flux density from the voltage integral"),
                     column("dc_bias_field", "DC bias field round(mean H) of the measured field (A/m)"),
                     column("duty_p", "duty of the positive slope (per unit); null for sinusoidal excitation (upstream -1)"),
                     column("duty_n", "duty of the negative slope (per unit); null for sinusoidal excitation (upstream -1)"),
                     {"name": "waveform", "unit": "", "definition": "excitation class: sinusoidal (Duty_P = -1), triangular "
                                                                    "(|Duty_P + Duty_N - 1| <= 1e-6) or trapezoidal"},
                     column("temperature", "oil-bath set point", "degC"),
                     column("loss_density", "mean of v*i divided by the effective volume of the specimen")],
            rows=rows, writer=writer, provenance=_provenance("normalize_princeton_magnet", [mat["id"]]),
            method=meta.get("info_processing"), specimen=f"{material} toroid {core.get('core')}; Ae {core.get('Ae')} m^2; le {core.get('le')} m; "
                                                         f"Ve {core.get('Ve')} m^3; N1 {core.get('N1')}; N2 {core.get('N2')}",
            instrument=meta.get("info_setup"), description=f"Complete packaged MagNet scalar database for {material} ({n} rows).",
            flags=flags, dataset_ref="dataset.princeton-magnet", underlying_source=f"princeton-magnet:{slug(material)}",
            extra={"upstream_file": raw.name, "upstream_sha256": raw.sha256, "excitation_counts": meta.get("excitation_counts"),
                   "specimen_core": {k: core.get(k) for k in ("core", "Ae", "le", "Ve", "N1", "N2", "units") if k in core}})
        measurements.append(rec)
        summary.count("measurement")
        summary.rows("core_loss", n)
    summary.count("magnetic_material", len(materials))
    dataset = dataset_record(
        rid="dataset.princeton-magnet", name="Princeton MagNet packaged core-loss databases", source_id=SOURCE_ID,
        title="Princeton MagNet: packaged scalar core-loss databases (10 materials)", license=LICENSE,
        url=f"{REPOSITORY}/tree/{COMMIT}/src/magnet/data", version=COMMIT, source_commit=COMMIT, publisher="Princeton University (Minjie Chen group)",
        redistribution=True, record_count=len(measurements), citation=CITATION if isinstance(CITATION, str) else "; ".join(CITATION),
        checksums={Path(f.name).name: f.sha256 for f in files}, materials=sorted(MANUFACTURERS),
        description="Large-signal toroid measurements (sinusoidal, triangular, trapezoidal; 25-90 degC; DC bias); independent "
                    "experimental data, unreviewed by AIPE.",
        evidence_items=[evidence("evidence.princeton-magnet-source", "source", "PrincetonUniversity/magnet", "MIT licence, LICENSE at the pinned commit.",
                                 COMMIT_TIME, LICENSE, url=f"{REPOSITORY}/blob/{COMMIT}/LICENSE", version=COMMIT)],
        provenance=_provenance("princeton_magnet_dataset"))
    shard_files = {"materials": writer.shards(materials, "materials", "materials", 100),
                   "measurements": writer.shards(measurements, "measurements", "measurements", 100)}
    writer.json("datasets/princeton-magnet.json", dataset)
    return finish_source(writer, summary, shard_files, ["datasets/princeton-magnet.json"])


__all__ = ["COMMIT", "LICENSE", "SOURCE_ID", "raw_files", "run"]
