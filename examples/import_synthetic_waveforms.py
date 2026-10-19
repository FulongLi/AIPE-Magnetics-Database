#!/usr/bin/env python
"""Complete, executable, OFFLINE data-import example with clearly synthetic waveforms.

What it does (nothing is downloaded, nothing is measured, no loss value is claimed):

1. Generates synthetic flux-density sequences ``B(t)`` - one steady-state period per sequence -
   for two shapes (sinusoidal, symmetric triangular), three frequencies (50 / 100 / 200 kHz) and
   three peak values (50 / 100 / 200 mT): 18 sequences x ``--samples`` points, deliberately in
   **mT** and **kHz** so that the importer has to normalise units.
2. Saves them the way a user would hand them over (``waveforms-synthetic-source.npz``).
3. Imports that file with :func:`aipe_magnetics.data.importers.import_waveform_sequences`, which
   writes the AIPE HDF5 waveform file (layout ``aipe.magnetics.waveforms/0.2``: SI units, original
   units retained as attributes, NaN = missing, per-array SHA-256, gzip chunks) and builds the
   ``waveform_dataset`` record with the file checksum in ``artifacts``.
4. Writes the companion ``dataset`` record (source identity, version, licence, file checksums) and two
   parametric ``excitation_waveform`` records (one per shape) that reference the dataset.
5. Validates every record against schema 0.2.0 (JSON Schema + semantics + cross-references),
   verifies the HDF5 checksums and prints a summary.

Conditions stored per sequence are **frequency and temperature only**; the records state that the
data are synthetic (``synthetic: true``, assumption evidence, ``aipe.synthetic`` extension with the
generator parameters). Equations: sinusoid ``B(t) = Bpk sin(2 pi t / T)``; triangle
``B(t) = Bpk (2/pi) asin(sin(2 pi t / T))`` (zero at t = 0, +Bpk at T/4, -Bpk at 3T/4);
samples at ``t_k = k T / N``, ``k = 0 .. N-1``.

Usage::

    python examples/import_synthetic_waveforms.py [--out DIR] [--overwrite] [--samples N]

Default output directory: ``examples/synthetic`` (the committed fixture set, < 200 kB).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from aipe_magnetics import TOOL_ID, TOOL_NAME, __version__
from aipe_magnetics.common.jsonio import dump, load, sha256_file
from aipe_magnetics.data.hdf5 import WaveformMeta, WaveformStore
from aipe_magnetics.data.importers import import_waveform_sequences
from aipe_magnetics.data.records import SCHEMA_VERSION, MagneticRecord
from aipe_magnetics.data.schema import validate_catalogue, validate_record

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "synthetic"

SOURCE_NPZ = "waveforms-synthetic-source.npz"
WAVEFORM_H5 = "waveforms-synthetic.h5"
DATASET_ID = "dataset.aipe-synthetic-waveforms"
WAVEFORM_ID = "waveform-dataset.aipe-synthetic-flux-density"
EXCITATION_IDS = {"sinusoidal": "excitation.aipe-synthetic-sinusoidal", "triangular": "excitation.aipe-synthetic-triangular"}
RECORD_FILES = {
    DATASET_ID: "dataset.aipe-synthetic-waveforms.json",
    WAVEFORM_ID: "waveform-dataset.aipe-synthetic-flux-density.json",
    EXCITATION_IDS["sinusoidal"]: "excitation.aipe-synthetic-sinusoidal.json",
    EXCITATION_IDS["triangular"]: "excitation.aipe-synthetic-triangular.json",
}

SHAPES = ("sinusoidal", "triangular")
FREQUENCIES_KHZ = (50.0, 100.0, 200.0)
PEAKS_MT = (50.0, 100.0, 200.0)
TEMPERATURE_DEGC = 25.0
CREATED_AT = "2026-10-19T00:00:00Z"  # fixed: authoring date of the example, not a build timestamp

EVIDENCE = {
    "id": "evidence.aipe-synthetic-waveform-generator",
    "kind": "assumption",
    "reference": "examples/import_synthetic_waveforms.py (AIPE Magnetics v0.2 synthetic waveform generator)",
    "description": ("Sequences are generated analytically (sinusoid and arcsin(sin) triangle). They are not "
                    "measurements, belong to no material and carry no loss information."),
    "created_at": CREATED_AT,
    "review_status": "unreviewed",
    "license": "NOASSERTION",
}


# --------------------------------------------------------------------------------------------------
# Synthetic generation
# --------------------------------------------------------------------------------------------------
def generate_sequences(n_samples: int = 256) -> dict[str, Any]:
    """Return the synthetic sequence set (mT / kHz / degC as a user might supply them)."""
    if n_samples < 8:
        raise ValueError("n_samples must be at least 8")
    phase = 2.0 * np.pi * np.arange(n_samples) / n_samples  # t_k / T * 2 pi, k = 0..N-1
    rows: list[np.ndarray] = []
    shapes: list[str] = []
    freqs: list[float] = []
    peaks: list[float] = []
    for shape in SHAPES:
        unit_wave = np.sin(phase) if shape == "sinusoidal" else (2.0 / np.pi) * np.arcsin(np.sin(phase))
        for f_khz in FREQUENCIES_KHZ:
            for b_mt in PEAKS_MT:
                rows.append(b_mt * unit_wave)
                shapes.append(shape)
                freqs.append(f_khz)
                peaks.append(b_mt)
    flux_density_mT = np.vstack(rows)
    return {
        "flux_density_mT": flux_density_mT,
        "frequency_kHz": np.asarray(freqs, dtype=np.float64),
        "temperature_degC": np.full(len(rows), TEMPERATURE_DEGC, dtype=np.float64),
        "shapes": shapes,
        "peaks_mT": peaks,
        "n_samples": n_samples,
    }


# --------------------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------------------
def _envelope(record_id: str, kind: str, name: str, description: str, properties: dict[str, Any], **extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "id": record_id, "kind": kind, "name": name, "synthetic": True,
        "description": description, "license": "NOASSERTION", "properties": properties,
        "evidence_refs": [EVIDENCE["id"]], "evidence": [dict(EVIDENCE)],
        "provenance": {"activity": "generated by examples/import_synthetic_waveforms.py",
                       "tool": {"id": TOOL_ID, "name": TOOL_NAME, "version": __version__}, "inputs": [],
                       "normalization": "none; record authored directly in SI"},
        "tags": ["synthetic", "example"],
    }
    data.update(extra)
    return data


def build_dataset_record(generated: dict[str, Any], checksums: dict[str, str]) -> dict[str, Any]:
    return _envelope(
        DATASET_ID, "dataset", "AIPE synthetic flux-density waveform set (illustrative)",
        "Synthetic B(t) sequences generated analytically for the data-import example. Not a measurement; no material; "
        "no loss values.",
        {"source_id": "aipe.synthetic", "title": "AIPE Magnetics synthetic waveform example set",
         "publisher": "AIPE Magnetics (synthetic generator, this repository)", "version": __version__,
         "license": "NOASSERTION", "citation": "AIPE Magnetics v0.2, examples/import_synthetic_waveforms.py",
         "checksums": checksums, "record_count": int(generated["flux_density_mT"].shape[0]),
         "description": "18 sequences: 2 shapes x 3 frequencies (50/100/200 kHz) x 3 peaks (50/100/200 mT)."},
    )


def build_excitation_records(generated: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for shape in SHAPES:
        rows = [i for i, s in enumerate(generated["shapes"]) if s == shape]
        properties: dict[str, Any] = {
            "waveform_type": shape, "controlled_quantity": "flux_density", "timebase": "periodic_from_zero",
            "reference_direction": "winding reference direction (right-hand rule); B positive along it",
            "dataset_ref": WAVEFORM_ID,
            "description": (f"Parametric description of rows {rows[0]}..{rows[-1]} of /sequences/flux_density: one period per "
                            f"row, {generated['n_samples']} samples, frequencies {list(FREQUENCIES_KHZ)} kHz and peaks "
                            f"{list(PEAKS_MT)} mT (per-row values in /conditions/frequency and extensions['aipe.synthetic'])."),
        }
        if shape == "triangular":
            properties["duty"] = 0.5
        records.append(_envelope(
            EXCITATION_IDS[shape], "excitation_waveform", f"Synthetic {shape} flux-density excitation (illustrative)",
            f"{shape.capitalize()} B(t) family of the synthetic example; no loss information.", properties))
    return records


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------
def run(out: Path, *, overwrite: bool = False, n_samples: int = 256, quiet: bool = False) -> dict[str, Any]:
    """Generate, import, validate; returns a summary dict. Raises on any validation failure."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    outputs = [out / SOURCE_NPZ, out / WAVEFORM_H5] + [out / name for name in RECORD_FILES.values()]
    existing = [p for p in outputs if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{len(existing)} output file(s) already exist in {out} (e.g. {existing[0].name}); pass --overwrite")
    for path in existing:
        path.unlink()

    generated = generate_sequences(n_samples)
    np.savez(out / SOURCE_NPZ, flux_density_mT=generated["flux_density_mT"], frequency_kHz=generated["frequency_kHz"],
             temperature_degC=generated["temperature_degC"])

    # The dataset record is the catalogue context of the import (the waveform record's dataset_ref must
    # resolve); its checksum of the HDF5 file is added once that file exists.
    dataset_record = build_dataset_record(generated, {SOURCE_NPZ: sha256_file(out / SOURCE_NPZ)})
    meta = WaveformMeta(
        source_id="aipe.synthetic", dataset_version=__version__, license="NOASSERTION", timebase="uniform_per_period",
        description=("Synthetic B(t): one period per row; rows 0-8 sinusoidal, rows 9-17 symmetric triangular; "
                     "frequency and temperature per row in /conditions; no loss values."))
    result = import_waveform_sequences(
        out / SOURCE_NPZ, out / WAVEFORM_H5,
        arrays={"flux_density": ("flux_density_mT", "mT")},
        conditions={"frequency": ("frequency_kHz", "kHz"), "temperature": ("temperature_degC", "degC")},
        meta=meta, record_id=WAVEFORM_ID, name="Synthetic flux-density waveform file (illustrative)", evidence=EVIDENCE,
        synthetic=True, dataset_ref=DATASET_ID, catalogue=[dataset_record], tags=["synthetic", "example"],
        description="HDF5 waveform file of the synthetic example; see the aipe.synthetic extension for the generator parameters.",
        destination_record=None)
    dataset_record["properties"]["checksums"][WAVEFORM_H5] = sha256_file(out / WAVEFORM_H5)
    waveform_record = result.record.to_json_dict()
    waveform_record["extensions"]["aipe.synthetic"] = {
        "generator": "examples/import_synthetic_waveforms.py", "shapes_by_row": generated["shapes"],
        "peak_mT_by_row": generated["peaks_mT"], "frequency_kHz_by_row": generated["frequency_kHz"].tolist(),
        "temperature_degC": TEMPERATURE_DEGC, "samples_per_period": n_samples,
        "equations": {"sinusoidal": "B(t) = Bpk sin(2 pi t / T)", "triangular": "B(t) = Bpk (2/pi) asin(sin(2 pi t / T))"},
    }
    records = [dataset_record, waveform_record, *build_excitation_records(generated)]

    errors: list[str] = []
    for record in records:
        errors.extend(f"{record['id']}: {e}" for e in validate_record(record))
    errors.extend(validate_catalogue(records))
    checksum_problems = WaveformStore().verify_checksums(out / WAVEFORM_H5)
    if errors or checksum_problems:
        raise RuntimeError("validation failed: " + "; ".join(errors + checksum_problems))

    for record in records:
        dump(record, out / RECORD_FILES[record["id"]], overwrite=False)
    # self-consistency of the written set
    parsed = [MagneticRecord.model_validate(load(out / name)) for name in RECORD_FILES.values()]
    if validate_catalogue(parsed):
        raise RuntimeError("written records do not cross-validate")

    h5_size = (out / WAVEFORM_H5).stat().st_size
    summary = {
        "out": str(out), "n_sequences": int(generated["flux_density_mT"].shape[0]), "n_samples": n_samples,
        "waveform_file": WAVEFORM_H5, "waveform_file_bytes": h5_size, "waveform_sha256": sha256_file(out / WAVEFORM_H5),
        "records": list(RECORD_FILES.values()), "validation_errors": 0, "checksum_mismatches": 0,
        "import_warnings": list(result.warnings),
    }
    if not quiet:
        print(f"synthetic waveforms written to {out}")
        print(f"  sequences: {summary['n_sequences']} x {n_samples} samples (B in T; original unit mT), conditions: frequency [Hz], temperature [K]")
        print(f"  {WAVEFORM_H5}: {h5_size} bytes, sha256 {summary['waveform_sha256'][:16]}..., checksums verified")
        for name in RECORD_FILES.values():
            print(f"  record: {name}")
        print("  validation: 4 records valid under schema 0.2.0; cross-references resolved; no loss values claimed (synthetic)")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline synthetic waveform import example (AIPE Magnetics v0.2).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs in --out")
    parser.add_argument("--samples", type=int, default=256, help="samples per period (default 256)")
    args = parser.parse_args(argv)
    try:
        run(args.out, overwrite=args.overwrite, n_samples=args.samples)
    except (FileExistsError, RuntimeError, ValueError) as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
