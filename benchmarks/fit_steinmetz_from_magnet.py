#!/usr/bin/env python3
"""Reproducibly fit classical Steinmetz parameter sets from the Princeton MagNet scalar databases.

Input: a local checkout of https://github.com/PrincetonUniversity/magnet (MIT) or any directory with
``<material>_database.h5`` files. Output: a JSON document with one parameter set per material and
measured temperature (sinusoidal excitation, zero DC bias), including fit ranges, metrics, the source
file SHA-256, the upstream commit and the exact method, so the derivation can be repeated.

Example::

    python benchmarks/fit_steinmetz_from_magnet.py --magnet-root ../_refs/magnet \
        --output src/aipe_magnetics/data/catalog/derived/princeton-magnet-steinmetz-fits.json \
        --fmin 50e3 --fmax 500e3 --bmin 0.01 --bmax 0.3

The fits are *derived data* produced by this script from measured data; they are stored with
``basis: measurement_fit`` and never presented as manufacturer values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aipe_magnetics import __version__
from aipe_magnetics.data.sources.princeton_magnet_db import (
    UPSTREAM_COMMIT,
    UPSTREAM_LICENSE,
    UPSTREAM_REPOSITORY,
    find_database_files,
    fit_steinmetz_by_temperature,
    read_magnet_database,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--magnet-root", type=Path, required=True, help="checkout of PrincetonUniversity/magnet or its data dir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--materials", nargs="*", default=None)
    parser.add_argument("--fmin", type=float, default=50e3, help="lower frequency bound of the fit window (Hz)")
    parser.add_argument("--fmax", type=float, default=500e3, help="upper frequency bound of the fit window (Hz)")
    parser.add_argument("--bmin", type=float, default=0.01, help="lower flux density amplitude bound (T)")
    parser.add_argument("--bmax", type=float, default=0.30, help="upper flux density amplitude bound (T)")
    parser.add_argument("--min-points", type=int, default=30)
    parser.add_argument("--commit", default=UPSTREAM_COMMIT, help="upstream commit hash of the checkout")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    files = find_database_files(args.magnet_root)
    materials = args.materials or sorted(files)
    document = {
        "document": "aipe.magnetics.derived.steinmetz-fits/0.2",
        "tool": {"id": "aipe.magnetics", "version": __version__, "script": "benchmarks/fit_steinmetz_from_magnet.py"},
        "method": ("Ordinary least squares in log space, log(Pv) = log(k) + alpha*log(f) + beta*log(B), on sinusoidal "
                   "(Duty_P == Duty_N == -1), zero-DC-bias rows, grouped by measured temperature (rounded to 1 degC), "
                   f"restricted to {args.fmin:g} <= f <= {args.fmax:g} Hz and {args.bmin:g} <= B <= {args.bmax:g} T. "
                   "Units: k in W/m^3 with f in Hz and B (amplitude) in T."),
        "source": {"repository": UPSTREAM_REPOSITORY, "commit": args.commit, "license": UPSTREAM_LICENSE,
                   "citation": ("D. Serrano et al., 'Why MagNet: Quantifying the Complexity of Modeling Power Magnetic Material "
                                "Characteristics', IEEE Trans. Power Electron., 2023, doi:10.1109/TPEL.2023.3291084; "
                                "H. Li et al., 'How MagNet: Machine Learning Framework for Modeling Power Magnetic Material "
                                "Characteristics', IEEE Trans. Power Electron., 2023, doi:10.1109/TPEL.2023.3309232"),
                   "files": {}},
        "fits": [],
    }
    for material in materials:
        if material not in files:
            print(f"warning: no database file for {material}", file=sys.stderr)
            continue
        db = read_magnet_database(files[material], material=material)
        document["source"]["files"][material] = {"file": files[material].name, "sha256": db.sha256, "n_points": db.n_points,
                                                  "core": db.core_info, "temperatures_degC": db.temperatures()}
        try:
            fits = fit_steinmetz_by_temperature(db, frequency_range=(args.fmin, args.fmax),
                                                flux_density_range=(args.bmin, args.bmax), min_points=args.min_points)
        except Exception as error:  # noqa: BLE001 - report and continue with other materials
            print(f"warning: {material}: {error}", file=sys.stderr)
            continue
        for fit in fits:
            document["fits"].append(fit.to_dict())
            print(f"{material:5s} {fit.temperature_degC:5.0f} degC  k={fit.k:.4g} alpha={fit.alpha:.4f} beta={fit.beta:.4f} "
                  f"n={fit.n_points:5d}  max|rel|={fit.max_rel_error:.1%} mean|rel|={fit.mean_abs_rel_error:.1%}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        print(f"refusing to overwrite {args.output} (use --overwrite)", file=sys.stderr)
        return 1
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(document['fits'])} parameter sets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
