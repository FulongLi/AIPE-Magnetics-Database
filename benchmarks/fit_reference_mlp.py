"""Fit the bundled reference ANN: a SYNTHETIC Steinmetz surface surrogate (numpy + scipy only).

Purpose
-------
Produce the weights and metadata in ``src/aipe_magnetics/models/datadriven/reference/`` that
demonstrate the data-driven adapter's *verified inference workflow*: loading existing weights with a
SHA-256 check, normalising inputs, running the numpy backend and de-normalising outputs. It is **not
a physical material model**: the training data are generated from the power law

    Pv [W/m^3] = k * f^alpha * Bpk^beta        (f in Hz, Bpk in T)

with the SYNTHETIC, illustrative parameter set ``k = 2.0, alpha = 1.45, beta = 2.6`` chosen only to
have a ferrite-like order of magnitude. No datasheet or measurement was used.

Method
------
* grid: ``n_f x n_b`` log-spaced points, f in [20 kHz, 500 kHz], Bpk in [20 mT, 300 mT]
* features: ``x = (log10 f, log10 Bpk)`` standardised (z-score) over the grid; target
  ``z = (log10 Pv - mean) / std``; both encoded as ``NormalizationSpec(kind="log10", mean, std)``
* network: MLP 2 -> 8 -> 8 -> 1, tanh hidden layers, identity output (``family="ann"``)
* loss: mean squared error in z-space, analytic gradient by manual back-propagation
* optimiser: full-batch L-BFGS-B (``scipy.optimize.minimize``), fixed seed for the initialisation
* metrics: maximum relative error of Pv on the training grid and on an independent log-uniform
  random validation set (fixed seed) - target max relative error < 2 %

Everything is deterministic for a given numpy/scipy build; the metadata records the versions used.

Usage::

    python benchmarks/fit_reference_mlp.py [--out DIR] [--max-iter N] [--seed S]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.optimize import minimize

from aipe_magnetics import __version__
from aipe_magnetics.common.jsonio import sha256_file
from aipe_magnetics.common.results import ModelIdentity, SourceReference
from aipe_magnetics.models.contract import ValidityDomain, ValidityRange
from aipe_magnetics.models.datadriven.adapter import InferenceConfig, ModelPackage, NormalizationSpec, WeightsBundle
from aipe_magnetics.models.datadriven.architectures import ArchitectureSpec
from aipe_magnetics.models.datadriven.numpy_backend import write_aipe_json

# ----------------------------------------------------------------------------------------------------
# SYNTHETIC parameter set (illustrative only - not a material)
# ----------------------------------------------------------------------------------------------------
SYNTHETIC_K = 2.0
SYNTHETIC_ALPHA = 1.45
SYNTHETIC_BETA = 2.6
F_RANGE_HZ = (20e3, 500e3)
B_RANGE_T = (0.02, 0.3)
GRID = (25, 25)
HIDDEN_SIZES = [8, 8]
SEED = 20261019
MAX_ITER = 20000  # one L-BFGS-B iteration cap for fit(), run() and the CLI (recorded in the metadata)
VALIDATION_SEED = 777
VALIDATION_POINTS = 500
MODEL_ID = "aipe.magnetics.model.datadriven.reference-steinmetz-ann"
MATERIAL_ID = "material.synthetic-steinmetz-reference"
WEIGHTS_NAME = "steinmetz_surrogate_ann.aipe.json"
METADATA_NAME = "metadata.json"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "src" / "aipe_magnetics" / "models" / "datadriven" / "reference"


def steinmetz_surface(f: np.ndarray, b: np.ndarray, *, k: float = SYNTHETIC_K, alpha: float = SYNTHETIC_ALPHA,
                      beta: float = SYNTHETIC_BETA) -> np.ndarray:
    return k * np.power(f, alpha) * np.power(b, beta)


def training_grid(n_f: int = GRID[0], n_b: int = GRID[1]) -> tuple[np.ndarray, np.ndarray]:
    f = np.logspace(np.log10(F_RANGE_HZ[0]), np.log10(F_RANGE_HZ[1]), n_f)
    b = np.logspace(np.log10(B_RANGE_T[0]), np.log10(B_RANGE_T[1]), n_b)
    ff, bb = np.meshgrid(f, b, indexing="ij")
    return ff.ravel(), bb.ravel()


def validation_points(n: int = VALIDATION_POINTS, seed: int = VALIDATION_SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    f = 10 ** rng.uniform(np.log10(F_RANGE_HZ[0]), np.log10(F_RANGE_HZ[1]), n)
    b = 10 ** rng.uniform(np.log10(B_RANGE_T[0]), np.log10(B_RANGE_T[1]), n)
    return f, b


# ----------------------------------------------------------------------------------------------------
# Tiny MLP with manual back-propagation (tanh hidden layers, identity output)
# ----------------------------------------------------------------------------------------------------
def layer_sizes(hidden: list[int]) -> list[tuple[int, int]]:
    sizes = [2, *hidden, 1]
    return [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]


def init_theta(hidden: list[int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    for n_in, n_out in layer_sizes(hidden):
        limit = np.sqrt(6.0 / (n_in + n_out))  # Glorot-uniform
        parts.append(rng.uniform(-limit, limit, size=(n_out, n_in)).ravel())
        parts.append(np.zeros(n_out))
    return np.concatenate(parts)


def unflatten(theta: np.ndarray, hidden: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    layers = []
    offset = 0
    for n_in, n_out in layer_sizes(hidden):
        w = theta[offset:offset + n_out * n_in].reshape(n_out, n_in)
        offset += n_out * n_in
        b = theta[offset:offset + n_out]
        offset += n_out
        layers.append((w, b))
    return layers


def forward(theta: np.ndarray, x: np.ndarray, hidden: list[int]) -> np.ndarray:
    a = x
    layers = unflatten(theta, hidden)
    for i, (w, b) in enumerate(layers):
        z = a @ w.T + b
        a = z if i == len(layers) - 1 else np.tanh(z)
    return a


def loss_and_grad(theta: np.ndarray, x: np.ndarray, y: np.ndarray, hidden: list[int]) -> tuple[float, np.ndarray]:
    layers = unflatten(theta, hidden)
    activations = [x]
    a = x
    for i, (w, b) in enumerate(layers):
        z = a @ w.T + b
        a = z if i == len(layers) - 1 else np.tanh(z)
        activations.append(a)
    n = x.shape[0]
    residual = activations[-1] - y
    loss = float(np.mean(residual ** 2))
    delta = 2.0 * residual / n  # dL/da_L
    grads: list[np.ndarray] = []
    for i in range(len(layers) - 1, -1, -1):
        w, _ = layers[i]
        a_prev, a_cur = activations[i], activations[i + 1]
        dz = delta if i == len(layers) - 1 else delta * (1.0 - a_cur ** 2)
        grads.insert(0, np.sum(dz, axis=0))          # db
        grads.insert(0, (dz.T @ a_prev).ravel())      # dW (out, in) flattened
        delta = dz @ w
    return loss, np.concatenate(grads)


def fit(*, hidden: list[int] | None = None, seed: int = SEED, max_iter: int = MAX_ITER, grid: tuple[int, int] = GRID
        ) -> dict[str, Any]:
    """Train the surrogate; returns parameters, normalisation statistics and training details."""
    hidden = list(HIDDEN_SIZES if hidden is None else hidden)
    f, b = training_grid(*grid)
    pv = steinmetz_surface(f, b)
    log_f, log_b, log_p = np.log10(f), np.log10(b), np.log10(pv)
    stats = {"frequency": (float(log_f.mean()), float(log_f.std())),
             "flux_density_peak": (float(log_b.mean()), float(log_b.std())),
             "loss_density": (float(log_p.mean()), float(log_p.std()))}
    x = np.stack([(log_f - stats["frequency"][0]) / stats["frequency"][1],
                  (log_b - stats["flux_density_peak"][0]) / stats["flux_density_peak"][1]], axis=1)
    y = ((log_p - stats["loss_density"][0]) / stats["loss_density"][1])[:, None]
    theta0 = init_theta(hidden, seed)
    result = minimize(loss_and_grad, theta0, args=(x, y, hidden), jac=True, method="L-BFGS-B",
                      options={"maxiter": max_iter, "maxfun": 4 * max_iter, "ftol": 1e-14, "gtol": 1e-9})
    theta = np.asarray(result.x, dtype=np.float64)
    params: dict[str, np.ndarray] = {}
    for i, (w, bias) in enumerate(unflatten(theta, hidden)):
        params[f"layers.{i}.weight"] = w.copy()
        params[f"layers.{i}.bias"] = bias.copy()
    return {"params": params, "stats": stats, "hidden": hidden, "seed": seed, "grid": grid,
            "optimizer": {"method": "L-BFGS-B", "iterations": int(result.nit), "function_evaluations": int(result.nfev),
                          "final_mse_normalised": float(result.fun), "success": bool(result.success),
                          "message": str(result.message), "max_iter": max_iter}}


def predict_pv(fit_result: dict[str, Any], f: np.ndarray, b: np.ndarray) -> np.ndarray:
    stats, hidden = fit_result["stats"], fit_result["hidden"]
    theta = np.concatenate([np.concatenate([fit_result["params"][f"layers.{i}.weight"].ravel(),
                                            fit_result["params"][f"layers.{i}.bias"]])
                            for i in range(len(layer_sizes(hidden)))])
    x = np.stack([(np.log10(f) - stats["frequency"][0]) / stats["frequency"][1],
                  (np.log10(b) - stats["flux_density_peak"][0]) / stats["flux_density_peak"][1]], axis=1)
    z = forward(theta, x, hidden)[:, 0]
    return 10 ** (z * stats["loss_density"][1] + stats["loss_density"][0])


def evaluate(fit_result: dict[str, Any]) -> dict[str, float]:
    f_tr, b_tr = training_grid(*fit_result["grid"])
    f_va, b_va = validation_points()
    rel_tr = np.abs(predict_pv(fit_result, f_tr, b_tr) / steinmetz_surface(f_tr, b_tr) - 1.0)
    rel_va = np.abs(predict_pv(fit_result, f_va, b_va) / steinmetz_surface(f_va, b_va) - 1.0)
    return {"max_relative_error_training_grid": float(rel_tr.max()), "rms_relative_error_training_grid": float(np.sqrt(np.mean(rel_tr ** 2))),
            "max_relative_error_validation": float(rel_va.max()), "rms_relative_error_validation": float(np.sqrt(np.mean(rel_va ** 2))),
            "validation_points": int(f_va.size), "training_points": int(f_tr.size)}


# ----------------------------------------------------------------------------------------------------
# Package assembly
# ----------------------------------------------------------------------------------------------------
def architecture(hidden: list[int]) -> ArchitectureSpec:
    return ArchitectureSpec(
        family="ann", name=f"steinmetz-surrogate-mlp-2x{'x'.join(str(h) for h in hidden)}x1",
        hyperparameters={"kind": "mlp", "hidden_sizes": list(hidden), "activation": "tanh", "output_activation": "identity"},
        input_features=["frequency", "flux_density_peak"], output_features=["loss_density"],
        input_units={"frequency": "Hz", "flux_density_peak": "T"}, output_units={"loss_density": "W/m^3"},
        description="MLP surrogate of log10(Pv) as a function of (log10 f, log10 Bpk); SYNTHETIC training surface.")


def sources(weights_sha256: str, seed: int) -> list[SourceReference]:
    return [
        SourceReference(
            id="src.steinmetz-1892", kind="publication",
            reference="C. P. Steinmetz, 'On the law of hysteresis', Transactions of the American Institute of Electrical "
                      "Engineers, vol. IX, no. 1, pp. 1-64, 1892 (reprinted in Proc. IEEE, vol. 72, no. 2, pp. 197-221, 1984)",
            note="Origin of the power-law loss description; only the functional form Pv = k f^alpha Bpk^beta is used, "
                 "with SYNTHETIC parameter values."),
        SourceReference(
            id="src.reinert-2001-mse", kind="publication",
            reference="J. Reinert, A. Brockmeyer, R. W. De Doncker, 'Calculation of losses in ferro- and ferrimagnetic "
                      "materials based on the modified Steinmetz equation', IEEE Trans. Ind. Appl., vol. 37, no. 4, "
                      "pp. 1055-1061, 2001", locator="Eq. (1): classical Steinmetz equation Pv = k f^alpha B^beta",
            note="Reference for the classical Steinmetz form used as the synthetic training surface."),
        SourceReference(
            id="src.synthetic-steinmetz-parameters", kind="assumption",
            reference=f"SYNTHETIC illustrative parameter set k={SYNTHETIC_K}, alpha={SYNTHETIC_ALPHA}, beta={SYNTHETIC_BETA} "
                      "(Pv in W/m^3, f in Hz, Bpk in T); not a measured or datasheet material",
            note="synthetic - chosen only for a ferrite-like order of magnitude"),
        SourceReference(
            id="src.reference-mlp-training", kind="internal", version=__version__, sha256=weights_sha256,
            reference=f"benchmarks/fit_reference_mlp.py: numpy MLP fitted with scipy L-BFGS-B, seed {seed}, full-batch, "
                      "on the synthetic grid",
            note="synthetic training data generated by the script itself; weights file digest in sha256"),
    ]


def build_package(fit_result: dict[str, Any], metrics: dict[str, float], weights_name: str, weights_sha256: str,
                  weights_size: int) -> ModelPackage:
    stats = fit_result["stats"]
    spec = architecture(fit_result["hidden"])
    return ModelPackage(
        identity=ModelIdentity(id=MODEL_ID, name="Reference ANN surrogate of a SYNTHETIC Steinmetz surface",
                               version="0.2.0", family="data_driven", method_version="numpy-mlp/0.2"),
        architecture=spec,
        weights=WeightsBundle(format="aipe-json", path=weights_name, sha256=weights_sha256, size_bytes=weights_size,
                              source=sources(weights_sha256, fit_result["seed"])[3],
                              note="dense layer list (W out x in, b, activation) as JSON lists; no binary blobs"),
        inference=InferenceConfig(
            input_normalization={"frequency": NormalizationSpec(kind="log10", mean=stats["frequency"][0], std=stats["frequency"][1]),
                                 "flux_density_peak": NormalizationSpec(kind="log10", mean=stats["flux_density_peak"][0],
                                                                        std=stats["flux_density_peak"][1])},
            output_denormalization={"loss_density": NormalizationSpec(kind="log10", mean=stats["loss_density"][0],
                                                                      std=stats["loss_density"][1])},
            dtype="float64", batch_size=4096, device="cpu",
            domain_map={"frequency": "frequency", "flux_density_peak": "flux_density_peak"}),
        domain=ValidityDomain(
            materials=[MATERIAL_ID], excitation=["sinusoidal"],
            frequency=ValidityRange(min=F_RANGE_HZ[0], max=F_RANGE_HZ[1], unit="Hz"),
            flux_density_peak=ValidityRange(min=B_RANGE_T[0], max=B_RANGE_T[1], unit="T"),
            temperature=None, dc_bias="unsupported", requires_history=False,
            numerical_constraints=["frequency and flux_density_peak must be strictly positive (log10 input features)"],
            known_limitations=[
                "SYNTHETIC surrogate: trained on a power law with illustrative parameters, not on measured data; "
                "it does not describe any real material.",
                "No temperature dependence and no DC-bias dependence are modelled.",
                "Sinusoidal-excitation Steinmetz surface only; the surrogate interpolates inside the training grid "
                "(20-500 kHz, 20-300 mT) and must not be extrapolated.",
            ]),
        sources=sources(weights_sha256, fit_result["seed"]), synthetic=True,
        description="Demonstrates the verified inference workflow of the data-driven adapter (sha256-checked weights, "
                    "normalisation, numpy MLP inference). Target: log10(Pv) = f(log10 f, log10 Bpk) of a SYNTHETIC "
                    "Steinmetz surface.",
        training={
            "basis": "synthetic",
            "script": "benchmarks/fit_reference_mlp.py",
            "seed": fit_result["seed"],
            "validation_seed": VALIDATION_SEED,
            "synthetic_parameters": {"k": SYNTHETIC_K, "alpha": SYNTHETIC_ALPHA, "beta": SYNTHETIC_BETA,
                                     "equation": "Pv = k * f**alpha * Bpk**beta", "units": "Pv W/m^3, f Hz, Bpk T"},
            "grid": {"n_frequency": fit_result["grid"][0], "n_flux_density_peak": fit_result["grid"][1],
                     "frequency_hz": list(F_RANGE_HZ), "flux_density_peak_t": list(B_RANGE_T), "spacing": "log"},
            "features": {"inputs": "log10 then z-score (NormalizationSpec kind='log10' with mean/std)",
                         "output": "z-scored log10(Pv); inverse 10**(z*std+mean)"},
            "network": {"layers": "2 -> " + " -> ".join(str(h) for h in fit_result["hidden"]) + " -> 1",
                        "activation": "tanh", "output_activation": "identity",
                        "parameter_count": spec.parameter_count()},
            "optimizer": fit_result["optimizer"],
            "metrics": metrics,
            "target": {"max_relative_error": 0.02},
            "environment": {"numpy": np.__version__, "scipy": scipy.__version__, "aipe_magnetics": __version__},
        })


def run(out_dir: Path, *, max_iter: int = MAX_ITER, seed: int = SEED, hidden: list[int] | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_result = fit(hidden=hidden, seed=seed, max_iter=max_iter)
    metrics = evaluate(fit_result)
    spec = architecture(fit_result["hidden"])
    weights_path = write_aipe_json(out_dir / WEIGHTS_NAME, spec, fit_result["params"],
                                   extra={"synthetic": True, "note": "SYNTHETIC Steinmetz-surface surrogate; see metadata.json"},
                                   overwrite=True)
    digest = sha256_file(weights_path)
    package = build_package(fit_result, metrics, WEIGHTS_NAME, digest, weights_path.stat().st_size)
    package.save(out_dir / METADATA_NAME, overwrite=True)
    return {"weights": weights_path, "metadata": out_dir / METADATA_NAME, "sha256": digest, "metrics": metrics,
            "optimizer": fit_result["optimizer"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory (default: package reference dir)")
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    info = run(args.out, max_iter=args.max_iter, seed=args.seed)
    print(f"weights : {info['weights']}")
    print(f"metadata: {info['metadata']}")
    print(f"sha256  : {info['sha256']}")
    print(f"L-BFGS-B: {info['optimizer']['iterations']} iterations, final normalised MSE {info['optimizer']['final_mse_normalised']:.3e}")
    for key, value in info["metrics"].items():
        print(f"{key}: {value}")
    ok = info["metrics"]["max_relative_error_training_grid"] < 0.02
    print("TARGET max relative error < 2 % on the training grid:", "MET" if ok else "NOT MET")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
