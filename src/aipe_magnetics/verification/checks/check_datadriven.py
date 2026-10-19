"""Verification checks for the data-driven model adapter family (ANN, LSTM, Neural ODE, backends).

All checks are deterministic, offline, fast and leave the repository untouched (temporary copies
are used wherever files are tampered with). Capabilities:

* ``aipe.magnetics.model.datadriven.reference-steinmetz-ann`` - the bundled SYNTHETIC reference ANN
* ``aipe.magnetics.model.datadriven.lstm``                    - numpy LSTM implementation
* ``aipe.magnetics.model.datadriven.neural-ode``              - Neural-ODE reference implementation
* ``aipe.magnetics.model.datadriven``                         - adapter: integrity, domain policy, backends
* ``aipe.magnetics.integration.magnethub``                    - MagNet-Hub input/output mapping (no download)
"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from aipe_magnetics.common.errors import DataIntegrityError, InvalidInputError, OutOfDomainError, UnsupportedOperationError
from aipe_magnetics.common.jsonio import dump, load as load_json, sha256_file
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import ModelIdentity, SourceReference
from aipe_magnetics.models.contract import ValidityDomain, ValidityRange
from aipe_magnetics.models.datadriven import (
    REFERENCE_MODEL_ID,
    ArchitectureSpec,
    DataDrivenModel,
    InferenceConfig,
    ModelPackage,
    NeuralODEModel,
    WeightsBundle,
    get_backend,
    linear_forced_response,
    load_reference_package,
    reference_package_path,
    reference_steinmetz_ann,
    write_aipe_json,
)
from aipe_magnetics.models.datadriven.magnethub_backend import (
    MAGNETHUB_INSTALL_HINT,
    collect_outputs,
    kelvin_to_celsius,
    magnethub_available,
    magnethub_package,
    prepare_inputs,
)
from aipe_magnetics.models.datadriven.numpy_backend import build_network
from aipe_magnetics.models.datadriven.torch_backend import TORCH_INSTALL_HINT, torch_available
from aipe_magnetics.verification.evidence import OperatingRange
from aipe_magnetics.verification.suite import CheckResult, check

CAP_REFERENCE = REFERENCE_MODEL_ID
CAP_ADAPTER = "aipe.magnetics.model.datadriven"
CAP_LSTM = "aipe.magnetics.model.datadriven.lstm"
CAP_NEURAL_ODE = "aipe.magnetics.model.datadriven.neural-ode"
CAP_MAGNETHUB = "aipe.magnetics.integration.magnethub"
MATERIAL_ID = "material.synthetic-steinmetz-reference"
REFERENCE_RANGE = OperatingRange(
    materials=[MATERIAL_ID], excitation=["sinusoidal"],
    frequency=ValidityRange(min=20e3, max=500e3, unit="Hz"),
    flux_density_peak=ValidityRange(min=0.02, max=0.3, unit="T"),
    note="training grid of the SYNTHETIC reference surrogate")

_SYNTHETIC_SOURCE = SourceReference(id="src.check-synthetic-weights", kind="assumption",
                                    reference="hand-constructed synthetic weights used by a verification check",
                                    note="synthetic; carries no physical meaning")


def _identity(suffix: str) -> ModelIdentity:
    return ModelIdentity(id=f"aipe.magnetics.model.datadriven.check-{suffix}", name=f"check {suffix}", version="0.2.0",
                         family="data_driven")


# --------------------------------------------------------------------------------------------------
# Helpers shared with the pytest suite
# --------------------------------------------------------------------------------------------------
def build_linear_neural_ode_package(directory: str | Path, *, a: float, b: float) -> ModelPackage:
    """Neural-ODE package whose two identity dense layers compute exactly ``f(y, u) = -a*y + b*u``.

    Layer 0 (3 x 2, identity) copies ``[y, u]`` into the first two hidden units and zeroes the
    third; layer 1 (1 x 3, identity) has weights ``[-a, b, 0]``. ``W1 @ W0 = [[-a, b]]`` exactly.
    """
    directory = Path(directory)
    spec = ArchitectureSpec(
        family="neural_ode", name="linear-vector-field-check",
        hyperparameters={"kind": "mlp_vector_field", "state_size": 1, "input_size": 1, "hidden_sizes": [3],
                         "activation": "identity", "output_activation": "identity"},
        input_features=["y", "u"], output_features=["y"], input_units={"y": "1", "u": "1"}, output_units={"y": "1"})
    params = {
        "layers.0.weight": np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        "layers.0.bias": np.zeros(3),
        "layers.1.weight": np.array([[-a, b, 0.0]]),
        "layers.1.bias": np.zeros(1),
    }
    weights = write_aipe_json(directory / "linear_vector_field.aipe.json", spec, params, overwrite=True)
    package = ModelPackage(
        identity=_identity("neural-ode-linear"), architecture=spec,
        weights=WeightsBundle(format="aipe-json", path=weights.name, sha256=sha256_file(weights), source=_SYNTHETIC_SOURCE),
        inference=InferenceConfig(), domain=ValidityDomain(), sources=[_SYNTHETIC_SOURCE], synthetic=True,
        description="dy/dt = -a*y + b*u(t) encoded as an identity-activation MLP (closed-form reference case)")
    return package.with_base_dir(directory)


def lstm_check_spec(output_mode: str = "last") -> ArchitectureSpec:
    return ArchitectureSpec(
        family="lstm", name=f"lstm-check-{output_mode}",
        hyperparameters={"kind": "lstm", "hidden_size": 4, "num_layers": 2, "output_mode": output_mode,
                         "head_hidden_sizes": [3], "head_activation": "tanh", "head_output_activation": "identity"},
        input_features=["flux_density", "frequency"], output_features=["field", "loss_density"],
        sequence_features=["flux_density"] + (["field", "loss_density"] if output_mode == "sequence" else []),
        input_units={"flux_density": "T", "frequency": "Hz"}, output_units={"field": "A/m", "loss_density": "W/m^3"})


def random_parameters(spec: ArchitectureSpec, seed: int = 7, scale: float = 0.5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: rng.normal(scale=scale, size=shape) for name, shape in spec.expected_parameter_shapes().items()}


def naive_activation(name: str, s: float) -> float:
    """Scalar activations written from their definitions with ``math`` only (independent of the backend)."""
    if name == "identity":
        return s
    if name == "tanh":
        return math.tanh(s)
    if name == "relu":
        return s if s > 0.0 else 0.0
    if name == "sigmoid":
        return 1.0 / (1.0 + math.exp(-s))
    if name == "gelu":
        return 0.5 * s * (1.0 + math.erf(s / math.sqrt(2.0)))
    raise ValueError(f"naive reference: unsupported activation {name!r}")


def naive_lstm_forward(spec: ArchitectureSpec, params: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """Independent, scalar-loop LSTM + dense head using only ``math`` (no vectorisation, no shared code).

    Implements the PyTorch gate equations literally: for every sample, layer and step, each of the
    ``4H`` gate pre-activations is accumulated term by term, gates are sliced in the order i, f, g, o.
    The head applies every activation of the descriptor explicitly (:func:`naive_activation`).
    """
    hp = spec.typed_hyperparameters()
    hidden = int(hp.hidden_size)  # type: ignore[union-attr]
    n_batch, length, n_in = x.shape

    def sigmoid(v: float) -> float:
        return 1.0 / (1.0 + math.exp(-v))

    def dense_stack(vector: list[float]) -> list[float]:
        for i, (_, _, act) in enumerate(spec.dense_layer_sizes()):
            w, bias = params[f"head.{i}.weight"], params[f"head.{i}.bias"]
            out = []
            for r in range(w.shape[0]):
                s = float(bias[r])
                for j in range(w.shape[1]):
                    s += float(w[r, j]) * vector[j]
                out.append(naive_activation(act, s))
            vector = out
        return vector

    results = []
    for n in range(n_batch):
        layer_input = [[float(x[n, t, j]) for j in range(n_in)] for t in range(length)]
        for k in range(int(hp.num_layers)):  # type: ignore[union-attr]
            w_ih, w_hh = params[f"lstm.weight_ih_l{k}"], params[f"lstm.weight_hh_l{k}"]
            b_ih, b_hh = params[f"lstm.bias_ih_l{k}"], params[f"lstm.bias_hh_l{k}"]
            h = [0.0] * hidden
            c = [0.0] * hidden
            sequence = []
            for t in range(length):
                x_t = layer_input[t]
                gates = []
                for g in range(4 * hidden):
                    s = float(b_ih[g]) + float(b_hh[g])
                    for j in range(len(x_t)):
                        s += float(w_ih[g, j]) * x_t[j]
                    for j in range(hidden):
                        s += float(w_hh[g, j]) * h[j]
                    gates.append(s)
                i_g = [sigmoid(gates[j]) for j in range(hidden)]
                f_g = [sigmoid(gates[hidden + j]) for j in range(hidden)]
                g_g = [math.tanh(gates[2 * hidden + j]) for j in range(hidden)]
                o_g = [sigmoid(gates[3 * hidden + j]) for j in range(hidden)]
                c = [f_g[j] * c[j] + i_g[j] * g_g[j] for j in range(hidden)]
                h = [o_g[j] * math.tanh(c[j]) for j in range(hidden)]
                sequence.append(list(h))
            layer_input = sequence
        if hp.output_mode == "sequence":  # type: ignore[union-attr]
            results.append([dense_stack(step) for step in layer_input])
        else:
            results.append(dense_stack(layer_input[-1]))
    return np.asarray(results, dtype=np.float64)


def _copy_reference(directory: Path) -> tuple[Path, Path]:
    metadata = reference_package_path()
    package = load_reference_package()
    weights = package.weights_path()
    shutil.copy(metadata, directory / metadata.name)
    shutil.copy(weights, directory / weights.name)
    return directory / metadata.name, directory / weights.name


# --------------------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------------------
@check(
    id="datadriven.reference-ann-loads-and-is-deterministic",
    capability=CAP_REFERENCE, level="L1", kind="software",
    description="The bundled SYNTHETIC reference ANN package validates, its weights digest verifies, repeated inference "
                "is bit-identical and a batch evaluation agrees with per-sample evaluations to 1e-12 relative.",
    method="Load reference/metadata.json, recompute the weights SHA-256, run predict() twice on the same inputs "
           "(bit-for-bit comparison) and compare a 5-sample batch evaluation against per-sample evaluations with "
           "np.allclose(rtol=1e-12, atol=0); bit equality between batched and single evaluations is not a property of "
           "the model because BLAS gemm and gemv kernels round differently.",
    range=REFERENCE_RANGE,
)
def reference_ann_loads_and_is_deterministic() -> CheckResult:
    package = load_reference_package()
    model = reference_steinmetz_ann()
    notes: list[str] = []
    ok = package.synthetic is True and package.identity.id == REFERENCE_MODEL_ID
    ok &= model.integrity["verified"] is True and model.integrity["sha256_actual"] == package.weights.sha256
    ok &= package.domain.materials == [MATERIAL_ID]
    ok &= not model.load_warnings
    inputs = {"frequency": [25e3, 100e3, 400e3, 50e3, 250e3], "flux_density_peak": [0.05, 0.1, 0.25, 0.2, 0.03]}
    first = model.predict(inputs)
    second = model.predict(inputs)
    batch = np.asarray(first.diagnostics["outputs"]["loss_density"])
    repeat_identical = bool(np.array_equal(batch, np.asarray(second.diagnostics["outputs"]["loss_density"])))
    ok &= repeat_identical
    singles = np.array([model.predict({"frequency": f, "flux_density_peak": b}).values["loss_density"].value
                        for f, b in zip(inputs["frequency"], inputs["flux_density_peak"])])
    batch_vs_single = float(np.max(np.abs(batch / singles - 1.0)))
    batch_consistent = bool(np.allclose(batch, singles, rtol=1e-12, atol=0.0))
    ok &= batch_consistent
    scalar = model.predict({"frequency": 1e5, "flux_density_peak": 0.1})
    ok &= scalar.values["loss_density"].unit == "W/m^3" and scalar.diagnostics["backend"] == "numpy"
    ok &= any(lim.code == "synthetic_parameters" for lim in scalar.limitations)
    declared = package.training.get("metrics", {})
    ok &= float(declared.get("max_relative_error_training_grid", 1.0)) < 0.02
    notes.append("the package is SYNTHETIC (training basis 'synthetic'); it demonstrates the inference workflow only")
    return CheckResult(passed=bool(ok), notes=notes, metrics={
        "sha256_verified": model.integrity["verified"], "parameter_count": package.architecture.parameter_count(),
        "declared_max_rel_error_training_grid": declared.get("max_relative_error_training_grid"),
        "repeat_bit_identical": repeat_identical, "batch_vs_single_max_rel_diff": batch_vs_single,
        "batch_consistent_rtol_1e-12": batch_consistent})


@check(
    id="datadriven.reference-ann-reproduces-steinmetz",
    capability=CAP_REFERENCE, level="L2", kind="analytical_reference",
    description="On an independent random grid the reference ANN reproduces the SYNTHETIC Steinmetz surface it was "
                "trained on to better than 3 % maximum relative error.",
    method="Draw 400 log-uniform (f, Bpk) points (seed 20261020) inside the training ranges, evaluate the surrogate and "
           "compare with Pv = k f^alpha Bpk^beta using the synthetic parameters recorded in the package metadata.",
    range=REFERENCE_RANGE,
)
def reference_ann_reproduces_steinmetz() -> CheckResult:
    package = load_reference_package()
    model = reference_steinmetz_ann()
    synthetic = package.training["synthetic_parameters"]
    k, alpha, beta = float(synthetic["k"]), float(synthetic["alpha"]), float(synthetic["beta"])
    rng = np.random.default_rng(20261020)
    f = 10 ** rng.uniform(np.log10(20e3), np.log10(500e3), 400)
    b = 10 ** rng.uniform(np.log10(0.02), np.log10(0.3), 400)
    predicted = np.asarray(model.predict({"frequency": f, "flux_density_peak": b}).diagnostics["outputs"]["loss_density"])
    analytic = k * f ** alpha * b ** beta
    rel = np.abs(predicted / analytic - 1.0)
    max_rel = float(rel.max())
    return CheckResult(passed=max_rel < 0.03, metrics={
        "max_relative_error": max_rel, "rms_relative_error": float(np.sqrt(np.mean(rel ** 2))), "points": int(f.size),
        "synthetic_parameters": {"k": k, "alpha": alpha, "beta": beta}},
        notes=["analytic reference is the SYNTHETIC training surface itself (no experimental data involved)"])


@check(
    id="datadriven.weights-integrity",
    capability=CAP_ADAPTER, level="L1", kind="invalid_input_rejection",
    description="Tampered or mis-declared weights are rejected with DataIntegrityError; a missing digest is reported as "
                "'weights_unverified'; executable TorchScript bundles without a digest are refused.",
    method="Copy the reference package to a temporary directory and (a) perturb one weight, (b) declare a wrong digest, "
           "(c) drop the digest, (d) declare a 'torchscript' bundle without digest; observe the adapter's behaviour.",
)
def weights_integrity() -> CheckResult:
    outcomes: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # untampered copy loads and verifies
        clean = root / "clean"
        clean.mkdir()
        metadata, weights = _copy_reference(clean)
        outcomes["clean_verified"] = DataDrivenModel.from_path(metadata).integrity["verified"]
        # (a) perturb one weight, keep the declared digest
        tampered = root / "tampered"
        tampered.mkdir()
        metadata_t, weights_t = _copy_reference(tampered)
        data = load_json(weights_t)
        data["layers"][0]["W"][0][0] += 1e-3
        dump(data, weights_t, overwrite=True)
        try:
            DataDrivenModel.from_path(metadata_t)
            outcomes["tampered_rejected"] = False
        except DataIntegrityError as error:
            outcomes["tampered_rejected"] = "sha256" in error.message
        # (b) wrong declared digest
        wrong = root / "wrong"
        wrong.mkdir()
        metadata_w, _ = _copy_reference(wrong)
        meta = load_json(metadata_w)
        meta["weights"]["sha256"] = "0" * 64
        dump(meta, metadata_w, overwrite=True)
        try:
            DataDrivenModel.from_path(metadata_w)
            outcomes["wrong_digest_rejected"] = False
        except DataIntegrityError:
            outcomes["wrong_digest_rejected"] = True
        # (c) no digest -> loads with explicit warning on the model and on every output
        nodigest = root / "nodigest"
        nodigest.mkdir()
        metadata_n, _ = _copy_reference(nodigest)
        meta = load_json(metadata_n)
        del meta["weights"]["sha256"]
        dump(meta, metadata_n, overwrite=True)
        model = DataDrivenModel.from_path(metadata_n)
        output = model.predict({"frequency": 1e5, "flux_density_peak": 0.1})
        outcomes["missing_digest_warns"] = (any(w.code == "weights_unverified" for w in model.load_warnings)
                                            and any(w.code == "weights_unverified" for w in output.warnings)
                                            and model.integrity["verified"] is False)
        # (d) executable format without digest is refused before anything is loaded
        exe = root / "exe"
        exe.mkdir()
        (exe / "model.pt").write_bytes(b"not a real archive")
        package = load_reference_package()
        bundle = package.weights.model_copy(update={"format": "torchscript", "path": "model.pt", "sha256": None})
        exe_package = package.model_copy(update={"weights": bundle}).with_base_dir(exe)
        try:
            DataDrivenModel(exe_package)
            outcomes["torchscript_without_digest_refused"] = False
        except DataIntegrityError as error:
            outcomes["torchscript_without_digest_refused"] = "sha256" in error.message
    passed = all(bool(v) for v in outcomes.values())
    return CheckResult(passed=passed, metrics=outcomes)


@check(
    id="datadriven.lstm-numpy-matches-independent-step-loop",
    capability=CAP_LSTM, level="L2", kind="independent_numerical",
    description="The vectorised numpy LSTM (PyTorch gate layout i,f,g,o; W_ih, W_hh, b_ih, b_hh) agrees with an "
                "independent scalar-loop implementation for sequence-to-one and sequence-to-sequence heads.",
    method="Two stacked LSTM layers (H=4) with a tanh/identity dense head and random N(0, 0.5^2) weights (seed 7) are "
           "evaluated on random inputs (N=3, L=6) by both implementations; compare max |difference|.",
)
def lstm_numpy_matches_independent_step_loop() -> CheckResult:
    rng = np.random.default_rng(11)
    metrics: dict[str, Any] = {}
    worst = 0.0
    for mode in ("last", "sequence"):
        spec = lstm_check_spec(mode)
        params = random_parameters(spec, seed=7)
        x = rng.normal(size=(3, 6, 2))
        vectorised = build_network(spec, params)(x)
        naive = naive_lstm_forward(spec, params, x)
        diff = float(np.max(np.abs(vectorised - naive)))
        metrics[f"max_abs_diff_{mode}"] = diff
        metrics[f"output_shape_{mode}"] = list(vectorised.shape)
        worst = max(worst, diff)
    metrics["parameter_count"] = lstm_check_spec("last").parameter_count()
    return CheckResult(passed=worst < 1e-12, metrics=metrics)


@check(
    id="datadriven.neural-ode-linear-case-closed-form",
    capability=CAP_NEURAL_ODE, level="L2", kind="analytical_reference",
    description="A Neural ODE whose identity-activation MLP vector field equals dy/dt = -a*y + b*u(t) reproduces the "
                "closed-form solution for a constant-plus-sinusoidal input to better than 1e-6.",
    method="Hand-constructed weights (a=1000 1/s, b=2.5), y0=0.3, u(t)=0.4+0.2 sin(2*pi*1000 t), DOP853 with "
           "rtol=1e-10/atol=1e-12 on [0, 5 ms]; compare with the analytic response at 101 points; also verify the "
           "vector field exactly and the sampled-input (linear interpolation) path for a constant input.",
)
def neural_ode_linear_case_closed_form() -> CheckResult:
    a, b, y0, u0, u1, omega = 1000.0, 2.5, 0.3, 0.4, 0.2, 2 * math.pi * 1000.0
    with tempfile.TemporaryDirectory() as tmp:
        package = build_linear_neural_ode_package(tmp, a=a, b=b)
        model = NeuralODEModel(package, rtol=1e-10, atol=1e-12)
        # exact vector field
        probe_y = np.array([[0.3], [-1.2], [2.0]])
        probe_u = np.array([[0.4], [0.0], [-0.7]])
        field_err = float(np.max(np.abs(model.vector_field(probe_y, probe_u) - (-a * probe_y + b * probe_u))))
        t = np.linspace(0.0, 5e-3, 101)
        out = model.simulate(y0=[y0], t_eval=t, u=lambda tt: [u0 + u1 * math.sin(omega * tt)])
        numeric = np.asarray(out.diagnostics["trajectory"]["y"])
        exact = linear_forced_response(t, y0=y0, a=a, b=b, u0=u0, u1=u1, omega=omega)
        err_sinusoid = float(np.max(np.abs(numeric - exact)))
        # sampled constant input through the interpolation path
        t_u = np.linspace(-1e-4, 6e-3, 25)
        out_c = model.simulate(y0=[y0], t_eval=t, u=(t_u, np.full((t_u.size, 1), u0)))
        err_constant = float(np.max(np.abs(np.asarray(out_c.diagnostics["trajectory"]["y"])
                                           - linear_forced_response(t, y0=y0, a=a, b=b, u0=u0))))
        final_ok = bool(abs(out.values["y"].value - float(exact[-1])) < 1e-6)
    passed = bool(field_err < 1e-12 and err_sinusoid < 1e-6 and err_constant < 1e-6 and final_ok)
    return CheckResult(passed=passed, metrics={
        "max_abs_error_sinusoidal_input": err_sinusoid, "max_abs_error_constant_input": err_constant,
        "vector_field_max_abs_error": field_err, "nfev": out.diagnostics["nfev"], "method": out.diagnostics["method"],
        "integrator": out.diagnostics["integrator"],
        "dynamic_ode_module_available": out.diagnostics["dynamic_ode_module_available"]})


@check(
    id="datadriven.out-of-domain-rejected",
    capability=CAP_ADAPTER, level="L1", kind="invalid_input_rejection",
    description="Inputs outside the declared validity domain are rejected (policy 'reject') or flagged (policy 'warn'); "
                "impossible or malformed inputs raise InvalidInputError; unit-bearing inputs are converted, not guessed; "
                "explicit conditions cannot override conditions derived from the inputs; validity ranges must be SI.",
    method="Evaluate the reference ANN at 2 MHz, 0.5 T, a batch with one bad sample, negative/NaN/missing/mis-dimensioned "
           "inputs, an empty batch, an explicit 'frequency' condition contradicting the 2 MHz input, a mis-sized "
           "condition array and a 100 kHz Quantity; validate a package copy whose frequency range is declared in kHz; "
           "observe the raised error classes and the warn-policy output.",
    range=REFERENCE_RANGE,
)
def out_of_domain_rejected() -> CheckResult:
    model = reference_steinmetz_ann()
    outcomes: dict[str, Any] = {}

    def expect(name: str, exc: type[Exception], **kwargs: Any) -> None:
        try:
            model.predict(**kwargs)
            outcomes[name] = False
        except exc:
            outcomes[name] = True
        except Exception as other:  # noqa: BLE001 - a wrong error class is a failure
            outcomes[name] = f"wrong error {type(other).__name__}"

    expect("frequency_2MHz", OutOfDomainError, inputs={"frequency": 2e6, "flux_density_peak": 0.1})
    expect("flux_0p5T", OutOfDomainError, inputs={"frequency": 1e5, "flux_density_peak": 0.5})
    expect("batch_with_one_bad_sample", OutOfDomainError, inputs={"frequency": [1e5, 1e5, 9e5], "flux_density_peak": 0.1})
    expect("negative_frequency", InvalidInputError, inputs={"frequency": -1e5, "flux_density_peak": 0.1})
    expect("nan_input", InvalidInputError, inputs={"frequency": float("nan"), "flux_density_peak": 0.1})
    expect("missing_feature", InvalidInputError, inputs={"frequency": 1e5})
    expect("unexpected_feature", InvalidInputError, inputs={"frequency": 1e5, "flux_density_peak": 0.1, "temperature": 300.0})
    expect("wrong_dimension_quantity", InvalidInputError,
           inputs={"frequency": Quantity(value=1e5, unit="Hz"), "flux_density_peak": Quantity(value=0.1, unit="A/m")})
    expect("boolean_input", InvalidInputError, inputs={"frequency": True, "flux_density_peak": 0.1})
    expect("unsupported_material", OutOfDomainError, inputs={"frequency": 1e5, "flux_density_peak": 0.1}, material="N87")
    expect("dc_bias_unsupported", OutOfDomainError, inputs={"frequency": 1e5, "flux_density_peak": 0.1}, dc_bias=True)
    expect("empty_batch", InvalidInputError, inputs={"frequency": np.array([]), "flux_density_peak": np.array([])})
    # an explicit condition must not override the condition derived from the actual input (2 MHz stays 2 MHz)
    expect("explicit_condition_overriding_input", InvalidInputError,
           inputs={"frequency": 2e6, "flux_density_peak": 0.1}, conditions={"frequency": 1e5})
    expect("condition_array_length_mismatch", InvalidInputError,
           inputs={"frequency": 1e5, "flux_density_peak": 0.1}, conditions={"temperature": [300.0, 300.0, 300.0]})
    # a validity range declared in a non-SI unit is refused when the package is validated
    non_si = load_reference_package().to_json_dict()
    non_si["domain"]["frequency"] = {"min": 20.0, "max": 500.0, "unit": "kHz"}
    try:
        ModelPackage.model_validate(non_si)
        outcomes["non_si_domain_unit_rejected"] = False
    except ValueError:  # pydantic ValidationError
        outcomes["non_si_domain_unit_rejected"] = True
    warned = model.predict({"frequency": 2e6, "flux_density_peak": 0.1}, policy="warn")
    outcomes["warn_policy_flags"] = (warned.domain.inside is False and any(w.code == "out_of_domain" for w in warned.warnings)
                                     and "loss_density" in warned.values)
    in_hz = model.predict({"frequency": 1e5, "flux_density_peak": 0.1}).values["loss_density"].value
    in_khz = model.predict({"frequency": Quantity(value=100.0, unit="kHz"),
                            "flux_density_peak": Quantity(value=100.0, unit="mT")}).values["loss_density"].value
    outcomes["quantity_units_converted"] = in_hz == in_khz
    outcomes["inside_domain_reported"] = model.predict({"frequency": 1e5, "flux_density_peak": 0.1}).domain.inside is True
    return CheckResult(passed=all(v is True for v in outcomes.values()), metrics=outcomes)


@check(
    id="datadriven.missing-optional-backend-is-explicit",
    capability=CAP_ADAPTER, level="L1", kind="software",
    description="Optional backends never fall back silently: without torch the torch backend raises "
                "UnsupportedOperationError naming aipe-magnetics[torch]; with torch, a state-dict export round-trips "
                "through the torch and numpy backends with identical results. Unsupported formats are explicit.",
    method="Probe torch availability; absent: request torch-state-dict/torchscript backends and load a package, plus the "
           "'equinox' format and the numpy transformer path. Present: build an MLP and an LSTM module from a descriptor, "
           "save the state dict, load it through TorchBackend and through NumpyBackend (npz) and compare outputs.",
)
def missing_optional_backend_is_explicit() -> CheckResult:
    outcomes: dict[str, Any] = {"torch_available": torch_available()}
    package = load_reference_package()
    # explicit errors that do not depend on torch
    try:
        get_backend("equinox", "lstm")
        outcomes["equinox_explicit"] = False
    except UnsupportedOperationError:
        outcomes["equinox_explicit"] = True
    try:
        get_backend("aipe-json", "transformer")
        outcomes["transformer_numpy_explicit"] = False
    except UnsupportedOperationError as error:
        outcomes["transformer_numpy_explicit"] = "torchscript" in error.message
    if not outcomes["torch_available"]:
        outcomes["path"] = "torch absent"
        for fmt in ("torch-state-dict", "torchscript"):
            backend = get_backend(fmt, "ann")
            bundle = package.weights.model_copy(update={"format": fmt})
            try:
                backend.load(package.model_copy(update={"weights": bundle}))
                outcomes[f"{fmt}_explicit"] = False
            except UnsupportedOperationError as error:
                outcomes[f"{fmt}_explicit"] = TORCH_INSTALL_HINT in error.message
        passed = all(bool(v) for k, v in outcomes.items() if k not in ("torch_available", "path"))
        return CheckResult(passed=passed, metrics=outcomes, notes=["torch is not installed in this environment; the "
                                                                   "equivalence path was not executed here"])
    # torch present: state-dict export / load equivalence (ann and lstm)
    from aipe_magnetics.models.datadriven.torch_backend import build_module, import_torch, save_state_dict, state_dict_to_numpy

    torch = import_torch()
    outcomes["path"] = "torch present"
    outcomes["torch_version"] = str(torch.__version__)
    rng = np.random.default_rng(3)
    worst = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        specs = {
            "ann": ArchitectureSpec(family="ann", name="ann-roundtrip",
                                    hyperparameters={"hidden_sizes": [6, 5], "activation": "gelu", "output_activation": "identity"},
                                    input_features=["a", "b", "c"], output_features=["p", "q"]),
            "lstm": lstm_check_spec("last"),
        }
        for label, spec in specs.items():
            torch.manual_seed(0)
            module = build_module(spec, torch).double()
            state_path = save_state_dict(module, root / f"{label}.pt", torch)
            params = state_dict_to_numpy(module.state_dict())
            np.savez(root / f"{label}.npz", **params)
            common = dict(identity=_identity(f"roundtrip-{label}"), architecture=spec, sources=[_SYNTHETIC_SOURCE],
                          synthetic=True, inference=InferenceConfig(dtype="float64"))
            torch_pkg = ModelPackage(weights=WeightsBundle(format="torch-state-dict", path=state_path.name,
                                                           sha256=sha256_file(state_path), source=_SYNTHETIC_SOURCE),
                                     **common).with_base_dir(root)
            numpy_pkg = ModelPackage(weights=WeightsBundle(format="npz", path=f"{label}.npz",
                                                           sha256=sha256_file(root / f"{label}.npz"), source=_SYNTHETIC_SOURCE),
                                     **common).with_base_dir(root)
            if label == "ann":
                inputs = {name: rng.normal(size=4) for name in spec.input_features}
            else:
                inputs = {"flux_density": rng.normal(size=(4, 6)), "frequency": rng.normal(size=4)}
            out_torch = DataDrivenModel(torch_pkg).predict(inputs)
            out_numpy = DataDrivenModel(numpy_pkg).predict(inputs)
            for name in spec.output_features:
                diff = float(np.max(np.abs(np.asarray(out_torch.diagnostics["outputs"][name])
                                           - np.asarray(out_numpy.diagnostics["outputs"][name]))))
                outcomes[f"max_abs_diff_{label}_{name}"] = diff
                worst = max(worst, diff)
            outcomes[f"torch_backend_used_{label}"] = out_torch.diagnostics["backend"] == "torch"
    outcomes["max_abs_diff"] = worst
    passed = worst < 1e-10 and all(bool(v) for k, v in outcomes.items() if k.startswith(("torch_backend_used", "equinox", "transformer")))
    return CheckResult(passed=passed, metrics=outcomes)


@check(
    id="datadriven.magnethub-adapter-mapping",
    capability=CAP_MAGNETHUB, level="L1", kind="software",
    description="The MagNet-Hub adapter maps AIPE inputs to the upstream call convention (B in T with the runtime "
                "sequence length, f in Hz, T converted K -> degC) and validates upstream outputs; without the package "
                "it fails explicitly with an installation hint. Nothing is downloaded.",
    method="Exercise prepare_inputs/collect_outputs/kelvin_to_celsius and the magnethub_package() descriptor; when "
           "magnethub is not importable, request the backend and confirm UnsupportedOperationError with the hint.",
)
def magnethub_adapter_mapping() -> CheckResult:
    outcomes: dict[str, Any] = {"magnethub_available": magnethub_available()}
    b = 0.1 * np.sin(np.linspace(0.0, 2 * np.pi, 1024, endpoint=False))
    b_seq, f, t_degc = prepare_inputs({"flux_density": b, "frequency": np.array([1e5]), "temperature": np.array([298.15])}, 1024)
    outcomes["kelvin_to_celsius"] = abs(float(t_degc[0]) - 25.0) < 1e-9 and abs(float(kelvin_to_celsius(np.array([373.15]))[0]) - 100.0) < 1e-9
    outcomes["shapes"] = b_seq.shape == (1, 1024) and f.shape == (1,)
    try:
        prepare_inputs({"flux_density": b[:512], "frequency": np.array([1e5]), "temperature": np.array([298.15])}, 1024)
        outcomes["length_mismatch_rejected"] = False
    except InvalidInputError:
        outcomes["length_mismatch_rejected"] = True
    try:
        kelvin_to_celsius(np.array([0.0]))
        outcomes["nonpositive_kelvin_rejected"] = False
    except InvalidInputError:
        outcomes["nonpositive_kelvin_rejected"] = True
    collected = collect_outputs(np.array([1234.5], dtype=np.float32), np.ones((1, 1024), dtype=np.float32), 1)
    outcomes["outputs_collected"] = collected["loss_density"].dtype == np.float64 and collected["field"].shape == (1, 1024)
    try:
        collect_outputs(np.array([1.0, 2.0]), np.ones((1, 1024)), 1)
        outcomes["bad_output_shape_rejected"] = False
    except DataIntegrityError:
        outcomes["bad_output_shape_rejected"] = True
    package = magnethub_package("N87", "paderborn")
    outcomes["descriptor_valid"] = (package.architecture.is_external and package.weights.format == "external-package"
                                    and package.synthetic is False and package.architecture.unit_of("temperature") == "K")
    backend = get_backend("external-package", "ann")
    outcomes["backend_is_magnethub"] = backend.name == "magnethub"
    if not outcomes["magnethub_available"]:
        try:
            backend.load(package)
            outcomes["missing_package_explicit"] = False
        except UnsupportedOperationError as error:
            outcomes["missing_package_explicit"] = MAGNETHUB_INSTALL_HINT.split(" (")[0] in error.message
        notes = ["magnethub is not installed; the real inference path is covered only by the external pytest test"]
    else:
        notes = ["magnethub is importable; real inference is exercised by the external pytest test, not here"]
    passed = all(bool(v) for k, v in outcomes.items() if k != "magnethub_available")
    return CheckResult(passed=passed, metrics=outcomes, notes=notes)


CHECKS = (
    reference_ann_loads_and_is_deterministic,
    reference_ann_reproduces_steinmetz,
    weights_integrity,
    lstm_numpy_matches_independent_step_loop,
    neural_ode_linear_case_closed_form,
    out_of_domain_rejected,
    missing_optional_backend_is_explicit,
    magnethub_adapter_mapping,
)

__all__ = ["CHECKS", "build_linear_neural_ode_package", "lstm_check_spec", "naive_activation", "naive_lstm_forward",
           "random_parameters", *(fn.__name__ for fn in CHECKS)]
