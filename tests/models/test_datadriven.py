"""Tests for ``aipe_magnetics.models.datadriven``: descriptors, backends, adapter, reference workflow and checks.

Nothing here downloads anything. Optional backends (torch, magnethub) are exercised on *both* paths:
when absent the explicit error is asserted, when present the real round trip runs. The only
skipping test is the ``external`` MagNet-Hub inference test, which reports why it skipped.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.special import erf

from aipe_magnetics.common.errors import (
    ConvergenceError,
    DataIntegrityError,
    DataNotFoundError,
    InvalidInputError,
    OutOfDomainError,
    UnitError,
    UnsupportedOperationError,
)
from aipe_magnetics.common.jsonio import dump, load as load_json, sha256_file
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import ModelIdentity, SourceReference
from aipe_magnetics.models.contract import MagneticModel, ValidityDomain, ValidityRange
from aipe_magnetics.models.datadriven import (
    MODELS,
    REFERENCE_MODEL_ID,
    ArchitectureSpec,
    DataDrivenModel,
    InferenceConfig,
    ModelPackage,
    NeuralODEModel,
    NormalizationSpec,
    NumpyBackend,
    WeightsBundle,
    available_backends,
    coerce_inputs,
    convert_array,
    get_backend,
    linear_forced_response,
    load_reference_package,
    reference_package_path,
    reference_steinmetz_ann,
    register_backend,
    unregister_backend,
    write_aipe_json,
)
from aipe_magnetics.models.datadriven import numpy_backend as nb
from aipe_magnetics.models.datadriven.architectures import stack_inputs, unstack_outputs
from aipe_magnetics.models.datadriven.magnethub_backend import (
    MAGNETHUB_INSTALL_HINT,
    MagnetHubBackend,
    collect_outputs,
    kelvin_to_celsius,
    magnethub_available,
    magnethub_package,
    prepare_inputs,
)
from aipe_magnetics.models.datadriven.torch_backend import TORCH_INSTALL_HINT, TorchBackend, torch_available
from aipe_magnetics.verification.checks import check_datadriven as checks
from aipe_magnetics.verification.suite import registered_checks, run_check

REPO = Path(__file__).resolve().parents[2]
SRC = SourceReference(id="src.test-synthetic", kind="assumption", reference="synthetic hand-made weights for tests",
                      note="synthetic")
IDENT = ModelIdentity(id="aipe.magnetics.model.datadriven.test", name="test", version="0.0.1", family="data_driven")


def _package(tmp: Path, spec: ArchitectureSpec, params: dict, *, fmt: str = "aipe-json", sha: bool = True,
             inference: InferenceConfig | None = None, domain: ValidityDomain | None = None) -> ModelPackage:
    if fmt == "aipe-json":
        path = write_aipe_json(tmp / "w.aipe.json", spec, params, overwrite=True)
    else:
        path = tmp / "w.npz"
        np.savez(path, **params)
    bundle = WeightsBundle(format=fmt, path=path.name, sha256=sha256_file(path) if sha else None, source=SRC)
    return ModelPackage(identity=IDENT, architecture=spec, weights=bundle, inference=inference or InferenceConfig(),
                        domain=domain or ValidityDomain(), sources=[SRC], synthetic=True).with_base_dir(tmp)


@pytest.fixture
def linear_ann(tmp_path: Path) -> DataDrivenModel:
    """ANN computing loss_density = 2*frequency + 3*flux_density_peak + 1 (identity activation)."""
    spec = ArchitectureSpec(family="ann", name="linear", hyperparameters={"hidden_sizes": []},
                            input_features=["frequency", "flux_density_peak"], output_features=["loss_density"],
                            input_units={"frequency": "Hz", "flux_density_peak": "T"}, output_units={"loss_density": "W/m^3"})
    params = {"layers.0.weight": np.array([[2.0, 3.0]]), "layers.0.bias": np.array([1.0])}
    domain = ValidityDomain(frequency=ValidityRange(min=1e3, max=1e6, unit="Hz"),
                            flux_density_peak=ValidityRange(min=0.0, max=0.5, unit="T"),
                            temperature=ValidityRange(min=250.0, max=400.0, unit="K"), excitation=["sinusoidal"],
                            materials=["material.test"])
    inference = InferenceConfig(domain_map={"frequency": "frequency", "flux_density_peak": "flux_density_peak"})
    return DataDrivenModel(_package(tmp_path, spec, params, inference=inference, domain=domain))


# --------------------------------------------------------------------------------------------------
# Import hygiene
# --------------------------------------------------------------------------------------------------
def test_importing_the_package_loads_no_optional_dependency():
    code = ("import sys; import aipe_magnetics.models.datadriven as m; "
            "print('torch' in sys.modules, 'magnethub' in sys.modules, 'jax' in sys.modules)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["False", "False", "False"]


# --------------------------------------------------------------------------------------------------
# Architecture descriptors
# --------------------------------------------------------------------------------------------------
def test_architecture_spec_validation_rejects_inconsistent_descriptors():
    with pytest.raises(ValidationError):
        ArchitectureSpec(family="cnn", name="x", input_features=["a"], output_features=["b"])
    with pytest.raises(ValidationError):  # ann takes no sequences
        ArchitectureSpec(family="ann", name="x", input_features=["a"], output_features=["b"], sequence_features=["a"])
    with pytest.raises(ValidationError):  # lstm needs a sequence input
        ArchitectureSpec(family="lstm", name="x", hyperparameters={"hidden_size": 2}, input_features=["a"], output_features=["b"])
    with pytest.raises(ValidationError):  # transformer heads must divide d_model
        ArchitectureSpec(family="transformer", name="x", hyperparameters={"d_model": 7, "nhead": 2, "num_layers": 1, "dim_feedforward": 8},
                         input_features=["a"], output_features=["b"], sequence_features=["a"])
    with pytest.raises(ValidationError):  # neural_ode outputs must be the state names
        ArchitectureSpec(family="neural_ode", name="x", hyperparameters={"state_size": 1, "input_size": 1},
                         input_features=["y", "u"], output_features=["z"])
    with pytest.raises(ValidationError):  # unknown unit
        ArchitectureSpec(family="ann", name="x", input_features=["a"], output_features=["b"], input_units={"a": "furlong"})
    with pytest.raises(ValidationError):  # duplicate features
        ArchitectureSpec(family="ann", name="x", input_features=["a", "a"], output_features=["b"])
    with pytest.raises(ValidationError):  # unknown hyper-parameter key (extra='forbid')
        ArchitectureSpec(family="ann", name="x", hyperparameters={"hidden_sizes": [4], "dropout": 0.1},
                         input_features=["a"], output_features=["b"])
    with pytest.raises(ValidationError):  # non-positive hidden size
        ArchitectureSpec(family="ann", name="x", hyperparameters={"hidden_sizes": [0]}, input_features=["a"], output_features=["b"])


def test_expected_parameter_shapes_and_counts():
    mlp = ArchitectureSpec(family="ann", name="m", hyperparameters={"hidden_sizes": [8, 8]},
                           input_features=["f", "b"], output_features=["p"])
    assert mlp.parameter_count() == 2 * 8 + 8 + 8 * 8 + 8 + 8 * 1 + 1 == 105
    assert mlp.expected_parameter_shapes()["layers.0.weight"] == (8, 2)
    lstm = checks.lstm_check_spec("last")
    shapes = lstm.expected_parameter_shapes()
    assert shapes["lstm.weight_ih_l0"] == (16, 2) and shapes["lstm.weight_hh_l1"] == (16, 4)
    assert shapes["head.0.weight"] == (3, 4) and shapes["head.1.weight"] == (2, 3)
    transformer = ArchitectureSpec(family="transformer", name="t",
                                   hyperparameters={"d_model": 8, "nhead": 2, "num_layers": 1, "dim_feedforward": 16},
                                   input_features=["b"], output_features=["p"], sequence_features=["b"])
    with pytest.raises(UnsupportedOperationError):
        transformer.expected_parameter_shapes()
    external = magnethub_package("N87").architecture
    assert external.is_external
    with pytest.raises(UnsupportedOperationError):
        external.expected_parameter_shapes()


def test_stack_and_unstack_feature_layouts():
    spec = checks.lstm_check_spec("sequence")
    x = {"flux_density": np.arange(12.0).reshape(2, 6), "frequency": np.array([1.0, 2.0])}
    tensor = stack_inputs(spec, x)
    assert tensor.shape == (2, 6, 2)
    assert np.all(tensor[1, :, 1] == 2.0) and np.array_equal(tensor[0, :, 0], np.arange(6.0))
    outputs = unstack_outputs(spec, np.zeros((2, 6, 2)))
    assert set(outputs) == {"field", "loss_density"} and outputs["field"].shape == (2, 6)
    with pytest.raises(InvalidInputError):
        stack_inputs(spec, {"flux_density": np.zeros((2, 6)), "frequency": np.zeros(3)})
    with pytest.raises(InvalidInputError):
        unstack_outputs(spec, np.zeros((2, 6, 3)))


# --------------------------------------------------------------------------------------------------
# Normalisation and units
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("spec", [
    NormalizationSpec(kind="none"),
    NormalizationSpec(kind="zscore", mean=2.0, std=0.5),
    NormalizationSpec(kind="minmax", min=-1.0, max=3.0),
    NormalizationSpec(kind="log10"),
    NormalizationSpec(kind="log10", mean=4.5, std=0.3),
])
def test_normalisation_round_trip(spec: NormalizationSpec):
    x = np.array([0.5, 1.0, 20.0, 3e5])
    np.testing.assert_allclose(spec.inverse(spec.forward(x)), x, rtol=1e-12)


def test_normalisation_rejects_invalid_statistics_and_non_positive_log_inputs():
    for bad in ({"kind": "zscore", "mean": 0.0}, {"kind": "zscore", "mean": 0.0, "std": 0.0},
                {"kind": "minmax", "min": 1.0, "max": 1.0}, {"kind": "none", "mean": 1.0},
                {"kind": "log10", "min": 0.0, "max": 1.0}, {"kind": "zscore", "mean": float("nan"), "std": 1.0}):
        with pytest.raises(ValidationError):
            NormalizationSpec(**bad)
    with pytest.raises(InvalidInputError):
        NormalizationSpec(kind="log10").forward(np.array([1.0, 0.0]), name="frequency")


def test_convert_array_is_elementwise_and_dimension_checked():
    np.testing.assert_allclose(convert_array(np.array([1.0, 2.5]), "kHz", "Hz"), [1000.0, 2500.0])
    np.testing.assert_allclose(convert_array(np.array([25.0, 100.0]), "degC", "K"), [298.15, 373.15])
    np.testing.assert_allclose(convert_array(np.array([298.15]), "K", "degC"), [25.0])
    with pytest.raises(UnitError):
        convert_array(np.array([1.0]), "Hz", "T")


# --------------------------------------------------------------------------------------------------
# Numpy backend
# --------------------------------------------------------------------------------------------------
def test_activation_functions_match_reference_formulas():
    x = np.linspace(-4, 4, 41)
    np.testing.assert_allclose(nb.activation("gelu")(x), 0.5 * x * (1 + erf(x / math.sqrt(2))), rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(nb.activation("sigmoid")(x), 1 / (1 + np.exp(-x)), rtol=1e-14)
    np.testing.assert_array_equal(nb.activation("relu")(x), np.where(x > 0, x, 0.0))
    np.testing.assert_array_equal(nb.activation("identity")(x), x)
    np.testing.assert_allclose(nb.activation("tanh")(x), np.tanh(x))
    with pytest.raises(InvalidInputError):
        nb.activation("swish")


def test_mlp_forward_matches_manual_computation_and_layer_validation():
    rng = np.random.default_rng(0)
    w0, b0, w1, b1 = rng.normal(size=(5, 3)), rng.normal(size=5), rng.normal(size=(2, 5)), rng.normal(size=2)
    layers = [nb.DenseLayer(w0, b0, "tanh"), nb.DenseLayer(w1, b1, "identity")]
    x = rng.normal(size=(7, 3))
    expected = np.tanh(x @ w0.T + b0) @ w1.T + b1
    np.testing.assert_allclose(nb.mlp_forward(layers, x), expected, rtol=1e-14)
    with pytest.raises(DataIntegrityError):
        nb.DenseLayer(w0, np.zeros(4), "tanh")
    with pytest.raises(DataIntegrityError):
        nb.DenseLayer(np.array([[np.nan]]), np.zeros(1), "tanh")
    with pytest.raises(InvalidInputError):
        nb.mlp_forward(layers, np.zeros((2, 4)))


def test_aipe_json_loader_cross_checks_architecture(tmp_path: Path):
    spec = ArchitectureSpec(family="ann", name="a", hyperparameters={"hidden_sizes": [4], "activation": "tanh"},
                            input_features=["x1", "x2"], output_features=["y"])
    params = checks.random_parameters(spec, seed=1)
    path = write_aipe_json(tmp_path / "a.json", spec, params)
    loaded, tags, header = nb.load_aipe_json(path)
    assert header["family"] == "ann" and tags == {"layers.0": "tanh", "layers.1": "identity"}
    assert isinstance(nb.build_network(spec, loaded, tags), nb.MLPNetwork)
    other = spec.model_copy(update={"hyperparameters": {"hidden_sizes": [5]}})
    with pytest.raises(DataIntegrityError):  # shape mismatch
        nb.build_network(other, loaded, tags)
    relu = spec.model_copy(update={"hyperparameters": {"hidden_sizes": [4], "activation": "relu"}})
    with pytest.raises(DataIntegrityError):  # activation tag disagrees with the descriptor
        nb.build_network(relu, loaded, tags)
    with pytest.raises(DataIntegrityError):  # missing / unexpected parameter names
        nb.build_network(spec, {k: v for k, v in loaded.items() if k != "layers.0.bias"}, {})
    data = load_json(path)
    data["format"] = "something-else"
    dump(data, tmp_path / "bad.json")
    with pytest.raises(DataIntegrityError):
        nb.load_aipe_json(tmp_path / "bad.json")
    (tmp_path / "dup.json").write_text('{"format": "aipe-json", "format": "aipe-json"}', encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        nb.load_aipe_json(tmp_path / "dup.json")
    with pytest.raises(UnsupportedOperationError):
        nb.load_parameters(path, "torchscript")


def test_lstm_single_step_hand_computed():
    # H = 1, in = 1: W_ih rows (i, f, g, o) = [1, 2, 3, 4], W_hh = 0, biases 0; x = 0.5, h = c = 0
    layer = nb.LSTMLayer(np.array([[1.0], [2.0], [3.0], [4.0]]), np.zeros((4, 1)), np.zeros(4), np.zeros(4))
    h, c = nb.lstm_step(layer, np.array([[0.5]]), np.zeros((1, 1)), np.zeros((1, 1)))
    i, f, g, o = (1 / (1 + math.exp(-0.5)), 1 / (1 + math.exp(-1.0)), math.tanh(1.5), 1 / (1 + math.exp(-2.0)))
    assert c[0, 0] == pytest.approx(f * 0.0 + i * g, rel=1e-14)
    assert h[0, 0] == pytest.approx(o * math.tanh(i * g), rel=1e-14)
    outputs, h_final, c_final = nb.lstm_forward([layer], np.array([[[0.5]]]))
    assert outputs.shape == (1, 1, 1) and h_final[0][0, 0] == pytest.approx(h[0, 0]) and c_final[0][0, 0] == pytest.approx(c[0, 0])
    with pytest.raises(DataIntegrityError):
        nb.LSTMLayer(np.zeros((6, 1)), np.zeros((6, 1)), np.zeros(6), np.zeros(6))  # 4H must divide


@pytest.mark.parametrize("mode", ["last", "sequence"])
def test_lstm_vectorised_matches_naive_loop(mode: str):
    spec = checks.lstm_check_spec(mode)
    params = checks.random_parameters(spec, seed=7)
    x = np.random.default_rng(5).normal(size=(2, 5, 2))
    vectorised = nb.build_network(spec, params)(x)
    naive = checks.naive_lstm_forward(spec, params, x)
    assert vectorised.shape == naive.shape
    np.testing.assert_allclose(vectorised, naive, rtol=0, atol=1e-13)


def test_npz_lstm_package_round_trip(tmp_path: Path):
    spec = checks.lstm_check_spec("last")
    params = checks.random_parameters(spec, seed=9)
    model = DataDrivenModel(_package(tmp_path, spec, params, fmt="npz"))
    b = np.random.default_rng(1).normal(size=(3, 5))
    out = model.predict({"flux_density": b, "frequency": [1e5, 2e5, 3e5]})
    direct = nb.build_network(spec, params)(stack_inputs(spec, {"flux_density": b, "frequency": np.array([1e5, 2e5, 3e5])}))
    np.testing.assert_allclose(out.diagnostics["outputs"]["field"], direct[:, 0], rtol=1e-14)
    np.testing.assert_allclose(out.diagnostics["outputs"]["loss_density"], direct[:, 1], rtol=1e-14)
    assert model.integrity["verified"] and out.diagnostics["sequence_length"] == 5


# --------------------------------------------------------------------------------------------------
# Adapter: DataDrivenModel
# --------------------------------------------------------------------------------------------------
def test_predict_scalar_batch_broadcast_and_quantities(linear_ann: DataDrivenModel):
    assert isinstance(linear_ann, MagneticModel)
    out = linear_ann.predict({"frequency": 1e5, "flux_density_peak": 0.1})
    assert out.values["loss_density"].unit == "W/m^3"
    assert out.values["loss_density"].value == pytest.approx(2e5 + 0.3 + 1.0)
    assert out.domain.inside and out.diagnostics["batch_size"] == 1 and out.diagnostics["backend"] == "numpy"
    batch = linear_ann.predict({"frequency": [1e5, 2e5], "flux_density_peak": 0.1})
    assert batch.values == {} and batch.diagnostics["outputs"]["loss_density"] == pytest.approx([2e5 + 1.3, 4e5 + 1.3])
    same = linear_ann.evaluate({"frequency": 1e5, "flux_density_peak": 0.1})
    assert same.values["loss_density"] == out.values["loss_density"]
    khz = linear_ann.predict({"frequency": Quantity(value=100.0, unit="kHz"), "flux_density_peak": {"value": 100.0, "unit": "mT"}})
    assert khz.values["loss_density"].value == pytest.approx(out.values["loss_density"].value)
    assert any(l.code == "synthetic_parameters" for l in out.limitations)
    assert any(a.id == "input_units" for a in out.assumptions)
    assert "package" in linear_ann.describe()


def test_predict_rejects_invalid_inputs(linear_ann: DataDrivenModel):
    good = {"frequency": 1e5, "flux_density_peak": 0.1}
    for bad in ({"frequency": float("nan"), "flux_density_peak": 0.1}, {"frequency": 1e5},
                {**good, "temperature": 300.0}, {"frequency": True, "flux_density_peak": 0.1},
                {"frequency": "1e5", "flux_density_peak": 0.1}, {"frequency": [[1e5]], "flux_density_peak": 0.1},
                {"frequency": [1e5, 2e5], "flux_density_peak": [0.1, 0.1, 0.1]},
                {"frequency": -1e5, "flux_density_peak": 0.1},
                {"frequency": Quantity(value=1e5, unit="Hz"), "flux_density_peak": Quantity(value=0.1, unit="A/m")}):
        with pytest.raises(InvalidInputError):
            linear_ann.predict(bad)
    with pytest.raises(InvalidInputError):
        linear_ann.predict(good, policy="clip")
    with pytest.raises(InvalidInputError):
        linear_ann.predict([1e5, 0.1])  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError):
        linear_ann.evaluate()


def test_domain_policy_reject_warn_and_explicit_conditions(linear_ann: DataDrivenModel):
    with pytest.raises(OutOfDomainError):
        linear_ann.predict({"frequency": 5e6, "flux_density_peak": 0.1})
    with pytest.raises(OutOfDomainError):
        linear_ann.predict({"frequency": [1e5, 5e6], "flux_density_peak": 0.1})
    warned = linear_ann.predict({"frequency": 5e6, "flux_density_peak": 0.1}, policy="warn")
    assert warned.domain.inside is False and any(w.code == "out_of_domain" for w in warned.warnings)
    assert "frequency" in warned.domain.violations[0]
    with pytest.raises(OutOfDomainError):
        linear_ann.predict({"frequency": 1e5, "flux_density_peak": 0.1}, conditions={"temperature": 500.0})
    ok = linear_ann.predict({"frequency": 1e5, "flux_density_peak": 0.1}, conditions={"temperature": Quantity(value=25.0, unit="degC")})
    assert ok.domain.inside and ok.domain.checked["temperature"] == "[250, 400] K"
    with pytest.raises(InvalidInputError):
        linear_ann.predict({"frequency": 1e5, "flux_density_peak": 0.1}, conditions={"temperature": 0.0})
    with pytest.raises(InvalidInputError):
        linear_ann.predict({"frequency": 1e5, "flux_density_peak": 0.1}, conditions={"pressure": 1.0})
    with pytest.raises(OutOfDomainError):
        linear_ann.predict({"frequency": 1e5, "flux_density_peak": 0.1}, material="material.other")
    with pytest.raises(OutOfDomainError):
        linear_ann.predict({"frequency": 1e5, "flux_density_peak": 0.1}, excitation="square")
    with pytest.raises(OutOfDomainError):
        linear_ann.predict({"frequency": 1e5, "flux_density_peak": 0.1}, dc_bias=True)
    assert linear_ann.predict({"frequency": 1e5, "flux_density_peak": 0.1}, material="material.test",
                              excitation="sinusoidal").domain.inside


def test_weights_integrity_policy(tmp_path: Path):
    spec = ArchitectureSpec(family="ann", name="linear", hyperparameters={"hidden_sizes": []},
                            input_features=["a"], output_features=["b"])
    params = {"layers.0.weight": np.array([[2.0]]), "layers.0.bias": np.array([0.0])}
    package = _package(tmp_path, spec, params)
    assert DataDrivenModel(package).integrity["verified"] is True
    # tamper after declaring the digest
    data = load_json(package.weights_path())
    data["layers"][0]["W"][0][0] = 2.5
    dump(data, package.weights_path(), overwrite=True)
    with pytest.raises(DataIntegrityError):
        DataDrivenModel(package)
    # missing digest -> warning on model and outputs
    unverified = _package(tmp_path, spec, params, sha=False)
    model = DataDrivenModel(unverified)
    assert [w.code for w in model.load_warnings] == ["weights_unverified"] and model.integrity["verified"] is False
    assert any(w.code == "weights_unverified" for w in model.predict({"a": 1.0}).warnings)
    # wrong digest
    wrong = unverified.model_copy(update={"weights": unverified.weights.model_copy(update={"sha256": "0" * 64})}).with_base_dir(tmp_path)
    with pytest.raises(DataIntegrityError):
        DataDrivenModel(wrong)
    # missing file
    missing = unverified.model_copy(update={"weights": unverified.weights.model_copy(update={"path": "nope.json"})}).with_base_dir(tmp_path)
    with pytest.raises(DataNotFoundError):
        DataDrivenModel(missing)
    # relative path without base directory
    with pytest.raises(InvalidInputError):
        ModelPackage.model_validate(unverified.model_dump(mode="json")).weights_path()
    # executable format without digest is refused before torch is even needed
    (tmp_path / "m.pt").write_bytes(b"xx")
    exe = unverified.model_copy(update={"weights": unverified.weights.model_copy(update={"format": "torchscript", "path": "m.pt"})}).with_base_dir(tmp_path)
    with pytest.raises(DataIntegrityError):
        DataDrivenModel(exe)
    with pytest.raises(ValidationError):  # malformed digest string
        WeightsBundle(format="npz", path="w.npz", sha256="XYZ", source=SRC)


def test_model_package_validation_and_round_trip(tmp_path: Path):
    spec = ArchitectureSpec(family="ann", name="linear", hyperparameters={"hidden_sizes": []},
                            input_features=["a"], output_features=["b"], input_units={"a": "Hz"})
    params = {"layers.0.weight": np.array([[2.0]]), "layers.0.bias": np.array([0.0])}
    package = _package(tmp_path, spec, params)
    package.save(tmp_path / "metadata.json")
    reloaded = ModelPackage.load(tmp_path / "metadata.json")
    assert reloaded.model_dump(mode="json") == package.model_dump(mode="json")
    assert DataDrivenModel.from_path(tmp_path / "metadata.json").predict({"a": 3.0}).values["b"].value == 6.0
    base = package.model_dump(mode="json")
    with pytest.raises(ValidationError):  # wrong family
        ModelPackage.model_validate({**base, "identity": {**base["identity"], "family": "empirical"}})
    with pytest.raises(ValidationError):  # synthetic but not labelled
        ModelPackage.model_validate({**base, "sources": [{"id": "s", "kind": "internal", "reference": "weights"}]})
    with pytest.raises(ValidationError):  # claims real data without a real source
        ModelPackage.model_validate({**base, "synthetic": False})
    with pytest.raises(ValidationError):  # unknown normalisation key
        ModelPackage.model_validate({**base, "inference": {"input_normalization": {"zz": {"kind": "none"}}}})
    with pytest.raises(ValidationError):  # domain_map unit mismatch (a is in Hz, temperature needs K)
        ModelPackage.model_validate({**base, "inference": {"domain_map": {"temperature": "a"}}})
    with pytest.raises(ValidationError):  # external-package weights must not carry a path
        WeightsBundle(format="external-package", path="x", uri="pkg:pypi/x", source=SRC)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        ModelPackage.load(tmp_path / "broken.json")


def test_backend_registry_and_explicit_unsupported_paths(linear_ann: DataDrivenModel):
    table = available_backends()
    assert set(table) >= {"numpy", "torch", "magnethub"}
    assert isinstance(get_backend("aipe-json", "ann"), NumpyBackend)
    assert isinstance(get_backend("torch-state-dict", "lstm"), TorchBackend)
    assert isinstance(get_backend("external-package", "ann"), MagnetHubBackend)
    with pytest.raises(InvalidInputError):
        get_backend("onnx", "ann")
    with pytest.raises(UnsupportedOperationError):
        get_backend("equinox", "lstm")
    with pytest.raises(UnsupportedOperationError):
        get_backend("aipe-json", "transformer")
    with pytest.raises(UnsupportedOperationError):
        get_backend("torch-state-dict", "transformer")
    with pytest.raises(InvalidInputError):
        get_backend("npz", "neural_ode").run(linear_ann.package, {})  # missing features are rejected explicitly
    with pytest.raises(InvalidInputError):
        NumpyBackend().forward(np.zeros((1, 2)))  # not loaded -> explicit error, never a silent zero output
    with pytest.raises(UnsupportedOperationError):
        linear_ann.package.model_copy(update={"weights": linear_ann.package.weights.model_copy(
            update={"format": "external-package", "path": None, "uri": "pkg:x"})}).weights_path()


def test_coerce_inputs_shapes():
    spec = checks.lstm_check_spec("last")
    arrays = coerce_inputs(spec, {"flux_density": np.zeros(6), "frequency": [1.0, 2.0]})
    assert arrays["flux_density"].shape == (2, 6) and arrays["frequency"].shape == (2,)
    fixed = spec.model_copy(update={"sequence_length": 8})
    with pytest.raises(InvalidInputError):
        coerce_inputs(fixed, {"flux_density": np.zeros(6), "frequency": 1.0})
    with pytest.raises(InvalidInputError):
        coerce_inputs(spec, {"flux_density": np.zeros((2, 6)), "frequency": [1.0, 2.0, 3.0]})
    with pytest.raises(InvalidInputError):
        coerce_inputs(spec, {"flux_density": np.zeros((2, 2, 6)), "frequency": 1.0})


# --------------------------------------------------------------------------------------------------
# Reference ANN package (SYNTHETIC)
# --------------------------------------------------------------------------------------------------
def test_reference_package_is_self_consistent_and_labelled_synthetic():
    package = load_reference_package()
    assert reference_package_path().name == "metadata.json" and package.synthetic is True
    assert package.identity.id == REFERENCE_MODEL_ID and package.identity.family == "data_driven"
    assert sha256_file(package.weights_path()) == package.weights.sha256
    assert package.training["basis"] == "synthetic"
    assert package.training["metrics"]["max_relative_error_training_grid"] < 0.02
    assert any("synthetic" in (s.note or "").lower() or "synthetic" in s.reference.lower() for s in package.sources)
    assert any(s.kind == "publication" for s in package.sources)
    assert package.architecture.parameter_count() == 105
    assert MODELS and MODELS[0].model_id == REFERENCE_MODEL_ID  # type: ignore[attr-defined]
    model = MODELS[0]()
    assert isinstance(model, DataDrivenModel) and model.integrity["verified"] is True and not model.load_warnings


def test_reference_ann_reproduces_the_synthetic_surface_and_honours_domain():
    package = load_reference_package()
    model = reference_steinmetz_ann()
    k, alpha, beta = (package.training["synthetic_parameters"][key] for key in ("k", "alpha", "beta"))
    rng = np.random.default_rng(123)
    f = 10 ** rng.uniform(np.log10(20e3), np.log10(500e3), 100)
    b = 10 ** rng.uniform(np.log10(0.02), np.log10(0.3), 100)
    predicted = np.asarray(model.predict({"frequency": f, "flux_density_peak": b}).diagnostics["outputs"]["loss_density"])
    assert np.max(np.abs(predicted / (k * f ** alpha * b ** beta) - 1)) < 0.03
    single = model.predict({"frequency": 1e5, "flux_density_peak": 0.1})
    assert single.values["loss_density"].value == pytest.approx(k * 1e5 ** alpha * 0.1 ** beta, rel=0.01)
    assert single.values["loss_density"].unit == "W/m^3"
    with pytest.raises(OutOfDomainError):
        model.predict({"frequency": 1e6, "flux_density_peak": 0.1})
    with pytest.raises(OutOfDomainError):  # 0 T is a legal peak but below the 20 mT training range
        model.predict({"frequency": 1e5, "flux_density_peak": 0.0})
    with pytest.raises(InvalidInputError):  # a negative peak magnitude is impossible, not merely out of domain
        model.predict({"frequency": 1e5, "flux_density_peak": -0.1})
    with pytest.raises(InvalidInputError):  # under policy='warn' the log10 feature still refuses a zero value
        model.predict({"frequency": 1e5, "flux_density_peak": 0.0}, policy="warn")


# --------------------------------------------------------------------------------------------------
# Neural ODE
# --------------------------------------------------------------------------------------------------
def test_neural_ode_linear_case_matches_closed_form(tmp_path: Path):
    a, b, y0, u0 = 500.0, 1.5, -0.2, 0.8
    package = checks.build_linear_neural_ode_package(tmp_path, a=a, b=b)
    model = NeuralODEModel(package, rtol=1e-10, atol=1e-12)
    assert isinstance(model, MagneticModel) and model.state_features == ["y"] and model.input_features == ["u"]
    np.testing.assert_allclose(model.vector_field(np.array([0.3]), np.array([0.4])), [-a * 0.3 + b * 0.4], rtol=1e-14)
    t = np.linspace(0, 8e-3, 81)
    out = model.evaluate(y0=[y0], t_eval=t, u=lambda tt: [u0])
    exact = linear_forced_response(t, y0=y0, a=a, b=b, u0=u0)
    np.testing.assert_allclose(out.diagnostics["trajectory"]["y"], exact, rtol=0, atol=1e-6)
    assert out.values["y"].value == pytest.approx(exact[-1], abs=1e-6) and out.values["y"].unit == "1"
    assert out.diagnostics["integrator"] == "scipy.integrate.solve_ivp" and out.diagnostics["method"] == "DOP853"
    assert isinstance(out.diagnostics["dynamic_ode_module_available"], bool)
    sampled = model.simulate(y0=[y0], t_eval=t, u=(np.array([0.0, 1e-2]), np.array([[u0], [u0]])))
    np.testing.assert_allclose(sampled.diagnostics["trajectory"]["y"], exact, rtol=0, atol=1e-6)


def test_neural_ode_rejects_invalid_inputs_and_reports_solver_failure(tmp_path: Path):
    package = checks.build_linear_neural_ode_package(tmp_path, a=100.0, b=1.0)
    model = NeuralODEModel(package)
    t = np.linspace(0, 1e-2, 11)
    with pytest.raises(InvalidInputError):
        model.simulate(y0=[0.0, 1.0], t_eval=t, u=lambda tt: [1.0])
    with pytest.raises(InvalidInputError):
        model.simulate(y0=[0.0], t_eval=t[::-1], u=lambda tt: [1.0])
    with pytest.raises(InvalidInputError):
        model.simulate(y0=[0.0], t_eval=t)  # u required
    with pytest.raises(InvalidInputError):
        model.simulate(y0=[0.0], t_eval=t, u=lambda tt: [1.0, 2.0])
    with pytest.raises(InvalidInputError):
        model.simulate(y0=[0.0], t_eval=t, u=(np.array([0.0, 5e-3]), np.array([[1.0], [1.0]])))  # does not cover interval
    with pytest.raises(InvalidInputError):
        model.simulate(y0=[float("nan")], t_eval=t, u=lambda tt: [1.0])
    with pytest.raises(InvalidInputError):
        model.simulate(y0=[0.0], t_eval=t, u=lambda tt: [1.0], rtol=-1.0)
    with pytest.raises(InvalidInputError):
        model.evaluate([0.0])  # positional arguments are not accepted
    with pytest.raises(UnsupportedOperationError):
        DataDrivenModel(package).predict({"y": 0.0, "u": 1.0})  # vector fields are integrated, not "predicted"

    class Failed:
        success = False
        message = "step size too small"

    failing = NeuralODEModel(package, integrator=lambda *args, **kwargs: Failed())
    with pytest.raises(ConvergenceError):
        failing.simulate(y0=[0.0], t_eval=t, u=lambda tt: [1.0])
    log_package = package.model_copy(update={"inference": InferenceConfig(
        input_normalization={"u": NormalizationSpec(kind="log10")})}).with_base_dir(tmp_path)
    with pytest.raises(InvalidInputError):
        NeuralODEModel(log_package)
    ann = load_reference_package()
    with pytest.raises(InvalidInputError):
        NeuralODEModel(ann)


# --------------------------------------------------------------------------------------------------
# Optional backends (both paths execute real code)
# --------------------------------------------------------------------------------------------------
def test_torch_backend_is_explicit_when_absent_or_equivalent_when_present(tmp_path: Path):
    package = load_reference_package()
    if not torch_available():
        for fmt in ("torch-state-dict", "torchscript"):
            with pytest.raises(UnsupportedOperationError) as info:
                get_backend(fmt, "ann").load(package.model_copy(update={"weights": package.weights.model_copy(update={"format": fmt})}))
            assert TORCH_INSTALL_HINT in str(info.value)
        return
    result = checks.missing_optional_backend_is_explicit()
    assert result.passed, result.metrics
    assert result.metrics["path"] == "torch present" and result.metrics["max_abs_diff"] < 1e-10


def test_magnethub_mapping_functions_without_the_package():
    b = 0.2 * np.sin(np.linspace(0, 2 * np.pi, 1024, endpoint=False))
    b_seq, f, t = prepare_inputs({"flux_density": b, "frequency": np.array([2e5]), "temperature": np.array([298.15])}, 1024)
    assert b_seq.shape == (1, 1024) and f[0] == 2e5 and t[0] == pytest.approx(25.0)
    np.testing.assert_allclose(kelvin_to_celsius(np.array([273.15, 373.15])), [0.0, 100.0])
    with pytest.raises(InvalidInputError):
        kelvin_to_celsius(np.array([-5.0]))
    with pytest.raises(InvalidInputError):
        prepare_inputs({"flux_density": b[:128], "frequency": np.array([2e5]), "temperature": np.array([298.15])}, 1024)
    with pytest.raises(InvalidInputError):
        prepare_inputs({"flux_density": b, "frequency": np.array([-2e5]), "temperature": np.array([298.15])}, 1024)
    out = collect_outputs(np.array([10.0]), np.zeros((1, 1024)), 1)
    assert out["loss_density"].shape == (1,) and out["field"].shape == (1, 1024)
    with pytest.raises(DataIntegrityError):
        collect_outputs(np.array([10.0]), None, 1)
    package = magnethub_package("N87", "paderborn")
    assert package.architecture.sequence_length == 1024 and package.weights.format == "external-package"
    assert package.domain.materials == ["N87"] and package.synthetic is False
    with pytest.raises(InvalidInputError):
        magnethub_package("N87", "oxford")


def test_magnethub_backend_is_explicit_when_absent_or_runs_when_present():
    package = magnethub_package("N87", "paderborn", sequence_length=None)
    backend = get_backend("external-package", "ann")
    assert isinstance(backend, MagnetHubBackend)
    if magnethub_available():  # e.g. Python >= 3.13 with mag-net-hub installed: run the real path instead of failing
        model = DataDrivenModel(package)
        length = model.backend.expected_sequence_length  # type: ignore[attr-defined]
        b = 0.1 * np.sin(np.linspace(0, 2 * np.pi, length, endpoint=False))
        out = model.predict({"flux_density": b, "frequency": 1e5, "temperature": 298.15})
        assert out.diagnostics["backend"] == "magnethub" and out.values["loss_density"].value > 0
        assert any(w.code == "weights_unverified" for w in out.warnings)
        return
    with pytest.raises(UnsupportedOperationError) as info:
        backend.load(package)
    assert "mag-net-hub" in str(info.value) and MAGNETHUB_INSTALL_HINT.split(" (")[0] in str(info.value)
    with pytest.raises(UnsupportedOperationError):
        DataDrivenModel(package)


@pytest.mark.external
def test_magnethub_real_inference_external():
    """Genuine MagNet-Hub inference (N87, Paderborn CNN). Skips with an explicit reason when the package is absent."""
    try:
        import magnethub  # noqa: F401
    except ImportError as error:
        pytest.skip(f"external dependency magnethub not installed ({error}); {MAGNETHUB_INSTALL_HINT}")
    model = DataDrivenModel(magnethub_package("N87", "paderborn", sequence_length=None))
    length = model.backend.expected_sequence_length  # type: ignore[attr-defined]
    b = 0.1 * np.sin(np.linspace(0, 2 * np.pi, length, endpoint=False))
    out = model.predict({"flux_density": b, "frequency": 1e5, "temperature": 298.15})
    assert out.values["loss_density"].unit == "W/m^3" and out.values["loss_density"].value > 0
    assert np.asarray(out.diagnostics["outputs"]["field"]).shape == (1, length)


# --------------------------------------------------------------------------------------------------
# Benchmark script
# --------------------------------------------------------------------------------------------------
def _load_benchmark_module():
    path = REPO / "benchmarks" / "fit_reference_mlp.py"
    spec = importlib.util.spec_from_file_location("fit_reference_mlp", path)
    assert spec is not None and spec.loader is not None, f"benchmark script missing at {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_gradient_matches_finite_differences():
    bench = _load_benchmark_module()
    hidden = [3]
    theta = bench.init_theta(hidden, seed=1)
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(9, 2)), rng.normal(size=(9, 1))
    loss, grad = bench.loss_and_grad(theta, x, y, hidden)
    numeric = np.zeros_like(theta)
    for i in range(theta.size):
        step = np.zeros_like(theta)
        step[i] = 1e-6
        numeric[i] = (bench.loss_and_grad(theta + step, x, y, hidden)[0] - bench.loss_and_grad(theta - step, x, y, hidden)[0]) / 2e-6
    np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-8)


def test_benchmark_script_produces_a_loadable_package(tmp_path: Path):
    bench = _load_benchmark_module()
    info = bench.run(tmp_path / "ref", max_iter=20000)
    assert info["metrics"]["max_relative_error_training_grid"] < 0.02
    model = DataDrivenModel.from_path(info["metadata"])
    assert model.integrity["verified"] and model.package.synthetic and model.package.weights.sha256 == info["sha256"]
    committed = load_reference_package()
    assert model.package.architecture == committed.architecture
    assert model.package.inference.input_normalization == committed.inference.input_normalization
    f, b = 1.2e5, 0.08
    refit = model.predict({"frequency": f, "flux_density_peak": b}).values["loss_density"].value
    bundled = reference_steinmetz_ann().predict({"frequency": f, "flux_density_peak": b}).values["loss_density"].value
    assert refit == pytest.approx(bundled, rel=5e-3)


# --------------------------------------------------------------------------------------------------
# Verification checks of this module
# --------------------------------------------------------------------------------------------------
EXPECTED_CHECKS = {
    "datadriven.reference-ann-loads-and-is-deterministic": ("L1", checks.CAP_REFERENCE),
    "datadriven.reference-ann-reproduces-steinmetz": ("L2", checks.CAP_REFERENCE),
    "datadriven.weights-integrity": ("L1", checks.CAP_ADAPTER),
    "datadriven.lstm-numpy-matches-independent-step-loop": ("L2", checks.CAP_LSTM),
    "datadriven.neural-ode-linear-case-closed-form": ("L2", checks.CAP_NEURAL_ODE),
    "datadriven.out-of-domain-rejected": ("L1", checks.CAP_ADAPTER),
    "datadriven.missing-optional-backend-is-explicit": ("L1", checks.CAP_ADAPTER),
    "datadriven.magnethub-adapter-mapping": ("L1", checks.CAP_MAGNETHUB),
}


def test_checks_are_registered_with_expected_levels_and_capabilities():
    registry = registered_checks()
    for check_id, (level, capability) in EXPECTED_CHECKS.items():
        assert check_id in registry, f"{check_id} not registered"
        assert registry[check_id].level == level and registry[check_id].capability == capability
    assert {fn.check_id for fn in checks.CHECKS} == set(EXPECTED_CHECKS)  # type: ignore[attr-defined]


@pytest.mark.parametrize("check_fn", checks.CHECKS, ids=lambda fn: fn.check_id)
def test_every_datadriven_check_passes(check_fn):
    result = check_fn()
    assert result.passed, f"{check_fn.check_id} failed: metrics={result.metrics} notes={result.notes}"
    outcome = run_check(registered_checks()[check_fn.check_id])
    assert outcome.status == "pass", outcome.error
    assert outcome.duration_s < 5.0, f"{check_fn.check_id} took {outcome.duration_s:.2f} s"
    record = outcome.to_record()
    assert record.status == "pass" and record.capability.startswith("aipe.magnetics.")


# --------------------------------------------------------------------------------------------------
# Regression tests for the independent review findings (fix stage)
# --------------------------------------------------------------------------------------------------
_LINEAR_SPEC = ArchitectureSpec(family="ann", name="linear", hyperparameters={"hidden_sizes": []},
                                input_features=["frequency", "flux_density_peak"], output_features=["loss_density"],
                                input_units={"frequency": "Hz", "flux_density_peak": "T"}, output_units={"loss_density": "W/m^3"})
_LINEAR_PARAMS = {"layers.0.weight": np.array([[2.0, 3.0]]), "layers.0.bias": np.array([1.0])}


def test_explicit_conditions_cannot_override_conditions_derived_from_inputs(linear_ann: DataDrivenModel):
    good = {"frequency": 1e5, "flux_density_peak": 0.1}
    with pytest.raises(InvalidInputError, match="domain_map"):  # 5 MHz input, 100 kHz "condition": no silent override
        linear_ann.predict({"frequency": 5e6, "flux_density_peak": 0.1}, conditions={"frequency": 1e5})
    with pytest.raises(InvalidInputError):  # even a consistent value is refused: the input is the only source
        linear_ann.predict(good, conditions={"flux_density_peak": 0.1})
    ref = reference_steinmetz_ann()
    with pytest.raises(InvalidInputError):
        ref.predict({"frequency": 2e6, "flux_density_peak": 0.1}, conditions={"frequency": 1e5})
    # unmapped conditions: one value or one value per sample, never truncated or silently broadcast
    for bad in ([300.0, 300.0, 300.0], [[300.0]], [], np.zeros((2, 1))):
        with pytest.raises(InvalidInputError):
            linear_ann.predict(good, conditions={"temperature": bad})
    batch = {"frequency": [1e5, 2e5, 3e5], "flux_density_peak": 0.1}
    assert linear_ann.predict(batch, conditions={"temperature": 300.0}).domain.inside
    assert linear_ann.predict(batch, conditions={"temperature": [300.0, 300.0, 300.0]}).domain.inside
    with pytest.raises(OutOfDomainError):
        linear_ann.predict(batch, conditions={"temperature": [300.0, 300.0, 900.0]})
    warned = linear_ann.predict(batch, conditions={"temperature": [300.0, 300.0, 900.0]}, policy="warn")
    assert warned.domain.inside is False and "temperature: 1 of 3" in warned.domain.violations[0]
    with pytest.raises(InvalidInputError):  # impossible values stay invalid inputs
        linear_ann.predict(batch, conditions={"temperature": [300.0, -1.0, 300.0]})


def test_validity_ranges_must_be_declared_in_si_units(tmp_path: Path):
    inference = InferenceConfig(domain_map={"frequency": "frequency"})
    with pytest.raises(ValidationError, match="SI"):
        _package(tmp_path, _LINEAR_SPEC, _LINEAR_PARAMS, inference=inference,
                 domain=ValidityDomain(frequency=ValidityRange(min=20.0, max=500.0, unit="kHz")))
    with pytest.raises(ValidationError):  # unknown unit
        _package(tmp_path, _LINEAR_SPEC, _LINEAR_PARAMS, domain=ValidityDomain(temperature=ValidityRange(min=1.0, max=2.0, unit="furlong")))
    with pytest.raises(ValidationError):  # the metadata ingestion path rejects it as well
        ModelPackage.model_validate({**load_reference_package().to_json_dict(),
                                     "domain": {"frequency": {"min": 20.0, "max": 500.0, "unit": "kHz"}}})
    package = _package(tmp_path, _LINEAR_SPEC, _LINEAR_PARAMS, inference=inference,
                       domain=ValidityDomain(frequency=ValidityRange(min=1e3, max=1e6, unit="Hz")))
    model = DataDrivenModel(package)
    assert model.predict({"frequency": 1e5, "flux_density_peak": 0.1}).domain.inside
    with pytest.raises(OutOfDomainError):
        model.predict({"frequency": 100.0, "flux_density_peak": 0.1})
    model.domain = ValidityDomain(frequency=ValidityRange(min=20.0, max=500.0, unit="kHz"))  # bypassed validation
    with pytest.raises(DataIntegrityError, match="SI"):
        model.predict({"frequency": 1e5, "flux_density_peak": 0.1})


def test_batch_and_single_sample_evaluations_agree_within_rounding():
    ref = reference_steinmetz_ann()
    rng = np.random.default_rng(2026)
    f = 10 ** rng.uniform(np.log10(20e3), np.log10(500e3), 16)
    b = 10 ** rng.uniform(np.log10(0.02), np.log10(0.3), 16)
    batch = np.asarray(ref.predict({"frequency": f, "flux_density_peak": b}).diagnostics["outputs"]["loss_density"])
    singles = np.array([ref.predict({"frequency": fi, "flux_density_peak": bi}).values["loss_density"].value
                        for fi, bi in zip(f, b)])
    np.testing.assert_allclose(batch, singles, rtol=1e-12, atol=0.0)  # BLAS kernels differ by a few ulp, not more
    result = checks.reference_ann_loads_and_is_deterministic()
    assert result.passed and result.metrics["repeat_bit_identical"] is True
    assert result.metrics["batch_consistent_rtol_1e-12"] is True and result.metrics["batch_vs_single_max_rel_diff"] < 1e-12


def test_empty_batches_and_empty_sequences_are_rejected(linear_ann: DataDrivenModel, tmp_path: Path):
    with pytest.raises(InvalidInputError, match="N = 0"):
        linear_ann.predict({"frequency": np.array([]), "flux_density_peak": np.array([])})
    with pytest.raises(InvalidInputError):
        linear_ann.predict({"frequency": np.array([]), "flux_density_peak": 0.1})
    spec = checks.lstm_check_spec("last")
    lstm = DataDrivenModel(_package(tmp_path, spec, checks.random_parameters(spec, seed=7)))
    with pytest.raises(InvalidInputError, match="L >= 1"):
        lstm.predict({"flux_density": np.zeros((1, 0)), "frequency": 1e5})
    with pytest.raises(InvalidInputError):
        coerce_inputs(spec, {"flux_density": np.zeros((2, 0)), "frequency": [1.0, 2.0]})
    with pytest.raises(InvalidInputError):
        stack_inputs(spec, {"flux_density": np.zeros((2, 0)), "frequency": np.zeros(2)})
    with pytest.raises(InvalidInputError):
        stack_inputs(spec, {"flux_density": np.zeros((0, 4)), "frequency": np.zeros(0)})
    with pytest.raises(InvalidInputError):
        stack_inputs(_LINEAR_SPEC, {"frequency": np.zeros(0), "flux_density_peak": np.zeros(0)})


def test_npz_loader_reports_wrong_content_with_typed_errors(tmp_path: Path):
    np.save(tmp_path / "w.npy", np.zeros(3))
    with pytest.raises(DataIntegrityError, match="not an npz archive"):
        nb.load_npz(tmp_path / "w.npy")
    for name, payload in (("empty.npz", b""), ("trunc.npz", b"PK\x03\x04trunc"), ("garbage.npz", b"garbage")):
        (tmp_path / name).write_bytes(payload)
        with pytest.raises(DataIntegrityError):
            nb.load_npz(tmp_path / name)
    np.savez(tmp_path / "text.npz", a=np.array(["x", "y"]))
    with pytest.raises(DataIntegrityError):
        nb.load_npz(tmp_path / "text.npz")
    with pytest.raises(DataNotFoundError):
        nb.load_npz(tmp_path / "missing.npz")
    np.savez(tmp_path / "ok.npz", **_LINEAR_PARAMS)
    from_path, from_bytes = nb.load_npz(tmp_path / "ok.npz"), nb.load_npz((tmp_path / "ok.npz").read_bytes())
    assert set(from_path) == set(from_bytes) == set(_LINEAR_PARAMS)
    np.testing.assert_array_equal(from_path["layers.0.weight"], from_bytes["layers.0.weight"])
    path = write_aipe_json(tmp_path / "ok.json", _LINEAR_SPEC, _LINEAR_PARAMS)
    assert nb.load_aipe_json(path)[2] == nb.load_aipe_json(path.read_bytes())[2]
    (tmp_path / "latin.json").write_bytes(b"\xff\xfe{}")
    with pytest.raises(DataIntegrityError):
        nb.load_aipe_json(tmp_path / "latin.json")


def test_model_package_load_reports_missing_or_unreadable_metadata(tmp_path: Path):
    with pytest.raises(DataNotFoundError):
        ModelPackage.load(tmp_path / "does-not-exist.json")
    with pytest.raises(DataNotFoundError):
        ModelPackage.load(tmp_path)  # a directory
    with pytest.raises(DataNotFoundError):
        DataDrivenModel.from_path(tmp_path / "nope.json")
    (tmp_path / "bad.json").write_bytes(b"\xff\xfe not utf-8 \x00")
    with pytest.raises(DataIntegrityError):
        ModelPackage.load(tmp_path / "bad.json")


def test_neural_ode_conditions_and_dc_bias_follow_the_adapter_conventions(tmp_path: Path):
    package = checks.build_linear_neural_ode_package(tmp_path, a=100.0, b=1.0)
    package = package.model_copy(update={"domain": ValidityDomain(
        temperature=ValidityRange(min=250.0, max=400.0, unit="K"), dc_bias="unsupported")}).with_base_dir(tmp_path)
    model = NeuralODEModel(package)
    t = np.linspace(0, 1e-2, 11)

    def run(**kwargs):
        return model.simulate(y0=[0.0], t_eval=t, u=lambda tt: [1.0], **kwargs)

    ok = run(conditions={"temperature": Quantity(value=25.0, unit="degC")})  # converted, not rejected
    assert ok.domain.inside and ok.domain.checked["temperature"] == "[250, 400] K"
    with pytest.raises(OutOfDomainError):
        run(conditions={"temperature": 500.0})
    warned = run(conditions={"temperature": 500.0}, policy="warn")
    assert warned.domain.inside is False and any(w.code == "out_of_domain" for w in warned.warnings)
    with pytest.raises(InvalidInputError):  # arrays are not truncated to their first element
        run(conditions={"temperature": [300.0, 900.0]})
    with pytest.raises(InvalidInputError):  # impossible values are invalid inputs, not merely out of domain
        run(conditions={"temperature": -5.0})
    with pytest.raises(InvalidInputError):
        run(conditions={"frequency": 0.0})
    with pytest.raises(InvalidInputError):
        run(conditions={"temperature": Quantity(value=1.0, unit="T")})
    with pytest.raises(InvalidInputError):
        run(conditions={"pressure": 1.0})
    with pytest.raises(OutOfDomainError):  # dc_bias is part of the MagneticModel contract
        run(dc_bias=True)
    assert run(dc_bias=False).domain.inside


def test_neural_ode_validates_solver_settings_before_integrating(tmp_path: Path):
    package = checks.build_linear_neural_ode_package(tmp_path, a=100.0, b=1.0)
    model = NeuralODEModel(package)
    t = np.linspace(0, 1e-2, 11)
    for kwargs in ({"max_step": 0.0}, {"max_step": -1.0}, {"max_step": float("nan")}, {"method": "bogus"},
                   {"method": ""}, {"atol": 0.0}, {"rtol": float("inf")}, {"rtol": True}):
        with pytest.raises(InvalidInputError):
            model.simulate(y0=[0.0], t_eval=t, u=lambda tt: [1.0], **kwargs)
    with pytest.raises(InvalidInputError):
        NeuralODEModel(package, method="bogus")
    with pytest.raises(InvalidInputError):
        NeuralODEModel(package, max_step=0.0)
    out = model.simulate(y0=[0.0], t_eval=t, u=lambda tt: [1.0], method="RK45", max_step=1e-3)
    assert out.diagnostics["method"] == "RK45" and out.diagnostics["max_step"] == 1e-3
    exact = linear_forced_response(t, y0=0.0, a=100.0, b=1.0, u0=1.0)
    np.testing.assert_allclose(out.diagnostics["trajectory"]["y"], exact, rtol=0, atol=1e-6)

    def picky(*args, **kwargs):
        raise ValueError("custom integrator refuses")

    with pytest.raises(InvalidInputError, match="integrator rejected"):
        NeuralODEModel(package, integrator=picky, method="anything").simulate(y0=[0.0], t_eval=t, u=lambda tt: [1.0])
    with pytest.raises(InvalidInputError, match="u\\(t\\) must return"):  # typed errors from u(t) pass through unchanged
        model.simulate(y0=[0.0], t_eval=t, u=lambda tt: [1.0, 2.0])


def test_aipe_json_header_is_cross_checked_against_the_architecture(tmp_path: Path):
    package = load_reference_package()
    weights = tmp_path / package.weights_path().name
    data = load_json(package.weights_path())
    meta = package.to_json_dict()

    def rewrite() -> None:
        dump(data, weights, overwrite=True)
        meta["weights"]["sha256"] = sha256_file(weights)
        meta["weights"]["size_bytes"] = weights.stat().st_size
        dump(meta, tmp_path / "metadata.json", overwrite=True)

    data["family"] = "neural_ode"
    rewrite()
    with pytest.raises(DataIntegrityError, match="family"):
        DataDrivenModel.from_path(tmp_path / "metadata.json")
    data["family"], data["architecture"] = "ann", "something-else"
    rewrite()
    with pytest.raises(DataIntegrityError, match="architecture"):
        DataDrivenModel.from_path(tmp_path / "metadata.json")
    del data["architecture"]  # the architecture name is optional in the header; the family is not
    rewrite()
    assert DataDrivenModel.from_path(tmp_path / "metadata.json").integrity["verified"] is True
    with pytest.raises(DataIntegrityError):
        nb.check_aipe_json_header({}, package.architecture)


def test_unhonoured_inference_dtype_is_reported_not_silent(tmp_path: Path):
    model = DataDrivenModel(_package(tmp_path, _LINEAR_SPEC, _LINEAR_PARAMS, inference=InferenceConfig(dtype="float32")))
    assert [w.code for w in model.load_warnings] == ["dtype_ignored"]
    out = model.predict({"frequency": 1.0, "flux_density_peak": 1.0})
    assert any(w.code == "dtype_ignored" for w in out.warnings)
    assert out.diagnostics["dtype"] == "float64" and out.diagnostics["dtype_configured"] == "float32"
    honoured = DataDrivenModel(_package(tmp_path, _LINEAR_SPEC, _LINEAR_PARAMS))
    assert not honoured.load_warnings
    assert honoured.predict({"frequency": 1.0, "flux_density_peak": 1.0}).diagnostics["dtype"] == "float64"


def test_backend_registry_validates_registrations():
    with pytest.raises(InvalidInputError):
        register_backend("probe-unknown-family", NumpyBackend, {"npz": ["cnn"]})
    with pytest.raises(InvalidInputError):
        register_backend("probe-unknown-format", NumpyBackend, {"onnx": ["ann"]})
    with pytest.raises(InvalidInputError):
        register_backend("probe-empty", NumpyBackend, {})
    with pytest.raises(InvalidInputError):  # the default table cannot be replaced by accident
        register_backend("numpy", NumpyBackend, {"npz": ["ann"]})
    assert isinstance(get_backend("aipe-json", "ann"), NumpyBackend)
    assert "probe-unknown-family" not in available_backends()
    try:
        register_backend("probe-ok", NumpyBackend, {"npz": ["ann"]})
        with pytest.raises(InvalidInputError):
            register_backend("probe-ok", NumpyBackend, {"npz": ["lstm"]})
        register_backend("probe-ok", NumpyBackend, {"npz": ["lstm"]}, replace=True)
        assert available_backends()["probe-ok"] == {"npz": ["lstm"]}
    finally:
        assert unregister_backend("probe-ok") is True
    assert unregister_backend("probe-ok") is False


def test_write_aipe_json_protects_reserved_header_keys(tmp_path: Path):
    for key in ("format", "format_version", "family", "architecture", "layers", "lstm", "head"):
        with pytest.raises(InvalidInputError, match="reserved"):
            write_aipe_json(tmp_path / "x.json", _LINEAR_SPEC, _LINEAR_PARAMS, extra={key: "evil"})
    path = write_aipe_json(tmp_path / "x.json", _LINEAR_SPEC, _LINEAR_PARAMS, extra={"note": "fine"})
    assert nb.load_aipe_json(path)[2]["note"] == "fine"


def test_hyperparameters_are_validated_strictly_and_stored_in_typed_form():
    for bad in ([8.0], ["8"], [True], [0], [-1], 8):
        with pytest.raises(ValidationError):
            ArchitectureSpec(family="ann", name="x", hyperparameters={"hidden_sizes": bad}, input_features=["a"], output_features=["b"])
    with pytest.raises(ValidationError):
        ArchitectureSpec(family="lstm", name="x", hyperparameters={"hidden_size": 2.0}, input_features=["a"],
                         output_features=["b"], sequence_features=["a"])
    spec = ArchitectureSpec(family="ann", name="x", hyperparameters={"hidden_sizes": (4, 2)}, input_features=["a"], output_features=["b"])
    assert spec.hyperparameters == {"kind": "mlp", "hidden_sizes": [4, 2], "activation": "tanh", "output_activation": "identity"}
    assert spec.hyperparameters == spec.typed_hyperparameters().model_dump()
    assert ArchitectureSpec.model_validate(spec.model_dump()).hyperparameters == spec.hyperparameters
    assert ArchitectureSpec.model_validate_json(spec.model_dump_json()) == spec
    external = magnethub_package("N87").architecture
    assert external.hyperparameters["kind"] == "external" and external.hyperparameters["material"] == "N87"
    committed = load_reference_package()
    assert committed.to_json_dict()["architecture"]["hyperparameters"] == load_json(reference_package_path())["architecture"]["hyperparameters"]


def test_weights_path_policy_and_size_declaration(tmp_path: Path):
    package = _package(tmp_path, _LINEAR_SPEC, _LINEAR_PARAMS)
    escaped = package.model_copy(update={"weights": package.weights.model_copy(update={"path": "../w.aipe.json"})}
                                 ).with_base_dir(tmp_path / "inner")
    with pytest.raises(InvalidInputError, match="escapes"):
        escaped.weights_path()
    with pytest.raises(InvalidInputError):
        DataDrivenModel(escaped)
    absolute = package.model_copy(update={"weights": package.weights.model_copy(
        update={"path": str((tmp_path / "w.aipe.json").resolve())})})  # trusted-caller feature, still digest-checked
    assert DataDrivenModel(absolute).integrity["verified"] is True
    wrong_size = package.model_copy(update={"weights": package.weights.model_copy(update={"size_bytes": 1})}).with_base_dir(tmp_path)
    with pytest.raises(DataIntegrityError, match="size_bytes"):
        DataDrivenModel(wrong_size)
    size = (tmp_path / "w.aipe.json").stat().st_size
    right_size = package.model_copy(update={"weights": package.weights.model_copy(update={"size_bytes": size})}).with_base_dir(tmp_path)
    assert DataDrivenModel(right_size).integrity["verified"] is True
    data, digest = package.read_weights()
    assert digest == package.weights.sha256 == sha256_file(tmp_path / "w.aipe.json") and len(data) == size
    assert package.read_weights()[0] is data  # cached until released: the backend parses this very buffer
    package.release_weights()
    assert package.read_weights()[0] is not data
    package.release_weights()


def test_naive_lstm_reference_implements_every_head_activation():
    for act in ("relu", "gelu", "sigmoid", "identity", "tanh"):
        spec = ArchitectureSpec(family="lstm", name=f"lstm-{act}", input_features=["b", "f"], output_features=["p"],
                                sequence_features=["b"], hyperparameters={"hidden_size": 3, "num_layers": 1, "head_hidden_sizes": [2],
                                                                          "head_activation": act, "head_output_activation": act})
        params = checks.random_parameters(spec, seed=3)
        x = np.random.default_rng(4).normal(size=(2, 4, 2))
        np.testing.assert_allclose(nb.build_network(spec, params)(x), checks.naive_lstm_forward(spec, params, x), rtol=0, atol=1e-13)
    assert checks.naive_activation("relu", -1.0) == 0.0 and checks.naive_activation("gelu", 0.0) == 0.0
    with pytest.raises(ValueError):
        checks.naive_activation("swish", 0.1)


def test_benchmark_iteration_cap_is_one_constant():
    import inspect

    bench = _load_benchmark_module()
    assert bench.MAX_ITER == 20000
    assert inspect.signature(bench.fit).parameters["max_iter"].default == bench.MAX_ITER
    assert inspect.signature(bench.run).parameters["max_iter"].default == bench.MAX_ITER
    assert load_reference_package().training["optimizer"]["max_iter"] == bench.MAX_ITER
