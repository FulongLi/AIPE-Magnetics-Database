"""Tests for aipe_magnetics.models.empirical.steinmetz."""

from __future__ import annotations

import json

import numpy as np
import pytest

from aipe_magnetics.common.errors import DataNotFoundError, InvalidInputError, NumericalError, OutOfDomainError
from aipe_magnetics.common.results import SourceReference
from aipe_magnetics.data.records import RangeSI, SteinmetzParameterSet
from aipe_magnetics.models.contract import CoreLossModel, ModelOutput, ValidityRange
from aipe_magnetics.models.empirical import MODELS, MODEL_IDS
from aipe_magnetics.models.empirical.steinmetz import (
    FIT_TEMPERATURE_TOLERANCE_K,
    STEINMETZ_MODEL_ID,
    STEINMETZ_SOURCE,
    SteinmetzModel,
    select_parameter_set,
)

# SYNTHETIC parameters: k=1 W/m^3 (f in Hz, B in T), alpha=1.5, beta=2.5 -> Pv(100 kHz, 0.1 T) = 100 kW/m^3.
K, ALPHA, BETA = 1.0, 1.5, 2.5
NAN, INF = float("nan"), float("inf")
SYNTHETIC_SOURCE = SourceReference(id="internal.test-synthetic", kind="internal",
                                   reference="synthetic parameters for unit tests (not a material fit)")


def make_model(**kwargs) -> SteinmetzModel:
    defaults = dict(material_id="synthetic.test-material", fit_temperature=373.15, frequency_range=(1e3, 1e6),
                    flux_density_range=(1e-3, 0.3), sources=[SYNTHETIC_SOURCE], synthetic=True)
    defaults.update(kwargs)
    return SteinmetzModel(K, ALPHA, BETA, **defaults)


# --------------------------------------------------------------------------------------------------
# identity, domain, construction
# --------------------------------------------------------------------------------------------------
def test_identity_and_registry():
    model = make_model()
    assert isinstance(model, CoreLossModel)
    assert model.identity.id == STEINMETZ_MODEL_ID == "aipe.magnetics.model.steinmetz"
    assert model.identity.family == "empirical" and model.identity.version == "0.2.0"
    assert SteinmetzModel in MODELS and MODEL_IDS[STEINMETZ_MODEL_ID] is SteinmetzModel
    described = model.describe()
    assert described["identity"]["id"] == STEINMETZ_MODEL_ID
    assert {p["name"] for p in described["parameters"]} == {"k", "alpha", "beta"}
    json.dumps(described)


def test_domain_declaration():
    model = make_model()
    assert model.domain.excitation == ["sinusoidal"]
    assert model.domain.dc_bias == "unsupported" and model.domain.requires_history is False
    assert model.domain.materials == ["synthetic.test-material"]
    assert model.domain.frequency == ValidityRange(min=1e3, max=1e6, unit="Hz")
    assert model.domain.flux_density_peak == ValidityRange(min=1e-3, max=0.3, unit="T")
    # implied +-5 K band around the fit temperature when no explicit range is given
    assert model.temperature_range_implied
    assert model.domain.temperature == ValidityRange(min=373.15 - FIT_TEMPERATURE_TOLERANCE_K,
                                                     max=373.15 + FIT_TEMPERATURE_TOLERANCE_K, unit="K")
    assert any("synthetic" in item.lower() for item in model.domain.known_limitations)
    bare = SteinmetzModel(K, ALPHA, BETA)
    assert bare.domain.materials is None and bare.domain.temperature is None and bare.domain.frequency is None


def test_ranges_accept_other_units_and_record_types():
    model = SteinmetzModel(K, ALPHA, BETA, frequency_range=RangeSI(min=10, max=500, unit="kHz"),
                           temperature_range=RangeSI(min=25, max=100, unit="degC"),
                           flux_density_range=ValidityRange(min=10, max=200, unit="mT"))
    assert model.domain.frequency == ValidityRange(min=1e4, max=5e5, unit="Hz")
    assert model.domain.temperature.min == pytest.approx(298.15) and model.domain.temperature.max == pytest.approx(373.15)
    assert model.domain.flux_density_peak == ValidityRange(min=0.01, max=0.2, unit="T")
    assert not model.temperature_range_implied


@pytest.mark.parametrize("kwargs", [
    dict(k=0.0), dict(k=-1.0), dict(k=NAN), dict(k="1"), dict(alpha=0.0), dict(alpha=True), dict(beta=-2.5),
    dict(beta=INF), dict(fit_temperature=0.0), dict(frequency_range=(1e6, 1e3)),
    dict(frequency_range=ValidityRange(min=1, max=2, unit="T")), dict(frequency_range=(1e3,)),
    dict(flux_density_range=(NAN, 0.3)), dict(sources=["not a source"]), dict(material_id=""),
])
def test_constructor_rejects_invalid_parameters(kwargs):
    params = dict(k=K, alpha=ALPHA, beta=BETA)
    params.update(kwargs)
    with pytest.raises(InvalidInputError):
        SteinmetzModel(params.pop("k"), params.pop("alpha"), params.pop("beta"), **params)


# --------------------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------------------
def test_loss_density_value_and_envelope():
    model = make_model()
    output = model.loss_density(frequency=1e5, flux_density_peak=0.1, temperature=373.15)
    assert isinstance(output, ModelOutput)
    quantity = output.value("loss_density")
    assert quantity.unit == "W/m^3" and quantity.value == pytest.approx(1e5, rel=1e-12)
    assert output.model.id == STEINMETZ_MODEL_ID
    assert output.domain.inside and output.warnings == []
    assert output.diagnostics["out_of_domain"] is False and output.diagnostics["zero_excitation"] is False
    assert {a.id for a in output.assumptions} == {"steinmetz_parameters", "temperature_handling"}
    codes = {item.code for item in output.limitations}
    assert {"sinusoidal_only", "no_dc_bias", "no_minor_loops", "temperature_specific_parameters", "fit_range_only",
            "synthetic_parameters"} <= codes
    assert [s.id for s in output.sources] == [STEINMETZ_SOURCE.id, SYNTHETIC_SOURCE.id]
    json.dumps(output.model_dump(mode="json"))


def test_closed_form_matches_formula():
    model = SteinmetzModel(2.5, 1.3, 2.7)
    assert model.closed_form(12345.0, 0.0789) == pytest.approx(2.5 * 12345.0 ** 1.3 * 0.0789 ** 2.7, rel=1e-14)
    assert model.closed_form(12345.0, 0.0) == 0.0
    with pytest.raises(InvalidInputError):
        model.closed_form(-1.0, 0.1)


def test_evaluate_passthrough_from_contract():
    model = make_model()
    direct = model.loss_density(frequency=2e5, flux_density_peak=0.05)
    via_contract = model.evaluate(frequency=2e5, flux_density_peak=0.05)
    assert via_contract.value("loss_density") == direct.value("loss_density")


def test_zero_excitation_returns_zero_with_info_warning():
    model = make_model()
    output = model.loss_density(frequency=1e5, flux_density_peak=0.0)
    assert output.value("loss_density").value == 0.0 and output.value("loss_density").unit == "W/m^3"
    assert [(w.code, w.severity) for w in output.warnings] == [("zero_excitation", "info")]
    assert output.diagnostics["zero_excitation"] is True
    # zero flux is not checked against the flux-density range (min 1 mT) ...
    assert output.domain.inside
    # ... but the frequency still is
    with pytest.raises(OutOfDomainError):
        model.loss_density(frequency=1e7, flux_density_peak=0.0)


@pytest.mark.parametrize("kwargs", [
    dict(frequency=-1e5, flux_density_peak=0.1), dict(frequency=0.0, flux_density_peak=0.1),
    dict(frequency=NAN, flux_density_peak=0.1), dict(frequency=INF, flux_density_peak=0.1),
    dict(frequency=True, flux_density_peak=0.1), dict(frequency="1e5", flux_density_peak=0.1),
    dict(frequency=None, flux_density_peak=0.1), dict(frequency=1e5, flux_density_peak=-0.1),
    dict(frequency=1e5, flux_density_peak=INF), dict(frequency=1e5, flux_density_peak=NAN),
    dict(frequency=1e5, flux_density_peak=[0.1]), dict(frequency=1e5, flux_density_peak=0.1, temperature=0.0),
    dict(frequency=1e5, flux_density_peak=0.1, temperature=NAN),
    dict(frequency=1e5, flux_density_peak=0.1, policy="clip"), dict(frequency=1e5, flux_density_peak=0.1, policy=None),
    dict(frequency=1e5, flux_density_peak=0.1, waveform="sine"),
])
def test_invalid_inputs_raise_invalid_input_error(kwargs):
    with pytest.raises(InvalidInputError):
        make_model().loss_density(**kwargs)


def test_domain_policy_reject_and_warn():
    model = make_model()
    with pytest.raises(OutOfDomainError) as excinfo:
        model.loss_density(frequency=1e7, flux_density_peak=0.1)
    assert "frequency" in str(excinfo.value) and excinfo.value.details["model"] == STEINMETZ_MODEL_ID
    output = model.loss_density(frequency=1e7, flux_density_peak=0.1, policy="warn")
    assert output.diagnostics["out_of_domain"] is True and not output.domain.inside
    assert [w.code for w in output.warnings] == ["out_of_domain"]
    assert output.value("loss_density").value == pytest.approx(K * 1e7 ** ALPHA * 0.1 ** BETA)
    with pytest.raises(OutOfDomainError):
        model.loss_density(frequency=1e5, flux_density_peak=0.5)
    with pytest.raises(OutOfDomainError):
        model.loss_density(frequency=1e5, flux_density_peak=0.1, temperature=298.15)
    assert model.loss_density(frequency=1e5, flux_density_peak=0.1, temperature=370.0).domain.inside


def test_temperature_assumptions_are_stated():
    model = make_model()
    without = model.loss_density(frequency=1e5, flux_density_peak=0.1)
    statement = next(a.statement for a in without.assumptions if a.id == "temperature_handling")
    assert "373.15" in statement and "No temperature given" in statement
    with_t = model.loss_density(frequency=1e5, flux_density_peak=0.1, temperature=375.0)
    statement = next(a.statement for a in with_t.assumptions if a.id == "temperature_handling")
    assert "375" in statement and "without temperature correction" in statement
    assert with_t.diagnostics["temperature"] == 375.0


@pytest.mark.parametrize("frequency,peak", [(1e300, 1e300), (1e200, 1e100), (1e308, 1.0)])
def test_overflow_raises_numerical_error(frequency, peak):
    model = SteinmetzModel(K, ALPHA, BETA)
    with pytest.raises(NumericalError):
        model.loss_density(frequency=frequency, flux_density_peak=peak)
    with pytest.raises(NumericalError):
        model.closed_form(frequency, peak)


# --------------------------------------------------------------------------------------------------
# vectorised helper
# --------------------------------------------------------------------------------------------------
def test_loss_density_array_matches_scalar_and_broadcasts():
    model = make_model()
    frequencies = np.array([1e3, 1e4, 1e5])[:, None]
    peaks = np.array([0.01, 0.1])[None, :]
    grid = model.loss_density_array(frequencies, peaks)
    assert grid.shape == (3, 2)
    for i, f in enumerate(frequencies[:, 0]):
        for j, b in enumerate(peaks[0]):
            assert grid[i, j] == pytest.approx(model.loss_density(frequency=f, flux_density_peak=b).value(
                "loss_density").value, rel=1e-14)
    assert model.loss_density_array([1e4, 1e5], [0.0, 0.0]).tolist() == [0.0, 0.0]
    assert model.loss_density_array(1e5, 0.1) == pytest.approx(1e5)


def test_loss_density_array_validation_and_domain():
    model = make_model()
    with pytest.raises(InvalidInputError):
        model.loss_density_array([-1e5], [0.1])
    with pytest.raises(InvalidInputError):
        model.loss_density_array([1e5], [-0.1])
    with pytest.raises(InvalidInputError):
        model.loss_density_array([1e5], [NAN])
    with pytest.raises(InvalidInputError):
        model.loss_density_array([1e5, 1e4, 1e3], [0.1, 0.2])
    with pytest.raises(InvalidInputError):
        model.loss_density_array([1e5], [0.1], policy="clip")
    with pytest.raises(OutOfDomainError):
        model.loss_density_array([1e7, 1e5], [0.1, 0.1])
    with pytest.raises(OutOfDomainError):
        model.loss_density_array([1e5], [0.1], temperature=298.15)
    warned = model.loss_density_array([1e7, 1e5], [0.1, 0.1], policy="warn")
    assert warned.shape == (2,)
    assert model.in_domain_mask([1e7, 1e5, 1e5], [0.1, 0.1, 0.0]).tolist() == [False, True, True]
    assert model.in_domain_mask([1e5], [0.5]).tolist() == [False]
    with pytest.raises(NumericalError):
        model.loss_density_array([1e300], [1e300], policy="warn")


# --------------------------------------------------------------------------------------------------
# parameter sets
# --------------------------------------------------------------------------------------------------
def make_set(id_: str, k: float = 1.0, temperature: float | None = None, temperature_range=None,
             basis: str = "synthetic", **kwargs) -> SteinmetzParameterSet:
    data = dict(id=id_, k={"value": k, "unit": "W/m^3"}, alpha=ALPHA, beta=BETA, basis=basis)
    if temperature is not None:
        data["temperature"] = {"value": temperature, "unit": "K"}
    if temperature_range is not None:
        data["temperature_range"] = temperature_range
    data.update(kwargs)
    return SteinmetzParameterSet.model_validate(data)


def test_from_parameter_set():
    record = make_set("synthetic.set-100c", k=0.8, temperature=373.15,
                      temperature_range={"min": 353.15, "max": 393.15, "unit": "K"},
                      frequency_range={"min": 20, "max": 500, "unit": "kHz"},
                      flux_density_range={"min": 10, "max": 200, "unit": "mT"}, source_ref="evidence.synthetic",
                      note="synthetic")
    model = SteinmetzModel.from_parameter_set(record, material_id="synthetic.mat", sources=[SYNTHETIC_SOURCE])
    assert model.k == 0.8 and model.alpha == ALPHA and model.beta == BETA
    assert model.material_id == "synthetic.mat" and model.parameter_set_id == "synthetic.set-100c"
    assert model.synthetic and model.parameter_basis == "synthetic" and model.source_ref == "evidence.synthetic"
    assert model.fit_temperature == 373.15 and not model.temperature_range_implied
    assert model.domain.frequency == ValidityRange(min=2e4, max=5e5, unit="Hz")
    assert model.domain.flux_density_peak == ValidityRange(min=0.01, max=0.2, unit="T")
    assert model.domain.temperature == ValidityRange(min=353.15, max=393.15, unit="K")
    assert model.parameters[0].source_ref == "evidence.synthetic"
    output = model.loss_density(frequency=1e5, flux_density_peak=0.1, temperature=360.0)
    assert any(item.code == "synthetic_parameters" for item in output.limitations)
    assert "SYNTHETIC" in next(a.statement for a in output.assumptions if a.id == "steinmetz_parameters")
    assert output.value("loss_density").value == pytest.approx(0.8e5)
    with pytest.raises(InvalidInputError):
        SteinmetzModel.from_parameter_set({"k": 1.0})


def test_from_parameter_set_datasheet_basis_is_reported():
    record = make_set("synthetic.set-ds", basis="datasheet_fit", temperature=298.15)
    model = SteinmetzModel.from_parameter_set(record)
    assert not model.synthetic
    output = model.loss_density(frequency=1e5, flux_density_peak=0.1)
    assumption = next(a for a in output.assumptions if a.id == "steinmetz_parameters")
    assert assumption.basis == "datasheet"
    assert not any(item.code == "synthetic_parameters" for item in output.limitations)


def test_select_parameter_set_by_range_and_fit_temperature():
    cold = make_set("synthetic.set-25c", temperature=298.15)
    hot = make_set("synthetic.set-100c", temperature=373.15, temperature_range={"min": 353.15, "max": 393.15, "unit": "K"})
    celsius = make_set("synthetic.set-c", temperature_range={"min": 120, "max": 140, "unit": "degC"})
    sets = [cold, hot, celsius]
    assert select_parameter_set(sets, 298.15) is cold
    assert select_parameter_set(sets, 298.15 + FIT_TEMPERATURE_TOLERANCE_K) is cold
    assert select_parameter_set(sets, 360.0) is hot
    assert select_parameter_set(sets, 393.15) is hot
    assert select_parameter_set(sets, 400.0) is celsius  # 126.85 degC lies in [120, 140] degC
    with pytest.raises(DataNotFoundError) as excinfo:
        select_parameter_set(sets, 320.0)
    message = str(excinfo.value)
    assert "320" in message and "synthetic.set-25c" in message and "synthetic.set-100c" in message
    assert excinfo.value.details["temperature"] == 320.0 and len(excinfo.value.details["available"]) == 3
    with pytest.raises(DataNotFoundError):
        select_parameter_set(sets, 298.15 + FIT_TEMPERATURE_TOLERANCE_K + 1e-6)  # no nearest-neighbour fallback


def test_select_parameter_set_tie_breaking_and_unstated_temperature():
    wide = make_set("synthetic.wide", temperature_range={"min": 250, "max": 450, "unit": "K"})
    narrow = make_set("synthetic.narrow", temperature_range={"min": 290, "max": 310, "unit": "K"})
    fitted = make_set("synthetic.fitted", temperature=305.0, temperature_range={"min": 250, "max": 450, "unit": "K"})
    unstated = make_set("synthetic.unstated")
    assert select_parameter_set([wide, narrow], 300.0) is narrow  # narrower range first
    assert select_parameter_set([wide, narrow, fitted], 300.0) is fitted  # a stated fit temperature wins
    assert select_parameter_set([unstated, wide], 300.0) is wide
    with pytest.raises(DataNotFoundError) as excinfo:
        select_parameter_set([unstated], 300.0)
    assert "never selected" in str(excinfo.value)


@pytest.mark.parametrize("args", [([], NAN), ([], -5.0), ([], "300"), ("not a list", 300.0), ([1, 2], 300.0)])
def test_select_parameter_set_invalid_inputs(args):
    sets, temperature = args
    with pytest.raises(InvalidInputError):
        select_parameter_set(sets, temperature)
