"""Tests for the Bertotti loss-separation model and its verification checks."""

from __future__ import annotations

import math

import pytest

from aipe_magnetics.common.errors import InvalidInputError, NumericalError, OutOfDomainError
from aipe_magnetics.common.results import SourceReference
from aipe_magnetics.models.contract import CoreLossModel, ModelOutput
from aipe_magnetics.models.physics import BERTOTTI_MODEL_ID, MODEL_IDS, MODELS, BertottiLossModel
from aipe_magnetics.verification.checks import check_physics


@pytest.fixture
def model() -> BertottiLossModel:
    # synthetic coefficients for API tests only; not a material fit
    return BertottiLossModel(k_h=20.0, beta_h=1.8, k_c=0.05, k_e=0.5, frequency_range=(10.0, 1e5),
                             flux_density_range=(0.0, 2.0), temperature_range=(293.15, 373.15))


class TestBertotti:
    def test_identity_and_contract(self, model):
        assert isinstance(model, CoreLossModel)
        assert model.identity.id == BERTOTTI_MODEL_ID == "aipe.magnetics.model.bertotti"
        assert BertottiLossModel in MODELS and MODEL_IDS[BERTOTTI_MODEL_ID] is BertottiLossModel
        assert [p.name for p in model.parameters] == ["k_h", "beta_h", "k_c", "k_e"]
        assert model.domain.excitation == ["sinusoidal"] and model.domain.dc_bias == "unsupported"
        assert any(s.id == "pub.bertotti-1988" for s in model.sources)
        described = model.describe()
        assert described["identity"]["id"] == BERTOTTI_MODEL_ID and "frequency" in described["domain"]

    def test_components_and_values(self, model):
        f, b = 1000.0, 1.0
        parts = model.components(f, b)
        assert parts["hysteresis"] == pytest.approx(20.0 * f * b ** 1.8)
        assert parts["classical"] == pytest.approx(0.05 * f ** 2 * b ** 2)
        assert parts["excess"] == pytest.approx(0.5 * f ** 1.5 * b ** 1.5)
        assert parts["total"] == pytest.approx(sum(parts[k] for k in ("hysteresis", "classical", "excess")))
        out = model.loss_density(frequency=f, flux_density_peak=b, temperature=300.0)
        assert isinstance(out, ModelOutput) and out.domain.inside
        assert out.values["loss_density"].unit == "W/m^3"
        assert out.values["loss_density"].value == pytest.approx(parts["total"])
        assert out.values["energy_per_cycle_density"].value == pytest.approx(parts["total"] / f)
        assert out.values["energy_per_cycle_density"].unit == "J/m^3"
        assert {a.id for a in out.assumptions} >= {"bertotti.sinusoidal", "bertotti.fit"}
        assert out.limitations and out.warnings == []
        out.model_dump(mode="json")
        # evaluate() delegates to loss_density
        assert model.evaluate(frequency=f, flux_density_peak=b).values["loss_density"].value == pytest.approx(parts["total"])

    def test_zero_excitation(self, model):
        assert model.loss_density(frequency=0.0, flux_density_peak=1.0, policy="warn").values["loss_density"].value == 0.0
        assert model.loss_density(frequency=1e3, flux_density_peak=0.0).values["loss_density"].value == 0.0
        assert "energy_per_cycle_density" not in model.loss_density(frequency=0.0, flux_density_peak=1.0,
                                                                    policy="warn").values

    def test_temperature_info_warning(self, model):
        out = model.loss_density(frequency=1e3, flux_density_peak=1.0)
        assert any(w.code == "temperature_unknown" for w in out.warnings)

    @pytest.mark.parametrize("kwargs", [
        {"k_h": -1.0}, {"beta_h": 0.0}, {"beta_h": -2.0}, {"k_c": float("nan")}, {"k_e": float("inf")},
        {"k_h": True}, {"beta_h": "1.8"}, {"k_c": None},
    ])
    def test_invalid_coefficients(self, kwargs):
        params = {"k_h": 1.0, "beta_h": 1.8, "k_c": 0.0, "k_e": 0.0, **kwargs}
        with pytest.raises(InvalidInputError):
            BertottiLossModel(**params)

    def test_invalid_inputs(self, model):
        with pytest.raises(InvalidInputError):
            model.loss_density(frequency=-1.0, flux_density_peak=1.0)
        with pytest.raises(InvalidInputError):
            model.loss_density(frequency=1e3, flux_density_peak=float("nan"))
        with pytest.raises(InvalidInputError):
            model.loss_density(frequency=1e3, flux_density_peak=1.0, temperature=-5.0)
        with pytest.raises(InvalidInputError):
            model.loss_density(frequency=1e3, flux_density_peak=1.0, waveform="square")
        with pytest.raises(InvalidInputError):
            BertottiLossModel(1.0, 1.8, 0.0, 0.0, flux_density_range=(2.0, 1.0))

    def test_domain_policy(self, model):
        with pytest.raises(OutOfDomainError):
            model.loss_density(frequency=1e6, flux_density_peak=1.0)
        with pytest.raises(OutOfDomainError):
            model.loss_density(frequency=1e3, flux_density_peak=1.0, temperature=500.0)
        out = model.loss_density(frequency=1e6, flux_density_peak=3.0, policy="warn")
        assert not out.domain.inside and len(out.domain.violations) == 2
        assert any(w.code == "out_of_domain" for w in out.warnings)

    def test_overflow_is_numerical_error(self):
        unbounded = BertottiLossModel(k_h=1.0, beta_h=1.8, k_c=1.0, k_e=1.0)
        with pytest.raises(NumericalError):
            unbounded.loss_density(frequency=1e3, flux_density_peak=1e200)
        with pytest.raises(NumericalError):
            unbounded.loss_density(frequency=1e200, flux_density_peak=1.0)

    def test_from_lamination(self):
        sigma, d = 2.0e6, 0.35e-3
        k_c = BertottiLossModel.classical_eddy_coefficient(sigma, d)
        assert k_c == pytest.approx(math.pi ** 2 * sigma * d ** 2 / 6.0)
        model = BertottiLossModel.from_lamination(10.0, 1.7, 0.2, conductivity=sigma, thickness=d,
                                                  source=SourceReference(id="src.test", kind="assumption",
                                                                         reference="synthetic test coefficients"))
        assert model.k_c == pytest.approx(k_c) and model.lamination == {"conductivity": sigma, "thickness": d}
        assert all(p.source_ref == "src.test" for p in model.parameters)
        out = model.loss_density(frequency=50.0, flux_density_peak=1.0)
        assert any(a.id == "bertotti.thin-lamination" for a in out.assumptions)
        with pytest.raises(InvalidInputError):
            BertottiLossModel.classical_eddy_coefficient(sigma, 0.0)
        with pytest.raises(InvalidInputError):
            BertottiLossModel.classical_eddy_coefficient(float("nan"), d)


class TestBertottiRegressions:
    def test_numpy_scalars_are_accepted_like_python_numbers(self):
        import numpy as np

        model = BertottiLossModel(np.float32(20.0), np.float64(1.8), np.int64(0), np.float32(0.5),
                                  frequency_range=(np.float32(10.0), 1e5))
        out = model.loss_density(frequency=np.int64(1000), flux_density_peak=np.float32(1.0))
        assert out.values["loss_density"].value == pytest.approx(20.0 * 1000.0 + 0.5 * 1000.0 ** 1.5)
        assert isinstance(model.k_h, float) and model.domain.frequency.min == 10.0
        with pytest.raises(InvalidInputError):
            BertottiLossModel(np.bool_(True), 1.8, 0.0, 0.0)

    def test_lamination_provenance_is_validated(self):
        sigma, d = 2.0e6, 0.35e-3
        k_c = BertottiLossModel.classical_eddy_coefficient(sigma, d)
        model = BertottiLossModel(20.0, 1.8, k_c, 0.5, lamination={"conductivity": sigma, "thickness": d})
        assert model.lamination == {"conductivity": sigma, "thickness": d}
        out = model.loss_density(frequency=50.0, flux_density_peak=1.0)
        assert any(a.id == "bertotti.thin-lamination" for a in out.assumptions)
        for bad in ({"sigma": sigma}, {"conductivity": sigma}, {"conductivity": sigma, "thickness": d, "width": 1.0},
                    {"conductivity": -sigma, "thickness": d}, "steel"):
            with pytest.raises(InvalidInputError):
                BertottiLossModel(20.0, 1.8, k_c, 0.5, lamination=bad)
        with pytest.raises(InvalidInputError):  # k_c not derived from the lamination data: the assumption would be false
            BertottiLossModel(20.0, 1.8, 0.05, 0.5, lamination={"conductivity": sigma, "thickness": d})


BERTOTTI_CHECKS = [
    check_physics.bertotti_scaling_and_limiting_cases,
    check_physics.bertotti_invalid_parameters_rejected,
    check_physics.bertotti_classical_eddy_coefficient,
]


@pytest.mark.parametrize("func", BERTOTTI_CHECKS, ids=[f.check_id for f in BERTOTTI_CHECKS])
def test_bertotti_checks_pass(func):
    result = func()
    assert result.passed, f"{func.check_id} failed: {result.metrics}"
