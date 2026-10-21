"""LossEvaluator flux-density range warnings name the violated bound (regression for the two-sided lower-range check)."""

from __future__ import annotations

import numpy as np

from aipe_magnetics.database import FluxShape, LossEvaluator, material_data


def _evaluator() -> LossEvaluator:
    md = material_data("N87")
    return LossEvaluator(md.id, md.steinmetz_sets)


def test_value_above_the_fitted_range_is_reported_as_exceeding_only():
    ev = _evaluator().loss_density(100e3, 0.35, 373.15, FluxShape.triangular(0.5), policy="warn")  # 100 degC datasheet fit: 50..200 mT
    assert any("exceeds" in w for w in ev.warnings)
    assert not any("below" in w for w in ev.warnings)


def test_value_below_the_fitted_range_is_reported_as_below_only():
    ev = _evaluator().loss_density(100e3, 0.02, 373.15, FluxShape.triangular(0.5), policy="warn")
    assert any("below" in w for w in ev.warnings)
    assert not any("exceeds" in w for w in ev.warnings)


def test_array_input_checks_its_largest_and_smallest_non_zero_elements():
    peaks = np.array([0.0, 0.02, 0.1, 0.35])
    ev = _evaluator().loss_density(100e3, peaks, 373.15, FluxShape.triangular(0.5), policy="warn")
    assert np.asarray(ev.loss_density).shape == peaks.shape
    assert any("exceeds" in w for w in ev.warnings) and any("below" in w and "smallest non-zero" in w for w in ev.warnings)
    inside = _evaluator().loss_density(100e3, np.array([0.06, 0.1, 0.15]), 373.15, FluxShape.triangular(0.5), policy="warn")
    assert not any("exceeds" in w or "below" in w for w in inside.warnings)
