"""Empirical core-loss models: classical Steinmetz equation and the improved Generalized Steinmetz Equation.

``MODELS`` lists the model classes (factories) of this family for registry / discovery code.
"""

from aipe_magnetics.models.empirical.igse import (
    DC_BIAS_RELATIVE_THRESHOLD,
    IGSE_MODEL_ID,
    IGSE_SOURCE,
    SAMPLING_WARNING_THRESHOLD,
    IGSEModel,
    abs_cos_power_integral,
    abs_cos_power_integral_closed_form,
    igse_coefficient,
    igse_coefficient_venkatachalam_approximation,
    igse_trapezoidal_closed_form,
    igse_triangular_closed_form,
)
from aipe_magnetics.models.empirical.steinmetz import (
    FIT_TEMPERATURE_TOLERANCE_K,
    MODEL_VERSION,
    STEINMETZ_MODEL_ID,
    STEINMETZ_SOURCE,
    SteinmetzModel,
    SteinmetzParameterisedModel,
    select_parameter_set,
)

MODELS = [SteinmetzModel, IGSEModel]
MODEL_IDS = {model.MODEL_ID: model for model in MODELS}

__all__ = [
    "MODELS",
    "MODEL_IDS",
    "MODEL_VERSION",
    "STEINMETZ_MODEL_ID",
    "STEINMETZ_SOURCE",
    "IGSE_MODEL_ID",
    "IGSE_SOURCE",
    "DC_BIAS_RELATIVE_THRESHOLD",
    "SAMPLING_WARNING_THRESHOLD",
    "FIT_TEMPERATURE_TOLERANCE_K",
    "SteinmetzParameterisedModel",
    "SteinmetzModel",
    "IGSEModel",
    "select_parameter_set",
    "abs_cos_power_integral",
    "abs_cos_power_integral_closed_form",
    "igse_coefficient",
    "igse_coefficient_venkatachalam_approximation",
    "igse_trapezoidal_closed_form",
    "igse_triangular_closed_form",
]
