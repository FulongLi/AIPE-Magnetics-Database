"""Physics-based model family.

* :mod:`anhysteretic` - Langevin anhysteretic magnetisation (numerically stable).
* :mod:`jiles_atherton` - scalar Jiles-Atherton hysteresis model solved as an ODE driven by ``H(t)``.
* :mod:`bertotti` - Bertotti loss separation (hysteresis + classical eddy + excess).
* :mod:`reference_parameters` - source references and the (illustrative) reference parameter set.

``MODELS`` lists the model classes of this family for discovery by the engineering/API layers.
"""

from aipe_magnetics.models.physics.anhysteretic import (
    anhysteretic_magnetisation,
    anhysteretic_susceptibility,
    langevin,
    langevin_derivative,
)
from aipe_magnetics.models.physics.bertotti import BERTOTTI_MODEL_ID, BertottiLossModel
from aipe_magnetics.models.physics.jiles_atherton import (
    JA_MODEL_ID,
    JATrace,
    JilesAthertonModel,
    JilesAthertonODE,
    loop_metrics,
)
from aipe_magnetics.models.physics.reference_parameters import (
    BERTOTTI_1988,
    JA_1986,
    JA_ILLUSTRATIVE,
    JILES_1992_MINOR_LOOPS,
    JTD_1992,
    LAMINATION_EXAMPLE,
    REFERENCE_PARAMETER_SETS,
    JilesAthertonParameterSet,
    get_parameter_set,
)

MODELS = [JilesAthertonModel, BertottiLossModel]
MODEL_IDS = {JA_MODEL_ID: JilesAthertonModel, BERTOTTI_MODEL_ID: BertottiLossModel}

__all__ = [
    "MODELS", "MODEL_IDS",
    "anhysteretic_magnetisation", "anhysteretic_susceptibility", "langevin", "langevin_derivative",
    "BERTOTTI_MODEL_ID", "BertottiLossModel",
    "JA_MODEL_ID", "JATrace", "JilesAthertonModel", "JilesAthertonODE", "loop_metrics",
    "BERTOTTI_1988", "JA_1986", "JA_ILLUSTRATIVE", "JILES_1992_MINOR_LOOPS", "JTD_1992", "LAMINATION_EXAMPLE",
    "REFERENCE_PARAMETER_SETS", "JilesAthertonParameterSet", "get_parameter_set",
]
