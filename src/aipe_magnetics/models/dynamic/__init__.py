"""Dynamic / ODE model family: excitation, ODE contract, solver, convergence and steady-state tools.

Public API::

    from aipe_magnetics.models.dynamic import (
        Excitation, ODEModel, SolverConfig, ODESolution, solve, convergence_study, periodic_steady_state,
        cycle_energy_density, trapezoid, cumulative_trapezoid, LinearRL, NonlinearInductor,
    )

All mathematics is callable without any language model; see the module docstrings for the exact
equations and conventions.
"""

from aipe_magnetics.models.dynamic.excitation import DEFAULT_CLOSURE_TOLERANCE, EXCITATION_KINDS, Excitation
from aipe_magnetics.models.dynamic.examples import FiniteTimeBlowUp, LinearRL, NonlinearInductor
from aipe_magnetics.models.dynamic.ode import (
    RATIO_FREE_FRACTION,
    SOLVER_METHODS,
    ODEModel,
    PeriodicSteadyState,
    SolverConfig,
    convergence_study,
    periodic_steady_state,
    solve,
)
from aipe_magnetics.models.dynamic.solution import (
    ODESolution,
    cumulative_trapezoid,
    cycle_energy_density,
    interpolated_crossings,
    loop_closure_error,
    trapezoid,
)

__all__ = [
    "DEFAULT_CLOSURE_TOLERANCE", "EXCITATION_KINDS", "Excitation",
    "RATIO_FREE_FRACTION", "SOLVER_METHODS", "ODEModel", "PeriodicSteadyState", "SolverConfig", "convergence_study",
    "periodic_steady_state", "solve",
    "ODESolution", "cumulative_trapezoid", "cycle_energy_density", "interpolated_crossings", "loop_closure_error",
    "trapezoid",
    "FiniteTimeBlowUp", "LinearRL", "NonlinearInductor",
]
