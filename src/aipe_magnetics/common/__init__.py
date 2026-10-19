"""Shared primitives: errors, units, quantities, result envelopes, deterministic JSON I/O."""

from aipe_magnetics.common.errors import (
    AIPEMagneticsError,
    AuthorizationError,
    ConvergenceError,
    DataIntegrityError,
    DataNotFoundError,
    InvalidInputError,
    LicenseRestrictionError,
    NumericalError,
    OutOfDomainError,
    UnitError,
    UnsupportedOperationError,
)
from aipe_magnetics.common.quantity import Quantity, Uncertainty, require_dimension, require_nonnegative, require_positive
from aipe_magnetics.common.results import (
    Assumption,
    EngineeringResult,
    ErrorRecord,
    Limitation,
    ModelIdentity,
    ResultValue,
    SourceReference,
    ToolIdentity,
    VerificationStatus,
    Warning_,
    tool_identity,
)
from aipe_magnetics.common.units import MU_0, UnitRegistry, convert, si_unit_of, to_si, units

__all__ = [
    "AIPEMagneticsError", "AuthorizationError", "ConvergenceError", "DataIntegrityError", "DataNotFoundError",
    "InvalidInputError", "LicenseRestrictionError", "NumericalError", "OutOfDomainError", "UnitError",
    "UnsupportedOperationError",
    "Quantity", "Uncertainty", "require_dimension", "require_nonnegative", "require_positive",
    "Assumption", "EngineeringResult", "ErrorRecord", "Limitation", "ModelIdentity", "ResultValue",
    "SourceReference", "ToolIdentity", "VerificationStatus", "Warning_", "tool_identity",
    "MU_0", "UnitRegistry", "convert", "si_unit_of", "to_si", "units",
]
