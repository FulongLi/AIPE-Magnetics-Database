"""Explicit error hierarchy.

Design rules:

* Invalid engineering inputs are *rejected*, never silently clipped or modified
  (:class:`InvalidInputError`, :class:`OutOfDomainError`).
* Missing data is reported as missing (:class:`DataNotFoundError`), never guessed.
* Numerical failures are surfaced (:class:`ConvergenceError`, :class:`NumericalError`).
* Rights problems are explicit (:class:`LicenseRestrictionError`).
"""

from __future__ import annotations

from typing import Any


class AIPEMagneticsError(Exception):
    """Base class for all package errors. ``code`` is a stable machine-readable identifier."""

    code = "aipe_magnetics_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class InvalidInputError(AIPEMagneticsError, ValueError):
    """An input is missing, has the wrong type/dimension, or is physically impossible (e.g. negative frequency)."""

    code = "invalid_input"


class UnitError(InvalidInputError):
    """A unit is unknown or dimensionally incompatible with the requested conversion."""

    code = "unit_error"


class OutOfDomainError(AIPEMagneticsError, ValueError):
    """Inputs are valid but outside the declared validity domain of a model or dataset.

    Raised when the domain policy is ``reject``. With policy ``warn`` the evaluation proceeds and the
    result carries an out-of-range verification status instead.
    """

    code = "out_of_domain"


class UnsupportedOperationError(AIPEMagneticsError, NotImplementedError):
    """The requested capability is declared but not implemented for these inputs (never a silent fallback)."""

    code = "unsupported_operation"


class DataNotFoundError(AIPEMagneticsError, LookupError):
    """A record, material, core or dataset could not be found."""

    code = "data_not_found"


class DataIntegrityError(AIPEMagneticsError, ValueError):
    """Checksum mismatch, schema violation, duplicate identifier or inconsistent dataset metadata."""

    code = "data_integrity"


class LicenseRestrictionError(AIPEMagneticsError, PermissionError):
    """The requested redistribution or bundling is not permitted by the source licence."""

    code = "license_restriction"


class NumericalError(AIPEMagneticsError, ArithmeticError):
    """Overflow, NaN/Inf or a nonphysical numerical result."""

    code = "numerical_error"


class ConvergenceError(NumericalError):
    """An iterative or time-stepping method did not converge to the requested tolerance."""

    code = "convergence_error"


class AuthorizationError(AIPEMagneticsError, PermissionError):
    """An operation that modifies user files was requested without explicit authorization."""

    code = "authorization_required"
