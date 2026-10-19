"""Langevin anhysteretic magnetisation used by the Jiles-Atherton model.

Definitions (Jiles & Atherton 1986, eq. for the anhysteretic magnetisation)::

    M_an(He) = Ms * L(He / a),      L(x) = coth(x) - 1/x          (Langevin function)
    dM_an/dHe = (Ms / a) * L'(x),   L'(x) = 1/x^2 - 1/sinh(x)^2

with ``Ms`` the saturation magnetisation (A/m), ``a`` the anhysteretic shape parameter (A/m) and
``He = H + alpha*M`` the effective field (A/m).

Numerical evaluation
--------------------
* ``coth(x) - 1/x`` suffers catastrophic cancellation for small ``|x|``.  For ``|x| < 0.05`` the
  Taylor series is used::

      L(x)  = x/3 - x^3/45 + 2x^5/945 - x^7/4725 + 2x^9/93555 + O(x^11)
      L'(x) = 1/3 - x^2/15 + 2x^4/189 - x^6/675 + 2x^8/10395 + O(x^10)

  The three-term series ``x/3 - x^3/45 + 2x^5/945`` requested for ``|x| < 1e-3`` is contained in it;
  the truncation error of the five-term series at ``|x| = 0.05`` is below 1e-20 (next term
  ``1382 x^11 / 638512875``), far below double precision round-off of the direct formula there.
* ``coth(x)`` is evaluated as ``1/tanh(x)``, which cannot overflow; ``1/sinh(x)^2`` is replaced by
  ``0`` for ``|x| > 20`` where it is below 1.7e-17 and numerically irrelevant compared with ``1/x^2``.
* Limits: ``L(x) -> +-1`` and ``L'(x) -> 0`` for ``|x| -> inf``; ``L(0) = 0`` and ``L'(0) = 1/3`` so the
  initial anhysteretic susceptibility is ``dM_an/dHe(0) = Ms / (3a)``.

Scalar helpers (``langevin_scalar`` and ``langevin_derivative_scalar``) use ``math`` for speed inside
ODE right-hand sides; the array functions are vectorised with numpy and return floats for scalar
inputs.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from aipe_magnetics.common.errors import InvalidInputError

__all__ = ["LANGEVIN_SERIES_THRESHOLD", "langevin", "langevin_derivative", "langevin_scalar",
           "langevin_derivative_scalar", "anhysteretic_magnetisation", "anhysteretic_susceptibility"]

LANGEVIN_SERIES_THRESHOLD = 0.05
_LARGE = 20.0


def langevin_scalar(x: float) -> float:
    """``L(x) = coth(x) - 1/x`` for a Python float (series for small ``|x|``)."""
    ax = abs(x)
    if ax < LANGEVIN_SERIES_THRESHOLD:
        x2 = x * x
        return x * (1.0 / 3.0 + x2 * (-1.0 / 45.0 + x2 * (2.0 / 945.0 + x2 * (-1.0 / 4725.0 + x2 * (2.0 / 93555.0)))))
    return 1.0 / math.tanh(x) - 1.0 / x


def langevin_derivative_scalar(x: float) -> float:
    """``L'(x) = 1/x^2 - 1/sinh(x)^2`` for a Python float (series for small ``|x|``)."""
    ax = abs(x)
    if ax < LANGEVIN_SERIES_THRESHOLD:
        x2 = x * x
        return 1.0 / 3.0 + x2 * (-1.0 / 15.0 + x2 * (2.0 / 189.0 + x2 * (-1.0 / 675.0 + x2 * (2.0 / 10395.0))))
    inv_x2 = 1.0 / (x * x)
    if ax > _LARGE:
        return inv_x2
    s = math.sinh(x)
    return inv_x2 - 1.0 / (s * s)


def _prepare(x: Any) -> tuple[np.ndarray, bool]:
    scalar = np.ndim(x) == 0
    arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise InvalidInputError("Langevin argument must be finite")
    return arr, scalar


def langevin(x: Any) -> Any:
    """Vectorised ``L(x)``; returns a float for scalar input."""
    arr, scalar = _prepare(x)
    out = np.empty_like(arr)
    small = np.abs(arr) < LANGEVIN_SERIES_THRESHOLD
    xs = arr[small]
    x2 = xs * xs
    out[small] = xs * (1.0 / 3.0 + x2 * (-1.0 / 45.0 + x2 * (2.0 / 945.0 + x2 * (-1.0 / 4725.0 + x2 * (2.0 / 93555.0)))))
    xl = arr[~small]
    out[~small] = 1.0 / np.tanh(xl) - 1.0 / xl
    return float(out) if scalar else out


def langevin_derivative(x: Any) -> Any:
    """Vectorised ``L'(x)``; returns a float for scalar input."""
    arr, scalar = _prepare(x)
    out = np.empty_like(arr)
    small = np.abs(arr) < LANGEVIN_SERIES_THRESHOLD
    xs = arr[small]
    x2 = xs * xs
    out[small] = 1.0 / 3.0 + x2 * (-1.0 / 15.0 + x2 * (2.0 / 189.0 + x2 * (-1.0 / 675.0 + x2 * (2.0 / 10395.0))))
    mid = (~small) & (np.abs(arr) <= _LARGE)
    xm = arr[mid]
    out[mid] = 1.0 / (xm * xm) - 1.0 / np.sinh(xm) ** 2
    large = np.abs(arr) > _LARGE
    xb = arr[large]
    out[large] = 1.0 / (xb * xb)
    return float(out) if scalar else out


def _check_params(Ms: float, a: float) -> tuple[float, float]:
    for name, value in (("Ms", Ms), ("a", a)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise InvalidInputError(f"{name} must be a real number", details={"name": name})
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise InvalidInputError(f"{name} must be finite and positive, got {value}", details={"name": name})
    return float(Ms), float(a)


def anhysteretic_magnetisation(He: Any, Ms: float, a: float) -> Any:
    """``M_an(He) = Ms * L(He/a)`` in A/m."""
    Ms, a = _check_params(Ms, a)
    arr, scalar = _prepare(He)
    out = Ms * langevin(arr / a)
    return float(out) if scalar else out


def anhysteretic_susceptibility(He: Any, Ms: float, a: float) -> Any:
    """``dM_an/dHe = (Ms/a) * L'(He/a)`` (dimensionless)."""
    Ms, a = _check_params(Ms, a)
    arr, scalar = _prepare(He)
    out = (Ms / a) * langevin_derivative(arr / a)
    return float(out) if scalar else out
