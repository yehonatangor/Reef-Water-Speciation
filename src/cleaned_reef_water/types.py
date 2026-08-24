"""Shared type aliases, enumerations and validation helpers."""

from __future__ import annotations

import enum
from typing import Final, TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = [
    "ArrayLike",
    "FloatArray",
    "PHScale",
    "ABSOLUTE_ZERO_C",
    "KELVIN_OFFSET",
    "GAS_CONSTANT",
    "FARADAY",
    "ATM_IN_PA",
    "LN10",
    "as_array",
    "validate_salinity",
    "validate_temperature",
]

#: Scalar or array input accepted by the public API.
ArrayLike: TypeAlias = float | int | npt.NDArray[np.floating]

#: Canonical floating point array type returned by the public API.
FloatArray: TypeAlias = npt.NDArray[np.float64]

#: Offset between the Celsius and Kelvin scales (exact, by definition).
KELVIN_OFFSET: Final[float] = 273.15

#: Absolute zero expressed in degrees Celsius.
ABSOLUTE_ZERO_C: Final[float] = -273.15

#: Molar gas constant, J K^-1 mol^-1 (CODATA, quoted in Guide Chapter 5 §1).
GAS_CONSTANT: Final[float] = 8.314472

#: Faraday constant, C mol^-1 (CODATA, quoted in Guide Chapter 5 §1).
FARADAY: Final[float] = 96485.3399

#: One standard atmosphere expressed in pascal (exact, by definition).
ATM_IN_PA: Final[float] = 101325.0

#: Natural logarithm of ten, cached because it appears in every pK conversion.
LN10: Final[float] = float(np.log(10.0))


class PHScale(enum.Enum):
    r"""The four pH scales in routine use in chemical oceanography.

    Attributes
    ----------
    FREE
        Free hydrogen ion concentration scale.
    TOTAL
        Total hydrogen ion scale (free plus bisulfate).  This is the package
        default and the scale recommended by Dickson, Sabine & Christian
        (2007), *Guide to Best Practices for Ocean CO2 Measurements*.
    SEAWATER
        Seawater scale (free plus bisulfate plus hydrogen fluoride).
    NBS
        NBS/NIST activity scale.  Not a concentration scale.
    """

    FREE = "free"
    TOTAL = "total"
    SEAWATER = "sws"
    NBS = "nbs"

    @property
    def is_concentration_scale(self) -> bool:
        """Return ``True`` for the three thermodynamic concentration scales."""
        return self is not PHScale.NBS

    def __str__(self) -> str:  # pragma: no cover - trivial
        """Return the short scale identifier, e.g. ``"total"``."""
        return self.value


def as_array(value: ArrayLike) -> FloatArray:
    """Coerce ``value`` to a contiguous float64 array.

    Parameters
    ----------
    value
        Scalar or array-like numeric input.

    Returns
    -------
    FloatArray
        A ``float64`` array.  Scalars become zero-dimensional arrays, which
        broadcast correctly against every other input and preserve scalar
        semantics when the caller extracts ``.item()``.

    Raises
    ------
    TypeError
        If ``value`` cannot be interpreted as a floating point array.
    """
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError(f"expected numeric input, got {type(value)!r}") from exc
    return array


def validate_salinity(
    salinity: ArrayLike,
    *,
    low: float = 0.0,
    high: float = 50.0,
    name: str = "salinity",
) -> FloatArray:
    """Validate and coerce a practical salinity.

    Parameters
    ----------
    salinity
        Practical salinity (PSS-78), dimensionless.
    low, high
        Inclusive bounds outside which a :class:`ValueError` is raised.  The
        default range is deliberately wider than the validity range of any
        individual equilibrium constant; per-constant range checking is the
        responsibility of the individual constant functions, which emit
        warnings rather than errors.
    name
        Name used in the error message.

    Returns
    -------
    FloatArray
        The validated salinity as a float64 array.

    Raises
    ------
    ValueError
        If any element is non-finite or outside ``[low, high]``.
    """
    array = as_array(salinity)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(array < low) or np.any(array > high):
        raise ValueError(
            f"{name} out of range: expected {low} <= S <= {high}, "
            f"got min={float(np.min(array))}, max={float(np.max(array))}"
        )
    return array


def validate_temperature(
    t_c: ArrayLike,
    *,
    low: float = -2.0,
    high: float = 50.0,
    name: str = "temperature",
) -> FloatArray:
    """Validate a Celsius temperature and return it in kelvin.

    Parameters
    ----------
    t_c
        Temperature in degrees Celsius.
    low, high
        Inclusive bounds in degrees Celsius.
    name
        Name used in the error message.

    Returns
    -------
    FloatArray
        Absolute temperature in kelvin.

    Raises
    ------
    ValueError
        If any element is non-finite or outside ``[low, high]``.
    """
    array = as_array(t_c)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(array < low) or np.any(array > high):
        raise ValueError(
            f"{name} out of range: expected {low} <= t/degC <= {high}, "
            f"got min={float(np.min(array))}, max={float(np.max(array))}"
        )
    return array + KELVIN_OFFSET
