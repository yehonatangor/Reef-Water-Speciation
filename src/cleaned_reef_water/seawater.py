"""Bulk seawater properties and conservative composition."""

from __future__ import annotations

import enum
from typing import Final

from .types import (
    ArrayLike,
    FloatArray,
    as_array,
    validate_salinity,
    validate_temperature,
)

__all__ = [
    "BoronFormulation",
    "CHLORINITY_PER_SALINITY",
    "density_seawater",
    "ionic_strength",
    "chlorinity",
    "total_borate",
    "total_sulfate",
    "total_fluoride",
    "total_calcium",
    "molar_to_molal",
    "molal_to_molar",
]

#: Salinity-to-chlorinity conversion, ``Cl = S / 1.80655`` (PSS-78 definition).
CHLORINITY_PER_SALINITY: Final[float] = 1.80655

#: Relative atomic mass of boron, IUPAC 2006 (Guide Chapter 5, Table 1).
_ATOMIC_MASS_BORON: Final[float] = 10.811

# --- Millero & Poisson (1981) equation-of-state coefficients ----------------
# Standard Mean Ocean Water density polynomial, equation (7).
_SMOW_COEFFS: Final[tuple[float, ...]] = (
    999.842594,
    6.793952e-2,
    -9.095290e-3,
    1.001685e-4,
    -1.120083e-6,
    6.536332e-9,
)
# Coefficient A of S^1, equation (8).
_A_COEFFS: Final[tuple[float, ...]] = (
    8.24493e-1,
    -4.0899e-3,
    7.6438e-5,
    -8.2467e-7,
    5.3875e-9,
)
# Coefficient B of S^1.5, equation (9).
_B_COEFFS: Final[tuple[float, ...]] = (-5.72466e-3, 1.0227e-4, -1.6546e-6)
# Coefficient C of S^2, equation (10).
_C_COEFF: Final[float] = 4.8314e-4


class BoronFormulation(enum.Enum):
    """Choice of boron-to-salinity proportionality constant.

    Attributes
    ----------
    UPPSTROM1974
        ``0.232 mg-B / kg / permil-Cl``.  The historical default used by
        CO2SYS and by the Guide to Best Practices.
    LEE2010
        ``0.2414 mg-B / kg / permil-Cl``.  Based on 139 samples spanning three
        ocean basins and argued by the authors to supersede Uppström.
    """

    UPPSTROM1974 = "uppstrom1974"
    LEE2010 = "lee2010"

    @property
    def boron_chlorinity_ratio(self) -> float:
        """Boron-to-chlorinity ratio in ``mg-B kg-soln^-1 permil^-1``."""
        return 0.232 if self is BoronFormulation.UPPSTROM1974 else 0.2414


def density_seawater(salinity: ArrayLike, t_c: ArrayLike) -> FloatArray:
    r"""Density of seawater at one atmosphere, in ``kg m^-3``.

    Parameters
    ----------
    salinity
        Practical salinity, valid for ``0 <= S <= 42``.
    t_c
        Temperature in degrees Celsius, valid for ``0 <= t <= 40``.

    Returns
    -------
    FloatArray
        Density in ``kg m^-3``.

    Examples
    --------
    The Guide to Best Practices quotes ``1023.343 kg m^-3`` at ``S = 35`` and
    ``t = 25 degC``:

    >>> float(round(density_seawater(35.0, 25.0), 3))
    1023.343
    """
    s = validate_salinity(salinity, low=0.0, high=42.0)
    t = as_array(t_c)
    validate_temperature(t, low=0.0, high=40.0)

    rho_smow = sum(c * t**i for i, c in enumerate(_SMOW_COEFFS))
    a = sum(c * t**i for i, c in enumerate(_A_COEFFS))
    b = sum(c * t**i for i, c in enumerate(_B_COEFFS))
    return rho_smow + a * s + b * s**1.5 + _C_COEFF * s**2


def chlorinity(salinity: ArrayLike) -> FloatArray:
    """Chlorinity in permil from practical salinity.

    Parameters
    ----------
    salinity
        Practical salinity.

    Returns
    -------
    FloatArray
        Chlorinity, ``Cl = S / 1.80655``.
    """
    return validate_salinity(salinity) / CHLORINITY_PER_SALINITY


def ionic_strength(salinity: ArrayLike) -> FloatArray:
    r"""Ionic strength of seawater in ``mol kg-H2O^-1``.

    Parameters
    ----------
    salinity
        Practical salinity.

    Returns
    -------
    FloatArray
        Ionic strength in ``mol kg-H2O^-1``.
    """
    s = validate_salinity(salinity)
    return 19.924 * s / (1000.0 - 1.005 * s)


def total_borate(
    salinity: ArrayLike,
    formulation: BoronFormulation = BoronFormulation.UPPSTROM1974,
) -> FloatArray:
    r"""Total dissolved boron in ``mol kg-soln^-1``.

    Parameters
    ----------
    salinity
        Practical salinity.
    formulation
        Which published boron-to-chlorinity ratio to use.

    Returns
    -------
    FloatArray
        Total borate in ``mol kg-soln^-1``.

    Examples
    --------
    >>> float(round(total_borate(35.0) * 1e6, 2))
    415.76
    >>> float(round(total_borate(35.0, BoronFormulation.LEE2010) * 1e6, 2))
    432.6
    """
    ratio = formulation.boron_chlorinity_ratio
    milligram_per_kg = ratio * chlorinity(salinity)
    return milligram_per_kg * 1e-3 / _ATOMIC_MASS_BORON


def total_sulfate(salinity: ArrayLike) -> FloatArray:
    r"""Total sulfate in ``mol kg-soln^-1``.

    Parameters
    ----------
    salinity
        Practical salinity.

    Returns
    -------
    FloatArray
        Total sulfate in ``mol kg-soln^-1``.

    Examples
    --------
    >>> float(round(total_sulfate(35.0) * 1e3, 4))
    28.2354
    """
    return (0.14 / 96.062) * chlorinity(salinity)


def total_fluoride(salinity: ArrayLike) -> FloatArray:
    r"""Total fluoride in ``mol kg-soln^-1``.

    Parameters
    ----------
    salinity
        Practical salinity.

    Returns
    -------
    FloatArray
        Total fluoride in ``mol kg-soln^-1``.

    Examples
    --------
    >>> float(round(total_fluoride(35.0) * 1e6, 2))
    68.32
    """
    return (6.7e-5 / 18.9984) * chlorinity(salinity)


def total_calcium(salinity: ArrayLike) -> FloatArray:
    r"""Total calcium in ``mol kg-soln^-1``.

    Parameters
    ----------
    salinity
        Practical salinity.

    Returns
    -------
    FloatArray
        Total calcium in ``mol kg-soln^-1``.

    Examples
    --------
    >>> float(round(total_calcium(35.0) * 1e3, 4))
    10.2846
    """
    return (0.02128 / 40.087) * chlorinity(salinity)


def molar_to_molal(
    concentration_mol_per_litre: ArrayLike,
    salinity: ArrayLike,
    t_c: ArrayLike,
) -> FloatArray:
    """Convert ``mol L^-1`` to ``mol kg-soln^-1``.

    Parameters
    ----------
    concentration_mol_per_litre
        Volumetric concentration in ``mol L^-1``.
    salinity
        Practical salinity of the solution.
    t_c
        Temperature in degrees Celsius at which the volume was measured.

    Returns
    -------
    FloatArray
        Gravimetric concentration in ``mol kg-soln^-1``.
    """
    rho_kg_per_l = density_seawater(salinity, t_c) / 1000.0
    return as_array(concentration_mol_per_litre) / rho_kg_per_l


def molal_to_molar(
    concentration_mol_per_kg: ArrayLike,
    salinity: ArrayLike,
    t_c: ArrayLike,
) -> FloatArray:
    """Convert ``mol kg-soln^-1`` to ``mol L^-1``.

    Parameters
    ----------
    concentration_mol_per_kg
        Gravimetric concentration in ``mol kg-soln^-1``.
    salinity
        Practical salinity of the solution.
    t_c
        Temperature in degrees Celsius.

    Returns
    -------
    FloatArray
        Volumetric concentration in ``mol L^-1``.
    """
    rho_kg_per_l = density_seawater(salinity, t_c) / 1000.0
    return as_array(concentration_mol_per_kg) * rho_kg_per_l
