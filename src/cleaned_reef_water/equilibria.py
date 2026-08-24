"""Stoichiometric equilibrium constants for the seawater acid--base system."""

from __future__ import annotations

import enum
import warnings
from typing import Final

import numpy as np

from .seawater import ionic_strength
from .types import (
    ArrayLike,
    FloatArray,
    PHScale,
    as_array,
    validate_salinity,
    validate_temperature,
)

__all__ = [
    "CarbonicFormulation",
    "k_co2_solubility",
    "k_bisulfate_free",
    "k_fluoride_free",
    "k_fluoride_total",
    "k_borate_total",
    "k_water_sws",
    "k_water_total",
    "k_carbonic_total",
    "k_phosphoric_sws",
    "k_phosphoric_total",
    "k_silicate_sws",
    "k_silicate_total",
    "k_calcite",
    "k_aragonite",
]


class CarbonicFormulation(enum.Enum):
    """Available parameterisations of ``K_1`` and ``K_2``.

    Attributes
    ----------
    LUEKER2000
        Lueker, Dickson & Keeling (2000), *Marine Chemistry* **70**, 105--119.
        A refit of the Mehrbach *et al.* (1973) data onto the total scale,
        validated against laboratory ``pCO2`` measurements.  **Recommended by
        the Guide and the package default.**  Valid ``19 <= S <= 43``,
        ``2 <= t <= 35 degC``.
    ROY1993
        Roy *et al.* (1993), *Marine Chemistry* **44**, 249--267.  Direct
        potentiometric determination on the total scale in synthetic seawater.
        Valid ``5 <= S <= 45``, ``0 <= t <= 45 degC``.
    MILLERO2006
        Millero *et al.* (2006), *Marine Chemistry* **100**, 80--94.  A fit
        spanning brackish to hypersaline water on the seawater scale.  Valid
        ``1 <= S <= 50``, ``0 <= t <= 50 degC``.
    DICKSON_MILLERO1987
        Dickson & Millero (1987), *Deep-Sea Research* **34**, 1733--1743,
        Table 5, ``0 <= S <= 40`` form, on the seawater scale.  Retained for
        backwards compatibility with the legacy code base.  **Note:** this
        parameterisation is internally inconsistent with the check value
        printed in the source paper (see ``docs/audit_report.md``); it is not
        recommended for new work.
    """

    LUEKER2000 = "lueker2000"
    ROY1993 = "roy1993"
    MILLERO2006 = "millero2006"
    DICKSON_MILLERO1987 = "dickson_millero1987"

    @property
    def native_scale(self) -> PHScale:
        """PH scale on which the published fit is defined."""
        if self in (CarbonicFormulation.LUEKER2000, CarbonicFormulation.ROY1993):
            return PHScale.TOTAL
        return PHScale.SEAWATER

    @property
    def salinity_range(self) -> tuple[float, float]:
        """Inclusive salinity validity range of the published fit."""
        return {
            CarbonicFormulation.LUEKER2000: (19.0, 43.0),
            CarbonicFormulation.ROY1993: (5.0, 45.0),
            CarbonicFormulation.MILLERO2006: (1.0, 50.0),
            CarbonicFormulation.DICKSON_MILLERO1987: (0.0, 40.0),
        }[self]

    @property
    def temperature_range(self) -> tuple[float, float]:
        """Inclusive temperature validity range in degrees Celsius."""
        return {
            CarbonicFormulation.LUEKER2000: (2.0, 35.0),
            CarbonicFormulation.ROY1993: (0.0, 45.0),
            CarbonicFormulation.MILLERO2006: (0.0, 50.0),
            CarbonicFormulation.DICKSON_MILLERO1987: (0.0, 45.0),
        }[self]


def _warn_if_out_of_range(
    salinity: FloatArray,
    t_c: FloatArray,
    s_range: tuple[float, float],
    t_range: tuple[float, float],
    label: str,
) -> None:
    """Emit a :class:`UserWarning` when inputs leave a fit's validity range."""
    if np.any(salinity < s_range[0]) or np.any(salinity > s_range[1]):
        warnings.warn(
            f"{label}: salinity outside fitted range {s_range}; extrapolating",
            UserWarning,
            stacklevel=3,
        )
    if np.any(t_c < t_range[0]) or np.any(t_c > t_range[1]):
        warnings.warn(
            f"{label}: temperature outside fitted range {t_range} degC; "
            "extrapolating",
            UserWarning,
            stacklevel=3,
        )


def k_co2_solubility(salinity: ArrayLike, t_c: ArrayLike) -> FloatArray:
    r"""Henry's law constant ``K_0`` for CO2, in ``mol kg-soln^-1 atm^-1``.

    Parameters
    ----------
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.

    Returns
    -------
    FloatArray
        ``K_0`` in ``mol kg-soln^-1 atm^-1``.  ``K_0`` is not an acid
        dissociation constant and carries no pH scale.

    Examples
    --------
    >>> import numpy as np
    >>> float(round(np.log(k_co2_solubility(35.0, 25.0)), 4))
    -3.5617
    """
    s = validate_salinity(salinity)
    temp = validate_temperature(t_c)
    t100 = temp / 100.0
    ln_k0 = (
        93.4517 / t100
        - 60.2409
        + 23.3585 * np.log(t100)
        + s * (0.023517 - 0.023656 * t100 + 0.0047036 * t100**2)
    )
    return np.exp(ln_k0)


def k_bisulfate_free(salinity: ArrayLike, t_c: ArrayLike) -> FloatArray:
    r"""Bisulfate dissociation constant ``K_S`` on the **free** scale.

    Parameters
    ----------
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.

    Returns
    -------
    FloatArray
        ``K_S`` in ``mol kg-soln^-1``, free scale.

    Examples
    --------
    >>> import numpy as np
    >>> float(round(np.log(k_bisulfate_free(35.0, 25.0)), 2))
    -2.3
    """
    s = validate_salinity(salinity)
    temp = validate_temperature(t_c)
    ln_t = np.log(temp)
    i = ionic_strength(s)
    sqrt_i = np.sqrt(i)

    ln_ks = (
        -4276.1 / temp
        + 141.328
        - 23.093 * ln_t
        + (-13856.0 / temp + 324.57 - 47.986 * ln_t) * sqrt_i
        + (35474.0 / temp - 771.54 + 114.723 * ln_t) * i
        - (2698.0 / temp) * i**1.5
        + (1776.0 / temp) * i**2
        + np.log(1.0 - 0.001005 * s)
    )
    return as_array(np.exp(ln_ks))


def k_fluoride_free(salinity: ArrayLike, t_c: ArrayLike) -> FloatArray:
    r"""Hydrogen fluoride dissociation constant ``K_F`` on the **free** scale.

    Parameters
    ----------
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.

    Returns
    -------
    FloatArray
        ``K_F`` in ``mol kg-soln^-1``, free scale.
    """
    s = validate_salinity(salinity)
    temp = validate_temperature(t_c)
    ln_kf = (
        1590.2 / temp
        - 12.641
        + 1.525 * np.sqrt(ionic_strength(s))
        + np.log(1.0 - 0.001005 * s)
    )
    return as_array(np.exp(ln_kf))


def k_fluoride_total(salinity: ArrayLike, t_c: ArrayLike) -> FloatArray:
    r"""Hydrogen fluoride dissociation constant ``K_F`` on the **total** scale.

    Parameters
    ----------
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.

    Returns
    -------
    FloatArray
        ``K_F`` in ``mol kg-soln^-1``, total scale.

    Examples
    --------
    >>> import numpy as np
    >>> float(round(np.log(k_fluoride_total(35.0, 25.0)), 2))
    -6.09
    """
    s = validate_salinity(salinity)
    temp = validate_temperature(t_c)
    return np.exp(874.0 / temp - 9.68 + 0.111 * np.sqrt(s))


def k_borate_total(salinity: ArrayLike, t_c: ArrayLike) -> FloatArray:
    r"""Boric acid dissociation constant ``K_B`` on the **total** scale.

    Parameters
    ----------
    salinity
        Practical salinity, valid ``0 <= S <= 45``.
    t_c
        Temperature in degrees Celsius, valid ``0 <= t <= 45``.

    Returns
    -------
    FloatArray
        ``K_B`` in ``mol kg-soln^-1``, total scale.

    Examples
    --------
    >>> import numpy as np
    >>> float(round(np.log(k_borate_total(35.0, 25.0)), 4))
    -19.7964
    """
    s = validate_salinity(salinity)
    t_celsius = as_array(t_c)
    temp = validate_temperature(t_celsius)
    _warn_if_out_of_range(s, t_celsius, (0.0, 45.0), (0.0, 45.0), "K_B (Dickson 1990b)")

    sqrt_s = np.sqrt(s)
    ln_kb = (
        (
            -8966.90
            - 2890.53 * sqrt_s
            - 77.942 * s
            + 1.728 * s**1.5
            - 0.0996 * s**2
        )
        / temp
        + 148.0248
        + 137.1942 * sqrt_s
        + 1.62142 * s
        + (-24.4344 - 25.085 * sqrt_s - 0.2474 * s) * np.log(temp)
        + 0.053105 * sqrt_s * temp
    )
    return as_array(np.exp(ln_kb))


def k_water_sws(salinity: ArrayLike, t_c: ArrayLike) -> FloatArray:
    r"""Ion product of water ``K_W`` on the **seawater** scale.

    Parameters
    ----------
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.

    Returns
    -------
    FloatArray
        ``K_W`` in ``(mol kg-soln^-1)^2``, seawater scale.
    """
    s = validate_salinity(salinity)
    temp = validate_temperature(t_c)
    ln_t = np.log(temp)
    ln_kw = (
        -13847.26 / temp
        + 148.9802
        - 23.6521 * ln_t
        + (118.67 / temp - 5.977 + 1.0495 * ln_t) * np.sqrt(s)
        - 0.01615 * s
    )
    return as_array(np.exp(ln_kw))


def k_water_total(salinity: ArrayLike, t_c: ArrayLike) -> FloatArray:
    r"""Ion product of water ``K_W`` on the **total** scale.

    Parameters
    ----------
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.

    Returns
    -------
    FloatArray
        ``K_W`` in ``(mol kg-soln^-1)^2``, total scale.

    Examples
    --------
    >>> import numpy as np
    >>> float(round(np.log(k_water_total(35.0, 25.0)), 4))
    -30.4411
    """
    from .scales import convert_constant

    return convert_constant(
        k_water_sws(salinity, t_c), salinity, t_c, PHScale.SEAWATER, PHScale.TOTAL
    )


def _k_carbonic_lueker2000(
    s: FloatArray, temp: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Lueker *et al.* (2000) ``K_1``/``K_2``, total scale."""
    ln_t = np.log(temp)
    log10_k1 = (
        -3633.86 / temp + 61.2172 - 9.67770 * ln_t + 0.011555 * s - 0.0001152 * s**2
    )
    log10_k2 = (
        -471.78 / temp - 25.9290 + 3.16967 * ln_t + 0.01781 * s - 0.0001122 * s**2
    )
    return 10.0**log10_k1, 10.0**log10_k2


def _k_carbonic_roy1993(
    s: FloatArray, temp: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Roy *et al.* (1993) ``K_1``/``K_2``, total scale, mol kg-soln^-1."""
    ln_t = np.log(temp)
    sqrt_s = np.sqrt(s)
    ln_k1 = (
        2.83655
        - 2307.1266 / temp
        - 1.5529413 * ln_t
        + (-0.20760841 - 4.0484 / temp) * sqrt_s
        + 0.08468345 * s
        - 0.00654208 * s**1.5
        + np.log(1.0 - 0.001005 * s)
    )
    ln_k2 = (
        -9.226508
        - 3351.6106 / temp
        - 0.2005743 * ln_t
        + (-0.106901773 - 23.9722 / temp) * sqrt_s
        + 0.1130822 * s
        - 0.00846934 * s**1.5
        + np.log(1.0 - 0.001005 * s)
    )
    return np.exp(ln_k1), np.exp(ln_k2)


def _k_carbonic_millero2006(
    s: FloatArray, temp: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Millero *et al.* (2006) ``K_1``/``K_2``, seawater scale."""
    sqrt_s = np.sqrt(s)
    ln_t = np.log(temp)
    pk1_0 = -126.34048 + 6320.813 / temp + 19.568224 * ln_t
    a1 = 13.4191 * sqrt_s + 0.0331 * s - 5.33e-5 * s**2
    b1 = -530.123 * sqrt_s - 6.103 * s
    c1 = -2.06950 * sqrt_s
    pk1 = a1 + b1 / temp + c1 * ln_t + pk1_0

    pk2_0 = -90.18333 + 5143.692 / temp + 14.613358 * ln_t
    a2 = 21.0894 * sqrt_s + 0.1248 * s - 3.687e-4 * s**2
    b2 = -772.483 * sqrt_s - 20.051 * s
    c2 = -3.3336 * sqrt_s
    pk2 = a2 + b2 / temp + c2 * ln_t + pk2_0
    return 10.0**-pk1, 10.0**-pk2


def _k_carbonic_dm1987(
    s: FloatArray, temp: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Dickson & Millero (1987) Table 5 ``0 <= S <= 40`` form, seawater scale."""
    ln_t = np.log(temp)
    sqrt_s = np.sqrt(s)
    pk1_0 = 6320.81 / temp - 126.3405 + 19.568 * ln_t
    pk2_0 = 5143.69 / temp - 90.1833 + 14.613 * ln_t
    pk1 = pk1_0 + (-840.39 / temp + 19.894 - 3.0189 * ln_t) * sqrt_s + 0.00668 * s
    pk2 = pk2_0 + (-690.59 / temp + 17.176 - 2.6719 * ln_t) * sqrt_s + 0.0217 * s
    return 10.0**-pk1, 10.0**-pk2


_CARBONIC_DISPATCH: Final[dict[CarbonicFormulation, object]] = {
    CarbonicFormulation.LUEKER2000: _k_carbonic_lueker2000,
    CarbonicFormulation.ROY1993: _k_carbonic_roy1993,
    CarbonicFormulation.MILLERO2006: _k_carbonic_millero2006,
    CarbonicFormulation.DICKSON_MILLERO1987: _k_carbonic_dm1987,
}


def k_carbonic_total(
    salinity: ArrayLike,
    t_c: ArrayLike,
    formulation: CarbonicFormulation = CarbonicFormulation.LUEKER2000,
) -> tuple[FloatArray, FloatArray]:
    """Carbonic acid constants ``K_1`` and ``K_2`` on the **total** scale.

    Parameters
    ----------
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.
    formulation
        Which published parameterisation to use.

    Returns
    -------
    tuple of FloatArray
        ``(K_1, K_2)`` in ``mol kg-soln^-1``, total scale.

    Examples
    --------
    >>> import numpy as np
    >>> k1, k2 = k_carbonic_total(35.0, 25.0)
    >>> float(round(np.log10(k1), 4))
    -5.8472
    >>> float(round(np.log10(k2), 4))
    -8.966
    """
    from .scales import convert_constant  # local import avoids a cycle

    s = validate_salinity(salinity)
    t_celsius = as_array(t_c)
    temp = validate_temperature(t_celsius)
    _warn_if_out_of_range(
        s,
        t_celsius,
        formulation.salinity_range,
        formulation.temperature_range,
        f"K_1/K_2 ({formulation.value})",
    )

    k1, k2 = _CARBONIC_DISPATCH[formulation](s, temp)  # type: ignore[operator]
    if formulation.native_scale is PHScale.TOTAL:
        return k1, k2
    return (
        convert_constant(k1, s, t_celsius, formulation.native_scale, PHScale.TOTAL),
        convert_constant(k2, s, t_celsius, formulation.native_scale, PHScale.TOTAL),
    )


def k_phosphoric_sws(
    salinity: ArrayLike, t_c: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Phosphoric acid constants on the **seawater** scale.

    Parameters
    ----------
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.

    Returns
    -------
    tuple of FloatArray
        ``(K_1P, K_2P, K_3P)`` in ``mol kg-soln^-1``, seawater scale.
    """
    s = validate_salinity(salinity)
    temp = validate_temperature(t_c)
    ln_t = np.log(temp)
    sqrt_s = np.sqrt(s)

    ln_k1p = (
        -4576.752 / temp
        + 115.54
        - 18.453 * ln_t
        + (-106.736 / temp + 0.69171) * sqrt_s
        + (-0.65643 / temp - 0.01844) * s
    )
    ln_k2p = (
        -8814.715 / temp
        + 172.1033
        - 27.927 * ln_t
        + (-160.340 / temp + 1.3566) * sqrt_s
        + (0.37335 / temp - 0.05778) * s
    )
    ln_k3p = (
        -3070.75 / temp
        - 18.126
        + (17.27039 / temp + 2.81197) * sqrt_s
        + (-44.99486 / temp - 0.09984) * s
    )
    return np.exp(ln_k1p), np.exp(ln_k2p), np.exp(ln_k3p)


def k_phosphoric_total(
    salinity: ArrayLike, t_c: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Phosphoric acid constants ``K_1P``, ``K_2P``, ``K_3P``, **total** scale.

    Parameters
    ----------
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.

    Returns
    -------
    tuple of FloatArray
        ``(K_1P, K_2P, K_3P)`` in ``mol kg-soln^-1``, total scale.

    Examples
    --------
    >>> import numpy as np
    >>> k1p, k2p, k3p = k_phosphoric_total(35.0, 25.0)
    >>> [float(round(np.log(k), 3)) for k in (k1p, k2p, k3p)]
    [-3.719, -13.735, -20.245]
    """
    from .scales import convert_constant

    return tuple(  # type: ignore[return-value]
        convert_constant(k, salinity, t_c, PHScale.SEAWATER, PHScale.TOTAL)
        for k in k_phosphoric_sws(salinity, t_c)
    )


def k_silicate_sws(salinity: ArrayLike, t_c: ArrayLike) -> FloatArray:
    """Silicic acid first dissociation constant ``K_Si``, **seawater** scale.

    Parameters
    ----------
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.

    Returns
    -------
    FloatArray
        ``K_Si`` in ``mol kg-soln^-1``, seawater scale.
    """
    s = validate_salinity(salinity)
    temp = validate_temperature(t_c)
    i = ionic_strength(s)
    ln_t = np.log(temp)
    ln_ksi = (
        -8904.2 / temp
        + 117.40
        - 19.334 * ln_t
        + (-458.79 / temp + 3.5913) * np.sqrt(i)
        + (188.74 / temp - 1.5998) * i
        + (-12.1652 / temp + 0.07871) * i**2
        + np.log(1.0 - 0.001005 * s)
    )
    return as_array(np.exp(ln_ksi))


def k_silicate_total(salinity: ArrayLike, t_c: ArrayLike) -> FloatArray:
    """Silicic acid first dissociation constant ``K_Si``, **total** scale.

    Parameters
    ----------
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.

    Returns
    -------
    FloatArray
        ``K_Si`` in ``mol kg-soln^-1``, total scale.

    Examples
    --------
    >>> import numpy as np
    >>> float(round(np.log(k_silicate_total(35.0, 25.0)), 3))
    -21.614
    """
    from .scales import convert_constant

    return convert_constant(
        k_silicate_sws(salinity, t_c), salinity, t_c, PHScale.SEAWATER, PHScale.TOTAL
    )


def _k_solubility(
    salinity: ArrayLike,
    t_c: ArrayLike,
    log_k0_a: float,
    log_k0_c: float,
    b0: float,
    b1: float,
    b2: float,
    c0: float,
    d0: float,
) -> FloatArray:
    """Shared implementation of the Mucci (1983) solubility products."""
    s = validate_salinity(salinity)
    t_celsius = as_array(t_c)
    temp = validate_temperature(t_celsius)
    _warn_if_out_of_range(s, t_celsius, (5.0, 44.0), (5.0, 40.0), "K_sp (Mucci 1983)")

    log_k0 = log_k0_a - 0.077993 * temp + log_k0_c / temp + 71.595 * np.log10(temp)
    log_ksp = (
        log_k0
        + (b0 + b1 * temp + b2 / temp) * np.sqrt(s)
        + c0 * s
        + d0 * s**1.5
    )
    return as_array(10.0**log_ksp)


def k_calcite(salinity: ArrayLike, t_c: ArrayLike) -> FloatArray:
    r"""Stoichiometric solubility product of calcite, ``(mol kg-soln^-1)^2``.

    Parameters
    ----------
    salinity
        Practical salinity, valid ``5 <= S <= 44``.
    t_c
        Temperature in degrees Celsius, valid ``5 <= t <= 40``.

    Returns
    -------
    FloatArray
        ``K_sp`` for calcite in ``(mol kg-soln^-1)^2``.
    """
    return _k_solubility(
        salinity,
        t_c,
        -171.9065,
        2839.319,
        -0.77712,
        2.8426e-3,
        178.34,
        -0.07711,
        4.1249e-3,
    )


def k_aragonite(salinity: ArrayLike, t_c: ArrayLike) -> FloatArray:
    r"""Stoichiometric solubility product of aragonite, ``(mol kg-soln^-1)^2``.

    Parameters
    ----------
    salinity
        Practical salinity, valid ``5 <= S <= 44``.
    t_c
        Temperature in degrees Celsius, valid ``5 <= t <= 40``.

    Returns
    -------
    FloatArray
        ``K_sp`` for aragonite in ``(mol kg-soln^-1)^2``.
    """
    return _k_solubility(
        salinity,
        t_c,
        -171.945,
        2903.293,
        -0.068393,
        1.7276e-3,
        88.135,
        -0.10018,
        5.9415e-3,
    )
