r"""Conversion between the free, total and seawater pH scales."""

from __future__ import annotations

import numpy as np

from .equilibria import k_bisulfate_free, k_fluoride_free
from .seawater import total_fluoride, total_sulfate
from .types import ArrayLike, FloatArray, PHScale, as_array

__all__ = [
    "free_to_total_factor",
    "free_to_sws_factor",
    "scale_factor",
    "convert_ph",
    "convert_constant",
]


def free_to_total_factor(salinity: ArrayLike, t_c: ArrayLike) -> FloatArray:
    """Return ``1 + S_T / K_S``.

    Parameters
    ----------
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.

    Returns
    -------
    FloatArray
        The dimensionless conversion factor.

    Examples
    --------
    >>> float(round(free_to_total_factor(35.0, 25.0), 4))
    1.2815

    The corresponding pH offset is ``log10(1.2815) = 0.108``, the familiar
    difference of about 0.11 between free-scale and total-scale pH at
    ``S = 35``.
    """
    return 1.0 + total_sulfate(salinity) / k_bisulfate_free(salinity, t_c)


def free_to_sws_factor(salinity: ArrayLike, t_c: ArrayLike) -> FloatArray:
    """Return ``1 + S_T / K_S + F_T / K_F``.

    Parameters
    ----------
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.

    Returns
    -------
    FloatArray
        The dimensionless conversion factor.

    Examples
    --------
    >>> float(round(free_to_sws_factor(35.0, 25.0), 4))
    1.3104

    The seawater and total scales differ by only
    ``log10(1.3104 / 1.2815) = 0.0097`` pH units, because hydrogen fluoride is
    a very minor proton acceptor compared with sulfate.
    """
    return free_to_total_factor(salinity, t_c) + total_fluoride(salinity) / (
        k_fluoride_free(salinity, t_c)
    )


def scale_factor(
    salinity: ArrayLike,
    t_c: ArrayLike,
    source: PHScale,
    target: PHScale,
) -> FloatArray:
    """Multiplicative factor converting ``[H+]`` from ``source`` to ``target``.

    Parameters
    ----------
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.
    source, target
        The pH scales to convert between.  Both must be concentration scales.

    Returns
    -------
    FloatArray
        Factor ``f`` such that ``[H+]_target = f * [H+]_source``.

    Raises
    ------
    ValueError
        If either scale is :attr:`~cleaned_reef_water.types.PHScale.NBS`.

    Examples
    --------
    Converting a seawater-scale constant to the total scale reduces it
    slightly, because the seawater scale counts hydrogen fluoride as well:

    >>> f = scale_factor(35.0, 25.0, PHScale.SEAWATER, PHScale.TOTAL)
    >>> round(float(f), 4)
    0.978

    Note that the return value is always an array (zero-dimensional for scalar
    input), consistent with the declared return type.  Call :func:`float` on it
    before using scalar-only operations such as :func:`round`.
    """
    for scale in (source, target):
        if not scale.is_concentration_scale:
            raise ValueError(
                "cannot convert to or from the NBS activity scale without an "
                "explicit total activity coefficient f_H; use a concentration "
                "scale (free, total or sws) instead"
            )
    unity: FloatArray = np.ones_like(as_array(salinity) + as_array(t_c))
    if source is target:
        return unity

    def to_free_factor(scale: PHScale) -> FloatArray:
        """Factor converting ``[H+]`` on ``scale`` to the free scale."""
        if scale is PHScale.FREE:
            return unity
        if scale is PHScale.TOTAL:
            return as_array(1.0 / free_to_total_factor(salinity, t_c))
        return as_array(1.0 / free_to_sws_factor(salinity, t_c))

    def from_free_factor(scale: PHScale) -> FloatArray:
        """Factor converting ``[H+]`` on the free scale to ``scale``."""
        if scale is PHScale.FREE:
            return unity
        if scale is PHScale.TOTAL:
            return free_to_total_factor(salinity, t_c)
        return free_to_sws_factor(salinity, t_c)

    return as_array(to_free_factor(source) * from_free_factor(target))


def convert_ph(
    ph: ArrayLike,
    salinity: ArrayLike,
    t_c: ArrayLike,
    source: PHScale,
    target: PHScale,
) -> FloatArray:
    """Convert a pH value between concentration scales.

    Parameters
    ----------
    ph
        pH on the ``source`` scale.
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.
    source, target
        The pH scales to convert between.

    Returns
    -------
    FloatArray
        pH on the ``target`` scale.

    Examples
    --------
    Free-scale pH is always numerically *higher* than total-scale pH, because
    the free scale counts fewer protons:

    >>> ph_total = convert_ph(8.1, 35.0, 25.0, PHScale.FREE, PHScale.TOTAL)
    >>> bool(ph_total < 8.1)
    True
    """
    factor = scale_factor(salinity, t_c, source, target)
    return as_array(as_array(ph) - np.log10(factor))


def convert_constant(
    constant: ArrayLike,
    salinity: ArrayLike,
    t_c: ArrayLike,
    source: PHScale,
    target: PHScale,
) -> FloatArray:
    """Convert an equilibrium constant between concentration scales.

    Parameters
    ----------
    constant
        An equilibrium constant carrying exactly one power of ``[H+]``, on the
        ``source`` scale.
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.
    source, target
        The pH scales to convert between.

    Returns
    -------
    FloatArray
        The constant on the ``target`` scale.

    Examples
    --------
    >>> import numpy as np
    >>> from cleaned_reef_water.equilibria import k_borate_total
    >>> kb_total = k_borate_total(35.0, 25.0)
    >>> kb_sws = convert_constant(
    ...     kb_total, 35.0, 25.0, PHScale.TOTAL, PHScale.SEAWATER)
    >>> bool(kb_sws > kb_total)
    True
    """
    return as_array(constant) * scale_factor(salinity, t_c, source, target)
