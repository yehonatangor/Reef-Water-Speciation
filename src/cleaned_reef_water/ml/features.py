r"""Physical summary statistics of a titration curve, for the encoder."""

from __future__ import annotations

from typing import Final

import numpy as np

from ..types import FloatArray

__all__ = ["FEATURE_NAMES", "PH_THRESHOLDS", "curve_features"]

#: pH values at which the titrant mass is interpolated.
PH_THRESHOLDS: Final[tuple[float, ...]] = (7.0, 6.0, 4.5, 4.0)

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "ph_initial",
    "ph_final",
    "ph_mean",
    "ph_std",
    "ph_range",
    "mass_final",
    *(f"mass_at_ph_{t:g}" for t in PH_THRESHOLDS),
    *(f"mass_at_ph_{t:g}_over_m0" for t in PH_THRESHOLDS),
)


def _mass_at_ph(
    ph: FloatArray, mass: FloatArray, threshold: float
) -> FloatArray:
    """Titrant mass where each curve first falls to ``threshold``.

    Parameters
    ----------
    ph
        ``(n, p)`` pH traces, decreasing along axis 1.
    mass
        ``(n, p)`` titrant masses in kg.
    threshold
        pH value to locate.

    Returns
    -------
    FloatArray
        ``(n,)`` interpolated titrant mass.
    """
    n, points = ph.shape
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        index = int(np.searchsorted(-ph[i], -threshold))
        index = min(max(index, 1), points - 1)
        upper, lower = ph[i, index - 1], ph[i, index]
        fraction = (upper - threshold) / (upper - lower + 1e-12)
        fraction = min(max(fraction, 0.0), 1.0)
        out[i] = mass[i, index - 1] + fraction * (
            mass[i, index] - mass[i, index - 1]
        )
    return out


def curve_features(
    curves: FloatArray, sample_mass: FloatArray
) -> FloatArray:
    """Compute the summary statistics in :data:`FEATURE_NAMES`.

    Parameters
    ----------
    curves
        ``(n, p, 2)`` array of measured ``(pH, titrant mass)``.
    sample_mass
        ``(n,)`` or ``(n, 1)`` sample mass in kg.

    Returns
    -------
    FloatArray
        ``(n, len(FEATURE_NAMES))`` of raw, unscaled features.  Scale them
        before use; :class:`~cleaned_reef_water.ml.normalization.Normalizer`
        does this when they are appended to the environment vector.

    Raises
    ------
    ValueError
        If ``curves`` is not three-dimensional or the batch sizes disagree.

    Examples
    --------
    >>> import numpy as np
    >>> ph = np.linspace(8.2, 3.5, 40)[None, :]
    >>> mass = np.linspace(0.0, 7e-4, 40)[None, :]
    >>> features = curve_features(
    ...     np.stack([ph, mass], axis=-1), np.array([0.015])
    ... )
    >>> features.shape == (1, len(FEATURE_NAMES))
    True
    >>> bool(features[0, 0] > features[0, 1])  # pH decreases
    True
    """
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim != 3:
        raise ValueError(f"curves must be 3-D, got shape {curves.shape}")
    mass_0 = np.asarray(sample_mass, dtype=np.float64).reshape(-1)
    if mass_0.shape[0] != curves.shape[0]:
        raise ValueError(
            f"curves and sample_mass must share the first dimension, got "
            f"{curves.shape[0]} and {mass_0.shape[0]}"
        )

    ph, mass = curves[:, :, 0], curves[:, :, 1]
    crossings = [_mass_at_ph(ph, mass, t) for t in PH_THRESHOLDS]

    columns = [
        ph[:, 0],
        ph[:, -1],
        ph.mean(axis=1),
        ph.std(axis=1),
        ph[:, 0] - ph[:, -1],
        mass[:, -1],
        *crossings,
        # Alkalinity is m_eq * C_a / m_0, so mass over sample mass tracks it.
        *(c / mass_0 for c in crossings),
    ]
    return np.stack(columns, axis=-1)
