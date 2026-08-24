r"""Robust numerical solvers for the seawater acid--base system."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import numpy as np

from .alkalinity import (
    EquilibriumConstants,
    speciate,
    total_alkalinity,
)
from .seawater import BoronFormulation

__all__ = [
    "ConvergenceError",
    "PH_LOWER_LIMIT",
    "PH_UPPER_LIMIT",
    "solve_scalar_bracketed",
    "ph_from_dic_alkalinity",
    "ph_from_alkalinity_and_totals",
]

#: Hardest physically meaningful acidity considered by the solvers.
PH_LOWER_LIMIT: Final[float] = 1.0

#: Hardest physically meaningful alkalinity considered by the solvers.
PH_UPPER_LIMIT: Final[float] = 13.0


class ConvergenceError(RuntimeError):
    """Raised when a solver cannot bracket or converge on a root.

    Attributes
    ----------
    bracket
        The widest bracket that was attempted.
    residuals
        Residual values at the bracket endpoints.
    """

    def __init__(
        self,
        message: str,
        bracket: tuple[float, float] | None = None,
        residuals: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(message)
        self.bracket = bracket
        self.residuals = residuals


def _brent(
    residual: Callable[[float], float],
    x_low: float,
    x_high: float,
    f_low: float,
    f_high: float,
    xtol: float,
    rtol: float,
    max_iterations: int = 100,
) -> float:
    """Brent's method on a bracket already known to straddle a root.

    Parameters
    ----------
    residual
        Continuous scalar function whose root is sought.
    x_low, x_high
        Bracket endpoints, with ``residual(x_low) * residual(x_high) <= 0``.
    f_low, f_high
        Residual values at those endpoints.
    xtol, rtol
        Absolute and relative convergence tolerances on the abscissa.  The
        iteration stops when the bisection half-width falls below
        ``(xtol + rtol * |x|) / 2``.
    max_iterations
        Iteration cap.  Brent's method converges superlinearly and 100 is far
        beyond what this residual needs; it exists so that a pathological
        residual terminates rather than hangs.

    Returns
    -------
    float
        The root.

    Examples
    --------
    >>> f = lambda x: x**3 - 2.0 * x - 5.0
    >>> root = _brent(f, 2.0, 3.0, f(2.0), f(3.0), 1e-12, 1e-15)
    >>> abs(root - 2.0945514815423265) < 1e-12
    True
    """
    x_pre, x_cur = x_low, x_high
    f_pre, f_cur = f_low, f_high
    x_blk = f_blk = s_pre = s_cur = 0.0

    if f_pre == 0.0:
        return x_pre
    if f_cur == 0.0:
        return x_cur

    for _ in range(max_iterations):
        # Keep a contrapoint on the opposite side of the root from x_cur.
        if f_pre * f_cur < 0.0:
            x_blk, f_blk = x_pre, f_pre
            s_pre = s_cur = x_cur - x_pre
        # Always iterate from the endpoint with the smaller residual.
        if abs(f_blk) < abs(f_cur):
            x_pre, x_cur, x_blk = x_cur, x_blk, x_cur
            f_pre, f_cur, f_blk = f_cur, f_blk, f_cur

        tolerance = 0.5 * (xtol + rtol * abs(x_cur))
        half_width = 0.5 * (x_blk - x_cur)
        if f_cur == 0.0 or abs(half_width) < tolerance:
            return x_cur

        if abs(s_pre) > tolerance and abs(f_cur) < abs(f_pre):
            if x_pre == x_blk:
                # Two distinct points only: secant step.
                step = -f_cur * (x_cur - x_pre) / (f_cur - f_pre)
            else:
                # Three distinct points: inverse quadratic interpolation.
                d_pre = (f_pre - f_cur) / (x_pre - x_cur)
                d_blk = (f_blk - f_cur) / (x_blk - x_cur)
                step = (
                    -f_cur
                    * (f_blk * d_blk - f_pre * d_pre)
                    / (d_blk * d_pre * (f_blk - f_pre))
                )
            # Accept the interpolation only if it is inside the bracket and shrinking.
            if 2.0 * abs(step) < min(abs(s_pre), 3.0 * abs(half_width) - tolerance):
                s_pre, s_cur = s_cur, step
            else:
                s_pre = s_cur = half_width
        else:
            s_pre = s_cur = half_width

        x_pre, f_pre = x_cur, f_cur
        if abs(s_cur) > tolerance:
            x_cur += s_cur
        else:
            x_cur += tolerance if half_width > 0.0 else -tolerance
        f_cur = residual(x_cur)

    return x_cur


def solve_scalar_bracketed(
    residual: Callable[[float], float],
    initial_bracket: tuple[float, float] = (6.0, 9.5),
    *,
    hard_limits: tuple[float, float] = (PH_LOWER_LIMIT, PH_UPPER_LIMIT),
    expansion: float = 1.6,
    max_expansions: int = 24,
    xtol: float = 1e-10,
    rtol: float = 8.881784197001252e-16,
) -> float:
    """Find a root by expanding a bracket until a sign change appears.

    Parameters
    ----------
    residual
        Continuous scalar function whose root is sought.
    initial_bracket
        Starting ``(low, high)`` bracket.  Chosen to cover ordinary seawater so
        that the common case converges without any expansion.
    hard_limits
        Absolute ``(low, high)`` limits beyond which expansion stops.
    expansion
        Multiplicative factor by which the bracket half-width grows each
        iteration.
    max_expansions
        Maximum number of expansion steps before giving up.
    xtol, rtol
        Absolute and relative convergence tolerances passed to :func:`_brent`.
        The default ``rtol`` is ``4 * np.finfo(float).eps``, the smallest value
        SciPy permits, since a pH is an order-one quantity and there is no
        reason to stop short of machine precision.

    Returns
    -------
    float
        The root.

    Raises
    ------
    ConvergenceError
        If no sign change is found within ``hard_limits``, or if the residual
        is non-finite anywhere in the bracket.

    Examples
    --------
    >>> solve_scalar_bracketed(lambda x: x - 7.0)
    7.0
    """
    low, high = initial_bracket
    hard_low, hard_high = hard_limits
    f_low, f_high = residual(low), residual(high)

    for _ in range(max_expansions):
        if not (np.isfinite(f_low) and np.isfinite(f_high)):
            raise ConvergenceError(
                "residual is not finite at the bracket endpoints",
                bracket=(low, high),
                residuals=(float(f_low), float(f_high)),
            )
        if f_low * f_high <= 0.0:
            return float(
                _brent(residual, low, high, f_low, f_high, xtol=xtol, rtol=rtol)
            )
        if low <= hard_low and high >= hard_high:
            break
        half_width = 0.5 * (high - low) * expansion
        midpoint = 0.5 * (low + high)
        low = max(hard_low, midpoint - half_width)
        high = min(hard_high, midpoint + half_width)
        f_low, f_high = residual(low), residual(high)

    raise ConvergenceError(
        "no sign change in the residual within the physically admissible pH "
        f"range {hard_limits}; the requested state may be thermodynamically "
        "inconsistent (for example alkalinity exceeding what the given DIC "
        "and boron can supply)",
        bracket=(low, high),
        residuals=(float(f_low), float(f_high)),
    )


def ph_from_alkalinity_and_totals(
    alkalinity: float,
    constants: EquilibriumConstants,
    *,
    dic: float = 0.0,
    total_boron: float | None = None,
    total_phosphate: float = 0.0,
    total_silicate: float = 0.0,
    boron_formulation: BoronFormulation = BoronFormulation.UPPSTROM1974,
    initial_bracket: tuple[float, float] = (6.0, 9.5),
) -> float:
    """Solve for total-scale pH given alkalinity and the conservative totals.

    Parameters
    ----------
    alkalinity
        Total alkalinity in ``mol kg-soln^-1``.
    constants
        Constants bundle from :func:`~cleaned_reef_water.alkalinity.constants_at`.
    dic
        Total dissolved inorganic carbon in ``mol kg-soln^-1``.
    total_boron
        Total borate in ``mol kg-soln^-1``; derived from salinity if ``None``.
    total_phosphate, total_silicate
        Nutrient totals in ``mol kg-soln^-1``.
    boron_formulation
        Used only when ``total_boron`` is ``None``.
    initial_bracket
        Starting pH bracket.

    Returns
    -------
    float
        pH on the total hydrogen ion scale.

    Raises
    ------
    ConvergenceError
        If the requested combination has no solution.
    """

    def residual(ph: float) -> float:
        species = speciate(
            ph,
            constants,
            dic=dic,
            total_boron=total_boron,
            total_phosphate=total_phosphate,
            total_silicate=total_silicate,
            boron_formulation=boron_formulation,
        )
        return float(total_alkalinity(species)) - alkalinity

    return solve_scalar_bracketed(residual, initial_bracket)


def ph_from_dic_alkalinity(
    dic: float,
    alkalinity: float,
    salinity: float,
    t_c: float,
    **kwargs: object,
) -> float:
    """Return total-scale pH from DIC and alkalinity.

    Parameters
    ----------
    dic
        Total dissolved inorganic carbon in ``mol kg-soln^-1``.
    alkalinity
        Total alkalinity in ``mol kg-soln^-1``.
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.
    **kwargs
        Forwarded to :func:`ph_from_alkalinity_and_totals`.

    Returns
    -------
    float
        pH on the total hydrogen ion scale.

    Examples
    --------
    Surface ocean water with ``DIC = 2000`` and ``A_T = 2300 umol/kg`` sits
    near pH 8.1:

    >>> ph = ph_from_dic_alkalinity(2000e-6, 2300e-6, 35.0, 25.0)
    >>> bool(8.0 < ph < 8.2)
    True
    """
    from .alkalinity import constants_at

    constants = constants_at(salinity, t_c)
    return ph_from_alkalinity_and_totals(
        alkalinity, constants, dic=dic, **kwargs  # type: ignore[arg-type]
    )
