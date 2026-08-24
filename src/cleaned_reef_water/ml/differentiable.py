r"""A differentiable TensorFlow reimplementation of the titration forward model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np

from ..alkalinity import DEFAULT_ORGANIC_PK, constants_at
from ..equilibria import CarbonicFormulation
from ..types import ArrayLike, FloatArray

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = [
    "CONSTANT_FIELDS",
    "PH_BRACKET",
    "alkalinity_from_ph",
    "carbonate_species",
    "pack_constants",
    "solve_ph",
    "titration_ph",
]

#: Column order of the packed constants tensor produced by
#: :func:`pack_constants`.
CONSTANT_FIELDS: Final[tuple[str, ...]] = (
    "k1",
    "k2",
    "kb",
    "kw",
    "k1p",
    "k2p",
    "k3p",
    "ksi",
    "k_hso4",
    "k_hf",
    "total_to_free",
    "total_sulfate_conc",
    "total_fluoride_conc",
)

#: Bracket for the pH bisection.  Wide enough for a titration carried well
#: past the carbonate endpoint (pH ~3) with margin on both sides.
PH_BRACKET: Final[tuple[float, float]] = (1.0, 13.0)

#: Finite-difference step used for the Newton denominator, in pH units.
_DERIVATIVE_STEP: Final[float] = 1e-5


def _tf() -> Any:
    """Import TensorFlow, with an actionable error if it is missing."""
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The differentiable forward model requires TensorFlow. "
            "Install it with:  pip install 'cleaned_reef_water[ml]'"
        ) from exc
    return tf


def pack_constants(
    salinity: ArrayLike,
    t_c: ArrayLike,
    formulation: CarbonicFormulation = CarbonicFormulation.LUEKER2000,
) -> FloatArray:
    """Evaluate the equilibrium constants and pack them into an array.

    Parameters
    ----------
    salinity, t_c
        Practical salinity and temperature in degrees Celsius.  Scalars or
        1-D arrays of matching length.
    formulation
        Carbonic acid parameterisation.

    Returns
    -------
    FloatArray
        ``(n, 13)`` array whose columns are :data:`CONSTANT_FIELDS`.

    Examples
    --------
    >>> packed = pack_constants([35.0, 33.0], [25.0, 20.0])
    >>> packed.shape
    (2, 13)
    >>> bool(np.all(packed > 0.0))
    True
    """
    salinity_array = np.atleast_1d(np.asarray(salinity, dtype=np.float64))
    t_c_array = np.atleast_1d(np.asarray(t_c, dtype=np.float64))
    if salinity_array.shape != t_c_array.shape:
        raise ValueError(
            f"salinity and t_c must have the same shape, got "
            f"{salinity_array.shape} and {t_c_array.shape}"
        )

    constants = constants_at(salinity_array, t_c_array, formulation)
    columns = [
        np.broadcast_to(
            np.asarray(getattr(constants, field), dtype=np.float64),
            salinity_array.shape,
        )
        for field in CONSTANT_FIELDS
    ]
    return np.stack(columns, axis=-1)


def _unpack(constants: Any) -> dict[str, Any]:
    """Split a packed ``(n, 13)`` tensor into named ``(n, 1)`` columns."""
    return {
        field: constants[:, index : index + 1]
        for index, field in enumerate(CONSTANT_FIELDS)
    }


def alkalinity_from_ph(
    ph: Any,
    constants: Any,
    dic: Any,
    total_boron: Any,
    total_phosphate: Any = 0.0,
    total_silicate: Any = 0.0,
    total_organic: Any = 0.0,
    pk_organic: Any = DEFAULT_ORGANIC_PK,
) -> Any:
    r"""Total alkalinity implied by a pH, as a differentiable tensor.

    Parameters
    ----------
    ph
        pH on the total scale, shape ``(n, p)`` or broadcastable to it.
    constants
        Packed constants, shape ``(n, 13)``; see :func:`pack_constants`.
    dic, total_boron, total_phosphate, total_silicate
        Conservative totals in ``mol kg-soln^-1``, already diluted.
    total_organic
        Lumped ionisable organic matter in ``mol kg-soln^-1``, already diluted.
        Defaults to zero, which reproduces the previous behaviour exactly.
    pk_organic
        Apparent :math:`\mathrm{p}K_a` of that lumped acid.  **Not** diluted:
        it is an equilibrium constant, not a concentration.

    Returns
    -------
    Tensor
        Total alkalinity in ``mol kg-soln^-1``.
    """
    tf = _tf()
    c = _unpack(constants)
    h = tf.pow(tf.constant(10.0, dtype=tf.float64), -ph)
    h_free = h * c["total_to_free"]

    denom_c = h * h + c["k1"] * h + c["k1"] * c["k2"]
    hco3 = dic * c["k1"] * h / denom_c
    co3 = dic * c["k1"] * c["k2"] / denom_c

    boh4 = total_boron * c["kb"] / (c["kb"] + h)
    oh = c["kw"] / h

    k1p, k2p, k3p = c["k1p"], c["k2p"], c["k3p"]
    denom_p = h**3 + k1p * h * h + k1p * k2p * h + k1p * k2p * k3p
    h3po4 = total_phosphate * h**3 / denom_p
    hpo4 = total_phosphate * k1p * k2p * h / denom_p
    po4 = total_phosphate * k1p * k2p * k3p / denom_p

    siooh3 = total_silicate * c["ksi"] / (c["ksi"] + h)

    # Lumped organic acid, HA <-> H+ + A-.
    k_organic = tf.pow(tf.constant(10.0, dtype=tf.float64), -pk_organic)
    organic_anion = total_organic * k_organic / (k_organic + h)

    hso4 = c["total_sulfate_conc"] / (1.0 + c["k_hso4"] / h_free)
    hf = c["total_fluoride_conc"] / (1.0 + c["k_hf"] / h_free)

    return (
        hco3
        + 2.0 * co3
        + boh4
        + oh
        + hpo4
        + 2.0 * po4
        + siooh3
        + organic_anion
        - h_free
        - hso4
        - hf
        - h3po4
    )


def carbonate_species(
    ph: Any,
    constants: Any,
    dic: Any,
    total_boron: Any,
) -> tuple[Any, Any, Any]:
    r"""Return ``(HCO3, CO3, B(OH)4)`` at a given pH, differentiably.

    Parameters
    ----------
    ph
        pH on the total scale.
    constants
        Packed constants, shape ``(n, 13)``.
    dic, total_boron
        Conservative totals in ``mol kg-soln^-1``.

    Returns
    -------
    tuple
        The three species this project reports, each the same shape as ``ph``.
    """
    tf = _tf()
    c = _unpack(constants)
    h = tf.pow(tf.constant(10.0, dtype=tf.float64), -ph)
    denominator = h * h + c["k1"] * h + c["k1"] * c["k2"]
    hco3 = dic * c["k1"] * h / denominator
    co3 = dic * c["k1"] * c["k2"] / denominator
    boh4 = total_boron * c["kb"] / (c["kb"] + h)
    return hco3, co3, boh4


def solve_ph(
    target_alkalinity: Any,
    constants: Any,
    dic: Any,
    total_boron: Any,
    total_phosphate: Any = 0.0,
    total_silicate: Any = 0.0,
    total_organic: Any = 0.0,
    pk_organic: Any = DEFAULT_ORGANIC_PK,
    n_iterations: int = 60,
) -> Any:
    r"""Solve for the pH at which the model alkalinity equals a target.

    Parameters
    ----------
    target_alkalinity
        Required alkalinity in ``mol kg-soln^-1``, shape ``(n, p)``.
    constants
        Packed constants, shape ``(n, 13)``.
    dic, total_boron, total_phosphate, total_silicate, total_organic
        Diluted conservative totals, broadcastable to ``target_alkalinity``.
    pk_organic
        Apparent pKa of the lumped organic acid; not diluted.
    n_iterations
        Bisection iterations.  60 halvings of the 12-unit bracket resolve pH
        to ~1e-17, below float64 resolution at pH 8, so the subsequent Newton
        step is a formality for the value and exists for the gradient.

    Returns
    -------
    Tensor
        pH on the total scale, differentiable with respect to every argument
        via the implicit function theorem.
    """
    tf = _tf()

    def residual(ph: Any) -> Any:
        return (
            alkalinity_from_ph(
                ph,
                constants,
                dic,
                total_boron,
                total_phosphate,
                total_silicate,
                total_organic,
                pk_organic,
            )
            - target_alkalinity
        )

    shape = tf.shape(target_alkalinity)
    low = tf.fill(shape, tf.constant(PH_BRACKET[0], dtype=tf.float64))
    high = tf.fill(shape, tf.constant(PH_BRACKET[1], dtype=tf.float64))

    # The bracket search carries no gradient, so detach every input to it.
    # This is what keeps 60 iterations off the tape entirely.
    for _ in range(n_iterations):
        middle = 0.5 * (low + high)
        too_acidic = tf.stop_gradient(residual(middle)) < 0.0
        low = tf.where(too_acidic, middle, low)
        high = tf.where(too_acidic, high, middle)
    ph_0 = tf.stop_gradient(0.5 * (low + high))

    # One Newton step supplies the exact implicit-function-theorem gradient.
    step = tf.constant(_DERIVATIVE_STEP, dtype=tf.float64)
    slope = tf.stop_gradient(
        (residual(ph_0 + step) - residual(ph_0 - step)) / (2.0 * step)
    )
    return ph_0 - residual(ph_0) / slope


def titration_ph(
    alkalinity: Any,
    dic: Any,
    constants: Any,
    titrant_mass: Any,
    sample_mass: Any,
    total_boron: Any,
    total_phosphate: Any = 0.0,
    total_silicate: Any = 0.0,
    total_organic: Any = 0.0,
    pk_organic: Any = DEFAULT_ORGANIC_PK,
    titrant_concentration: float = 0.1,
    n_iterations: int = 60,
) -> Any:
    r"""Render a full titration curve, differentiably.

    Parameters
    ----------
    alkalinity, dic
        Sample totals in ``mol kg-soln^-1``, shape ``(n, 1)``.
    constants
        Packed constants, shape ``(n, 13)``.
    titrant_mass
        Titrant mass at each point in kg, shape ``(n, p)``.  Multiply by a
        fitted pump-scale factor *before* calling if one is being estimated.
    sample_mass
        Sample mass in kg, shape ``(n, 1)``.
    total_boron, total_phosphate, total_silicate, total_organic
        Undiluted totals, shape ``(n, 1)``; diluted internally.
    pk_organic
        Apparent pKa of the lumped organic acid, shape ``(n, 1)``.  Deliberately
        **not** diluted -- diluting an equilibrium constant alongside the
        concentrations it governs is an easy and silent error, and the two
        arguments sit adjacent in the signature.
    titrant_concentration
        Titrant concentration in ``mol kg-soln^-1``.
    n_iterations
        Bisection iterations per point.

    Returns
    -------
    Tensor
        pH on the total scale, shape ``(n, p)``.
    """
    total = sample_mass + titrant_mass
    dilution = sample_mass / total
    titrant_dilution = titrant_mass / total

    target = alkalinity * dilution - titrant_concentration * titrant_dilution
    return solve_ph(
        target,
        constants,
        dic * dilution,
        total_boron * dilution,
        total_phosphate * dilution,
        total_silicate * dilution,
        total_organic * dilution,
        pk_organic,
        n_iterations=n_iterations,
    )
