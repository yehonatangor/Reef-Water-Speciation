r"""Forward model of a potentiometric acid titration of seawater."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .alkalinity import (
    DEFAULT_ORGANIC_PK,
    EquilibriumConstants,
    constants_at,
    speciate,
    total_alkalinity,
)
from .equilibria import CarbonicFormulation
from .seawater import BoronFormulation, total_borate
from .solvers import ConvergenceError, solve_scalar_bracketed
from .types import FloatArray

__all__ = [
    "Titrant",
    "STRONG_ACID",
    "TitrationCurve",
    "simulate_titration",
]


@dataclass(frozen=True, slots=True)
class Titrant:
    r"""A titrant characterised by its dissociation constants.

    Attributes
    ----------
    name
        Human-readable identifier.
    concentration
        Total titrant concentration in ``mol kg-soln^-1``.
    pk_values
        Successive ``pK_a`` values of the acid, in ascending order.  An empty
        tuple denotes a strong acid, which is treated as fully dissociated.
    """

    name: str
    concentration: float
    pk_values: tuple[float, ...] = ()

    @property
    def is_strong(self) -> bool:
        """``True`` when the titrant is modelled as fully dissociated."""
        return len(self.pk_values) == 0

    @property
    def dissociation_constants(self) -> tuple[float, ...]:
        """Successive ``K_a`` values derived from :attr:`pk_values`."""
        return tuple(10.0**-pk for pk in self.pk_values)

    def protons_released(self, h: FloatArray) -> FloatArray:
        """Mean number of protons released per mole of titrant.

        Parameters
        ----------
        h
            Hydrogen ion concentration in ``mol kg-soln^-1`` on the same scale
            as the titrant's ``pK`` values.

        Returns
        -------
        FloatArray
            Mean protons released, between ``0`` and ``len(pk_values)``.
        """
        if self.is_strong:
            return np.ones_like(h)
        constants = self.dissociation_constants
        n = len(constants)
        denominator = h**n
        numerator = np.zeros_like(h)
        cumulative = 1.0
        for j, k in enumerate(constants, start=1):
            cumulative = cumulative * k
            term = cumulative * h ** (n - j)
            denominator = denominator + term
            numerator = numerator + j * term
        return numerator / denominator

    def titratable_capacity(self) -> float:
        """Total proton concentration the titrant can supply, ``mol kg^-1``.

        Returns
        -------
        float
            ``concentration`` for a strong acid; ``concentration`` times the
            number of ionisable protons for a polyprotic acid.
        """
        return self.concentration * max(1, len(self.pk_values))


#: Guide SOP 3b titrant: 0.1 mol/kg HCl in 0.6 mol/kg NaCl, matching ionic strength.
STRONG_ACID: Final[Titrant] = Titrant(name="HCl", concentration=0.1)


@dataclass(frozen=True, slots=True)
class TitrationCurve:
    """Result of a simulated titration.

    Attributes
    ----------
    titrant_mass
        Mass of titrant added at each point, in kg.
    ph_total
        Equilibrium pH on the total hydrogen ion scale at each point.
    sample_mass
        Mass of the original sample, in kg.
    alkalinity
        Total alkalinity of the original sample, ``mol kg-soln^-1``.
    titrant
        The titrant used.
    equivalence_mass
        Mass of titrant at the alkalinity equivalence point, in kg.
    """

    titrant_mass: FloatArray
    ph_total: FloatArray
    sample_mass: float
    alkalinity: float
    titrant: Titrant
    equivalence_mass: float

    @property
    def n_points(self) -> int:
        """Number of titration points."""
        return int(self.titrant_mass.size)

    def as_volume_ml(self, titrant_density_kg_per_l: float = 1.0) -> FloatArray:
        """Titrant mass expressed as a volume in millilitres.

        Parameters
        ----------
        titrant_density_kg_per_l
            Density of the titrant solution in ``kg L^-1``.

        Returns
        -------
        FloatArray
            Titrant volume in mL.
        """
        return self.titrant_mass / titrant_density_kg_per_l * 1000.0


def simulate_titration(
    alkalinity: float,
    dic: float,
    salinity: float,
    t_c: float,
    *,
    titrant: Titrant = STRONG_ACID,
    sample_mass_kg: float = 0.015,
    n_points: int = 200,
    max_equivalence_fraction: float = 2.0,
    total_phosphate: float = 0.0,
    total_silicate: float = 0.0,
    total_organic: float = 0.0,
    pk_organic: float = DEFAULT_ORGANIC_PK,
    total_boron: float | None = None,
    boron_formulation: BoronFormulation = BoronFormulation.UPPSTROM1974,
    carbonic_formulation: CarbonicFormulation = CarbonicFormulation.LUEKER2000,
    constants: EquilibriumConstants | None = None,
) -> TitrationCurve:
    r"""Simulate a potentiometric acid titration of a seawater sample.

    Parameters
    ----------
    alkalinity
        Total alkalinity of the sample in ``mol kg-soln^-1``.
    dic
        Total dissolved inorganic carbon in ``mol kg-soln^-1``.
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.
    titrant
        Titrant model; see :data:`STRONG_ACID`.
    sample_mass_kg
        Mass of seawater sample, in kg.
    n_points
        Number of titration points, spaced uniformly in titrant mass.
    max_equivalence_fraction
        Titrate to this multiple of the equivalence point.
    total_phosphate, total_silicate
        Nutrient totals in ``mol kg-soln^-1``.
    total_organic, pk_organic
        Lumped organic acid total in ``mol kg-soln^-1`` and its ``pK_a`` on
        the total scale.  Organic matter titrates, so it belongs in the
        forward model rather than in the noise; a classical fit with no
        organic term absorbs it into carbonate alkalinity.  Zero by default,
        which reproduces the organic-free forward model exactly.
    total_boron
        Total borate in ``mol kg-soln^-1``; derived from salinity if ``None``.
    boron_formulation, carbonic_formulation
        Parameterisation choices.
    constants
        Pre-computed constants bundle, to avoid re-evaluating them inside a
        Monte Carlo loop.  Must correspond to ``salinity`` and ``t_c``.

    Returns
    -------
    TitrationCurve
        The simulated curve.

    Raises
    ------
    ValueError
        If ``n_points < 2``, or if any mass or concentration is non-positive.
    ConvergenceError
        If the equilibrium pH cannot be bracketed at some titration point.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points}")
    if sample_mass_kg <= 0.0:
        raise ValueError(f"sample_mass_kg must be positive, got {sample_mass_kg}")
    if titrant.concentration <= 0.0:
        raise ValueError("titrant concentration must be positive")
    if max_equivalence_fraction <= 0.0:
        raise ValueError("max_equivalence_fraction must be positive")

    if constants is None:
        constants = constants_at(salinity, t_c, carbonic_formulation)

    b_t = (
        float(total_borate(salinity, boron_formulation))
        if total_boron is None
        else total_boron
    )

    equivalence_mass = sample_mass_kg * alkalinity / titrant.titratable_capacity()
    masses = np.linspace(
        0.0, equivalence_mass * max_equivalence_fraction, n_points, dtype=np.float64
    )

    ph_values = np.empty(n_points, dtype=np.float64)
    bracket = (6.0, 9.5)

    for index, mass in enumerate(masses):
        dilution = sample_mass_kg / (sample_mass_kg + mass)
        titrant_dilution = mass / (sample_mass_kg + mass)

        def residual(
            ph: float,
            _d: float = dilution,
            _td: float = titrant_dilution,
        ) -> float:
            species = speciate(
                ph,
                constants,
                dic=dic * _d,
                total_boron=b_t * _d,
                total_phosphate=total_phosphate * _d,
                total_silicate=total_silicate * _d,
                # The organic total dilutes like every other conservative
                # total; its pK_a does not.
                total_organic=total_organic * _d,
                pk_organic=pk_organic,
            )
            if titrant.is_strong:
                acid_protons = titrant.concentration * _td
            else:
                acid_protons = (
                    titrant.concentration
                    * _td
                    * float(titrant.protons_released(np.asarray(10.0**-ph)))
                )
            modelled = float(total_alkalinity(species))
            return modelled - (alkalinity * _d - float(acid_protons))

        try:
            ph_values[index] = solve_scalar_bracketed(residual, bracket)
        except ConvergenceError as exc:
            raise ConvergenceError(
                f"titration point {index} (titrant mass {mass:.6g} kg) failed to "
                f"converge: {exc}",
                bracket=exc.bracket,
                residuals=exc.residuals,
            ) from exc
        # Warm-start the next point from the current solution: the curve is
        # monotone, so this is both safe and roughly twice as fast.
        bracket = (max(1.5, ph_values[index] - 1.5), ph_values[index] + 0.5)

    return TitrationCurve(
        titrant_mass=masses,
        ph_total=ph_values,
        sample_mass=sample_mass_kg,
        alkalinity=alkalinity,
        titrant=titrant,
        equivalence_mass=equivalence_mass,
    )
