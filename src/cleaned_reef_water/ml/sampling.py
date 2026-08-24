"""Sampling of physically valid seawater compositions for training data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..alkalinity import constants_at, speciate, total_alkalinity
from ..equilibria import CarbonicFormulation
from ..seawater import BoronFormulation, total_borate

__all__ = ["Composition", "SamplingRanges", "sample_composition"]


@dataclass(frozen=True, slots=True)
class Composition:
    """One physically valid seawater composition.

    Attributes
    ----------
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.
    dic
        Total dissolved inorganic carbon, ``mol kg-soln^-1``.
    total_boron
        Total borate, ``mol kg-soln^-1``.
    total_phosphate, total_silicate
        Nutrient totals, ``mol kg-soln^-1``.
    alkalinity
        Total alkalinity, ``mol kg-soln^-1``.
    ph_total
        Initial pH on the total scale.
    sample_mass_kg
        Mass of sample to be titrated, in kg.
    """

    salinity: float
    t_c: float
    dic: float
    total_boron: float
    total_phosphate: float
    total_silicate: float
    total_organic: float
    pk_organic: float
    alkalinity: float
    ph_total: float
    sample_mass_kg: float


@dataclass(frozen=True, slots=True)
class SamplingRanges:
    """Uniform sampling ranges for the composition generator.

    Attributes
    ----------
    salinity
        Practical salinity range.
    t_c
        Temperature range in degrees Celsius.
    dic
        DIC range in ``mol kg-soln^-1``.
    ph_total
        Initial total-scale pH range.
    boron_jitter
        Multiplicative jitter applied to the salinity-derived borate.
    phosphate_probability
        Probability that a sample carries non-zero phosphate.
    phosphate_log10
        ``log10`` range of phosphate in ``mol kg-soln^-1`` when non-zero.
    silicate_probability
        Probability that a sample carries non-zero silicate.
    silicate_log10
        ``log10`` range of silicate in ``mol kg-soln^-1`` when non-zero.
    sample_mass_kg
        Range of titrated sample mass, in kg.
    organic_probability
        Probability that a sample carries organic alkalinity.
    organic
        Range of lumped organic acid total in ``mol kg-soln^-1``.
    pk_organic
        Range of the organic ``pK_a`` on the total scale.
    """

    salinity: tuple[float, float] = (25.0, 40.0)
    t_c: tuple[float, float] = (22.0, 28.0)
    dic: tuple[float, float] = (1.0e-3, 5.0e-3)
    ph_total: tuple[float, float] = (7.6, 8.4)
    boron_jitter: tuple[float, float] = (0.85, 1.15)
    phosphate_probability: float = 0.25
    phosphate_log10: tuple[float, float] = (-6.0, -4.5)
    silicate_probability: float = 0.25
    silicate_log10: tuple[float, float] = (-6.0, -4.0)
    sample_mass_kg: tuple[float, float] = (0.0145, 0.0155)
    organic_probability: float = 0.5
    organic: tuple[float, float] = (0.0, 1.5e-4)
    pk_organic: tuple[float, float] = (4.0, 7.0)


def sample_composition(
    rng: np.random.Generator,
    ranges: SamplingRanges | None = None,
    formulation: CarbonicFormulation = CarbonicFormulation.LUEKER2000,
) -> Composition:
    """Draw one physically valid composition.

    Parameters
    ----------
    rng
        Random generator.  Passed explicitly rather than using the global
        NumPy random state, so that data generation is reproducible and
        parallelisable.
    ranges
        Sampling ranges; defaults to :class:`SamplingRanges`.
    formulation
        Carbonic acid parameterisation used to compute alkalinity.

    Returns
    -------
    Composition
        A feasible composition with its alkalinity and initial pH.
    """
    if ranges is None:
        ranges = SamplingRanges()

    salinity = float(rng.uniform(*ranges.salinity))
    t_c = float(rng.uniform(*ranges.t_c))
    dic = float(rng.uniform(*ranges.dic))
    ph = float(rng.uniform(*ranges.ph_total))

    boron = float(total_borate(salinity, BoronFormulation.UPPSTROM1974)) * float(
        rng.uniform(*ranges.boron_jitter)
    )
    phosphate = (
        float(10.0 ** rng.uniform(*ranges.phosphate_log10))
        if rng.random() < ranges.phosphate_probability
        else 0.0
    )
    silicate = (
        float(10.0 ** rng.uniform(*ranges.silicate_log10))
        if rng.random() < ranges.silicate_probability
        else 0.0
    )

    organic = (
        float(rng.uniform(*ranges.organic))
        if rng.random() < ranges.organic_probability
        else 0.0
    )
    pk_organic = float(rng.uniform(*ranges.pk_organic))

    # pH and DIC are drawn; alkalinity is *derived*.
    constants = constants_at(salinity, t_c, formulation)
    species = speciate(
        ph,
        constants,
        dic=dic,
        total_boron=boron,
        total_phosphate=phosphate,
        total_silicate=silicate,
        total_organic=organic,
        pk_organic=pk_organic,
    )
    alkalinity = float(total_alkalinity(species))

    return Composition(
        salinity=salinity,
        t_c=t_c,
        dic=dic,
        total_boron=boron,
        total_phosphate=phosphate,
        total_silicate=silicate,
        total_organic=organic,
        pk_organic=pk_organic,
        alkalinity=alkalinity,
        ph_total=ph,
        sample_mass_kg=float(rng.uniform(*ranges.sample_mass_kg)),
    )
