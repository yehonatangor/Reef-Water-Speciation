r"""Total alkalinity, proton conditions and equilibrium speciation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .equilibria import (
    CarbonicFormulation,
    k_bisulfate_free,
    k_borate_total,
    k_carbonic_total,
    k_fluoride_free,
    k_phosphoric_total,
    k_silicate_total,
    k_water_total,
)
from .scales import free_to_total_factor
from .seawater import (
    BoronFormulation,
    total_borate,
    total_fluoride,
    total_sulfate,
)
from .types import ArrayLike, FloatArray, as_array

__all__ = [
    "EquilibriumConstants",
    "Speciation",
    "constants_at",
    "speciate",
    "total_alkalinity",
    "explicit_conservative_alkalinity",
    "DEFAULT_ORGANIC_PK",
]

#: Default ``pK_a`` of the lumped organic acid, on the **total** hydrogen ion
#: scale.
DEFAULT_ORGANIC_PK: float = 4.5


@dataclass(frozen=True, slots=True)
class EquilibriumConstants:
    """Bundle of stoichiometric constants evaluated at one ``(S, t)`` point.

    Attributes
    ----------
    k1, k2
        Carbonic acid first and second dissociation constants.
    kb
        Boric acid dissociation constant.
    kw
        Ion product of water, ``(mol kg-soln^-1)^2``.
    k1p, k2p, k3p
        Phosphoric acid dissociation constants.
    ksi
        Silicic acid first dissociation constant.
    k_hso4
        Bisulfate dissociation constant, **free** scale.
    k_hf
        Hydrogen fluoride dissociation constant, **free** scale.
    total_to_free
        Factor converting a total-scale ``[H+]`` to the free scale.
    total_sulfate_conc, total_fluoride_conc
        Conservative totals of sulfate and fluoride, cached here because they
        depend only on salinity and are needed on every call to
        :func:`speciate`.  Recomputing them inside the solver's inner loop
        dominated the runtime of a curve fit.
    salinity, t_c
        The conditions at which the bundle was evaluated.
    """

    k1: FloatArray
    k2: FloatArray
    kb: FloatArray
    kw: FloatArray
    k1p: FloatArray
    k2p: FloatArray
    k3p: FloatArray
    ksi: FloatArray
    k_hso4: FloatArray
    k_hf: FloatArray
    total_to_free: FloatArray
    total_sulfate_conc: FloatArray
    total_fluoride_conc: FloatArray
    salinity: FloatArray
    t_c: FloatArray


def constants_at(
    salinity: ArrayLike,
    t_c: ArrayLike,
    formulation: CarbonicFormulation = CarbonicFormulation.LUEKER2000,
) -> EquilibriumConstants:
    """Evaluate every equilibrium constant at one set of conditions.

    Parameters
    ----------
    salinity
        Practical salinity.
    t_c
        Temperature in degrees Celsius.
    formulation
        Carbonic acid parameterisation to use.

    Returns
    -------
    EquilibriumConstants
        All constants, on the total scale except ``k_hso4`` and ``k_hf``.
    """
    k1, k2 = k_carbonic_total(salinity, t_c, formulation)
    k1p, k2p, k3p = k_phosphoric_total(salinity, t_c)
    return EquilibriumConstants(
        k1=k1,
        k2=k2,
        kb=k_borate_total(salinity, t_c),
        kw=k_water_total(salinity, t_c),
        k1p=k1p,
        k2p=k2p,
        k3p=k3p,
        ksi=k_silicate_total(salinity, t_c),
        k_hso4=k_bisulfate_free(salinity, t_c),
        k_hf=k_fluoride_free(salinity, t_c),
        total_to_free=1.0 / free_to_total_factor(salinity, t_c),
        total_sulfate_conc=total_sulfate(salinity),
        total_fluoride_conc=total_fluoride(salinity),
        salinity=as_array(salinity),
        t_c=as_array(t_c),
    )


@dataclass(frozen=True, slots=True)
class Speciation:
    """Concentrations of every acid--base species, in ``mol kg-soln^-1``.

    Attributes
    ----------
    h_total
        Hydrogen ion concentration on the total scale.
    h_free
        Hydrogen ion concentration on the free scale.
    co2, hco3, co3
        Dissolved CO2 (including carbonic acid), bicarbonate and carbonate.
    boh4
        Borate ion.
    oh
        Hydroxide.
    h3po4, h2po4, hpo4, po4
        The four phosphate species.
    sioh4, siooh3
        Silicic acid and its conjugate base.
    hso4
        Bisulfate, computed on the free scale.
    hf
        Undissociated hydrogen fluoride, computed on the free scale.
    organic_anion
        Deprotonated lumped organic acid, ``A^-``.  Zero unless
        ``total_organic`` is supplied to :func:`speciate`.
    """

    h_total: FloatArray
    h_free: FloatArray
    co2: FloatArray
    hco3: FloatArray
    co3: FloatArray
    boh4: FloatArray
    oh: FloatArray
    h3po4: FloatArray
    h2po4: FloatArray
    hpo4: FloatArray
    po4: FloatArray
    sioh4: FloatArray
    siooh3: FloatArray
    hso4: FloatArray
    hf: FloatArray
    organic_anion: FloatArray

    @property
    def dic(self) -> FloatArray:
        """Total dissolved inorganic carbon, ``CO2 + HCO3 + CO3``."""
        return self.co2 + self.hco3 + self.co3

    @property
    def ph_total(self) -> FloatArray:
        """PH on the total scale."""
        return -np.log10(self.h_total)


def speciate(
    ph_total: ArrayLike,
    constants: EquilibriumConstants,
    *,
    dic: ArrayLike = 0.0,
    total_boron: ArrayLike | None = None,
    total_phosphate: ArrayLike = 0.0,
    total_silicate: ArrayLike = 0.0,
    total_organic: ArrayLike = 0.0,
    pk_organic: ArrayLike = DEFAULT_ORGANIC_PK,
    boron_formulation: BoronFormulation = BoronFormulation.UPPSTROM1974,
) -> Speciation:
    r"""Distribute conservative totals over species at a fixed pH.

    Parameters
    ----------
    ph_total
        pH on the total hydrogen ion scale.
    constants
        Constants bundle from :func:`constants_at`.
    dic
        Total dissolved inorganic carbon in ``mol kg-soln^-1``.
    total_boron
        Total borate in ``mol kg-soln^-1``.  If ``None``, computed from
        salinity using ``boron_formulation``.
    total_phosphate, total_silicate
        Nutrient totals in ``mol kg-soln^-1``.
    total_organic
        Total lumped organic acid in ``mol kg-soln^-1``.  Defaults to zero,
        in which case every returned value is bit-identical to the result
        without an organic term.
    pk_organic
        ``pK_a`` of the lumped organic acid on the **total** hydrogen ion
        scale; see :data:`DEFAULT_ORGANIC_PK`.
    boron_formulation
        Used only when ``total_boron`` is ``None``.

    Returns
    -------
    Speciation
        All species concentrations.
    """
    h = 10.0 ** -as_array(ph_total)
    h_free = h * constants.total_to_free
    c_t = as_array(dic)
    b_t = (
        total_borate(constants.salinity, boron_formulation)
        if total_boron is None
        else as_array(total_boron)
    )
    p_t = as_array(total_phosphate)
    si_t = as_array(total_silicate)

    denom_c = h * h + constants.k1 * h + constants.k1 * constants.k2
    co2 = c_t * h * h / denom_c
    hco3 = c_t * constants.k1 * h / denom_c
    co3 = c_t * constants.k1 * constants.k2 / denom_c

    boh4 = b_t * constants.kb / (constants.kb + h)
    oh = constants.kw / h

    k1p, k2p, k3p = constants.k1p, constants.k2p, constants.k3p
    denom_p = h**3 + k1p * h * h + k1p * k2p * h + k1p * k2p * k3p
    h3po4 = p_t * h**3 / denom_p
    h2po4 = p_t * k1p * h * h / denom_p
    hpo4 = p_t * k1p * k2p * h / denom_p
    po4 = p_t * k1p * k2p * k3p / denom_p

    siooh3 = si_t * constants.ksi / (constants.ksi + h)
    sioh4 = si_t - siooh3

    hso4 = constants.total_sulfate_conc / (1.0 + constants.k_hso4 / h_free)
    hf = constants.total_fluoride_conc / (1.0 + constants.k_hf / h_free)

    # Lumped organic acid, HA <-> H+ + A-.
    org_t = as_array(total_organic)
    k_org = 10.0 ** -as_array(pk_organic)
    organic_anion = org_t * k_org / (k_org + h)

    return Speciation(
        h_total=h,
        h_free=h_free,
        co2=co2,
        hco3=hco3,
        co3=co3,
        boh4=boh4,
        oh=oh,
        h3po4=h3po4,
        h2po4=h2po4,
        hpo4=hpo4,
        po4=po4,
        sioh4=sioh4,
        siooh3=siooh3,
        hso4=hso4,
        hf=hf,
        organic_anion=organic_anion,
    )


def total_alkalinity(speciation: Speciation) -> FloatArray:
    """Total alkalinity from a :class:`Speciation`, in ``mol kg-soln^-1``.

    Parameters
    ----------
    speciation
        Species concentrations from :func:`speciate`.

    Returns
    -------
    FloatArray
        Total alkalinity in ``mol kg-soln^-1``.
    """
    return (
        speciation.hco3
        + 2.0 * speciation.co3
        + speciation.boh4
        + speciation.oh
        + speciation.hpo4
        + 2.0 * speciation.po4
        + speciation.siooh3
        + speciation.organic_anion
        - speciation.h_free
        - speciation.hso4
        - speciation.hf
        - speciation.h3po4
    )


def explicit_conservative_alkalinity(
    sodium: ArrayLike,
    potassium: ArrayLike,
    magnesium: ArrayLike,
    calcium: ArrayLike,
    strontium: ArrayLike,
    chloride: ArrayLike,
    bromide: ArrayLike,
    nitrate: ArrayLike,
    total_sulfate_conc: ArrayLike,
    total_fluoride_conc: ArrayLike,
    total_phosphate: ArrayLike,
) -> FloatArray:
    r"""Total alkalinity from conservative ion concentrations alone.

    Parameters
    ----------
    sodium, potassium, magnesium, calcium, strontium
        Conservative cation concentrations in ``mol kg-soln^-1``.
    chloride, bromide, nitrate
        Conservative anion concentrations in ``mol kg-soln^-1``.
    total_sulfate_conc, total_fluoride_conc, total_phosphate
        Total concentrations of the corresponding acid systems.

    Returns
    -------
    FloatArray
        Total alkalinity in ``mol kg-soln^-1``.
    """
    return (
        as_array(sodium)
        + as_array(potassium)
        + 2.0 * as_array(magnesium)
        + 2.0 * as_array(calcium)
        + 2.0 * as_array(strontium)
        - as_array(chloride)
        - as_array(bromide)
        - as_array(nitrate)
        - 2.0 * as_array(total_sulfate_conc)
        - as_array(total_fluoride_conc)
        - as_array(total_phosphate)
    )
