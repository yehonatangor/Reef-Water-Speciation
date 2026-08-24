"""Reef water carbonate chemistry, on the total hydrogen ion scale.

Examples
--------
>>> from cleaned_reef_water import ph_from_dic_alkalinity
>>> ph = ph_from_dic_alkalinity(2000e-6, 2300e-6, 35.0, 25.0)
>>> bool(8.0 < ph < 8.1)
True
"""

from __future__ import annotations

from .alkalinity import (
    EquilibriumConstants,
    Speciation,
    constants_at,
    explicit_conservative_alkalinity,
    speciate,
    total_alkalinity,
)
from .baseline import InversionResult, LeastSquaresInverse, NuisanceModel
from .equilibria import (
    CarbonicFormulation,
    k_aragonite,
    k_bisulfate_free,
    k_borate_total,
    k_calcite,
    k_carbonic_total,
    k_co2_solubility,
    k_fluoride_free,
    k_fluoride_total,
    k_phosphoric_total,
    k_silicate_total,
    k_water_total,
)
from .scales import convert_constant, convert_ph, scale_factor
from .seawater import (
    BoronFormulation,
    density_seawater,
    ionic_strength,
    molal_to_molar,
    molar_to_molal,
    total_borate,
    total_calcium,
    total_fluoride,
    total_sulfate,
)
from .solvers import (
    ConvergenceError,
    ph_from_alkalinity_and_totals,
    ph_from_dic_alkalinity,
)
from .titration import STRONG_ACID, Titrant, TitrationCurve, simulate_titration
from .types import PHScale

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "BoronFormulation",
    "CarbonicFormulation",
    "ConvergenceError",
    "EquilibriumConstants",
    "InversionResult",
    "LeastSquaresInverse",
    "NuisanceModel",
    "PHScale",
    "STRONG_ACID",
    "Speciation",
    "TitrationCurve",
    "Titrant",
    "constants_at",
    "convert_constant",
    "convert_ph",
    "density_seawater",
    "explicit_conservative_alkalinity",
    "ionic_strength",
    "k_aragonite",
    "k_bisulfate_free",
    "k_borate_total",
    "k_calcite",
    "k_carbonic_total",
    "k_co2_solubility",
    "k_fluoride_free",
    "k_fluoride_total",
    "k_phosphoric_total",
    "k_silicate_total",
    "k_water_total",
    "molal_to_molar",
    "molar_to_molal",
    "ph_from_alkalinity_and_totals",
    "ph_from_dic_alkalinity",
    "scale_factor",
    "simulate_titration",
    "speciate",
    "total_alkalinity",
    "total_borate",
    "total_calcium",
    "total_fluoride",
    "total_sulfate",
]
