r"""Classical least-squares inversion of a titration curve."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .alkalinity import constants_at, speciate
from .equilibria import CarbonicFormulation
from .seawater import BoronFormulation
from .solvers import ConvergenceError, ph_from_alkalinity_and_totals
from .titration import STRONG_ACID, Titrant, simulate_titration
from .types import FloatArray

__all__ = ["NuisanceModel", "InversionResult", "LeastSquaresInverse"]


class NuisanceModel(enum.Enum):
    """Which instrument nuisance parameters the fit is allowed to absorb.

    Attributes
    ----------
    NONE
        Fit ``(A_T, C_T)`` only.  Assumes a perfectly calibrated electrode.
        Included as a deliberately weak reference; not recommended.
    OFFSET
        Fit ``(A_T, C_T, E0)``.  This is Dickson's (1981) procedure and the
        fair baseline for any comparison.
    OFFSET_AND_SLOPE
        Fit ``(A_T, C_T, E0, s)``.  Absorbs a non-ideal Nernstian slope as
        well, which matters for low-cost electrodes.
    """

    NONE = "none"
    OFFSET = "offset"
    OFFSET_AND_SLOPE = "offset_and_slope"

    @property
    def n_parameters(self) -> int:
        """Total number of free parameters, including ``A_T`` and ``C_T``."""
        return {
            NuisanceModel.NONE: 2,
            NuisanceModel.OFFSET: 3,
            NuisanceModel.OFFSET_AND_SLOPE: 4,
        }[self]


@dataclass(frozen=True, slots=True)
class InversionResult:
    """Outcome of a least-squares inversion.

    Attributes
    ----------
    alkalinity, dic
        Fitted totals in ``mol kg-soln^-1``.
    ph_initial
        Total-scale pH of the undisturbed sample implied by the fit.
    hco3, co3, boh4
        Speciation of the undisturbed sample, ``mol kg-soln^-1``.
    offset, slope
        Fitted nuisance parameters.  ``offset`` is in pH units; ``slope`` is
        dimensionless and equals 1 for an ideal electrode.
    residual_rms
        Root-mean-square pH residual of the fit.
    n_function_evaluations
        Number of residual evaluations used by the optimiser.
    success
        Whether the optimiser reported convergence.
    """

    alkalinity: float
    dic: float
    ph_initial: float
    hco3: float
    co3: float
    boh4: float
    offset: float
    slope: float
    residual_rms: float
    n_function_evaluations: int
    success: bool


@dataclass(frozen=True, slots=True)
class LeastSquaresInverse:
    r"""Estimate ``(A_T, C_T)`` and speciation from a measured titration curve.

    Attributes
    ----------
    sample_mass_kg
        Mass of the titrated sample, in kg.
    titrant
        Titrant model.  Must match the one actually used.
    nuisance
        Which nuisance parameters to fit.
    carbonic_formulation
        Carbonic acid parameterisation, which must match the one assumed when
        interpreting the result.
    boron_formulation
        Boron-to-salinity relationship.
    n_curve_points
        Number of points used when evaluating the forward model.  Defaults to
        matching the observed curve.
    max_nfev
        Optimiser evaluation budget.
    """

    sample_mass_kg: float = 0.015
    titrant: Titrant = STRONG_ACID
    nuisance: NuisanceModel = NuisanceModel.OFFSET
    carbonic_formulation: CarbonicFormulation = CarbonicFormulation.LUEKER2000
    boron_formulation: BoronFormulation = BoronFormulation.UPPSTROM1974
    calibration_ph: float = 8.0936
    max_nfev: int = 400
    alkalinity_bounds_mmol: tuple[float, float] = (0.1, 10.0)
    dic_bounds_mmol: tuple[float, float] = (0.1, 10.0)
    offset_bounds_ph: tuple[float, float] = (-0.5, 0.5)
    slope_bounds: tuple[float, float] = (0.8, 1.2)

    def _forward(
        self,
        alkalinity: float,
        dic: float,
        salinity: float,
        t_c: float,
        n_points: int,
        max_fraction: float,
    ) -> FloatArray:
        """Evaluate the forward model on the observed titrant grid."""
        curve = simulate_titration(
            alkalinity,
            dic,
            salinity,
            t_c,
            titrant=self.titrant,
            sample_mass_kg=self.sample_mass_kg,
            n_points=n_points,
            max_equivalence_fraction=max_fraction,
            boron_formulation=self.boron_formulation,
            carbonic_formulation=self.carbonic_formulation,
        )
        return curve.ph_total

    def fit(
        self,
        titrant_mass: FloatArray,
        ph_observed: FloatArray,
        salinity: float,
        t_c: float,
        *,
        initial_alkalinity: float = 2.3e-3,
        initial_dic: float = 2.0e-3,
    ) -> InversionResult:
        """Invert one titration curve.

        Parameters
        ----------
        titrant_mass
            Observed titrant mass at each point, in kg.
        ph_observed
            Observed pH at each point, total scale.
        salinity, t_c
            Measured salinity and temperature.  Errors in these propagate into
            the result, which is intentional -- a real instrument does not know
            them exactly.
        initial_alkalinity, initial_dic
            Starting guesses in ``mol kg-soln^-1``.

        Returns
        -------
        InversionResult
            Fitted totals, speciation and nuisance parameters.

        Raises
        ------
        ValueError
            If the inputs have different lengths or fewer than four points,
            which is below the number of free parameters.
        """
        mass = np.asarray(titrant_mass, dtype=np.float64)
        observed = np.asarray(ph_observed, dtype=np.float64)
        if mass.shape != observed.shape:
            raise ValueError(
                f"titrant_mass and ph_observed must have the same shape, got "
                f"{mass.shape} and {observed.shape}"
            )
        n_points = int(mass.size)
        if n_points < 4:
            raise ValueError(
                f"need at least 4 titration points to fit, got {n_points}"
            )

        # Reproduce the observed titrant grid as a multiple of the equivalence mass.
        final_mass = float(mass[-1])

        def unpack(p: FloatArray) -> tuple[float, float, float, float]:
            alkalinity = float(p[0]) * 1e-3
            dic = float(p[1]) * 1e-3
            offset = float(p[2]) if self.nuisance.n_parameters >= 3 else 0.0
            slope = float(p[3]) if self.nuisance.n_parameters >= 4 else 1.0
            return alkalinity, dic, offset, slope

        def residual(p: FloatArray) -> FloatArray:
            alkalinity, dic, offset, slope = unpack(p)
            if alkalinity <= 0.0 or dic <= 0.0:
                return np.full(n_points, 1e3)
            equivalence = self.sample_mass_kg * alkalinity / (
                self.titrant.titratable_capacity()
            )
            if equivalence <= 0.0:
                return np.full(n_points, 1e3)
            try:
                modelled = self._forward(
                    alkalinity,
                    dic,
                    salinity,
                    t_c,
                    n_points,
                    final_mass / equivalence,
                )
            except (ConvergenceError, ValueError):
                return np.full(n_points, 1e3)
            apparent = (
                self.calibration_ph
                + slope * (modelled - self.calibration_ph)
                + offset
            )
            return apparent - observed

        n_free = self.nuisance.n_parameters
        start = np.array(
            [initial_alkalinity * 1e3, initial_dic * 1e3, 0.0, 1.0][:n_free]
        )
        # Physically motivated bounds.
        lower = np.array(
            [
                self.alkalinity_bounds_mmol[0],
                self.dic_bounds_mmol[0],
                self.offset_bounds_ph[0],
                self.slope_bounds[0],
            ][:n_free]
        )
        upper = np.array(
            [
                self.alkalinity_bounds_mmol[1],
                self.dic_bounds_mmol[1],
                self.offset_bounds_ph[1],
                self.slope_bounds[1],
            ][:n_free]
        )
        start = np.clip(start, lower, upper)
        solution = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            xtol=1e-12,
            ftol=1e-12,
            max_nfev=self.max_nfev,
        )
        alkalinity, dic, offset, slope = unpack(solution.x)

        constants = constants_at(salinity, t_c, self.carbonic_formulation)
        try:
            ph_initial = ph_from_alkalinity_and_totals(
                alkalinity,
                constants,
                dic=dic,
                boron_formulation=self.boron_formulation,
            )
            species = speciate(
                ph_initial,
                constants,
                dic=dic,
                boron_formulation=self.boron_formulation,
            )
            hco3, co3, boh4 = (
                float(species.hco3),
                float(species.co3),
                float(species.boh4),
            )
        except ConvergenceError:
            ph_initial = float("nan")
            hco3 = co3 = boh4 = float("nan")

        return InversionResult(
            alkalinity=alkalinity,
            dic=dic,
            ph_initial=float(ph_initial),
            hco3=hco3,
            co3=co3,
            boh4=boh4,
            offset=offset,
            slope=slope,
            residual_rms=float(np.sqrt(np.mean(solution.fun**2))),
            n_function_evaluations=int(solution.nfev),
            success=bool(solution.success),
        )

    def fit_speciation_only(
        self,
        titrant_mass: FloatArray,
        ph_observed: FloatArray,
        salinity: float,
        t_c: float,
    ) -> tuple[float, float, float]:
        """Return only ``(HCO3, CO3, B(OH)4)`` in ``mol kg-soln^-1``.

        Parameters
        ----------
        titrant_mass, ph_observed
            The observed curve.
        salinity, t_c
            Measured environmental conditions.

        Returns
        -------
        tuple of float
            ``(hco3, co3, boh4)``.
        """
        result = self.fit(titrant_mass, ph_observed, salinity, t_c)
        return result.hco3, result.co3, result.boh4
