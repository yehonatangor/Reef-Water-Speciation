r"""Physically grounded instrument noise model for titration curves."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import FARADAY, GAS_CONSTANT, KELVIN_OFFSET, LN10, FloatArray

__all__ = ["NoiseModel", "NoisyCurve", "apply_noise"]


@dataclass(frozen=True, slots=True)
class NoiseModel:
    """Tolerances of a realistic automated alkalinity titrator.

    Attributes
    ----------
    calibration_ph
        pH of the buffer or certified reference material against which the
        electrode was calibrated.  Slope errors pivot about this value.  The
        default of ``8.0936`` is the Tris CRM value quoted in Guide SOP 6a.
    slope_error_sd
        Standard deviation of the fractional Nernstian slope error.  A slope
        of 99.5 % of theoretical is a typical acceptance limit.
    offset_sd
        Standard deviation of the calibration offset, in pH units.
    electrode_noise_sd
        Standard deviation of the random electrode noise, in pH units.
    ar1_rho_range
        Range from which the AR(1) autocorrelation coefficient is drawn.
    junction_drift_sd
        Standard deviation of the total liquid-junction drift accumulated
        over the whole run, in pH units.
    co2_loss_fraction_range
        Range of the fraction of dissolved CO2 lost by degassing by the end
        of the run.
    co2_time_constant_range
        Range of the degassing time constant, expressed as a fraction of the
        total run duration.
    burette_scale_error_sd
        Standard deviation of the systematic burette scale error, as a
        fraction.
    burette_jitter_sd
        Standard deviation of per-step dispensing jitter, as a fraction of the
        step size.
    temperature_error_sd
        Standard deviation of the temperature measurement error, in degrees
        Celsius.
    salinity_error_sd
        Standard deviation of the practical salinity measurement error.
    quantisation_step_ph
        ADC resolution in pH units.  ``0.0`` disables quantisation.  Use
        :meth:`quantisation_from_adc` to derive it from bit depth and range.
    nbs_calibration_bias_mean, nbs_calibration_bias_sd
        Mean and spread of the systematic offset introduced by calibrating
        against NBS-scale pH 4/7/10 buffers rather than a Tris CRM.  Set the
        mean to ``0.0`` for CRM-calibrated instruments.
    """

    calibration_ph: float = 8.0936
    slope_error_sd: float = 0.004
    offset_sd: float = 0.005
    electrode_noise_sd: float = 0.003
    ar1_rho_range: tuple[float, float] = (0.3, 0.7)
    junction_drift_sd: float = 0.010
    co2_loss_fraction_range: tuple[float, float] = (0.0, 0.05)
    co2_time_constant_range: tuple[float, float] = (0.1, 0.5)
    burette_scale_error_sd: float = 0.0015
    burette_jitter_sd: float = 0.0008
    temperature_error_sd: float = 0.05
    salinity_error_sd: float = 0.02
    quantisation_step_ph: float = 0.0
    nbs_calibration_bias_mean: float = 0.0
    nbs_calibration_bias_sd: float = 0.0

    @staticmethod
    def quantisation_from_adc(
        bits: int, full_scale_volts: float, t_c: float = 25.0
    ) -> float:
        """ADC resolution expressed in pH units.

        Parameters
        ----------
        bits
            ADC bit depth.
        full_scale_volts
            Full-scale input span of the ADC, in volts.
        t_c
            Temperature in degrees Celsius, which sets the Nernstian slope.

        Returns
        -------
        float
            pH resolution per least significant bit.

        Examples
        --------
        A 10-bit ADC over 5 V -- the classic Arduino front end:

        >>> round(NoiseModel.quantisation_from_adc(10, 5.0), 4)
        0.0825

        A 16-bit ADS1115 over +/-2.048 V is effectively continuous:

        >>> round(NoiseModel.quantisation_from_adc(16, 4.096), 5)
        0.00106
        """
        if bits <= 0:
            raise ValueError(f"bits must be positive, got {bits}")
        if full_scale_volts <= 0.0:
            raise ValueError(
                f"full_scale_volts must be positive, got {full_scale_volts}"
            )
        millivolts_per_lsb = full_scale_volts * 1000.0 / (2**bits)
        slope = (GAS_CONSTANT * (t_c + KELVIN_OFFSET) * LN10 / FARADAY) * 1000.0
        return float(millivolts_per_lsb / slope)

    @classmethod
    def laboratory(cls) -> NoiseModel:
        """Tolerances of a well-maintained automated titrator (Guide SOP 3b).

        Returns
        -------
        NoiseModel
            Laboratory-grade tolerances.
        """
        return cls()

    @classmethod
    def hobbyist(cls) -> NoiseModel:
        """Tolerances of a sub-USD-200 pH probe and a stepper syringe pump.

        Returns
        -------
        NoiseModel
            Hobbyist-grade tolerances.
        """
        return cls(
            slope_error_sd=0.03,
            offset_sd=0.03,
            electrode_noise_sd=0.012,
            ar1_rho_range=(0.4, 0.85),
            junction_drift_sd=0.050,
            co2_loss_fraction_range=(0.0, 0.12),
            co2_time_constant_range=(0.1, 0.5),
            burette_scale_error_sd=0.008,
            burette_jitter_sd=0.004,
            temperature_error_sd=0.3,
            salinity_error_sd=0.5,
            quantisation_step_ph=cls.quantisation_from_adc(16, 4.096),
            nbs_calibration_bias_mean=0.13,
            nbs_calibration_bias_sd=0.04,
        )

    def nernst_slope_mv_per_ph(self, t_c: float) -> float:
        """Theoretical Nernstian slope in mV per pH unit at ``t_c``.

        Parameters
        ----------
        t_c
            Temperature in degrees Celsius.

        Returns
        -------
        float
            ``RT ln(10) / F`` in millivolts per pH unit -- 59.16 mV at 25 degC.

        Examples
        --------
        >>> float(round(NoiseModel().nernst_slope_mv_per_ph(25.0), 2))
        59.16
        """
        return (
            GAS_CONSTANT * (t_c + KELVIN_OFFSET) * LN10 / FARADAY
        ) * 1000.0


@dataclass(frozen=True, slots=True)
class NoisyCurve:
    """A titration curve with instrument error applied.

    Attributes
    ----------
    titrant_mass
        Apparent titrant mass, carrying burette error.
    ph_measured
        Apparent pH, carrying electrode error.
    salinity_measured, t_c_measured
        Apparent environmental conditions.
    truth
        The realised nuisance parameters, retained for sensitivity analysis.
    """

    titrant_mass: FloatArray
    ph_measured: FloatArray
    salinity_measured: float
    t_c_measured: float
    truth: dict[str, float] = field(default_factory=dict)


def _ar1_series(
    n: int, rho: float, sd: float, rng: np.random.Generator
) -> FloatArray:
    """Stationary AR(1) series with marginal standard deviation ``sd``."""
    white = rng.normal(0.0, sd, n)
    series = np.empty(n, dtype=np.float64)
    series[0] = white[0]
    scale = np.sqrt(1.0 - rho**2)
    for i in range(1, n):
        series[i] = rho * series[i - 1] + scale * white[i]
    return series


def apply_noise(
    titrant_mass: FloatArray,
    ph_true: FloatArray,
    salinity: float,
    t_c: float,
    *,
    model: NoiseModel | None = None,
    rng: np.random.Generator | None = None,
) -> NoisyCurve:
    """Apply the full instrument noise model to a clean titration curve.

    Parameters
    ----------
    titrant_mass
        Clean titrant mass at each point, in kg.
    ph_true
        Clean equilibrium pH at each point, total scale.
    salinity
        True practical salinity.
    t_c
        True temperature in degrees Celsius.
    model
        Tolerances to use; defaults to :class:`NoiseModel`.
    rng
        Random generator.  A fresh default generator is created if ``None``.

    Returns
    -------
    NoisyCurve
        The perturbed curve and the realised nuisance parameters.

    Raises
    ------
    ValueError
        If ``titrant_mass`` and ``ph_true`` have different lengths, or fewer
        than two points.
    """
    if model is None:
        model = NoiseModel()
    if rng is None:
        rng = np.random.default_rng()

    n = int(np.size(ph_true))
    if np.size(titrant_mass) != n:
        raise ValueError(
            f"titrant_mass and ph_true must have equal length, got "
            f"{np.size(titrant_mass)} and {n}"
        )
    if n < 2:
        raise ValueError("a titration curve needs at least two points")

    ph = np.asarray(ph_true, dtype=np.float64).copy()
    mass = np.asarray(titrant_mass, dtype=np.float64).copy()
    progress = np.linspace(0.0, 1.0, n)

    # --- 1.
    loss_fraction = rng.uniform(*model.co2_loss_fraction_range)
    tau = rng.uniform(*model.co2_time_constant_range)
    escaped = loss_fraction * (1.0 - np.exp(-progress / tau))
    ph = ph - np.log10(1.0 - escaped)

    # --- 2. Nernstian slope error and calibration offset --------------------
    slope = 1.0 + rng.normal(0.0, model.slope_error_sd)
    offset = rng.normal(0.0, model.offset_sd)
    ph = model.calibration_ph + slope * (ph - model.calibration_ph) + offset

    # --- 3. Correlated electrode noise --------------------------------------
    rho = rng.uniform(*model.ar1_rho_range)
    ph = ph + _ar1_series(n, rho, model.electrode_noise_sd, rng)

    # --- 4. Liquid junction drift -------------------------------------------
    drift_total = rng.normal(0.0, model.junction_drift_sd)
    ph = ph + drift_total * progress

    # --- 4b.
    nbs_bias = 0.0
    if model.nbs_calibration_bias_mean or model.nbs_calibration_bias_sd:
        nbs_bias = rng.normal(
            model.nbs_calibration_bias_mean, model.nbs_calibration_bias_sd
        )
        ph = ph + nbs_bias

    # --- 5. Burette scale error and per-step jitter -------------------------
    scale_error = rng.normal(0.0, model.burette_scale_error_sd)
    steps = np.diff(mass, prepend=0.0)
    jitter = rng.normal(0.0, model.burette_jitter_sd, n) * steps
    mass = np.cumsum(steps * (1.0 + scale_error) + jitter)
    mass[0] = 0.0

    # --- 6.
    if model.quantisation_step_ph > 0.0:
        step = model.quantisation_step_ph
        ph = np.round(ph / step) * step

    # --- 7. Environmental measurement error ---------------------------------
    salinity_measured = salinity + rng.normal(0.0, model.salinity_error_sd)
    t_c_measured = t_c + rng.normal(0.0, model.temperature_error_sd)

    return NoisyCurve(
        titrant_mass=mass,
        ph_measured=ph,
        salinity_measured=float(salinity_measured),
        t_c_measured=float(t_c_measured),
        truth={
            "slope": float(slope),
            "offset": float(offset),
            "ar1_rho": float(rho),
            "junction_drift": float(drift_total),
            "co2_loss_fraction": float(loss_fraction),
            "co2_tau": float(tau),
            "burette_scale_error": float(scale_error),
            "nbs_calibration_bias": float(nbs_bias),
        },
    )
