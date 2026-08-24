"""The classical least-squares inverse: correctness and noise behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from cleaned_reef_water import constants_at, simulate_titration, speciate
from cleaned_reef_water.baseline import LeastSquaresInverse, NuisanceModel
from cleaned_reef_water.ml.noise import NoiseModel, apply_noise

S_REF, T_REF = 35.0, 25.0
ALK, DIC = 2300e-6, 2000e-6


@pytest.fixture(scope="module")
def clean_curve():
    return simulate_titration(ALK, DIC, S_REF, T_REF, n_points=60)


class TestExactRecovery:
    """On a noiseless curve the inverse must be exact."""

    @pytest.mark.parametrize("nuisance", list(NuisanceModel))
    def test_recovers_totals(self, clean_curve, nuisance):
        result = LeastSquaresInverse(nuisance=nuisance).fit(
            clean_curve.titrant_mass, clean_curve.ph_total, S_REF, T_REF
        )
        assert result.success
        assert result.alkalinity == pytest.approx(ALK, rel=1e-6)
        assert result.dic == pytest.approx(DIC, rel=1e-6)

    def test_recovers_speciation(self, clean_curve):
        result = LeastSquaresInverse().fit(
            clean_curve.titrant_mass, clean_curve.ph_total, S_REF, T_REF
        )
        truth = speciate(
            clean_curve.ph_total[0], constants_at(S_REF, T_REF), dic=DIC
        )
        assert result.hco3 == pytest.approx(float(truth.hco3), rel=1e-5)
        assert result.co3 == pytest.approx(float(truth.co3), rel=1e-5)
        assert result.boh4 == pytest.approx(float(truth.boh4), rel=1e-5)

    def test_residual_is_negligible(self, clean_curve):
        result = LeastSquaresInverse().fit(
            clean_curve.titrant_mass, clean_curve.ph_total, S_REF, T_REF
        )
        assert result.residual_rms < 1e-6

    def test_speciation_only_wrapper_agrees(self, clean_curve):
        inverse = LeastSquaresInverse()
        full = inverse.fit(clean_curve.titrant_mass, clean_curve.ph_total, S_REF, T_REF)
        short = inverse.fit_speciation_only(
            clean_curve.titrant_mass, clean_curve.ph_total, S_REF, T_REF
        )
        assert short == pytest.approx((full.hco3, full.co3, full.boh4), rel=1e-12)


class TestNuisanceParameters:
    def test_parameter_counts(self):
        assert NuisanceModel.NONE.n_parameters == 2
        assert NuisanceModel.OFFSET.n_parameters == 3
        assert NuisanceModel.OFFSET_AND_SLOPE.n_parameters == 4

    def test_offset_absorbs_a_systematic_shift(self, clean_curve):
        """A constant pH offset must be recovered, not pushed into A_T."""
        shifted = clean_curve.ph_total + 0.10
        without = LeastSquaresInverse(nuisance=NuisanceModel.NONE).fit(
            clean_curve.titrant_mass, shifted, S_REF, T_REF
        )
        with_offset = LeastSquaresInverse(nuisance=NuisanceModel.OFFSET).fit(
            clean_curve.titrant_mass, shifted, S_REF, T_REF
        )
        assert with_offset.offset == pytest.approx(0.10, abs=1e-3)
        assert abs(with_offset.alkalinity - ALK) < abs(without.alkalinity - ALK)
        assert with_offset.alkalinity == pytest.approx(ALK, rel=1e-3)

    def test_slope_is_unity_when_not_fitted(self, clean_curve):
        result = LeastSquaresInverse(nuisance=NuisanceModel.OFFSET).fit(
            clean_curve.titrant_mass, clean_curve.ph_total, S_REF, T_REF
        )
        assert result.slope == 1.0


class TestNoiseDegradation:
    """Accuracy must degrade in the expected direction and magnitude."""

    def _mae(self, model, nuisance, n=6):
        errors = []
        for seed in range(n):
            curve = simulate_titration(ALK, DIC, S_REF, T_REF, n_points=50)
            noisy = apply_noise(
                curve.titrant_mass,
                curve.ph_total,
                S_REF,
                T_REF,
                model=model,
                rng=np.random.default_rng(seed),
            )
            result = LeastSquaresInverse(nuisance=nuisance).fit(
                noisy.titrant_mass,
                noisy.ph_measured,
                noisy.salinity_measured,
                noisy.t_c_measured,
            )
            errors.append(abs(result.alkalinity - ALK) * 1e6)
        return float(np.mean(errors))

    def test_hobbyist_is_worse_than_laboratory(self):
        lab = self._mae(NoiseModel.laboratory(), NuisanceModel.OFFSET_AND_SLOPE)
        hob = self._mae(NoiseModel.hobbyist(), NuisanceModel.OFFSET_AND_SLOPE)
        assert hob > lab

    def test_fitting_the_offset_helps_hobbyist_alkalinity(self):
        """The NBS bias must be absorbed by the offset parameter."""
        naive = self._mae(NoiseModel.hobbyist(), NuisanceModel.NONE)
        fitted = self._mae(NoiseModel.hobbyist(), NuisanceModel.OFFSET)
        assert fitted < naive / 2.0

    def test_laboratory_accuracy_is_plausible(self):
        mae = self._mae(NoiseModel.laboratory(), NuisanceModel.OFFSET_AND_SLOPE)
        assert mae < 60.0, "laboratory fit should stay well inside 60 umol/kg"


class TestOptimiserBounds:
    """Guard against the divergence found in the 100-curve benchmark."""

    def test_fitted_slope_stays_physical(self):
        inverse = LeastSquaresInverse(nuisance=NuisanceModel.OFFSET_AND_SLOPE)
        model = NoiseModel.laboratory()
        for seed, alk in enumerate((1.2e-3, 1.3e-3, 2.3e-3, 4.0e-3)):
            curve = simulate_titration(alk, DIC, S_REF, T_REF, n_points=60)
            noisy = apply_noise(
                curve.titrant_mass, curve.ph_total, S_REF, T_REF,
                model=model, rng=np.random.default_rng(seed),
            )
            result = inverse.fit(
                noisy.titrant_mass, noisy.ph_measured,
                noisy.salinity_measured, noisy.t_c_measured,
            )
            assert 0.8 <= result.slope <= 1.2, "slope escaped physical bounds"
            assert abs(result.offset) <= 0.5, "offset escaped physical bounds"

    def test_low_alkalinity_does_not_diverge(self):
        """The specific regime that failed: A_T near 1200 umol/kg."""
        inverse = LeastSquaresInverse(nuisance=NuisanceModel.OFFSET_AND_SLOPE)
        alk = 1.22e-3
        curve = simulate_titration(alk, 1.1e-3, S_REF, T_REF, n_points=60)
        noisy = apply_noise(
            curve.titrant_mass, curve.ph_total, S_REF, T_REF,
            model=NoiseModel.laboratory(), rng=np.random.default_rng(79),
        )
        result = inverse.fit(
            noisy.titrant_mass, noisy.ph_measured,
            noisy.salinity_measured, noisy.t_c_measured,
        )
        assert abs(result.alkalinity - alk) * 1e6 < 100.0

    def test_bounds_are_configurable(self):
        inverse = LeastSquaresInverse(slope_bounds=(0.95, 1.05))
        assert inverse.slope_bounds == (0.95, 1.05)


class TestValidation:
    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same shape"):
            LeastSquaresInverse().fit(np.zeros(10), np.zeros(9), S_REF, T_REF)

    def test_too_few_points_rejected(self):
        with pytest.raises(ValueError, match="at least 4"):
            LeastSquaresInverse().fit(np.zeros(3), np.zeros(3), S_REF, T_REF)
