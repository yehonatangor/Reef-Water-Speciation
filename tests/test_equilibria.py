"""Benchmark every equilibrium constant against published reference values."""

from __future__ import annotations

import numpy as np
import pytest

from cleaned_reef_water.equilibria import (
    CarbonicFormulation,
    k_aragonite,
    k_bisulfate_free,
    k_borate_total,
    k_calcite,
    k_carbonic_total,
    k_co2_solubility,
    k_fluoride_free,
    k_fluoride_total,
    k_phosphoric_sws,
    k_phosphoric_total,
    k_silicate_total,
    k_water_sws,
    k_water_total,
)

S_REF = 35.0
T_REF = 25.0


class TestGuideCheckValues:
    """Constants that must reproduce the Guide's printed check values exactly."""

    @pytest.mark.parametrize(
        ("label", "value", "expected", "decimals"),
        [
            ("ln K0", np.log(k_co2_solubility(S_REF, T_REF)), -3.5617, 4),
            ("ln KS", np.log(k_bisulfate_free(S_REF, T_REF)), -2.30, 2),
            ("ln KB", np.log(k_borate_total(S_REF, T_REF)), -19.7964, 4),
            ("ln KF", np.log(k_fluoride_total(S_REF, T_REF)), -6.09, 2),
        ],
    )
    def test_guide_reference(self, label, value, expected, decimals):
        assert round(float(value), decimals) == pytest.approx(
            expected, abs=10**-decimals
        ), label

    def test_carbonic_lueker_matches_guide(self):
        k1, k2 = k_carbonic_total(S_REF, T_REF, CarbonicFormulation.LUEKER2000)
        assert float(np.log10(k1)) == pytest.approx(-5.8472, abs=1e-4)
        assert float(np.log10(k2)) == pytest.approx(-8.9660, abs=1e-4)


class TestScaleConversionDeparture:
    """Constants where this package departs from the Guide's 0.015 shortcut."""

    def test_water_sws_matches_millero1995(self):
        """Millero (1995) as published, before any scale conversion."""
        value = float(np.log(k_water_sws(S_REF, T_REF)))
        assert value == pytest.approx(-30.4187, abs=1e-3)

    def test_water_total_is_exact_conversion_not_shortcut(self):
        exact = k_water_total(S_REF, T_REF)
        shortcut = k_water_sws(S_REF, T_REF) * np.exp(-0.015)
        # The Guide's shortcut and the exact conversion differ by 0.73 %.
        assert float(exact / shortcut) == pytest.approx(np.exp(-0.00729), rel=1e-3)

    def test_phosphoric_sws_matches_millero1995_constants(self):
        """The published SWS constants are the total values plus 0.015."""
        sws = k_phosphoric_sws(S_REF, T_REF)
        total = k_phosphoric_total(S_REF, T_REF)
        for k_sws, k_total in zip(sws, total, strict=True):
            assert float(np.log(k_sws) - np.log(k_total)) == pytest.approx(
                0.02229, abs=1e-4
            )

    def test_silicate_and_water_use_the_same_conversion_factor(self):
        from cleaned_reef_water.equilibria import k_silicate_sws

        ratio_si = float(k_silicate_total(S_REF, T_REF) / k_silicate_sws(S_REF, T_REF))
        ratio_w = float(k_water_total(S_REF, T_REF) / k_water_sws(S_REF, T_REF))
        assert ratio_si == pytest.approx(ratio_w, rel=1e-12)


class TestMucciSolubility:
    """Mucci (1983) internal consistency and published ratios."""

    def test_aragonite_calcite_ratio_at_25c(self):
        """Mucci constrained the ratio to 1.50 +/- 0.03 at 25 degC, S = 0."""
        # At S = 0 the salinity terms vanish and only the log K^0 terms remain.
        import warnings
        warnings.simplefilter("ignore", UserWarning)
        log_ratio = float(
            np.log10(k_aragonite(0.0, 25.0)) - np.log10(k_calcite(0.0, 25.0))
        )
        assert 10**log_ratio == pytest.approx(1.50, abs=0.03)

    def test_ratio_matches_analytic_difference(self):
        """Log Ka - log Kc must equal -0.0385 + 63.974/T identically."""
        temp = 25.0 + 273.15
        expected = -0.0385 + 63.974 / temp
        actual = float(
            np.log10(k_aragonite(0.0, 25.0)) - np.log10(k_calcite(0.0, 25.0))
        )
        assert actual == pytest.approx(expected, abs=1e-9)

    def test_aragonite_more_soluble_than_calcite(self):
        assert float(k_aragonite(S_REF, T_REF)) > float(k_calcite(S_REF, T_REF))


class TestCarbonicFormulations:
    """All four parameterisations must be plausible and mutually consistent."""

    @pytest.mark.parametrize("formulation", list(CarbonicFormulation))
    def test_pk_values_are_physically_plausible(self, formulation):
        k1, k2 = k_carbonic_total(S_REF, T_REF, formulation)
        assert 5.7 < -float(np.log10(k1)) < 6.0
        assert 8.8 < -float(np.log10(k2)) < 9.1
        assert float(k1) > float(k2), "K1 must exceed K2"

    def test_formulations_agree_within_published_uncertainty(self):
        """Spread across formulations should sit near the quoted 0.02 in pK."""
        pk1 = [
            -float(np.log10(k_carbonic_total(S_REF, T_REF, f)[0]))
            for f in CarbonicFormulation
        ]
        assert max(pk1) - min(pk1) < 0.05

    def test_dickson_millero_1987_paper_check_value_discrepancy(self):
        """Document the internal inconsistency in Dickson & Millero (1987)."""
        temp = 298.15
        ln_t, sqrt_s = np.log(temp), np.sqrt(35.0)
        pk1_0 = 6320.81 / temp - 126.3405 + 19.568 * ln_t
        pk2_0 = 5143.69 / temp - 90.1833 + 14.613 * ln_t
        pk1 = (
            pk1_0
            + (-840.39 / temp + 19.894 - 3.0189 * ln_t) * sqrt_s
            + 0.00668 * 35.0
        )
        pk2 = pk2_0 + (-690.59 / temp + 17.176 - 2.6719 * ln_t) * sqrt_s + 0.0217 * 35.0
        assert pk2 == pytest.approx(8.9358, abs=1e-4), "pK2 matches the paper exactly"
        assert pk1 == pytest.approx(5.8435, abs=1e-4), "pK1 evaluates to 5.8435"
        assert abs(pk1 - 5.8477) > 3e-3, "paper's printed check value is 5.8477"


class TestRangeWarnings:
    """Extrapolation is allowed but must never be silent."""

    def test_warns_outside_salinity_range(self):
        with pytest.warns(UserWarning, match="salinity outside fitted range"):
            k_carbonic_total(5.0, 25.0, CarbonicFormulation.LUEKER2000)

    def test_warns_outside_temperature_range(self):
        with pytest.warns(UserWarning, match="temperature outside fitted range"):
            k_carbonic_total(35.0, 40.0, CarbonicFormulation.LUEKER2000)

    def test_no_warning_inside_range(self, recwarn):
        k_carbonic_total(35.0, 25.0, CarbonicFormulation.LUEKER2000)
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


class TestVectorisation:
    """Every constant must broadcast over arrays."""

    @pytest.mark.parametrize(
        "func",
        [
            k_co2_solubility,
            k_bisulfate_free,
            k_borate_total,
            k_water_total,
            k_silicate_total,
            k_calcite,
            k_aragonite,
            k_fluoride_free,
        ],
    )
    def test_broadcasts(self, func):
        salinity = np.array([30.0, 35.0, 38.0])
        result = func(salinity, 25.0)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_scalar_and_array_agree(self):
        scalar = float(k_borate_total(35.0, 25.0))
        array = k_borate_total(np.array([35.0]), np.array([25.0]))
        assert float(array[0]) == pytest.approx(scalar, rel=1e-15)


class TestMonotonicity:
    """Qualitative thermodynamic behaviour."""

    def test_k1_increases_with_salinity(self):
        salinities = np.array([20.0, 25.0, 30.0, 35.0, 40.0])
        k1, _ = k_carbonic_total(salinities, 25.0)
        assert np.all(np.diff(k1) > 0)

    def test_kw_increases_with_temperature(self):
        temps = np.array([5.0, 15.0, 25.0, 35.0])
        assert np.all(np.diff(k_water_total(35.0, temps)) > 0)

    def test_calcite_solubility_has_a_maximum_near_10c(self):
        """K_sp of calcite is not monotone in temperature."""
        temps = np.arange(5.0, 40.0, 1.0)
        ksp = k_calcite(35.0, temps)
        peak = float(temps[int(np.argmax(ksp))])
        assert 8.0 <= peak <= 13.0
        # Strictly decreasing once past the maximum.
        assert np.all(np.diff(ksp[temps >= 15.0]) < 0)
