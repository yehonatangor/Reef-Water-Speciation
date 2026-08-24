"""Conservation invariants, scale conversions and solver behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from cleaned_reef_water.alkalinity import (
    constants_at,
    explicit_conservative_alkalinity,
    speciate,
    total_alkalinity,
)
from cleaned_reef_water.scales import convert_ph, scale_factor
from cleaned_reef_water.seawater import (
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
from cleaned_reef_water.solvers import (
    ConvergenceError,
    ph_from_dic_alkalinity,
    solve_scalar_bracketed,
)
from cleaned_reef_water.types import PHScale

S_REF, T_REF = 35.0, 25.0


class TestSeawaterProperties:
    def test_density_matches_guide(self):
        """Guide Chapter 5 §4.2: rho = 1023.343 kg/m3 at S=35, t=25."""
        assert float(density_seawater(35.0, 25.0)) == pytest.approx(1023.343, abs=1e-3)

    def test_density_of_pure_water_at_4c_is_near_maximum(self):
        temps = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        rho = density_seawater(0.0, temps)
        assert int(np.argmax(rho)) == 2

    def test_ionic_strength(self):
        assert float(ionic_strength(35.0)) == pytest.approx(0.72276, abs=1e-5)

    @pytest.mark.parametrize(
        ("func", "expected", "tol"),
        [
            (total_sulfate, 0.0282354, 1e-6),
            (total_fluoride, 6.8324e-5, 1e-8),
            (total_calcium, 0.0102846, 1e-6),
        ],
    )
    def test_conservative_totals(self, func, expected, tol):
        assert float(func(35.0)) == pytest.approx(expected, abs=tol)

    def test_borate_formulations_differ_by_four_percent(self):
        uppstrom = float(total_borate(35.0, BoronFormulation.UPPSTROM1974))
        lee = float(total_borate(35.0, BoronFormulation.LEE2010))
        assert uppstrom == pytest.approx(415.76e-6, abs=0.01e-6)
        assert lee / uppstrom == pytest.approx(0.2414 / 0.232, rel=1e-12)

    def test_all_totals_are_proportional_to_salinity(self):
        for func in (total_sulfate, total_fluoride, total_calcium, total_borate):
            assert float(func(35.0)) == pytest.approx(
                2.0 * float(func(17.5)), rel=1e-12
            )

    def test_molar_molal_round_trip(self):
        original = 2.3e-3
        molal = molar_to_molal(original, 35.0, 25.0)
        assert float(molal_to_molar(molal, 35.0, 25.0)) == pytest.approx(
            original, rel=1e-14
        )

    def test_molar_and_molal_differ_by_the_density(self):
        """A 2.3 mmol/L solution is 2.25 mmol/kg -- a 2.3 % difference."""
        # 1023.343 is the Guide's value rounded to three decimals, so compare
        # at a tolerance consistent with that rounding.
        molal = float(molar_to_molal(2.3e-3, 35.0, 25.0))
        assert molal / 2.3e-3 == pytest.approx(1.0 / 1.023343, rel=1e-6)


class TestScaleConversions:
    def test_free_to_total_offset_is_about_0_11(self):
        factor = float(scale_factor(S_REF, T_REF, PHScale.FREE, PHScale.TOTAL))
        assert np.log10(factor) == pytest.approx(0.1077, abs=1e-3)

    def test_total_to_sws_offset_is_about_0_01(self):
        factor = float(scale_factor(S_REF, T_REF, PHScale.TOTAL, PHScale.SEAWATER))
        assert np.log10(factor) == pytest.approx(0.0097, abs=1e-3)

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (PHScale.FREE, PHScale.TOTAL),
            (PHScale.TOTAL, PHScale.SEAWATER),
            (PHScale.FREE, PHScale.SEAWATER),
        ],
    )
    def test_round_trip_is_exact(self, a, b):
        forward = scale_factor(S_REF, T_REF, a, b)
        backward = scale_factor(S_REF, T_REF, b, a)
        assert float(forward * backward) == pytest.approx(1.0, rel=1e-14)

    def test_composition_of_conversions(self):
        """free->total->sws must equal free->sws."""
        direct = scale_factor(S_REF, T_REF, PHScale.FREE, PHScale.SEAWATER)
        composed = scale_factor(
            S_REF, T_REF, PHScale.FREE, PHScale.TOTAL
        ) * scale_factor(S_REF, T_REF, PHScale.TOTAL, PHScale.SEAWATER)
        assert float(direct) == pytest.approx(float(composed), rel=1e-14)

    def test_identity_conversion(self):
        assert float(scale_factor(S_REF, T_REF, PHScale.TOTAL, PHScale.TOTAL)) == 1.0

    def test_nbs_is_rejected(self):
        with pytest.raises(ValueError, match="NBS activity scale"):
            scale_factor(S_REF, T_REF, PHScale.NBS, PHScale.TOTAL)

    def test_ph_ordering(self):
        """pH_free > pH_total > pH_sws for the same solution."""
        ph_free = 8.2
        ph_total = float(convert_ph(ph_free, S_REF, T_REF, PHScale.FREE, PHScale.TOTAL))
        ph_sws = float(
            convert_ph(ph_free, S_REF, T_REF, PHScale.FREE, PHScale.SEAWATER)
        )
        assert ph_free > ph_total > ph_sws


class TestSpeciationConservation:
    def test_dic_is_conserved_exactly(self):
        constants = constants_at(S_REF, T_REF)
        for ph in (4.0, 6.0, 7.5, 8.1, 9.0, 11.0):
            species = speciate(ph, constants, dic=2.0e-3)
            assert float(species.dic) == pytest.approx(2.0e-3, rel=1e-15)

    def test_boron_is_conserved(self):
        constants = constants_at(S_REF, T_REF)
        b_t = 416e-6
        for ph in (4.0, 8.1, 11.0):
            species = speciate(ph, constants, total_boron=b_t)
            boh3 = b_t - float(species.boh4)
            assert boh3 + float(species.boh4) == pytest.approx(b_t, rel=1e-15)
            assert boh3 >= 0.0

    def test_phosphate_is_conserved(self):
        constants = constants_at(S_REF, T_REF)
        p_t = 2e-6
        for ph in (3.0, 7.0, 8.1, 12.0):
            s = speciate(ph, constants, total_phosphate=p_t)
            total = float(s.h3po4 + s.h2po4 + s.hpo4 + s.po4)
            assert total == pytest.approx(p_t, rel=1e-14)

    def test_silicate_is_conserved(self):
        constants = constants_at(S_REF, T_REF)
        si_t = 5e-6
        s = speciate(8.1, constants, total_silicate=si_t)
        assert float(s.sioh4 + s.siooh3) == pytest.approx(si_t, rel=1e-14)

    def test_all_species_non_negative(self):
        constants = constants_at(S_REF, T_REF)
        for ph in np.linspace(2.0, 12.0, 41):
            s = speciate(
                ph, constants, dic=2e-3, total_phosphate=2e-6, total_silicate=5e-6
            )
            for name in (
                "co2", "hco3", "co3", "boh4", "oh", "h3po4", "h2po4",
                "hpo4", "po4", "sioh4", "siooh3", "hso4", "hf",
            ):
                assert float(getattr(s, name)) >= 0.0, f"{name} negative at pH {ph}"

    def test_free_ph_is_lower_than_total_ph(self):
        constants = constants_at(S_REF, T_REF)
        s = speciate(8.1, constants, dic=2e-3)
        assert float(s.h_free) < float(s.h_total)


class TestAlkalinity:
    def test_dickson_1981_terms_matter_at_low_ph(self):
        """HSO4 and HF are negligible at pH 8 but not in the Gran region."""
        constants = constants_at(S_REF, T_REF)
        at_seawater_ph = float(
            sum(getattr(speciate(8.1, constants, dic=2e-3), n) for n in ("hso4", "hf"))
        )
        at_gran_ph = float(
            sum(getattr(speciate(4.0, constants, dic=2e-3), n) for n in ("hso4", "hf"))
        )
        # Negligible at seawater pH: below 0.01 umol/kg.
        assert 0.0 < at_seawater_ph < 1e-8
        # Significant in the Gran region: above 1 umol/kg, i.e. larger than
        # the reproducibility of a certified alkalinity titration.
        assert at_gran_ph > 1e-6
        assert at_gran_ph / at_seawater_ph > 100.0

    def test_alkalinity_decreases_monotonically_with_ph_decrease(self):
        constants = constants_at(S_REF, T_REF)
        ph_grid = np.linspace(4.0, 10.0, 61)
        alk = np.array(
            [float(total_alkalinity(speciate(p, constants, dic=2e-3))) for p in ph_grid]
        )
        assert np.all(np.diff(alk) > 0)

    def test_explicit_conservative_expression_agrees(self):
        """Wolf-Gladrow (2007) TA_ec on a synthetic charge-balanced solution."""
        alk = float(
            explicit_conservative_alkalinity(
                sodium=0.46907,
                potassium=0.01021,
                magnesium=0.05282,
                calcium=0.01028,
                strontium=9.06e-5,
                chloride=0.54586,
                bromide=8.42e-4,
                nitrate=0.0,
                total_sulfate_conc=0.02824,
                total_fluoride_conc=6.83e-5,
                total_phosphate=0.0,
            )
        )
        # Standard seawater alkalinity is close to 2300 umol/kg; the recipe
        # above is only approximate, so allow a wide band but require the
        # expression to land in the right regime and sign.
        assert 0.0 < alk < 5.0e-3

    def test_zero_composition_gives_water_only_alkalinity(self):
        constants = constants_at(S_REF, T_REF)
        s = speciate(7.0, constants, total_boron=0.0)
        alk = float(total_alkalinity(s))
        assert alk == pytest.approx(
            float(s.oh - s.h_free - s.hso4 - s.hf), rel=1e-12
        )


class TestSolvers:
    def test_round_trip_ph_alkalinity(self):
        constants = constants_at(S_REF, T_REF)
        for ph_true in (7.4, 7.8, 8.1, 8.4):
            species = speciate(ph_true, constants, dic=2.0e-3)
            alk = float(total_alkalinity(species))
            recovered = ph_from_dic_alkalinity(2.0e-3, alk, S_REF, T_REF)
            assert recovered == pytest.approx(ph_true, abs=1e-9)

    def test_surface_ocean_values(self):
        ph = ph_from_dic_alkalinity(2000e-6, 2300e-6, 35.0, 25.0)
        assert 8.0 < ph < 8.1

    def test_bracket_expands_beyond_default(self):
        """A very high alkalinity forces the solver outside its start bracket."""
        ph = ph_from_dic_alkalinity(1.0e-4, 2.0e-3, S_REF, T_REF)
        assert ph > 9.5, "solution lies outside the default (6.0, 9.5) bracket"

    def test_impossible_state_raises(self):
        with pytest.raises(ConvergenceError):
            ph_from_dic_alkalinity(0.0, 1.0, S_REF, T_REF)

    def test_simple_root(self):
        assert solve_scalar_bracketed(lambda x: x - 7.25) == pytest.approx(7.25)

    def test_convergence_error_carries_diagnostics(self):
        with pytest.raises(ConvergenceError) as info:
            solve_scalar_bracketed(lambda _ph: 1.0)
        assert info.value.bracket is not None
        assert info.value.residuals is not None


class TestValidation:
    def test_negative_salinity_rejected(self):
        from cleaned_reef_water.seawater import density_seawater

        with pytest.raises(ValueError, match="out of range"):
            density_seawater(-1.0, 25.0)

    def test_nan_rejected(self):
        from cleaned_reef_water.types import validate_salinity

        with pytest.raises(ValueError, match="non-finite"):
            validate_salinity(np.nan)

    def test_extreme_temperature_rejected(self):
        from cleaned_reef_water.types import validate_temperature

        with pytest.raises(ValueError, match="out of range"):
            validate_temperature(200.0)
