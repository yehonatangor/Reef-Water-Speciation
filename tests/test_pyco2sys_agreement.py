"""Head-to-head agreement with PyCO2SYS, the reference implementation."""

from __future__ import annotations

import numpy as np
import pytest

from cleaned_reef_water.equilibria import (
    CarbonicFormulation,
    k_bisulfate_free,
    k_borate_total,
    k_carbonic_total,
    k_co2_solubility,
    k_fluoride_free,
    k_water_sws,
)

pyco2_p1atm = pytest.importorskip(
    "PyCO2SYS.equilibria.p1atm",
    reason="PyCO2SYS is needed for the cross-implementation check",
)

#: Grid spanning the sampler's range and then some, so the salinity and
#: temperature *dependence* is exercised rather than a single point.
SALINITIES = (20.0, 25.0, 30.0, 35.0, 40.0)
TEMPERATURES = (5.0, 15.0, 25.0, 35.0)

#: Agreement required, as a relative difference.
TOLERANCE = 1.0e-5


def _grid():
    """Every (salinity, temperature) pair, with temperature also in kelvin."""
    for salinity in SALINITIES:
        for t_c in TEMPERATURES:
            yield salinity, t_c, t_c + 273.15


class TestCarbonicAcid:
    """K1 and K2 on the total scale, Lueker et al. (2000)."""

    def test_k1_and_k2_match_across_the_grid(self):
        worst_k1 = worst_k2 = 0.0
        for salinity, t_c, t_k in _grid():
            k1, k2 = k_carbonic_total(
                salinity, t_c, CarbonicFormulation.LUEKER2000
            )
            ref1, ref2 = pyco2_p1atm.kH2CO3_TOT_LDK00(t_k, salinity)
            worst_k1 = max(worst_k1, abs(float(k1) / float(ref1) - 1.0))
            worst_k2 = max(worst_k2, abs(float(k2) / float(ref2) - 1.0))
        assert worst_k1 < TOLERANCE, f"K1 worst relative difference {worst_k1:.2e}"
        assert worst_k2 < TOLERANCE, f"K2 worst relative difference {worst_k2:.2e}"


class TestOtherConstants:
    """One test per constant, so a failure names the culprit directly."""

    def test_boric_acid(self):
        """Dickson (1990b), total scale in both libraries."""
        worst = 0.0
        for salinity, t_c, t_k in _grid():
            ours = float(k_borate_total(salinity, t_c))
            theirs = float(pyco2_p1atm.kBOH3_TOT_D90b(t_k, salinity))
            worst = max(worst, abs(ours / theirs - 1.0))
        assert worst < TOLERANCE, f"K_B worst relative difference {worst:.2e}"

    def test_water(self):
        """Millero (1995)."""
        worst = 0.0
        for salinity, t_c, t_k in _grid():
            ours = float(k_water_sws(salinity, t_c))
            theirs = float(pyco2_p1atm.kH2O_SWS_M95(t_k, salinity))
            worst = max(worst, abs(ours / theirs - 1.0))
        assert worst < TOLERANCE, f"K_W worst relative difference {worst:.2e}"

    def test_bisulfate(self):
        """Dickson (1990a), free scale by definition in both."""
        worst = 0.0
        for salinity, t_c, t_k in _grid():
            ours = float(k_bisulfate_free(salinity, t_c))
            theirs = float(pyco2_p1atm.kHSO4_FREE_D90a(t_k, salinity))
            worst = max(worst, abs(ours / theirs - 1.0))
        assert worst < TOLERANCE, f"K_S worst relative difference {worst:.2e}"

    def test_hydrogen_fluoride(self):
        """Dickson & Riley (1979), free scale."""
        worst = 0.0
        for salinity, t_c, t_k in _grid():
            ours = float(k_fluoride_free(salinity, t_c))
            theirs = float(pyco2_p1atm.kHF_FREE_DR79(t_k, salinity))
            worst = max(worst, abs(ours / theirs - 1.0))
        assert worst < TOLERANCE, f"K_F worst relative difference {worst:.2e}"

    def test_co2_solubility(self):
        """Weiss (1974).  No pH scale -- it is a solubility, not a dissociation."""
        worst = 0.0
        for salinity, t_c, t_k in _grid():
            ours = float(k_co2_solubility(salinity, t_c))
            theirs = float(pyco2_p1atm.kCO2_W74(t_k, salinity))
            worst = max(worst, abs(ours / theirs - 1.0))
        assert worst < TOLERANCE, f"K_0 worst relative difference {worst:.2e}"


class TestAgreementIsReported:
    """Print the actual agreement, so the README's claim has a live source."""

    def test_report_worst_case_agreement(self, capsys):
        """Not an assertion so much as a measurement with a floor under it."""
        comparisons = {
            "K1": lambda s, t, tk: (
                float(k_carbonic_total(s, t, CarbonicFormulation.LUEKER2000)[0]),
                float(pyco2_p1atm.kH2CO3_TOT_LDK00(tk, s)[0]),
            ),
            "K2": lambda s, t, tk: (
                float(k_carbonic_total(s, t, CarbonicFormulation.LUEKER2000)[1]),
                float(pyco2_p1atm.kH2CO3_TOT_LDK00(tk, s)[1]),
            ),
            "K_B": lambda s, t, tk: (
                float(k_borate_total(s, t)),
                float(pyco2_p1atm.kBOH3_TOT_D90b(tk, s)),
            ),
            "K_W": lambda s, t, tk: (
                float(k_water_sws(s, t)),
                float(pyco2_p1atm.kH2O_SWS_M95(tk, s)),
            ),
            "K_S": lambda s, t, tk: (
                float(k_bisulfate_free(s, t)),
                float(pyco2_p1atm.kHSO4_FREE_D90a(tk, s)),
            ),
            "K_F": lambda s, t, tk: (
                float(k_fluoride_free(s, t)),
                float(pyco2_p1atm.kHF_FREE_DR79(tk, s)),
            ),
            "K_0": lambda s, t, tk: (
                float(k_co2_solubility(s, t)),
                float(pyco2_p1atm.kCO2_W74(tk, s)),
            ),
        }
        worst_overall = 0.0
        lines = []
        for name, get in comparisons.items():
            worst = max(
                abs(get(s, t, tk)[0] / get(s, t, tk)[1] - 1.0)
                for s, t, tk in _grid()
            )
            worst_overall = max(worst_overall, worst)
            lines.append(f"  {name:5s} {worst:.2e}")

        with capsys.disabled():
            print(
                f"\nPyCO2SYS agreement over "
                f"{len(SALINITIES) * len(TEMPERATURES)} (S, T) points, "
                f"worst relative difference:"
            )
            print("\n".join(lines))
            print(f"  {'ALL':5s} {worst_overall:.2e}")

        assert worst_overall < TOLERANCE
        assert np.isfinite(worst_overall)


class TestSolubilityProducts:
    """Mucci (1983) calcite and aragonite, the one source that cannot be read."""

    @staticmethod
    def _reference():
        solubility = pytest.importorskip("PyCO2SYS.solubility")
        return solubility.k_calcite_M83, solubility.k_aragonite_M83

    def test_calcite_matches(self):
        from cleaned_reef_water.equilibria import k_calcite

        calcite, _ = self._reference()
        worst = 0.0
        for salinity, t_c, t_k in _grid():
            ours = float(k_calcite(salinity, t_c))
            # Pbar = 0: this library is 1 atm only, and at zero the pressure
            # term is identically one.
            theirs = float(calcite(t_k, salinity, 0.0, 83.14462618))
            worst = max(worst, abs(ours / theirs - 1.0))
        assert worst < TOLERANCE, f"Ksp calcite worst difference {worst:.2e}"

    def test_aragonite_matches(self):
        from cleaned_reef_water.equilibria import k_aragonite

        _, aragonite = self._reference()
        worst = 0.0
        for salinity, t_c, t_k in _grid():
            ours = float(k_aragonite(salinity, t_c))
            theirs = float(aragonite(t_k, salinity, 0.0, 83.14462618))
            worst = max(worst, abs(ours / theirs - 1.0))
        assert worst < TOLERANCE, f"Ksp aragonite worst difference {worst:.2e}"

    def test_aragonite_is_more_soluble_than_calcite(self):
        """Mucci's own qualitative result, independent of any coefficient."""
        from cleaned_reef_water.equilibria import k_aragonite, k_calcite

        for salinity, t_c, _ in _grid():
            ratio = float(k_aragonite(salinity, t_c)) / float(
                k_calcite(salinity, t_c)
            )
            assert 1.3 < ratio < 1.8, f"S={salinity}, t={t_c}: ratio {ratio:.3f}"
        reference = float(k_aragonite(35.0, 25.0)) / float(k_calcite(35.0, 25.0))
        assert reference == pytest.approx(1.517, abs=0.01)
