"""Organic alkalinity: backwards compatibility, physics, and the bias it creates."""

from __future__ import annotations

import numpy as np
import pytest

from cleaned_reef_water import simulate_titration
from cleaned_reef_water.alkalinity import (
    DEFAULT_ORGANIC_PK,
    constants_at,
    speciate,
    total_alkalinity,
)
from cleaned_reef_water.baseline import LeastSquaresInverse, NuisanceModel
from cleaned_reef_water.ml.dataset import META_COLUMNS, generate_dataset
from cleaned_reef_water.ml.sampling import SamplingRanges, sample_composition

S_REF, T_REF = 35.0, 25.0
ALK, DIC = 2300e-6, 2000e-6


class TestBackwardsCompatibility:
    """Zero organic must reproduce the previous behaviour exactly."""

    def test_speciation_is_unchanged(self):
        constants = constants_at(S_REF, T_REF)
        without = speciate(8.1, constants, dic=DIC)
        with_zero = speciate(8.1, constants, dic=DIC, total_organic=0.0)
        for name in ("hco3", "co3", "boh4", "oh", "h_free", "hso4", "hf"):
            assert float(getattr(without, name)) == float(getattr(with_zero, name))
        assert float(with_zero.organic_anion) == 0.0

    def test_alkalinity_is_unchanged(self):
        constants = constants_at(S_REF, T_REF)
        without = total_alkalinity(speciate(8.1, constants, dic=DIC))
        with_zero = total_alkalinity(
            speciate(8.1, constants, dic=DIC, total_organic=0.0)
        )
        assert float(without) == float(with_zero)

    def test_titration_curve_is_bit_identical(self):
        without = simulate_titration(ALK, DIC, S_REF, T_REF, n_points=40)
        with_zero = simulate_titration(
            ALK, DIC, S_REF, T_REF, n_points=40, total_organic=0.0
        )
        assert np.array_equal(without.ph_total, with_zero.ph_total)


class TestPhysics:
    def test_organic_anion_follows_the_dissociation_curve(self):
        """A weak acid is fully dissociated well above its pKa, and not below."""
        constants = constants_at(S_REF, T_REF)
        total = 1e-4
        high = speciate(8.1, constants, dic=DIC, total_organic=total, pk_organic=4.5)
        low = speciate(3.0, constants, dic=DIC, total_organic=total, pk_organic=4.5)
        assert float(high.organic_anion) == pytest.approx(total, rel=1e-3)
        assert float(low.organic_anion) < 0.04 * total

    def test_at_the_pka_exactly_half_is_dissociated(self):
        constants = constants_at(S_REF, T_REF)
        total = 1e-4
        at_pka = speciate(
            DEFAULT_ORGANIC_PK, constants, dic=DIC, total_organic=total,
            pk_organic=DEFAULT_ORGANIC_PK,
        )
        assert float(at_pka.organic_anion) == pytest.approx(0.5 * total, rel=1e-9)

    def test_organic_adds_to_alkalinity(self):
        """The anion is a proton acceptor, so it raises measured alkalinity."""
        constants = constants_at(S_REF, T_REF)
        base = speciate(8.1, constants, dic=DIC)
        with_org = speciate(8.1, constants, dic=DIC, total_organic=1e-4, pk_organic=4.5)
        added = float(total_alkalinity(with_org)) - float(total_alkalinity(base))
        assert added == pytest.approx(float(with_org.organic_anion), rel=1e-12)
        assert added > 0.0

    def test_organic_anion_never_exceeds_the_total(self):
        constants = constants_at(S_REF, T_REF)
        total = 1e-4
        for ph in (2.0, 4.0, 4.76, 6.0, 8.0, 10.0):
            found = speciate(
                ph, constants, dic=DIC, total_organic=total, pk_organic=4.76
            )
            assert 0.0 <= float(found.organic_anion) <= total


class TestSampler:
    def test_composition_is_self_consistent(self):
        """The forward model must reproduce the drawn pH at zero titrant."""
        rng = np.random.default_rng(3)
        for _ in range(5):
            drawn = sample_composition(rng)
            curve = simulate_titration(
                drawn.alkalinity, drawn.dic, drawn.salinity, drawn.t_c,
                sample_mass_kg=drawn.sample_mass_kg, n_points=20,
                total_boron=drawn.total_boron,
                total_phosphate=drawn.total_phosphate,
                total_silicate=drawn.total_silicate,
                total_organic=drawn.total_organic,
                pk_organic=drawn.pk_organic,
            )
            assert curve.ph_total[0] == pytest.approx(drawn.ph_total, abs=1e-9)

    def test_ranges_are_respected(self):
        rng = np.random.default_rng(11)
        ranges = SamplingRanges()
        drawn = [sample_composition(rng) for _ in range(60)]
        organic = np.array([d.total_organic for d in drawn])
        pk = np.array([d.pk_organic for d in drawn])
        assert np.all(organic >= 0.0)
        assert np.all(organic <= ranges.organic[1])
        assert np.all((pk >= ranges.pk_organic[0]) & (pk <= ranges.pk_organic[1]))
        assert 0.0 < (organic > 0).mean() < 1.0, "both cases must occur"

    def test_zero_probability_reproduces_organic_free_water(self):
        import dataclasses

        ranges = dataclasses.replace(SamplingRanges(), organic_probability=0.0)
        rng = np.random.default_rng(5)
        drawn = [sample_composition(rng, ranges) for _ in range(20)]
        assert all(d.total_organic == 0.0 for d in drawn)

    def test_dataset_records_organic(self):
        data = generate_dataset(12, seed=5, n_points=30, add_noise=False)
        assert data.metadata.shape[1] == len(META_COLUMNS)
        organic = data.metadata[:, META_COLUMNS.index("total_organic")]
        pk = data.metadata[:, META_COLUMNS.index("pk_organic")]
        assert np.all(organic >= 0.0)
        assert np.all((pk >= 4.0) & (pk <= 7.0))


class TestClassicalInverseIsBiased:
    """The result that motivates the whole feature."""

    def test_classical_inverse_is_biased_by_organic_alkalinity(self):
        """The bias is systematic, large, and proportional to organic content."""
        data = generate_dataset(45, seed=9, n_points=60, add_noise=False)
        mass_col = META_COLUMNS.index("sample_mass_kg")
        organic_col = META_COLUMNS.index("total_organic")

        errors, organic = [], []
        for i in range(len(data.curves)):
            meta = data.metadata[i]
            result = LeastSquaresInverse(
                sample_mass_kg=float(meta[mass_col]),
                nuisance=NuisanceModel.OFFSET_AND_SLOPE,
            ).fit(
                data.curves[i, :, 1], data.curves[i, :, 0],
                float(data.env[i, 0]), float(data.env[i, 1]),
            )
            if not np.isfinite(result.hco3):
                continue
            errors.append((result.hco3 - data.labels[i, 0]) * 1e6)
            organic.append(meta[organic_col] * 1e6)

        errors = np.asarray(errors)
        organic = np.asarray(organic)
        clean = organic < 1e-9

        assert clean.sum() >= 10 and (~clean).sum() >= 10, "need both groups"
        # On organic-free water the inverse is essentially exact, which is the
        # control: these curves carry no instrument noise at all, so anything
        # left is model misspecification and there is none.
        assert np.abs(errors[clean]).mean() < 15.0
        # With organic present it acquires a large POSITIVE bias -- it must
        # attribute the organic proton acceptors to the carbonate system.
        assert errors[~clean].mean() > 20.0
        # And that is a bias, not merely extra scatter.
        assert errors[~clean].mean() > 4.0 * abs(errors[clean].mean())
        # The load-bearing assertion: the error tracks the organic content,
        # which is what makes this a term the inverse is missing rather than
        # organic water simply being harder to fit.
        correlation = float(np.corrcoef(organic, errors)[0, 1])
        assert correlation > 0.5, f"error must track organic content, got {correlation}"
        slope = float(np.polyfit(organic, errors, 1)[0])
        assert 0.2 < slope < 1.0, f"expected ~0.57 umol/umol, got {slope}"


class TestIdentifiability:
    r"""Organic concentration is nearly degenerate with the carbonate totals."""

    @staticmethod
    def _explained_fraction(
        pk_organic: float,
        n_points: int = 40,
        salinity: float = S_REF,
        t_c: float = T_REF,
    ) -> float:
        """Fraction of the organic sensitivity lying in the span of (A_T, C_T)."""
        alkalinity, dic, organic = 2.30e-3, 2.00e-3, 7.5e-5

        def curve(a=alkalinity, c=dic, o=organic):
            return simulate_titration(
                a, c, salinity, t_c, n_points=n_points,
                total_organic=o, pk_organic=pk_organic,
            ).ph_total

        step = 1e-6
        basis = np.stack([
            (curve(a=alkalinity + step) - curve(a=alkalinity - step)) / 2.0,
            (curve(c=dic + step) - curve(c=dic - step)) / 2.0,
        ])
        target = (curve(o=organic + step) - curve(o=organic - step)) / 2.0
        coefficients, *_ = np.linalg.lstsq(basis.T, target, rcond=None)
        residual = target - basis.T @ coefficients
        return 1.0 - float(residual @ residual) / float(target @ target)

    def test_organic_pka_range_straddles_carbonic_pk1(self):
        """The physical cause.  If this stops being true, the rest changes."""
        constants = constants_at(S_REF, T_REF)
        pk1 = -np.log10(float(constants.k1))
        assert pk1 == pytest.approx(5.847, abs=0.01)
        low, high = SamplingRanges().pk_organic
        assert low < pk1 < high, (
            f"carbonic pK1 {pk1:.3f} must lie inside the organic pKa draw "
            f"range {(low, high)} for the degeneracy claim to hold"
        )

    def test_organic_is_nearly_degenerate_with_the_carbonate_totals(self):
        """At the centre of the draw range, ~99% of the signature is redundant."""
        explained = self._explained_fraction(5.5)
        assert explained > 0.98, (
            f"expected near-total degeneracy, got {explained:.4f} explained"
        )

    def test_degeneracy_is_worst_exactly_at_carbonic_pk1(self):
        """Not a coincidence of parameterisation: it tracks the overlap."""
        at_pk1 = self._explained_fraction(5.85)
        at_edge = self._explained_fraction(4.0)
        assert at_pk1 > 0.999, f"expected total degeneracy at pK1, got {at_pk1:.4f}"
        assert at_edge < 0.80, f"expected recoverable at pKa 4.0, got {at_edge:.4f}"
        assert at_pk1 > at_edge

    def test_degeneracy_holds_across_the_composition_range(self):
        """Not an artefact of one reference state."""
        for salinity, t_c in ((20.0, 10.0), (35.0, 32.0)):
            explained = self._explained_fraction(
                5.5, salinity=salinity, t_c=t_c
            )
            assert explained > 0.97, (
                f"S={salinity} T={t_c}: expected near-degeneracy, got "
                f"{explained:.4f}"
            )
