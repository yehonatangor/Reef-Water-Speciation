"""Forward titration model, instrument noise and the ML data pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from cleaned_reef_water.alkalinity import constants_at, speciate, total_alkalinity
from cleaned_reef_water.ml.dataset import (
    META_COLUMNS,
    generate_dataset,
    split_dataset,
)
from cleaned_reef_water.ml.noise import NoiseModel, apply_noise
from cleaned_reef_water.ml.normalization import Normalizer
from cleaned_reef_water.ml.sampling import sample_composition
from cleaned_reef_water.solvers import ph_from_dic_alkalinity
from cleaned_reef_water.titration import STRONG_ACID, Titrant, simulate_titration

S_REF, T_REF = 35.0, 25.0
ALK, DIC = 2300e-6, 2000e-6


@pytest.fixture(scope="module")
def hcl_curve():
    return simulate_titration(ALK, DIC, S_REF, T_REF, titrant=STRONG_ACID, n_points=120)


class TestTitrationForwardModel:
    def test_first_point_is_the_undisturbed_sample(self, hcl_curve):
        """The curve must start at m = 0, not at a small non-zero aliquot."""
        assert hcl_curve.titrant_mass[0] == 0.0
        independent = ph_from_dic_alkalinity(DIC, ALK, S_REF, T_REF)
        assert float(hcl_curve.ph_total[0]) == pytest.approx(independent, abs=1e-9)

    def test_ph_decreases_monotonically(self, hcl_curve):
        assert np.all(np.diff(hcl_curve.ph_total) < 0.0)

    def test_equivalence_point_near_ph_4_5(self, hcl_curve):
        offset = np.abs(hcl_curve.titrant_mass - hcl_curve.equivalence_mass)
        index = int(np.argmin(offset))
        assert 4.0 < float(hcl_curve.ph_total[index]) < 5.0

    def test_equivalence_mass_is_exact(self, hcl_curve):
        expected = hcl_curve.sample_mass * ALK / STRONG_ACID.concentration
        assert hcl_curve.equivalence_mass == pytest.approx(expected, rel=1e-15)

    def test_alkalinity_is_recoverable_from_the_curve(self, hcl_curve):
        """At the equivalence point the added protons must equal the alkalinity."""
        protons = hcl_curve.equivalence_mass * STRONG_ACID.concentration
        assert protons / hcl_curve.sample_mass == pytest.approx(ALK, rel=1e-12)

    def test_higher_alkalinity_needs_more_titrant(self):
        low = simulate_titration(2000e-6, DIC, S_REF, T_REF, n_points=20)
        high = simulate_titration(2600e-6, DIC, S_REF, T_REF, n_points=20)
        assert high.equivalence_mass > low.equivalence_mass

    def test_all_points_finite(self, hcl_curve):
        assert np.all(np.isfinite(hcl_curve.ph_total))

    def test_volume_conversion(self, hcl_curve):
        volumes = hcl_curve.as_volume_ml(titrant_density_kg_per_l=1.02)
        assert volumes[0] == 0.0
        assert np.all(np.diff(volumes) > 0)

    @pytest.mark.parametrize("bad", [{"n_points": 1}, {"sample_mass_kg": 0.0}])
    def test_invalid_arguments_rejected(self, bad):
        with pytest.raises(ValueError):
            simulate_titration(ALK, DIC, S_REF, T_REF, **bad)


#: A generic triprotic weak acid used only to exercise the polyprotic
#: mathematics that underpins ``docs/titrant_selection.md``.
_WEAK_TRIPROTIC = Titrant(name="weak triprotic", concentration=0.05,
                          pk_values=(3.13, 4.76, 6.40))


class TestStrongAcidTitrant:
    """The recommended titrant introduces no unknown parameters."""

    def test_always_releases_exactly_one_proton(self):
        h = 10.0 ** -np.linspace(1.0, 11.0, 25)
        assert np.allclose(STRONG_ACID.protons_released(h), 1.0)

    def test_titratable_capacity_is_the_concentration(self):
        assert STRONG_ACID.titratable_capacity() == STRONG_ACID.concentration

    def test_buffer_capacity_is_identically_zero(self):
        """The reason a strong acid is used: it cannot bias the result."""
        h = 10.0 ** -np.linspace(2.0, 10.0, 25)
        n = STRONG_ACID.protons_released(h)
        assert np.allclose(n * (1.0 - n), 0.0, atol=1e-15)


class TestPolyproticMathematics:
    """Exercise the general n-bar machinery documented in titrant_selection.md."""

    def test_releases_all_protons_at_high_ph(self):
        assert float(
            _WEAK_TRIPROTIC.protons_released(np.asarray(1e-9))
        ) == pytest.approx(3.0, abs=0.05)

    def test_releases_no_protons_at_low_ph(self):
        assert float(_WEAK_TRIPROTIC.protons_released(np.asarray(1e-1))) < 0.1

    def test_protons_released_is_monotone_in_ph(self):
        h = 10.0 ** -np.linspace(1.0, 11.0, 50)
        assert np.all(np.diff(_WEAK_TRIPROTIC.protons_released(h)) > 0)

    def test_titratable_capacity_replaces_the_magic_1_8(self):
        """Legacy code divided by an unexplained 1.8; capacity is exact."""
        assert _WEAK_TRIPROTIC.titratable_capacity() == pytest.approx(
            3.0 * _WEAK_TRIPROTIC.concentration
        )

    def test_monoprotic_derivative_matches_closed_form(self):
        """d(n)/d(pK) = -ln(10) * n * (1 - n) for a monoprotic acid."""
        acid = Titrant(name="monoprotic", concentration=0.1, pk_values=(4.0,))
        for ph in (2.0, 3.0, 4.0, 5.0, 6.0):
            h = np.asarray(10.0**-ph)
            n = float(acid.protons_released(h))
            eps = 1e-6
            hi = Titrant("hi", 0.1, (4.0 + eps,)).protons_released(h)
            lo = Titrant("lo", 0.1, (4.0 - eps,)).protons_released(h)
            numeric = float((hi - lo) / (2 * eps))
            closed_form = -np.log(10.0) * n * (1.0 - n)
            assert numeric == pytest.approx(closed_form, rel=1e-5)

    def test_weak_titrant_reaches_a_higher_end_ph_than_a_strong_one(self):
        """A weak titrant cannot drive the sample as far -- the core problem."""
        weak = simulate_titration(
            ALK, DIC, S_REF, T_REF, titrant=_WEAK_TRIPROTIC, n_points=60
        )
        strong = simulate_titration(
            ALK, DIC, S_REF, T_REF, titrant=STRONG_ACID, n_points=60
        )
        assert float(weak.ph_total[-1]) > float(strong.ph_total[-1]) + 1.0
        assert np.all(np.diff(weak.ph_total) < 0.0)


class TestNoiseModel:
    @pytest.fixture(scope="class")
    def clean(self):
        return simulate_titration(ALK, DIC, S_REF, T_REF, n_points=150)

    def test_noise_magnitude_is_realistic(self, clean):
        residual_sd = []
        for seed in range(60):
            noisy = apply_noise(
                clean.titrant_mass,
                clean.ph_total,
                S_REF,
                T_REF,
                rng=np.random.default_rng(seed),
            )
            residual_sd.append(float(np.std(noisy.ph_measured - clean.ph_total)))
        mean_sd = float(np.mean(residual_sd))
        assert 0.002 < mean_sd < 0.02, f"pH noise sd {mean_sd} is not realistic"

    def test_nernst_slope_is_59_16_mv_at_25c(self):
        slope = NoiseModel().nernst_slope_mv_per_ph(25.0)
        assert slope == pytest.approx(59.16, abs=0.01)

    def test_slope_error_pivots_about_the_calibration_ph(self, clean):
        """At the calibration pH a pure slope error must have no effect."""
        model = NoiseModel(
            calibration_ph=float(clean.ph_total[0]),
            slope_error_sd=0.05,
            offset_sd=0.0,
            electrode_noise_sd=0.0,
            junction_drift_sd=0.0,
            co2_loss_fraction_range=(0.0, 0.0),
            burette_scale_error_sd=0.0,
            burette_jitter_sd=0.0,
        )
        noisy = apply_noise(
            clean.titrant_mass,
            clean.ph_total,
            S_REF,
            T_REF,
            model=model,
            rng=np.random.default_rng(3),
        )
        assert float(noisy.ph_measured[0]) == pytest.approx(
            float(clean.ph_total[0]), abs=1e-12
        )
        assert abs(float(noisy.ph_measured[-1] - clean.ph_total[-1])) > 1e-4

    def test_co2_degassing_raises_ph(self, clean):
        model = NoiseModel(
            slope_error_sd=0.0,
            offset_sd=0.0,
            electrode_noise_sd=0.0,
            junction_drift_sd=0.0,
            co2_loss_fraction_range=(0.05, 0.05),
        )
        noisy = apply_noise(
            clean.titrant_mass, clean.ph_total, S_REF, T_REF,
            model=model, rng=np.random.default_rng(0),
        )
        delta = noisy.ph_measured - clean.ph_total
        assert delta[0] == pytest.approx(0.0, abs=1e-12)
        assert np.all(delta >= -1e-12), "degassing can only raise pH"
        assert float(delta[-1]) == pytest.approx(0.0223, abs=0.005)

    def test_burette_error_is_sub_percent(self, clean):
        noisy = apply_noise(
            clean.titrant_mass, clean.ph_total, S_REF, T_REF,
            rng=np.random.default_rng(11),
        )
        relative = abs(float(noisy.titrant_mass[-1] / clean.titrant_mass[-1] - 1.0))
        assert relative < 0.01, "legacy +/-1.5 % burette error was unrealistic"

    def test_environment_carries_measurement_error(self, clean):
        noisy = apply_noise(
            clean.titrant_mass, clean.ph_total, S_REF, T_REF,
            rng=np.random.default_rng(5),
        )
        assert noisy.salinity_measured != S_REF
        assert noisy.t_c_measured != T_REF
        assert abs(noisy.salinity_measured - S_REF) < 0.2
        assert abs(noisy.t_c_measured - T_REF) < 0.5

    def test_reproducible_given_a_seed(self, clean):
        args = (clean.titrant_mass, clean.ph_total, S_REF, T_REF)
        a = apply_noise(*args, rng=np.random.default_rng(7))
        b = apply_noise(*args, rng=np.random.default_rng(7))
        assert np.array_equal(a.ph_measured, b.ph_measured)

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="equal length"):
            apply_noise(np.zeros(5), np.zeros(4), S_REF, T_REF)


class TestSamplingAndDataset:
    def test_sampled_composition_is_self_consistent(self):
        rng = np.random.default_rng(0)
        for _ in range(25):
            comp = sample_composition(rng)
            constants = constants_at(comp.salinity, comp.t_c)
            species = speciate(
                comp.ph_total,
                constants,
                dic=comp.dic,
                total_boron=comp.total_boron,
                total_phosphate=comp.total_phosphate,
                total_silicate=comp.total_silicate,
                # Organic matter titrates, so it is part of the alkalinity the
                # sampler derives.
                total_organic=comp.total_organic,
                pk_organic=comp.pk_organic,
            )
            assert float(total_alkalinity(species)) == pytest.approx(
                comp.alkalinity, rel=1e-12
            )

    def test_boron_is_decorrelated_from_salinity(self):
        rng = np.random.default_rng(1)
        comps = [sample_composition(rng) for _ in range(300)]
        salinity = np.array([c.salinity for c in comps])
        boron = np.array([c.total_boron for c in comps])
        r = float(np.corrcoef(salinity, boron)[0, 1])
        assert abs(r) < 0.95, "boron must not be a deterministic function of salinity"

    def test_dataset_shapes_and_finiteness(self):
        data = generate_dataset(12, seed=2, n_points=60)
        assert data.curves.shape == (12, 60, 2)
        assert data.env.shape == (12, 2)
        assert data.labels.shape == (12, 3)
        assert np.all(np.isfinite(data.curves))
        assert np.all(data.labels > 0.0)

    def test_dataset_is_reproducible(self):
        a = generate_dataset(6, seed=3, n_points=40)
        b = generate_dataset(6, seed=3, n_points=40)
        assert np.array_equal(a.labels, b.labels)
        assert np.array_equal(a.curves, b.curves)

    def test_labels_use_true_initial_ph_not_first_curve_point(self):
        """Labels must come from the undisturbed sample, analytically."""
        data = generate_dataset(8, seed=4, n_points=40, add_noise=False)
        for i in range(8):
            dic = data.metadata[i, META_COLUMNS.index("dic")]
            salinity = data.metadata[i, META_COLUMNS.index("salinity_true")]
            t_c = data.metadata[i, META_COLUMNS.index("t_c_true")]
            constants = constants_at(salinity, t_c)
            ph = data.curves[i, 0, 0]
            species = speciate(ph, constants, dic=dic)
            assert float(species.hco3) == pytest.approx(data.labels[i, 0], rel=1e-6)

    def test_metadata_records_sample_mass_and_nutrients(self):
        """Regression: the inverse needs m_0, and it must come from the data."""
        data = generate_dataset(8, seed=11, n_points=30, add_noise=False)
        assert data.metadata.shape[1] == len(META_COLUMNS)
        mass = data.metadata[:, META_COLUMNS.index("sample_mass_kg")]
        assert np.all((mass >= 0.0145) & (mass <= 0.0155))
        assert mass.std() > 0.0, "sample mass must actually vary, not be constant"
        boron = data.metadata[:, META_COLUMNS.index("total_boron")]
        assert np.all(boron > 0.0)

    def test_recorded_sample_mass_makes_the_inverse_accurate(self):
        """With the true m_0 the noise-free inverse must be sub-umol/kg."""
        from cleaned_reef_water.baseline import LeastSquaresInverse

        data = generate_dataset(4, seed=12, n_points=50, add_noise=False)
        for i in range(4):
            meta = data.metadata[i]
            result = LeastSquaresInverse(
                sample_mass_kg=float(meta[META_COLUMNS.index("sample_mass_kg")])
            ).fit(
                data.curves[i, :, 1],
                data.curves[i, :, 0],
                float(meta[META_COLUMNS.index("salinity_true")]),
                float(meta[META_COLUMNS.index("t_c_true")]),
            )
            truth = float(meta[META_COLUMNS.index("alkalinity")])
            assert abs(result.alkalinity - truth) * 1e6 < 2.0

    def test_defaulting_the_sample_mass_is_much_worse(self):
        """Pins the magnitude of the bug this metadata column exists to fix."""
        from cleaned_reef_water.baseline import LeastSquaresInverse

        data = generate_dataset(6, seed=13, n_points=50, add_noise=False)
        with_true, with_default = [], []
        for i in range(6):
            meta = data.metadata[i]
            truth = float(meta[META_COLUMNS.index("alkalinity")])
            args = (
                data.curves[i, :, 1],
                data.curves[i, :, 0],
                float(meta[META_COLUMNS.index("salinity_true")]),
                float(meta[META_COLUMNS.index("t_c_true")]),
            )
            true_mass = float(meta[META_COLUMNS.index("sample_mass_kg")])
            with_true.append(
                abs(
                    LeastSquaresInverse(sample_mass_kg=true_mass).fit(*args).alkalinity
                    - truth
                )
                * 1e6
            )
            with_default.append(
                abs(
                    LeastSquaresInverse(sample_mass_kg=0.015).fit(*args).alkalinity
                    - truth
                )
                * 1e6
            )
        assert np.mean(with_true) < 2.0
        assert np.mean(with_default) > 10.0 * np.mean(with_true)

    def test_split_is_disjoint_and_complete(self):
        data = generate_dataset(20, seed=5, n_points=30)
        train, val, test = split_dataset(data)
        assert len(train["curves"]) + len(val["curves"]) + len(test["curves"]) == 20
        all_alk = np.concatenate(
            [p["metadata"][:, 1] for p in (train, val, test)]
        )
        assert np.allclose(np.sort(all_alk), np.sort(data.metadata[:, 1]))

    def test_split_rejects_impossible_fractions(self):
        data = generate_dataset(4, seed=6, n_points=20)
        with pytest.raises(ValueError, match="leave a test set"):
            split_dataset(data, train_fraction=0.9, val_fraction=0.2)


class TestNormalizer:
    @pytest.fixture(scope="class")
    def fitted(self):
        rng = np.random.default_rng(0)
        curves = rng.normal(3.0, 2.0, (50, 20, 2))
        env = rng.normal(35.0, 1.0, (50, 2))
        labels = rng.uniform(1e-4, 2e-3, (50, 3))
        return curves, env, labels, Normalizer.fit(curves, env, labels)

    def test_standardises_to_zero_mean_unit_variance(self, fitted):
        curves, env, labels, norm = fitted
        scaled = norm.transform_curves(curves)
        assert float(scaled.mean()) == pytest.approx(0.0, abs=1e-12)
        assert float(scaled.std()) == pytest.approx(1.0, abs=1e-12)

    def test_label_round_trip(self, fitted):
        _, _, labels, norm = fitted
        recovered = norm.inverse_labels(norm.transform_labels(labels))
        assert np.allclose(recovered, labels, rtol=1e-14)

    def test_labels_are_not_centred_by_default(self):
        """Softplus is strictly positive, so scaled labels must be too."""
        rng = np.random.default_rng(0)
        labels = rng.uniform(1e-4, 2e-3, (200, 3))
        norm = Normalizer.fit(np.zeros((200, 5, 2)), np.zeros((200, 2)), labels)
        scaled = norm.transform_labels(labels)
        assert np.all(scaled > 0.0), "softplus cannot represent these targets"
        assert np.all(norm.label_mean == 0.0)

    def test_scaled_targets_start_near_one(self):
        """Targets must sit near 1.0, where the softplus head is initialised."""
        rng = np.random.default_rng(0)
        labels = np.abs(
            np.stack(
                [rng.normal(m, s, 2000) for m, s in
                 ((2721e-6, 1053e-6), (310e-6, 197e-6), (81.5e-6, 37.6e-6))],
                axis=1,
            )
        )
        blank_curves = np.zeros((2000, 5, 2))
        blank_env = np.zeros((2000, 2))

        by_mean = Normalizer.fit(blank_curves, blank_env, labels)
        scaled = by_mean.transform_labels(labels)
        assert np.allclose(scaled.mean(axis=0), 1.0, atol=1e-9)
        assert np.all(scaled > 0.0)

        by_std = Normalizer.fit(
            blank_curves, blank_env, labels, label_scale="std"
        )
        assert np.all(by_std.transform_labels(labels).mean(axis=0) > 1.5)

    def test_unknown_label_scale_rejected(self):
        with pytest.raises(ValueError, match="label_scale"):
            Normalizer.fit(
                np.zeros((10, 5, 2)), np.zeros((10, 2)), np.ones((10, 3)),
                label_scale="zscore",
            )

    def test_centring_is_available_but_opt_in(self):
        rng = np.random.default_rng(0)
        labels = rng.uniform(1e-4, 2e-3, (200, 3))
        norm = Normalizer.fit(
            np.zeros((200, 5, 2)), np.zeros((200, 2)), labels, center_labels=True
        )
        scaled = norm.transform_labels(labels)
        assert np.any(scaled < 0.0)
        assert np.allclose(norm.inverse_labels(scaled), labels, rtol=1e-14)

    def test_constant_feature_does_not_produce_nan(self):
        curves = np.ones((10, 5, 2))
        env = np.ones((10, 2))
        labels = np.ones((10, 3))
        norm = Normalizer.fit(curves, env, labels)
        assert np.all(np.isfinite(norm.transform_curves(curves)))

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="share the first dimension"):
            Normalizer.fit(np.zeros((10, 5, 2)), np.zeros((9, 2)), np.zeros((10, 3)))

    def test_requires_3d_curves(self):
        with pytest.raises(ValueError, match="must be 3-D"):
            Normalizer.fit(np.zeros((10, 5)), np.zeros((10, 2)), np.zeros((10, 3)))

    def test_save_and_load_round_trip(self, tmp_path, fitted):
        _, _, _, norm = fitted
        path = tmp_path / "norm.npz"
        norm.save(path)
        loaded = Normalizer.load(path)
        assert np.allclose(loaded.label_mean, norm.label_mean)
        assert np.allclose(loaded.curve_std, norm.curve_std)
