"""The physics-constrained autoencoder: structure, gradients, conventions."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from cleaned_reef_water.ml.dataset import (
    META_COLUMNS,
    NUISANCE_COLUMNS,
    generate_dataset,
)
from cleaned_reef_water.ml.differentiable import pack_constants, titration_ph
from cleaned_reef_water.ml.noise import NoiseModel
from cleaned_reef_water.ml.sampling import SamplingRanges

tf = pytest.importorskip("tensorflow", reason="the autoencoder needs TensorFlow")

from cleaned_reef_water.ml.autoencoder import (  # noqa: E402
    LATENT_BOUNDS,
    LATENT_INDEX,
    LATENT_NAMES,
    N_COMPOSITION_LATENTS,
    build_physics_autoencoder,
    decode_latents,
)

N_POINTS = 24


def _inputs(dataset):
    """Assemble model inputs from a generated dataset."""
    meta = dataset.metadata
    return {
        "curve": dataset.curves.astype("float32"),
        "env": dataset.env.astype("float32"),
        "constants": pack_constants(meta[:, 2], meta[:, 3]),
        "titrant_mass": dataset.curves[:, :, 1],
        "sample_mass": meta[:, 4:5],
        "totals": meta[:, 5:8],
    }


@pytest.fixture(scope="module")
def dataset():
    """Generate a small noisy dataset."""
    return generate_dataset(6, seed=21, n_points=N_POINTS)


class TestStructure:
    def test_builds_and_shapes_are_right(self, dataset):
        model = build_physics_autoencoder(N_POINTS)
        out = model(_inputs(dataset), training=False)
        assert out["reconstruction"].shape == (6, N_POINTS)
        assert out["latents"].shape == (6, len(LATENT_NAMES))

    def test_untrained_latents_start_near_no_instrument_error(self, dataset):
        """Slope and pump scale must start close to 1.0."""
        model = build_physics_autoencoder(N_POINTS)
        latents = decode_latents(model(_inputs(dataset), training=False)["latents"])
        assert latents["slope"] == pytest.approx(1.0, abs=0.02)
        assert latents["pump_scale"] == pytest.approx(1.0, abs=0.01)
        assert latents["offset"] == pytest.approx(0.0, abs=0.05)

    def test_latents_always_respect_physical_bounds(self, dataset):
        """The squashing nonlinearity must make excursions impossible."""
        model = build_physics_autoencoder(N_POINTS)
        # The head is two-stage, so both halves must be driven to extremes.
        for name in ("totals_logits", "nuisance_logits"):
            layer = model.get_layer(name)
            layer.set_weights(
                [np.full_like(w, 50.0) for w in layer.get_weights()]
            )
        latents = decode_latents(model(_inputs(dataset), training=False)["latents"])
        for name in LATENT_NAMES:
            low, high = LATENT_BOUNDS[name]
            assert np.all(latents[name] >= low - 1e-12)
            assert np.all(latents[name] <= high + 1e-12)

    def test_decoder_has_no_trainable_weights(self):
        """All capacity must live in the encoder; the chemistry is fixed."""
        model = build_physics_autoencoder(N_POINTS)
        for name in ("reconstruction", "species"):
            assert model.get_layer(name).trainable_weights == []

    def test_nuisance_head_is_conditioned_on_the_totals(self):
        """The two-stage head must actually be two-stage."""
        model = build_physics_autoencoder(N_POINTS)
        nuisance = model.get_layer("nuisance_logits")
        totals = model.get_layer("totals_logits")
        # Version-independent check: the nuisance head's input must be wider
        # than the shared features by exactly the composition block.
        assert totals.output.shape[-1] == N_COMPOSITION_LATENTS
        assert nuisance.output.shape[-1] == len(LATENT_NAMES) - N_COMPOSITION_LATENTS
        assert (
            nuisance.input.shape[-1]
            == totals.input.shape[-1] + N_COMPOSITION_LATENTS
        ), "nuisance head is not receiving the predicted composition"


class TestSpeciesHead:
    """The in-graph speciation must equal the validated NumPy chemistry."""

    def test_matches_numpy_speciation(self):
        from cleaned_reef_water.alkalinity import constants_at, speciate
        from cleaned_reef_water.ml.differentiable import (
            carbonate_species,
            solve_ph,
        )
        from cleaned_reef_water.seawater import total_borate

        salinity, t_c = np.array([35.0, 31.0]), np.array([25.0, 22.0])
        alkalinity = np.array([[2.30e-3], [3.10e-3]])
        dic = np.array([[2.00e-3], [2.75e-3]])
        boron = np.array([[float(total_borate(s))] for s in salinity])
        constants = pack_constants(salinity, t_c)

        ph = solve_ph(
            tf.constant(alkalinity), tf.constant(constants),
            tf.constant(dic), tf.constant(boron),
        )
        hco3, co3, boh4 = carbonate_species(
            ph, tf.constant(constants), tf.constant(dic), tf.constant(boron)
        )

        for i in range(2):
            expected = speciate(
                float(ph.numpy()[i, 0]),
                constants_at(salinity[i], t_c[i]),
                dic=float(dic[i, 0]),
                total_boron=float(boron[i, 0]),
            )
            assert float(hco3.numpy()[i, 0]) == pytest.approx(
                float(expected.hco3), rel=1e-10
            )
            assert float(co3.numpy()[i, 0]) == pytest.approx(
                float(expected.co3), rel=1e-10
            )
            assert float(boh4.numpy()[i, 0]) == pytest.approx(
                float(expected.boh4), rel=1e-10
            )


#: One latent vector, written by name so that inserting a latent breaks this
#: file loudly at the point of construction instead of silently shifting what
#: every positional literal means.
_EXAMPLE_LATENTS = {
    "alkalinity": 2.3e-3,
    "alk_minus_dic": 3.0e-4,
    "total_organic": 7.5e-5,
    "pk_organic": 4.5,
    "offset": 0.01,
    "slope": 1.0,
    "pump_scale": 1.002,
}


def _latent_row(**overrides: float) -> np.ndarray:
    """Build a ``(1, len(LATENT_NAMES))`` latent row in the canonical order."""
    values = {**_EXAMPLE_LATENTS, **overrides}
    return np.array([[values[name] for name in LATENT_NAMES]])


class TestDecodeLatents:
    def test_round_trips_names(self):
        out = decode_latents(_latent_row())
        assert sorted(out) == sorted(LATENT_NAMES)
        assert out["pump_scale"][0] == pytest.approx(1.002)
        assert out["total_organic"][0] == pytest.approx(7.5e-5)

    def test_second_latent_is_the_difference_not_dic(self):
        """A_T minus C_T is the latent; DIC is derived."""
        from cleaned_reef_water.ml.autoencoder import totals_from_latents

        alkalinity, dic = totals_from_latents(_latent_row())
        assert float(alkalinity[0, 0]) == pytest.approx(2.3e-3)
        assert float(dic[0, 0]) == pytest.approx(2.0e-3)

    def test_rejects_wrong_width(self):
        with pytest.raises(ValueError, match="shape"):
            decode_latents(np.zeros((2, 3)))


class TestGradientFlow:
    def test_every_encoder_layer_receives_gradient(self, dataset):
        """Guards the latent-head initialiser."""
        model = build_physics_autoencoder(N_POINTS)
        inputs = _inputs(dataset)
        target = tf.constant(dataset.curves[:, :, 0])
        with tf.GradientTape() as tape:
            out = model(inputs, training=True)
            loss = tf.reduce_mean(tf.square(out["reconstruction"] - target))
        gradients = tape.gradient(loss, model.trainable_variables)
        assert all(g is not None for g in gradients), "a variable is disconnected"
        live = [bool(np.any(np.abs(g.numpy()) > 0.0)) for g in gradients]
        assert all(live), f"{live.count(False)} weight tensors got zero gradient"


class TestPumpScaleConvention:
    """The recorded ``pump_scale`` must be what the decoder multiplies by."""

    def test_recorded_scale_reconstructs_the_measured_curve(self):
        isolated = dataclasses.replace(
            NoiseModel.hobbyist(),
            slope_error_sd=0.0,
            offset_sd=0.0,
            electrode_noise_sd=0.0,
            junction_drift_sd=0.0,
            burette_jitter_sd=0.0,
            temperature_error_sd=0.0,
            salinity_error_sd=0.0,
            quantisation_step_ph=0.0,
            nbs_calibration_bias_mean=0.0,
            nbs_calibration_bias_sd=0.0,
            co2_loss_fraction_range=(0.0, 0.0),
        )
        # Organic-free: the differentiable decoder has no organic term, so
        # organic alkalinity would confound a test about the mass axis.
        data = generate_dataset(
            6, seed=21, n_points=N_POINTS, noise_model=isolated,
            ranges=dataclasses.replace(SamplingRanges(), organic_probability=0.0),
        )
        meta = data.metadata
        scale = data.nuisance[:, NUISANCE_COLUMNS.index("pump_scale")][:, None]

        def render(mass_factor):
            return titration_ph(
                tf.constant(meta[:, 1:2]),
                tf.constant(meta[:, 0:1]),
                tf.constant(pack_constants(meta[:, 2], meta[:, 3])),
                tf.constant(data.curves[:, :, 1]) * mass_factor,
                tf.constant(meta[:, 4:5]),
                tf.constant(meta[:, 5:6]),
                tf.constant(meta[:, 6:7]),
                tf.constant(meta[:, 7:8]),
            ).numpy()

        measured = data.curves[:, :, 0]
        with_scale = np.abs(render(tf.constant(scale)) - measured).max()
        without = np.abs(render(1.0) - measured).max()

        assert with_scale < 1e-8, (
            f"recorded pump_scale does not reconstruct the curve "
            f"({with_scale:.3e} pH); the convention is inverted"
        )
        assert without > 1e-3, "burette error was not actually applied"

    def test_full_affine_round_trip_uses_the_calibration_pivot(self):
        """All three recorded nuisance latents must reconstruct the curve."""
        calibration_ph = 8.0936
        isolated = dataclasses.replace(
            NoiseModel.hobbyist(),
            electrode_noise_sd=0.0,
            junction_drift_sd=0.0,
            burette_jitter_sd=0.0,
            temperature_error_sd=0.0,
            salinity_error_sd=0.0,
            quantisation_step_ph=0.0,
            co2_loss_fraction_range=(0.0, 0.0),
        )
        data = generate_dataset(
            6, seed=31, n_points=N_POINTS, noise_model=isolated,
            ranges=dataclasses.replace(SamplingRanges(), organic_probability=0.0),
        )
        meta = data.metadata
        offset = data.nuisance[:, 0:1]
        slope = data.nuisance[:, 1:2]
        scale = data.nuisance[:, 2:3]

        clean = titration_ph(
            tf.constant(meta[:, 1:2]),
            tf.constant(meta[:, 0:1]),
            tf.constant(pack_constants(meta[:, 2], meta[:, 3])),
            tf.constant(data.curves[:, :, 1]) * tf.constant(scale),
            tf.constant(meta[:, 4:5]),
            tf.constant(meta[:, 5:6]),
            tf.constant(meta[:, 6:7]),
            tf.constant(meta[:, 7:8]),
        ).numpy()
        measured = data.curves[:, :, 0]

        pivot = np.abs(
            calibration_ph + slope * (clean - calibration_ph) + offset - measured
        ).max()
        naive = np.abs(slope * clean + offset - measured).max()
        assert pivot < 1e-8, f"pivot form does not round-trip ({pivot:.3e} pH)"
        assert naive > 1e-2, "the two forms should differ; check calibration_ph"

    def test_scale_is_the_reciprocal_of_the_recorded_error(self):
        data = generate_dataset(8, seed=22, n_points=N_POINTS)
        scale = data.nuisance[:, NUISANCE_COLUMNS.index("pump_scale")]
        assert np.all(np.abs(scale - 1.0) < 0.10)
        assert scale.std() > 0.0


class TestMetadataContract:
    def test_column_orders_are_what_the_model_indexes(self):
        """The decoder slices metadata positionally, so order is load-bearing."""
        assert META_COLUMNS[:4] == ("dic", "alkalinity", "salinity_true", "t_c_true")
        assert META_COLUMNS[4] == "sample_mass_kg"
        assert META_COLUMNS[5:8] == (
            "total_boron",
            "total_phosphate",
            "total_silicate",
        )
        assert META_COLUMNS[8:] == ("total_organic", "pk_organic")
        assert NUISANCE_COLUMNS == ("offset", "slope", "pump_scale")
        # The nuisances are the *trailing* latents, after the composition
        # block.
        assert LATENT_NAMES[N_COMPOSITION_LATENTS:] == NUISANCE_COLUMNS
        assert LATENT_NAMES[:N_COMPOSITION_LATENTS] == (
            "alkalinity",
            "alk_minus_dic",
            "total_organic",
            "pk_organic",
        )
        # Every latent must have a bound and a unique index, or the squashing
        # layer silently builds a shorter vector than the decoder reads.
        assert set(LATENT_BOUNDS) == set(LATENT_NAMES)
        assert {n: i for i, n in enumerate(LATENT_NAMES)} == LATENT_INDEX


class TestOrganicInTheDecoder:
    """The differentiable organic term against the validated NumPy chemistry."""

    @staticmethod
    def _numpy_curve(organic, pk, n_points=24):
        from cleaned_reef_water import simulate_titration

        return simulate_titration(
            2.3e-3, 2.0e-3, 35.0, 25.0,
            sample_mass_kg=0.015, n_points=n_points,
            total_organic=organic, pk_organic=pk,
        )

    def test_matches_numpy_titration_with_organic(self):
        """The TF curve must equal the NumPy curve, not merely resemble it."""
        from cleaned_reef_water.seawater import total_borate

        for organic, pk in ((0.0, 4.5), (7.5e-5, 4.5), (1.5e-4, 6.5)):
            expected = self._numpy_curve(organic, pk)
            got = titration_ph(
                alkalinity=tf.constant([[2.3e-3]], dtype=tf.float64),
                dic=tf.constant([[2.0e-3]], dtype=tf.float64),
                constants=tf.constant(pack_constants([35.0], [25.0])),
                titrant_mass=tf.constant(
                    expected.titrant_mass[None, :], dtype=tf.float64
                ),
                sample_mass=tf.constant([[0.015]], dtype=tf.float64),
                total_boron=tf.constant(
                    [[float(total_borate(35.0))]], dtype=tf.float64
                ),
                total_organic=tf.constant([[organic]], dtype=tf.float64),
                pk_organic=tf.constant([[pk]], dtype=tf.float64),
            )
            assert np.allclose(
                got.numpy()[0], expected.ph_total, atol=1e-8
            ), f"organic={organic}, pk={pk}"

    def test_zero_organic_is_unchanged(self):
        """Adding the term must not perturb the organic-free path at all."""
        from cleaned_reef_water.ml.differentiable import alkalinity_from_ph

        constants = tf.constant(pack_constants([35.0], [25.0]))
        ph = tf.constant([[8.1]], dtype=tf.float64)
        args = (ph, constants, tf.constant([[2.0e-3]], dtype=tf.float64),
                tf.constant([[4.16e-4]], dtype=tf.float64))
        without = float(alkalinity_from_ph(*args).numpy()[0, 0])
        with_zero = float(
            alkalinity_from_ph(*args, 0.0, 0.0, 0.0, 4.5).numpy()[0, 0]
        )
        assert without == with_zero

    def test_organic_raises_alkalinity_at_fixed_ph(self):
        """Sanity of sign: the conjugate base is a proton acceptor."""
        from cleaned_reef_water.ml.differentiable import alkalinity_from_ph

        constants = tf.constant(pack_constants([35.0], [25.0]))
        base = alkalinity_from_ph(
            tf.constant([[8.1]], dtype=tf.float64), constants,
            tf.constant([[2.0e-3]], dtype=tf.float64),
            tf.constant([[4.16e-4]], dtype=tf.float64),
        )
        raised = alkalinity_from_ph(
            tf.constant([[8.1]], dtype=tf.float64), constants,
            tf.constant([[2.0e-3]], dtype=tf.float64),
            tf.constant([[4.16e-4]], dtype=tf.float64),
            0.0, 0.0,
            tf.constant([[1.0e-4]], dtype=tf.float64),
            tf.constant([[4.5]], dtype=tf.float64),
        )
        added = float(raised.numpy()[0, 0]) - float(base.numpy()[0, 0])
        # At pH 8.1, five units above the pKa, essentially all of it.
        assert added == pytest.approx(1.0e-4, rel=1e-3)

    def test_gradient_reaches_both_organic_latents(self):
        """No gradient means the encoder can never learn to set them."""
        from cleaned_reef_water.seawater import total_borate

        expected = self._numpy_curve(7.5e-5, 5.0, n_points=12)
        organic = tf.Variable([[7.5e-5]], dtype=tf.float64)
        pk = tf.Variable([[5.0]], dtype=tf.float64)

        def curve(org, pka):
            return titration_ph(
                alkalinity=tf.constant([[2.3e-3]], dtype=tf.float64),
                dic=tf.constant([[2.0e-3]], dtype=tf.float64),
                constants=tf.constant(pack_constants([35.0], [25.0])),
                titrant_mass=tf.constant(
                    expected.titrant_mass[None, :], dtype=tf.float64
                ),
                sample_mass=tf.constant([[0.015]], dtype=tf.float64),
                total_boron=tf.constant(
                    [[float(total_borate(35.0))]], dtype=tf.float64
                ),
                total_organic=org,
                pk_organic=pka,
            )

        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(curve(organic, pk))
        d_organic, d_pk = tape.gradient(loss, [organic, pk])

        assert d_organic is not None and d_pk is not None, "no gradient path"

        for variable, analytic, step in (
            (organic, d_organic, 1e-9),
            (pk, d_pk, 1e-6),
        ):
            start = variable.numpy().copy()
            variable.assign(start + step)
            high = float(tf.reduce_sum(curve(organic, pk)).numpy())
            variable.assign(start - step)
            low = float(tf.reduce_sum(curve(organic, pk)).numpy())
            variable.assign(start)
            numeric = (high - low) / (2.0 * step)
            assert float(analytic.numpy()[0, 0]) == pytest.approx(
                numeric, rel=1e-5
            )
