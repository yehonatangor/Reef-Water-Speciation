"""The differentiable forward model must match the exact one, in value and slope."""

from __future__ import annotations

import numpy as np
import pytest

from cleaned_reef_water import simulate_titration, speciate
from cleaned_reef_water.alkalinity import constants_at, total_alkalinity
from cleaned_reef_water.ml.differentiable import (
    CONSTANT_FIELDS,
    alkalinity_from_ph,
    pack_constants,
    titration_ph,
)
from cleaned_reef_water.seawater import total_borate

tf = pytest.importorskip("tensorflow", reason="differentiable model needs TensorFlow")

S_REF, T_REF = 35.0, 25.0


@pytest.fixture(scope="module")
def batch():
    """Build a small batch spanning the sampled composition ranges."""
    rng = np.random.default_rng(3)
    n, points = 6, 40
    alkalinity = rng.uniform(1.5e-3, 4.0e-3, (n, 1))
    dic = alkalinity * rng.uniform(0.80, 0.95, (n, 1))
    salinity = rng.uniform(30.0, 36.0, n)
    t_c = rng.uniform(22.0, 28.0, n)
    sample_mass = rng.uniform(0.0145, 0.0155, (n, 1))
    phosphate = np.where(rng.random((n, 1)) < 0.5, 1e-5, 0.0)
    silicate = np.where(rng.random((n, 1)) < 0.5, 5e-5, 0.0)
    boron = np.array([float(total_borate(s)) for s in salinity]).reshape(n, 1)

    reference = np.empty((n, points))
    masses = np.empty((n, points))
    for i in range(n):
        curve = simulate_titration(
            float(alkalinity[i, 0]),
            float(dic[i, 0]),
            float(salinity[i]),
            float(t_c[i]),
            sample_mass_kg=float(sample_mass[i, 0]),
            n_points=points,
            total_phosphate=float(phosphate[i, 0]),
            total_silicate=float(silicate[i, 0]),
            total_boron=float(boron[i, 0]),
        )
        reference[i] = curve.ph_total
        masses[i] = curve.titrant_mass

    return {
        "alkalinity": alkalinity,
        "dic": dic,
        "salinity": salinity,
        "t_c": t_c,
        "sample_mass": sample_mass,
        "phosphate": phosphate,
        "silicate": silicate,
        "boron": boron,
        "reference_ph": reference,
        "masses": masses,
        "constants": pack_constants(salinity, t_c),
    }


def _curve(data, alkalinity=None, dic=None):
    """Render the batch through the differentiable model."""
    return titration_ph(
        tf.constant(data["alkalinity"]) if alkalinity is None else alkalinity,
        tf.constant(data["dic"]) if dic is None else dic,
        tf.constant(data["constants"]),
        tf.constant(data["masses"]),
        tf.constant(data["sample_mass"]),
        tf.constant(data["boron"]),
        tf.constant(data["phosphate"]),
        tf.constant(data["silicate"]),
    )


class TestPackConstants:
    def test_shape_and_positivity(self):
        packed = pack_constants([35.0, 33.0], [25.0, 20.0])
        assert packed.shape == (2, len(CONSTANT_FIELDS))
        assert np.all(packed > 0.0)

    def test_columns_match_the_named_constants(self):
        packed = pack_constants(S_REF, T_REF)
        constants = constants_at(S_REF, T_REF)
        for index, field in enumerate(CONSTANT_FIELDS):
            assert packed[0, index] == pytest.approx(
                float(getattr(constants, field)), rel=1e-15
            ), f"column {index} is not {field}"

    def test_mismatched_shapes_rejected(self):
        with pytest.raises(ValueError, match="same shape"):
            pack_constants([35.0, 33.0], [25.0])


class TestAlkalinityAgreesWithNumpy:
    """The tensor speciation must equal the validated NumPy speciation."""

    @pytest.mark.parametrize("ph", [4.0, 6.5, 8.1, 9.5])
    def test_matches_numpy_total_alkalinity(self, ph):
        constants = constants_at(S_REF, T_REF)
        dic, boron = 2.0e-3, float(total_borate(S_REF))
        phosphate, silicate = 1e-5, 5e-5

        expected = float(
            total_alkalinity(
                speciate(
                    ph,
                    constants,
                    dic=dic,
                    total_boron=boron,
                    total_phosphate=phosphate,
                    total_silicate=silicate,
                )
            )
        )
        # .item() rather than float(): converting a shape-(1, 1) tensor with
        # float() is a DeprecationWarning on some NumPy versions and a
        # TypeError on others.
        got = (
            alkalinity_from_ph(
                tf.constant([[ph]], dtype=tf.float64),
                tf.constant(pack_constants(S_REF, T_REF)),
                dic,
                boron,
                phosphate,
                silicate,
            )
            .numpy()
            .item()
        )
        assert got == pytest.approx(expected, rel=1e-12)


class TestCurveAgreesWithExactSolver:
    def test_ph_matches_simulate_titration(self, batch):
        got = _curve(batch).numpy()
        deviation = np.abs(got - batch["reference_ph"])
        assert deviation.max() < 1e-8, (
            f"differentiable model deviates from the exact solver by "
            f"{deviation.max():.3e} pH"
        )

    def test_curve_is_monotone_decreasing(self, batch):
        got = _curve(batch).numpy()
        assert np.all(np.diff(got, axis=1) < 0.0)

    def test_first_point_is_the_undisturbed_sample(self, batch):
        """At zero titrant the curve must give the sample's own pH."""
        got = _curve(batch).numpy()
        assert np.allclose(got[:, 0], batch["reference_ph"][:, 0], atol=1e-8)


class TestGradients:
    """Without the implicit-function-theorem step these all return zero."""

    def _analytic_and_numeric(self, batch, key):
        variable = tf.Variable(batch[key])
        with tf.GradientTape() as tape:
            curve = _curve(
                batch,
                alkalinity=variable if key == "alkalinity" else None,
                dic=variable if key == "dic" else None,
            )
            loss = tf.reduce_sum(curve * curve)
        analytic = tape.gradient(loss, variable).numpy()

        def loss_at(values):
            curve = _curve(
                batch,
                alkalinity=values if key == "alkalinity" else None,
                dic=values if key == "dic" else None,
            )
            return float(tf.reduce_sum(curve * curve))

        step = 1e-9
        numeric = np.zeros_like(analytic)
        for i in range(analytic.shape[0]):
            up, down = batch[key].copy(), batch[key].copy()
            up[i] += step
            down[i] -= step
            numeric[i] = (loss_at(up) - loss_at(down)) / (2.0 * step)
        return analytic, numeric

    @pytest.mark.parametrize("key", ["alkalinity", "dic"])
    def test_matches_finite_differences(self, batch, key):
        analytic, numeric = self._analytic_and_numeric(batch, key)
        relative = np.abs(analytic - numeric) / np.abs(numeric)
        assert relative.max() < 1e-6, (
            f"d/d{key} disagrees with finite differences by {relative.max():.3e}"
        )

    def test_gradient_is_not_silently_zero(self, batch):
        """Guards the Newton correction."""
        variable = tf.Variable(batch["alkalinity"])
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(_curve(batch, alkalinity=variable))
        gradient = tape.gradient(loss, variable).numpy()
        assert np.all(np.abs(gradient) > 1e-6)

    def test_gradient_flows_to_titrant_mass(self, batch):
        """The pump-scale parameter multiplies the mass axis, so this must work."""
        scale = tf.Variable(np.ones((batch["alkalinity"].shape[0], 1)))
        with tf.GradientTape() as tape:
            curve = titration_ph(
                tf.constant(batch["alkalinity"]),
                tf.constant(batch["dic"]),
                tf.constant(batch["constants"]),
                tf.constant(batch["masses"]) * scale,
                tf.constant(batch["sample_mass"]),
                tf.constant(batch["boron"]),
                tf.constant(batch["phosphate"]),
                tf.constant(batch["silicate"]),
            )
            loss = tf.reduce_sum(curve)
        gradient = tape.gradient(loss, scale).numpy()
        assert np.all(np.abs(gradient) > 1e-6)
