"""Dataset persistence, validation, and the model builders."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cleaned_reef_water.ml.dataset import (
    META_COLUMNS,
    NUISANCE_COLUMNS,
    Dataset,
    generate_dataset,
)


def _write(path, data, provenance=None, **overrides):
    """Write a dataset to npz, with optional corruptions."""
    arrays = {
        "curves": data.curves,
        "env": data.env,
        "labels": data.labels,
        "metadata": data.metadata,
        "clean_ph": data.clean_ph,
        "nuisance": data.nuisance,
        "provenance": json.dumps(provenance or {"instrument": "hobbyist"}),
    }
    arrays.update(overrides)
    for key in [k for k, v in overrides.items() if v is None]:
        del arrays[key]
    np.savez_compressed(path, **arrays)
    return path


@pytest.fixture(scope="module")
def data():
    """Generate a small dataset once."""
    return generate_dataset(6, seed=77, n_points=20)


class TestRoundTrip:
    def test_loads_what_was_written(self, tmp_path, data):
        path = _write(tmp_path / "d.npz", data)
        loaded, provenance = Dataset.from_npz(path)
        assert np.array_equal(loaded.curves, data.curves)
        assert np.array_equal(loaded.metadata, data.metadata)
        assert np.array_equal(loaded.nuisance, data.nuisance)
        assert np.array_equal(loaded.clean_ph, data.clean_ph)
        assert provenance["instrument"] == "hobbyist"

    def test_metadata_and_nuisance_have_the_declared_widths(self, data):
        assert data.metadata.shape[1] == len(META_COLUMNS)
        assert data.nuisance.shape[1] == len(NUISANCE_COLUMNS)


class TestRejectsStaleFiles:
    """Each of these silently degraded results before the guard existed."""

    @pytest.mark.parametrize("dropped", ["clean_ph", "nuisance", "labels"])
    def test_missing_array_is_rejected(self, tmp_path, data, dropped):
        path = _write(tmp_path / f"no_{dropped}.npz", data, **{dropped: None})
        with pytest.raises(ValueError, match=dropped):
            Dataset.from_npz(path)

    def test_narrow_metadata_is_rejected_with_the_reason(self, tmp_path, data):
        """The 4-column format that omitted the sample mass."""
        path = _write(tmp_path / "old.npz", data, metadata=data.metadata[:, :4])
        with pytest.raises(ValueError, match="sample mass"):
            Dataset.from_npz(path)

    def test_rejection_message_says_how_to_fix_it(self, tmp_path, data):
        path = _write(tmp_path / "old2.npz", data, metadata=data.metadata[:, :4])
        with pytest.raises(ValueError, match="generate_dataset.py"):
            Dataset.from_npz(path)


class TestNoiseFreeDatasets:
    def test_nuisance_is_the_identity_when_noise_is_off(self):
        """No noise means no offset, unit slope, unit pump scale."""
        clean = generate_dataset(4, seed=78, n_points=20, add_noise=False)
        assert np.allclose(clean.nuisance[:, 0], 0.0)
        assert np.allclose(clean.nuisance[:, 1], 1.0)
        assert np.allclose(clean.nuisance[:, 2], 1.0)

    def test_measured_curve_equals_clean_curve_when_noise_is_off(self):
        clean = generate_dataset(4, seed=79, n_points=20, add_noise=False)
        assert np.allclose(clean.curves[:, :, 0], clean.clean_ph)


class TestModelBuilders:
    """Smoke tests for the reference architectures."""

    def test_speciation_cnn_builds_and_predicts_non_negative(self):
        tf = pytest.importorskip("tensorflow")
        from cleaned_reef_water.ml.architecture import build_speciation_cnn

        model = build_speciation_cnn(n_points=32)
        out = model.predict(
            [np.zeros((3, 32, 2), np.float32), np.zeros((3, 2), np.float32)],
            verbose=0,
        )
        assert out.shape == (3, 3)
        assert np.all(out >= 0.0), "softplus head must not emit negatives"
        del tf

    def test_denoiser_builds_and_returns_a_curve(self):
        tf = pytest.importorskip("tensorflow")
        from cleaned_reef_water.ml.architecture import build_denoiser_cnn

        model = build_denoiser_cnn(n_points=32)
        curve = np.full((3, 32, 2), 8.0, np.float32)
        out = model.predict([curve, np.zeros((3, 2), np.float32)], verbose=0)
        assert out.shape == (3, 32)
        assert np.all(np.isfinite(out))
        del tf

    def test_denoiser_starts_as_the_identity(self):
        """An untrained denoiser must return its input unchanged."""
        pytest.importorskip("tensorflow")
        from cleaned_reef_water.ml.architecture import build_denoiser_cnn

        model = build_denoiser_cnn(n_points=32)
        measured = np.linspace(8.2, 3.5, 32, dtype=np.float32)
        curve = np.stack(
            [np.tile(measured, (2, 1)), np.zeros((2, 32), np.float32)], axis=-1
        )
        out = model.predict([curve, np.zeros((2, 2), np.float32)], verbose=0)
        assert np.allclose(out, measured, atol=1e-4)
