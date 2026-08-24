"""Feature and label scaling, fitted on the training split only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..types import FloatArray

__all__ = ["Normalizer", "EPSILON"]

#: Floor applied to every standard deviation before division.
EPSILON: float = 1e-12


@dataclass(frozen=True, slots=True)
class Normalizer:
    """Z-score statistics for curve, environment and label tensors.

    Attributes
    ----------
    curve_mean, curve_std
        Per-channel statistics of shape ``(1, 1, n_channels)``.
    env_mean, env_std
        Per-feature statistics of shape ``(n_env,)``.
    label_mean, label_std
        Per-target offset and divisor, shape ``(n_targets,)``.  ``label_mean``
        is all zeros unless fitted with ``center_labels=True``.  ``label_std``
        holds whichever divisor ``label_scale`` selected -- the label mean by
        default, not the standard deviation.  See the module docstring.
    """

    curve_mean: FloatArray
    curve_std: FloatArray
    env_mean: FloatArray
    env_std: FloatArray
    label_mean: FloatArray
    label_std: FloatArray

    @classmethod
    def fit(
        cls,
        curves: FloatArray,
        env: FloatArray,
        labels: FloatArray,
        *,
        center_labels: bool = False,
        label_scale: str = "mean",
    ) -> Normalizer:
        """Fit statistics on a training split.

        Parameters
        ----------
        curves
            Array of shape ``(n_samples, n_points, n_channels)``.
        env
            Array of shape ``(n_samples, n_env)``.
        labels
            Array of shape ``(n_samples, n_targets)``.
        center_labels
            Subtract the label mean as well as dividing.  Defaults to
            ``False`` because the CNN uses a ``softplus`` output head, which
            cannot represent negative values; see the module docstring.
        label_scale
            ``"mean"`` (default) divides each target by its mean, putting every
            scaled target at 1.0 on average.  ``"std"`` divides by the standard
            deviation, the classical choice, which leaves the targets far from
            where a softplus head starts.

        Returns
        -------
        Normalizer
            Fitted statistics.

        Raises
        ------
        ValueError
            If the arrays disagree on ``n_samples``, ``curves`` is not
            three-dimensional, or ``label_scale`` is unrecognised.
        """
        if label_scale not in ("mean", "std"):
            raise ValueError(
                f"label_scale must be 'mean' or 'std', got {label_scale!r}"
            )
        if curves.ndim != 3:
            raise ValueError(f"curves must be 3-D, got shape {curves.shape}")
        n = curves.shape[0]
        if env.shape[0] != n or labels.shape[0] != n:
            raise ValueError(
                "curves, env and labels must share the first dimension; got "
                f"{curves.shape[0]}, {env.shape[0]}, {labels.shape[0]}"
            )
        def floor(values: FloatArray) -> FloatArray:
            """Clamp a standard deviation away from zero."""
            return np.maximum(values, EPSILON)

        return cls(
            curve_mean=curves.mean(axis=(0, 1), keepdims=True),
            curve_std=floor(curves.std(axis=(0, 1), keepdims=True)),
            env_mean=env.mean(axis=0),
            env_std=floor(env.std(axis=0)),
            label_mean=(
                labels.mean(axis=0)
                if center_labels
                else np.zeros(labels.shape[1], dtype=np.float64)
            ),
            label_std=floor(
                np.abs(labels.mean(axis=0))
                if label_scale == "mean"
                else labels.std(axis=0)
            ),
        )

    def transform_curves(self, curves: FloatArray) -> FloatArray:
        """Standardise a curve tensor."""
        return (curves - self.curve_mean) / self.curve_std

    def transform_env(self, env: FloatArray) -> FloatArray:
        """Standardise an environment tensor."""
        return (env - self.env_mean) / self.env_std

    def transform_labels(self, labels: FloatArray) -> FloatArray:
        """Standardise a label tensor."""
        return (labels - self.label_mean) / self.label_std

    def inverse_labels(self, labels: FloatArray) -> FloatArray:
        """Map standardised labels back to physical units.

        Parameters
        ----------
        labels
            Standardised predictions of shape ``(n_samples, n_targets)``.

        Returns
        -------
        FloatArray
            Predictions in ``mol kg-soln^-1``.
        """
        return labels * self.label_std + self.label_mean

    def save(self, path: str | Path) -> None:
        """Write the statistics to a ``.npz`` file.

        Parameters
        ----------
        path
            Destination path.
        """
        np.savez(
            path,
            curve_mean=self.curve_mean,
            curve_std=self.curve_std,
            env_mean=self.env_mean,
            env_std=self.env_std,
            label_mean=self.label_mean,
            label_std=self.label_std,
        )

    @classmethod
    def load(cls, path: str | Path) -> Normalizer:
        """Read statistics previously written by :meth:`save`.

        Parameters
        ----------
        path
            Source path.

        Returns
        -------
        Normalizer
            The stored statistics.
        """
        with np.load(path) as data:
            return cls(**{key: data[key] for key in data.files})
