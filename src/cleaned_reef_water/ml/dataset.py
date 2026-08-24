"""Synthetic dataset generation and reproducible splitting."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..alkalinity import constants_at, speciate
from ..equilibria import CarbonicFormulation
from ..solvers import ConvergenceError
from ..titration import STRONG_ACID, Titrant, simulate_titration
from ..types import FloatArray
from .noise import NoiseModel, apply_noise
from .sampling import SamplingRanges, sample_composition

__all__ = [
    "META_COLUMNS",
    "NUISANCE_COLUMNS",
    "Dataset",
    "generate_dataset",
    "split_dataset",
]

#: Column order of :attr:`Dataset.nuisance`, the realised per-curve instrument error.
NUISANCE_COLUMNS: tuple[str, ...] = ("offset", "slope", "pump_scale")

#: Column order of :attr:`Dataset.metadata`.
META_COLUMNS: tuple[str, ...] = (
    "dic",
    "alkalinity",
    "salinity_true",
    "t_c_true",
    "sample_mass_kg",
    "total_boron",
    "total_phosphate",
    "total_silicate",
    "total_organic",
    "pk_organic",
)


@dataclass(frozen=True, slots=True)
class Dataset:
    """A generated synthetic dataset.

    Attributes
    ----------
    curves
        ``(n, n_points, 2)`` array of ``(pH, titrant mass)``.
    env
        ``(n, 2)`` array of measured ``(salinity, temperature)``.
    labels
        ``(n, 3)`` array of ``(HCO3, CO3, B(OH)4)`` in ``mol kg-soln^-1``.
    metadata
        ``(n, 8)`` array with columns given by :data:`META_COLUMNS`.  Includes
        the true sample mass and the nutrient totals, both of which the
        chemical inverse and the differentiable decoder require.
    clean_ph
        ``(n, n_points)`` array of the noise-free pH curve.  This is the
        training target for the denoiser in
        :func:`~cleaned_reef_water.ml.architecture.build_denoiser_cnn`: the
        forward model gives it to us exactly, for free, which makes it a far
        richer supervision signal than three scalar labels.
    nuisance
        ``(n, 3)`` array of the realised instrument error, with columns given
        by :data:`NUISANCE_COLUMNS`.  All zeros-and-ones (the no-error values)
        when ``add_noise`` is ``False``.
    n_failed
        Number of draws discarded because the forward model did not converge.
    """

    curves: FloatArray
    env: FloatArray
    labels: FloatArray
    metadata: FloatArray
    clean_ph: FloatArray
    nuisance: FloatArray
    n_failed: int

    @classmethod
    def from_npz(cls, path: str | Path) -> tuple[Dataset, dict[str, object]]:
        """Load a dataset written by ``scripts/generate_dataset.py``.

        Parameters
        ----------
        path
            Path to the ``.npz`` file.

        Returns
        -------
        tuple
            ``(dataset, provenance)``.

        Raises
        ------
        ValueError
            If the file predates any required array or metadata column.
        """
        required = ("curves", "env", "labels", "metadata", "clean_ph", "nuisance")
        regenerate = (
            "Regenerate it with:\n"
            "    python scripts/generate_dataset.py --instrument <grade>"
        )
        with np.load(path, allow_pickle=False) as stored:
            missing = [key for key in required if key not in stored]
            if missing:
                raise ValueError(
                    f"{path} is missing {missing}; it predates the current "
                    f"format. {regenerate}"
                )
            if stored["metadata"].shape[1] != len(META_COLUMNS):
                raise ValueError(
                    f"{path} has {stored['metadata'].shape[1]} metadata "
                    f"columns, expected {len(META_COLUMNS)} {META_COLUMNS}. "
                    f"Without the true sample mass the inverse assumes a "
                    f"nominal 0.015 kg and carries ~41 umol/kg of avoidable "
                    f"error. {regenerate}"
                )
            dataset = cls(
                curves=stored["curves"],
                env=stored["env"],
                labels=stored["labels"],
                metadata=stored["metadata"],
                clean_ph=stored["clean_ph"],
                nuisance=stored["nuisance"],
                n_failed=0,
            )
            provenance = json.loads(str(stored["provenance"]))
        return dataset, provenance


def generate_dataset(
    n_samples: int,
    *,
    seed: int = 0,
    n_points: int = 200,
    titrant: Titrant = STRONG_ACID,
    ranges: SamplingRanges | None = None,
    noise_model: NoiseModel | None = None,
    formulation: CarbonicFormulation = CarbonicFormulation.LUEKER2000,
    add_noise: bool = True,
    progress_every: int = 0,
) -> Dataset:
    """Generate synthetic titration curves with matched speciation labels.

    Parameters
    ----------
    n_samples
        Number of successful samples to produce.
    seed
        Seed for the generator.  Data generation is fully reproducible.
    n_points
        Points per curve.
    titrant
        Titrant model.
    ranges
        Composition sampling ranges.
    noise_model
        Instrument noise tolerances.
    formulation
        Carbonic acid parameterisation.
    add_noise
        When ``False``, returns clean curves and exact environmental values,
        which is useful for isolating model error from measurement error.
    progress_every
        Print a progress line every this many successful samples.  ``0``
        disables progress output.  Useful for locating a crash: the last line
        printed bounds where the failure occurred.

    Returns
    -------
    Dataset
        The generated dataset.

    Raises
    ------
    ValueError
        If ``n_samples`` is not positive.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    rng = np.random.default_rng(seed)
    curves = np.empty((n_samples, n_points, 2), dtype=np.float64)
    env = np.empty((n_samples, 2), dtype=np.float64)
    labels = np.empty((n_samples, 3), dtype=np.float64)
    metadata = np.empty((n_samples, len(META_COLUMNS)), dtype=np.float64)
    clean_ph = np.empty((n_samples, n_points), dtype=np.float64)
    nuisance = np.empty((n_samples, len(NUISANCE_COLUMNS)), dtype=np.float64)

    filled = 0
    failed = 0
    started = time.perf_counter()
    while filled < n_samples:
        composition = sample_composition(rng, ranges, formulation)
        constants = constants_at(composition.salinity, composition.t_c, formulation)
        try:
            curve = simulate_titration(
                composition.alkalinity,
                composition.dic,
                composition.salinity,
                composition.t_c,
                titrant=titrant,
                sample_mass_kg=composition.sample_mass_kg,
                n_points=n_points,
                total_phosphate=composition.total_phosphate,
                total_silicate=composition.total_silicate,
                total_organic=composition.total_organic,
                pk_organic=composition.pk_organic,
                total_boron=composition.total_boron,
                constants=constants,
            )
        except ConvergenceError:
            failed += 1
            continue

        species = speciate(
            composition.ph_total,
            constants,
            dic=composition.dic,
            total_boron=composition.total_boron,
            total_phosphate=composition.total_phosphate,
            total_silicate=composition.total_silicate,
            total_organic=composition.total_organic,
            pk_organic=composition.pk_organic,
        )

        if add_noise:
            noisy = apply_noise(
                curve.titrant_mass,
                curve.ph_total,
                composition.salinity,
                composition.t_c,
                model=noise_model,
                rng=rng,
            )
            ph_axis, mass_axis = noisy.ph_measured, noisy.titrant_mass
            s_obs, t_obs = noisy.salinity_measured, noisy.t_c_measured
            # ``offset`` is the drawn calibration offset PLUS the NBS
            # calibration bias.
            realised = (
                noisy.truth["offset"] + noisy.truth["nbs_calibration_bias"],
                noisy.truth["slope"],
                1.0 / (1.0 + noisy.truth["burette_scale_error"]),
            )
        else:
            ph_axis, mass_axis = curve.ph_total, curve.titrant_mass
            s_obs, t_obs = composition.salinity, composition.t_c
            realised = (0.0, 1.0, 1.0)

        if not np.all(np.isfinite(ph_axis)):
            failed += 1
            continue

        curves[filled, :, 0] = ph_axis
        curves[filled, :, 1] = mass_axis
        clean_ph[filled] = curve.ph_total
        env[filled] = (s_obs, t_obs)
        labels[filled] = (
            float(species.hco3),
            float(species.co3),
            float(species.boh4),
        )
        metadata[filled] = (
            composition.dic,
            composition.alkalinity,
            composition.salinity,
            composition.t_c,
            composition.sample_mass_kg,
            composition.total_boron,
            composition.total_phosphate,
            composition.total_silicate,
            composition.total_organic,
            composition.pk_organic,
        )
        nuisance[filled] = realised
        filled += 1
        if progress_every and filled % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"    {filled}/{n_samples} curves, {failed} discarded, "
                f"{elapsed:.0f}s",
                flush=True,
            )

    return Dataset(
        curves=curves,
        env=env,
        labels=labels,
        metadata=metadata,
        clean_ph=clean_ph,
        nuisance=nuisance,
        n_failed=failed,
    )


def split_dataset(
    dataset: Dataset,
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[dict[str, FloatArray], dict[str, FloatArray], dict[str, FloatArray]]:
    """Split a dataset into train, validation and test partitions.

    Parameters
    ----------
    dataset
        The dataset to split.
    train_fraction, val_fraction
        Fractions of the data assigned to training and validation.  The
        remainder becomes the test set.
    seed
        Seed for the shuffling permutation.

    Returns
    -------
    tuple of dict
        ``(train, val, test)``, each a dict with keys ``curves``, ``env``,
        ``labels`` and ``metadata``.

    Raises
    ------
    ValueError
        If the fractions do not leave a non-empty test set.
    """
    if not 0.0 < train_fraction < 1.0 or not 0.0 < val_fraction < 1.0:
        raise ValueError("fractions must lie strictly between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            "train_fraction + val_fraction must be < 1 to leave a test set, got "
            f"{train_fraction} + {val_fraction}"
        )

    n = dataset.curves.shape[0]
    order = np.random.default_rng(seed).permutation(n)
    n_train = int(train_fraction * n)
    n_val = int((train_fraction + val_fraction) * n)
    slices = (order[:n_train], order[n_train:n_val], order[n_val:])

    return tuple(  # type: ignore[return-value]
        {
            "curves": dataset.curves[idx],
            "env": dataset.env[idx],
            "labels": dataset.labels[idx],
            "metadata": dataset.metadata[idx],
            "clean_ph": dataset.clean_ph[idx],
            "nuisance": dataset.nuisance[idx],
        }
        for idx in slices
    )
