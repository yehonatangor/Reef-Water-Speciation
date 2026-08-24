r"""Physics-constrained autoencoder: CNN encoder, forward-model decoder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from .differentiable import (
    CONSTANT_FIELDS,
    carbonate_species,
    solve_ph,
    titration_ph,
)

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = [
    "LATENT_BOUNDS",
    "LATENT_INDEX",
    "LATENT_NAMES",
    "N_COMPOSITION_LATENTS",
    "build_physics_autoencoder",
    "decode_latents",
    "register_serializable_layers",
    "totals_from_latents",
]


def _latent(latents: Any, name: str) -> Any:
    """Return one latent as an ``(n, 1)`` column, selected by name.

    Parameters
    ----------
    latents
        ``(n, len(LATENT_NAMES))`` array or tensor.
    name
        A key of :data:`LATENT_INDEX`.

    Returns
    -------
    Tensor
        Shape ``(n, 1)``, so it broadcasts against ``(n, p)`` curves.

    Raises
    ------
    KeyError
        If ``name`` is not a latent.  This is the point of the function: a
        mistyped name fails immediately, whereas a mistyped integer index
        returns a different latent and fails silently.
    """
    index = LATENT_INDEX[name]
    return latents[:, index : index + 1]


def totals_from_latents(latents: Any) -> tuple[Any, Any]:
    """Return ``(alkalinity, dic)`` from the latent parameterisation.

    Parameters
    ----------
    latents
        ``(n, len(LATENT_NAMES))`` array or tensor, ordered by :data:`LATENT_NAMES`.

    Returns
    -------
    tuple
        ``(alkalinity, dic)``, each ``(n, 1)``.
    """
    alkalinity = latents[:, 0:1]
    return alkalinity, alkalinity - latents[:, 1:2]

#: Order of the latent vector.  Fixed, because it is read positionally by the
#: decoder and by every evaluation script.
LATENT_NAMES: Final[tuple[str, ...]] = (
    "alkalinity",
    "alk_minus_dic",
    "total_organic",
    "pk_organic",
    "offset",
    "slope",
    "pump_scale",
)

#: How many leading latents describe the *water* rather than the *instrument*.
N_COMPOSITION_LATENTS: Final[int] = 4

#: Column index of each latent.
LATENT_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(LATENT_NAMES)
}

#: Bounds on each latent, in the units of :data:`LATENT_NAMES`.
LATENT_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "alkalinity": (0.5e-3, 6.0e-3),
    "alk_minus_dic": (1.0e-5, 1.5e-3),
    "total_organic": (0.0, 2.25e-4),
    "pk_organic": (3.5, 7.5),
    "offset": (-0.5, 0.5),
    "slope": (0.8, 1.2),
    "pump_scale": (0.90, 1.10),
}


def _tf() -> Any:
    """Import TensorFlow, with an actionable error if it is missing."""
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The physics-constrained autoencoder requires TensorFlow. "
            "Install it with:  pip install 'cleaned_reef_water[ml]'"
        ) from exc
    return tf


def _register(cls: Any) -> Any:
    """Register a layer for Keras serialisation, if Keras is importable."""
    try:
        import keras  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover - depends on environment
        return cls
    registered: Any = keras.saving.register_keras_serializable(
        package="cleaned_reef_water"
    )(cls)
    return registered


#: Cache for :func:`_make_layers`.
_LAYER_CACHE: list[tuple[Any, Any, Any]] = []


def register_serializable_layers() -> None:
    """Make this module's custom layers visible to ``keras.models.load_model``.

    Examples
    --------
    >>> import tensorflow as tf                          # doctest: +SKIP
    >>> register_serializable_layers()                   # doctest: +SKIP
    >>> model = tf.keras.models.load_model(path, compile=False)  # doctest: +SKIP
    """
    _make_layers()


def _make_layers() -> tuple[Any, Any, Any]:
    """Build the two custom layers lazily, so importing needs no TensorFlow."""
    if _LAYER_CACHE:
        return _LAYER_CACHE[0]
    tf = _tf()
    layers = tf.keras.layers
    # Bound to a local so mypy sees an untyped base rather than an attribute
    # of a runtime value, which it cannot resolve as a class.
    layer_base: Any = layers.Layer

    @_register
    class PhysicalLatents(layer_base):  # type: ignore[misc, valid-type]
        """Squash unconstrained activations into physically bounded latents."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._low = tf.constant(
                [LATENT_BOUNDS[name][0] for name in LATENT_NAMES], dtype=tf.float64
            )
            self._high = tf.constant(
                [LATENT_BOUNDS[name][1] for name in LATENT_NAMES], dtype=tf.float64
            )

        def call(self, inputs: Any) -> Any:
            raw = tf.cast(inputs, tf.float64)
            return self._low + (self._high - self._low) * tf.sigmoid(raw)

        def compute_output_shape(self, input_shape: Any) -> Any:
            return input_shape

    @_register
    class TitrationDecoder(layer_base):  # type: ignore[misc, valid-type]
        r"""Render latents into a measured-space pH curve. No trainable weights."""

        def __init__(
            self,
            titrant_concentration: float = 0.1,
            calibration_ph: float = 8.0936,
            n_iterations: int = 60,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.titrant_concentration = titrant_concentration
            self.calibration_ph = calibration_ph
            self.n_iterations = n_iterations

        def call(self, inputs: list[Any]) -> Any:
            latents, constants, titrant_mass, sample_mass, totals = inputs
            latents = tf.cast(latents, tf.float64)
            alkalinity, dic = totals_from_latents(latents)
            clean = titration_ph(
                alkalinity=alkalinity,
                dic=dic,
                constants=tf.cast(constants, tf.float64),
                titrant_mass=(
                    tf.cast(titrant_mass, tf.float64) * _latent(latents, "pump_scale")
                ),
                sample_mass=tf.cast(sample_mass, tf.float64),
                total_boron=tf.cast(totals[:, 0:1], tf.float64),
                total_phosphate=tf.cast(totals[:, 1:2], tf.float64),
                total_silicate=tf.cast(totals[:, 2:3], tf.float64),
                total_organic=_latent(latents, "total_organic"),
                pk_organic=_latent(latents, "pk_organic"),
                titrant_concentration=self.titrant_concentration,
                n_iterations=self.n_iterations,
            )
            pivot = tf.constant(self.calibration_ph, dtype=tf.float64)
            return (
                pivot
                + _latent(latents, "slope") * (clean - pivot)
                + _latent(latents, "offset")
            )

        def compute_output_shape(self, input_shape: Any) -> Any:
            return input_shape[2]

        def get_config(self) -> dict[str, Any]:
            config: dict[str, Any] = super().get_config()
            config.update(
                {
                    "titrant_concentration": self.titrant_concentration,
                    "calibration_ph": self.calibration_ph,
                    "n_iterations": self.n_iterations,
                }
            )
            return config

    @_register
    class SpeciesHead(layer_base):  # type: ignore[misc, valid-type]
        r"""Compute ``(HCO3, CO3, B(OH)4)`` from the latents. No weights."""

        def __init__(self, n_iterations: int = 60, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.n_iterations = n_iterations

        def call(self, inputs: list[Any]) -> Any:
            latents, constants, totals = inputs
            latents = tf.cast(latents, tf.float64)
            constants = tf.cast(constants, tf.float64)
            boron = tf.cast(totals[:, 0:1], tf.float64)
            alkalinity, dic = totals_from_latents(latents)
            # At zero titrant the target alkalinity is A_T itself.
            ph = solve_ph(
                alkalinity,
                constants,
                dic,
                boron,
                total_organic=_latent(latents, "total_organic"),
                pk_organic=_latent(latents, "pk_organic"),
                n_iterations=self.n_iterations,
            )
            hco3, co3, boh4 = carbonate_species(ph, constants, dic, boron)
            return tf.concat([hco3, co3, boh4], axis=-1)

        def compute_output_shape(self, input_shape: Any) -> Any:
            return (input_shape[0][0], 3)

        def get_config(self) -> dict[str, Any]:
            config: dict[str, Any] = super().get_config()
            config.update({"n_iterations": self.n_iterations})
            return config

    _LAYER_CACHE.append((PhysicalLatents, TitrationDecoder, SpeciesHead))
    return _LAYER_CACHE[0]


def decode_latents(latents: Any) -> dict[str, Any]:
    """Split a latent array into named columns.

    Parameters
    ----------
    latents
        ``(n, len(LATENT_NAMES))`` array ordered as :data:`LATENT_NAMES`.

    Returns
    -------
    dict
        One entry per latent, each ``(n,)``.

    Examples
    --------
    >>> import numpy as np
    >>> out = decode_latents(
    ...     np.array([[2.3e-3, 3.0e-4, 7.5e-5, 4.5, 0.01, 1.0, 1.002]])
    ... )
    >>> [f"{name}={float(out[name][0]):g}" for name in LATENT_NAMES]
    ['alkalinity=0.0023', 'alk_minus_dic=0.0003', 'total_organic=7.5e-05', \
'pk_organic=4.5', 'offset=0.01', 'slope=1', 'pump_scale=1.002']
    >>> float(out["alkalinity"][0] - out["alk_minus_dic"][0])  # DIC
    0.002
    """
    import numpy as np

    array = np.asarray(latents, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(LATENT_NAMES):
        raise ValueError(
            f"latents must have shape (n, {len(LATENT_NAMES)}), got {array.shape}"
        )
    return {name: array[:, i] for i, name in enumerate(LATENT_NAMES)}


def build_physics_autoencoder(
    n_points: int,
    *,
    n_curve_channels: int = 2,
    n_env_features: int = 2,
    filters: tuple[int, ...] = (32, 64, 128),
    kernel_size: int = 5,
    dense_units: int = 256,
    dropout: float = 0.1,
    titrant_concentration: float = 0.1,
    calibration_ph: float = 8.0936,
    n_solver_iterations: int = 60,
) -> Any:
    r"""Build the physics-constrained autoencoder.

    Parameters
    ----------
    n_points
        Titration points per curve.
    n_curve_channels
        Channels in the curve input; 2 for ``(pH, titrant mass)``.
    n_env_features
        Environmental scalars; 2 for ``(salinity, temperature)``.
    filters, kernel_size, dense_units, dropout
        Encoder capacity.
    titrant_concentration
        Titrant concentration in ``mol kg-soln^-1``.
    calibration_ph
        pH at which the electrode was calibrated.  The slope error pivots
        about this point; it must match the noise model's ``calibration_ph``.
    n_solver_iterations
        Bisection iterations in the decoder.  These carry no tape memory, so
        the cost is compute-only.

    Returns
    -------
    tf.keras.Model
        Inputs, all keyword-named:

        ``curve``
            ``(n_points, n_curve_channels)`` normalised measured curve.
        ``env``
            ``(n_env_features,)`` normalised salinity and temperature.
        ``constants``
            ``(13,)`` packed equilibrium constants; see
            :func:`~cleaned_reef_water.ml.differentiable.pack_constants`.
        ``titrant_mass``
            ``(n_points,)`` measured titrant mass in kg, **unnormalised**.
        ``sample_mass``
            ``(1,)`` sample mass in kg, **unnormalised**.
        ``totals``
            ``(3,)`` boron, phosphate and silicate totals in mol/kg.

        Outputs:

        ``reconstruction``
            ``(n_points,)`` predicted *measured* pH curve.
        ``latents``
            ``(len(LATENT_NAMES),)`` physical parameters, in that order.
    """
    tf = _tf()
    layers = tf.keras.layers
    latent_layer_cls, decoder_cls, species_cls = _make_layers()

    curve_input = layers.Input(shape=(n_points, n_curve_channels), name="curve")
    env_input = layers.Input(shape=(n_env_features,), name="env")
    constants_input = layers.Input(shape=(len(CONSTANT_FIELDS),), name="constants")
    mass_input = layers.Input(shape=(n_points,), name="titrant_mass")
    sample_mass_input = layers.Input(shape=(1,), name="sample_mass")
    totals_input = layers.Input(shape=(3,), name="totals")

    # Fused early so the convolutions condition on it: S and T change the shape.
    env_sequence = layers.RepeatVector(n_points)(env_input)
    x = layers.Concatenate(axis=-1)([curve_input, env_sequence])

    for width in filters:
        x = layers.Conv1D(width, kernel_size, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

    # Never global average pooling: it is invariant to the inflection position.
    x = layers.Flatten()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units // 2, activation="relu")(x)

    # A linear fit on these summary statistics alone reaches sd 28.0, against 58.9.
    features = layers.Dense(dense_units // 2, activation="relu")(env_input)
    features = layers.Dense(dense_units // 2, activation="relu")(features)
    x = layers.Concatenate()([x, features, env_input])

    # Zero bias centres every latent, so slope and pump scale start at unity.
    totals_logits = layers.Dense(
        N_COMPOSITION_LATENTS,
        kernel_initializer=tf.keras.initializers.RandomNormal(
            stddev=1e-3, seed=0
        ),
        bias_initializer="zeros",
        name="totals_logits",
    )(x)
    conditioned = layers.Concatenate()([x, totals_logits])
    nuisance_logits = layers.Dense(
        len(LATENT_NAMES) - N_COMPOSITION_LATENTS,
        kernel_initializer=tf.keras.initializers.RandomNormal(
            stddev=1e-3, seed=1
        ),
        bias_initializer="zeros",
        name="nuisance_logits",
    )(conditioned)
    raw = layers.Concatenate(name="latent_logits")(
        [totals_logits, nuisance_logits]
    )
    latents = latent_layer_cls(name="latents", dtype="float64")(raw)

    reconstruction = decoder_cls(
        titrant_concentration=titrant_concentration,
        calibration_ph=calibration_ph,
        n_iterations=n_solver_iterations,
        name="reconstruction",
        dtype="float64",
    )([latents, constants_input, mass_input, sample_mass_input, totals_input])

    species = species_cls(
        n_iterations=n_solver_iterations, name="species", dtype="float64"
    )([latents, constants_input, totals_input])

    return tf.keras.Model(
        inputs={
            "curve": curve_input,
            "env": env_input,
            "constants": constants_input,
            "titrant_mass": mass_input,
            "sample_mass": sample_mass_input,
            "totals": totals_input,
        },
        outputs={
            "reconstruction": reconstruction,
            "latents": latents,
            "species": species,
        },
        name="physics_autoencoder",
    )
