r"""TensorFlow/Keras architecture for carbonate speciation from titration curves."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import tensorflow as tf

__all__ = ["build_speciation_cnn", "build_denoiser_cnn", "OUTPUT_NAMES"]

#: Order of the regression targets emitted by the network.
OUTPUT_NAMES: tuple[str, ...] = ("hco3", "co3", "boh4")


def build_speciation_cnn(
    n_points: int = 200,
    n_curve_channels: int = 2,
    n_env_features: int = 2,
    n_outputs: int = 3,
    *,
    dropout: float = 0.2,
    filters: tuple[int, ...] = (32, 64, 128),
    kernel_sizes: tuple[int, ...] = (7, 5, 3),
    dense_units: int = 128,
    name: str = "speciation_cnn",
) -> tf.keras.Model:
    """Build the two-branch 1-D CNN.

    Parameters
    ----------
    n_points
        Number of points per titration curve.
    n_curve_channels
        Channels in the curve tensor -- normally ``2`` for ``(pH, titrant
        mass)``.
    n_env_features
        Number of environmental scalars, normally ``2`` for ``(S, t)``.
    n_outputs
        Number of regression targets.
    dropout
        Dropout rate applied in the dense head.
    filters
        Number of filters in each convolutional block.
    kernel_sizes
        Kernel size for each convolutional block.  Must match ``filters`` in
        length.
    dense_units
        Width of the dense layer in the head.
    name
        Model name.

    Returns
    -------
    tf.keras.Model
        A compiled-ready functional model with inputs ``["curve", "env"]``.

    Raises
    ------
    ImportError
        If TensorFlow is not installed.  The chemistry core of this package
        has no TensorFlow dependency; only this module does.
    ValueError
        If ``filters`` and ``kernel_sizes`` differ in length.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "TensorFlow is required for cleaned_reef_water.ml.architecture. "
            "Install it with `pip install 'cleaned-reef-water[ml]'`. The "
            "chemistry modules do not require TensorFlow."
        ) from exc

    if len(filters) != len(kernel_sizes):
        raise ValueError(
            f"filters and kernel_sizes must have equal length, got "
            f"{len(filters)} and {len(kernel_sizes)}"
        )

    layers = tf.keras.layers
    curve_input = layers.Input(shape=(n_points, n_curve_channels), name="curve")
    env_input = layers.Input(shape=(n_env_features,), name="env")

    # Early fusion: broadcast the environmental scalars along the sequence.
    env_sequence = layers.RepeatVector(n_points)(env_input)
    x = layers.Concatenate(axis=-1)([curve_input, env_sequence])

    for n_filters, kernel in zip(filters, kernel_sizes, strict=True):
        x = layers.Conv1D(n_filters, kernel, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        # Strided pooling downsamples but, unlike global average pooling,
        # retains where along the curve each feature occurred.
        x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, env_input])  # late fusion as well
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units // 2, activation="relu")(x)
    # Initialise the output bias so softplus(b) == 1.0, matching the scaled
    # targets produced by Normalizer(label_scale="mean").
    output = layers.Dense(
        n_outputs,
        activation="softplus",
        bias_initializer=tf.keras.initializers.Constant(0.5413248546),
        name="speciation",
    )(x)

    return tf.keras.Model(
        inputs=[curve_input, env_input], outputs=output, name=name
    )


def build_denoiser_cnn(
    n_points: int = 200,
    n_curve_channels: int = 2,
    n_env_features: int = 2,
    *,
    filters: tuple[int, ...] = (32, 64, 128),
    kernel_size: int = 7,
    name: str = "titration_denoiser",
) -> tf.keras.Model:
    r"""Build a 1-D encoder--decoder that removes instrument artefacts.

    Parameters
    ----------
    n_points
        Points per curve.  Must be divisible by ``2 ** len(filters)``.
    n_curve_channels
        Channels in the input curve, normally ``2`` for ``(pH, titrant mass)``.
    n_env_features
        Number of environmental scalars, normally ``2`` for ``(S, t)``.
    filters
        Channel width at each encoder level.
    kernel_size
        Convolution width.  Wide kernels help: drift and slope error are
        low-frequency features.
    name
        Model name.

    Returns
    -------
    tf.keras.Model
        Inputs ``["curve", "env"]``, output ``(n_points,)`` denoised pH.

    Raises
    ------
    ImportError
        If TensorFlow is not installed.
    ValueError
        If ``n_points`` is not divisible by ``2 ** len(filters)``.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "TensorFlow is required for cleaned_reef_water.ml.architecture. "
            "Install it with `pip install 'cleaned-reef-water[ml]'`."
        ) from exc

    stride_total = 2 ** len(filters)
    if n_points % stride_total:
        raise ValueError(
            f"n_points ({n_points}) must be divisible by {stride_total} "
            f"for {len(filters)} down/up-sampling levels"
        )

    layers = tf.keras.layers
    curve_input = layers.Input(shape=(n_points, n_curve_channels), name="curve")
    env_input = layers.Input(shape=(n_env_features,), name="env")

    env_sequence = layers.RepeatVector(n_points)(env_input)
    x = layers.Concatenate(axis=-1)([curve_input, env_sequence])

    skips = []
    for width in filters:
        x = layers.Conv1D(width, kernel_size, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        skips.append(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters[-1], kernel_size, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for width, skip in zip(reversed(filters), reversed(skips), strict=True):
        x = layers.UpSampling1D(size=2)(x)
        x = layers.Concatenate(axis=-1)([x, skip])
        x = layers.Conv1D(width, kernel_size, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    # Predict a residual correction to the measured pH rather than the curve
    # itself.
    correction = layers.Conv1D(
        1,
        1,
        padding="same",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="correction",
    )(x)
    measured_ph = layers.Lambda(
        lambda t: t[:, :, :1], name="measured_ph"
    )(curve_input)
    output = layers.Add(name="denoised_ph")([measured_ph, correction])
    output = layers.Reshape((n_points,), name="denoised")(output)

    return tf.keras.Model(
        inputs=[curve_input, env_input], outputs=output, name=name
    )
