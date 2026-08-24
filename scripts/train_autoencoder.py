#!/usr/bin/env python3
r"""Train the physics-constrained autoencoder and evaluate it end to end.

Examples
--------
::

    python scripts/generate_dataset.py --instrument hobbyist
    python scripts/train_autoencoder.py --dataset data/dataset_hobbyist.npz
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

# Run from a source checkout as well as an installed package.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cleaned_reef_water import __version__, constants_at, speciate

# Safe at module scope: ``ml.autoencoder`` imports TensorFlow lazily, inside
# ``build_physics_autoencoder``, so naming it here costs nothing at import time.
from cleaned_reef_water.ml.autoencoder import LATENT_INDEX, LATENT_NAMES
from cleaned_reef_water.ml.dataset import (
    META_COLUMNS,
    NUISANCE_COLUMNS,
    Dataset,
    split_dataset,
)
from cleaned_reef_water.ml.features import curve_features
from cleaned_reef_water.ml.normalization import Normalizer
from cleaned_reef_water.seawater import total_borate

SPECIES = ("hco3", "co3", "boh4")

#: Characteristic scale of each latent, used to make the supervised loss
#: dimensionless.
_LATENT_SCALE_BY_NAME: dict[str, float] = {
    "alkalinity": 1e-4,
    "alk_minus_dic": 2e-5,
    "total_organic": 2e-5,
    "pk_organic": 0.5,
    "offset": 0.05,
    "slope": 0.03,
    "pump_scale": 0.008,
}

#: Ordered to match :data:`LATENT_NAMES`, by construction rather than by hand.
LATENT_SCALES = np.array(
    [_LATENT_SCALE_BY_NAME[name] for name in LATENT_NAMES], dtype=np.float64
)

#: Latents carrying ``--weight-totals`` in the supervised loss.
TOTALS_LATENTS: tuple[str, ...] = ("alkalinity", "alk_minus_dic")

#: Characteristic pH residual, for scaling the reconstruction term.
PH_SCALE = 0.02

#: Characteristic species error, for scaling the species term.
SPECIES_SCALE = 2e-5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument(
        "--weight-reconstruction",
        type=float,
        default=1.0,
        help="weight on the curve reconstruction term",
    )
    parser.add_argument(
        "--weight-totals",
        type=float,
        default=1.0,
        help="weight on supervised (A_T, C_T)",
    )
    parser.add_argument(
        "--weight-nuisance",
        type=float,
        default=1.0,
        help="weight on supervised (offset, slope, pump_scale); setting this "
        "to 0 reproduces the degenerate reconstruction-only failure",
    )
    parser.add_argument(
        "--weight-species",
        type=float,
        default=1.0,
        help="weight on supervised (HCO3, CO3, B(OH)4).  Species depend on "
        "(A_T, C_T) jointly, so a loss on the two totals separately leaves "
        "their error correlation unconstrained; this term supplies it",
    )
    parser.add_argument(
        "--solver-iterations",
        type=int,
        default=40,
        help="bisection iterations in the decoder; 40 resolves pH to ~1e-11 "
        "and is cheaper than the default 60",
    )
    parser.add_argument(
        "--use-true-nutrients",
        action="store_true",
        help="DIAGNOSTIC ONLY: feed the generator's phosphate and silicate to "
        "the decoder.  These are unknown on a real sample, so a result "
        "obtained with this flag is not comparable to the baseline",
    )
    parser.add_argument("--n-eval", type=int, default=0,
                        help="curves to evaluate; 0 means the whole test split")
    parser.add_argument(
        "--n-timed",
        type=int,
        default=10,
        help="curves used to time the classical inverse for the speed "
        "comparison; kept small because it calls scipy in-process",
    )
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="skip the classical-inverse speed benchmark entirely",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--enable-onednn", action="store_true")
    return parser.parse_args(argv)


def encoder_env(part: dict[str, np.ndarray]) -> np.ndarray:
    """Environmental features the *encoder* sees: ``(S, T, sample_mass)``."""
    mass = part["metadata"][:, META_COLUMNS.index("sample_mass_kg"), None]
    return np.concatenate(
        [part["env"], mass, curve_features(part["curves"], mass)], axis=1
    )


def build_inputs(
    part: dict[str, np.ndarray],
    normalizer: Normalizer,
    use_true_nutrients: bool,
) -> dict[str, np.ndarray]:
    """Assemble model inputs from a dataset split."""
    from cleaned_reef_water.ml.differentiable import pack_constants

    meta = part["metadata"]
    salinity_measured = part["env"][:, 0]
    boron = np.asarray(
        [float(total_borate(s)) for s in salinity_measured], dtype=np.float64
    ).reshape(-1, 1)
    if use_true_nutrients:
        nutrients = meta[:, 6:8]
    else:
        nutrients = np.zeros((meta.shape[0], 2), dtype=np.float64)

    # Encoder inputs feed convolutions, not chemistry, so float32 is enough.
    return {
        "curve": normalizer.transform_curves(part["curves"]).astype(np.float32),
        "env": normalizer.transform_env(encoder_env(part)).astype(np.float32),
        "constants": pack_constants(salinity_measured, part["env"][:, 1]),
        "titrant_mass": part["curves"][:, :, 1],
        "sample_mass": meta[:, 4:5],
        "totals": np.concatenate([boron, nutrients], axis=1),
    }


def build_targets(part: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Assemble the reconstruction and latent targets."""
    meta = part["metadata"]

    def column(name: str) -> np.ndarray:
        index = META_COLUMNS.index(name)
        return meta[:, index : index + 1]

    latents = np.concatenate(
        [
            column("alkalinity"),
            column("alkalinity") - column("dic"),  # A_T - C_T, not C_T
            column("total_organic"),
            column("pk_organic"),
            part["nuisance"],  # offset, slope, pump_scale
        ],
        axis=1,
    )
    expected = len(LATENT_NAMES)
    if latents.shape[1] != expected:
        raise ValueError(
            f"built {latents.shape[1]} latent targets but LATENT_NAMES has "
            f"{expected}: {LATENT_NAMES}"
        )
    return {
        "reconstruction": part["curves"][:, :, 0],
        "latents": latents,
        "species": part["labels"],
    }


def main(argv: list[str] | None = None) -> int:
    """Train the autoencoder and report end-to-end accuracy."""
    args = parse_args(argv)
    if not args.dataset.exists():
        print(f"dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    print(f"[1/5] loading {args.dataset} ...", flush=True)
    try:
        data, data_provenance = Dataset.from_npz(args.dataset)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    instrument = data_provenance.get("instrument", "unknown")
    n_points = int(data.curves.shape[1])
    print(f"      {data.curves.shape}, instrument={instrument}", flush=True)
    if args.use_true_nutrients:
        print(
            "      WARNING: --use-true-nutrients is set. The decoder is being "
            "given information a real sample does not provide; this result is "
            "not comparable to results/baseline.json.",
            flush=True,
        )

    if not args.enable_onednn:
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow is not installed.", file=sys.stderr)
        return 1
    from cleaned_reef_water.ml.autoencoder import (
        build_physics_autoencoder,
        decode_latents,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[2/5] splitting and normalising ...", flush=True)
    train, val, test = split_dataset(data, seed=args.split_seed)
    normalizer = Normalizer.fit(
        train["curves"], encoder_env(train), train["labels"]
    )
    normalizer.save(args.output_dir / "autoencoder_normalizer.npz")

    inputs = {
        name: build_inputs(part, normalizer, args.use_true_nutrients)
        for name, part in (("train", train), ("val", val), ("test", test))
    }
    targets = {
        name: build_targets(part)
        for name, part in (("train", train), ("val", val), ("test", test))
    }

    print("[3/5] training ...", flush=True)
    print(
        f"      loss scales: pH {PH_SCALE}, latents "
        f"{dict(zip(LATENT_NAMES, LATENT_SCALES, strict=True))}",
        flush=True,
    )
    tf.random.set_seed(args.train_seed)
    model = build_physics_autoencoder(
        n_points=n_points,
        n_curve_channels=int(data.curves.shape[2]),
        n_env_features=encoder_env(train).shape[1],
        n_solver_iterations=args.solver_iterations,
    )

    scales = tf.constant(LATENT_SCALES, dtype=tf.float64)
    weight_totals = tf.constant(args.weight_totals, dtype=tf.float64)
    weight_nuisance = tf.constant(args.weight_nuisance, dtype=tf.float64)

    # This split is NOT the same as N_COMPOSITION_LATENTS, and the difference
    # is deliberate.
    totals_mask = tf.constant(
        [[name in TOTALS_LATENTS for name in LATENT_NAMES]], dtype=tf.float64
    )
    n_totals = float(len(TOTALS_LATENTS))
    n_other = float(len(LATENT_NAMES) - len(TOTALS_LATENTS))

    def latent_loss(y_true: object, y_pred: object) -> object:
        """Scaled supervised error, weighted separately for totals/nuisance."""
        z = (tf.cast(y_pred, tf.float64) - tf.cast(y_true, tf.float64)) / scales
        squared = tf.square(z)
        totals = tf.reduce_sum(squared * totals_mask, axis=-1) / n_totals
        nuisance = tf.reduce_sum(squared * (1.0 - totals_mask), axis=-1) / n_other
        return weight_totals * totals + weight_nuisance * nuisance

    def species_loss(y_true: object, y_pred: object) -> object:
        """Scaled supervised error on the species themselves."""
        z = (
            tf.cast(y_pred, tf.float64) - tf.cast(y_true, tf.float64)
        ) / SPECIES_SCALE
        return tf.reduce_mean(tf.square(z), axis=-1)

    def reconstruction_loss(y_true: object, y_pred: object) -> object:
        """Mean squared pH error, scaled to a characteristic residual."""
        z = (tf.cast(y_pred, tf.float64) - tf.cast(y_true, tf.float64)) / PH_SCALE
        return tf.reduce_mean(tf.square(z), axis=-1)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate, clipnorm=1.0
        ),
        loss={
            "reconstruction": reconstruction_loss,
            "latents": latent_loss,
            "species": species_loss,
        },
        loss_weights={
            "reconstruction": args.weight_reconstruction,
            "latents": 1.0,
            "species": args.weight_species,
        },
    )
    history = model.fit(
        inputs["train"],
        targets["train"],
        validation_data=(inputs["val"], targets["val"]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            # Monitor the LATENT loss, not the total.
            tf.keras.callbacks.EarlyStopping(
                monitor="val_latents_loss",
                mode="min",
                patience=args.patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_latents_loss", mode="min", factor=0.5,
                patience=8, min_lr=1e-6
            ),
            # The OS can kill training mid-run, silently and without a traceback.
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(args.output_dir / "autoencoder_best.keras"),
                monitor="val_latents_loss",
                mode="min",
                save_best_only=True,
                verbose=0,
            ),
            # Append the per-epoch history so it survives a kill too.
            tf.keras.callbacks.CSVLogger(
                str(args.output_dir / "training_history.csv"), append=True
            ),
        ],
        verbose=2,
    )
    model.save(args.output_dir / "autoencoder.keras")

    print("[4/5] predicting on the test split ...", flush=True)
    # Amortised inference speed is the point of this model, so it is measured
    # rather than asserted.
    model.predict(
        {k: v[:8] for k, v in inputs["test"].items()}, verbose=0
    )  # warm up the graph so compilation is not counted
    started = time.perf_counter()
    predicted = model.predict(inputs["test"], verbose=0)
    network_seconds = time.perf_counter() - started
    n_test = len(test["curves"])
    print(
        f"      {1e3 * network_seconds / n_test:.3f} ms/curve "
        f"({n_test} curves in {network_seconds:.2f}s)",
        flush=True,
    )
    latents = decode_latents(predicted["latents"])
    truth = targets["test"]["latents"]

    n_eval = args.n_eval or len(test["curves"])
    n_eval = min(n_eval, len(test["curves"]))

    print(f"[5/5] speciating {n_eval} predictions exactly ...", flush=True)
    # Vectorised rather than a Python loop over ``ph_from_dic_alkalinity``.
    from cleaned_reef_water.ml.differentiable import pack_constants, solve_ph

    alkalinity = latents["alkalinity"][:n_eval, None]
    dic = alkalinity - latents["alk_minus_dic"][:n_eval, None]
    salinity = test["env"][:n_eval, 0]
    t_c = test["env"][:n_eval, 1]
    boron = np.asarray(
        [float(total_borate(s)) for s in salinity], dtype=np.float64
    ).reshape(-1, 1)

    # The organic latents MUST be carried into this solve.
    organic = latents["total_organic"][:n_eval, None]
    pk_organic = latents["pk_organic"][:n_eval, None]

    ph = np.asarray(
        solve_ph(
            tf.constant(alkalinity),
            tf.constant(pack_constants(salinity, t_c)),
            tf.constant(dic),
            tf.constant(boron),
            total_organic=tf.constant(organic),
            pk_organic=tf.constant(pk_organic),
        )
    ).ravel()

    # speciate is pure algebra and broadcasts, so this is one vectorised call.
    species = speciate(
        ph,
        constants_at(salinity, t_c),
        dic=dic.ravel(),
        total_boron=boron.ravel(),
        total_organic=organic.ravel(),
        pk_organic=pk_organic.ravel(),
    )

    finite = np.isfinite(ph)
    n_failed_solve = int((~finite).sum())
    if n_failed_solve:
        print(f"      {n_failed_solve} solves did not converge", flush=True)

    errors: dict[str, list[float]] = {}
    errors["alkalinity"] = (
        (alkalinity.ravel() - truth[:n_eval, 0]) * 1e6
    ).tolist()
    # truth column 1 is A_T - C_T, so recover true DIC the same way.
    truth_dic = truth[:n_eval, 0] - truth[:n_eval, 1]
    errors["dic"] = ((dic.ravel() - truth_dic) * 1e6).tolist()
    for j, name in enumerate(SPECIES):
        residual = (
            np.asarray(getattr(species, name)).ravel() - test["labels"][:n_eval, j]
        ) * 1e6
        errors[name] = residual[finite].tolist()

    metrics = {}
    for name, values in errors.items():
        array = np.asarray(values)
        metrics[name] = {
            "bias": round(float(np.mean(array)), 3),
            "sd": round(float(np.std(array)), 3),
            "mae": round(float(np.mean(np.abs(array))), 3),
            "median_abs": round(float(np.median(np.abs(array))), 3),
            "max_abs": round(float(np.max(np.abs(array))), 3),
        }

    # Joint error structure of the two totals.
    d_alk = np.asarray(errors["alkalinity"])
    d_dic = np.asarray(errors["dic"])
    joint = {
        "corr_dA_dC": round(float(np.corrcoef(d_alk, d_dic)[0, 1]), 4),
        "sd_dA_minus_dC": round(float(np.std(d_alk - d_dic)), 3),
        "classical_reference": {
            "corr_dA_dC": 0.933,
            "sd_dA_minus_dC": 11.46,
            "measured_on": "organic-free data (pre-organic dataset)",
            "comparable": False,
        },
    }
    organic_fraction = float(
        (data.metadata[:, META_COLUMNS.index("total_organic")] > 1e-12).mean()
    )
    print(
        f"\njoint error structure: corr(dA,dC) {joint['corr_dA_dC']:+.3f}  "
        f"sd(dA-dC) {joint['sd_dA_minus_dC']:.2f}"
    )
    if organic_fraction > 0.0:
        # The stored reference was measured on organic-free curves.
        print(
            f"      NOT comparable to the stored classical reference "
            f"(+0.933, 11.46):\n      that was measured on organic-free "
            f"curves and {100 * organic_fraction:.0f}% of these carry "
            f"organic.\n      Run scripts/compare_organic.py for a "
            f"like-for-like number."
        )
    else:
        print("      (classical on the same problem: +0.933, 11.46)")

    # Recovery of every latent that is not one of the two carbonate totals: the
    # instrument nuisances, and now the two organic latents.
    nuisance_recovery = {}
    for name in (*NUISANCE_COLUMNS, "total_organic", "pk_organic"):
        true_column = truth[:n_eval, LATENT_INDEX[name]]
        residual = latents[name][:n_eval] - true_column
        nuisance_recovery[name] = {
            "bias": round(float(np.mean(residual)), 6),
            "sd": round(float(np.std(residual)), 6),
            "true_sd": round(float(np.std(true_column)), 6),
            # >1 carries information beyond the prior; ~1 is the population mean.
            "recovery_ratio": round(
                float(np.std(true_column) / max(np.std(residual), 1e-12)), 3
            ),
        }

    payload = {
        "provenance": {
            "package_version": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "tensorflow": tf.__version__,
            "command": " ".join(sys.argv),
            "seeds": {"split": args.split_seed, "train": args.train_seed},
            "dataset": data_provenance,
        },
        "mode": "physics_constrained_autoencoder",
        "instrument": instrument,
        "used_true_nutrients": args.use_true_nutrients,
        "loss_weights": {
            "reconstruction": args.weight_reconstruction,
            "totals": args.weight_totals,
            "nuisance": args.weight_nuisance,
            "species": args.weight_species,
        },
        "loss_scales": {
            "ph": PH_SCALE,
            "species": SPECIES_SCALE,
            **dict(zip(LATENT_NAMES, LATENT_SCALES.tolist(), strict=True)),
        },
        "epochs_run": len(history.history["loss"]),
        "inference": {
            "network_ms_per_curve": round(1e3 * network_seconds / n_test, 4),
        },
        "n_evaluated": len(errors["alkalinity"]),
        "metrics_umol_per_kg": metrics,
        "nuisance_recovery": nuisance_recovery,
        "joint_error_structure": joint,
        "metadata_columns": list(META_COLUMNS),
    }
    # Written BEFORE the classical timing below, which calls into scipy and has
    # crashed the process once already.
    output = args.output_dir / f"autoencoder_{instrument}.json"
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf8")

    print(
        f"\n{'quantity':12s}{'bias':>10s}{'sd':>10s}{'MAE':>10s}"
        f"{'median':>10s}{'max':>10s}"
    )
    for name in ("alkalinity", "dic", *SPECIES):
        v = metrics[name]
        print(
            f"{name:12s}{v['bias']:+10.2f}{v['sd']:10.2f}{v['mae']:10.2f}"
            f"{v['median_abs']:10.2f}{v['max_abs']:10.2f}"
        )
    print("\nunits: umol/kg   |   compare against results/baseline.json")

    print(
        f"\n{'latent':14s}{'bias':>12s}{'residual sd':>14s}"
        f"{'true sd':>12s}{'recovery':>10s}"
    )
    for name in nuisance_recovery:
        v = nuisance_recovery[name]
        print(
            f"{name:14s}{v['bias']:+12.5f}{v['sd']:14.5f}"
            f"{v['true_sd']:12.5f}{v['recovery_ratio']:10.2f}"
        )
    print(
        "\nrecovery = true sd / residual sd.  Above 1 means the latent was "
        "actually\nrecovered from the curve; ~1 means the encoder is "
        "predicting the population\nmean and the curve does not identify it."
        "\n\npk_organic is expected near 1.0: it has no effect at all when "
        "total_organic\nis zero, which is about half the training set.  See "
        "docs/machine_learning.md 6a."
    )
    print(f"written to {output}")

    if args.skip_timing:
        return 0

    # Optional and last, because it calls scipy in-process with TensorFlow.
    # Everything above is already on disk.
    from cleaned_reef_water.baseline import LeastSquaresInverse, NuisanceModel

    n_timed = min(args.n_timed, n_eval)
    classical = LeastSquaresInverse(nuisance=NuisanceModel.OFFSET_AND_SLOPE)
    started = time.perf_counter()
    for index in range(n_timed):
        classical.fit(
            test["curves"][index, :, 1],
            test["curves"][index, :, 0],
            float(test["env"][index, 0]),
            float(test["env"][index, 1]),
        )
    classical_seconds = (time.perf_counter() - started) / n_timed
    network_each = network_seconds / n_test
    speedup = classical_seconds / network_each
    print(
        f"\nspeed: network {network_each * 1e3:.3f} ms/curve  vs  classical "
        f"{classical_seconds * 1e3:.1f} ms/curve  = {speedup:.0f}x faster"
    )
    payload["inference"].update(
        {
            "classical_ms_per_curve": round(classical_seconds * 1e3, 2),
            "speedup": round(speedup, 1),
            "n_timed": n_timed,
        }
    )
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
