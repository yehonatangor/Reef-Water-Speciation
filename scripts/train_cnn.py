#!/usr/bin/env python3
"""Train the speciation CNN on a pre-generated dataset and evaluate it."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cleaned_reef_water import __version__
from cleaned_reef_water.ml.dataset import Dataset, split_dataset
from cleaned_reef_water.ml.normalization import Normalizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="npz written by scripts/generate_dataset.py",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--enable-onednn",
        action="store_true",
        help=(
            "re-enable TensorFlow's oneDNN kernels; off by default because "
            "they crash intermittently on native Windows and make results "
            "non-reproducible"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the full training pipeline."""
    args = parse_args(argv)
    if not args.dataset.exists():
        print(
            f"dataset not found: {args.dataset}\n"
            "Generate it first:\n"
            "    python scripts/generate_dataset.py --instrument hobbyist",
            file=sys.stderr,
        )
        return 1

    print(f"[1/4] loading {args.dataset} ...")
    try:
        data, data_provenance = Dataset.from_npz(args.dataset)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    instrument = data_provenance.get("instrument", "unknown")
    n_points = int(data.curves.shape[1])
    print(f"      {data.curves.shape}, instrument={instrument}")

    if not args.enable_onednn:
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    try:
        import tensorflow as tf
    except ImportError:
        print(
            "TensorFlow is not installed. Install it with:\n"
            "    pip install 'cleaned-reef-water[ml]'",
            file=sys.stderr,
        )
        return 1

    from cleaned_reef_water.ml.architecture import OUTPUT_NAMES, build_speciation_cnn

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[2/4] splitting and normalising ...")
    train, val, test = split_dataset(data, seed=args.split_seed)
    norm = Normalizer.fit(train["curves"], train["env"], train["labels"])
    norm.save(args.output_dir / "normalizer.npz")

    def prepare(part: dict[str, np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
        return (
            [norm.transform_curves(part["curves"]), norm.transform_env(part["env"])],
            norm.transform_labels(part["labels"]),
        )

    x_train, y_train = prepare(train)
    x_val, y_val = prepare(val)
    x_test, y_test = prepare(test)

    print("[3/4] training ...")
    tf.random.set_seed(args.train_seed)
    model = build_speciation_cnn(
        n_points=n_points, n_curve_channels=int(data.curves.shape[2])
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate, clipnorm=1.0
        ),
        loss="mse",
        metrics=["mae"],
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
            ),
        ],
        verbose=2,
    )
    model.save(args.output_dir / "speciation_cnn.keras")

    print("[4/4] evaluating on the held-out test split ...")
    predicted = norm.inverse_labels(model.predict(x_test, verbose=0))
    truth = test["labels"]
    metrics = {}
    for index, name in enumerate(OUTPUT_NAMES):
        error = (predicted[:, index] - truth[:, index]) * 1e6
        metrics[name] = {
            "bias": round(float(np.mean(error)), 3),
            "sd": round(float(np.std(error)), 3),
            "mae": round(float(np.mean(np.abs(error))), 3),
            "rmse": round(float(np.sqrt(np.mean(error**2))), 3),
        }

    payload = {
        "provenance": {
            "package_version": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "tensorflow": tf.__version__,
            "onednn_enabled": bool(args.enable_onednn),
            "command": " ".join(sys.argv),
            "seeds": {"split": args.split_seed, "train": args.train_seed},
            "dataset": data_provenance,
        },
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "instrument": instrument,
        "n_train": int(len(train["curves"])),
        "n_test": int(len(test["curves"])),
        "epochs_run": len(history.history["loss"]),
        "test_metrics_umol_per_kg": metrics,
    }
    output = args.output_dir / f"cnn_{instrument}.json"
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf8")

    print(f"\n{'species':10s}{'bias':>10s}{'sd':>10s}{'MAE':>10s}{'RMSE':>10s}")
    for name, values in metrics.items():
        print(
            f"{name:10s}{values['bias']:+10.2f}{values['sd']:10.2f}"
            f"{values['mae']:10.2f}{values['rmse']:10.2f}"
        )
    print("\nunits: umol/kg")
    print(
        "Compare against results/baseline.json. The target is to MATCH the\n"
        "least-squares inverse, not beat it: that fit is already at or near the\n"
        "Cramer-Rao bound (docs/information_limits.md). A result well BELOW the\n"
        "bound -- 18.4 umol/kg hobbyist, 3.5 laboratory -- indicates a leak."
    )
    print(f"written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
