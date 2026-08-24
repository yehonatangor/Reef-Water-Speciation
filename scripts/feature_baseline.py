#!/usr/bin/env python3
r"""Reproduce the linear-regression baseline that motivated ``ml/features.py``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Part of the documented-claims tooling, so it must run from a bare checkout.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cleaned_reef_water.ml.dataset import (
    META_COLUMNS,
    Dataset,
    generate_dataset,
    split_dataset,
)
from cleaned_reef_water.ml.features import (
    FEATURE_NAMES,
    curve_features,
)
from cleaned_reef_water.ml.noise import NoiseModel

#: The figure recorded in ``ml/features.py`` for the trained convolutional
#: stack, for context.  Not recomputed here -- see the module docstring.
CNN_REFERENCE_SD = 58.9


def design_matrix(part: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Build the feature matrix and the alkalinity target for one split.

    Returns
    -------
    tuple
        ``(X, y)`` with a bias column appended to ``X`` and ``y`` in mol/kg.
    """
    mass = part["metadata"][:, META_COLUMNS.index("sample_mass_kg"), None]
    features = curve_features(part["curves"], mass)
    # Salinity and temperature are measured on a real instrument, so a fair
    # baseline gets them too.
    design = np.concatenate([features, part["env"][:, :2], mass], axis=1)
    design = np.concatenate([design, np.ones((len(design), 1))], axis=1)
    target = part["metadata"][:, META_COLUMNS.index("alkalinity")]
    return design, target


def fit_and_score(train, test) -> tuple[float, float, float]:
    """Fit least squares on ``train`` and score on ``test``.

    Returns
    -------
    tuple
        ``(sd, mae, bias)`` of the alkalinity error in umol/kg.
    """
    x_train, y_train = design_matrix(train)
    x_test, y_test = design_matrix(test)
    coefficients, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
    error = (x_test @ coefficients - y_test) * 1e6
    return float(np.std(error)), float(np.mean(np.abs(error))), float(np.mean(error))


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=None,
        help="existing .npz; if omitted, a small one is generated on the fly",
    )
    parser.add_argument("--n-samples", type=int, default=1200,
                        help="curves to generate when --dataset is not given")
    parser.add_argument(
        "--instrument", choices=("hobbyist", "laboratory"), default="hobbyist",
        help=(
            "noise preset when generating.  This matters enormously and the "
            "original claim did not state it: the same fit gives ~5 umol/kg "
            "on laboratory noise and ~28 on hobbyist"
        ),
    )
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Fit the linear baseline and report."""
    args = parse_args(argv)

    if args.dataset:
        print(f"loading {args.dataset} ...", flush=True)
        data, provenance = Dataset.from_npz(args.dataset)
        source = str(args.dataset)
    else:
        print(f"generating {args.n_samples} curves "
              f"({args.n_points} points, {args.instrument} noise, "
              f"seed {args.seed}) ...", flush=True)
        print("  pass --dataset data/dataset_hobbyist.npz to use the full set")
        data = generate_dataset(
            args.n_samples, seed=args.seed, n_points=args.n_points,
            noise_model=getattr(NoiseModel, args.instrument)(),
        )
        provenance = {"instrument": args.instrument}
        source = f"generated, n={args.n_samples}"

    train, _, test = split_dataset(data, seed=args.split_seed)
    sd, mae, bias = fit_and_score(train, test)

    print(f"\nsource        : {source}")
    if provenance:
        print(f"instrument    : {provenance.get('instrument', 'unknown')}")
    print(f"train / test  : {len(train['curves'])} / {len(test['curves'])}")
    print(f"features      : {len(FEATURE_NAMES)} curve statistics "
          f"+ S, t, sample mass + bias")

    print("\nLinear regression on the summary features, held-out split:")
    print(f"  alkalinity error sd    {sd:8.2f} umol/kg")
    print(f"  alkalinity error MAE   {mae:8.2f} umol/kg")
    print(f"  bias                   {bias:+8.2f} umol/kg")

    print(f"\nFor context, the trained convolutional stack recorded in "
          f"ml/features.py:\n  alkalinity error sd    "
          f"{CNN_REFERENCE_SD:8.2f} umol/kg  (NOT recomputed here -- "
          f"needs scripts/train_cnn.py)")
    if sd < CNN_REFERENCE_SD:
        print(f"\n  The linear fit is {CNN_REFERENCE_SD / sd:.2f}x better, "
              "which is the finding that\n  motivated supplying these "
              "statistics to the encoder directly.")
    else:
        print("\n  The linear fit does NOT beat the recorded CNN figure here. "
              "That may simply\n  mean this run used fewer curves; try "
              "--dataset with the full training set.")

    print("\nMost informative single features, by |t|-like ratio of "
          "coefficient to spread:")
    x_train, y_train = design_matrix(train)
    coefficients, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
    names = (*FEATURE_NAMES, "salinity", "t_c", "sample_mass", "bias")
    scale = x_train.std(axis=0)
    influence = np.abs(coefficients * scale) * 1e6
    for index in np.argsort(influence)[::-1][:5]:
        print(f"  {names[index]:<28}{influence[index]:9.2f} umol/kg per sd")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "source": source,
            "n_train": len(train["curves"]),
            "n_test": len(test["curves"]),
            "linear_sd_umol_per_kg": round(sd, 3),
            "linear_mae_umol_per_kg": round(mae, 3),
            "linear_bias_umol_per_kg": round(bias, 3),
            "cnn_reference_sd_umol_per_kg": CNN_REFERENCE_SD,
            "cnn_reference_recomputed": False,
        }, indent=2))
        print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
