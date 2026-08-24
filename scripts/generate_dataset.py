#!/usr/bin/env python3
r"""Generate a synthetic titration dataset, resumably, and write it to ``.npz``.

Examples
--------
Single-process run (fine where the crash does not occur)::

    python scripts/generate_dataset.py --instrument hobbyist

Smoke test::

    python scripts/generate_dataset.py --n-samples 200 --output data/smoke.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

# Run from a source checkout as well as an installed package.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cleaned_reef_water import __version__
from cleaned_reef_water.ml.dataset import generate_dataset
from cleaned_reef_water.ml.noise import NoiseModel
from cleaned_reef_water.ml.sampling import SamplingRanges

INSTRUMENTS = {
    "laboratory": NoiseModel.laboratory,
    "hobbyist": NoiseModel.hobbyist,
}

_ARRAY_KEYS = ("curves", "env", "labels", "metadata", "clean_ph", "nuisance")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-samples", type=int, default=16000)
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--instrument", choices=list(INSTRUMENTS), default="hobbyist"
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="generate clean curves, to isolate model error from measurement error",
    )
    parser.add_argument(
        "--organic-probability",
        type=float,
        default=SamplingRanges().organic_probability,
        help=(
            "fraction of samples containing organic alkalinity. "
            "Pass 0.0 to generate the organic-free CONTROL set: the trained "
            "model must still work on water that has none, and that cannot be "
            "checked against the pre-organic .v1 datasets because those have "
            "8 metadata columns and are rejected by the loader"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="curves per chunk file",
    )
    parser.add_argument(
        "--max-chunks-per-run",
        type=int,
        default=0,
        help=(
            "stop after this many chunks and exit with code 2 so a shell loop "
            "can restart in a fresh process; 0 means do them all"
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="print a progress line every N curves (0 to disable)",
    )
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        help="do not delete the per-chunk files after merging",
    )
    return parser.parse_args(argv)


def part_path(output: Path, index: int) -> Path:
    """Return the filename for chunk ``index``."""
    return output.with_name(f"{output.stem}.part{index:03d}.npz")


def main(argv: list[str] | None = None) -> int:
    """Generate any missing chunks, merging when the set is complete."""
    args = parse_args(argv)
    output = args.output or Path(f"data/dataset_{args.instrument}.npz")
    output.parent.mkdir(parents=True, exist_ok=True)

    chunk = max(1, min(args.chunk_size, args.n_samples))
    n_chunks = -(-args.n_samples // chunk)  # ceiling division
    sizes = [
        min(chunk, args.n_samples - i * chunk) for i in range(n_chunks)
    ]

    missing = [i for i in range(n_chunks) if not part_path(output, i).exists()]
    if missing:
        print(
            f"{n_chunks - len(missing)}/{n_chunks} chunks already present; "
            f"{len(missing)} to go",
            flush=True,
        )

    budget = args.max_chunks_per_run or len(missing)
    for index in missing[:budget]:
        size = sizes[index]
        print(
            f"chunk {index + 1}/{n_chunks}: generating {size} curves "
            f"({args.instrument} noise, {args.n_points} points) ...",
            flush=True,
        )
        started = time.perf_counter()
        part = generate_dataset(
            size,
            seed=args.seed + index * chunk,
            n_points=args.n_points,
            ranges=dataclasses.replace(
                SamplingRanges(), organic_probability=args.organic_probability
            ),
            noise_model=INSTRUMENTS[args.instrument](),
            add_noise=not args.no_noise,
            progress_every=args.progress_every,
        )
        np.savez_compressed(
            part_path(output, index),
            curves=part.curves,
            env=part.env,
            labels=part.labels,
            metadata=part.metadata,
            clean_ph=part.clean_ph,
            nuisance=part.nuisance,
            n_failed=np.asarray(part.n_failed),
        )
        print(
            f"  wrote {part_path(output, index).name} "
            f"({time.perf_counter() - started:.0f}s)",
            flush=True,
        )

    remaining = [i for i in range(n_chunks) if not part_path(output, i).exists()]
    if remaining:
        print(
            f"\n{n_chunks - len(remaining)}/{n_chunks} chunks done, "
            f"{len(remaining)} remaining. Run again to continue.",
            flush=True,
        )
        return 2

    print("\nall chunks present; merging ...", flush=True)
    merged: dict[str, list[np.ndarray]] = {key: [] for key in _ARRAY_KEYS}
    n_failed = 0
    for index in range(n_chunks):
        with np.load(part_path(output, index)) as stored:
            for key in _ARRAY_KEYS:
                merged[key].append(stored[key])
            n_failed += int(stored["n_failed"])

    arrays = {key: np.concatenate(values) for key, values in merged.items()}
    provenance = {
        "package_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "instrument": args.instrument,
        "seed": args.seed,
        "n_samples": int(arrays["curves"].shape[0]),
        "n_points": args.n_points,
        "add_noise": not args.no_noise,
        "organic_probability": args.organic_probability,
        "n_failed": n_failed,
        "chunk_size": chunk,
        "n_chunks": n_chunks,
        "command": " ".join(sys.argv),
    }
    np.savez_compressed(output, provenance=json.dumps(provenance), **arrays)

    if not args.keep_parts:
        for index in range(n_chunks):
            part_path(output, index).unlink()

    size_mb = output.stat().st_size / 1e6
    print(f"  {arrays['curves'].shape}, {n_failed} discarded, {size_mb:.1f} MB")
    print(f"written to {output}")
    print(f"\nnow run:  python scripts/train_cnn.py --dataset {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
