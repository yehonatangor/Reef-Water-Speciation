#!/usr/bin/env python3
r"""Measure whether carrying an organic term actually buys anything."""


from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Run from a source checkout as well as an installed package.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cleaned_reef_water import __version__, constants_at, speciate
from cleaned_reef_water.baseline import LeastSquaresInverse, NuisanceModel
from cleaned_reef_water.ml.autoencoder import (
    LATENT_NAMES,
    register_serializable_layers,
)
from cleaned_reef_water.ml.dataset import (
    META_COLUMNS,
    Dataset,
    split_dataset,
)
from cleaned_reef_water.ml.normalization import Normalizer
from cleaned_reef_water.seawater import total_borate

SPECIES = ("hco3", "co3", "boh4")

#: Attempts allowed before giving up on the chunked classical phase.
MAX_CLASSICAL_ATTEMPTS = 400

#: Pause between attempts.
CLASSICAL_RETRY_SECONDS = 3.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("results/autoencoder_best.keras"),
        help="trained model; the best checkpoint, not the last epoch",
    )
    parser.add_argument(
        "--normalizer",
        type=Path,
        default=Path("results/autoencoder_normalizer.npz"),
        help="normalizer saved alongside the model; must be the same run",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/dataset_hobbyist.npz"),
        help="the organic-containing dataset the model was TRAINED on",
    )
    parser.add_argument(
        "--control",
        type=Path,
        default=Path("data/dataset_hobbyist_control.npz"),
        help="an organic-free dataset, generated with --organic-probability 0",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help=(
            "MUST match the training run, or the 'held-out' test split "
            "contains curves the model was fitted on"
        ),
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=600,
        help=(
            "curves per group.  Each classical fit is an optimisation over a "
            "forward model, so this dominates the runtime: ~0.2 s/curve"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/organic_comparison.json"),
    )
    parser.add_argument(
        "--only",
        choices=(
            "control_organic_free",
            "mixed_organic_present",
            "mixed_organic_only",
        ),
        default=None,
        help=(
            "run a single group and stop.  Results accumulate in --output, so "
            "running each group in its own process survives anything that "
            "kills the interpreter"
        ),
    )
    parser.add_argument(
        "--phase",
        choices=("all", "classical", "network"),
        default="all",
        help=(
            "'classical' fits the curves with the reference method and "
            "imports no TensorFlow; 'network' runs the encoder and reads "
            "those cached fits back.  They MUST stay in separate processes: "
            "scipy.optimize.least_squares and TensorFlow together segfault on "
            "Windows.  'all' spawns the classical phase as a subprocess"
        ),
    )
    parser.add_argument(
        "--classical-chunk-size",
        type=int,
        default=100,
        help=(
            "curves per classical process.  The classical inverse crashes the "
            "interpreter intermittently on Windows, so each chunk is fitted "
            "in a fresh process and written before the next begins.  Results "
            "are unchanged: curves are identified by absolute index, so the "
            "merge does not depend on how the work was divided.  0 disables "
            "chunking"
        ),
    )
    parser.add_argument(
        "--max-chunks-per-run",
        type=int,
        default=0,
        help="stop after this many chunks and exit 2; 0 means as many as fit",
    )
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        help="keep the per-chunk files after merging",
    )
    parser.add_argument("--enable-onednn", action="store_true")
    return parser.parse_args(argv)


def load_completed_groups(output: Path) -> dict[str, object]:
    """Read back any groups a previous run already finished.

    Returns
    -------
    dict
        Empty if the file is absent or unreadable.  A truncated JSON file --
        the likely result of a hard power-off mid-write -- is treated as no
        results rather than raising, because the alternative is a script that
        cannot be restarted without manual cleanup.
    """
    if not output.exists():
        return {}
    try:
        payload = json.loads(output.read_text())
    except (json.JSONDecodeError, OSError):
        print(f"{output} is unreadable; starting over", file=sys.stderr)
        return {}
    groups = payload.get("groups", {})
    return groups if isinstance(groups, dict) else {}


def write_payload(
    args: argparse.Namespace,
    provenance: dict[str, object],
    control_provenance: dict[str, object],
    groups: dict[str, object],
) -> None:
    """Write results to disk, including partial ones."""
    import tensorflow as tf

    payload = {
        "provenance": {
            "package_version": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "tensorflow": tf.__version__,
            "command": " ".join(sys.argv),
            "model": str(args.model),
            "split_seed": args.split_seed,
            "n_eval": args.n_eval,
            "dataset": provenance,
            "control": control_provenance,
            "latent_names": list(LATENT_NAMES),
        },
        "groups": groups,
    }
    # Write to a temporary file and replace, so that a crash during the write
    # cannot leave a half-finished file.
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(args.output)


def organic_fraction(data: Dataset) -> float:
    """Fraction of curves that actually contain organic alkalinity."""
    return float(
        (data.metadata[:, META_COLUMNS.index("total_organic")] > 1e-12).mean()
    )


def summarise(errors: np.ndarray, organic: np.ndarray) -> dict[str, float]:
    """Bias, scatter and the dependence of the error on organic content.

    Parameters
    ----------
    errors
        Signed errors in umol/kg.
    organic
        Organic content of the same curves in umol/kg.

    Returns
    -------
    dict
        ``bias`` is the headline: misspecification is systematic, so it shows
        up here rather than in ``sd``.  ``corr_with_organic`` and ``slope``
        separate "this water is harder" from "this estimator is missing a
        term"; they are ``None`` when the organic content does not vary, which
        is the case for the control group by construction.
    """
    out: dict[str, float] = {
        "n": int(errors.size),
        "bias": round(float(np.mean(errors)), 3),
        "sd": round(float(np.std(errors)), 3),
        "mae": round(float(np.mean(np.abs(errors))), 3),
        "max_abs": round(float(np.max(np.abs(errors))), 3),
    }
    if errors.size >= 3 and float(np.std(organic)) > 1e-9:
        out["corr_with_organic"] = round(
            float(np.corrcoef(organic, errors)[0, 1]), 4
        )
        out["slope_per_umol_organic"] = round(
            float(np.polyfit(organic, errors, 1)[0]), 4
        )
    return out


def network_species(
    model: object,
    normalizer: Normalizer,
    part: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Run the encoder and speciate its latents exactly.

    Returns
    -------
    tuple
        ``(species, alkalinity, seconds)``.  Speciation goes through the
        validated NumPy chemistry rather than the in-graph approximation, so
        the network and the classical inverse are scored by the same code.
    """
    import tensorflow as tf

    from cleaned_reef_water.ml.autoencoder import decode_latents
    from cleaned_reef_water.ml.differentiable import pack_constants, solve_ph
    from cleaned_reef_water.ml.features import curve_features

    mass = part["metadata"][:, META_COLUMNS.index("sample_mass_kg"), None]
    env = np.concatenate(
        [part["env"], mass, curve_features(part["curves"], mass)], axis=1
    )
    salinity, t_c = part["env"][:, 0], part["env"][:, 1]
    boron = np.asarray(
        [float(total_borate(s)) for s in salinity], dtype=np.float64
    ).reshape(-1, 1)

    inputs = {
        "curve": normalizer.transform_curves(part["curves"]).astype(np.float32),
        "env": normalizer.transform_env(env).astype(np.float32),
        "constants": pack_constants(salinity, t_c),
        "titrant_mass": part["curves"][:, :, 1],
        "sample_mass": mass,
        "totals": np.concatenate(
            [boron, np.zeros((len(boron), 2), dtype=np.float64)], axis=1
        ),
    }

    model.predict({k: v[:8] for k, v in inputs.items()}, verbose=0)  # warm up
    started = time.perf_counter()
    predicted = model.predict(inputs, verbose=0)
    seconds = time.perf_counter() - started

    latents = decode_latents(predicted["latents"])
    alkalinity = latents["alkalinity"][:, None]
    dic = alkalinity - latents["alk_minus_dic"][:, None]
    organic = latents["total_organic"][:, None]
    pk_organic = latents["pk_organic"][:, None]

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

    species = speciate(
        ph,
        constants_at(salinity, t_c),
        dic=dic.ravel(),
        total_boron=boron.ravel(),
        total_organic=organic.ravel(),
        pk_organic=pk_organic.ravel(),
    )
    return (
        {name: np.asarray(getattr(species, name)).ravel() for name in SPECIES},
        alkalinity.ravel(),
        seconds,
        organic.ravel(),
    )


def classical_species(
    part: dict[str, np.ndarray],
    start: int = 0,
    count: int | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Fit curves ``start`` to ``start + count`` with the classical inverse.

    Returns
    -------
    tuple
        ``(species, alkalinity, seconds)`` for the requested slice, with
        non-converged fits left as ``nan`` for the caller to mask rather than
        silently dropped.
    """
    mass_column = META_COLUMNS.index("sample_mass_kg")
    total = len(part["curves"])
    stop = total if count is None else min(start + count, total)
    n = stop - start
    out = {name: np.full(n, np.nan) for name in SPECIES}
    alkalinity = np.full(n, np.nan)

    started = time.perf_counter()
    for offset in range(n):
        i = start + offset
        result = LeastSquaresInverse(
            sample_mass_kg=float(part["metadata"][i, mass_column]),
            nuisance=NuisanceModel.OFFSET_AND_SLOPE,
        ).fit(
            part["curves"][i, :, 1],
            part["curves"][i, :, 0],
            float(part["env"][i, 0]),
            float(part["env"][i, 1]),
        )
        for name in SPECIES:
            out[name][offset] = getattr(result, name)
        alkalinity[offset] = result.alkalinity
        if (offset + 1) % 50 == 0:
            print(f"      classical {offset + 1}/{n}", flush=True)
    return out, alkalinity, time.perf_counter() - started


def cache_path(args: argparse.Namespace, key: str) -> Path:
    """Where the merged classical fits for one group are stored."""
    return args.output.parent / "organic_classical" / f"{key}.npz"


def chunk_path(args: argparse.Namespace, key: str, index: int) -> Path:
    """Where one chunk of one group's classical fits is stored.

    The classical inverse crashes the interpreter intermittently on Windows,
    so each chunk is fitted in a fresh process and written before the next
    begins.  A crash then costs one chunk rather than the run.  Curves are
    identified by absolute index, so the merged result does not depend on how
    the work was divided.
    """
    return (
        args.output.parent / "organic_classical" / f"{key}.part{index:03d}.npz"
    )


def merge_chunks(
    args: argparse.Namespace, key: str, n: int, chunk: int
) -> bool:
    """Concatenate a group's chunks into its merged cache.

    Returns
    -------
    bool
        ``True`` when every chunk was present and the merge was written.
    """
    n_chunks = -(-n // chunk)
    paths = [chunk_path(args, key, i) for i in range(n_chunks)]
    if not all(p.exists() for p in paths):
        return False

    # numpy's loader is lazy; Windows will not delete a file it holds open.
    loaded: list[dict[str, np.ndarray]] = []
    for p in paths:
        with np.load(p) as handle:
            loaded.append({key: np.array(handle[key]) for key in handle.files})

    merged = {
        name: np.concatenate([p[name] for p in loaded]) for name in SPECIES
    }
    alkalinity = np.concatenate([p["alkalinity"] for p in loaded])
    seconds = float(sum(float(p["seconds"]) for p in loaded))

    path = cache_path(args, key)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        alkalinity=alkalinity,
        seconds=np.asarray(seconds),
        **merged,
    )
    temporary.replace(path)
    if not args.keep_parts:
        for p in paths:
            # A leftover chunk is harmless: the merged cache is what downstream reads.
            with contextlib.suppress(OSError):
                p.unlink(missing_ok=True)
    return True


def run_classical_phase(args: argparse.Namespace) -> int:
    """Fit every group with the classical inverse and cache the results.

    Returns
    -------
    int
        ``0`` when every group is cached, ``2`` when work remains.
    """
    groups = build_groups(args)
    if not groups:
        return 1

    # An import of TensorFlow anywhere upstream reintroduces the segfault.
    if "tensorflow" in sys.modules:
        print(
            "TensorFlow is loaded in the classical phase.  That combination "
            "segfaults on Windows (D23) -- something on the import path "
            "pulled it in.  Refusing to run.",
            file=sys.stderr,
        )
        return 1

    done_this_run = 0
    budget = args.max_chunks_per_run or 10**9

    for key, label, part in groups:
        if args.only and key != args.only:
            continue
        path = cache_path(args, key)
        if path.exists():
            print(f"[{label}] already cached", flush=True)
            continue

        n = len(part["curves"])
        chunk = n if args.classical_chunk_size <= 0 else min(
            args.classical_chunk_size, n
        )
        n_chunks = -(-n // chunk)
        path.parent.mkdir(parents=True, exist_ok=True)

        for index in range(n_chunks):
            part_file = chunk_path(args, key, index)
            if part_file.exists():
                continue
            if done_this_run >= budget:
                break
            start = index * chunk
            size = min(chunk, n - start)
            suffix = "" if n_chunks == 1 else f" chunk {index + 1}/{n_chunks}"
            print(f"\n[{label}]{suffix} {size} classical fits ...", flush=True)
            species, alkalinity, seconds = classical_species(part, start, size)
            temporary = part_file.with_suffix(".tmp.npz")
            np.savez(
                temporary,
                alkalinity=alkalinity,
                seconds=np.asarray(seconds),
                **species,
            )
            temporary.replace(part_file)
            print(f"      {1e3 * seconds / size:.1f} ms/curve", flush=True)
            done_this_run += 1

        merge_chunks(args, key, n, chunk)

    missing = [
        key
        for key, _, _ in groups
        if not cache_path(args, key).exists()
    ]
    if missing:
        print(f"still to fit: {', '.join(missing)}", file=sys.stderr)
        return 2
    return 0


def load_classical(args: argparse.Namespace, key: str, n: int) -> tuple[
    dict[str, np.ndarray], np.ndarray, float
]:
    """Read one group's cached classical fits.

    Raises
    ------
    FileNotFoundError
        If the classical phase has not been run for this group.
    ValueError
        If the cache holds a different number of curves than the group being
        scored -- which means ``--n-eval`` or ``--split-seed`` changed between
        phases and the two estimators would be compared on different water.
    """
    path = cache_path(args, key)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run --phase classical first"
        )
    cached = np.load(path)
    if len(cached["alkalinity"]) != n:
        raise ValueError(
            f"{path} holds {len(cached['alkalinity'])} curves but this group "
            f"has {n}.  --n-eval or --split-seed changed between phases; "
            f"delete {path.parent} and re-run the classical phase."
        )
    species = {name: cached[name] for name in SPECIES}
    return species, cached["alkalinity"], float(cached["seconds"])


def evaluate(
    key: str,
    label: str,
    part: dict[str, np.ndarray],
    model: object,
    normalizer: Normalizer,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Score both estimators on one group of curves."""
    print(f"\n[{label}] {len(part['curves'])} curves", flush=True)
    organic = part["metadata"][:, META_COLUMNS.index("total_organic")] * 1e6
    truth_alk = part["metadata"][:, META_COLUMNS.index("alkalinity")]

    net, net_alk, net_seconds, net_organic = network_species(
        model, normalizer, part
    )
    print(f"      network   {1e3 * net_seconds / len(part['curves']):.3f} ms/curve")
    cls, cls_alk, cls_seconds = load_classical(args, key, len(part["curves"]))
    print(f"      classical {1e3 * cls_seconds / len(part['curves']):.1f} ms/curve")

    # A fit that did not converge is excluded from BOTH estimators, so the two
    # are always scored on an identical set of curves.
    finite = np.isfinite(cls_alk) & np.isfinite(net_alk)
    for name in SPECIES:
        finite &= np.isfinite(cls[name]) & np.isfinite(net[name])
    n_dropped = int((~finite).sum())
    if n_dropped:
        print(f"      {n_dropped} curves dropped (a fit did not converge)")

    results: dict[str, object] = {
        "n_curves": int(finite.sum()),
        "n_dropped": n_dropped,
        "ms_per_curve": {
            "network": round(1e3 * net_seconds / len(part["curves"]), 4),
            "classical": round(1e3 * cls_seconds / len(part["curves"]), 2),
        },
        "organic_umol_per_kg": {
            "mean": round(float(organic.mean()), 2),
            "max": round(float(organic.max()), 2),
            "fraction_present": round(float((organic > 1e-9).mean()), 3),
        },
        # What the encoder thinks the organic content is.
        "predicted_organic_umol_per_kg": {
            "mean": round(float(1e6 * net_organic[finite].mean()), 2),
            "sd": round(float(1e6 * net_organic[finite].std()), 2),
            "min": round(float(1e6 * net_organic[finite].min()), 2),
            "max": round(float(1e6 * net_organic[finite].max()), 2),
            "true_mean": round(float(organic[finite].mean()), 2),
        },
    }

    for j, name in enumerate(SPECIES):
        results[name] = {
            "network": summarise(
                (net[name][finite] - part["labels"][finite, j]) * 1e6,
                organic[finite],
            ),
            "classical": summarise(
                (cls[name][finite] - part["labels"][finite, j]) * 1e6,
                organic[finite],
            ),
        }
    results["alkalinity"] = {
        "network": summarise(
            (net_alk[finite] - truth_alk[finite]) * 1e6, organic[finite]
        ),
        "classical": summarise(
            (cls_alk[finite] - truth_alk[finite]) * 1e6, organic[finite]
        ),
    }
    return results


def report(name: str, block: dict[str, object]) -> None:
    """Print one group's bicarbonate result, which is the headline."""
    hco3 = block["hco3"]
    print(f"\n{name}  ({block['n_curves']} curves)")
    print(f"  {'estimator':12s}{'bias':>10s}{'sd':>10s}{'MAE':>10s}{'corr(org)':>12s}")
    for who in ("classical", "network"):
        v = hco3[who]
        corr = v.get("corr_with_organic")
        print(
            f"  {who:12s}{v['bias']:+10.2f}{v['sd']:10.2f}{v['mae']:10.2f}"
            f"{'  n/a' if corr is None else f'{corr:+12.3f}'}"
        )
    predicted = block["predicted_organic_umol_per_kg"]
    print(
        f"  organic umol/kg: true {predicted['true_mean']:.1f}, "
        f"predicted {predicted['mean']:.1f} +- {predicted['sd']:.1f} "
        f"(range {predicted['min']:.1f}-{predicted['max']:.1f})"
    )


def build_groups(
    args: argparse.Namespace,
) -> list[tuple[str, str, dict[str, np.ndarray]]]:
    """Assemble the curve subsets both phases score.

    Returns
    -------
    list
        ``(key, label, part)`` triples, or empty if a validation check failed.
    """
    print(f"loading {args.dataset} ...", flush=True)
    data, _ = Dataset.from_npz(args.dataset)
    if organic_fraction(data) <= 0.0:
        print(
            f"{args.dataset} contains no organic alkalinity, so there is "
            "nothing to compare; are --dataset and --control swapped?",
            file=sys.stderr,
        )
        return []

    # The same split seed as training, or the "held-out" curves are not held
    # out and every number below is optimistic.
    _, _, test = split_dataset(data, seed=args.split_seed)
    take = min(args.n_eval, len(test["curves"]))
    mixed = {k: v[:take] for k, v in test.items()}

    print(f"loading {args.control} ...", flush=True)
    control_data, _ = Dataset.from_npz(args.control)
    control_fraction = organic_fraction(control_data)
    if control_fraction > 0.0:
        print(
            f"{args.control} has organic alkalinity on "
            f"{100 * control_fraction:.1f}% of its curves, so it cannot serve "
            "as the organic-free control.  Regenerate it with "
            "--organic-probability 0.0",
            file=sys.stderr,
        )
        return []
    control = {
        k: v[: min(args.n_eval, len(control_data.curves))]
        for k, v in {
            "curves": control_data.curves,
            "env": control_data.env,
            "labels": control_data.labels,
            "metadata": control_data.metadata,
        }.items()
    }

    # Scored on the organic half alone: averaging with the clean half halves the bias.
    organic = mixed["metadata"][:, META_COLUMNS.index("total_organic")]
    with_organic = {k: v[organic > 1e-9] for k, v in mixed.items()}

    groups = [
        ("control_organic_free", "control", control),
        ("mixed_organic_present", "mixed", mixed),
    ]
    if len(with_organic["curves"]) >= 20:
        groups.append(("mixed_organic_only", "organic only", with_organic))
    return groups


def run_network_phase(args: argparse.Namespace) -> int:
    """Run the encoder, combine with the cached classical fits, and report."""
    import tensorflow as tf

    print(f"loading {args.model} ...", flush=True)
    normalizer = Normalizer.load(args.normalizer)
    # Force the lazy custom-layer definitions to register before load_model.
    register_serializable_layers()
    model = tf.keras.models.load_model(args.model, compile=False)

    pending = build_groups(args)
    if not pending:
        return 1
    _, provenance = Dataset.from_npz(args.dataset)
    _, control_provenance = Dataset.from_npz(args.control)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    groups = load_completed_groups(args.output)
    if groups:
        print(f"\nresuming: {', '.join(groups)} already done", flush=True)

    for key, label, part in pending:
        if key in groups:
            continue
        if args.only and key != args.only:
            continue
        try:
            groups[key] = evaluate(key, label, part, model, normalizer, args)
        except (FileNotFoundError, ValueError) as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 1
        # Written after EVERY group, not at the end.
        write_payload(args, provenance, control_provenance, groups)
        print(f"      saved {key} to {args.output}", flush=True)

    missing = [key for key, _, _ in pending if key not in groups]
    if missing:
        print(
            f"\nnot yet run: {', '.join(missing)}.  Re-run the same command "
            "to continue.",
            file=sys.stderr,
        )
        return 2

    print("\n" + "=" * 66)
    print("BICARBONATE ERROR, umol/kg".center(66))
    print("=" * 66)
    for name, block in groups.items():
        report(name, block)

    control_net = groups["control_organic_free"]["hco3"]["network"]
    control_cls = groups["control_organic_free"]["hco3"]["classical"]
    print(
        "\nRead it in this order:"
        "\n  1. On organic-free water the network must not have got worse. "
        f"\n     network MAE {control_net['mae']:.2f} vs classical "
        f"{control_cls['mae']:.2f}."
        "\n     If the network is much worse here, the organic latents cost "
        "more than\n     they bought and nothing below is worth reporting."
        "\n  2. Only then, on organic water: the classical bias should be "
        "large and\n     correlated with organic content; the network's "
        "should not be."
    )
    print(f"\nwritten to {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the requested phase."""
    args = parse_args(argv)
    if not args.enable_onednn:
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    for path in (args.dataset, args.control):
        if not path.exists():
            print(f"not found: {path}", file=sys.stderr)
            return 1
    if args.phase != "classical":
        for path in (args.model, args.normalizer):
            if not path.exists():
                print(f"not found: {path}", file=sys.stderr)
                return 1

    if args.phase == "classical":
        return run_classical_phase(args)
    if args.phase == "network":
        return run_network_phase(args)

    forwarded = [a for a in (argv if argv is not None else sys.argv[1:])
                 if not a.startswith("--phase")]
    # One chunk per subprocess, overriding whatever was passed: the point of
    # the loop is that each chunk gets a fresh interpreter.
    command = [sys.executable, str(Path(__file__).resolve()), "--phase",
               "classical", *forwarded, "--max-chunks-per-run", "1"]

    # One fresh interpreter per chunk; retry only on exit 2 or a signal death.
    print("running the classical phase in clean interpreters ...", flush=True)
    crashes = 0
    for _ in range(MAX_CLASSICAL_ATTEMPTS):
        completed = subprocess.run(command, check=False)
        status = completed.returncode
        if status == 0:
            break
        if status == 2:
            continue
        if status < 128:
            print(
                f"the classical phase exited with {status}, which is an "
                "error rather than a crash.  Stopping.",
                file=sys.stderr,
            )
            return status
        crashes += 1
        print(f"  (crashed with status {status} - resuming)", flush=True)
        time.sleep(CLASSICAL_RETRY_SECONDS)
    else:
        print(
            f"the classical phase did not finish in "
            f"{MAX_CLASSICAL_ATTEMPTS} attempts.",
            file=sys.stderr,
        )
        return 1

    if crashes:
        print(f"classical phase complete after {crashes} crash(es).",
              flush=True)
    return run_network_phase(args)


if __name__ == "__main__":
    raise SystemExit(main())
