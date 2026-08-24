#!/usr/bin/env python3
"""Benchmark the classical least-squares inverse under realistic noise.

Examples
--------
Run the default comparison and write ``results/baseline.json``::

    python scripts/run_baseline.py

If it segfaults, drive it with the loop script, which chunks the work, retries
only on crashes, and pauses between attempts::

    bash scripts/run_baseline_loop.sh

    # gentler, if repeated restarts destabilise the machine
    CHUNK_SIZE=5 SLEEP_SECONDS=5 bash scripts/run_baseline_loop.sh

Run only hobbyist hardware with more replicates::

    python scripts/run_baseline.py --instrument hobbyist --n-curves 200
"""

from __future__ import annotations

import argparse
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

from cleaned_reef_water import (
    __version__,
    constants_at,
    simulate_titration,
    speciate,
)
from cleaned_reef_water.baseline import LeastSquaresInverse, NuisanceModel
from cleaned_reef_water.ml.noise import NoiseModel, apply_noise
from cleaned_reef_water.ml.sampling import (
    SamplingRanges,
    sample_composition,
)

INSTRUMENTS = {
    "laboratory": NoiseModel.laboratory,
    "hobbyist": NoiseModel.hobbyist,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-curves", type=int, default=100)
    parser.add_argument("--n-points", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--instrument",
        choices=[*INSTRUMENTS, "all"],
        default="all",
        help="instrument grade to simulate",
    )
    parser.add_argument(
        "--nuisance",
        choices=[m.value for m in NuisanceModel] + ["all"],
        default="all",
        help="which nuisance parameters the fit may absorb",
    )
    parser.add_argument("--output", type=Path, default=Path("results/baseline.json"))
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help=(
            "curves per chunk file; 0 (default) runs each case in one go. "
            "Set this if scipy segfaults mid-run -- results are unchanged, "
            "since each curve is seeded independently of the chunking"
        ),
    )
    parser.add_argument(
        "--max-chunks-per-run",
        type=int,
        default=0,
        help=(
            "stop after this many chunks and exit 2 so a shell loop can "
            "restart the interpreter; 0 means do as many as possible"
        ),
    )
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        help="do not delete the per-chunk files after merging",
    )
    return parser.parse_args(argv)


def part_path(output: Path, instrument: str, nuisance: str, index: int) -> Path:
    """Return the sidecar filename for one chunk of one case."""
    return output.with_suffix(f".{instrument}.{nuisance}.part{index:04d}.json")


def run_curves(
    instrument: str,
    nuisance: NuisanceModel,
    start: int,
    count: int,
    n_points: int,
    seed: int,
) -> dict[str, object]:
    """Benchmark ``count`` curves starting at index ``start``.

    Parameters
    ----------
    instrument
        Key into :data:`INSTRUMENTS`.
    nuisance
        Nuisance-parameter model for the fit.
    start, count
        Half-open range of curve indices to evaluate.
    n_points
        Points per titration curve.
    seed
        Base seed.  Each curve uses ``seed + index`` so results are
        reproducible and independent of ordering **and of chunking** -- curve
        7 is the same curve whether it is the eighth of one long run or the
        third of a chunk starting at 5.

    Returns
    -------
    dict
        Raw per-curve errors in micromol per kilogram, plus the failure count
        and elapsed seconds.  Deliberately unsummarised: a mean of means is
        not a mean, so the statistics are computed once over the merged set.
    """
    model = INSTRUMENTS[instrument]()
    ranges = SamplingRanges()
    errors: dict[str, list[float]] = {
        "alkalinity": [],
        "dic": [],
        "hco3": [],
        "co3": [],
        "boh4": [],
    }
    n_failed = 0
    started = time.perf_counter()

    for index in range(start, start + count):
        rng = np.random.default_rng(seed + index)
        composition = sample_composition(rng, ranges)
        try:
            # Phosphate and silicate MUST be passed.
            clean = simulate_titration(
                composition.alkalinity,
                composition.dic,
                composition.salinity,
                composition.t_c,
                sample_mass_kg=composition.sample_mass_kg,
                n_points=n_points,
                total_boron=composition.total_boron,
                total_phosphate=composition.total_phosphate,
                total_silicate=composition.total_silicate,
            )
        except Exception:  # noqa: BLE001 - forward model failure, count and skip
            n_failed += 1
            continue

        noisy = apply_noise(
            clean.titrant_mass,
            clean.ph_total,
            composition.salinity,
            composition.t_c,
            model=model,
            rng=rng,
        )
        inverse = LeastSquaresInverse(
            sample_mass_kg=composition.sample_mass_kg, nuisance=nuisance
        )
        result = inverse.fit(
            noisy.titrant_mass,
            noisy.ph_measured,
            noisy.salinity_measured,
            noisy.t_c_measured,
        )
        if not np.isfinite(result.hco3):
            n_failed += 1
            continue

        # The truth must be the speciation of the composition actually drawn,
        # nutrients included -- matching how ``ml.dataset`` builds its labels.
        constants = constants_at(composition.salinity, composition.t_c)
        truth = speciate(
            composition.ph_total,
            constants,
            dic=composition.dic,
            total_boron=composition.total_boron,
            total_phosphate=composition.total_phosphate,
            total_silicate=composition.total_silicate,
        )
        errors["alkalinity"].append((result.alkalinity - composition.alkalinity) * 1e6)
        errors["dic"].append((result.dic - composition.dic) * 1e6)
        errors["hco3"].append((result.hco3 - float(truth.hco3)) * 1e6)
        errors["co3"].append((result.co3 - float(truth.co3)) * 1e6)
        errors["boh4"].append((result.boh4 - float(truth.boh4)) * 1e6)

    return {
        "start": start,
        "count": count,
        "n_failed": n_failed,
        "seconds": round(time.perf_counter() - started, 2),
        "errors": errors,
    }


def summarise(
    instrument: str,
    nuisance: NuisanceModel,
    parts: list[dict[str, object]],
    n_points: int,
) -> dict[str, object]:
    """Reduce raw per-curve errors from one or more chunks to statistics."""
    ordered = sorted(parts, key=lambda p: int(p["start"]))
    errors: dict[str, list[float]] = {
        "alkalinity": [], "dic": [], "hco3": [], "co3": [], "boh4": [],
    }
    for part in ordered:
        for name, values in part["errors"].items():  # type: ignore[union-attr]
            errors[name].extend(values)

    summary: dict[str, object] = {
        "instrument": instrument,
        "nuisance": nuisance.value,
        "n_curves": sum(int(p["count"]) for p in ordered),
        "n_points": n_points,
        "n_failed": sum(int(p["n_failed"]) for p in ordered),
        "seconds": round(sum(float(p["seconds"]) for p in ordered), 2),
        "n_chunks": len(ordered),
    }
    for name, values in errors.items():
        array = np.asarray(values)
        summary[name] = {
            "bias": round(float(np.mean(array)), 3),
            "sd": round(float(np.std(array)), 3),
            "mae": round(float(np.mean(np.abs(array))), 3),
            "rmse": round(float(np.sqrt(np.mean(array**2))), 3),
            "p95_abs": round(float(np.percentile(np.abs(array), 95)), 3),
            "median_abs": round(float(np.median(np.abs(array))), 3),
            "max_abs": round(float(np.max(np.abs(array))), 3),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and write results to JSON."""
    args = parse_args(argv)
    instruments = list(INSTRUMENTS) if args.instrument == "all" else [args.instrument]
    nuisances = (
        list(NuisanceModel)
        if args.nuisance == "all"
        else [NuisanceModel(args.nuisance)]
    )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    chunk = args.n_curves if args.chunk_size <= 0 else min(
        args.chunk_size, args.n_curves
    )
    n_chunks = -(-args.n_curves // chunk)  # ceiling division
    budget = args.max_chunks_per_run or (n_chunks * len(instruments) * len(nuisances))
    done_this_run = 0

    # Pass one: compute any chunk that is not already on disk.
    for instrument in instruments:
        for nuisance in nuisances:
            for index in range(n_chunks):
                path = part_path(output, instrument, nuisance.value, index)
                if path.exists():
                    continue
                if done_this_run >= budget:
                    break
                start = index * chunk
                count = min(chunk, args.n_curves - start)
                label = f"{instrument:11s} / {nuisance.value:17s}"
                suffix = "" if n_chunks == 1 else f" chunk {index + 1}/{n_chunks}"
                print(f"running {label}{suffix} ...", flush=True)
                part = run_curves(
                    instrument, nuisance, start, count, args.n_points, args.seed
                )
                path.write_text(json.dumps(part), encoding="utf8")
                done_this_run += 1

    # Pass two: is every chunk of every case present?
    missing = [
        (instrument, nuisance, index)
        for instrument in instruments
        for nuisance in nuisances
        for index in range(n_chunks)
        if not part_path(output, instrument, nuisance.value, index).exists()
    ]
    if missing:
        total = n_chunks * len(instruments) * len(nuisances)
        print(
            f"\n{total - len(missing)}/{total} chunks done, {len(missing)} to go."
            f"\nRun again to resume, or drive it with:"
            f"\n  bash scripts/run_baseline_loop.sh"
            f"\n(which retries only on crashes, pauses between attempts, and"
            f"\n stops on real errors instead of looping forever)"
        )
        return 2

    cases = []
    for instrument in instruments:
        for nuisance in nuisances:
            parts = [
                json.loads(
                    part_path(output, instrument, nuisance.value, i).read_text(
                        encoding="utf8"
                    )
                )
                for i in range(n_chunks)
            ]
            cases.append(summarise(instrument, nuisance, parts, args.n_points))

    payload = {
        "provenance": {
            "package_version": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "seed": args.seed,
            "command": " ".join(sys.argv),
            "chunk_size": chunk,
            "n_chunks": n_chunks,
        },
        "cases": cases,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf8")

    if not args.keep_parts:
        for instrument in instruments:
            for nuisance in nuisances:
                for index in range(n_chunks):
                    part_path(output, instrument, nuisance.value, index).unlink(
                        missing_ok=True
                    )

    print(f"\n{'instrument':12s}{'nuisance':18s}{'A_T bias':>10s}"
          f"{'A_T sd':>9s}{'A_T MAE':>10s}{'A_T med':>9s}{'A_T max':>10s}"
          f"{'HCO3 MAE':>10s}")
    for case in cases:
        alk = case["alkalinity"]
        hco3 = case["hco3"]
        print(
            f"{case['instrument']:12s}{case['nuisance']:18s}"
            f"{alk['bias']:+10.2f}{alk['sd']:9.2f}{alk['mae']:10.2f}"
            f"{alk['median_abs']:9.2f}{alk['max_abs']:10.1f}{hco3['mae']:10.2f}"
        )
    print("\nunits: umol/kg   |   CRM titration reproducibility: +/-2 umol/kg")
    print(f"written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
