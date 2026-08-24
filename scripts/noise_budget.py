#!/usr/bin/env python3
r"""Reproduce the instrument noise budget in ``machine_learning.md`` §6.3."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

# Part of the documented-claims tooling, so it must run from a bare checkout.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cleaned_reef_water import simulate_titration
from cleaned_reef_water.ml.noise import NoiseModel, apply_noise

#: Reference sample: ordinary reef seawater, mid-range.
ALKALINITY, DIC = 2.30e-3, 2.00e-3
SALINITY, T_C = 35.0, 25.0

#: Each mechanism, and the ``NoiseModel`` fields that switch it on.
MECHANISMS: dict[str, tuple[str, ...]] = {
    "Nernstian slope": ("slope_error_sd",),
    "CO2 degassing": ("co2_loss_fraction_range",),
    "Electrode AR(1)": ("electrode_noise_sd",),
    "Junction drift": ("junction_drift_sd",),
    "Calibration offset": ("offset_sd",),
    "NBS calibration bias": ("nbs_calibration_bias_mean", "nbs_calibration_bias_sd"),
    "Burette scale + jitter": ("burette_scale_error_sd", "burette_jitter_sd"),
    "ADC quantisation": ("quantisation_step_ph",),
    "Salinity measurement": ("salinity_error_sd",),
    "Temperature measurement": ("temperature_error_sd",),
}

#: Fields that are not tolerances and must survive being zeroed.
_STRUCTURAL = ("calibration_ph", "ar1_rho_range", "co2_time_constant_range")


def _silenced(model: NoiseModel, keep: tuple[str, ...]) -> NoiseModel:
    """Return a copy of ``model`` with every tolerance zeroed except ``keep``."""
    changes: dict[str, object] = {}
    for field in dataclasses.fields(model):
        name = field.name
        if name in keep or name in _STRUCTURAL:
            continue
        current = getattr(model, name)
        changes[name] = (0.0, 0.0) if isinstance(current, tuple) else 0.0
    # ar1_rho_range only matters when electrode noise is on; leaving it intact
    # is harmless because it multiplies a zero amplitude.
    return dataclasses.replace(model, **changes)


def residual_statistics(
    model: NoiseModel, n_realisations: int, n_points: int, seed: int
) -> tuple[float, float]:
    """Measure the pH residual sd and worst case over many noise realisations.

    Returns
    -------
    tuple
        ``(mean sd, mean max-abs)`` in pH units.
    """
    curve = simulate_titration(
        ALKALINITY, DIC, SALINITY, T_C, n_points=n_points
    )
    rng = np.random.default_rng(seed)
    sds, maxima = [], []
    for _ in range(n_realisations):
        noisy = apply_noise(
            curve.titrant_mass, curve.ph_total, SALINITY, T_C,
            model=model, rng=rng,
        )
        residual = np.asarray(noisy.ph_measured) - np.asarray(curve.ph_total)
        sds.append(float(np.std(residual)))
        maxima.append(float(np.max(np.abs(residual))))
    return float(np.mean(sds)), float(np.mean(maxima))


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", choices=("hobbyist", "laboratory"),
                        default="hobbyist")
    parser.add_argument("--n-realisations", type=int, default=200)
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Compute and print the noise budget."""
    args = parse_args(argv)
    model = getattr(NoiseModel, args.instrument)()

    print(f"Instrument noise budget -- {args.instrument} preset")
    print(f"A_T = {1e6 * ALKALINITY:.0f}, DIC = {1e6 * DIC:.0f} umol/kg, "
          f"S = {SALINITY:g}, t = {T_C:g} C")
    print(f"{args.n_realisations} realisations of a {args.n_points}-point "
          f"curve, seed {args.seed}\n")

    total_sd, total_max = residual_statistics(
        model, args.n_realisations, args.n_points, args.seed
    )
    print("All mechanisms active:")
    print(f"  pH residual sd    {total_sd:.4f}")
    print(f"  max abs residual  {total_max:.4f}\n")

    print("Term by term, every other tolerance zeroed:")
    print(f"  {'mechanism':<26}{'sd':>10}{'max':>10}{'% of total':>12}")
    print("  " + "-" * 58)

    rows = []
    for name, fields in MECHANISMS.items():
        sd, worst = residual_statistics(
            _silenced(model, fields), args.n_realisations,
            args.n_points, args.seed,
        )
        rows.append((name, sd, worst))
    rows.sort(key=lambda row: row[1], reverse=True)
    for name, sd, worst in rows:
        share = 100.0 * sd / total_sd if total_sd else 0.0
        print(f"  {name:<26}{sd:>10.5f}{worst:>10.5f}{share:>11.1f}%")

    quadrature = float(np.sqrt(sum(sd**2 for _, sd, _ in rows)))
    print(f"\n  {'quadrature sum':<26}{quadrature:>10.5f}")
    print(f"  {'actual total':<26}{total_sd:>10.5f}")
    print(
        "\nThe two disagree because several terms are correlated along the\n"
        "curve rather than independent -- the slope error pivots about the\n"
        "calibration pH and CO2 loss grows monotonically through the run --\n"
        "so they do not add in quadrature.  Read the ranking, not the sum."
    )

    dominant = rows[0]
    print(f"\nDominant term: {dominant[0]} at {dominant[1]:.5f} pH "
          f"({100 * dominant[1] / total_sd:.0f}% of the total sd).")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "instrument": args.instrument,
            "n_realisations": args.n_realisations,
            "n_points": args.n_points,
            "seed": args.seed,
            "total": {"sd": round(total_sd, 6), "max_abs": round(total_max, 6)},
            "by_mechanism": {
                name: {"sd": round(sd, 6), "max_abs": round(worst, 6)}
                for name, sd, worst in rows
            },
            "quadrature_sum": round(quadrature, 6),
        }, indent=2))
        print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
