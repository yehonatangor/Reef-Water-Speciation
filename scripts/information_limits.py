#!/usr/bin/env python3
r"""Compute the Cramer-Rao bound on alkalinity and species accuracy.

Examples
--------
::

    python scripts/information_limits.py
    python scripts/information_limits.py --verify --output results/limits.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import sys
from pathlib import Path

import numpy as np

# Run from a source checkout as well as an installed package.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cleaned_reef_water import __version__, simulate_titration, speciate
from cleaned_reef_water.alkalinity import constants_at
from cleaned_reef_water.ml.noise import NoiseModel, apply_noise
from cleaned_reef_water.solvers import ph_from_dic_alkalinity

SPECIES = ("hco3", "co3", "boh4")
SALINITY, TEMPERATURE = 35.0, 25.0
SAMPLE_MASS, TITRANT = 0.015, 0.1
CALIBRATION_PH = 8.0936

#: Measured performance of the classical inverse, from ``run_baseline.py``
#: with ``OFFSET_AND_SLOPE``.  The bound must not exceed these.
MEASURED_BASELINE = {"hobbyist": 22.0, "laboratory": 3.5}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-points", type=int, default=40)
    parser.add_argument("--n-monte-carlo", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="assert the bound does not exceed the measured baseline",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def jacobian(
    alkalinity: float, dic: float, n_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(G, clean_ph, titrant_mass)`` for one composition."""
    curve = simulate_titration(
        alkalinity,
        dic,
        SALINITY,
        TEMPERATURE,
        sample_mass_kg=SAMPLE_MASS,
        n_points=n_points,
    )
    clean, mass = curve.ph_total, curve.titrant_mass

    def forward(
        a: float, c: float, offset: float, slope: float, scale: float
    ) -> np.ndarray:
        equivalence = SAMPLE_MASS * a / TITRANT
        rendered = simulate_titration(
            a,
            c,
            SALINITY,
            TEMPERATURE,
            sample_mass_kg=SAMPLE_MASS,
            n_points=n_points,
            max_equivalence_fraction=(mass[-1] * scale) / equivalence,
        )
        return CALIBRATION_PH + slope * (rendered.ph_total - CALIBRATION_PH) + offset

    centre = np.array([alkalinity, dic, 0.0, 1.0, 1.0])
    steps = np.array([1e-8, 1e-8, 1e-5, 1e-5, 1e-5])
    g = np.empty((n_points, 5))
    for j in range(5):
        up, down = centre.copy(), centre.copy()
        up[j] += steps[j]
        down[j] -= steps[j]
        g[:, j] = (forward(*up) - forward(*down)) / (2.0 * steps[j])
    return g, clean, mass


def residual_covariance(
    model: NoiseModel,
    clean: np.ndarray,
    mass: np.ndarray,
    n_monte_carlo: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Covariance of the noise that no fitted parameter can absorb."""
    stripped = dataclasses.replace(
        model,
        slope_error_sd=0.0,
        offset_sd=0.0,
        nbs_calibration_bias_mean=0.0,
        nbs_calibration_bias_sd=0.0,
        burette_scale_error_sd=0.0,
        burette_jitter_sd=0.0,
    )
    residuals = np.array(
        [
            apply_noise(
                mass,
                clean,
                SALINITY,
                TEMPERATURE,
                model=stripped,
                rng=np.random.default_rng(seed + k),
            ).ph_measured
            - clean
            for k in range(n_monte_carlo)
        ]
    )
    return np.cov(residuals, rowvar=False), float(residuals.std())


def parameter_covariance(
    g: np.ndarray, sigma: np.ndarray, burette_sd: float
) -> np.ndarray:
    """Inverse Fisher information for ``(A_T, C_T, offset, slope)``."""
    total = (
        sigma
        + burette_sd**2 * np.outer(g[:, 4], g[:, 4])
        + 1e-14 * np.eye(sigma.shape[0])
    )
    fitted = g[:, :4]
    information = fitted.T @ np.linalg.inv(total) @ fitted
    return np.linalg.inv(information)


def species_sd(alkalinity: float, dic: float, covariance: np.ndarray) -> np.ndarray:
    """Propagate the totals covariance onto species by the delta method."""

    def species(a: float, c: float) -> np.ndarray:
        ph = ph_from_dic_alkalinity(c, a, SALINITY, TEMPERATURE)
        found = speciate(ph, constants_at(SALINITY, TEMPERATURE), dic=c)
        return np.array([float(getattr(found, name)) for name in SPECIES])

    step = 1e-8
    jac = np.empty((len(SPECIES), 2))
    for j in range(2):
        up, down = [alkalinity, dic], [alkalinity, dic]
        up[j] += step
        down[j] -= step
        jac[:, j] = (species(*up) - species(*down)) / (2.0 * step)
    return np.sqrt(np.diag(jac @ covariance[:2, :2] @ jac.T))


def main(argv: list[str] | None = None) -> int:
    """Compute and print every bound reported in the documentation."""
    args = parse_args(argv)
    instruments = {
        "hobbyist": NoiseModel.hobbyist(),
        "laboratory": NoiseModel.laboratory(),
    }
    report: dict[str, object] = {}

    print("=" * 72)
    print("1. Is the classical estimator already optimal?")
    print("=" * 72)
    print(f"{'instrument':16s}{'noise sd':>10}{'CRLB A_T':>11}{'measured':>11}"
          f"{'headroom':>11}")
    optimality = {}
    for name, model in instruments.items():
        g, clean, mass = jacobian(2300e-6, 2000e-6, args.n_points)
        sigma, noise_sd = residual_covariance(
            model, clean, mass, args.n_monte_carlo, args.seed
        )
        covariance = parameter_covariance(g, sigma, model.burette_scale_error_sd)
        bound = float(np.sqrt(covariance[0, 0]) * 1e6)
        measured = MEASURED_BASELINE[name]
        optimality[name] = {
            "crlb_umol_per_kg": round(bound, 2),
            "measured_umol_per_kg": measured,
            "residual_noise_sd_ph": round(noise_sd, 5),
        }
        print(f"{name:16s}{noise_sd:10.4f}{bound:11.2f}{measured:11.1f}"
              f"{measured / bound:10.2f}x")
        if args.verify and bound > measured * 1.02:
            print(
                f"\nFAIL: the bound ({bound:.2f}) exceeds the measured error "
                f"({measured:.1f}) for {name}. An estimator cannot beat its "
                f"own Cramer-Rao bound, so the bound is mis-specified.",
                file=sys.stderr,
            )
            return 1
    report["optimality"] = optimality

    print()
    print("=" * 72)
    print("2. Where is the leverage? (hobbyist)")
    print("=" * 72)
    hobbyist = instruments["hobbyist"]
    leverage = {}
    variants = {
        "as modelled": {},
        "junction drift removed": {"junction_drift_sd": 0.0},
        "CO2 degassing removed": {"co2_loss_fraction_range": (0.0, 0.0)},
        "electrode noise halved": {"electrode_noise_sd": 0.006},
        "electrode noise removed": {"electrode_noise_sd": 0.0},
        "burette 0.8% -> 0.4%": {"burette_scale_error_sd": 0.004},
        "burette 0.8% -> 0.1%": {"burette_scale_error_sd": 0.001},
    }
    print(f"{'change':32s}{'noise sd':>10}{'CRLB A_T':>11}")
    for label, changes in variants.items():
        model = dataclasses.replace(hobbyist, **changes)
        g, clean, mass = jacobian(2300e-6, 2000e-6, args.n_points)
        sigma, noise_sd = residual_covariance(
            model, clean, mass, args.n_monte_carlo, args.seed
        )
        covariance = parameter_covariance(g, sigma, model.burette_scale_error_sd)
        bound = float(np.sqrt(covariance[0, 0]) * 1e6)
        leverage[label] = round(bound, 2)
        print(f"{label:32s}{noise_sd:10.4f}{bound:11.2f}")
    report["leverage_hobbyist"] = leverage

    print()
    print("=" * 72)
    print("3. Species accuracy across the composition range (hobbyist)")
    print("=" * 72)
    print(f"{'A_T':>6}{'DIC/A_T':>9}{'pH':>7}{'A_T sd':>9}"
          + "".join(f"{name + ' sd':>10}" for name in SPECIES)
          + f"{'worst rel':>11}")
    grid = []
    for alkalinity in (1500e-6, 2300e-6, 3200e-6, 4000e-6):
        for ratio in (0.80, 0.88, 0.95):
            dic = alkalinity * ratio
            g, clean, mass = jacobian(alkalinity, dic, args.n_points)
            sigma, _ = residual_covariance(
                hobbyist, clean, mass, args.n_monte_carlo, args.seed
            )
            covariance = parameter_covariance(
                g, sigma, hobbyist.burette_scale_error_sd
            )
            sd = species_sd(alkalinity, dic, covariance)
            ph = ph_from_dic_alkalinity(dic, alkalinity, SALINITY, TEMPERATURE)
            values = speciate(ph, constants_at(SALINITY, TEMPERATURE), dic=dic)
            magnitudes = np.array(
                [float(getattr(values, name)) for name in SPECIES]
            )
            worst = float(np.max(sd / magnitudes) * 100.0)
            alkalinity_sd = float(np.sqrt(covariance[0, 0]) * 1e6)
            grid.append(
                {
                    "alkalinity_umol_per_kg": round(alkalinity * 1e6),
                    "dic_over_alkalinity": ratio,
                    "initial_ph": round(float(ph), 3),
                    "alkalinity_sd": round(alkalinity_sd, 2),
                    "species_sd_umol_per_kg": {
                        name: round(float(v) * 1e6, 2)
                        for name, v in zip(SPECIES, sd, strict=True)
                    },
                    "worst_relative_percent": round(worst, 2),
                }
            )
            print(f"{alkalinity * 1e6:6.0f}{ratio:9.2f}{float(ph):7.2f}"
                  f"{alkalinity_sd:9.1f}"
                  + "".join(f"{v * 1e6:10.1f}" for v in sd)
                  + f"{worst:10.1f}%")
    report["species_grid_hobbyist"] = grid
    print("\nunits: umol/kg.  'worst rel' is the largest relative sd of any "
          "species.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provenance": {
                "package_version": __version__,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "command": " ".join(sys.argv),
                "n_points": args.n_points,
                "n_monte_carlo": args.n_monte_carlo,
                "seed": args.seed,
            },
            **report,
        }
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf8")
        print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
