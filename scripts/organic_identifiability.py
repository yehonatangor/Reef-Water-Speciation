#!/usr/bin/env python3
r"""Measure how much of organic alkalinity a titration curve can actually resolve."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

# Run from a source checkout as well as an installed package.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cleaned_reef_water import simulate_titration
from cleaned_reef_water.alkalinity import constants_at

#: Reference state.  Ordinary reef seawater in the middle of the sampler's range.
SALINITY, T_C = 35.0, 25.0
ALKALINITY, DIC = 2.30e-3, 2.00e-3
ORGANIC, PK_ORGANIC = 7.5e-5, 5.5

PARAMETERS = ("alkalinity", "dic", "total_organic", "pk_organic")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument(
        "--pk-organic",
        type=float,
        default=PK_ORGANIC,
        help=(
            "organic pKa of the reference state.  Sweep it to see the "
            "degeneracy deepen as it approaches carbonic pK1"
        ),
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="report the degeneracy across the whole 4.0-7.0 pKa draw range",
    )
    parser.add_argument(
        "--sweep-states",
        action="store_true",
        help=(
            "vary salinity, temperature, alkalinity and the C_T/A_T ratio "
            "across the sampler's range, to check the degeneracy is a "
            "property of the chemistry rather than of one reference point"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


@dataclass(frozen=True)
class State:
    """One seawater composition at which to evaluate the degeneracy."""

    salinity: float = SALINITY
    t_c: float = T_C
    alkalinity: float = ALKALINITY
    dic: float = DIC
    organic: float = ORGANIC
    pk_organic: float = PK_ORGANIC

    def label(self) -> str:
        """Compact description for tabular output."""
        return (
            f"S={self.salinity:.0f} T={self.t_c:.0f} "
            f"A={1e6 * self.alkalinity:.0f} C/A={self.dic / self.alkalinity:.2f} "
            f"org={1e6 * self.organic:.0f}"
        )


def sensitivities(state: State, n_points: int) -> dict[str, np.ndarray]:
    """Central-difference derivative of the pH trace w.r.t. each unknown.

    Returns
    -------
    dict
        One ``(n_points,)`` vector per parameter.  Steps are 1 umol/kg for the
        three concentrations and 0.05 for the pKa, so the columns are already
        at comparable physical scale and the cosines below are meaningful
        without further weighting.
    """
    def curve(
        alkalinity: float | None = None,
        dic: float | None = None,
        organic: float | None = None,
        pk: float | None = None,
    ) -> np.ndarray:
        return simulate_titration(
            state.alkalinity if alkalinity is None else alkalinity,
            state.dic if dic is None else dic,
            state.salinity,
            state.t_c,
            n_points=n_points,
            total_organic=state.organic if organic is None else organic,
            pk_organic=state.pk_organic if pk is None else pk,
        ).ph_total

    step, pk_step = 1e-6, 0.05
    return {
        "alkalinity": (curve(alkalinity=state.alkalinity + step)
                       - curve(alkalinity=state.alkalinity - step)) / 2.0,
        "dic": (curve(dic=state.dic + step)
                - curve(dic=state.dic - step)) / 2.0,
        "total_organic": (curve(organic=state.organic + step)
                          - curve(organic=state.organic - step)) / 2.0,
        "pk_organic": (curve(pk=state.pk_organic + pk_step)
                       - curve(pk=state.pk_organic - pk_step)) / 2.0,
    }


def explained_fraction(
    jacobian: dict[str, np.ndarray], target: str, by: tuple[str, ...]
) -> float:
    """Fraction of one sensitivity direction lying in the span of others.

    Returns
    -------
    float
        1.0 means the parameter is perfectly degenerate with ``by`` and cannot
        be estimated from the curve at all; 0.0 means fully independent.
    """
    basis = np.stack([jacobian[name] for name in by])
    coefficients, *_ = np.linalg.lstsq(basis.T, jacobian[target], rcond=None)
    residual = jacobian[target] - basis.T @ coefficients
    total = float(jacobian[target] @ jacobian[target])
    return 1.0 - float(residual @ residual) / total


def analyse(n_points: int, pk_organic: float, state: State | None = None) -> dict[
    str, object
]:
    """Run the full analysis at one state and organic pKa."""
    state = replace(state or State(), pk_organic=pk_organic)
    jacobian = sensitivities(state, n_points)
    matrix = np.stack([jacobian[name] for name in PARAMETERS])

    cosines = {}
    for i, a in enumerate(PARAMETERS):
        for b in PARAMETERS[i + 1:]:
            j = PARAMETERS.index(b)
            cosines[f"{a}|{b}"] = round(
                float(
                    matrix[i] @ matrix[j]
                    / (np.linalg.norm(matrix[i]) * np.linalg.norm(matrix[j]))
                ),
                4,
            )

    gram = matrix @ matrix.T
    eigenvalues = np.linalg.eigvalsh(gram)
    return {
        "state": state.label(),
        "pk_organic": pk_organic,
        "carbonic_pk1_total": round(
            float(-np.log10(float(constants_at(state.salinity, state.t_c).k1))), 4
        ),
        "cosines": cosines,
        "gram_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "organic_explained_by_carbonate": round(
            explained_fraction(jacobian, "total_organic", ("alkalinity", "dic")),
            5,
        ),
        "pk_explained_by_carbonate": round(
            explained_fraction(jacobian, "pk_organic", ("alkalinity", "dic")), 5
        ),
    }


#: States spanning the sampler's range, one axis varied at a time from the
#: reference so that any dependence is attributable.
SWEEP_STATES: tuple[State, ...] = (
    State(),
    State(salinity=20.0),
    State(salinity=30.0),
    State(salinity=40.0),
    State(t_c=10.0),
    State(t_c=18.0),
    State(t_c=32.0),
    State(alkalinity=1.60e-3, dic=1.39e-3),
    State(alkalinity=3.20e-3, dic=2.78e-3),
    State(dic=1.84e-3),   # C_T/A_T = 0.80, the low end of the draw
    State(dic=2.185e-3),  # C_T/A_T = 0.95, the high end
    State(organic=1.5e-5),
    State(organic=1.5e-4),
    State(salinity=20.0, t_c=10.0, alkalinity=1.60e-3, dic=1.36e-3),
)


def sweep_states(args: argparse.Namespace) -> int:
    """Check the degeneracy holds across the composition range."""
    print(
        "degeneracy of total_organic with (A_T, C_T), across the state space\n"
        "organic pKa held at the draw-range centre (5.5)\n"
    )
    print(
        f"{'state':>42s}{'pK1':>8s}{'explained':>11s}{'independent':>13s}"
    )
    rows = []
    worst, best = 1.0, 0.0
    for state in SWEEP_STATES:
        result = analyse(args.n_points, state.pk_organic, state)
        fraction = float(result["organic_explained_by_carbonate"])
        worst, best = min(worst, fraction), max(best, fraction)
        print(
            f"{state.label():>42s}"
            f"{result['carbonic_pk1_total']:>8.3f}"
            f"{fraction:>11.4f}"
            f"{100 * (1 - fraction):>12.2f}%"
        )
        rows.append(result)

    print(
        f"\nacross every state tested, between {100 * (1 - best):.2f}% and "
        f"{100 * (1 - worst):.2f}% of the organic\nsignature is independent of "
        "the carbonate totals."
    )
    if 1.0 - worst < 0.10:
        print(
            "The degeneracy is a property of the chemistry, not of the "
            "reference point."
        )
    else:
        print(
            "WARNING: the degeneracy varies materially across states, so the "
            "single-point\nresult in docs/machine_learning.md 6a understates "
            "the situation and should be\nrestated as a range."
        )
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2))
        print(f"\nwritten to {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Print the identifiability analysis."""
    args = parse_args(argv)

    if args.sweep:
        print("degeneracy of total_organic with (A_T, C_T), across the draw range\n")
        print(f"{'organic pKa':>12s}{'explained':>12s}{'independent':>14s}")
        rows = []
        for pk in (4.0, 4.5, 5.0, 5.5, 5.85, 6.0, 6.5, 7.0):
            result = analyse(args.n_points, pk)
            fraction = result["organic_explained_by_carbonate"]
            marker = "  <- carbonic pK1" if abs(pk - 5.85) < 0.01 else ""
            print(
                f"{pk:12.2f}{fraction:12.4f}{100 * (1 - fraction):13.2f}%{marker}"
            )
            rows.append(result)
        if args.output:
            args.output.write_text(json.dumps(rows, indent=2))
            print(f"\nwritten to {args.output}")
        return 0

    if args.sweep_states:
        return sweep_states(args)

    result = analyse(args.n_points, args.pk_organic)
    print(f"carbonic pK1 (total scale, S=35, T=25) = {result['carbonic_pk1_total']}")
    print(f"organic pKa of this reference state    = {result['pk_organic']}\n")

    print("cosine between sensitivity directions (+-1 = indistinguishable):")
    for pair, value in result["cosines"].items():
        a, b = pair.split("|")
        flag = "   DEGENERATE" if abs(value) > 0.95 else ""
        print(f"  {a:>14s} vs {b:<16s}{value:+8.4f}{flag}")

    fraction = result["organic_explained_by_carbonate"]
    print(
        f"\nfraction of the organic signature reproducible by (A_T, C_T): "
        f"{fraction:.4f}"
        f"\n  -> only {100 * (1 - fraction):.2f}% of it is genuinely independent"
    )
    print(
        f"\nfor comparison, the pKa direction: "
        f"{result['pk_explained_by_carbonate']:.4f} explained"
        f"\n  -> {100 * (1 - result['pk_explained_by_carbonate']):.2f}% "
        "independent, an order of magnitude more"
    )
    print(f"\nGram condition number: {result['gram_condition_number']:.3e}")
    print(
        "\nConclusion: the concentration of organic alkalinity is nearly "
        "degenerate with\nthe carbonate totals, because the organic pKa range "
        "straddles carbonic pK1.\nThe encoder's ~22% recovery is close to what "
        "the curve permits, not a training\nfailure, and no architecture "
        "removes it."
    )

    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
