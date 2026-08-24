#!/usr/bin/env python3
r"""Recompute every documented number that can be checked in seconds."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Part of `make check`, so it must run from a bare checkout as pytest does.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cleaned_reef_water import constants_at, simulate_titration
from cleaned_reef_water.equilibria import (
    CarbonicFormulation,
    k_bisulfate_free,
    k_borate_total,
    k_carbonic_total,
    k_co2_solubility,
    k_fluoride_total,
    k_water_sws,
    k_water_total,
)
from cleaned_reef_water.ml.sampling import SamplingRanges
from cleaned_reef_water.scales import PHScale, scale_factor
from cleaned_reef_water.seawater import (
    density_seawater,
    ionic_strength,
    total_borate,
)

S_REF, T_REF = 35.0, 25.0


@dataclass
class Check:
    """One documented number, recomputed."""

    claim: str
    where: str
    source: str
    computed: float
    expected: float
    tolerance: float
    relative: bool = False
    note: str = ""
    _passed: bool = field(init=False, default=False)

    def evaluate(self) -> bool:
        """Compare and cache the outcome."""
        difference = abs(self.computed - self.expected)
        if self.relative:
            difference /= abs(self.expected)
        self._passed = bool(difference <= self.tolerance)
        return self._passed

    @property
    def passed(self) -> bool:
        """Whether the recomputed value matches."""
        return self._passed


def chemistry_checks() -> list[Check]:
    """Constants and bulk properties, against their published values."""
    k1, k2 = k_carbonic_total(S_REF, T_REF, CarbonicFormulation.LUEKER2000)
    exact = float(k_water_total(S_REF, T_REF))
    shortcut = float(k_water_sws(S_REF, T_REF)) * float(np.exp(-0.015))

    return [
        Check(
            "pK1 = 5.8472 at S=35, T=25",
            "05_equilb_const.md",
            "Guide ch.5, Lueker et al. (2000)",
            float(-np.log10(k1)), 5.8472, 1e-4,
        ),
        Check(
            "pK2 = 8.9660 at S=35, T=25",
            "05_equilb_const.md",
            "Guide ch.5, Lueker et al. (2000)",
            float(-np.log10(k2)), 8.9660, 1e-4,
        ),
        Check(
            "ln K0 = -3.5617",
            "05_equilb_const.md",
            "Guide ch.5, Weiss (1974)",
            float(np.log(k_co2_solubility(S_REF, T_REF))), -3.5617, 1e-4,
        ),
        Check(
            "ln KB = -19.7964",
            "05_equilb_const.md",
            "Guide ch.5, Dickson (1990b)",
            float(np.log(k_borate_total(S_REF, T_REF))), -19.7964, 1e-4,
        ),
        Check(
            "ln KS = -2.30",
            "05_equilb_const.md",
            "Guide ch.5, Dickson (1990a)",
            float(np.log(k_bisulfate_free(S_REF, T_REF))), -2.30, 1e-2,
        ),
        Check(
            "ln KF = -6.09",
            "05_equilb_const.md",
            "Guide ch.5",
            float(np.log(k_fluoride_total(S_REF, T_REF))), -6.09, 1e-2,
        ),
        Check(
            "ln KW(SWS) = -30.4187",
            "05_equilb_const.md",
            "Millero (1995), as published",
            float(np.log(k_water_sws(S_REF, T_REF))), -30.4187, 1e-3,
        ),
        Check(
            "density = 1023.343 kg/m3",
            "04_bulk_swtr_prp.md",
            "Guide ch.5 §4.2, Millero & Poisson (1981)",
            float(density_seawater(S_REF, T_REF)), 1023.343, 1e-3,
        ),
        Check(
            "ionic strength = 0.72276 mol/kg-H2O",
            "04_bulk_swtr_prp.md",
            "Dickson et al. (2007)",
            float(ionic_strength(S_REF)), 0.72276, 1e-5,
        ),
        Check(
            "total borate = 415.76 umol/kg (Uppstrom 1974)",
            "04_bulk_swtr_prp.md",
            "Uppstrom (1974)",
            float(total_borate(S_REF)) * 1e6, 415.76, 0.01,
        ),
        Check(
            "free->total scale factor: log10 = 0.1077",
            "06_pH_conv.md",
            "derived from KS, KF and the totals",
            float(np.log10(scale_factor(
                S_REF, T_REF, PHScale.FREE, PHScale.TOTAL))),
            0.1077, 1e-3,
        ),
        Check(
            "exact SWS->Total conversion differs from the Guide's "
            "0.015 shortcut by 0.73%",
            "06_pH_conv.md",
            "this library's deliberate departure",
            float(exact / shortcut), float(np.exp(-0.00729)), 1e-3, relative=True,
            note="the shortcut is an approximation; the exact form is used here",
        ),
    ]


def cross_implementation_checks() -> list[Check]:
    """Agreement with PyCO2SYS, if it is installed."""
    try:
        from PyCO2SYS.equilibria import p1atm
    except ImportError:
        return []

    salinities, temperatures = (20.0, 30.0, 35.0, 40.0), (5.0, 15.0, 25.0, 35.0)
    worst = 0.0
    for salinity in salinities:
        for t_c in temperatures:
            t_k = t_c + 273.15
            ours = k_carbonic_total(
                salinity, t_c, CarbonicFormulation.LUEKER2000
            )
            theirs = p1atm.kH2CO3_TOT_LDK00(t_k, salinity)
            for a, b in zip(ours, theirs, strict=True):
                worst = max(worst, abs(float(a) / float(b) - 1.0))
            for ours_f, theirs_f in (
                (k_borate_total(salinity, t_c),
                 p1atm.kBOH3_TOT_D90b(t_k, salinity)),
                (k_water_sws(salinity, t_c),
                 p1atm.kH2O_SWS_M95(t_k, salinity)),
                (k_bisulfate_free(salinity, t_c),
                 p1atm.kHSO4_FREE_D90a(t_k, salinity)),
                (k_co2_solubility(salinity, t_c),
                 p1atm.kCO2_W74(t_k, salinity)),
            ):
                worst = max(worst, abs(float(ours_f) / float(theirs_f) - 1.0))

    return [
        Check(
            "agreement with PyCO2SYS is exact to float64 round-off",
            "17_verification.md",
            f"PyCO2SYS v1.8.3, {len(salinities) * len(temperatures)} (S,T) points",
            worst, 0.0, 1e-12,
            note="native pH scales; see tests/test_pyco2sys_agreement.py",
        )
    ]


def identifiability_checks() -> list[Check]:
    """Recompute the organic-alkalinity degeneracy from the sensitivity Jacobian."""
    alkalinity, dic, organic = 2.30e-3, 2.00e-3, 7.5e-5
    step, n_points = 1e-6, 200   # matches organic_identifiability.py exactly

    def curve(a=alkalinity, c=dic, o=organic, pk=5.5):
        return simulate_titration(
            a, c, S_REF, T_REF, n_points=n_points,
            total_organic=o, pk_organic=pk,
        ).ph_total

    def explained(pk: float) -> float:
        basis = np.stack([
            (curve(a=alkalinity + step, pk=pk)
             - curve(a=alkalinity - step, pk=pk)) / 2.0,
            (curve(c=dic + step, pk=pk) - curve(c=dic - step, pk=pk)) / 2.0,
        ])
        target = (curve(o=organic + step, pk=pk)
                  - curve(o=organic - step, pk=pk)) / 2.0
        coefficients, *_ = np.linalg.lstsq(basis.T, target, rcond=None)
        residual = target - basis.T @ coefficients
        return 1.0 - float(residual @ residual) / float(target @ target)

    pk1 = float(-np.log10(float(constants_at(S_REF, T_REF).k1)))
    low, high = SamplingRanges().pk_organic

    return [
        Check(
            "carbonic pK1 (5.847) lies inside the organic pKa draw range",
            "15_organic_alkalinity.md",
            "computed from the library; range from SamplingRanges",
            float(low < pk1 < high), 1.0, 0.0,
            note=f"pK1 = {pk1:.3f}, organic pKa drawn from {low}-{high}",
        ),
        Check(
            "98.76% of the organic signature is explained by (A_T, C_T)",
            "15_organic_alkalinity.md",
            "least squares on the sensitivity Jacobian",
            100.0 * explained(5.5), 98.76, 0.5,
        ),
        Check(
            "at pKa = carbonic pK1 the degeneracy is near-total (0.04% left)",
            "15_organic_alkalinity.md",
            "same, evaluated at pKa 5.85",
            100.0 * (1.0 - explained(5.85)), 0.04, 0.05,
        ),
        Check(
            "at pKa 4.0 organic becomes measurably identifiable (29.11%)",
            "15_organic_alkalinity.md",
            "same, evaluated at pKa 4.0",
            100.0 * (1.0 - explained(4.0)), 29.11, 1.0,
        ),
    ]


#: Claims that cannot be recomputed in seconds, with what does reproduce them.
EXPENSIVE = (
    ("Cramer-Rao bounds; classical inverse is near-optimal",
     "10_info_limits.md",
     "python scripts/information_limits.py --verify", "seconds"),
    ("Classical baseline accuracy",
     "12_classical_inverse.md",
     "python scripts/run_baseline.py", "~10 min"),
    ("Trained model accuracy and inference speed",
     "16_results.md",
     "python scripts/train_autoencoder.py --dataset data/dataset_hobbyist.npz",
     "hours, needs data"),
    ("Organic: learned vs classical, both populations",
     "16_results.md",
     "make compare", "~10 min, needs a trained model"),
    ("Instrument noise budget, term by term",
     "19_noise_budget.md",
     "python scripts/noise_budget.py --instrument laboratory", "~2 min"),
    ("Linear regression on the summary features beats the raw CNN",
     "13_learned_inverse.md",
     "python scripts/feature_baseline.py --instrument hobbyist",
     "~3 min; CNN side needs train_cnn.py"),
)


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true",
                        help="only report failures and the summary")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Recompute every fast claim and report."""
    args = parse_args(argv)

    groups = [
        ("Chemistry: constants and bulk properties", chemistry_checks()),
        ("Cross-implementation agreement", cross_implementation_checks()),
        ("Organic alkalinity identifiability", identifiability_checks()),
    ]

    failures = 0
    for title, checks in groups:
        if not checks:
            print(f"\n{title}\n  skipped (PyCO2SYS not installed)")
            continue
        print(f"\n{title}")
        print("-" * 78)
        for check in checks:
            ok = check.evaluate()
            failures += not ok
            if args.quiet and ok:
                continue
            mark = "ok  " if ok else "FAIL"
            print(f"  [{mark}] {check.claim}")
            print(f"         stated in : {check.where}")
            print(f"         source    : {check.source}")
            print(f"         computed  : {check.computed:.6g}"
                  f"   expected: {check.expected:.6g}")
            if check.note:
                print(f"         note      : {check.note}")

    print("\n\nClaims that need a longer run")
    print("-" * 78)
    for claim, where, command, cost in EXPENSIVE:
        print(f"  {claim}\n    stated in : {where}\n    reproduce : {command}"
              f"\n    cost      : {cost}")

    total = sum(len(c) for _, c in groups)
    print("\n" + "=" * 78)
    if failures:
        print(f"{failures} of {total} fast checks FAILED -- the documentation "
              "and the code disagree.")
        return 1
    print(f"All {total} fast checks passed: every recomputable number in the "
          "documentation\nstill matches what the code produces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
