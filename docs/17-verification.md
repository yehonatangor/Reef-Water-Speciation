# Verification

Every number in this documentation falls into one of three categories, and the category is stated wherever the number appears.

## Independently cross-checked

Nine constants are compared against a separately maintained implementation of the same published equations, across a grid of twenty salinity and temperature combinations.

| Constant | Worst relative difference |
|---|---|
| $K_1$ | $0$ |
| $K_2$ | $0$ |
| $K_B$ | $0$ |
| $K_W$ | $0$ |
| $K_S$ | $2.00\times10^{-15}$ |
| $K_F$ | $4.44\times10^{-16}$ |
| $K_0$ | $0$ |
| **All** | $\mathbf{2.00\times10^{-15}}$ |

Calcite and aragonite solubility are compared separately and agree to the same tolerance.

Five of the seven agree **bit-for-bit**. The two that do not differ in the last representable digit, which is the expected consequence of evaluating the same expression with operations in a different order.

Two implementations agreeing to fifteen digits do not do so by accident. A transcription error in any coefficient would be visible immediately.

Phosphate and silicate are excluded from this comparison because the reference implementation reports them on a different pH scale; agreement there would test a convention rather than the chemistry.

## Checked against published values

Six constants reproduce the printed check values of the standard reference at $S = 35$, $t = 25^\circ\mathrm{C}$:

| Quantity | Value |
|---|---|
| $\ln K_0$ | $-3.5617$ |
| $\ln K_S$ | $-2.30$ |
| $\ln K_B$ | $-19.7964$ |
| $\ln K_F$ | $-6.09$ |
| $\log_{10} K_1$ | $-5.8472$ |
| $\log_{10} K_2$ | $-8.9660$ |

Bulk properties likewise:

| Quantity | Value |
|---|---|
| Density | $1023.343\ \mathrm{kg\,m^{-3}}$ |
| Ionic strength | $0.72276\ \mathrm{mol\,kg\text{-}H_2O^{-1}}$ |
| Total borate | $415.76\ \mu\mathrm{mol\,kg^{-1}}$ |
| Free to total scale factor | $\log_{10} = 0.1077$ |

Water, phosphate and silicate are verified against their source parameterisation only, since the exact pH-scale conversion used here departs deliberately from the additive approximation the reference recommends. The two differ by $0.73\%$.

## Recomputed on demand

Seventeen documented numbers are recomputed from the code in a few seconds, each reported alongside the section that states it and the source it came from:

```bash
python scripts/verify_claims.py
```

The command exits non-zero if any documented value no longer matches what the code produces, so it functions as a release gate as well as a reading aid.

## Requiring a run

Three categories of number cannot be checked in seconds and are therefore outside the automatic gate:

| Claim | Command | Cost |
|---|---|---|
| Classical benchmark | `python scripts/run_baseline.py` | ~10 min |
| Accuracy floor | `python scripts/information_limits.py --verify` | seconds |
| Learned inverse results | `python scripts/train_autoencoder.py` | hours, needs a dataset |

The classical benchmark in particular must be regenerated after any change to the sampler, the forward model or the optimiser. Nothing enforces this automatically.

## Determinism

Every generated dataset is seeded, so the same command produces byte-identical data on any machine. Data generation is chunked, and each chunk is seeded by its index rather than by its position in the run, so the assembled result does not depend on how the work was divided across processes.

Result files record the package version, interpreter version, platform, every seed, and the full command line that produced them.

## Reproducing everything

In dependency order, from a clean checkout:

```bash
python -m ruff check .
python -m mypy src/
python -m pytest -m "not tensorflow"
python -m pytest -m tensorflow
python scripts/verify_claims.py

python scripts/information_limits.py --verify --output results/limits.json
python scripts/organic_identifiability.py --sweep --sweep-states
python scripts/run_baseline.py --n-curves 100 --n-points 100

python scripts/generate_dataset.py --instrument hobbyist --n-samples 16000
python scripts/train_autoencoder.py --dataset data/dataset_hobbyist.npz --output-dir results
python scripts/compare_organic.py --split-seed 42
```

The two test invocations run in separate processes deliberately: the numerical libraries used by the optimiser and by the training framework do not tear down reliably in one interpreter, and the symptom is a core dump after every test has passed.
