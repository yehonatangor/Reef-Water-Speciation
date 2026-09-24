# Reef Water

Seawater carbonate speciation from an acid titration: a verified forward model, a classical inverse, an information-theoretic accuracy floor, and a learned inverse that runs in one forward pass.

Every equilibrium constant is traceable to its primary source and agrees with an independent implementation of the same equations to floating-point round-off.

## Install

```bash
python -m venv .venv
source .venv/Scripts/activate     # Git Bash on Windows
pip install -e ".[dev,ml,notebook]"
```

Requires Python 3.10 or later. NumPy and SciPy are the only hard dependencies. The `ml` extra adds TensorFlow, `notebook` adds matplotlib and Jupyter for the figures, and `validate` adds PyCO2SYS for the independent cross-check.

## Quick start

```python
import cleaned_reef_water as crw

ph = crw.ph_from_dic_alkalinity(dic=2000e-6, alkalinity=2300e-6,
                                salinity=35.0, t_c=25.0)
# 8.0459

constants = crw.constants_at(salinity=35.0, t_c=25.0)
species = crw.speciate(ph, constants, dic=2000e-6)
# hco3 1775.4, co3 213.4, boh4 91.2 umol/kg
```

Simulating a measurement and inverting it:

```python
curve = crw.simulate_titration(alkalinity=2300e-6, dic=2000e-6,
                               salinity=35.0, t_c=25.0,
                               sample_mass_kg=0.015, n_points=200)

from cleaned_reef_water import LeastSquaresInverse, NuisanceModel

result = LeastSquaresInverse(
    sample_mass_kg=0.015,
    nuisance=NuisanceModel.OFFSET_AND_SLOPE,
).fit(curve.titrant_mass, curve.ph_total, 35.0, 25.0)
```

Everything takes and returns mol per kilogram of solution. Multiply by 1e6 for the micromole figures used throughout the documentation.

## Conventions

| | |
|---|---|
| pH scale | total hydrogen ion |
| Concentration | mol per kg solution |
| Temperature | degrees Celsius at the interface |
| Pressure | 1 atm |
| Reported quantities | micromole per kg |

## Two inverses

Least squares against the exact forward model. No training, no dataset. Fits the two carbonate totals plus calibration offset and Nernstian slope. This is the reference every other method is measured against, and it sits at the information floor on laboratory-grade data.

Physics-constrained autoencoder. A convolutional encoder predicts seven physical parameters, which pass through the exact chemistry implemented differentiably. The decoder has no trainable weights, so the output is a physically valid state by construction. One forward pass replaces the optimisation, and it carries an organic alkalinity term the classical fit lacks.

## Reproducing

```bash
make check          # lint, types, tests, documented numbers
make baseline       # classical benchmark, ~10 min
make limits         # accuracy floor, seconds
make data           # synthetic datasets, hours, resumable
make autoencoder    # train the recommended model
make compare        # learned against classical on organic water
```

Each stage writes JSON recording the package version, interpreter, platform, seeds and command line.

`REPRODUCE.md` is the full runbook: every command in cost order, what each one produces, which document it fills, and how to read the saved results back.

Data generation and training run in separate processes deliberately: the numerical libraries used by the optimiser and by the training framework do not tear down reliably in one interpreter, and the symptom is a core dump after every test has passed.

Long-running stages are chunked and resumable. If one crashes, re-running continues from the chunks already written:

```bash
bash scripts/resume.sh scripts/generate_dataset.py --instrument hobbyist
```

## Verifying the documented numbers

```bash
python scripts/verify_claims.py
```

Recomputes every documented value that can be checked in seconds, printing where each is stated, where it came from, and whether it still holds. Exits non-zero on disagreement, so it works as a release gate.

## Testing

```bash
python -m pytest -m "not tensorflow"
python -m pytest -m tensorflow
```

Two invocations, for the reason above. Coverage includes agreement with an independent implementation across a grid of salinity and temperature, mass conservation to 1e-14, round-trip pH scale conversion, and exactness of the chunked benchmark against a single-process run.

## Documentation

Read in order:

| | |
|---|---|
| 01 | Overview |
| 02 | Background |
| 03 | Scales and units |
| 04 | Bulk seawater properties |
| 05 | Equilibrium constants |
| 06 | pH scale conversion |
| 07 | Total alkalinity |
| 08 | The titration forward model |
| 09 | Numerical method |
| 10 | Information limits |
| 11 | Instrument model |
| 12 | The classical inverse |
| 13 | The learned inverse |
| 14 | Training data |
| 15 | Organic alkalinity |
| 16 | Results |
| 17 | Verification |
| 18 | Titrant selection |
| 19 | Noise budget |
| 20 | Future work |
| 21 | Doing it again |

## Figures

```bash
pip install -e ".[notebook]"
jupyter lab notebooks/curves.ipynb
```

Regenerates every figure from the forward model and the committed result files. The chemistry sections need only the package; the results sections read `results/*.json` and skip themselves cleanly if those files are absent, so the notebook runs from a bare checkout.

## Limitations

Everything is simulated. No physical instrument exists, and the noise model is built from component specifications and physical reasoning rather than from a calibration campaign against real hardware.

Pressure corrections are absent; all constants are at 1 atm.

Organic matter is modelled as a single lumped monoprotic acid standing in for a mixture spread over many dissociation constants. The identifiability conclusion is robust to that approximation; the achievable correction is conditional on it.

Recovering organic content is bounded by the chemistry rather than by the model: at the centre of the plausible range, under two percent of the organic signature is separable from the carbonate totals.

## Licence

MIT — see [`LICENSE`](LICENSE).


