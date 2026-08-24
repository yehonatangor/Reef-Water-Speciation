# Training data

No physical instrument exists, so every curve is simulated. The ground truth is therefore exact, and the quantity of data is limited only by compute.

## What varies

Each sample draws independently:

| Quantity | Range |
|---|---|
| Salinity | $25$–$40$ |
| Temperature | $22$–$28^\circ\mathrm{C}$ |
| Target pH, total scale | $7.6$–$8.4$ |
| Dissolved inorganic carbon | $1000$–$5000\ \mu\mathrm{mol\,kg^{-1}}$ |
| Sample mass | $0.0145$–$0.0155\ \mathrm{kg}$ |
| Total boron | from salinity, scaled by $0.85$–$1.15$ |
| Phosphate | zero with probability $0.75$, else $10^{-6}$–$10^{-4.5}\ \mathrm{mol\,kg^{-1}}$ |
| Silicate | zero with probability $0.75$, else $10^{-6}$–$10^{-4}\ \mathrm{mol\,kg^{-1}}$ |
| Organic acid total | zero with probability $0.5$, else $0$–$150\ \mu\mathrm{mol\,kg^{-1}}$ |
| Organic $\mathrm{p}K_a$ | $4.0$–$7.0$ |

Instrument nuisances are drawn per sample from the grade in use. A model trained with them fixed learns nothing about estimating them and does not transfer.

## Drawing the composition

Alkalinity and carbon are not drawn independently. Doing so produces thermodynamically impossible combinations, and discarding the failures biases the set toward easy compositions.

Instead the target pH and the carbon total are drawn from the ranges above, and alkalinity is derived from them through the forward chemistry. Every sample is then feasible by construction, and the distribution over pH is controlled directly rather than emerging from a rejection step.

The boron jitter is wider than analytical uncertainty in the boron-to-salinity ratio would justify. It stands in for genuine variation in that ratio between water bodies, which is not well constrained for aquarium water.

Organic matter is added on top of the derived alkalinity rather than displacing carbonate, which is the physically correct relationship: an organic acid contributes proton acceptors of its own.

## Organic acid ranges

Reef aquarium water typically carries $1$–$2\ \mathrm{mg\,C\,L^{-1}}$ of total organic carbon, roughly $80$–$170\ \mu\mathrm{mol\,C\,kg^{-1}}$, which sits inside the coastal range. The *concentration* therefore transfers from the literature.

The *composition* does not. Coastal and estuarine dissolved organic matter is dominated by terrestrial humic and fulvic acids; aquarium organics are produced in place: coral mucus, algal exudates, dissolved protein, with different functional-group chemistry at the same carbon concentration.

Reef systems also have a source with no coastal analogue: deliberate organic carbon dosing with vinegar, vodka or biopellets. Acetate has $\mathrm{p}K_a = 4.76$, directly in the region where the carbonate endpoint is determined.

The ranges above are therefore deliberately broad rather than centred on any literature value. The intent is a model robust across the uncertainty rather than tuned to one guess. These are assumptions, not measurements.

## Labels

Labels are computed from the drawn composition, not read off the simulated curve.

Reading them off the curve would make the label a function of the same solver that produced the input, so any solver bias would cancel between input and target and the model would appear more accurate than it is.

## Avoided shortcuts

The clean curve is never an input. It is stored for diagnostics only. A noise-free curve is a bijection of the parameters, so supplying it hands the model the answer.

True salinity and temperature are never inputs. The measured values are, with their uncertainty, because those are what an instrument reports.

Nutrients are passed to both simulator and label calculation. Omitting them from either side leaves the two solving different problems.

## Scaling

The curve, the environment vector and the labels are standardised before training, and the statistics are computed on the training split alone. Fitting them on the whole dataset would let the held-out set influence the transform applied to itself, which inflates the reported accuracy by an amount that is small, real, and impossible to detect after the fact.

Inputs are centred and divided by their standard deviation. Labels are divided by their mean rather than their standard deviation, which puts every target near $1.0$ on average and makes a squared error on alkalinity comparable to one on carbonate despite an order of magnitude between their physical sizes. Any divisor is floored away from zero, so a target with no spread in the training split cannot produce an infinite scaled value.

The statistics are saved alongside the model. Applying a model with a different transform from the one it was trained under produces predictions that are wrong by a fixed factor and otherwise look reasonable, which is a failure worth making impossible rather than merely unlikely.

## Seeds

Four random streams are used, and they are kept distinct on purpose.

| Stream | Role |
|---|---|
| Dataset seed | draws every composition and instrument state |
| Control seed | draws the organic-free control set |
| Split seed | divides a dataset into training, validation and held-out parts |
| Training seed | weight initialisation and batch order |

1. The control seed differs from the dataset seed.

    Sampling is deterministic, so reusing the dataset seed with organic switched off would reproduce the same compositions and instrument states, and the control would consist entirely of water the model had trained on.

2. The split seed is held fixed across training runs.

    Varying it alongside the training seed would confound two sources of variation: a difference between runs could then be initialisation or a different held-out set, with no way to separate them.

3. The training seed is the only one varied deliberately.

    Training is stochastic through initialisation, batch order and the epoch at which early stopping fires. A single run is one sample from that distribution, so results are reported across several runs with the spread stated.

## Generation and storage

Data generation never loads the machine learning framework. The numerical libraries used by the simulator and by the training framework do not coexist reliably in one interpreter on Windows, and separating the two processes removes the failure entirely.

Generation is chunked and resumable. Each chunk is seeded by its index, so the assembled dataset is identical regardless of how many chunks any one process completed. An interruption costs one chunk rather than the run.

Files record their provenance: package version, interpreter version, platform, every seed and the full command line. A dataset whose column count does not match the current schema is rejected on load rather than silently reinterpreted.

## Reproducing

```bash
python scripts/generate_dataset.py --instrument hobbyist --n-samples 16000
```

Sixteen thousand curves of two hundred points takes a few hours. The default seed is fixed, so the same command produces the same data on any machine.
