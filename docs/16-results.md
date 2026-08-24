# Results

All figures in $\mu\mathrm{mol\,kg^{-1}}$ unless stated. Hobbyist instrument grade, salinity and temperature drawn across their full sampled ranges, evaluated on a held-out split of 2400 curves never seen in training.

Learned results are the median across three training runs differing only in initialisation seed, with the range in brackets. The held-out split is identical across those runs, so the spread measures training variance alone.

## Accuracy floor

The bound below which no estimator can go, at $S = 35$, $t = 25\,^\circ\mathrm{C}$, $A_T = 2300$, $C_T = 2000\ \mu\mathrm{mol\,kg^{-1}}$, instrument nuisances marginalised, $40$ points per curve. The noise figure is the total effective pH residual, not the electrode term alone.

| Instrument grade | Effective noise, pH | $A_T$ bound | Measured at the same point count | Headroom |
|---|---|---|---|---|
| Laboratory | $0.0088$ | $3.53$ | $3.5$ | $0.99\times$ |
| Hobbyist | $0.0349$ | $18.41$ | $22.0$ | $1.20\times$ |

The laboratory case is at the floor to within measurement precision. The hobbyist case sits twenty percent above it, and that twenty percent is the entire margin available to any method whatsoever.

The measured column here is a $40$-point simulation matched to the bound. The benchmark in the next section uses $100$ points and is therefore not directly comparable to this bound: more points move the bound down as well as the measurement.

## Classical inverse

Least squares against the exact forward model, fitting calibration offset and Nernstian slope alongside the two totals. One hundred curves at one hundred points, on organic-free water.

| Instrument | $A_T$ bias | $A_T$ sd | $A_T$ MAE | $A_T$ median | $\mathrm{HCO_3^-}$ MAE |
|---|---|---|---|---|---|
| Laboratory | $-0.39$ | $5.30$ | $3.95$ | $3.19$ | $23.95$ |
| Hobbyist | $+4.82$ | $30.76$ | $23.03$ | $17.86$ | $32.26$ |

## Learned inverse

| Quantity | bias | sd | MAE | median | max |
|---|---|---|---|---|---|
| Alkalinity | $-1.69$ | $35.02$ | $24.23$ | $17.75$ | $490.81$ |
| Dissolved inorganic carbon | $-4.87$ | $51.58$ | $39.93$ | $32.50$ | $323.15$ |
| Bicarbonate | $-5.85$ | $52.53$ | $40.67$ | $32.96$ | $254.00$ |
| Carbonate | $+1.21$ | $16.01$ | $11.63$ | $8.49$ | $98.11$ |
| Borate | $+0.54$ | $7.54$ | $5.99$ | $5.03$ | $28.23$ |

Ranges across the three seeds, for the two headline quantities:

| | median | range |
|---|---|---|
| Alkalinity MAE | $24.23$ | $23.98$–$25.00$ |
| Bicarbonate MAE | $40.67$ | $40.28$–$41.44$ |

**Run-to-run variation is under five percent.** Three independent initialisations converge to the same accuracy, so the figures above are properties of the architecture rather than of one fortunate run.

### Against the classical inverse

The two tables above are not comparable to each other. The classical benchmark runs $100$ curves at $100$ points on organic-free water; the learned figures are $2400$ curves at $200$ points on water that carries organic acid half the time. Reading one against the other compares two estimators on two different problems.

A matched comparison requires both estimators on the same curves. That is what the organic study below provides: identical inputs, identical point count, both methods run in the same pass. On its mixed population of $600$ curves, half of which carry organic acid:

| | learned | classical |
|---|---|---|
| Alkalinity bias | $-1.92$ | $-4.30$ |
| Alkalinity MAE | $23.79$ | $23.35$ |
| Alkalinity max | $370.16$ | $99.00$ |
| Bicarbonate MAE | $41.27$ | $40.61$ |

**On alkalinity the two methods are within two percent of each other**, $23.79$ against $23.35$. Bicarbonate is within two percent as well, $41.27$ against $40.61$.

The difference is in the tail, not the typical case. The learned model's largest alkalinity error is $370.16$ against a mean absolute error of $23.79$, a ratio of sixteen; the classical fit's ratio is four. A small number of curves are fitted badly and they widen the learned distribution without moving its centre.

That is the precise statement: the typical curve is solved about as well as least squares solves it, a minority of hard curves is not, and the residual gap on bicarbonate is the error-correlation gap discussed below.

## Inference speed

| | per curve | range across runs |
|---|---|---|
| Learned inverse | $8.58\ \mathrm{ms}$ | $2.21$–$8.58$ |
| Classical inverse | $3503\ \mathrm{ms}$ | $1696$–$3538$ |

**Two to three orders of magnitude faster.** Measured speedups across all runs span $408\times$ to $767\times$.

Neither figure is stable to better than a factor of a few on identical code, so the ratio should be read as an order of magnitude rather than a multiplier. The classical timing depends on how many iterations each curve happens to need, and both sides move with machine load.

The claim that survives every measurement: the learned inverse costs single-digit milliseconds and the classical inverse costs seconds. That is the difference between an embedded controller and a laptop, which is the point.

## Parameter recovery

The ratio of true standard deviation to residual standard deviation. Near one means the encoder is predicting the population mean and the curve does not identify that parameter.

| Parameter | recovery | range |
|---|---|---|
| Nernstian slope | $2.86$ | $2.86$–$2.93$ |
| Calibration offset | $1.74$ | $1.73$–$1.81$ |
| Organic total | $1.14$ | $1.13$–$1.14$ |
| Organic $\mathrm{p}K_a$ | $1.07$ | $1.07$–$1.07$ |
| Pump scale | $0.98$ | $0.97$–$0.98$ |

**The calibration parameters are recovered well**, at $2.86$ and $1.74$. That is the mechanism the whole inexpensive-instrument argument depends on: the offset and slope of an uncalibrated electrode are estimable from the curve itself.

**Pump scale is not recovered at all.** It sits below one, meaning the encoder does no better than guessing the population mean.

That result agrees with the accuracy floor, reached by an independent route. Pump error acts on the horizontal axis rather than the vertical, so it is invisible to any measure of pH scatter and equally invisible to a network reading the curve. It is simultaneously the largest single lever on achievable accuracy, since improving delivery from $0.8\%$ to $0.1\%$ moves the bound from $18.41$ to $4.20$, and the one quantity the measurement cannot determine.

**A quantity that cannot be inferred must be fixed in hardware.** Gravimetric titrant delivery removes the term entirely.

**Organic recovery is $1.14$**, barely above the population mean, which is what the identifiability analysis predicts: at the centre of the sampled range under two percent of the organic signature is separable from the carbonate totals. The organic $\mathrm{p}K_a$ at $1.07$ is expected, since it has no effect whatever when the organic total is zero.

## Joint error structure

Carbonate speciation depends on the difference between the two totals, so the correlation between their errors matters more than either alone.

| | $\mathrm{corr}(\Delta A_T, \Delta C_T)$ | $\mathrm{sd}(\Delta A_T - \Delta C_T)$ |
|---|---|---|
| Learned | $0.547$ | $43.40$ |

Range across seeds: correlation $0.534$–$0.585$, difference error $43.10$–$43.73$. Training ran for $71$, $79$ and $96$ epochs before early stopping.

Least squares earns a high correlation structurally, because a single optimisation couples both totals through one residual and an error in one is compensated by a matched error in the other. A feed-forward encoder predicts the two independently and has no such coupling.

This is the mechanism behind the bicarbonate gap: the network matches the classical fit on each total individually and loses on their difference, which is the quantity speciation depends on.

## Organic water against the matched control

Three populations. The control is generated with organic disabled and an independent seed, evaluated but never trained on. Bicarbonate error, $\mu\mathrm{mol\,kg^{-1}}$.

| Population | Estimator | bias | sd | MAE | max |
|---|---|---|---|---|---|
| Organic-free control, $n = 600$ | classical | $+9.75$ | $23.36$ | $\mathbf{19.27}$ | $84.5$ |
| | learned | $-26.09$ | $37.93$ | $35.68$ | $222.3$ |
| Mixed, organic present, $n = 600$ | classical | $+34.87$ | $43.83$ | $40.61$ | $195.4$ |
| | learned | $\mathbf{-4.89}$ | $53.74$ | $41.27$ | $254.0$ |
| Organic-carrying only, $n = 313$ | classical | $+57.68$ | $44.55$ | $59.29$ | $195.4$ |
| | learned | $+16.92$ | $53.49$ | $\mathbf{43.11}$ | $254.0$ |

Read the three rows in order, because the first constrains what the others are allowed to mean.

### On organic-free water the learned model is worse

MAE $35.68$ against $19.27$, and a bias of $-26.09$ where the classical fit has $+9.75$.

The cause is direct: the encoder predicts a mean organic content of $32.4\ \mu\mathrm{mol\,kg^{-1}}$ on water containing exactly zero, never falling below $7.1$. The chemistry decoder then subtracts that phantom organic contribution from the carbonate budget, and bicarbonate comes out low by very nearly the amount implied.

**This is the cost of carrying the organic term**, and it is paid on every sample that contains no organic acid.

### On organic-carrying water the sign of the bias reverses

The classical bias climbs from $+9.75$ on clean water to $+34.87$ on mixed water to $+57.68$ on the organic-carrying subset. It has no organic term, so proton acceptance by organic acids is attributed to the carbonate system, and the error grows with organic content.

The learned bias goes the other way: $-26.09$, $-4.89$, $+16.92$ across the same three populations. Near-zero on the mixed population is the interesting entry: averaged over water that half contains organic acid, the two failure modes cancel.

On the organic-carrying subset the learned model wins on MAE, $43.11$ against $59.29$, a reduction of $27\%$.

### What the trade actually is

The learned model does not measure organic content well. On the organic-carrying subset it predicts a mean of $46.8$ against a true $70.9\ \mu\mathrm{mol\,kg^{-1}}$, and its predictions shrink toward the middle of the range from both directions.

It does not need to. Allocating proton acceptance to something other than carbonate is enough to remove most of the classical bias, and that is what the numbers show.

The cost is a larger scatter, $53.5$ against $44.6$, and worse performance on clean water. **Whether the trade is worth taking depends on how often the water actually contains organic acid.** For reef aquarium water, with organic carbon dosing in routine use, that is most of the time.

The control row is not optional. A model carrying organic parameters is expected to be worse on water containing none, and reporting only the favourable population would overstate the effect substantially.
