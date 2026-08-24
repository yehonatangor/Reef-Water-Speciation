# The classical inverse

Recovering composition from a measured curve, by least squares against the exact forward model.

## Formulation

The forward model maps a composition and an instrument state to a curve. The inverse minimises the sum of squared pH residuals over the parameters that are unknown:

$$\hat{\boldsymbol\theta} = \arg\min_{\boldsymbol\theta} \sum_{i=1}^{N} \left[\mathrm{pH}_i^{\mathrm{meas}} - \mathrm{pH}^{\mathrm{model}}(m_i;\boldsymbol\theta)\right]^{2}$$

with $\boldsymbol\theta = (A_T,\,C_T,\,\delta,\,s)$. Speciation follows analytically from the fitted totals; it is not fitted separately, which would permit a speciation inconsistent with the alkalinity that produced it.

The optimisation is Levenberg–Marquardt on the residual vector, using the analytic forward model at every evaluation.

## Which nuisances to fit

Three configurations are meaningful.

**Neither offset nor slope.** Assumes perfect calibration. Included only as a control: it demonstrates how much of the error on inexpensive hardware is calibration rather than chemistry.

**Offset only.** Recovers alkalinity well, because $A_T$ and $\delta$ are strongly correlated and freeing $\delta$ absorbs the dominant systematic. It degrades speciation, because with only $\delta$ free the fit buys an accurate $A_T$ by displacing error into $C_T$, and speciation depends on both.

**Offset and slope.** The configuration to use. Four parameters against $200$ points remains heavily overdetermined, and it is the only configuration that recovers both totals well enough for the speciation to be meaningful.

A three-parameter fit is not a valid baseline for a speciation comparison.

## Bounds are mandatory

Unbounded, the optimiser escapes into a degenerate basin: it fits a Nernstian slope of three to four times theoretical, reports convergence, and returns alkalinity errors in the thousands of $\mu\mathrm{mol\,kg^{-1}}$. The residual is genuinely small there, because a badly wrong slope combined with badly wrong totals can reproduce a curve.

Physically motivated bounds remove the basin entirely:

$$A_T,\,C_T \in [0.1,\,10]\ \mathrm{mmol\,kg^{-1}}, \qquad \delta \in [-0.5,\,0.5]\ \mathrm{pH}, \qquad s \in [0.8,\,1.2]$$

These are not regularisation. They are statements about what an electrode can physically do, and a fit outside them is not a worse answer but a meaningless one.

## Diagnosing divergence

Mean absolute error conceals this failure mode, because a handful of divergent fits among a hundred inflate the mean while leaving the median untouched. The benchmark therefore reports median and maximum alongside the mean.

A maximum three to six times the mean absolute error indicates tail noise. A maximum orders of magnitude above the median indicates divergence, and the bounds are the fix.

## Benchmark

Six cases, two instrument grades against three nuisance configurations, $100$ independent curves of $100$ points each, at $S = 35$ and $t = 25\,^\circ\mathrm{C}$, on organic-free water. All figures in $\mu\mathrm{mol\,kg^{-1}}$.

The organic-free condition matters when reading these numbers against any other table. Least squares has no organic term, so on water that carries organic acid its bias is substantially worse than shown here.

| Instrument | Nuisances fitted | $A_T$ bias | $A_T$ sd | $A_T$ MAE | $A_T$ median | $A_T$ max | $\mathrm{HCO_3^-}$ MAE |
|---|---|---|---|---|---|---|---|
| Laboratory | none | $+7.05$ | $13.31$ | $11.66$ | $9.38$ | $41.2$ | $24.01$ |
| Laboratory | offset | $-0.79$ | $5.66$ | $4.29$ | $2.99$ | $20.0$ | $26.60$ |
| Laboratory | offset and slope | $-0.39$ | $5.30$ | $\mathbf{3.95}$ | $3.19$ | $18.7$ | $\mathbf{23.95}$ |
| Hobbyist | none | $+124.84$ | $109.48$ | $128.56$ | $115.36$ | $443.3$ | $76.21$ |
| Hobbyist | offset | $+2.64$ | $32.98$ | $25.17$ | $19.27$ | $99.2$ | $93.34$ |
| Hobbyist | offset and slope | $+4.82$ | $30.76$ | $\mathbf{23.03}$ | $17.86$ | $108.7$ | $\mathbf{32.26}$ |

Certified reference material reproducibility for a good alkalinity titration is $\pm 2\ \mu\mathrm{mol\,kg^{-1}}$.

## What the table shows

**Calibration error dominates inexpensive hardware.** Hobbyist alkalinity bias falls from $+124.84$ to $+4.82\ \mu\mathrm{mol\,kg^{-1}}$, a factor of twenty-six, the moment the calibration parameters become free. The limitation is the assumption of calibration, not the electrode.

**Fitting the offset alone improves alkalinity and degrades speciation.** Hobbyist bicarbonate error rises from $76.21$ to $93.34$ when only the offset is free, then falls to $32.26$ once the slope is added, better than the no-nuisance case by a factor of two. With a single calibration parameter the fit buys an accurate total by displacing error into the carbon term, and speciation depends on both.

**Every maximum is three to five times its mean absolute error.** No case diverges.

**The laboratory grade is at the information floor.** Comparing its $3.95\ \mu\mathrm{mol\,kg^{-1}}$ directly against the bound of $3.53$ would be a mistake, because that bound is computed at $40$ points and this table is measured at $100$: both figures move with point count. Evaluated at a matched $40$ points the laboratory grade reaches $3.5$ against a bound of $3.53$, and the hobbyist grade $22.0$ against $18.41$, roughly $20\%$ of headroom, which is the whole margin available to any method.

## Reproducing it

```bash
python scripts/run_baseline.py --n-curves 100 --n-points 100 --seed 42
```

Each curve is seeded independently of run order, so the result does not depend on how the work is divided across processes.
