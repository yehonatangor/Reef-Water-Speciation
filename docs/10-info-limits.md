# Information limits

What accuracy is achievable from a titration curve, and by any method whatsoever.

## The bound

For an unbiased estimator of parameters $\boldsymbol\theta$ from measurements corrupted by noise of covariance $\Sigma$, the covariance of the estimate satisfies

$$\operatorname{cov}(\hat{\boldsymbol\theta}) \;\succeq\; \mathbf{F}^{-1}, \qquad \mathbf{F} = \mathbf{J}^{\top}\Sigma^{-1}\mathbf{J}$$

where $\mathbf{J}_{ij} = \partial\,\mathrm{pH}_i / \partial\theta_j$ is the sensitivity of each measured point to each parameter. The Cramér–Rao bound $\mathbf{F}^{-1}$ is a property of the measurement, not of any algorithm. No estimator of any kind can beat it.

This converts an unanswerable question into an arithmetic one. Rather than asking whether an estimator is good, one asks how close it sits to the floor.

## Construction

The parameter vector is

$$\boldsymbol\theta = (A_T,\; C_T,\; \delta,\; s,\; \kappa)$$

comprising the two carbonate totals and three instrument nuisances: calibration offset $\delta$ in pH, fractional Nernstian slope $s$, and pump delivery scale $\kappa$. The nuisances are unknown at inference and are therefore estimated jointly and **marginalised out** by inverting the full matrix and retaining the block for $(A_T, C_T)$.

Treating a nuisance as known produces a bound far tighter than any real instrument can achieve.

### The noise covariance

$\Sigma$ combines four independent contributions:

| Term | Form |
|---|---|
| Electrode noise | AR(1), correlated between adjacent points |
| Calibration drift | rank-one, constant across the curve |
| ADC quantisation | white, $q/\sqrt{12}$ for step $q$ |
| Titrant delivery | propagated through $\partial\,\mathrm{pH}/\partial m$ |

The correlated term matters. Treating $N$ points as independent improves the bound by $\sqrt{N}$ for free, which is fiction: electrode noise is autocorrelated and a $200$-point curve does not contain $200$ independent measurements.

### Self-consistency

A bound that kept improving without limit as points were added would indicate that the covariance was being treated as diagonal somewhere: with correlated noise, additional points stop contributing independent information, so the bound must flatten. The correlated term is therefore carried explicitly in $\Sigma$ rather than assumed away.

What is asserted on every run is the weaker but directly falsifiable condition that the bound never exceeds the measured error, since no estimator can beat its own bound. A mis-specified covariance would fail it. Sweeping the point count and confirming the flattening is a stronger check and is not part of the released verification.

## Result

At $S = 35$, $t = 25\,^\circ\mathrm{C}$, $A_T = 2300$, $C_T = 2000\ \mu\mathrm{mol\,kg^{-1}}$, $40$ points per curve. The noise figure is the total effective pH residual, combining every mechanism, not the electrode term alone.

The measured column is a simulation at the same point count as the bound. Comparing a bound computed at one point count against a measurement taken at another is not meaningful, since both move together.

| Instrument grade | Effective noise, pH | $A_T$ bound | Measured | Headroom |
|---|---|---|---|---|
| Laboratory | $0.0088$ | $3.53$ | $3.5$ | $0.99\times$ |
| Hobbyist | $0.0349$ | $18.41$ | $22.0$ | $1.20\times$ |

**The laboratory case is at the floor to within measurement precision.** The hobbyist case sits $20\%$ above it, which is the entire headroom available to any method whatsoever.

A learned inverse is therefore to be judged on **matching** these, and one that appears to beat them by a wide margin has acquired information it will not possess at inference time.

## Where the leverage is

Removing one error mechanism at a time, on the hobbyist grade, shows which of them the bound actually depends on.

| Change | Effective noise, pH | $A_T$ bound |
|---|---|---|
| As modelled | $0.0349$ | $18.41$ |
| Junction drift removed | $0.0186$ | $18.42$ |
| Carbon dioxide loss removed | $0.0319$ | $18.38$ |
| Electrode noise halved | $0.0333$ | $17.77$ |
| Electrode noise removed | $0.0328$ | $5.81$ |
| Pump error $0.8\% \to 0.4\%$ | $0.0349$ | $9.77$ |
| Pump error $0.8\% \to 0.1\%$ | $0.0349$ | $4.20$ |

Three findings, and the first is counterintuitive.

**Halving the noise is worth almost nothing; removing it is worth everything.** Electrode noise halved moves the bound by $3\%$; electrode noise removed moves it by a factor of three. The mechanism is not smooth improvement but a threshold: as long as any correlated electrode noise remains, it dominates the terms that would otherwise limit the fit.

**Pump precision is the cheapest available gain.** Improving delivery from $0.8\%$ to $0.1\%$ takes the bound from $18.41$ to $4.20$, a factor of four, without touching the electrode. The effective pH noise does not change at all, because the pump error enters through the horizontal axis rather than the vertical, which makes it invisible to any measure of pH scatter.

**Junction drift and carbon dioxide loss are nearly free.** Removing junction drift halves the apparent pH noise and moves the bound by $0.05\%$. Both are large in the residual and irrelevant to the answer, because both are absorbed by the fitted nuisance parameters.

The practical consequence for an instrument designer is that money spent on the pump buys more than money spent on the probe.

## Species accuracy across the composition range

The bound on each species at the hobbyist grade, over the sampled range of alkalinity and carbon ratio.

| $A_T$ | $C_T/A_T$ | pH | $A_T$ sd | $\mathrm{HCO_3^-}$ sd | $\mathrm{CO_3^{2-}}$ sd | $\mathrm{B(OH)_4^-}$ sd | worst relative |
|---|---|---|---|---|---|---|---|
| $1500$ | $0.80$ | $8.20$ | $12.1$ | $8.9$ | $2.7$ | $1.3$ | $1.5\%$ |
| $1500$ | $0.95$ | $7.60$ | $12.2$ | $11.8$ | $2.3$ | $1.4$ | $3.9\%$ |
| $2300$ | $0.80$ | $8.29$ | $18.4$ | $13.1$ | $4.4$ | $1.3$ | $1.4\%$ |
| $2300$ | $0.95$ | $7.66$ | $18.5$ | $17.8$ | $3.7$ | $1.5$ | $3.6\%$ |
| $3200$ | $0.80$ | $8.34$ | $25.4$ | $17.9$ | $6.4$ | $1.2$ | $1.3\%$ |
| $3200$ | $0.95$ | $7.69$ | $25.6$ | $24.5$ | $5.3$ | $1.5$ | $3.5\%$ |
| $4000$ | $0.80$ | $8.37$ | $31.7$ | $22.2$ | $8.1$ | $1.2$ | $1.3\%$ |
| $4000$ | $0.95$ | $7.71$ | $31.9$ | $30.5$ | $6.7$ | $1.5$ | $3.4\%$ |

The alkalinity bound scales almost exactly with alkalinity itself, so relative accuracy is roughly constant across the range.

Carbon-rich water is harder, and the worst relative figure is always the carbonate one. The mechanism is not that carbonate is measured less precisely: at $A_T = 2300$ its absolute bound actually improves slightly, from $4.4$ to $3.7\ \mu\mathrm{mol\,kg^{-1}}$, as the ratio rises from $0.80$ to $0.95$. What changes is how much carbonate is there to measure. The higher carbon fraction lowers the initial pH from $8.29$ to $7.66$, and carbonate ion falls with it from $317$ to $101\ \mu\mathrm{mol\,kg^{-1}}$. A nearly constant absolute error divided by a concentration three times smaller is the whole of the effect: $4.4/317 = 1.4\%$ against $3.7/101 = 3.7\%$.

Borate is recovered to $1.5\ \mu\mathrm{mol\,kg^{-1}}$ throughout, since it is fixed by salinity rather than fitted.

## Why denoising before fitting cannot help

A decomposition into "clean the curve, then invert it" is bounded above by inverting directly.

A noise-free titration curve is generated by four numbers, two totals and two calibration parameters, so it occupies a four-dimensional manifold in the $200$-dimensional space of possible curves. Least squares against the exact forward model already projects the noisy observation onto that manifold, and for white noise that projection is the maximum-likelihood one.

An intermediate denoising stage must commit to a single clean curve before the fit sees the data. That commitment is irreversible, and any information it discards cannot be recovered downstream. At best it reproduces the projection least squares already performs; at worst it discards signal.

## What a learned inverse offers instead

Not accuracy. The classical fit is already at the floor, so accuracy is unavailable as a target.

**Speed.** The classical inverse solves an optimisation per sample. A network evaluates one forward pass, which is the difference between a laptop and an embedded controller.

**Structure the classical fit lacks.** The least-squares model has no term for organic acids, so on water containing them it is fitting a model that cannot represent the sample. A learned model carrying an organic term is not merely faster there; it is solving the correct problem.

## Caveats

The bound assumes the forward model is exact and the noise model is correct. It is a statement about the estimation problem posed, not about a physical instrument. Systematic error in the forward model, or noise with structure not captured by $\Sigma$, moves the achievable accuracy without moving the computed bound.
