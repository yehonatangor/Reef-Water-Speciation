# Organic alkalinity

Dissolved organic acids accept protons in the same pH region as bicarbonate. This section establishes how much of that contribution is measurable at all, and what follows from the answer.

## The identifiability limit

An organic acid of dissociation constant $K_a$ contributes to the alkalinity balance as

$$\frac{O_T K_a}{K_a + [\mathrm{H^+}]}$$

which is a sigmoid in pH centred at $\mathrm{p}K_a$. Bicarbonate contributes a sigmoid centred at carbonic $\mathrm{p}K_1$. Two sigmoids of similar centre and similar amplitude are not distinguishable from a titration curve.

The question is quantitative. Take the sensitivity of the curve to each parameter, and ask what fraction of the organic sensitivity lies in the span of the carbonate sensitivities:

$$\text{explained} \;=\; 1 - \frac{\lVert \mathbf{J}_{O} - \mathbf{B}\hat{\mathbf{c}} \rVert^{2}}{\lVert \mathbf{J}_{O} \rVert^{2}}, \qquad \mathbf{B} = [\mathbf{J}_{A_T},\ \mathbf{J}_{C_T}]$$

A value of one means the organic contribution can be mimicked exactly by adjusting the carbonate totals, and no estimator can separate them.

## Result

At $S = 35$, $t = 25^\circ\mathrm{C}$, $\mathrm{p}K_a = 5.5$:

$$\text{explained} = 98.76\%$$

Sweeping the organic $\mathrm{p}K_a$ across its full drawn range shows the degeneracy tracking the overlap with carbonic $\mathrm{p}K_1 = 5.847$:

| Organic $\mathrm{p}K_a$ | Explained | Independent |
|---|---|---|
| $4.00$ | $0.7089$ | $29.11\%$ |
| $4.50$ | $0.8289$ | $17.11\%$ |
| $5.00$ | $0.9313$ | $6.87\%$ |
| $5.50$ | $0.9876$ | $1.24\%$ |
| $\mathbf{5.85}$ | $\mathbf{0.9996}$ | $\mathbf{0.04\%}$ |
| $6.00$ | $0.9987$ | $0.13\%$ |
| $6.50$ | $0.9701$ | $2.99\%$ |
| $7.00$ | $0.8985$ | $10.15\%$ |

The curve is not monotone. It falls to a minimum exactly at carbonic $\mathrm{p}K_1$ and rises symmetrically on either side, because what governs identifiability is the *separation* between the two dissociation constants rather than the value of either.

At $\mathrm{p}K_a$ equal to carbonic $\mathrm{p}K_1$ the degeneracy is total. An organic acid dissociating at pH $5.85$ *is* a bicarbonate ion, as far as a titration curve can determine.

The practical consequence is that the recoverable fraction depends on which organic acids are present. Acetate, from carbon dosing, sits at $\mathrm{p}K_a = 4.76$ where roughly a tenth of its signature is separable. An acid near $5.85$ is invisible.

## Robustness across the state space

Holding the organic $\mathrm{p}K_a$ at the centre of its range and varying everything else:

| State | Carbonic $\mathrm{p}K_1$ | Independent |
|---|---|---|
| $S=35$, $t=25^\circ\mathrm{C}$, $A_T=2300$, $C_T/A_T=0.87$ | $5.847$ | $1.24\%$ |
| $S=20$ | $5.925$ | $1.59\%$ |
| $S=40$ | $5.833$ | $1.19\%$ |
| $t=10^\circ\mathrm{C}$ | $5.993$ | $2.20\%$ |
| $t=32^\circ\mathrm{C}$ | $5.792$ | $0.96\%$ |
| $A_T=1600$ | $5.847$ | $1.26\%$ |
| $A_T=3200$ | $5.847$ | $1.23\%$ |
| $C_T/A_T=0.80$ | $5.847$ | $1.29\%$ |
| $C_T/A_T=0.95$ | $5.847$ | $1.40\%$ |
| $O_T=15\ \mu\mathrm{mol\,kg^{-1}}$ | $5.847$ | $1.26\%$ |
| $O_T=150\ \mu\mathrm{mol\,kg^{-1}}$ | $5.847$ | $1.23\%$ |
| $S=20$, $t=10^\circ\mathrm{C}$, $A_T=1600$ | $6.072$ | $2.58\%$ |

Between $0.96\%$ and $2.58\%$ across every state tested. The degeneracy is a property of the chemistry rather than of the reference point chosen to demonstrate it.

Temperature has the largest effect, and through a single mechanism: carbonic $\mathrm{p}K_1$ shifts from $5.79$ at $32^\circ\mathrm{C}$ to $5.99$ at $10^\circ\mathrm{C}$, moving it further from the fixed organic $\mathrm{p}K_a$ of $5.5$ and widening the separation the sweep above measures. Cold, fresh, low-alkalinity water is the most favourable case, and it still yields under $3\%$.

This is a property of the measurement not of the model. It bounds every estimator, classical or learned. A method reporting organic content to better than a few percent of its true variation has not solved the problem; it has acquired information the curve does not contain.

## Why it still matters

A blind spot that cannot be resolved can still be *accounted for*.

The classical fit has no organic term at all. Presented with water containing organic acids, it attributes their proton acceptance to the carbonate system, and the resulting error appears in the speciation rather than in the alkalinity. The total is right; its division among species is not.

A model carrying an organic term does not need to measure that term accurately to benefit. It needs only to allocate the acceptance to something other than carbonate. The correction is bounded by the degeneracy above, but it is not zero, and it acts on precisely the quantity the classical method gets wrong.

## The control is not optional

A learned model with organic latents is expected to be **worse** on organic-free water, because it spends parameters on a quantity that is not present.

Any comparison on organic-containing water must therefore be accompanied by a matched control: the same generator, the same instrument, the same protocol, organic switched off. Reporting the favourable case alone would overstate a real effect.

### Why the control uses a different seed

The control is generated with organic probability zero and a seed distinct from the training set, and this is not incidental.

Sampling is deterministic in the seed. Reusing the training seed with organic disabled would draw the identical sequence of salinities, temperatures, carbon totals and instrument states, differing only in the organic term. Every composition in the control would then be one the model had already been trained on.

The resulting comparison would measure how well the model reproduces water it has memorised, not how it behaves on water it has not seen. A separate seed makes the control an independent draw from the same distribution, which is what the comparison requires.

The control is evaluated only. It never enters training.

## What the recovery ratio measures

For each latent, the ratio of the true standard deviation to the residual standard deviation:

$$\text{recovery} = \frac{\mathrm{sd}(\theta)}{\mathrm{sd}(\theta - \hat\theta)}$$

A value near one means the encoder is predicting the population mean and the curve does not identify the parameter. Values meaningfully above one indicate genuine recovery.

The organic $\mathrm{p}K_a$ is expected near one regardless of model quality: it has no effect whatever when the organic total is zero, which is half the training set.

## Reproducing

```bash
python scripts/organic_identifiability.py --sweep --sweep-states
```

Pure chemistry. No dataset, no trained model, seconds to run, identical on any machine.
