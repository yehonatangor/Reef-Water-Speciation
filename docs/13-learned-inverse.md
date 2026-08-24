# The learned inverse

A neural network that recovers composition in one forward pass, with the chemistry inside the network rather than in the loss.

## Why learn an inverse at all

The classical fit sits at the information floor, so accuracy is not available as a target. Two things are.

**Amortised inference.** The optimisation costs on the order of a second per sample and requires a working optimiser at run time. A trained network costs one forward pass, which is what makes an embedded implementation possible.

**A structural gap.** The least-squares model carries no organic acid term. On water containing dissolved organic matter it is fitting a model that cannot represent the sample, and no amount of optimisation repairs that.

## Architecture

$$\text{curve} \;\xrightarrow{\ \text{encoder}\ }\; \boldsymbol\theta \;\xrightarrow{\ \text{forward model}\ }\; \widehat{\text{curve}}$$

A convolutional encoder maps the measured curve, together with salinity, temperature, sample mass and a set of summary statistics, to seven physical parameters. Those parameters are then passed through the exact chemistry, implemented differentiably, which renders the curve they imply. The loss compares that reconstruction to the measurement.

The decoder has no trainable weights. All capacity is in the encoder.

### Consequence

The output is a physically valid state by construction. The network proposes a composition; the chemistry determines what curve it produces. There is no configuration of weights that emits a curve no chemistry could generate, at any loss weighting, at any point in training.

This is stronger than adding a physics penalty to the loss. A penalty introduces a weight with no principled value, and whenever that weight is too small the model buys lower data loss with physics violations. An enforced constraint cannot be traded off.

## The seven parameters

| Parameter | Meaning |
|---|---|
| $A_T$ | total alkalinity |
| $A_T - C_T$ | alkalinity minus dissolved inorganic carbon |
| $O_T$ | lumped organic acid total |
| $\mathrm{p}K_{\mathrm{org}}$ | organic acid dissociation constant |
| $\delta$ | calibration offset |
| $s$ | fractional Nernstian slope |
| $\kappa$ | pump delivery scale |

The last three are instrument nuisances. They are unwanted, but they are unknown, so the model must estimate them to recover the first two.

### Why the difference rather than the carbon

Subtracting the carbon sum from the alkalinity sum cancels bicarbonate, the largest term in both:

$$A_T - C_T = [\mathrm{CO_3^{2-}}] - [\mathrm{CO_2}] + [\mathrm{B(OH)_4^-}] + [\mathrm{OH^-}] - [\mathrm{H^+}]$$

At $S = 35$, $t = 25^\circ\mathrm{C}$, $A_T = 2300$ and $C_T = 2000\ \mu\mathrm{mol\,kg^{-1}}$, the right-hand side evaluates to $213.40 - 11.23 + 91.15 + 6.69 - 0.01 = 300.01\ \mu\mathrm{mol\,kg^{-1}}$ against a difference of $300.00$.

Two reasons follow. The difference is close to the carbonate ion concentration, which is the quantity governing aragonite saturation and therefore the biologically meaningful one. And predicting two large numbers separately leaves their difference as the accident of two errors: $A_T$ and $C_T$ are of order $2000$ and their difference of order $300$, so a one percent error in each can be a fifteen percent error in the difference.

## Bounded outputs

Each parameter is squashed into a physical range by a bounded nonlinearity rather than penalised for leaving it:

| Parameter | Range |
|---|---|
| $A_T$ | $0.5$–$6.0\ \mathrm{mmol\,kg^{-1}}$ |
| $A_T - C_T$ | $0.01$–$1.5\ \mathrm{mmol\,kg^{-1}}$ |
| $O_T$ | $0$–$225\ \mu\mathrm{mol\,kg^{-1}}$ |
| $\mathrm{p}K_{\mathrm{org}}$ | $3.5$–$7.5$ |
| $\delta$ | $-0.5$–$0.5$ pH |
| $s$ | $0.8$–$1.2$ |
| $\kappa$ | $0.90$–$1.10$ |

The bounds exceed the sampled ranges deliberately. A parameter drawn at exactly its bound is attainable only in the limit of an infinite logit, and organic content is zero in half the training set.

## Differentiating through the root-find

The decoder solves for pH at every titration point, and there is no closed form. Gradients must pass through that solve.

Unrolling the iterations onto the tape works, is slow, and grows memory with iteration count. The implicit function theorem gives the gradient directly. For a residual $f(h;\boldsymbol\theta) = 0$,

$$\frac{\partial h}{\partial \boldsymbol\theta} = -\left(\frac{\partial f}{\partial h}\right)^{-1}\frac{\partial f}{\partial \boldsymbol\theta}$$

In practice: run a fixed bisection with gradients disabled, then take one Newton step on the tape. At convergence the residual is negligible, so that step changes the value by nothing and supplies exactly the right derivative. The cost is one extra residual evaluation regardless of how many iterations the solve took, and the memory is constant.

The derivative $\partial f/\partial h$ is available analytically, since every term in the alkalinity balance is differentiable in closed form.

## Loss

Three terms, weighted:

$$\mathcal{L} = \underbrace{\lVert \mathrm{pH^{meas}} - \widehat{\mathrm{pH}} \rVert^{2}}_{\text{reconstruction}} \;+\; \lambda_{\theta}\sum_j w_j\lVert \theta_j - \hat\theta_j \rVert^{2} \;+\; \lambda_{y}\lVert \mathbf{y} - \hat{\mathbf{y}} \rVert^{2}$$

The reconstruction term needs no labels. The parameter term supervises the latents directly, each scaled by its own magnitude so that alkalinity in $\mathrm{mol\,kg^{-1}}$ and slope near unity contribute comparably. The species term supervises the three outputs of interest.

Pure reconstruction is insufficient on its own: several parameter combinations produce nearly identical curves, and without direct supervision the encoder is free to choose among them arbitrarily.

## Summary statistics as input

Alongside the raw curve, the encoder receives fourteen physical summary statistics: the titrant mass at each of several fixed pH thresholds, both absolute and normalised by sample mass, the final mass, and the initial pH.

These are the quantities a chemist reads off a titration curve. Supplying them directly avoids requiring the convolutional stack to learn an interpolated crossing point from raw samples, which spends capacity on a quantity available in closed form.

### Evidence for supplying them

A linear regression on those fourteen statistics alone recovers alkalinity to $28.16\ \mu\mathrm{mol\,kg^{-1}}$ standard deviation, $22.81$ mean absolute error, bias $-0.88$, on hobbyist-grade data.

A convolutional network given the raw curve and no summary statistics reaches $58.90$. A linear model on the right features outperforms a convolutional stack on the raw signal by a factor of $2.09$.

That result determines the architecture. The summary statistics are not a convenience; the network is measurably worse without them.

Ranking the individual features by the ratio of fitted coefficient to its spread:

| Feature | Sensitivity |
|---|---|
| Titrant mass at pH $4$, per sample mass | $3163\ \mu\mathrm{mol\,kg^{-1}}$ per sd |
| Titrant mass at pH $4$ | $3062$ |
| Titrant mass at pH $4.5$, per sample mass | $1903$ |
| Titrant mass at pH $4.5$ | $1832$ |
| Final titrant mass | $1237$ |

The four most informative features are all crossing points in the equivalence region, and normalising by sample mass improves each, which is what the forward model predicts, since alkalinity is recovered as $m_{eq}C_a/m_0$.
