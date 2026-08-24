# Instrument model

The measured curve differs from the chemical curve. Five effects separate them, and each is modelled so that its parameters can be estimated rather than assumed.

## Calibration offset

A glass electrode reports

$$E = E_0 - \frac{RT}{F}\ln[\mathrm{H^+}]_T$$

and $E_0$ drifts with reference-junction condition, temperature history and age. An error in $E_0$ appears as a constant additive offset $\delta$ in every reported pH:

$$\mathrm{pH_{meas}} = \mathrm{pH} + \delta$$

This is the largest single error source on inexpensive hardware. Because it is constant along the curve while the alkalinity signal is not, it is separable, and fitting it recovers most of the loss.

## Nernstian slope

The theoretical response is $2.303\,RT/F = 59.16\ \mathrm{mV}$ per pH unit at $25\,^\circ\mathrm{C}$. Real electrodes achieve $95$–$102\%$ of this, degrading with age and membrane fouling. A fractional slope $s$ pivots the reported pH about the calibration point:

$$\mathrm{pH_{meas}} = \mathrm{pH_{cal}} + s\,(\mathrm{pH} - \mathrm{pH_{cal}}) + \delta$$

The pivot is the calibration pH, not zero, because a two-point calibration fixes the response there. Modelling the pivot incorrectly introduces an error that grows with distance from the calibration point, which is precisely the region a titration explores.

## Electrode noise

Successive readings are not independent. Junction potential and film effects give a first-order autoregressive structure between adjacent titration points:

$$\varepsilon_i = \rho\,\varepsilon_{i-1} + \sqrt{1-\rho^2}\,\eta_i, \qquad \eta_i \sim \mathcal{N}(0, \sigma^2)$$

with $\rho$ between $0.4$ and $0.85$ for inexpensive probes and $0.3$ to $0.7$ for laboratory grade. Ignoring the correlation and treating the record as white overstates the information content by roughly $\sqrt{N}$.

## Titrant delivery

A hobby-grade pump delivers a mass proportional to the commanded value with a fixed multiplicative error:

$$m_{\mathrm{actual}} = \kappa\,m_{\mathrm{commanded}}$$

Alkalinity is recovered as $A_T = m_{eq}C_a/m_0$, so a relative error in delivered mass is a relative error in alkalinity of the same magnitude. It is not recoverable from the curve shape alone and must be estimated jointly.

## Digitisation

An $n$-bit converter over a full-scale range $V$ quantises with step $q = V/2^n$, contributing uniform noise of standard deviation $q/\sqrt{12}$. For a $16$-bit converter over $\pm 2.048\ \mathrm{V}$ this is $0.00106$ pH per bit, small relative to electrode noise but genuinely independent between samples, and the only term that is.

## Additional physical effects

**Carbon dioxide loss.** An open cell loses $\mathrm{CO_2}$ during the run, reducing $C_T$ over time. Modelled as a first-order loss proportional to the supersaturation of dissolved $\mathrm{CO_2}$ relative to atmosphere.

**Temperature and salinity uncertainty.** Both are measured, not known. Errors in either propagate into every equilibrium constant and therefore into the whole curve.

## Instrument grades

Two presets bracket the plausible range:

| Parameter | Laboratory | Hobbyist |
|---|---|---|
| Electrode noise, pH | $0.003$ | $0.012$ |
| Calibration offset, pH | $0.005$ | $0.03$ |
| Junction drift, pH | $0.01$ | $0.05$ |
| Slope error | $0.4\%$ | $3\%$ |
| Autocorrelation $\rho$ | $0.3$–$0.7$ | $0.4$–$0.85$ |
| Pump scale error | $0.15\%$ | $0.8\%$ |
| Pump jitter | $0.08\%$ | $0.4\%$ |
| Quantisation step, pH | $0$ | $0.00106$ |
| Carbon dioxide loss | $0$–$5\%$ | $0$–$12\%$ |
| Salinity uncertainty | $\pm 0.02$ | $\pm 0.5$ |
| Temperature uncertainty, $^\circ\mathrm{C}$ | $\pm 0.05$ | $\pm 0.3$ |
| NBS calibration bias, pH | $0$ | $0.13 \pm 0.04$ |

Every accuracy figure quoted anywhere in this documentation is conditional on which grade produced it. The two differ by a factor of four in electrode noise and considerably more in the nuisance terms.

The laboratory grade takes quantisation as negligible, since a research-grade meter resolves below its own noise floor. The hobbyist grade carries an additional systematic the laboratory grade does not: a calibration bias from standardising against buffers of ionic strength near $0.1$ rather than against a seawater reference material.

## Why this matters for the inverse

The nuisance parameters are the reason a learned model must predict more than the composition. An encoder that outputs only $A_T$ and $C_T$ implicitly assumes a perfectly calibrated instrument, and its errors on real hardware are dominated by that assumption rather than by anything chemical.

Estimating offset, slope and pump scale alongside the composition is what allows an inexpensive instrument to approach the accuracy of an expensive one. The information to do so is present in the curve because the three act on it differently: an offset shifts it, a slope error tilts it about the calibration point, and a pump error rescales its horizontal axis.
