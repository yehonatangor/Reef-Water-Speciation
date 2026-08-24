# Noise budget

Which error mechanism dominates, measured by zeroing every other tolerance in turn.

## Method

Two hundred realisations of a two-hundred-point curve at $A_T = 2300$, $C_T = 2000\ \mu\mathrm{mol\,kg^{-1}}$, $S = 35$, $t = 25^\circ\mathrm{C}$. Each mechanism is measured with every other tolerance set to zero, so the figure is that mechanism's contribution alone.

## Laboratory grade

Total pH residual with all mechanisms active: $0.0071$, maximum absolute residual $0.0277$.

| Mechanism | sd | max | Share of total |
|---|---|---|---|
| Nernstian slope | $0.00485$ | $0.01581$ | $68.0\%$ |
| Electrode autocorrelation | $0.00298$ | $0.00885$ | $41.8\%$ |
| Carbon dioxide loss | $0.00281$ | $0.01074$ | $39.4\%$ |
| Junction drift | $0.00219$ | $0.00753$ | $30.7\%$ |
| Calibration offset | $0$ | $0.00422$ | $0\%$ |
| Pump scale and jitter | $0$ | $0$ | $0\%$ |
| Digitisation | $0$ | $0$ | $0\%$ |
| Salinity measurement | $0$ | $0$ | $0\%$ |
| Temperature measurement | $0$ | $0$ | $0\%$ |

## The shares do not sum to one

Quadrature sum $0.00671$ against an actual total of $0.00713$.

The terms are not independent along the curve. A slope error pivots about the calibration pH, so its contribution grows with distance from that point rather than scattering randomly. Carbon dioxide loss accumulates monotonically through the run. Two mechanisms that both grow toward the end of the titration are correlated, and correlated terms do not add in quadrature.

The table is therefore a ranking, not a decomposition. Read the order, not the sum.

## What dominates

**Nernstian slope, at $68\%$ of the total.** A slope error of $0.4\%$, well inside specification for a laboratory electrode, is the single largest contributor on the best hardware modelled here.

This is why the slope is a fitted parameter rather than an assumed constant. Fixing it at its theoretical value discards the dominant term.

**Calibration offset contributes zero to the scatter and $0.0042$ to the maximum.** It is a constant displacement of the whole curve: it changes every point identically, so it adds nothing to the residual spread while shifting the answer. A mechanism invisible to any measure of scatter can still be the largest source of error in the recovered alkalinity, which is exactly what the classical benchmark shows on hobbyist hardware.

**Pump error also contributes zero here**, and for a different reason: it acts on the horizontal axis. It changes which titrant mass each pH corresponds to, not the pH itself. The accuracy floor shows this term to be the largest single lever on the recovered alkalinity despite contributing nothing measurable to pH scatter.

## The general point

Three of the mechanisms that matter most to the answer contribute little or nothing to the pH residual: calibration offset, pump scale, and to a lesser extent junction drift.

A noise budget expressed in pH is therefore a poor guide to what limits accuracy. It ranks the mechanisms that make the curve look noisy, which is a different question from which mechanisms make the answer wrong.

## Reproducing

```bash
python scripts/noise_budget.py --instrument laboratory
python scripts/noise_budget.py --instrument hobbyist
```
