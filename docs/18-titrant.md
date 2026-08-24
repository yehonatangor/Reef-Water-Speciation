# Titrant selection

Why the titrant is hydrochloric acid, and why a weak acid cannot substitute.

## Strong against weak

A strong acid dissociates completely. Every mole delivered releases exactly one mole of protons, at every pH, and the mean protons released per mole of titrant is

$$\bar{n} \equiv 1$$

identically, with no dependence on anything.

A weak acid dissociates partly, and the fraction depends on the pH it finds itself in. For an $n$-protic weak acid,

$$\bar{n}(h) = \frac{\sum_{j=1}^{n} j\left(\prod_{i=1}^{j}K_i\right)h^{\,n-j}} {h^{n} + \sum_{j=1}^{n}\left(\prod_{i=1}^{j}K_i\right)h^{\,n-j}}$$

which is a function of the pH, which is a function of how much acid has been delivered, which is what $\bar{n}$ determines.

## The consequence

That circular dependency is solvable. The solution requires the titrant's dissociation constants to be known exactly.

They are not. Published $\mathrm{p}K_a$ values for common weak acids disagree at the second decimal place, and each shifts with temperature and ionic strength in a medium where most published determinations were not made: seawater at ionic strength $0.72\ \mathrm{mol\,kg\text{-}H_2O^{-1}}$.

An error in the titrant $\mathrm{p}K_a$ enters the recovered alkalinity as a **bias**. Bias is not reduced by averaging, by a better optimiser, or by more training data. It is a fixed displacement of every answer.

## Magnitude

A $\mathrm{p}K_a$ error of $0.05$, comfortably inside the spread of published values for acetic acid, displaces recovered alkalinity by an amount comparable to the entire instrument noise budget of a laboratory-grade titration.

The strong acid removes the term from the problem rather than requiring it to be estimated accurately.

## The chosen titrant

$$0.1\ \mathrm{mol\,kg^{-1}}\ \mathrm{HCl}\ \text{in}\ 0.6\ \mathrm{mol\,kg^{-1}}\ \mathrm{NaCl}$$

as specified by the standard operating procedure for open-cell alkalinity titration.

Two properties matter.

**Complete dissociation.** $\bar{n} \equiv 1$, so the equivalence mass is exact:

$$m_{eq} = \frac{m_0 A_T}{C_a}$$

with no empirical correction and no dependence on the pH path taken to reach it.

**Matched ionic strength.** The sodium chloride background holds the ionic strength of the titrated mixture near that of seawater throughout the run. Without it, the activity coefficients, and therefore every equilibrium constant, drift as the titration proceeds, and the constants evaluated at the start no longer describe the solution at the end.

## Concentration

The titrant concentration sets how much mass must be delivered to reach the equivalence point, and therefore how finely the pump must resolve.

At $0.1\ \mathrm{mol\,kg^{-1}}$ against a $0.015\ \mathrm{kg}$ sample of alkalinity $2300\ \mu\mathrm{mol\,kg^{-1}}$, the equivalence mass is

$$m_{eq} = \frac{0.015 \times 2300\times10^{-6}}{0.1} = 3.45\times10^{-4}\ \mathrm{kg}$$

roughly $0.35\ \mathrm{g}$. A titration overshooting the equivalence point by $30\%$ delivers about $0.45\ \mathrm{g}$ in total.

A more concentrated titrant requires less mass and therefore finer pump resolution for the same number of points. A more dilute one requires more mass and dilutes the sample further, which shifts every conservative total during the run. The stated concentration balances the two.

## Verification

For $A_T = 2300$, $C_T = 2000\ \mu\mathrm{mol\,kg^{-1}}$, $S = 35$, $t = 25^\circ\mathrm{C}$, hydrochloric titrant: the simulated curve begins at pH $8.0459$, agreeing with the independent solver to $10^{-9}$, decreases strictly monotonically, and crosses the equivalence point at pH $4.29$, inside the expected region.

That figure is the pH interpolated at the equivalence mass, which is independent of how finely the curve is sampled. The pH of the nearest sampled point is not: on a $200$-point grid it lands at $4.34$, and quoting that instead would tie a chemical property to an arbitrary choice of point count.
