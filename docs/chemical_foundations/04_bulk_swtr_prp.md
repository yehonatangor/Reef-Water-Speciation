# Bulk seawater properties

## Density (Millero & Poisson 1981)

$$\rho_{SW} = \rho_{SMOW} + A S + B S^{3/2} + C S^{2}$$

$$\rho_{SMOW} = 999.842594 + 6.793952\times10^{-2}\,t - 9.095290\times10^{-3}\,t^2 + 1.001685\times10^{-4}\,t^3 - 1.120083\times10^{-6}\,t^4 + 6.536332\times10^{-9}\,t^5$$

$$A = 8.24493\times10^{-1} - 4.0899\times10^{-3}t + 7.6438\times10^{-5}t^2 - 8.2467\times10^{-7}t^3 + 5.3875\times10^{-9}t^4$$

$$B = -5.72466\times10^{-3} + 1.0227\times10^{-4}t - 1.6546\times10^{-6}t^2, \qquad C = 4.8314\times10^{-4}$$

Valid $0 \le S \le 42$, $0 \le t \le 40\ ^\circ\mathrm{C}$. 

Verified: 

$\rho_{SW}(35, 25) = 1023.343\ \mathrm{kg\,m^{-3}}$, matches the Guides stated value.

## Ionic strength

$$I = \frac{19.924\,S}{1000 - 1.005\,S} \approx 0.02\,S$$

in $\mathrm{mol\,kg\text{-}H_2O^{-1}}$. 

Verified:

$I(35) = 0.72276$.

## Conservative totals

All scale linearly with chlorinity $\mathrm{Cl} = S/1.80655$:

| Species | Ratio | Source | Value at $S=35$ |
|---|---|---|---|
| $B_T$ | $0.232\ \mathrm{mg\,kg^{-1}}$ per ‰ Cl | Uppström (1974) | $415.76\ \mu\mathrm{mol\,kg^{-1}}$ |
| $B_T$ | $0.2414\ \mathrm{mg\,kg^{-1}}$ per ‰ Cl | Lee *et al.* (2010) | $432.60\ \mu\mathrm{mol\,kg^{-1}}$ |
| $S_T$ | $0.14/96.062$ | Morris & Riley (1966) | $28.235\ \mathrm{mmol\,kg^{-1}}$ |
| $F_T$ | $6.7\times10^{-5}/18.9984$ | Riley (1965) | $68.32\ \mu\mathrm{mol\,kg^{-1}}$ |
| $Ca_T$ | $0.02128/40.087$ | Riley & Tongudai (1967) | $10.285\ \mathrm{mmol\,kg^{-1}}$ |

$$B_T = \frac{r_{B/Cl}\times 10^{-3}}{M_B}\cdot\frac{S}{1.80655}, \qquad M_B = 10.811$$

The boron choice matters. The two ratios differ by 4%, which is $\approx 17\ \mu\mathrm{mol\,kg^{-1}}$ in $B_T$ and $\approx 4\ \mu\mathrm{mol\,kg^{-1}}$ in $A_T$ at seawater pH, comparable to the entire uncertainty budget of a CRM-calibrated titration. Lee *et al.* argue from 139 samples across three basins that $0.2414$ supersedes Uppström; the package defaults to Uppström for compatibility with CO2SYS and the Guide, and exposes the choice explicitly.
