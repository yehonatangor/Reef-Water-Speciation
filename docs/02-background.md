# Background

Marine acid-base chemistry is governed by a network of coupled equilibria, specifically the dissolved inorganic carbon and borate systems whose speciation shifts deterministically with pH, temperature, and salinity. Extracting total alkalinity from a titration curve requires translating these reactions into a proton-balance framework rather than relying on qualitative approximations. Because published equilibrium constants vary across measurement conventions, maintaining strict consistency across pH scales and temperature invariant mass units ($\mathrm{mol\,kg^{-1}}$) is essential to avoid systematic error. The following sections establish this thermodynamic ground truth, defining the mathematical foundations used to model the system and deconvolute titration data.

## Equilibrium constants

Equilibrium constants are ratios, not rates.

When a weak acid is sitting in water it is constantly splitting and recombining:

$$\mathrm{HA} \rightleftharpoons \mathrm{H^{+}} + \mathrm{A^{-}}$$

At equilibrium the ratio of the concentrations is fixed at a given temperature and salinity, and that ratio is the equilibrium constant:

$$K = \frac{[\mathrm{H^{+}}][\mathrm{A^{-}}]}{[\mathrm{HA}]}$$

Because these constants span many orders of magnitude they are quoted as $\mathrm{p}K = -\log_{10} K$. A $\mathrm{p} K$ of 5.85 means the acid is half dissociated at pH 5.85. Above that pH most of it has given up its proton, below it most has kept it. When you see two $\mathrm{p} K$ values close together, it means two different chemicals are doing the same thing at the same pH, and telling them apart from a titration curve is difficult.

## The carbonate system 

Dissolved inorganic carbon exists as three interconverting species:

$$\mathrm{CO_{2}} \;\rightleftharpoons\; \mathrm{HCO_3^{-}} \;\rightleftharpoons\; \mathrm{CO_3^{2-}}$$

with $ \mathrm{p} K_{1} \approx 5.85$ and $\mathrm{p} K_{2} \approx 8.97$ in seawater. 

At typical seawater pH of 8.1 the mixture is roughly 88% bicarbonate, 11% carbonate, 1% $\mathrm{CO}_{2}$. Add acid and the balance shifts left,eventually to $\mathrm{CO}_{2}$ which can escape as the system as a gas.

Boron matters aswell its pH is quite close to that of seawater, $\mathrm{B(OH)_{3}}/\mathrm{B(OH)_4^{-}}, \ \mathrm{p}K_{B} \approx 8.6$, and seawater has enough of it to contribute a few percent of the alkalinity. 

Phosphate, silicate, sulfate and fluoride each contribute a little and are all considered aswell.

## Alkalinity

The "capacity to absorb acid" needs to be made exact before it can be computed, and Dickson (1981) does that by defining alkalinity as the excess of proton acceptors over proton donors relative to a chosen zero point:

$$A_T = [\mathrm{HCO_3^{-}}] + 2[\mathrm{CO_3^{2-}}] + [\mathrm{B(OH)_4^{-}}] + [\mathrm{OH^{-}}] + \dots - [\mathrm{H^{+}}]_F - [\mathrm{HSO_4^{-}}] - [\mathrm{HF}] - \dots$$

Notice two things:

Carbonate counts twice because it can accept two protons. 

The definition is a conservative quantity: diluting the sample scales it, but adding CO₂ does not change it at all, because CO₂ is uncharged and contributes no proton acceptor. That last property is what makes alkalinity useful and why the kinetics matter.

## pH scales

There are three pH scales: Free, Total, and Seawater.

By classical convention "pH" is defined as the negative base-10 logarithm of hydrogen ion concentration ($\mathrm{pH} = -\log_{10}[\mathrm{H^{+}}]$), the parameter serves as a logarithmic measure of proton activity within an aqueous solution. However in seawater some hydrogen ions are stuck to sulfate and fluoride, and different conventions count them differently:

| Scale | Counts as "hydrogen" |
|---|---|
| Free | free H⁺ only |
| Total (used here) | free H⁺ + HSO₄⁻ |
| Seawater | free H⁺ + HSO₄⁻ + HF |

The three differ by roughly 0.1 pH, about 25% in concentration, and far larger than the measurement error we would consider. Published constants come on different scales, so mixing them produces answers that are wrong by more than the effect being measured. For this reason, every function in this library names its scale.

## Concentration

Why was mol per kilogram chosen rather than per litre?

Consider that volume changes with temperature; mass does not. A concentration in $\mathrm{mol\,L^{-1}}$ measured at $25^\circ\mathrm{C}$ means something different at $10^\circ\mathrm{C}$, while $\mathrm{mol\,kg^{-1}}$ is the same number everywhere. Oceanography uses $\mathrm{mol\,kg^{-1}}$ of solution for this specific reason, and so does this library.