# Total alkalinity

## The Dickson (1981) proton condition

Alkalinity is the proton deficit relative to a defined zero level of protons. Dickson's criterion: species formed from acids with $K > 10^{-4.5}$ at $S=35$, $t = 25^\circ\mathrm{C}$ are assigned to the donor (protonated) side.

$$\begin{aligned}
A_T &= [\mathrm{HCO_3^-}] + 2[\mathrm{CO_3^{2-}}] + [\mathrm{B(OH)_4^-}]
+ [\mathrm{OH^-}] + [\mathrm{HPO_4^{2-}}] + 2[\mathrm{PO_4^{3-}}] \\
&\quad + [\mathrm{SiO(OH)_3^-}] + [\mathrm{NH_3}] + [\mathrm{HS^-}] \\
&\quad - [\mathrm{H^+}]_F - [\mathrm{HSO_4^-}] - [\mathrm{HF}] - [\mathrm{H_3PO_4}]
\end{aligned}$$

Two features are routinely dropped by ad-hoc implementations and are implemented correctly here:

1. The hydrogen ion term is on the free scale, not total or SWS.
2. The $-[\mathrm{HSO_4^-}]$ and $-[\mathrm{HF}]$ terms are present.

Quantitatively, at $S=35$, $t=25^\circ\mathrm{C}$, $DIC = 2000\ \mu\mathrm{mol\,kg^{-1}}$:

| pH | $[\mathrm{HSO_4^-}] + [\mathrm{HF}]$ | Significance |
|---|---|---|
| 8.1 | $0.0022\ \mu\mathrm{mol\,kg^{-1}}$ | negligible |
| 4.0 | $24.13\ \mu\mathrm{mol\,kg^{-1}}$ | ≈ 10× the reproducibility of a CRM titration |

The ratio between the two is a factor of $1.1\times10^{4}$.

A curve-fitting estimate of $A_T$ spends most of its information in the pH 3–4 Gran region. Omitting these terms therefore biases the fitted alkalinity even though it barely affects the forward calculation at seawater pH.

Ammonia and sulfide are omitted from this implementation, which is a documented limitation: both are negligible in oxic reef water.

## The explicit conservative expression
  
Contains no equilibrium constants and no pH, (Wolf-Gladrow et al. 2007)

$$\begin{aligned}
TA_{ec} &= [\mathrm{Na^+}] + [\mathrm{K^+}] + 2[\mathrm{Mg^{2+}}] + 2[\mathrm{Ca^{2+}}]
+ 2[\mathrm{Sr^{2+}}] \\
&\quad - [\mathrm{Cl^-}] - [\mathrm{Br^-}] - [\mathrm{NO_3^-}]
- 2\,\mathrm{TSO_4} - \mathrm{TF} - \mathrm{TPO_4}
\end{aligned}$$

Its value is interpretive: any process that does not change a conservative ion concentration cannot change alkalinity. It immediately explains why CO₂ invasion leaves $A_T$ unchanged, why $\mathrm{CaCO_3}$ precipitation lowers it by exactly 2 equivalents per mole, and why nitrate assimilation raises it by 1. It is provided as an independent cross-check, not as the primary computation, since the full conservative ion suite is rarely measured.

## Speciation

The carbonate distribution shares a single denominator, which guarantees exact DIC closure in floating point:

$$D = [\mathrm{H^+}]^2 + K_1[\mathrm{H^+}] + K_1K_2$$

$$[\mathrm{CO_2^*}] = C_T\frac{[\mathrm{H^+}]^2}{D},\quad [\mathrm{HCO_3^-}] = C_T\frac{K_1[\mathrm{H^+}]}{D},\quad [\mathrm{CO_3^{2-}}] = C_T\frac{K_1K_2}{D}$$

Phosphate, with $D_P = [\mathrm{H^+}]^3 + K_{1P}[\mathrm{H^+}]^2 + K_{1P}K_{2P}[\mathrm{H^+}] + K_{1P}K_{2P}K_{3P}$:

$$[\mathrm{H_3PO_4}] = P_T\frac{[\mathrm{H^+}]^3}{D_P},\quad [\mathrm{H_2PO_4^-}] = P_T\frac{K_{1P}[\mathrm{H^+}]^2}{D_P},\quad [\mathrm{HPO_4^{2-}}] = P_T\frac{K_{1P}K_{2P}[\mathrm{H^+}]}{D_P},\quad [\mathrm{PO_4^{3-}}] = P_T\frac{K_{1P}K_{2P}K_{3P}}{D_P}$$

Monoprotic systems: 

$$[\mathrm{B(OH)_4^-}] = B_T K_B/(K_B + [\mathrm{H^+}]), [\mathrm{SiO(OH)_3^-}] = Si_T K_{Si}/(K_{Si} + [\mathrm{H^+}])$$

Free-scale species: $[\mathrm{H^+}]_F = [\mathrm{H^+}]_T / f_{F\to T}$:

$$[\mathrm{HSO_4^-}] = \frac{S_T}{1 + K_S/[\mathrm{H^+}]_F}, \qquad [\mathrm{HF}] = \frac{F_T}{1 + K_F/[\mathrm{H^+}]_F}$$

Verified:

DIC, $B_T$, $P_T$ and $Si_T$ are each conserved to $10^{-14}$ relative across pH 2–12, and no species is ever negative.
