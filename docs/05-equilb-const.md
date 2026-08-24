# Equilibrium Constants

## CO₂ solubility

No pH scale, (Weiss 1974).

$$\mathrm{CO_2(g)} \rightleftharpoons \mathrm{CO_2^*(aq)}, \qquad K_0 = \frac{[\mathrm{CO_2^*}]}{f(\mathrm{CO_2})}$$

$$\ln K_0 = 93.4517\frac{100}{T} - 60.2409 + 23.3585\ln\!\left(\frac{T}{100}\right) + S\!\left[0.023517 - 0.023656\frac{T}{100} + 0.0047036\left(\frac{T}{100}\right)^{2}\right]$$

Verified: 

$\ln K_0(35,25) = -3.5617$. Matches the guide value of −3.5617.

## Bisulfate 

Free scale, by definition (Dickson 1990a).

$$\mathrm{HSO_4^-} \rightleftharpoons \mathrm{H^+} + \mathrm{SO_4^{2-}}, \qquad K_S = \frac{[\mathrm{H^+}]_F[\mathrm{SO_4^{2-}}]}{[\mathrm{HSO_4^-}]}$$

$$\begin{aligned}
\ln K_S &= \frac{-4276.1}{T} + 141.328 - 23.093\ln T \\
&\quad + \left(\frac{-13856}{T} + 324.57 - 47.986\ln T\right)I^{1/2} \\
&\quad + \left(\frac{35474}{T} - 771.54 + 114.723\ln T\right)I \\
&\quad - \frac{2698}{T}I^{3/2} + \frac{1776}{T}I^{2} + \ln(1 - 0.001005\,S)
\end{aligned}$$

$K_S$ cannot be placed on the total scale. It is the constant that defines the free-to-total transformation. The trailing $\ln(1-0.001005S)$ converts from $\mathrm{mol\,kg\text{-}H_2O^{-1}}$ to $\mathrm{mol\,kg\text{-}soln^{-1}}$; omitting it causes a 3.5% error at $S=35$.

Verified:

$\ln K_S(35,25) = -2.2996$. Matches the guide value of $-2.30$.

## Hydrogen fluoride 

Free scale (Dickson & Riley 1979a), required for scale conversion:

$$\ln K_F = \frac{1590.2}{T} - 12.641 + 1.525\,I^{1/2} + \ln(1 - 0.001005\,S)$$

Total scale (Perez & Fraga 1987), preferred by the Guide for speciation:

$$\ln K_F = \frac{874}{T} - 9.68 + 0.111\,S^{1/2}$$

Verified:

$\ln K_F^{\mathrm{Total}}(35,25) = -6.092$. Matches the Guide value of $-6.09$.

## Boric acid

Total scale (Dickson 1990b)

$$\mathrm{B(OH)_3} + \mathrm{H_2O} \rightleftharpoons \mathrm{H^+} + \mathrm{B(OH)_4^-}$$

$$\begin{aligned}
\ln K_B &= \frac{-8966.90 - 2890.53\,S^{1/2} - 77.942\,S + 1.728\,S^{3/2} - 0.0996\,S^{2}}{T} \\
&\quad + 148.0248 + 137.1942\,S^{1/2} + 1.62142\,S \\
&\quad + \left(-24.4344 - 25.085\,S^{1/2} - 0.2474\,S\right)\ln T \\
&\quad + 0.053105\,S^{1/2}\,T
\end{aligned}$$

Verified:

$\ln K_B(35,25) = -19.7964$. Matches the Guide value; and matches PyCO2SYS to machine precision of $−19.7964$. 

## Carbonic acid

Carbonic acid has four parameterisations.

$$\mathrm{CO_2^*} + \mathrm{H_2O} \rightleftharpoons \mathrm{H^+} + \mathrm{HCO_3^-}, \qquad K_1 = \frac{[\mathrm{H^+}][\mathrm{HCO_3^-}]}{[\mathrm{CO_2^*}]}$$

$$\mathrm{HCO_3^-} \rightleftharpoons \mathrm{H^+} + \mathrm{CO_3^{2-}}, \qquad K_2 = \frac{[\mathrm{H^+}][\mathrm{CO_3^{2-}}]}{[\mathrm{HCO_3^-}]}$$

Total scale, Lueker, Dickson & Keeling (2000), package default:

$$\log_{10} K_1 = \frac{-3633.86}{T} + 61.2172 - 9.67770\ln T + 0.011555\,S - 0.0001152\,S^{2}$$

$$\log_{10} K_2 = \frac{-471.78}{T} - 25.9290 + 3.16967\ln T + 0.01781\,S - 0.0001122\,S^{2}$$

Total Scale, Roy et al. (1993), $\mathrm{mol\,kg\text{-}soln^{-1}}$:

$$\ln K_1 = 2.83655 - \frac{2307.1266}{T} - 1.5529413\ln T + \left(-0.20760841 - \frac{4.0484}{T}\right)S^{1/2} + 0.08468345\,S - 0.00654208\,S^{3/2} + \ln(1-0.001005S)$$

$$\ln K_2 = -9.226508 - \frac{3351.6106}{T} - 0.2005743\ln T + \left(-0.106901773 - \frac{23.9722}{T}\right)S^{1/2} + 0.1130822\,S - 0.00846934\,S^{3/2} + \ln(1-0.001005S)$$

Seawater Scale, Millero et al. (2006), converted internally.

Seawater Scale, Dickson & Millero (1987) Table 5, $0\le S\le 40$:

$$pK_1^* = pK_{1,0} + \left(\frac{-840.39}{T} + 19.894 - 3.0189\ln T\right)S^{1/2} + 0.00668\,S$$

$$pK_2^* = pK_{2,0} + \left(\frac{-690.59}{T} + 17.176 - 2.6719\ln T\right)S^{1/2} + 0.0217\,S$$

$$pK_{1,0} = \frac{6320.81}{T} - 126.3405 + 19.568\ln T, \qquad pK_{2,0} = \frac{5143.69}{T} - 90.1833 + 14.613\ln T$$

Verified against PyCO2SYS v1.8.3: all four formulations agree (Millero 2006 to $4.6\times10^{-7}$).

| Formulation | $pK_1$ | $pK_2$ | Native scale |
|---|---|---|---|
| Lueker 2000 | 5.8472 | 8.9660 | Total |
| Roy 1993 | 5.8563 | 8.9249 | Total |
| Millero 2006 | 5.8498 | 8.9733 | SWS |
| Dickson & Millero 1987 | 5.8532 | 8.9454 | SWS |

## Water 

Seawater Scale, (Millero 1995), converted exactly to Total.

$$\ln K_W = \frac{-13847.26}{T} + 148.9802 - 23.6521\ln T + \left(\frac{118.67}{T} - 5.977 + 1.0495\ln T\right)S^{1/2} - 0.01615\,S$$

## Phosphoric acid

Seawater Scale, (Millero 1995), converted exactly to Total.

$$\ln K_{1P} = \frac{-4576.752}{T} + 115.54 - 18.453\ln T + \left(\frac{-106.736}{T} + 0.69171\right)S^{1/2} + \left(\frac{-0.65643}{T} - 0.01844\right)S$$

$$\ln K_{2P} = \frac{-8814.715}{T} + 172.1033 - 27.927\ln T + \left(\frac{-160.340}{T} + 1.3566\right)S^{1/2} + \left(\frac{0.37335}{T} - 0.05778\right)S$$

$$\ln K_{3P} = \frac{-3070.75}{T} - 18.126 + \left(\frac{17.27039}{T} + 2.81197\right)S^{1/2} + \left(\frac{-44.99486}{T} - 0.09984\right)S$$

The Guide reaches Total by subtracting $0.015$ from each constant term ($115.525$, $172.0883$, $-18.141$). This package converts exactly instead.

## Silicic acid

Seawater Scale, (Millero 1995), converted exactly to Total.

$$\ln K_{Si} = \frac{-8904.2}{T} + 117.40 - 19.334\ln T + \left(\frac{-458.79}{T} + 3.5913\right)I^{1/2} + \left(\frac{188.74}{T} - 1.5998\right)I + \left(\frac{-12.1652}{T} + 0.07871\right)I^{2} + \ln(1-0.001005S)$$

## Calcite and aragonite solubility 

No pH scale, (Mucci 1983).

$$K_{sp} = [\mathrm{Ca^{2+}}][\mathrm{CO_3^{2-}}]$$

Thermodynamic (pure-water) limits, equations:

$$\log_{10}K_{sp}^{\circ}(\text{calcite}) = -171.9065 - 0.077993\,T + \frac{2839.319}{T} + 71.595\log_{10}T$$

$$\log_{10}K_{sp}^{\circ}(\text{aragonite}) = -171.945 - 0.077993\,T + \frac{2903.293}{T} + 71.595\log_{10}T$$

Salinity dependence, equations with coefficients. 

Note that Mucci found $b_3$, $c_1$–$c_3$ and $d_1$–$d_3$ unnecessary, so $B'$ reduces to three terms and $C'$, $D'$ to constants:

$$\log_{10}K_{sp}^{*} = \log_{10}K_{sp}^{\circ} + \left(b_0 + b_1 T + \frac{b_2}{T}\right)S^{1/2} + c_0 S + d_0 S^{3/2}$$

| Solid | $b_0$ | $b_1$ | $b_2$ | $c_0$ | $d_0$ | $\sigma$ |
|---|---|---|---|---|---|---|
| Calcite | $-0.77712$ | $2.8426\times10^{-3}$ | $178.34$ | $-0.07711$ | $4.1249\times10^{-3}$ | 0.010 |
| Aragonite | $-0.068393$ | $1.7276\times10^{-3}$ | $88.135$ | $-0.10018$ | $5.9415\times10^{-3}$ | 0.009 |

Verifying:

$$\log_{10}K_{sp}^{\circ}(\text{arag}) - \log_{10}K_{sp}^{\circ}(\text{calc}) = -0.0385 + \frac{63.974}{T}$$

At $t =25^\circ\mathrm{C}$ a ratio of $1.500$, exactly the value Mucci states he constrained the fit to. Additionally $pK_{sp}$ at $S=35$, $t =25^\circ\mathrm{C}$ reproduces PyCO2SYS to machine precision: calcite $6.36933$, aragonite $6.18831$.

Note: $K_{sp}$ contains no hydrogen ion and is therefore scale-independent. It is never passed through a pH scale conversion.
