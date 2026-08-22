# pH Scale Conversion

Since every acid dissociation constant is linear in $[\mathrm{H^+}]$, a constant transforms with the same factor as the hydrogen ion concentration:

$$\frac{K^{(\text{target})}}{K^{(\text{source})}} = \frac{[\mathrm{H^+}]^{(\text{target})}}{[\mathrm{H^+}]^{(\text{source})}}$$

$$f_{F\to T} = 1 + \frac{S_T}{K_S}, \qquad f_{F\to SWS} = 1 + \frac{S_T}{K_S} + \frac{F_T}{K_F}$$

$$f_{SWS\to T} = \frac{f_{F\to T}}{f_{F\to SWS}}$$

At $S=35$, $t =25^\circ{C}$:

$$f_{F\to T} = 1.28150, \quad f_{F\to SWS} = 1.31039, \quad f_{SWS\to T} = 0.97795$$

So:

$\mathrm{pH_{Free}} - \mathrm{pH_{Total}} = 0.1077$

$\mathrm{pH_{Total}} - \mathrm{pH_{SWS}} = 0.0097$

## Why the Guide's shortcut is rejected

The exact conversion is $\ln f_{SWS\to T} = -0.02229$. The Guide approximates this as $-0.015$, an error of $0.00729$ in $\ln K$, i.e. 0.73 %. The Guide labels this "approximately" and it was appropriate for hand calculation in 2007; it is unnecessary now. Using the exact factor moves this package from $7.3\times10^{-3}$ to $4.6\times10^{-7}$ relative agreement with PyCO2SYS on $K_W$, $K_{1P}$, $K_{2P}$, $K_{3P}$ and $K_{Si}$.

The residual $4.6\times10^{-7}$ is itself fully explained: PyCO2SYS uses $18.998$ for the relative atomic mass of fluorine where IUPAC and the Guide gives $18.9984$.