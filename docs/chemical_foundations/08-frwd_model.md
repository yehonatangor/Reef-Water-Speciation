# The titration forward model

Following Dickson (1981), for sample mass $m_0$ and titrant mass $m$:

$$A_T^{\mathrm{model}}(\mathrm{pH}) = \frac{m_0 A_T - m\,C_a\,\bar{n}(\mathrm{pH})}{m_0 + m}$$

where every conservative total is diluted by $m_0/(m_0+m)$ and $\bar{n}$ is the mean number of protons released per mole of titrant. For a strong acid $\bar{n} \equiv 1$. For an $n$-protic weak acid:

$$\bar{n}(h) = \frac{\sum_{j=1}^{n} j\left(\prod_{i=1}^{j}K_i\right)h^{\,n-j}} {h^{n} + \sum_{j=1}^{n}\left(\prod_{i=1}^{j}K_i\right)h^{\,n-j}}$$

The equivalence mass is exact:

$$m_{eq} = \frac{m_0 A_T}{C_{a,\text{titratable}}}$$

The titrant used by this package is $0.1\ \mathrm{mol\,kg^{-1}}$ HCl in $0.6\ \mathrm{mol\,kg^{-1}}$ NaCl. (Guide SOP 3b), for which $\bar{n} \equiv 1$ identically.

Verified: for $A_T = 2300$, $DIC = 2000\ \mu\mathrm{mol\,kg^{-1}}$, $S=35$, $t=25^\circ{C}$, HCl titrant, the curve starts at pH 8.0459 (identical to the independent DIC/$A_T$ solver to $10^{-9}$), is strictly monotone, and passes through pH 4.34 at the equivalence point, inside Dickson's expected 4.5 region.
