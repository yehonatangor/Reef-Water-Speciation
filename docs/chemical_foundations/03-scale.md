# Scale

Why use the total hydrogen ion scale, and why $\mathrm{mol\,kg\text{-}soln^{-1}}$?

## The four pH scales

A "hydrogen ion concentration" in seawater is ambiguous because the proton is distributed among free $\mathrm{H^+}$, $\mathrm{HSO_4^-}$ and $\mathrm{HF}$. Three self-consistent concentration scales exist:

$$[\mathrm{H^+}]_{\mathrm{Free}} = [\mathrm{H^+}]$$

$$[\mathrm{H^+}]_{\mathrm{Total}} = [\mathrm{H^+}] + [\mathrm{HSO_4^-}] = [\mathrm{H^+}]\left(1 + \frac{S_T}{K_S}\right)$$

$$[\mathrm{H^+}]_{\mathrm{SWS}} = [\mathrm{H^+}] + [\mathrm{HSO_4^-}] + [\mathrm{HF}] = [\mathrm{H^+}]\left(1 + \frac{S_T}{K_S} + \frac{F_T}{K_F}\right)$$

A fourth, the NBS scale, is defined operationally by NIST buffers of ionic strength $\approx 0.1$ and reports an activity of:

$$\mathrm{pH_{NBS}} = -\log_{10} a_{\mathrm{H^+}} = -\log_{10}\!\left(f_H\,[\mathrm{H^+}]_{\mathrm{SWS}}\right)$$

### Justification for choosing Total

**I. It is the scale on which the recommended constants are actually defined.**

The Guide to Best Practices states, *"with the exception of that for bisulfate ion, all acid dissociation constants are expressed in terms of 'total' hydrogen ion concentration."* Choosing Total means $K_1$, $K_2$, $K_B$ and $K_F$ require no conversion at all, every conversion is an opportunity for error, and this allows for one less point of failure.

**II. It is the scale of the certified reference materials.** 

Dickson's Tris CRM, the only traceable pH standard for seawater, is certified on the total scale (Guide SOP, $\mathrm{pH(S)_{TRIS}} = 8.0936$). Any laboratory calibrating an electrode against the CRM is already measuring total-scale pH. Reporting on any other scale requires a conversion that the calibration itself does not need.

**III. It is uniquely well-determined.** 

The Total scale depends on one quantity this package cannot avoid anyway, $K_S$ (Dickson 1990, $\pm 0.0021$ in $\ln K_S$). The SWS scale additionally requires $K_F$, whose free-scale determination (Dickson & Riley 1979) is the weakest constant in the system. The Free scale requires knowing $[\mathrm{HSO_4^-}]$ exactly to define anything measurable. Total minimises propagated uncertainty among the scales that a glass electrode can actually report.

**IV. NBS is not thermodynamically proportional.** 

$f_H$ is medium-specific and carries an uncertainty of order $0.01$ in pH which is larger than the entire Total/SWS difference. Mixing NBS-scale measurements with concentration-scale constants was the original sin of the Mehrbach 1973 dataset and required the Lueker 2000 refit to correct. This package therefore rejects NBS at the API boundary to avoid any issues.

**V. Interoperability.** 

PyCO2SYS, CO2SYS-MATLAB, and the GLODAP data products all default to or report on the Total scale.

**Consequence.** 

SWS-native parameterisations ($K_W$, phosphate, silicate, Millero 2006, Dickson & Millero 1987) are converted to Total by the exact factor, not by an additive approximation.

### Justification for $\mathrm{mol\,kg\text{-}soln^{-1}}$

Every equilibrium constant cited in this package was fitted with concentrations in $\mathrm{mol\,kg\text{-}soln^{-1}}$. Gravimetric concentration is temperature-independent: a sample warmed from $5 ^\circ\text{C}$ to $25 ^\circ\text{C}$ changes volume by 0.3% but its molality is invariant. Since a titration is performed at one temperature and the constants are evaluated at another whenever thermostatting is imperfect, volumetric units introduce a temperature-dependent bias.

Conversion uses the Millero & Poisson (1981) equation of state:

$$c\ [\mathrm{mol\,kg\text{-}soln^{-1}}] = \frac{c\ [\mathrm{mol\,L^{-1}}]}{\rho_{SW}/1000}$$

At $S=35$, $T=25^\circ\mathrm{C}$, $\rho_{SW} = 1023.343\ \mathrm{kg\,m^{-3}}$:

This means a numerically equal volumetric concentration is 2.3% larger than the gravimetric one. Applied to $A_T \approx 2300\ \mu\mathrm{mol\,kg^{-1}}$, the error from ignoring this is $\approx 54\ \mu\mathrm{mol\,kg^{-1}}$, roughly thirty times the reproducibility of a good alkalinity titration.