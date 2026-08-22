# Numerical method

The residual $f(\mathrm{pH}) = A_T^{\mathrm{model}}(\mathrm{pH}) - A_T^{\mathrm{obs}}$ is smooth and strictly monotone decreasing in $\mathrm{pH}$, ensuring that root bracketing is unconditionally safe.

The solver initializes on the bracket $[a_0, b_0] = [6.0, 9.5]$, chosen so that typical seawater samples converge without interval expansion. If $f(a_0) \cdot f(b_0) > 0$, the interval expands geometrically toward hard limits $[1.0, 13.0]$ until a sign change is detected, at which point Brent's method is invoked.

## Brent's Root-Finding

Given a current bracket $[a_k, b_k]$ with a previous iterate $c_k$ such that $f(b_k) \cdot f(a_k) < 0$ and $|f(b_k)| \le |f(a_k)|$, the algorithm dynamically selects between three interpolation strategies to compute the next iterate $s$:

* **Inverse Quadratic Interpolation (IQI):** Evaluated when $f(a_k) \neq f(c_k)$ and $f(b_k) \neq f(c_k)$:
$$s = \frac{a_k f(b_k) f(c_k)}{(f(a_k) - f(b_k))(f(a_k) - f(c_k))} + \frac{b_k f(a_k) f(c_k)}{(f(b_k) - f(a_k))(f(b_k) - f(c_k))} + \frac{c_k f(a_k) f(b_k)}{(f(c_k) - f(a_k))(f(c_k) - f(b_k))}$$

* **Secant Method:** Used when $f(a_k) = f(c_k)$:
$$s = b_k - f(b_k) \frac{b_k - a_k}{f(b_k) - f(a_k)}$$

* **Bisection Fallback:** Selected if the candidate step $s$ violates bounds or contracts too slowly ($|s - b_k| \ge \frac{1}{2} |b_k - c_k|$):
$$s = \frac{a_k + b_k}{2}$$

The bracket is updated to maintain $f(a_{k+1}) \cdot f(b_{k+1}) \le 0$ until convergence satisfies $|b_k - a_k| \le 2\epsilon_{\mathrm{tol}}$, combining the superlinear speed of open methods with the unconditional stability of bisection.