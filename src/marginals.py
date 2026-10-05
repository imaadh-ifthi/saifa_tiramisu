"""
Marginal distribution modeling.

For each asset and each timescale, we fit:

1. AR(1)-GARCH(1,1) filter.
2. Standardized residuals.
3. EVT Peaks-Over-Threshold tails using Generalized Pareto distributions.
4. Probability Integral Transform to uniforms.
"""

import warnings

import numpy as np
from arch import arch_model
from scipy.stats import genpareto


class MarginalARQGARCH:
    """
    Marginal model:
        AR(1)-GARCH(1,1) + EVT tail splicing.
    """

    def __init__(self, tail_quantile: float = 0.95):
        self.tail_quantile = float(tail_quantile)
        self.fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ewma_z(self, y: np.ndarray):
        """
        Fallback EWMA volatility standardization.
        """
        mu = float(np.mean(y))
        eps = y - mu

        sigma2 = np.empty_like(y, dtype=float)
        sigma2[0] = max(float(np.var(eps)), 1e-12)

        lam = 0.94

        for t in range(1, len(y)):
            sigma2[t] = lam * sigma2[t - 1] + (1.0 - lam) * eps[t - 1] ** 2

        sigma = np.sqrt(np.maximum(sigma2, 1e-12))
        z = eps / sigma

        var_next = lam * sigma2[-1] + (1.0 - lam) * eps[-1] ** 2

        return z, mu, var_next

    def _get_param(self, params: dict, candidates, default: float) -> float:
        """
        Robust parameter extraction from arch model params.
        """
        for key in candidates:
            if key in params:
                try:
                    return float(params[key])
                except Exception:
                    continue
        return float(default)

    def _fit_gpd(self, excess: np.ndarray):
        """
        Fit GPD to positive exceedances.
        """
        excess = np.asarray(excess, dtype=float)
        excess = excess[np.isfinite(excess)]
        excess = excess[excess > 0]

        if len(excess) < 10:
            beta = max(float(np.std(excess)) if len(excess) > 0 else 1e-6, 1e-6)
            return 0.0, beta

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xi, loc, beta = genpareto.fit(excess, floc=0)

            if not np.isfinite(beta) or beta <= 0:
                raise ValueError("Invalid GPD scale.")

            xi = float(np.clip(xi, -0.49, 0.49))
            beta = float(beta)

            return xi, beta

        except Exception:
            beta = max(float(np.mean(excess)), 1e-6)
            return 0.0, beta

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, x: np.ndarray):
        """
        Fit marginal model to a timescale component.
        """
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]

        if x.size == 0:
            x = np.zeros(200, dtype=float)

        if x.size < 150:
            x = np.concatenate([x, np.zeros(150 - x.size, dtype=float)])

        # Scale small financial coefficients for numerical stability.
        max_abs = float(np.max(np.abs(x))) if len(x) > 0 else 1.0
        self.scale = 100.0 if max_abs < 1.0 else 1.0

        y = x * self.scale

        mean_next_y = float(np.mean(y))
        var_next_y = float(np.var(y) + 1e-12)
        z = None

        # --------------------------------------------------------------
        # Try AR(1)-GARCH(1,1)
        # --------------------------------------------------------------
        try:
            am = arch_model(
                y,
                mean="AR",
                lags=1,
                vol="GARCH",
                p=1,
                q=1,
                dist="normal",
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = am.fit(disp="off", options={"maxiter": 200})

            if res.convergence_flag != 0:
                raise RuntimeError("GARCH optimizer failed to converge.")

            params = res.params.to_dict()

            mu = self._get_param(params, ["mu", "constant", "c"], 0.0)
            phi = self._get_param(params, ["ar[1]", "ar[1]", "lag[1]"], 0.0)
            omega = self._get_param(params, ["omega"], np.var(y) * 0.05)
            alpha = self._get_param(params, ["alpha[1]", "alpha"], 0.05)
            beta = self._get_param(params, ["beta[1]", "beta"], 0.90)

            # Parameter sanity.
            omega = max(float(omega), 1e-12)
            alpha = max(0.0, min(float(alpha), 0.999))
            beta = max(0.0, min(float(beta), 0.999))

            total = alpha + beta
            if total >= 0.999:
                scale_down = 0.998 / total
                alpha *= scale_down
                beta *= scale_down

            cond_vol = np.asarray(res.conditional_volatility, dtype=float)
            resid = np.asarray(res.resid, dtype=float)
            std_resid = np.asarray(res.std_resid, dtype=float)

            cond_vol = np.nan_to_num(cond_vol, nan=np.nanstd(y))
            resid = np.nan_to_num(resid, nan=0.0)
            std_resid = std_resid[np.isfinite(std_resid)]

            if len(std_resid) < 100:
                raise ValueError("Too few valid standardized residuals.")

            sigma_last = float(cond_vol[-1]) if np.isfinite(cond_vol[-1]) else float(np.std(y))
            eps_last = float(resid[-1]) if np.isfinite(resid[-1]) else float(y[-1] - mu)

            mean_next_y = mu + phi * (y[-1] - mu)
            var_next_y = omega + alpha * eps_last**2 + beta * sigma_last**2
            var_next_y = max(float(var_next_y), 1e-12)

            z = std_resid

        except Exception:
            # ----------------------------------------------------------
            # Fallback EWMA
            # ----------------------------------------------------------
            z, mu_y, var_next_y = self._ewma_z(y)
            mean_next_y = mu_y
            var_next_y = max(float(var_next_y), 1e-12)

        # --------------------------------------------------------------
        # Clean standardized residuals
        # --------------------------------------------------------------
        z = np.asarray(z, dtype=float)
        z = z[np.isfinite(z)]

        if len(z) < 100:
            z = (y - np.mean(y)) / max(np.std(y), 1e-12)

        z = np.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
        z = np.clip(z, -15.0, 15.0)

        if np.std(z) < 1e-8:
            rng = np.random.default_rng(0)
            z = z + rng.normal(0.0, 1e-4, size=len(z))

        self.z = z

        # --------------------------------------------------------------
        # Forecast parameters in original scale
        # --------------------------------------------------------------
        self.forecast_mean = float(mean_next_y / self.scale)
        self.forecast_sigma = float(np.sqrt(var_next_y) / self.scale)

        # --------------------------------------------------------------
        # EVT tail splicing
        # --------------------------------------------------------------
        lower_q = max(1.0 - self.tail_quantile, 0.01)
        upper_q = min(self.tail_quantile, 0.99)

        self.p_lower = float(lower_q)
        self.p_upper = float(upper_q)

        self.lower_thr = float(np.quantile(z, lower_q))
        self.upper_thr = float(np.quantile(z, upper_q))

        if self.upper_thr <= self.lower_thr:
            self.lower_thr = float(np.quantile(z, 0.05))
            self.upper_thr = float(np.quantile(z, 0.95))
            self.p_lower = 0.05
            self.p_upper = 0.95

        upper_excess = z[z > self.upper_thr] - self.upper_thr
        lower_excess = self.lower_thr - z[z < self.lower_thr]

        self.xi_u, self.beta_u = self._fit_gpd(upper_excess)
        self.xi_l, self.beta_l = self._fit_gpd(lower_excess)

        interior = z[(z >= self.lower_thr) & (z <= self.upper_thr)]

        if len(interior) < 50:
            interior = z

        self.interior = np.sort(interior)

        self.fitted = True

        return self

    # ------------------------------------------------------------------
    # Transform residuals to uniforms
    # ------------------------------------------------------------------
    def transform(self, z: np.ndarray) -> np.ndarray:
        """
        Probability Integral Transform.
        """
        if not self.fitted:
            raise RuntimeError("Marginal model is not fitted.")

        z = np.asarray(z, dtype=float)
        u = np.empty_like(z, dtype=float)

        lower_mask = z <= self.lower_thr
        upper_mask = z >= self.upper_thr
        interior_mask = ~(lower_mask | upper_mask)

        # Lower tail (survival-probability formulation).
        # As z decreases below the threshold, excess grows and G(excess)->1,
        # so (1 - G(excess))->0, giving u->0 for extreme losses.
        # At the threshold itself, excess=0, G(0)=0, so u = p_lower.
        if np.any(lower_mask):
            excess = self.lower_thr - z[lower_mask]
            excess = np.maximum(excess, 0.0)

            cond = genpareto.cdf(excess, self.xi_l, loc=0.0, scale=self.beta_l)
            cond = np.clip(cond, 0.0, 1.0)

            u[lower_mask] = self.p_lower * (1.0 - cond)

        # Upper tail.
        if np.any(upper_mask):
            excess = z[upper_mask] - self.upper_thr
            excess = np.maximum(excess, 0.0)

            cond = genpareto.cdf(excess, self.xi_u, loc=0.0, scale=self.beta_u)
            cond = np.clip(cond, 0.0, 1.0)

            u[upper_mask] = self.p_upper + (1.0 - self.p_upper) * cond

        # Interior empirical CDF.
        if np.any(interior_mask):
            ranks = np.searchsorted(self.interior, z[interior_mask], side="right")
            ranks = ranks / float(len(self.interior))
            ranks = np.clip(ranks, 0.0, 1.0)

            u[interior_mask] = self.p_lower + (self.p_upper - self.p_lower) * ranks

        return np.clip(u, 1e-6, 1.0 - 1e-6)

    # ------------------------------------------------------------------
    # Inverse transform uniforms to standardized residuals
    # ------------------------------------------------------------------
    def inverse_pit(self, u: np.ndarray) -> np.ndarray:
        """
        Inverse Probability Integral Transform.
        """
        if not self.fitted:
            raise RuntimeError("Marginal model is not fitted.")

        u = np.asarray(u, dtype=float)
        u = np.clip(u, 1e-6, 1.0 - 1e-6)

        z = np.empty_like(u, dtype=float)

        lower_mask = u <= self.p_lower
        upper_mask = u >= self.p_upper
        interior_mask = ~(lower_mask | upper_mask)

        # Lower tail (inverse of survival-probability formulation).
        # Forward: u = p_lower * (1 - G(excess))  =>  G(excess) = 1 - u/p_lower
        # So we invert G at (1 - u/p_lower) to recover excess.
        if np.any(lower_mask):
            cond = 1.0 - u[lower_mask] / self.p_lower
            cond = np.clip(cond, 1e-8, 1.0 - 1e-8)

            excess = genpareto.ppf(cond, self.xi_l, loc=0.0, scale=self.beta_l)
            excess = np.nan_to_num(excess, nan=0.0, posinf=10.0 * self.beta_l, neginf=0.0)

            z[lower_mask] = self.lower_thr - excess

        # Upper tail.
        if np.any(upper_mask):
            cond = (u[upper_mask] - self.p_upper) / max(1.0 - self.p_upper, 1e-12)
            cond = np.clip(cond, 1e-8, 1.0 - 1e-8)

            excess = genpareto.ppf(cond, self.xi_u, loc=0.0, scale=self.beta_u)
            excess = np.nan_to_num(excess, nan=0.0, posinf=10.0 * self.beta_u, neginf=0.0)

            z[upper_mask] = self.upper_thr + excess

        # Interior empirical quantile.
        if np.any(interior_mask):
            q = (u[interior_mask] - self.p_lower) / max(self.p_upper - self.p_lower, 1e-12)
            q = np.clip(q, 0.0, 1.0)

            z[interior_mask] = np.quantile(self.interior, q)

        return z

    # ------------------------------------------------------------------
    # Forecast one-step component from uniform draws
    # ------------------------------------------------------------------
    def forecast_component(self, u: np.ndarray) -> np.ndarray:
        """
        Convert copula uniform draws into forecasted wavelet-scale component.
        """
        z = self.inverse_pit(u)
        return self.forecast_mean + self.forecast_sigma * z
