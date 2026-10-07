"""
Copula modeling module.

Primary model:
    Regular Vine copula using pyvinecopulib.

Fallback:
    Gaussian copula using normal-score correlations.
"""

from itertools import combinations

import numpy as np
from scipy.stats import norm

from src.utils import regularize_corr


try:
    import pyvinecopulib as pv
    HAS_PYVINECOPULIB = True
except Exception:
    pv = None
    HAS_PYVINECOPULIB = False


class VineCopulaModel:
    """
    Multivariate copula model.

    Uses pyvinecopulib if available. Otherwise falls back to a Gaussian copula.
    """

    def __init__(self):
        self.kind = "none"
        self.dim = None
        self.fallback_reason = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, U: np.ndarray):
        """
        Fit copula to uniform matrix U of shape (T, d).
        """
        U = np.asarray(U, dtype=float)

        # Remove invalid rows.
        valid = np.all(np.isfinite(U), axis=1)
        U = U[valid]

        U = np.clip(U, 1e-6, 1.0 - 1e-6)

        if U.ndim != 2:
            raise ValueError("U must be a 2D array.")

        self.dim = U.shape[1]

        # ------------------------------------------------------------
        # Try vine copula
        # ------------------------------------------------------------
        if HAS_PYVINECOPULIB and U.shape[0] >= max(100, 10 * self.dim):
            try:
                controls = None

                # Try BIC-based control first.
                try:
                    controls = pv.FitControlsVinecop(
                        family_set=pv.BicopFamily.all(),
                        selection_criterion="bic",
                    )
                except Exception:
                    try:
                        controls = pv.FitControlsVinecop()
                    except Exception:
                        controls = None

                if controls is not None:
                    try:
                        vine = pv.Vinecop(U, controls=controls)
                    except TypeError:
                        vine = pv.Vinecop(U, fit_controls=controls)
                else:
                    vine = pv.Vinecop(U)

                self.vine = vine
                self.kind = "vine"
                self.fallback_reason = None

                return self

            except Exception as exc:
                self.fallback_reason = f"vine_fit_failed: {type(exc).__name__}: {exc}"

        # ------------------------------------------------------------
        # Gaussian copula fallback
        # ------------------------------------------------------------
        Z = norm.ppf(U)
        Z = np.nan_to_num(Z, nan=0.0, posinf=3.0, neginf=-3.0)

        if Z.shape[0] < 2:
            corr = np.eye(self.dim)
        else:
            corr = np.corrcoef(Z, rowvar=False)

        self.corr = regularize_corr(corr)
        self.kind = "gaussian"
        if self.fallback_reason is None:
            self.fallback_reason = "vine_unavailable_or_insufficient_sample"

        return self

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------
    def simulate(self, n: int, seed: int = None) -> np.ndarray:
        """
        Simulate uniform variates from the fitted copula.

        Returns
        -------
        np.ndarray
            Array of shape (n, d).
        """
        n = int(n)

        # ------------------------------------------------------------
        # Vine simulation
        # ------------------------------------------------------------
        if self.kind == "vine":
            try:
                if seed is not None:
                    rng = np.random.default_rng(seed)
                    seeds = rng.integers(1, 2_000_000_000, size=n).astype(np.int64)

                    try:
                        sim = self.vine.simulate(n, seeds=seeds)
                    except TypeError:
                        sim = self.vine.simulate(n)
                else:
                    sim = self.vine.simulate(n)

                sim = np.asarray(sim, dtype=float)

                # Some versions return shape (dim, n).
                if sim.shape == (self.dim, n):
                    sim = sim.T

                if sim.shape[1] != self.dim and sim.shape[0] == self.dim:
                    sim = sim.T

                sim = np.clip(sim, 1e-6, 1.0 - 1e-6)

                if sim.shape[0] >= n:
                    return sim[:n]

                # If too few simulations, pad with uniforms.
                pad = np.random.default_rng(seed).uniform(
                    low=1e-3,
                    high=1.0 - 1e-3,
                    size=(n - sim.shape[0], self.dim),
                )

                return np.vstack([sim, pad])

            except Exception:
                pass

        # ------------------------------------------------------------
        # Gaussian fallback simulation
        # ------------------------------------------------------------
        rng = np.random.default_rng(seed)

        Z = rng.multivariate_normal(
            mean=np.zeros(self.dim),
            cov=self.corr,
            size=n,
        )

        U = norm.cdf(Z)

        return np.clip(U, 1e-6, 1.0 - 1e-6)

    # ------------------------------------------------------------------
    # Tail dependence diagnostics
    # ------------------------------------------------------------------
    def tail_dependence(self, n: int = 5000, q: float = 0.05, seed: int = 1):
        """
        Estimate average pairwise lower and upper tail dependence.

        Returns
        -------
        lambda_L : float
            Average pairwise lower tail dependence.
        lambda_U : float
            Average pairwise upper tail dependence.
        """
        if self.dim is None or self.dim < 2:
            return 0.0, 0.0

        U = self.simulate(n=n, seed=seed)

        lower_values = []
        upper_values = []

        for i, j in combinations(range(self.dim), 2):
            lower_joint = np.mean((U[:, i] < q) & (U[:, j] < q))
            upper_joint = np.mean((U[:, i] > 1.0 - q) & (U[:, j] > 1.0 - q))

            lower_values.append(lower_joint / q)
            upper_values.append(upper_joint / q)

        lambda_L = float(np.mean(lower_values)) if lower_values else 0.0
        lambda_U = float(np.mean(upper_values)) if upper_values else 0.0

        return lambda_L, lambda_U
