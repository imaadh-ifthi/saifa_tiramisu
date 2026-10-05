"""
Full model classes.

1. MultiScaleWaveletVine:
   Wavelet decomposition + scale-specific marginals + scale-specific copulas.

2. GaussianAggregateBenchmark:
   Simple Gaussian copula benchmark on aggregate returns.
"""

import numpy as np

from src.modwt import AdditiveMODWT
from src.marginals import MarginalARQGARCH
from src.copulas import VineCopulaModel
from src.utils import regularize_corr
from src.cross_scale import CrossScaleCopulaCoupler
from scipy.stats import norm

class MultiScaleWaveletVine:
    """
    Multiscale wavelet-vine risk model.
    """

    def __init__(
        self,
        levels: int = 5,
        wavelet: str = "db4",
        seed: int = 42,
        tail_quantile: float = 0.95,
    ):
        self.levels = int(levels)
        self.wavelet = wavelet
        self.seed = int(seed)
        self.tail_quantile = float(tail_quantile)

        self.scale_models = {}
        self.tail_deps = {}
        self.scale_order = []

    # ------------------------------------------------------------------
    # Fit model
    # ------------------------------------------------------------------
    def fit(self, returns):
        """
        Fit the full multiscale model.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns in training window.
        """
        returns = returns.dropna()

        self.asset_names = list(returns.columns)
        self.n_assets = len(self.asset_names)

        decomp = AdditiveMODWT(wavelet=self.wavelet, levels=self.levels)

        details_by_asset = []
        smooth_by_asset = []

        # ------------------------------------------------------------
        # Decompose each asset
        # ------------------------------------------------------------
        for col in self.asset_names:
            x = returns[col].values
            details, smooth = decomp.decompose(x)

            details_by_asset.append(details)
            smooth_by_asset.append(smooth)

        # ------------------------------------------------------------
        # Fit scale-specific models
        # ------------------------------------------------------------
        self.scale_models = {}
        self.tail_deps = {}
        self.scale_order = []
        stress_matrix_cols = []

        for j in range(self.levels):
            scale_name = f"D{j + 1}"

            X = np.column_stack(
                [details_by_asset[i][j] for i in range(self.n_assets)]
            )

            U = self._fit_scale(scale_name, X)
            U_clipped = np.clip(U, 1e-6, 1.0 - 1e-6)
            stress_matrix_cols.append(np.mean(norm.ppf(U_clipped), axis=1))
            self.scale_order.append(scale_name)

        # Smooth component.
        scale_name = f"S{self.levels}"
        X_smooth = np.column_stack(smooth_by_asset)

        U = self._fit_scale(scale_name, X_smooth)
        U_clipped = np.clip(U, 1e-6, 1.0 - 1e-6)
        stress_matrix_cols.append(np.mean(norm.ppf(U_clipped), axis=1))
        self.scale_order.append(scale_name)

        # Fit cross-scale model
        stress_matrix = np.column_stack(stress_matrix_cols)
        self.cross_scale_coupler = CrossScaleCopulaCoupler(scale_names=self.scale_order)
        self.cross_scale_coupler.fit(stress_matrix)

        return self

    def _fit_scale(self, scale_name: str, X: np.ndarray):
        """
        Fit marginals and copula for one scale.
        """
        n_obs, n_assets = X.shape

        marginals = []
        U_list = []

        # Fit marginal model for each asset at this scale.
        for i in range(n_assets):
            m = MarginalARQGARCH(tail_quantile=self.tail_quantile)
            m.fit(X[:, i])

            u = m.transform(X[:, i])

            marginals.append(m)
            U_list.append(u)

        U = np.column_stack(U_list)

        cop = VineCopulaModel()
        cop.fit(U)

        lambda_L, lambda_U = cop.tail_dependence(
            n=5000,
            q=0.05,
            seed=self.seed,
        )

        self.scale_models[scale_name] = {
            "marginals": marginals,
            "copula": cop,
        }

        self.tail_deps[scale_name] = (lambda_L, lambda_U)
        
        return U

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------
    def simulate_portfolio(self, n_sim: int, weights: np.ndarray, seed: int = 42):
        """
        Simulate portfolio returns by simulating each scale, coupling them, and reconstructing.
        """
        n_sim = int(n_sim)
        weights = np.asarray(weights, dtype=float)

        scale_uniform_samples = {}

        for idx, scale_name in enumerate(self.scale_order):
            model = self.scale_models[scale_name]

            cop = model["copula"]

            U = cop.simulate(n_sim, seed=seed + idx + 1)

            # Safety check.
            if U.shape[0] < n_sim:
                pad = np.full((n_sim - U.shape[0], U.shape[1]), 0.5)
                U = np.vstack([U, pad])
            elif U.shape[0] > n_sim:
                U = U[:n_sim]
                
            scale_uniform_samples[scale_name] = U

        # Cross-scale coupling
        if hasattr(self, 'cross_scale_coupler'):
            coupled_samples = self.cross_scale_coupler.couple(scale_uniform_samples, seed=seed + 999)
        else:
            coupled_samples = scale_uniform_samples

        total_asset_returns = np.zeros((n_sim, self.n_assets), dtype=float)

        for scale_name in self.scale_order:
            model = self.scale_models[scale_name]
            marginals = model["marginals"]
            U_coupled = coupled_samples[scale_name]

            scale_components = np.zeros((n_sim, self.n_assets), dtype=float)

            for i, m in enumerate(marginals):
                scale_components[:, i] = m.forecast_component(U_coupled[:, i])

            total_asset_returns += scale_components

        portfolio_returns = total_asset_returns @ weights

        return portfolio_returns


class GaussianAggregateBenchmark:
    """
    Simple Gaussian copula benchmark applied directly to aggregate returns.

    This benchmark ignores:
        - heavy tails;
        - timescale separation;
        - asymmetric dependence.
    """

    def __init__(self):
        self.mu = None
        self.sigma = None
        self.corr = None
        self.dim = None

    def fit(self, returns):
        """
        Fit Gaussian benchmark.
        """
        returns = returns.dropna()

        self.asset_names = list(returns.columns)
        self.dim = len(self.asset_names)

        self.mu = returns.mean().values
        self.sigma = returns.std().values

        self.sigma = np.where(self.sigma < 1e-8, 1e-8, self.sigma)

        Z = (returns.values - self.mu) / self.sigma
        Z = np.nan_to_num(Z, nan=0.0)

        if Z.shape[0] < 2:
            corr = np.eye(self.dim)
        else:
            corr = np.corrcoef(Z, rowvar=False)

        self.corr = regularize_corr(corr)

        return self

    def simulate_portfolio(self, n_sim: int, weights: np.ndarray, seed: int = 1):
        """
        Simulate portfolio returns from Gaussian copula benchmark.
        """
        rng = np.random.default_rng(seed)

        Z = rng.multivariate_normal(
            mean=np.zeros(self.dim),
            cov=self.corr,
            size=int(n_sim),
        )

        X = self.mu + self.sigma * Z

        portfolio_returns = X @ np.asarray(weights, dtype=float)

        return portfolio_returns
