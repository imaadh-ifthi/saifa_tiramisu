import numpy as np
from scipy.stats import norm
from src.utils import regularize_corr

class CrossScaleCopulaCoupler:
    """
    Couples independently simulated scale scenarios by modelling their cross-scale
    dependence. Fits a Gaussian copula on the empirical ranks of the scalar
    scale-level stress scores, then reorders the simulated scenarios to match
    the simulated copula structure.
    """
    def __init__(self, scale_names=None):
        self.scale_names = scale_names
        self.n_scales = 0
        self.corr_matrix = None
        self.fitted = False
        
        # Diagnostics
        self.diag_historical_corr = None
        self.diag_simulated_before_corr = None
        self.diag_simulated_after_corr = None

    def fit(self, stress_matrix: np.ndarray):
        """
        Fit the cross-scale Gaussian copula using ranks of the stress matrix.
        """
        stress_matrix = np.asarray(stress_matrix, dtype=float)
        T, self.n_scales = stress_matrix.shape
        
        # 1. Convert historical stress scores to ranks
        ranks = np.argsort(np.argsort(stress_matrix, axis=0), axis=0)
        
        # 2. Convert ranks to standard-normal scores
        p = (ranks + 0.5) / T
        z = norm.ppf(p)
        
        # 3. Estimate correlation matrix
        if T < 2:
            corr = np.eye(self.n_scales)
        else:
            corr = np.corrcoef(z, rowvar=False)
            
        # 4. Regularize correlation matrix
        try:
            self.corr_matrix = regularize_corr(corr)
            if not np.all(np.isfinite(self.corr_matrix)):
                raise ValueError("Non-finite correlation matrix.")
        except Exception as e:
            print(f"[CrossScaleCopulaCoupler] Fallback to identity matrix due to: {e}")
            self.corr_matrix = np.eye(self.n_scales)
            
        self.diag_historical_corr = self.corr_matrix.copy()
        self.fitted = True
        return self

    def couple(self, scale_uniform_samples: dict, seed: int = 42):
        """
        Couple independently simulated scale scenarios by reordering their rows.
        """
        if not self.fitted:
            print("[CrossScaleCopulaCoupler] Not fitted. Returning uncoupled samples.")
            return scale_uniform_samples
            
        if self.scale_names is None:
            self.scale_names = list(scale_uniform_samples.keys())
            
        n_sim = len(scale_uniform_samples[self.scale_names[0]])
        
        # Calculate simulated stress matrix BEFORE coupling
        sim_stress = np.zeros((n_sim, self.n_scales))
        for j, scale in enumerate(self.scale_names):
            U = scale_uniform_samples[scale]
            U_clipped = np.clip(U, 1e-6, 1.0 - 1e-6)
            sim_stress[:, j] = np.mean(norm.ppf(U_clipped), axis=1)
            
        if n_sim > 1:
            self.diag_simulated_before_corr = np.corrcoef(sim_stress, rowvar=False)
        else:
            self.diag_simulated_before_corr = np.eye(self.n_scales)

        # Simulate cross-scale latent Gaussian sample
        rng = np.random.default_rng(seed)
        try:
            Z_scale = rng.multivariate_normal(
                mean=np.zeros(self.n_scales),
                cov=self.corr_matrix,
                size=n_sim,
                method='cholesky'
            )
        except Exception:
            Z_scale = rng.multivariate_normal(
                mean=np.zeros(self.n_scales),
                cov=np.eye(self.n_scales),
                size=n_sim,
            )

        coupled_samples = {}
        coupled_stress = np.zeros((n_sim, self.n_scales))
        
        for j, scale in enumerate(self.scale_names):
            U = scale_uniform_samples[scale]
            current_stress = sim_stress[:, j]
            target_latent = Z_scale[:, j]
            
            # Sort current scenarios by their stress
            sort_idx_current = np.argsort(current_stress)
            
            # Sort target latent values to find their ranks
            sort_idx_target = np.argsort(target_latent)
            
            # Create the permuted U array
            U_coupled = np.zeros_like(U)
            U_coupled[sort_idx_target] = U[sort_idx_current]
            
            coupled_samples[scale] = U_coupled
            coupled_stress[sort_idx_target, j] = current_stress[sort_idx_current]

        if n_sim > 1:
            self.diag_simulated_after_corr = np.corrcoef(coupled_stress, rowvar=False)
        else:
            self.diag_simulated_after_corr = np.eye(self.n_scales)

        return coupled_samples
