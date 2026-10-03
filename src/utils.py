"""
Utility functions.
"""

import numpy as np


def regularize_corr(corr: np.ndarray) -> np.ndarray:
    """
    Regularize a correlation matrix so that it is symmetric and positive definite.
    """
    corr = np.asarray(corr, dtype=float)
    corr = np.nan_to_num(corr, nan=0.0)

    # Force symmetry.
    corr = 0.5 * (corr + corr.T)

    # Force unit diagonal.
    np.fill_diagonal(corr, 1.0)

    try:
        np.linalg.cholesky(corr)
        return corr
    except np.linalg.LinAlgError:
        eigval, eigvec = np.linalg.eigh(corr)
        eigval = np.clip(eigval, 1e-8, None)
        corr = eigvec @ np.diag(eigval) @ eigvec.T

        d = np.sqrt(np.diag(corr))
        d[d == 0] = 1.0

        corr = corr / d[:, None] / d[None, :]
        np.fill_diagonal(corr, 1.0)

        # Final symmetry cleanup.
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)

        return corr
