"""
Risk measurement utilities.
"""

import numpy as np


def var_es(returns: np.ndarray, alpha: float = 0.01):
    """
    Compute Value-at-Risk and Expected Shortfall from simulated returns.

    Returns are expressed as arithmetic/log returns, so negative values are losses.

    Parameters
    ----------
    returns : np.ndarray
        Simulated portfolio returns.
    alpha : float
        Tail probability, e.g. 0.01 for 99% VaR/ES.

    Returns
    -------
    var : float
        alpha-quantile of returns.
    es : float
        average return conditional on return <= VaR.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    if len(returns) == 0:
        return 0.0, 0.0

    var = float(np.percentile(returns, 100.0 * alpha))

    tail = returns[returns <= var]

    if len(tail) == 0:
        es = var
    else:
        es = float(np.mean(tail))

    return var, es
