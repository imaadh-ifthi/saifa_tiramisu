"""
Statistical backtesting functions.

Includes:
    - Kupiec VaR test;
    - Christoffersen independence diagnostic;
    - Fissler-Ziegel joint VaR/ES loss;
    - Diebold-Mariano test.
"""

import numpy as np
from scipy.stats import chi2, norm


def kupiec_test(actual: np.ndarray, var_forecast: np.ndarray, alpha: float = 0.01):
    """
    Kupiec proportion-of-failures test.

    H0: VaR exceptions occur with probability alpha.
    """
    actual = np.asarray(actual, dtype=float)
    var_forecast = np.asarray(var_forecast, dtype=float)

    violations = (actual < var_forecast).astype(int)

    T = len(violations)
    x = int(violations.sum())

    if T == 0:
        return {"stat": np.nan, "p_value": np.nan, "violation_rate": np.nan}

    pi_hat = x / T

    eps = 1e-12

    log_likelihood_null = (
        (T - x) * np.log(max(1.0 - alpha, eps))
        + x * np.log(max(alpha, eps))
    )

    log_likelihood_alt = (
        (T - x) * np.log(max(1.0 - pi_hat, eps))
        + x * np.log(max(pi_hat, eps))
    )

    lr = -2.0 * (log_likelihood_null - log_likelihood_alt)
    lr = max(float(lr), 0.0)

    p_value = float(chi2.sf(lr, df=1))

    return {
        "stat": lr,
        "p_value": p_value,
        "violation_rate": pi_hat,
    }


def christoffersen_test(actual: np.ndarray, var_forecast: np.ndarray, alpha: float = 0.01):
    """
    Christoffersen independence test.

    This tests whether VaR violations are independent over time.
    """
    actual = np.asarray(actual, dtype=float)
    var_forecast = np.asarray(var_forecast, dtype=float)

    violations = (actual < var_forecast).astype(int)

    if len(violations) < 3:
        return {"stat": np.nan, "p_value": np.nan}

    n00 = n01 = n10 = n11 = 0

    for t in range(1, len(violations)):
        prev = violations[t - 1]
        curr = violations[t]

        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        elif prev == 1 and curr == 1:
            n11 += 1

    def safe_ll(n_success_before_0, n_success_before_1, pi):
        pi = min(max(float(pi), 1e-12), 1.0 - 1e-12)
        return (
            n_success_before_0 * np.log(1.0 - pi)
            + n_success_before_1 * np.log(pi)
        )

    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    log_likelihood_alt = safe_ll(n00, n01, pi0) + safe_ll(n10, n11, pi1)
    log_likelihood_null = safe_ll(n00 + n10, n01 + n11, pi)

    lr = -2.0 * (log_likelihood_null - log_likelihood_alt)
    lr = max(float(lr), 0.0)

    p_value = float(chi2.sf(lr, df=1))

    return {
        "stat": lr,
        "p_value": p_value,
    }


def fissler_ziegel_loss(
    actual: np.ndarray,
    var_forecast: np.ndarray,
    es_forecast: np.ndarray,
    alpha: float = 0.01,
):
    """
    Fissler-Ziegel FZ0 joint loss function for VaR and Expected Shortfall.

    This implementation follows the return-based sign convention:

        - actual returns are negative in losses;
        - VaR forecasts are negative tail quantiles;
        - ES forecasts are negative tail averages.

    Lower loss is better.
    """
    actual = np.asarray(actual, dtype=float)
    var_forecast = np.asarray(var_forecast, dtype=float)
    es_forecast = np.asarray(es_forecast, dtype=float)

    # ES must be negative under this convention.
    es_safe = np.where(es_forecast < -1e-12, es_forecast, -1e-12)

    indicator = (actual <= var_forecast).astype(float)

    term1 = -(indicator / (alpha * es_safe)) * (var_forecast - actual)
    term2 = var_forecast / es_safe
    term3 = np.log(-es_safe) - 1.0

    loss = term1 + term2 + term3

    loss = np.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=-1e6)

    return loss


def diebold_mariano_test(loss1: np.ndarray, loss2: np.ndarray):
    """
    Diebold-Mariano test for equal predictive accuracy.

    loss1 is usually the proposed model.
    loss2 is usually the benchmark.

    If loss1 has significantly lower mean loss, the DM statistic is negative.
    """
    loss1 = np.asarray(loss1, dtype=float)
    loss2 = np.asarray(loss2, dtype=float)

    d = loss1 - loss2
    d = d[np.isfinite(d)]

    n = len(d)

    if n < 3:
        return {"stat": np.nan, "p_value": np.nan}

    d_mean = float(np.mean(d))

    gamma0 = float(np.var(d, ddof=0))

    if n > 2:
        gamma1 = float(np.mean((d[:-1] - d_mean) * (d[1:] - d_mean)))
    else:
        gamma1 = 0.0

    var_d = (gamma0 + 2.0 * gamma1) / n

    if var_d <= 0:
        var_d = gamma0 / n

    if var_d <= 0:
        return {"stat": 0.0, "p_value": 1.0}

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = float(2.0 * norm.sf(abs(dm_stat)))

    return {
        "stat": float(dm_stat),
        "p_value": p_value,
    }
