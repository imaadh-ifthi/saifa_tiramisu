"""
Out-of-sample backtest engine.
"""

import numpy as np
import pandas as pd

from src.models import MultiScaleWaveletVine, GaussianAggregateBenchmark
from src.risk import var_es


def run_backtest(returns: pd.DataFrame, cfg):
    """
    Run rolling out-of-sample backtest.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily asset returns.
    cfg : module
        Config module.

    Returns
    -------
    results : pd.DataFrame
        Daily actual returns, VaR and ES forecasts for both models.
    tail_df : pd.DataFrame
        Estimated tail dependence by scale at each model refit.
    """
    returns = returns.dropna()

    n = len(returns)

    if n <= cfg.TRAIN_WINDOW + 10:
        raise RuntimeError("Not enough data for the selected training window.")

    start = cfg.TRAIN_WINDOW

    if cfg.TEST_DAYS is None:
        end = n
    else:
        end = min(n, start + int(cfg.TEST_DAYS))

    records = []
    tail_records = []

    model = None
    benchmark = None

    weights = np.asarray(cfg.WEIGHTS, dtype=float)
    weights = weights / weights.sum()

    total_days = end - start

    print(f"[Backtest] Starting backtest with {total_days} out-of-sample days.")

    for t in range(start, end):
        date = returns.index[t]

        train = returns.iloc[t - cfg.TRAIN_WINDOW : t]
        actual_return = float(returns.iloc[t].values @ weights)

        need_refit = (model is None) or ((t - start) % cfg.REFIT_EVERY == 0)

        if need_refit:
            print(f"[Backtest] Fitting models on window ending {date.date()}")

            model = MultiScaleWaveletVine(
                levels=cfg.LEVELS,
                wavelet=cfg.WAVELET,
                seed=cfg.SEED,
                tail_quantile=cfg.TAIL_QUANTILE,
            )
            model.fit(train)

            benchmark = GaussianAggregateBenchmark()
            benchmark.fit(train)

            tail_row = {"date": date}

            for scale_name, (lambda_l, lambda_u) in model.tail_deps.items():
                tail_row[f"{scale_name}_L"] = lambda_l
                tail_row[f"{scale_name}_U"] = lambda_u

            tail_records.append(tail_row)

        # --------------------------------------------------------------
        # Forecast distribution for date t
        # --------------------------------------------------------------
        sim_model = model.simulate_portfolio(
            n_sim=cfg.N_SIM,
            weights=weights,
            seed=cfg.SEED + t,
        )

        sim_bench = benchmark.simulate_portfolio(
            n_sim=cfg.N_SIM,
            weights=weights,
            seed=cfg.SEED + 1_000_000 + t,
        )

        model_var, model_es = var_es(sim_model, alpha=cfg.ALPHA)
        bench_var, bench_es = var_es(sim_bench, alpha=cfg.ALPHA)

        records.append(
            {
                "date": date,
                "actual_return": actual_return,
                "model_var": model_var,
                "model_es": model_es,
                "bench_var": bench_var,
                "bench_es": bench_es,
            }
        )

        completed = t - start + 1
        if completed % 10 == 0:
            print(f"[Backtest] Completed {completed}/{total_days} days.")

    results = pd.DataFrame(records)
    tail_df = pd.DataFrame(tail_records)

    return results, tail_df
