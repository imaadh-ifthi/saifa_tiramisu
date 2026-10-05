"""
Out-of-sample backtest engine.
"""

import numpy as np
import pandas as pd

from src.models import (
    HistoricalSimulationBenchmark,
    HeavyTailMarginalModel,
    MultiScaleWaveletVine,
)
from src.risk import var_es


def run_backtest(returns: pd.DataFrame, cfg):
    """
    Run rolling out-of-sample backtest.
    """
    returns = returns.dropna()
    n = len(returns)

    if n <= cfg.TRAIN_WINDOW + 10:
        raise RuntimeError("Not enough data for the selected training window.")

    # Allow robustness experiments to hold the OOS dates fixed while changing
    # the amount of historical data used for fitting.  In normal runs these
    # attributes are absent and the original TRAIN_WINDOW start is used.
    start = int(getattr(cfg, "OOS_START_INDEX", cfg.TRAIN_WINDOW))
    if start < int(cfg.TRAIN_WINDOW):
        raise ValueError("OOS_START_INDEX must be >= TRAIN_WINDOW to avoid look-ahead.")

    if cfg.TEST_DAYS is None:
        end = n
    else:
        end = min(n, start + int(cfg.TEST_DAYS))

    configured_end = getattr(cfg, "OOS_END_INDEX", None)
    if configured_end is not None:
        end = min(end, int(configured_end))

    if end <= start:
        raise RuntimeError("Empty or invalid OOS interval.")

    records = []
    tail_records = []

    models_state = {
        "M0": None,
        "M1": None,
        "M2": None,
        "M3": None,
    }

    weights = np.asarray(cfg.WEIGHTS, dtype=float)
    weights = weights / weights.sum()

    total_days = end - start

    print(f"[Backtest] Starting backtest with {total_days} out-of-sample days.")

    for t in range(start, end):
        date = returns.index[t]

        train = returns.iloc[t - cfg.TRAIN_WINDOW : t]
        actual_return = float(returns.iloc[t].values @ weights)

        need_refit = (models_state["M3"] is None) or ((t - start) % cfg.REFIT_EVERY == 0)

        if need_refit:
            date_str = date.date() if hasattr(date, "date") else date
            print(f"[Backtest] Fitting models on window ending {date_str}")
            
            # M0: Historical Simulation
            m0 = HistoricalSimulationBenchmark()
            m0.fit(train)
            
            # M1: Heavy-tail Marginal Model
            m1 = HeavyTailMarginalModel(tail_quantile=cfg.TAIL_QUANTILE, seed=cfg.SEED)
            m1.fit(train)
            
            # M2: Wavelet + Within-Scale Dependence (No Cross-Scale)
            m2 = MultiScaleWaveletVine(
                levels=cfg.LEVELS,
                wavelet=cfg.WAVELET,
                seed=cfg.SEED,
                tail_quantile=cfg.TAIL_QUANTILE,
                use_cross_scale=False,
            )
            m2.fit(train)

            # M3: Full Multiscale Cross-Scale Model
            m3 = MultiScaleWaveletVine(
                levels=cfg.LEVELS,
                wavelet=cfg.WAVELET,
                seed=cfg.SEED,
                tail_quantile=cfg.TAIL_QUANTILE,
                use_cross_scale=True,
            )
            m3.fit(train)

            models_state = {"M0": m0, "M1": m1, "M2": m2, "M3": m3}

            tail_row = {"date": date}
            for scale_name, (lambda_l, lambda_u) in m3.tail_deps.items():
                tail_row[f"{scale_name}_L"] = lambda_l
                tail_row[f"{scale_name}_U"] = lambda_u

            tail_records.append(tail_row)

        # Forecast distribution for date t
        sims = {}
        for m_name, model_obj in models_state.items():
            s = model_obj.simulate_portfolio(
                n_sim=cfg.N_SIM,
                weights=weights,
                seed=cfg.SEED + t,
            )
            sims[m_name] = s
            
        record = {
            "date": date,
            "actual_return": actual_return,
            "train_end": returns.index[t-1],
        }
        
        for m_name, s in sims.items():
            var_val, es_val = var_es(s, alpha=cfg.ALPHA)
            record[f"{m_name}_var"] = var_val
            record[f"{m_name}_es"] = es_val
            
        records.append(record)

        completed = t - start + 1
        if completed % 10 == 0:
            print(f"[Backtest] Completed {completed}/{total_days} days.")

    results = pd.DataFrame(records)
    tail_df = pd.DataFrame(tail_records)

    return results, tail_df
