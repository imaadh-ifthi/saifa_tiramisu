"""Robustness analysis with fixed OOS dates and data-derived conclusions."""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.backtest import run_backtest
from src.models import MultiScaleWaveletVine
from src.tail_dependence import aggregate_tail_dependence, aggregate_scale_difference
from src.backtests import fissler_ziegel_loss


def _copy_cfg(base_cfg):
    class CfgCopy:
        pass
    cfg = CfgCopy()
    for k in dir(base_cfg):
        if not k.startswith("__"):
            setattr(cfg, k, getattr(base_cfg, k))
    return cfg


def _direction(delta, ci_low, ci_high):
    if np.isfinite(ci_low) and ci_low > 0:
        return "D5>D1"
    if np.isfinite(ci_high) and ci_high < 0:
        return "D5<D1"
    if np.isclose(delta, 0.0, atol=1e-6):
        return "No clear difference"
    return "No clear difference"


def _tail_summary_for_training(train, cfg, n_boot=500):
    model = MultiScaleWaveletVine(
        levels=cfg.LEVELS,
        wavelet=cfg.WAVELET,
        seed=cfg.SEED,
        tail_quantile=cfg.TAIL_QUANTILE,
        use_cross_scale=True,
    )
    model.fit(train)
    d1 = aggregate_tail_dependence(model.empirical_U["D1"], 0.05, n_boot=n_boot, seed=cfg.SEED)
    d5 = aggregate_tail_dependence(model.empirical_U["D5"], 0.05, n_boot=n_boot, seed=cfg.SEED)
    diff = aggregate_scale_difference(
        model.empirical_U["D1"], model.empirical_U["D5"], 0.05, n_boot=n_boot, seed=cfg.SEED
    )
    return {
        "D1_tail_dependence": d1["estimate"],
        "D5_tail_dependence": d5["estimate"],
        "D1_ci_low": d1["ci_low"],
        "D1_ci_high": d1["ci_high"],
        "D5_ci_low": d5["ci_low"],
        "D5_ci_high": d5["ci_high"],
        "D5_minus_D1": diff["estimate"],
        "D5_minus_D1_ci_low": diff["ci_low"],
        "D5_minus_D1_ci_high": diff["ci_high"],
        "D5_minus_D1_p_value": diff["p_value"],
    }


def run_robustness(returns, base_cfg):
    print("\n" + "=" * 50)
    print("STARTING ROBUSTNESS ANALYSIS (FIX 7)")
    print("=" * 50)

    variations = [
        {"name": "Baseline", "param": "None", "changes": {}},
        {"name": "A_EVT_Threshold", "param": "TAIL_QUANTILE=0.90", "changes": {"TAIL_QUANTILE": 0.90}},
        {"name": "B_TrainWindow", "param": "TRAIN_WINDOW=750", "changes": {"TRAIN_WINDOW": 750}},
        {"name": "C_MonteCarlo", "param": "N_SIM=1000", "changes": {"N_SIM": 1000}},
        {"name": "D_RandomSeed", "param": "SEED=99", "changes": {"SEED": 99}},
    ]

    # All robustness cases use the SAME OOS dates. Only the training information
    # and/or stochastic configuration changes.
    returns = returns.dropna()
    common_start = int(base_cfg.TRAIN_WINDOW)
    common_end = min(len(returns), common_start + 252)
    if common_end <= common_start:
        raise RuntimeError("Not enough observations for the fixed robustness OOS window.")

    records = []
    baseline_direction = None

    for var in variations:
        print(f"\n[Robustness] {var['name']} ({var['param']})")
        cfg = _copy_cfg(base_cfg)
        for k, v in var["changes"].items():
            setattr(cfg, k, v)

        # Hold the OOS dates fixed across all robustness cases.
        cfg.OOS_START_INDEX = common_start
        cfg.OOS_END_INDEX = common_end
        cfg.TEST_DAYS = common_end - common_start

        t0 = time.perf_counter()
        res, _ = run_backtest(returns, cfg)
        runtime = time.perf_counter() - t0

        actual = res["actual_return"].to_numpy(dtype=float)
        var_forecast = res["M3_var"].to_numpy(dtype=float)
        es_forecast = res["M3_es"].to_numpy(dtype=float)
        breaches = actual < var_forecast
        fz = fissler_ziegel_loss(actual, var_forecast, es_forecast, cfg.ALPHA)

        # Tail-dependence robustness is evaluated at the first common forecast origin,
        # using only information available before that date.
        train_end = common_start
        train_start = train_end - int(cfg.TRAIN_WINDOW)
        if train_start < 0:
            raise RuntimeError("Robustness training window exceeds available pre-OOS history.")
        td = _tail_summary_for_training(returns.iloc[train_start:train_end], cfg)
        direction = _direction(td["D5_minus_D1"], td["D5_minus_D1_ci_low"], td["D5_minus_D1_ci_high"])
        if baseline_direction is None:
            baseline_direction = direction

        records.append({
            "robustness_case": var["name"],
            "parameter_change": var["param"],
            "OOS_start": returns.index[common_start],
            "OOS_end": returns.index[common_end - 1],
            "D1_tail_dependence": td["D1_tail_dependence"],
            "D5_tail_dependence": td["D5_tail_dependence"],
            "D1_ci_low": td["D1_ci_low"],
            "D1_ci_high": td["D1_ci_high"],
            "D5_ci_low": td["D5_ci_low"],
            "D5_ci_high": td["D5_ci_high"],
            "D5_minus_D1": td["D5_minus_D1"],
            "D5_minus_D1_ci_low": td["D5_minus_D1_ci_low"],
            "D5_minus_D1_ci_high": td["D5_minus_D1_ci_high"],
            "D5_minus_D1_p_value": td["D5_minus_D1_p_value"],
            "average_VaR": float(np.mean(var_forecast)),
            "average_ES": float(np.mean(es_forecast)),
            "breach_rate": float(np.mean(breaches)),
            "ES_score": float(np.mean(np.nan_to_num(fz, nan=1e6, posinf=1e6, neginf=1e6))),
            "observed_breaches": int(np.sum(breaches)),
            "conclusion_direction": direction,
            "conclusion_changed": bool(direction != baseline_direction),
            "runtime_seconds": runtime,
        })

    df = pd.DataFrame(records)
    out_dir = Path(base_cfg.RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "robustness_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(df["robustness_case"], df["average_VaR"])
    axes[0].set_title("Average M3 VaR")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(df["robustness_case"], df["average_ES"])
    axes[1].set_title("Average M3 ES")
    axes[1].tick_params(axis="x", rotation=45)

    axes[2].bar(df["robustness_case"], df["breach_rate"])
    axes[2].axhline(base_cfg.ALPHA, linestyle="--", label="Target")
    axes[2].set_title("M3 VaR Breach Rate")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "robustness_summary.png", dpi=200)
    plt.close(fig)

    print("[Robustness] Done. Results saved.")
    return df
