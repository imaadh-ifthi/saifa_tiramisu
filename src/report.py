"""
Reporting module.

Produces:
    - CSV outputs;
    - PNG figures;
    - summary.txt with statistical tests and recommendation.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR
from src.backtests import (
    kupiec_test,
    christoffersen_test,
    fissler_ziegel_loss,
    diebold_mariano_test,
)


def _scale_sort_key(scale_name: str):
    """
    Sort D1, D2, ..., D5, S5.
    """
    if scale_name.startswith("D"):
        try:
            return (0, int(scale_name[1:]))
        except Exception:
            return (0, 999)
    return (1, 0)


def save_report(results: pd.DataFrame, tail_df: pd.DataFrame, cfg):
    """
    Save all report artifacts.
    """
    RESULTS_DIR = Path(cfg.RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = results.copy()

    # Clean forecasts.
    for col in ["model_var", "model_es", "bench_var", "bench_es"]:
        results[col] = (
            results[col]
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0.0)
        )

    # --------------------------------------------------------------
    # Save raw outputs
    # --------------------------------------------------------------
    results.to_csv(RESULTS_DIR / "backtest_results.csv", index=False)

    if not tail_df.empty:
        tail_df.to_csv(RESULTS_DIR / "tail_dependence_by_scale.csv", index=False)

    # --------------------------------------------------------------
    # Statistical tests
    # --------------------------------------------------------------
    actual = results["actual_return"].values

    model_var = results["model_var"].values
    model_es = results["model_es"].values

    bench_var = results["bench_var"].values
    bench_es = results["bench_es"].values

    kupiec_model = kupiec_test(actual, model_var, cfg.ALPHA)
    kupiec_bench = kupiec_test(actual, bench_var, cfg.ALPHA)

    christ_model = christoffersen_test(actual, model_var, cfg.ALPHA)
    christ_bench = christoffersen_test(actual, bench_var, cfg.ALPHA)

    fz_model = fissler_ziegel_loss(actual, model_var, model_es, cfg.ALPHA)
    fz_bench = fissler_ziegel_loss(actual, bench_var, bench_es, cfg.ALPHA)

    fz_model = np.nan_to_num(fz_model, nan=1e6, posinf=1e6, neginf=-1e6)
    fz_bench = np.nan_to_num(fz_bench, nan=1e6, posinf=1e6, neginf=-1e6)

    dm = diebold_mariano_test(fz_model, fz_bench)

    # --------------------------------------------------------------
    # Tail dependence summary
    # --------------------------------------------------------------
    tail_ratio = np.nan
    lambda_first = np.nan
    lambda_last = np.nan
    scales = []

    if not tail_df.empty:
        lower_cols = [c for c in tail_df.columns if c.endswith("_L")]
        scales = sorted([c[:-2] for c in lower_cols], key=_scale_sort_key)

        if len(scales) >= 2:
            mean_tail = tail_df.mean(numeric_only=True)

            lambda_first = float(mean_tail.get(f"{scales[0]}_L", np.nan))
            lambda_last = float(mean_tail.get(f"{scales[-1]}_L", np.nan))

            if np.isfinite(lambda_first) and lambda_first > 1e-8:
                tail_ratio = lambda_last / lambda_first

    # --------------------------------------------------------------
    # Expected Shortfall comparison
    # --------------------------------------------------------------
    mean_model_es = float(np.mean(model_es))
    mean_bench_es = float(np.mean(bench_es))

    if mean_model_es < 0 and mean_bench_es < 0:
        es_underestimation = abs(mean_model_es) / abs(mean_bench_es) - 1.0
    else:
        es_underestimation = np.nan

    # --------------------------------------------------------------
    # Figure 1: tail dependence by horizon
    # --------------------------------------------------------------
    if not tail_df.empty and len(scales) > 0:
        mean_tail = tail_df.mean(numeric_only=True)

        lower_vals = []
        upper_vals = []

        for s in scales:
            lower_vals.append(float(mean_tail.get(f"{s}_L", np.nan)))
            upper_vals.append(float(mean_tail.get(f"{s}_U", np.nan)))

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(scales, lower_vals, marker="o", color="darkred", label="Lower tail dependence")
        ax.plot(scales, upper_vals, marker="s", color="steelblue", label="Upper tail dependence")

        ax.set_xlabel("Timescale / Horizon")
        ax.set_ylabel("Average pairwise tail dependence")
        ax.set_title("Tail Dependence Across Investment Horizons")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "tail_dependence_by_horizon.png", dpi=200)
        plt.close(fig)

    # --------------------------------------------------------------
    # Figure 2: VaR forecasts vs actual returns
    # --------------------------------------------------------------
    plot_df = results.tail(252).copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        plot_df["date"],
        plot_df["actual_return"],
        color="black",
        alpha=0.65,
        label="Portfolio return",
    )

    ax.plot(
        plot_df["date"],
        plot_df["model_var"],
        color="darkred",
        label="Wavelet-vine 99% VaR",
    )

    ax.plot(
        plot_df["date"],
        plot_df["bench_var"],
        color="steelblue",
        label="Gaussian benchmark 99% VaR",
        linestyle="--",
    )

    ax.set_title("Out-of-Sample 99% VaR Forecasts")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "var_forecasts.png", dpi=200)
    plt.close(fig)

    # --------------------------------------------------------------
    # Figure 3: cumulative Fissler-Ziegel loss
    # --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        results["date"],
        np.cumsum(fz_model),
        label="Wavelet-vine model",
        color="darkred",
    )

    ax.plot(
        results["date"],
        np.cumsum(fz_bench),
        label="Gaussian benchmark",
        color="steelblue",
        linestyle="--",
    )

    ax.set_title("Cumulative Fissler-Ziegel Joint VaR/ES Loss")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fissler_ziegel_cumulative_loss.png", dpi=200)
    plt.close(fig)

    # --------------------------------------------------------------
    # Summary text
    # --------------------------------------------------------------
    summary_lines = []

    summary_lines.append("=" * 70)
    summary_lines.append("QUANT EDGE 1.0: WAVELET-COPULA RISK FRAMEWORK SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append("")

    summary_lines.append("Backtest settings:")
    summary_lines.append(f"Training window: {cfg.TRAIN_WINDOW}")
    summary_lines.append(f"Test days: {len(results)}")
    summary_lines.append(f"Monte Carlo simulations per forecast: {cfg.N_SIM}")
    summary_lines.append(f"Confidence level: {100 * (1 - cfg.ALPHA):.0f}%")
    summary_lines.append(f"Wavelet levels: {cfg.LEVELS}")
    summary_lines.append("")

    summary_lines.append("VaR coverage tests:")
    summary_lines.append(
        f"Model Kupiec LR: {kupiec_model['stat']:.4f}, "
        f"p-value: {kupiec_model['p_value']:.4f}, "
        f"violation rate: {kupiec_model['violation_rate']:.4f}"
    )
    summary_lines.append(
        f"Benchmark Kupiec LR: {kupiec_bench['stat']:.4f}, "
        f"p-value: {kupiec_bench['p_value']:.4f}, "
        f"violation rate: {kupiec_bench['violation_rate']:.4f}"
    )
    summary_lines.append("")

    summary_lines.append("VaR independence tests:")
    summary_lines.append(
        f"Model Christoffersen LR: {christ_model['stat']:.4f}, "
        f"p-value: {christ_model['p_value']:.4f}"
    )
    summary_lines.append(
        f"Benchmark Christoffersen LR: {christ_bench['stat']:.4f}, "
        f"p-value: {christ_bench['p_value']:.4f}"
    )
    summary_lines.append("")

    summary_lines.append("Expected Shortfall joint loss:")
    summary_lines.append(f"Mean FZ loss model: {np.mean(fz_model):.6f}")
    summary_lines.append(f"Mean FZ loss benchmark: {np.mean(fz_bench):.6f}")
    summary_lines.append(
        f"Diebold-Mariano stat: {dm['stat']:.4f}, p-value: {dm['p_value']:.4f}"
    )
    summary_lines.append("")

    summary_lines.append("Tail dependence by horizon:")
    if not tail_df.empty and len(scales) > 0:
        mean_tail = tail_df.mean(numeric_only=True)
        for s in scales:
            l_val = mean_tail.get(f"{s}_L", np.nan)
            u_val = mean_tail.get(f"{s}_U", np.nan)
            summary_lines.append(
                f"{s}: lower tail = {l_val:.4f}, upper tail = {u_val:.4f}"
            )
    else:
        summary_lines.append("Tail dependence table unavailable.")
    summary_lines.append("")

    summary_lines.append("Economic interpretation:")
    summary_lines.append(
        f"Average model 99% ES: {mean_model_es:.4f}"
    )
    summary_lines.append(
        f"Average benchmark 99% ES: {mean_bench_es:.4f}"
    )

    if np.isfinite(es_underestimation):
        summary_lines.append(
            f"Benchmark ES underestimation relative to model: {100 * es_underestimation:.2f}%"
        )
    else:
        summary_lines.append("Benchmark ES underestimation: unavailable.")

    if np.isfinite(tail_ratio):
        summary_lines.append(
            f"Lower tail dependence ratio longest/shortest scale: {tail_ratio:.2f}x"
        )
    else:
        summary_lines.append("Lower tail dependence ratio: unavailable.")

    summary_lines.append("")
    summary_lines.append("=" * 70)
    summary_lines.append("CONCRETE RECOMMENDATION FOR RISK MANAGERS")
    summary_lines.append("=" * 70)
    summary_lines.append("")

    recommendation = f"""
Do not use one dependence structure for all risk horizons.

The framework estimates tail dependence separately at short, medium, and long
timescales. If lower tail dependence is higher at longer scales, then daily
or aggregate risk models can overstate diversification exactly when systemic
stress persists over several weeks.

Actionable rule:

1. Keep short-term liquidity and intraday limits based on high-frequency risk
   metrics, but do not use them to size systemic tail hedges.

2. Size strategic tail hedges, drawdown controls, and capital buffers using
   the low-frequency Expected Shortfall estimate.

3. If the low-frequency ES is {abs(mean_model_es):.4f} while the aggregate
   Gaussian benchmark ES is only {abs(mean_bench_es):.4f}, then relying on
   the aggregate benchmark may understate tail risk by approximately
   {100 * max(es_underestimation, 0.0):.1f}%.

4. Review hedge ratios whenever the long-scale lower tail dependence rises
   materially above the short-scale estimate.
"""

    summary_lines.append(recommendation)

    summary_text = "\n".join(summary_lines)

    with open(RESULTS_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)

    return summary_text
