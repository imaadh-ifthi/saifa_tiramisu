"""
Reporting module for Quant Edge 1.0 Ablation Framework.
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
    RESULTS_DIR_PATH = Path(cfg.RESULTS_DIR)
    RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)

    results = results.copy()
    
    models = ["M0", "M1", "M2", "M3"]
    model_labels = {
        "M0": "Historical Simulation Benchmark",
        "M1": "Heavy-Tail Marginals",
        "M2": "Wavelet + Within-Scale",
        "M3": "Full Cross-Scale Multiscale"
    }

    # Clean forecasts.
    for m in models:
        for col in [f"{m}_var", f"{m}_es"]:
            results[col] = (
                results[col]
                .replace([np.inf, -np.inf], np.nan)
                .ffill()
                .bfill()
                .fillna(0.0)
            )

    actual = results["actual_return"].values

    # 1. Output model_comparison_by_window.csv
    window_records = []
    for idx, row in results.iterrows():
        for m in models:
            var_val = row[f"{m}_var"]
            es_val = row[f"{m}_es"]
            ret = row["actual_return"]
            breach = 1 if ret < var_val else 0
            
            window_records.append({
                "forecast_date": row["date"],
                "training_window_end": row["train_end"],
                "model": m,
                "model_name": model_labels[m],
                "VaR": var_val,
                "ES": es_val,
                "realized_return": ret,
                "is_breach": breach
            })
    
    pd.DataFrame(window_records).to_csv(RESULTS_DIR_PATH / "model_comparison_by_window.csv", index=False)

    # 2. Output model_comparison.csv
    summary_records = []
    
    expected_breaches = len(actual) * cfg.ALPHA
    
    for m in models:
        m_var = results[f"{m}_var"].values
        m_es = results[f"{m}_es"].values
        
        breaches = actual < m_var
        obs_breaches = np.sum(breaches)
        breach_rate = obs_breaches / len(actual)
        
        avg_var = np.mean(m_var)
        avg_es = np.mean(m_es)
        avg_realized_loss = np.mean(actual[breaches]) if obs_breaches > 0 else np.nan
        
        kt = kupiec_test(actual, m_var, cfg.ALPHA)
        ct = christoffersen_test(actual, m_var, cfg.ALPHA)
        fz = fissler_ziegel_loss(actual, m_var, m_es, cfg.ALPHA)
        avg_fz = np.mean(np.nan_to_num(fz, nan=1e6, posinf=1e6, neginf=-1e6))
        
        summary_records.append({
            "model": m,
            "model_name": model_labels[m],
            "vaR_level": 1 - cfg.ALPHA,
            "expected_breaches": expected_breaches,
            "observed_breaches": obs_breaches,
            "breach_rate": breach_rate,
            "average_var": avg_var,
            "average_es": avg_es,
            "average_realized_breach_loss": avg_realized_loss,
            "coverage_test_statistic": kt["stat"],
            "coverage_test_pvalue": kt["p_value"],
            "independence_test_statistic": ct["stat"],
            "independence_test_pvalue": ct["p_value"],
            "es_score": avg_fz
        })
        
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(RESULTS_DIR_PATH / "model_comparison.csv", index=False)
    
    # 3. Figures
    # FIGURE 1: OOS VaR time series for all models
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results["date"], actual, color="black", alpha=0.5, label="Realized Portfolio Return")
    colors = {"M0": "blue", "M1": "green", "M2": "orange", "M3": "red"}
    styles = {"M0": "--", "M1": "-.", "M2": ":", "M3": "-"}
    
    for m in models:
        ax.plot(results["date"], results[f"{m}_var"], color=colors[m], linestyle=styles[m], label=model_labels[m])
        
    ax.set_title(f"Out-of-Sample {100*(1-cfg.ALPHA):.0f}% VaR Forecasts")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR_PATH / "oos_var_timeseries.png", dpi=200)
    plt.close(fig)
    
    # FIGURE 2: Average / distribution of OOS VaR by model
    fig, ax = plt.subplots(figsize=(8, 5))
    var_data = [results[f"{m}_var"].values for m in models]
    ax.boxplot(var_data, tick_labels=[model_labels[m] for m in models])
    ax.set_title("Distribution of OOS VaR Estimates")
    ax.set_ylabel("VaR")
    plt.xticks(rotation=15, ha='right')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR_PATH / "var_distribution.png", dpi=200)
    plt.close(fig)

    # FIGURE 3: Average OOS ES by model
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([model_labels[m] for m in models], df_summary["average_es"].values, color=["blue", "green", "orange", "red"])
    ax.set_title("Average Expected Shortfall")
    ax.set_ylabel("Expected Shortfall")
    plt.xticks(rotation=15, ha='right')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR_PATH / "avg_es.png", dpi=200)
    plt.close(fig)

    # FIGURE 4: Observed vs nominal VaR breach rate
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([model_labels[m] for m in models], df_summary["breach_rate"].values, color=["blue", "green", "orange", "red"])
    ax.axhline(cfg.ALPHA, color="black", linestyle="--", label="Nominal Target")
    ax.set_title("Empirical VaR Breach Rate")
    ax.set_ylabel("Breach Rate")
    ax.legend()
    plt.xticks(rotation=15, ha='right')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR_PATH / "breach_rate.png", dpi=200)
    plt.close(fig)
    
    # FIGURE 5: Ablation plot (Benchmark -> Heavy-tail -> Wavelet -> Full model)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([model_labels[m] for m in models], df_summary["es_score"].values, marker='o', color='purple')
    ax.set_title("Fissler-Ziegel Joint Loss Progression (Lower is better)")
    ax.set_ylabel("Average ES Score")
    plt.xticks(rotation=15, ha='right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR_PATH / "ablation_es_score.png", dpi=200)
    plt.close(fig)

    # 4. Write summary.txt
    s = []
    s.append("=" * 70)
    s.append("QUANT EDGE 1.0: ABLATION FRAMEWORK SUMMARY")
    s.append("=" * 70)
    s.append(f"Training Window: {cfg.TRAIN_WINDOW}")
    s.append(f"OOS Days: {len(results)}")
    
    s.append("\n--- ABLATION DELTAS ---")
    m0, m1, m2, m3 = df_summary.iloc[0], df_summary.iloc[1], df_summary.iloc[2], df_summary.iloc[3]
    
    def format_delta(m_post, m_pre, name):
        d_var = m_post['average_var'] - m_pre['average_var']
        d_es = m_post['average_es'] - m_pre['average_es']
        d_breach = m_post['breach_rate'] - m_pre['breach_rate']
        return f"{name}:\n  Avg VaR Diff: {d_var:.4f}\n  Avg ES Diff: {d_es:.4f}\n  Breach Rate Diff: {d_breach:.4f}\n"

    s.append(format_delta(m1, m0, "M1 - M0 (Heavy-tail vs Benchmark)"))
    s.append(format_delta(m2, m1, "M2 - M1 (Wavelets vs Heavy-tail)"))
    s.append(format_delta(m3, m2, "M3 - M2 (Cross-Scale vs Within-Scale)"))
    s.append(format_delta(m3, m0, "M3 - M0 (Full Model vs Benchmark)"))
    
    s.append("\n--- THE MOST IMPORTANT COMPARISON (IGNORE HORIZON DEPENDENCE VS FULL MODEL) ---")
    s.append("Model M1 (ignores multiscale/cross-scale) vs Model M3 (current FULL model).")
    s.append(f"VaR (M1): {m1['average_var']:.4f}  | VaR (M3): {m3['average_var']:.4f}")
    s.append(f"ES (M1): {m1['average_es']:.4f}  | ES (M3): {m3['average_es']:.4f}")
    s.append(f"Breaches (M1): {m1['observed_breaches']}  | Breaches (M3): {m3['observed_breaches']}")
    
    s.append("\n" + "=" * 70)
    s.append("CONCRETE RECOMMENDATION FOR RISK MANAGERS")
    s.append("=" * 70 + "\n")
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

3. If the low-frequency ES differs from the aggregate benchmark, rely on
   the structure that successfully incorporates cross-horizon coupling to
   prevent systemic risk underestimation.
"""
    s.append(recommendation)
    
    summary_text = "\n".join(s)
    with open(RESULTS_DIR_PATH / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    return summary_text
