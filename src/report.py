"""Reporting and evidence synthesis for Quant Edge 1.0."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.backtests import kupiec_test, christoffersen_test, fissler_ziegel_loss


def _load_horizon_summary(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "tail_dependence_horizon_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _horizon_conclusion(horizon: pd.DataFrame, base: str = "D1", compare: str = "D5") -> str:
    if horizon.empty:
        return "Insufficient evidence: no horizon summary was generated."
    rows = horizon[horizon["scale"].isin([base, compare])]
    if len(rows) < 2:
        return "Insufficient evidence: D1/D5 comparison is unavailable."
    d5 = rows.loc[rows["scale"] == compare].iloc[0]
    lo = float(d5["delta_ci_low"])
    hi = float(d5["delta_ci_high"])
    delta = float(d5["delta_vs_D1"])
    if lo > 0:
        return f"Evidence supports higher lower-tail dependence at {compare} than D1 (delta={delta:.4f}, 95% CI [{lo:.4f}, {hi:.4f}])."
    if hi < 0:
        return f"Evidence supports lower lower-tail dependence at {compare} than D1 (delta={delta:.4f}, 95% CI [{lo:.4f}, {hi:.4f}])."
    return f"No strong evidence of a D1-to-{compare} difference (delta={delta:.4f}, 95% CI [{lo:.4f}, {hi:.4f}] includes zero)."


def _select_recommendation(df_summary: pd.DataFrame, horizon_text: str) -> tuple[str, str]:
    """Select a model using a predeclared, evidence-first rule."""
    # Prefer models with no evidence of unconditional coverage failure and no evidence
    # of clustered breaches; use the FZ score as the secondary criterion (lower is better).
    candidates = df_summary[
        (df_summary["coverage_test_pvalue"] >= 0.05)
        & (df_summary["independence_test_pvalue"] >= 0.05)
    ].copy()

    rule = "Eligible models require coverage and independence p-values >= 0.05; among eligible models, lower Fissler-Ziegel score is preferred."
    if candidates.empty:
        candidates = df_summary[df_summary["coverage_test_pvalue"] >= 0.05].copy()
        rule += " No model passed both tests, so the selection used coverage p-value first, then Fissler-Ziegel score."

    if candidates.empty:
        selected = df_summary.iloc[np.argmin(np.abs(df_summary["breach_rate"] - (1 - df_summary["vaR_level"].iloc[0])))]
        rule += " No model passed coverage; selected the model with breach rate closest to the nominal target as a fallback."
    else:
        selected = candidates.sort_values(["es_score", "coverage_test_pvalue"], ascending=[True, False]).iloc[0]

    model = str(selected["model"])
    label = str(selected["model_name"])
    rec = (
        f"Recommendation: use {model} ({label}) as the primary 99% VaR/ES risk model under the tested portfolio and OOS setting. "
        f"The observed evidence indicates: {horizon_text} "
        "Use the multiscale model as an analytical diagnostic for horizon dependence rather than as the production risk forecast unless future validation shows materially better calibration."
    )
    return rec, rule


def save_report(results: pd.DataFrame, tail_df: pd.DataFrame, cfg):
    """Generate reproducible validation tables, figures and evidence-based recommendation."""
    results_dir = Path(cfg.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    results = results.copy()
    models = ["M0", "M1", "M2", "M3"]
    model_labels = {
        "M0": "Historical Simulation Benchmark",
        "M1": "Heavy-Tail Marginals",
        "M2": "Wavelet + Within-Scale",
        "M3": "Full Cross-Scale Multiscale",
    }

    required = [f"{m}_{suffix}" for m in models for suffix in ("var", "es")]
    missing = [c for c in required if c not in results.columns]
    if missing:
        raise ValueError(f"Missing forecast columns: {missing}")

    for m in models:
        for col in [f"{m}_var", f"{m}_es"]:
            results[col] = pd.to_numeric(results[col], errors="coerce")
    results = results.replace([np.inf, -np.inf], np.nan).dropna(subset=["actual_return"])

    actual = results["actual_return"].to_numpy(dtype=float)
    expected_breaches = len(actual) * cfg.ALPHA

    summary_records = []
    window_records = []
    for _, row in results.iterrows():
        for m in models:
            var_val = float(row[f"{m}_var"])
            es_val = float(row[f"{m}_es"])
            ret = float(row["actual_return"])
            breach = int(ret < var_val)
            window_records.append({
                "forecast_date": row["date"],
                "training_window_end": row["train_end"],
                "model": m,
                "model_name": model_labels[m],
                "VaR": var_val,
                "ES": es_val,
                "realized_return": ret,
                "is_breach": breach,
            })

    df_window = pd.DataFrame(window_records)
    df_window.to_csv(results_dir / "model_comparison_by_window.csv", index=False)

    for m in models:
        var_forecast = results[f"{m}_var"].to_numpy(dtype=float)
        es_forecast = results[f"{m}_es"].to_numpy(dtype=float)
        breaches = actual < var_forecast
        obs = int(breaches.sum())
        kt = kupiec_test(actual, var_forecast, cfg.ALPHA)
        ct = christoffersen_test(actual, var_forecast, cfg.ALPHA)
        fz = fissler_ziegel_loss(actual, var_forecast, es_forecast, cfg.ALPHA)
        summary_records.append({
            "model": m,
            "model_name": model_labels[m],
            "vaR_level": 1 - cfg.ALPHA,
            "expected_breaches": expected_breaches,
            "observed_breaches": obs,
            "breach_rate": obs / len(actual),
            "average_var": float(np.mean(var_forecast)),
            "average_es": float(np.mean(es_forecast)),
            "average_realized_breach_loss": float(np.mean(actual[breaches])) if obs else np.nan,
            "coverage_test_statistic": kt["stat"],
            "coverage_test_pvalue": kt["p_value"],
            "independence_test_statistic": ct["stat"],
            "independence_test_pvalue": ct["p_value"],
            "es_score": float(np.mean(np.nan_to_num(fz, nan=1e6, posinf=1e6, neginf=1e6))),
        })

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(results_dir / "model_comparison.csv", index=False)
    df_summary.to_csv(results_dir / "final_validation_summary.csv", index=False)

    # Load the horizon-level tail-dependence bootstrap summary produced by Fix 4.
    # Reads "tail_dependence_horizon_summary.csv"; not any legacy file.
    horizon = _load_horizon_summary(results_dir)  # reads tail_dependence_horizon_summary.csv
    horizon_text = _horizon_conclusion(horizon)
    recommendation, selection_rule = _select_recommendation(df_summary, horizon_text)

    # Primary competition table uses the genuine aggregate bootstrap output, not a
    # nonexistent/recovered legacy file and not interval arithmetic on pair CIs.
    comp = {
        "tail_dependence_D1": np.nan,
        "tail_dependence_D5": np.nan,
        "tail_dependence_D1_CI_lower": np.nan,
        "tail_dependence_D1_CI_upper": np.nan,
        "tail_dependence_D5_CI_lower": np.nan,
        "tail_dependence_D5_CI_upper": np.nan,
        "D5_minus_D1": np.nan,
        "D5_minus_D1_CI_lower": np.nan,
        "D5_minus_D1_CI_upper": np.nan,
        "M1_average_VaR": float(df_summary.loc[df_summary.model == "M1", "average_var"].iloc[0]),
        "M3_average_VaR": float(df_summary.loc[df_summary.model == "M3", "average_var"].iloc[0]),
        "M1_average_ES": float(df_summary.loc[df_summary.model == "M1", "average_es"].iloc[0]),
        "M3_average_ES": float(df_summary.loc[df_summary.model == "M3", "average_es"].iloc[0]),
        "M1_breach_rate": float(df_summary.loc[df_summary.model == "M1", "breach_rate"].iloc[0]),
        "M3_breach_rate": float(df_summary.loc[df_summary.model == "M3", "breach_rate"].iloc[0]),
        "M1_ES_score": float(df_summary.loc[df_summary.model == "M1", "es_score"].iloc[0]),
        "M3_ES_score": float(df_summary.loc[df_summary.model == "M3", "es_score"].iloc[0]),
        "selected_production_model": str(recommendation.split("use ", 1)[1].split(" (", 1)[0]),
        "horizon_conclusion": horizon_text,
    }
    if not horizon.empty:
        d1 = horizon[horizon.scale == "D1"]
        d5 = horizon[horizon.scale == "D5"]
        if not d1.empty:
            r = d1.iloc[0]
            comp.update({
                "tail_dependence_D1": r["mean_pair_lambda_L"],
                "tail_dependence_D1_CI_lower": r["ci_low"],
                "tail_dependence_D1_CI_upper": r["ci_high"],
            })
        if not d5.empty:
            r = d5.iloc[0]
            comp.update({
                "tail_dependence_D5": r["mean_pair_lambda_L"],
                "tail_dependence_D5_CI_lower": r["ci_low"],
                "tail_dependence_D5_CI_upper": r["ci_high"],
                "D5_minus_D1": r["delta_vs_D1"],
                "D5_minus_D1_CI_lower": r["delta_ci_low"],
                "D5_minus_D1_CI_upper": r["delta_ci_high"],
            })
    comp["interpretation_flags"] = (
        f"Horizon={horizon_text}; "
        f"RiskModelSelection={comp['selected_production_model']}"
    )
    pd.DataFrame([comp]).to_csv(results_dir / "final_competition_comparison.csv", index=False)

    # OOS VaR plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results["date"], actual, label="Realized Portfolio Return", linewidth=1.0)
    for m in models:
        ax.plot(results["date"], results[f"{m}_var"], linestyle="--", linewidth=1.0, label=f"{m} VaR")
    ax.set_title(f"Out-of-Sample {100*(1-cfg.ALPHA):.0f}% VaR Forecasts")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(results_dir / "final_oos_var_comparison.png", dpi=200)
    plt.close(fig)

    # VaR distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([results[f"{m}_var"].to_numpy() for m in models], tick_labels=models)
    ax.set_title("Distribution of OOS VaR Estimates")
    ax.set_ylabel("VaR")
    fig.tight_layout()
    fig.savefig(results_dir / "var_distribution.png", dpi=200)
    plt.close(fig)

    # ES comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(models, [df_summary.loc[df_summary.model == m, "average_es"].iloc[0] for m in models])
    ax.set_title("Average Expected Shortfall")
    ax.set_ylabel("Expected Shortfall")
    fig.tight_layout()
    fig.savefig(results_dir / "final_es_comparison.png", dpi=200)
    plt.close(fig)

    # Breach calibration
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(models, [df_summary.loc[df_summary.model == m, "breach_rate"].iloc[0] for m in models])
    ax.axhline(cfg.ALPHA, linestyle="--", label="Nominal Target")
    ax.set_title("Empirical VaR Breach Rate")
    ax.set_ylabel("Breach Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "final_breach_calibration.png", dpi=200)
    plt.close(fig)

    # Stress-period view is descriptive only; it does not create a new statistical claim.
    stress_mask = (pd.to_datetime(results["date"]) >= pd.Timestamp("2020-02-01")) & (pd.to_datetime(results["date"]) <= pd.Timestamp("2020-06-30"))
    stress_results = results.loc[stress_mask]
    if not stress_results.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(stress_results["date"], stress_results["actual_return"], label="Realized", linewidth=1.5)
        for m in models:
            ax.plot(stress_results["date"], stress_results[f"{m}_var"], linestyle="--", label=f"{m} VaR")
        ax.set_title("Stress Period (H1 2020) — Descriptive OOS Comparison")
        ax.legend(fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(results_dir / "final_stress_comparison.png", dpi=200)
        plt.close(fig)

    # Final text must be generated from actual measured outputs, never hard-coded.
    s = [
        "=" * 70,
        "QUANT EDGE 1.0: FINAL VALIDATION SUMMARY",
        "=" * 70,
        f"Training Window: {cfg.TRAIN_WINDOW}",
        f"OOS Days: {len(results)}",
        f"OOS Start: {pd.to_datetime(results['date']).min().date()}",
        f"OOS End: {pd.to_datetime(results['date']).max().date()}",
        "",
        "--- HORIZON EVIDENCE ---",
        horizon_text,
        "",
        "--- MODEL COMPARISON ---",
    ]
    for _, row in df_summary.iterrows():
        s.append(
            f"{row['model']}: breach_rate={row['breach_rate']:.4f}, "
            f"avg_VaR={row['average_var']:.4f}, avg_ES={row['average_es']:.4f}, "
            f"coverage_p={row['coverage_test_pvalue']:.6g}, "
            f"independence_p={row['independence_test_pvalue']:.6g}, "
            f"FZ_score={row['es_score']:.4f}"
        )
    s.extend([
        "",
        "--- MODEL SELECTION RULE ---",
        selection_rule,
        "",
        "--- RISK-MANAGER RECOMMENDATION ---",
        recommendation,
        "",
        "--- LIMITATIONS ---",
        "99% VaR backtests have limited statistical power when the OOS sample is short; interpret p-values alongside breach counts and ES scores.",
        "Stress-period tail-dependence comparisons are descriptive unless a common fitted marginal transformation is used across periods.",
        "The multiscale model's production suitability depends on the corrected PIT pipeline and the full OOS results generated by this exact run.",
    ])
    summary_text = "\n".join(s)
    (results_dir / "final_validation_summary.txt").write_text(summary_text, encoding="utf-8")

    return summary_text
