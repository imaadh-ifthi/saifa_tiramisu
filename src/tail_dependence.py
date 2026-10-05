import os
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import MultiScaleWaveletVine
from src.modwt import AdditiveMODWT
from config import RESULTS_DIR

def empirical_tail_dependence(u1, u2, q):
    """
    Empirical tail dependence for one pair of variables.
    Returns: lambda_L, lambda_U, sample_size, lower_events, upper_events
    """
    n = len(u1)
    lower_events = np.sum((u1 < q) & (u2 < q))
    upper_events = np.sum((u1 > 1.0 - q) & (u2 > 1.0 - q))
    return float(lower_events) / (n * q), float(upper_events) / (n * q), n, lower_events, upper_events

def bootstrap_tail_dependence(u1, u2, q, n_boot=2000, seed=42):
    """
    Nonparametric bootstrap of tail dependence.
    """
    rng = np.random.default_rng(seed)
    n = len(u1)
    boot_L = np.zeros(n_boot)
    boot_U = np.zeros(n_boot)
    
    # We can vectorize bootstrap for speed
    indices = rng.integers(0, n, size=(n_boot, n))
    
    for b in range(n_boot):
        idx = indices[b]
        l, u, _, _, _ = empirical_tail_dependence(u1[idx], u2[idx], q)
        boot_L[b] = l
        boot_U[b] = u
        
    return boot_L, boot_U



def aggregate_tail_dependence(U: np.ndarray, q: float, n_boot: int = 2000, seed: int = 42):
    """
    Mean pairwise empirical tail dependence with an observation-level bootstrap CI.

    The same bootstrap time indices are applied to every asset pair so the CI is for
    the mean of the ten pairwise coefficients, rather than an arithmetic average of
    pair-level confidence-interval endpoints.
    """
    U = np.asarray(U, dtype=float)
    n, n_assets = U.shape
    if n < 2:
        return {
            "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
            "bootstrap": np.array([], dtype=float),
        }

    pairs = list(combinations(range(n_assets), 2))

    def mean_stat(indices=None):
        X = U if indices is None else U[indices]
        vals = [
            empirical_tail_dependence(X[:, i], X[:, j], q)[0]
            for i, j in pairs
        ]
        return float(np.mean(vals))

    estimate = mean_stat()
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        bootstrap[b] = mean_stat(idx)

    ci = np.percentile(bootstrap, [2.5, 97.5])
    return {
        "estimate": estimate,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "bootstrap": bootstrap,
    }


def aggregate_scale_difference(U_base: np.ndarray, U_compare: np.ndarray, q: float, n_boot: int = 2000, seed: int = 42):
    """
    Paired observation-level bootstrap for the difference in mean pairwise tail dependence.
    """
    U_base = np.asarray(U_base, dtype=float)
    U_compare = np.asarray(U_compare, dtype=float)
    if U_base.shape != U_compare.shape:
        raise ValueError("Base and comparison PIT matrices must have identical shape and time alignment.")

    n, n_assets = U_base.shape
    pairs = list(combinations(range(n_assets), 2))

    def stat(indices):
        Xb = U_base[indices]
        Xc = U_compare[indices]
        vals_b = [empirical_tail_dependence(Xb[:, i], Xb[:, j], q)[0] for i, j in pairs]
        vals_c = [empirical_tail_dependence(Xc[:, i], Xc[:, j], q)[0] for i, j in pairs]
        return float(np.mean(vals_c) - np.mean(vals_b))

    point = stat(np.arange(n))
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        boot[b] = stat(rng.integers(0, n, size=n))

    ci = np.percentile(boot, [2.5, 97.5])
    return {
        "estimate": point,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "bootstrap": boot,
        "p_value": float(min(1.0, 2.0 * min(np.mean(boot <= 0.0), np.mean(boot >= 0.0)))),
    }




def benjamini_hochberg(p_values, alpha=0.05):
    """Return BH-FDR rejection flags and adjusted p-values without extra dependencies."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    order = np.argsort(p)
    sorted_p = p[order]
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for k in range(m - 1, -1, -1):
        rank = k + 1
        running = min(running, sorted_p[k] * m / rank)
        adjusted[k] = running

    p_adj = np.empty(m, dtype=float)
    p_adj[order] = np.clip(adjusted, 0.0, 1.0)
    critical = alpha * np.arange(1, m + 1) / m
    reject_sorted = sorted_p <= critical
    if np.any(reject_sorted):
        max_k = np.max(np.where(reject_sorted)[0])
        reject_sorted = np.arange(m) <= max_k
    reject = np.empty(m, dtype=bool)
    reject[order] = reject_sorted
    return reject, p_adj


def run_validation(returns, cfg):
    """
    Run full statistical tail dependence validation.
    """
    print("[Validation] Starting rigorous tail-dependence validation...")
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    asset_names = list(returns.columns)
    
    # Primary analysis uses a fixed broad historical sample.
    broad_train = returns.loc["2015-01-01":"2019-12-31"]
    if len(broad_train) < 100:
        print(f"[Validation] Warning: broad sample is too short ({len(broad_train)} obs). Skipping.")
        return

    print("[Validation] Fitting MultiScaleWaveletVine for broad historical sample...")
    broad_m = MultiScaleWaveletVine(
        levels=cfg.LEVELS,
        wavelet=cfg.WAVELET,
        seed=cfg.SEED,
        tail_quantile=cfg.TAIL_QUANTILE,
    )
    broad_m.fit(broad_train)
    
    thresholds = [0.025, 0.05, 0.10]
    n_boot = 2000
    
    results_pairwise = []
    
    print("[Validation] Computing pairwise tail dependence and bootstrap CIs...")
    # For every scale, pair, threshold, compute CI
    for scale in broad_m.scale_order:
        U = broad_m.empirical_U[scale]
        
        for idx_i, idx_j in combinations(range(len(asset_names)), 2):
            asset_i = asset_names[idx_i]
            asset_j = asset_names[idx_j]
            pair_name = f"{asset_i}-{asset_j}"
            
            u1 = U[:, idx_i]
            u2 = U[:, idx_j]
            
            for q in thresholds:
                # Point estimate
                lam_L, lam_U, n, ev_L, ev_U = empirical_tail_dependence(u1, u2, q)
                
                # Bootstrap
                boot_L, boot_U = bootstrap_tail_dependence(u1, u2, q, n_boot=n_boot, seed=cfg.SEED)
                ci_L = np.percentile(boot_L, [2.5, 97.5])
                ci_U = np.percentile(boot_U, [2.5, 97.5])
                
                results_pairwise.append({
                    "scale": scale,
                    "pair": pair_name,
                    "threshold_q": q,
                    "sample_size": n,
                    "tail_event_count_L": ev_L,
                    "tail_event_count_U": ev_U,
                    "lambda_L_est": lam_L,
                    "lambda_L_ci_low": ci_L[0],
                    "lambda_L_ci_high": ci_L[1],
                    "lambda_U_est": lam_U,
                    "lambda_U_ci_low": ci_U[0],
                    "lambda_U_ci_high": ci_U[1],
                })
                
    df_pw = pd.DataFrame(results_pairwise)
    df_pw.to_csv(out_dir / "tail_dependence_pairwise.csv", index=False)
    
    # Create CI dataset
    df_pw.to_csv(out_dir / "tail_dependence_ci.csv", index=False)
    
    # ---------------------------------------------------------
    # Scale Comparisons (D1 vs Dk, SJ)
    # ---------------------------------------------------------
    print("[Validation] Performing multiscale bootstrap comparisons (D1 vs others)...")
    comp_results = []
    
    # We will do comparisons on q=0.05 mostly
    U_D1 = broad_m.empirical_U["D1"]
    
    for scale in broad_m.scale_order[1:]:
        U_Dk = broad_m.empirical_U[scale]
        
        for idx_i, idx_j in combinations(range(len(asset_names)), 2):
            pair_name = f"{asset_names[idx_i]}-{asset_names[idx_j]}"
            
            u1_D1 = U_D1[:, idx_i]
            u2_D1 = U_D1[:, idx_j]
            
            u1_Dk = U_Dk[:, idx_i]
            u2_Dk = U_Dk[:, idx_j]
            
            for q in thresholds:
                lam_L_D1, _, _, _, _ = empirical_tail_dependence(u1_D1, u2_D1, q)
                lam_L_Dk, _, _, _, _ = empirical_tail_dependence(u1_Dk, u2_Dk, q)
                delta = lam_L_Dk - lam_L_D1
                
                # Bootstrap difference
                rng = np.random.default_rng(cfg.SEED)
                n = len(u1_D1)
                boot_deltas = np.zeros(n_boot)
                
                indices = rng.integers(0, n, size=(n_boot, n))
                for b in range(n_boot):
                    idx = indices[b]
                    l1, _, _, _, _ = empirical_tail_dependence(u1_D1[idx], u2_D1[idx], q)
                    lk, _, _, _, _ = empirical_tail_dependence(u1_Dk[idx], u2_Dk[idx], q)
                    boot_deltas[b] = lk - l1
                    
                ci_delta = np.percentile(boot_deltas, [2.5, 97.5])
                sig = not (ci_delta[0] <= 0 <= ci_delta[1])
                
                comp_results.append({
                    "pair": pair_name,
                    "threshold_q": q,
                    "scale_base": "D1",
                    "scale_compare": scale,
                    "delta_est": delta,
                    "delta_ci_low": ci_delta[0],
                    "delta_ci_high": ci_delta[1],
                    "excludes_zero": sig,
                    "bootstrap_p_value": float(min(1.0, 2.0 * min(np.mean(boot_deltas <= 0.0), np.mean(boot_deltas >= 0.0))))
                })
                
    df_comp = pd.DataFrame(comp_results)
    if not df_comp.empty:
        # Benjamini-Hochberg FDR correction over the exploratory scale/pair
        # bootstrap p-values at each threshold. This does not change estimates.
        for q in thresholds:
            mask = df_comp["threshold_q"] == q
            pvals = df_comp.loc[mask, "bootstrap_p_value"].to_numpy(dtype=float)
            if len(pvals):
                reject, p_adj = benjamini_hochberg(pvals, alpha=0.05)
                df_comp.loc[mask, "fdr_rejected_05"] = reject
                df_comp.loc[mask, "fdr_p_value"] = p_adj
    df_comp.to_csv(out_dir / "tail_dependence_scale_comparison.csv", index=False)

    # Aggregate horizon summary. This is the canonical source for the final report:
    # mean pairwise tail-dependence estimates with a genuine observation-level
    # bootstrap CI, plus a paired D1-vs-scale bootstrap difference.
    horizon_rows = []
    for scale in broad_m.scale_order:
        stats = aggregate_tail_dependence(broad_m.empirical_U[scale], q=0.05, n_boot=n_boot, seed=cfg.SEED)
        horizon_rows.append({
            "scale": scale,
            "threshold_q": 0.05,
            "mean_pair_lambda_L": stats["estimate"],
            "ci_low": stats["ci_low"],
            "ci_high": stats["ci_high"],
            "independence_reference": 0.05,
        })

    d1_U = broad_m.empirical_U["D1"]
    for row in horizon_rows:
        scale = row["scale"]
        if scale == "D1":
            row["delta_vs_D1"] = 0.0
            row["delta_ci_low"] = 0.0
            row["delta_ci_high"] = 0.0
            row["delta_bootstrap_p_value"] = np.nan
            row["delta_fdr_rejected_05"] = False
        else:
            diff = aggregate_scale_difference(d1_U, broad_m.empirical_U[scale], q=0.05, n_boot=n_boot, seed=cfg.SEED)
            row["delta_vs_D1"] = diff["estimate"]
            row["delta_ci_low"] = diff["ci_low"]
            row["delta_ci_high"] = diff["ci_high"]
            row["delta_bootstrap_p_value"] = diff["p_value"]
            row["delta_fdr_rejected_05"] = False

    df_horizon = pd.DataFrame(horizon_rows)
    valid_delta = df_horizon["delta_bootstrap_p_value"].notna()
    if valid_delta.any():
        reject, p_adj = benjamini_hochberg(
            df_horizon.loc[valid_delta, "delta_bootstrap_p_value"].to_numpy(dtype=float),
            alpha=0.05,
        )
        df_horizon.loc[valid_delta, "delta_fdr_rejected_05"] = reject
        df_horizon.loc[valid_delta, "delta_fdr_p_value"] = p_adj
    df_horizon.to_csv(out_dir / "tail_dependence_horizon_summary.csv", index=False)

    # ---------------------------------------------------------
    # Stress Period Comparison
    # ---------------------------------------------------------
    print("[Validation] Computing descriptive stress-period comparisons...")
    stress_results = []
    stress_start = pd.Timestamp("2020-02-01")
    stress_end = pd.Timestamp("2020-06-30")

    # Use the SAME broad-sample marginal fits for stress observations rather than
    # refitting a separate stress model.  The stress analysis is retrospective and
    # descriptive: it is not used in the rolling OOS forecasts.
    stress_full = returns.loc[:stress_end].copy()
    if not stress_full.empty and stress_full.index.min() <= stress_start:
        decomp = AdditiveMODWT(wavelet=cfg.WAVELET, levels=cfg.LEVELS)
        stress_components = {scale: [] for scale in broad_m.scale_order}
        for col in asset_names:
            details, smooth = decomp.decompose(stress_full[col].to_numpy(dtype=float))
            for j in range(cfg.LEVELS):
                stress_components[f"D{j + 1}"].append(details[j])
            stress_components[f"S{cfg.LEVELS}"].append(smooth)

        full_index = stress_full.index
        filter_mask = full_index >= pd.Timestamp("2020-01-01")
        stress_mask = (full_index >= stress_start) & (full_index <= stress_end)

        stress_U_by_scale = {}
        for scale in broad_m.scale_order:
            cols = []
            for i in range(len(asset_names)):
                component_all = np.asarray(stress_components[scale][i], dtype=float)[filter_mask]
                marginal = broad_m.scale_models[scale]["marginals"][i]
                # Carry the fitted volatility state through January before selecting
                # the February-June stress observations.
                z_after_training = marginal.standardize_new(component_all)
                u_after_training = marginal.transform(z_after_training)
                aligned_index = full_index[filter_mask]
                select_stress = (aligned_index >= stress_start) & (aligned_index <= stress_end)
                cols.append(u_after_training[select_stress])
            stress_U_by_scale[scale] = np.column_stack(cols)

        for scale in broad_m.scale_order:
            U_broad = broad_m.empirical_U[scale]
            U_stress = stress_U_by_scale[scale]
            for idx_i, idx_j in combinations(range(len(asset_names)), 2):
                pair_name = f"{asset_names[idx_i]}-{asset_names[idx_j]}"
                for q in [0.05, 0.10]:
                    lam_L_b, _, nb, _, _ = empirical_tail_dependence(
                        U_broad[:, idx_i], U_broad[:, idx_j], q
                    )
                    lam_L_s, _, ns, ev_Ls, _ = empirical_tail_dependence(
                        U_stress[:, idx_i], U_stress[:, idx_j], q
                    )
                    stress_results.append({
                        "scale": scale,
                        "pair": pair_name,
                        "threshold_q": q,
                        "lambda_L_broad": lam_L_b,
                        "lambda_L_stress": lam_L_s,
                        "N_broad": nb,
                        "N_stress": ns,
                        "events_stress": ev_Ls,
                        "comparison_type": "descriptive; same pre-stress fitted marginal models"
                    })

    df_stress = pd.DataFrame(stress_results)
    if not df_stress.empty:
        df_stress.to_csv(out_dir / "tail_dependence_stress.csv", index=False)

    # ---------------------------------------------------------
    # Figures
    # ---------------------------------------------------------
    print("[Validation] Generating figures...")
    df_q5 = df_pw[df_pw["threshold_q"] == 0.05]
    scale_order = broad_m.scale_order
    
    # 1. CI by horizon for average lower/upper tail dependence
    avg_ci_df = df_q5.groupby("scale")[["lambda_L_est", "lambda_L_ci_low", "lambda_L_ci_high", "lambda_U_est", "lambda_U_ci_low", "lambda_U_ci_high"]].mean().reindex(scale_order)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(scale_order))
    
    # L
    yerr_L = [avg_ci_df["lambda_L_est"] - avg_ci_df["lambda_L_ci_low"], avg_ci_df["lambda_L_ci_high"] - avg_ci_df["lambda_L_est"]]
    ax.errorbar(x_pos - 0.1, avg_ci_df["lambda_L_est"], yerr=yerr_L, fmt="o", color="darkred", label="Lower Tail (q=0.05)")
    
    # U
    yerr_U = [avg_ci_df["lambda_U_est"] - avg_ci_df["lambda_U_ci_low"], avg_ci_df["lambda_U_ci_high"] - avg_ci_df["lambda_U_est"]]
    ax.errorbar(x_pos + 0.1, avg_ci_df["lambda_U_est"], yerr=yerr_U, fmt="s", color="steelblue", label="Upper Tail (q=0.05)")
    
    ax.axhline(0.05, color="black", linestyle="--", alpha=0.5, label="Independence Reference")
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scale_order)
    ax.set_xlabel("Horizon / Scale")
    ax.set_ylabel("Empirical Tail Dependence")
    ax.set_title("Tail Dependence by Horizon (with Bootstrap 95% CIs)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "tail_dependence_by_horizon_ci.png", dpi=200)
    plt.close(fig)
    
    # 2. Pair Heatmap
    import seaborn as sns
    heatmap_data = df_q5.pivot(index="pair", columns="scale", values="lambda_L_est")[scale_order]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="Reds", fmt=".3f", ax=ax)
    ax.set_title("Lower-Tail Dependence (q=0.05) by Pair and Horizon")
    fig.tight_layout()
    fig.savefig(out_dir / "tail_dependence_pair_heatmap.png", dpi=200)
    plt.close(fig)
    
    # 3. Scale difference (Dk - D1)
    df_comp_q5 = df_comp[df_comp["threshold_q"] == 0.05]
    avg_diff = df_comp_q5.groupby("scale_compare")[["delta_est", "delta_ci_low", "delta_ci_high"]].mean().reindex(scale_order[1:])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos_diff = np.arange(len(scale_order) - 1)
    
    yerr_diff = [avg_diff["delta_est"] - avg_diff["delta_ci_low"], avg_diff["delta_ci_high"] - avg_diff["delta_est"]]
    ax.errorbar(x_pos_diff, avg_diff["delta_est"], yerr=yerr_diff, fmt="o", color="purple")
    ax.axhline(0, color="black", linestyle="--")
    
    ax.set_xticks(x_pos_diff)
    ax.set_xticklabels(scale_order[1:])
    ax.set_xlabel("Comparison Scale")
    ax.set_ylabel("Difference (Scale - D1)")
    ax.set_title("Lower-Tail Dependence Difference vs D1 (q=0.05)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "tail_dependence_scale_difference.png", dpi=200)
    plt.close(fig)
    
    # 4. Stress vs Broad
    if not df_stress.empty:
        df_stress_q10 = df_stress[df_stress["threshold_q"] == 0.10] # using 10% for stress for better sample sizes
        avg_stress = df_stress_q10.groupby("scale")[["lambda_L_broad", "lambda_L_stress"]].mean().reindex(scale_order)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        w = 0.35
        x = np.arange(len(scale_order))
        ax.bar(x - w/2, avg_stress["lambda_L_broad"], width=w, label="Broad Sample (2015-2019)")
        ax.bar(x + w/2, avg_stress["lambda_L_stress"], width=w, label="Stress Period (H1 2020)")
        ax.axhline(0.10, color="black", linestyle="--", label="Independence Reference")
        
        ax.set_xticks(x)
        ax.set_xticklabels(scale_order)
        ax.set_ylabel("Lower Tail Dependence (q=0.10)")
        ax.set_title("Tail Dependence: Broad Expansion vs COVID-19 Crash")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "tail_dependence_stress.png", dpi=200)
        plt.close(fig)

    print("[Validation] Tail-dependence validation complete.")
