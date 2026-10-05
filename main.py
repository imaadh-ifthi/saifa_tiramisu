"""
Main entry point for the SAIFA Quant Edge 1.0 submission.

Run:
    python main.py

Quick smoke test:
    python main.py --quick
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

import config as cfg

from src.data import load_returns
from src.backtest import run_backtest
from src.report import save_report




GENERATED_RESULT_FILES = [
    "backtest_results.csv",
    "model_comparison.csv",
    "model_comparison_by_window.csv",
    "final_validation_summary.csv",
    "final_validation_summary.txt",
    "final_competition_comparison.csv",
    "robustness_summary.csv",
    "tail_dependence_pairwise.csv",
    "tail_dependence_ci.csv",
    "tail_dependence_scale_comparison.csv",
    "tail_dependence_horizon_summary.csv",
    "tail_dependence_stress.csv",
    "tail_dependence_by_scale.csv",
    "tail_dependence_by_horizon.png",
    "tail_dependence_by_horizon_ci.png",
    "tail_dependence_pair_heatmap.png",
    "tail_dependence_scale_difference.png",
    "tail_dependence_stress.png",
    "oos_var_timeseries.png",
    "var_forecasts.png",
    "var_distribution.png",
    "avg_es.png",
    "breach_rate.png",
    "ablation_es_score.png",
    "fissler_ziegel_cumulative_loss.png",
    "final_oos_var_comparison.png",
    "final_es_comparison.png",
    "final_breach_calibration.png",
    "final_stress_comparison.png",
    "robustness_summary.png",
    "summary.txt",
]


def clean_results(results_dir):
    """Remove only known generated artifacts so results never mix runs."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    for name in GENERATED_RESULT_FILES:
        path = results_dir / name
        if path.exists():
            path.unlink()


def write_run_manifest(returns, cfg, results_dir):
    """Record the exact data/configuration used for a reproducible run."""
    data_bytes = returns.to_csv(date_format="%Y-%m-%d").encode("utf-8")
    manifest = {
        "data_source": "Yahoo Finance (via yfinance)",
        "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "assets": list(returns.columns),
        "data_start": str(returns.index.min().date()),
        "data_end": str(returns.index.max().date()),
        "observations": int(len(returns)),
        "levels": int(cfg.LEVELS),
        "wavelet": str(cfg.WAVELET),
        "train_window": int(cfg.TRAIN_WINDOW),
        "test_days": None if cfg.TEST_DAYS is None else int(cfg.TEST_DAYS),
        "refit_every": int(cfg.REFIT_EVERY),
        "n_sim": int(cfg.N_SIM),
        "alpha": float(cfg.ALPHA),
        "seed": int(cfg.SEED),
        "tail_quantile": float(cfg.TAIL_QUANTILE),
        "weights": [float(x) for x in np.asarray(cfg.WEIGHTS)],
    }
    Path(results_dir).joinpath("run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def apply_overrides(args):
    """
    Apply CLI overrides to the config module.
    """
    if args.quick:
        cfg.LEVELS = 3
        cfg.TEST_DAYS = 60
        cfg.N_SIM = 1000
        cfg.REFIT_EVERY = 20
        cfg.TRAIN_WINDOW = 750

    elif args.full:
        cfg.LEVELS = 5
        cfg.TEST_DAYS = None
        cfg.N_SIM = 10000
        cfg.REFIT_EVERY = 21
        cfg.TRAIN_WINDOW = 1000

    if args.levels is not None:
        cfg.LEVELS = int(args.levels)

    if args.test_days is not None:
        cfg.TEST_DAYS = int(args.test_days)

    if args.simulations is not None:
        cfg.N_SIM = int(args.simulations)

    if args.refit_every is not None:
        cfg.REFIT_EVERY = int(args.refit_every)

    if args.train_window is not None:
        cfg.TRAIN_WINDOW = int(args.train_window)


def main():
    parser = argparse.ArgumentParser(
        description="Quant Edge 1.0: Wavelet-Copula Market Risk Framework"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a faster smoke-test backtest.",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run a heavier full backtest.",
    )

    parser.add_argument(
        "--levels",
        type=int,
        default=None,
        help="Number of wavelet decomposition levels.",
    )

    parser.add_argument(
        "--test-days",
        type=int,
        default=None,
        help="Number of out-of-sample days.",
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=None,
        help="Monte Carlo simulations per forecast.",
    )

    parser.add_argument(
        "--refit-every",
        type=int,
        default=None,
        help="Refit model every N days.",
    )

    parser.add_argument(
        "--train-window",
        type=int,
        default=None,
        help="Training window length in days.",
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download of data.",
    )

    args = parser.parse_args()

    # Reproducibility seeds.
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    apply_overrides(args)

    print("=" * 70)
    print("SAIFA Quant Edge 1.0")
    print("Wavelet-Copula Market Risk Framework")
    print("=" * 70)

    returns = load_returns(force_refresh=args.refresh)

    # Ensure weights match the number of return columns.
    if len(cfg.WEIGHTS) != returns.shape[1]:
        cfg.WEIGHTS = np.ones(returns.shape[1]) / returns.shape[1]
    else:
        cfg.WEIGHTS = np.asarray(cfg.WEIGHTS, dtype=float)
        cfg.WEIGHTS = cfg.WEIGHTS / cfg.WEIGHTS.sum()

    results_dir = Path(cfg.RESULTS_DIR)
    clean_results(results_dir)
    write_run_manifest(returns, cfg, results_dir)

    from src.tail_dependence import run_validation
    run_validation(returns, cfg)

    results, tail_df = run_backtest(returns, cfg)

    save_report(results, tail_df, cfg)
    
    from src.robustness import run_robustness
    run_robustness(returns, cfg)

    print("\n[Done] Outputs saved to results/")


if __name__ == "__main__":
    main()
