"""
Main entry point for the SAIFA Quant Edge 1.0 submission.

Run:
    python main.py

Quick smoke test:
    python main.py --quick
"""

import argparse
import random

import numpy as np

import config as cfg

from src.data import load_returns
from src.backtest import run_backtest
from src.report import save_report


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

    results, tail_df = run_backtest(returns, cfg)

    save_report(results, tail_df, cfg)

    print("\n[Done] Outputs saved to results/")


if __name__ == "__main__":
    main()
