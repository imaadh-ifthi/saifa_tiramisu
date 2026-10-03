"""
Global configuration for the Quant Edge 1.0 submission.
"""

from pathlib import Path
import numpy as np


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Portfolio definition
# ------------------------------------------------------------
# We use ETF / liquid proxy instruments that are available from 2015.
# BTC-USD is included from 2015 onward.
ASSETS = {
    "SPY": "US Equity",
    "IEF": "US Treasury 7-10Y",
    "GLD": "Gold",
    "USO": "Crude Oil",
    "BTC-USD": "Bitcoin",
}

# Equal-weighted portfolio.
WEIGHTS = np.array([1.0 / len(ASSETS)] * len(ASSETS))


# ------------------------------------------------------------
# Sample period
# ------------------------------------------------------------
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"


# ------------------------------------------------------------
# Wavelet / multiresolution settings
# ------------------------------------------------------------
# db4 is the Daubechies wavelet with filter length 8.
WAVELET = "db4"

# Number of decomposition levels.
# D1: roughly 1-2 days
# D2: roughly 2-4 days
# D3: roughly 4-8 days
# D4: roughly 8-16 days
# D5: roughly 16-32 days
# S5: longer-term smooth component
LEVELS = 5


# ------------------------------------------------------------
# Backtest settings
# ------------------------------------------------------------
TRAIN_WINDOW = 1000
TEST_DAYS = 252
REFIT_EVERY = 21


# ------------------------------------------------------------
# Monte Carlo / risk settings
# ------------------------------------------------------------
N_SIM = 2000
ALPHA = 0.01
SEED = 42


# ------------------------------------------------------------
# Marginal / EVT settings
# ------------------------------------------------------------
TAIL_QUANTILE = 0.95
