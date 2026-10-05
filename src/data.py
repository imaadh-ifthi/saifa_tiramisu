"""
Data download and preparation.

Data is downloaded dynamically to keep the ZIP file small.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from config import ASSETS, START_DATE, END_DATE, DATA_DIR


def load_returns(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load daily log returns for the selected portfolio.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and one column per asset.
    """
    cache_path = DATA_DIR / "portfolio_returns.csv"

    if cache_path.exists() and not force_refresh:
        returns = pd.read_csv(cache_path, index_col=0, parse_dates=True)

        # Normalize legacy cache naming before returning cached data.  An older
        # generated file used ``BT-CUSD`` for Bitcoin; the current canonical
        # ticker is ``BTC-USD`` from config.ASSETS.
        if "BT-CUSD" in returns.columns and "BTC-USD" not in returns.columns:
            returns = returns.rename(columns={"BT-CUSD": "BTC-USD"})

        expected = list(ASSETS.keys())
        if all(col in returns.columns for col in expected):
            return returns[expected].dropna()

        print("[Data] Cached returns are missing expected assets; refreshing from Yahoo Finance...")

    tickers = list(ASSETS.keys())

    print("[Data] Downloading data from Yahoo Finance...")

    raw = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise RuntimeError("No data downloaded. Check internet connection or tickers.")

    # yfinance often returns MultiIndex columns.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw.copy()

    prices = prices.sort_index()

    # Use SPY as the trading calendar to avoid weekend crypto-only rows.
    if "SPY" in prices.columns:
        prices = prices.loc[prices["SPY"].notna()]

    # Small forward fill for holidays / minor missing observations.
    prices = prices.ffill(limit=5)

    # Keep only complete rows.
    prices = prices.dropna()

    if prices.shape[0] < 500:
        raise RuntimeError("Insufficient overlapping data for backtest.")

    # Reorder columns according to config.
    prices = prices[list(ASSETS.keys())]

    # Daily log returns.
    returns = np.log(prices / prices.shift(1)).dropna()

    returns.to_csv(cache_path)

    print(f"[Data] Saved returns with shape {returns.shape}")

    return returns
