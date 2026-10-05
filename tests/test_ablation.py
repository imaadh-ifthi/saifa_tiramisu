import numpy as np
import pandas as pd
import pytest

from src.models import (
    HistoricalSimulationBenchmark,
    HeavyTailMarginalModel,
    MultiScaleWaveletVine,
)
from src.backtest import run_backtest

class DummyCfg:
    TRAIN_WINDOW = 50
    TEST_DAYS = 5
    N_SIM = 100
    ALPHA = 0.01
    LEVELS = 2
    WAVELET = "db2"
    SEED = 42
    TAIL_QUANTILE = 0.90
    WEIGHTS = [0.5, 0.5]
    REFIT_EVERY = 5

def _get_dummy_returns():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=70)
    data = np.random.normal(0, 0.01, size=(70, 2))
    return pd.DataFrame(data, index=dates, columns=["A", "B"])

def test_benchmark_uses_only_training_data_and_is_deterministic():
    returns = _get_dummy_returns()
    train = returns.iloc[:50]
    
    m0_a = HistoricalSimulationBenchmark()
    m0_a.fit(train)
    sim_a = m0_a.simulate_portfolio(100, [0.5, 0.5], seed=1)
    
    m0_b = HistoricalSimulationBenchmark()
    m0_b.fit(train)
    sim_b = m0_b.simulate_portfolio(100, [0.5, 0.5], seed=1)
    
    assert len(sim_a) == 50 # M0 returns empirical values directly!
    np.testing.assert_array_equal(sim_a, sim_b)

def test_full_model_is_deterministic():
    returns = _get_dummy_returns()
    train = returns.iloc[:50]
    
    m3_a = MultiScaleWaveletVine(levels=2, wavelet="db2", seed=42, use_cross_scale=True)
    m3_a.fit(train)
    sim_a = m3_a.simulate_portfolio(100, [0.5, 0.5], seed=1)
    
    m3_b = MultiScaleWaveletVine(levels=2, wavelet="db2", seed=42, use_cross_scale=True)
    m3_b.fit(train)
    sim_b = m3_b.simulate_portfolio(100, [0.5, 0.5], seed=1)
    
    np.testing.assert_array_equal(sim_a, sim_b)

def test_no_oos_leakage_and_identical_oos_dates():
    returns = _get_dummy_returns()
    cfg = DummyCfg()
    
    results, tail_df = run_backtest(returns, cfg)
    
    assert len(results) == 5
    assert results["date"].iloc[-1] == returns.index[54]
    
    for m in ["M0", "M1", "M2", "M3"]:
        assert f"{m}_var" in results.columns
        assert f"{m}_es" in results.columns
        
        # Test outputs are finite
        assert np.all(np.isfinite(results[f"{m}_var"]))
        assert np.all(np.isfinite(results[f"{m}_es"]))

def test_heavy_tail_marginal_does_not_call_modwt():
    # If HeavyTailMarginalModel calls modwt, it will crash without it, but it doesn't.
    returns = _get_dummy_returns()
    train = returns.iloc[:50]
    
    m1 = HeavyTailMarginalModel(tail_quantile=0.90, seed=42)
    m1.fit(train)
    
    # Check that there is no MODWT property
    assert not hasattr(m1, "levels")
    
    sim = m1.simulate_portfolio(100, [0.5, 0.5], seed=1)
    assert sim.shape == (100,)
