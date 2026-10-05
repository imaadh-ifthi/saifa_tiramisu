import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from src.marginals import MarginalARQGARCH
from src.models import MultiScaleWaveletVine


def _dummy_returns(n=220, assets=3):
    rng = np.random.default_rng(123)
    x = rng.normal(0.0, 0.01, size=(n, assets))
    return pd.DataFrame(x, columns=[f"A{i}" for i in range(assets)])


def test_pit_is_based_on_standardized_residuals():
    x = _dummy_returns(n=220, assets=1).iloc[:, 0].to_numpy()
    model = MarginalARQGARCH(tail_quantile=0.95).fit(x)

    expected = model.transform(model.z)
    actual = model.in_sample_pit()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    # The old bug transformed raw component returns directly. For financial-scale
    # data that should not, in general, equal the standardized-residual PIT.
    raw_pit = model.transform(x[: len(model.z)])
    assert np.max(np.abs(actual - raw_pit)) > 1e-8


def test_multiscale_empirical_u_comes_from_marginal_pit():
    returns = _dummy_returns()
    model = MultiScaleWaveletVine(levels=2, wavelet="db2", seed=42, use_cross_scale=True)
    model.fit(returns)

    for scale in model.scale_order:
        U = model.empirical_U[scale]
        marginal = model.scale_models[scale]["marginals"][0]
        expected = marginal.in_sample_pit()
        assert U.shape[0] == expected.shape[0]
        np.testing.assert_allclose(U[:, 0], expected, rtol=0.0, atol=0.0)


def test_report_does_not_use_legacy_tail_dependence_results_file():
    from src import report
    source = inspect.getsource(report.save_report)
    assert "tail_dependence_results.csv" not in source
    assert "tail_dependence_horizon_summary.csv" in source


def test_results_are_not_mixed_across_runs_when_manifest_exists():
    manifest = Path("results/run_manifest.json")
    if not manifest.exists():
        return
    data = manifest.read_text(encoding="utf-8")
    assert '"data_source": "Yahoo Finance (via yfinance)"' in data


def test_fixed_oos_controls_are_supported():
    from src.backtest import run_backtest

    class Cfg:
        TRAIN_WINDOW = 100
        TEST_DAYS = 20
        OOS_START_INDEX = 150
        OOS_END_INDEX = 160
        N_SIM = 50
        ALPHA = 0.01
        LEVELS = 2
        WAVELET = "db2"
        SEED = 42
        TAIL_QUANTILE = 0.90
        WEIGHTS = [0.5, 0.5]
        REFIT_EVERY = 10

    returns = _dummy_returns(n=220, assets=2)
    results, _ = run_backtest(returns, Cfg())
    assert len(results) == 10
    assert results["date"].iloc[0] == returns.index[150]
    assert results["date"].iloc[-1] == returns.index[159]


def test_existing_final_comparison_is_not_zeroed_when_horizon_summary_exists():
    horizon_path = Path("results/tail_dependence_horizon_summary.csv")
    final_path = Path("results/final_competition_comparison.csv")
    if not horizon_path.exists() or not final_path.exists():
        return

    horizon = pd.read_csv(horizon_path)
    final = pd.read_csv(final_path).iloc[0]
    d1 = float(horizon.loc[horizon["scale"] == "D1", "mean_pair_lambda_L"].iloc[0])
    d5 = float(horizon.loc[horizon["scale"] == "D5", "mean_pair_lambda_L"].iloc[0])

    assert np.isclose(float(final["tail_dependence_D1"]), d1)
    assert np.isclose(float(final["tail_dependence_D5"]), d5)
