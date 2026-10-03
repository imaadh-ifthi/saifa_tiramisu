# Quant Edge 1.0: Wavelet-Copula Market Risk Framework

This repository contains an end-to-end Python implementation for the SAIFA Quant Edge 1.0 Initial Screening Challenge.

## Core question

Does tail dependence change with investment horizon, and what is the cost of ignoring this in portfolio risk measurement?

## Framework

The submission implements:

1. A multi-asset portfolio:
   - SPY: US equities
   - IEF: US 7-10 year Treasury exposure
   - GLD: gold
   - USO: crude oil exposure
   - BTC-USD: Bitcoin

2. Multiscale return decomposition:
   - wavelet-style multiresolution decomposition using Daubechies filters;
   - reflection boundary handling to reduce forward-looking boundary leakage.

3. Marginal models:
   - AR(1)-GARCH(1,1) filtering;
   - EVT Peaks-Over-Threshold tail splicing using the Generalized Pareto distribution;
   - Probability Integral Transform to uniform marginals.

4. Dependence models:
   - Regular Vine copulas at each timescale using `pyvinecopulib`;
   - Gaussian copula fallback if `pyvinecopulib` is unavailable.

5. Risk metrics:
   - 99% Value-at-Risk;
   - 99% Expected Shortfall.

6. Benchmarks:
   - Gaussian copula applied directly to aggregate returns, ignoring timescale separation and heavy tails.

7. Out-of-sample tests:
   - rolling/expanding backtest;
   - Kupiec VaR test;
   - Christoffersen independence diagnostic;
   - Fissler-Ziegel joint VaR/ES loss;
   - Diebold-Mariano test of predictive accuracy.

## One-command reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full default backtest:

```bash
python main.py
```

For a faster smoke test:

```bash
python main.py --quick
```

Outputs are written to:

```text
results/
```

including:

- `backtest_results.csv`
- `tail_dependence_by_scale.csv`
- `summary.txt`
- `tail_dependence_by_horizon.png`
- `var_forecasts.png`
- `fissler_ziegel_cumulative_loss.png`

## Data

Data is downloaded automatically from Yahoo Finance using `yfinance`.

The default sample starts in 2015 so that Bitcoin is available. The portfolio choices are justified in the report.

## AI disclosure

AI tools were used for coding assistance, debugging, and drafting documentation. The team remains fully responsible for all methodological choices, calculations, code, and statements in the submission.
