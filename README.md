# Quant Edge 1.0: Wavelet-Copula Market Risk Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Challenge: SAIFA Quant Edge 1.0](https://img.shields.io/badge/SAIFA-Quant%20Edge%201.0-green.svg)](https://saifa.org)

A study of whether joint downside behaviour differs across wavelet timescales, and whether a richer dependence model improves daily portfolio VaR / ES forecasts. It combines **normalized stationary wavelet (SWT/MODWT-equivalent) decomposition**, **AR(1)-GARCH(1,1) filtering**, **EVT generalized Pareto tails**, **regular vine copulas** (Gaussian fallback) and a **cross-scale Gaussian coupler**.

> **Status of this document.** It follows the project report ("Revised report draft"). The report's scoring was **corrected from saved forecasts** and the full refit of M1-M3 has **not** been independently regenerated. See [Known Issues & Reproducibility Status](#known-issues--reproducibility-status) before relying on any generated file in `results/`.

---

## Table of Contents

1. [Research Question](#research-question)
2. [Executive Summary](#executive-summary)
3. [Portfolio and Experimental Design](#portfolio-and-experimental-design)
4. [Models Compared](#models-compared)
5. [Methodology](#methodology)
6. [Results](#results)
7. [Recommendation](#recommendation)
8. [Known Issues & Reproducibility Status](#known-issues--reproducibility-status)
9. [Installation & Usage](#installation--usage)
10. [CLI Reference](#cli-reference)
11. [Project Directory Layout](#project-directory-layout)
12. [AI Disclosure](#ai-disclosure)

---

## Research Question

> **Does tail dependence change with investment horizon, and what does ignoring it do to measured risk?**

Wavelet scales are frequency bands *within daily returns*. They are not direct estimates of multi-day holding-period VaR. The model comparisons test the present architecture rather than isolating a universal causal effect of ignoring horizon dependence.

---

## Executive Summary

- **Dependence differs between endpoint scales, but not monotonically.** In the 2015-2019 analysis sample, the mean finite-threshold lower-tail coefficient (q = 0.05) is **0.0701 at D1** and **0.0334 at D5**. The paired D5 − D1 difference is **−0.0366** with an ordinary bootstrap 95% interval **[−0.0685, −0.0080]**. D4 has the highest coefficient, so there is no steady decline with scale. The result is sensitive to sample and bootstrap assumptions, and the BH-adjusted p-value for D5 is exactly 0.05 (borderline). No individual asset-pair difference survives FDR correction.
- **The multiscale forecasts under-cover realized losses.** Across 1,764 out-of-sample days (2018-12-24 to 2025-12-30), M2 and M3 breach their 99% VaR on **7.71%** and **7.20%** of days (target 1%). Both fail the coverage and independence diagnostics.
- **M0 and M1 are not rejected** by those diagnostics (breach rates 1.19% and 1.36%). A non-rejection is not proof of correct calibration.
- **Corrected scoring selects M0.** The repository's original FZ0 implementation had a sign error. After correcting it, the implemented selection rule prefers **M0 (historical simulation)** over M1: corrected mean FZ0 **−2.8080 (M0)** vs **−2.7862 (M1)**. The gap (about 0.022) is small and untested, so it is **not** evidence of statistically significant superiority.
- **Recommendation:** keep historical simulation as the operational baseline for this tested setting (M1 as comparator) and use multiscale analysis as a **diagnostic**, not to justify reduced risk limits.

---

## Portfolio and Experimental Design

### Portfolio and data

Equal-weighted (20% each) return proxy for five liquid instruments:

| Instrument | Exposure | Weight |
| :-- | :-- | :-- |
| SPY | US equity ETF | 20% |
| IEF | US Treasury 7-10 year ETF | 20% |
| GLD | Gold ETF | 20% |
| USO | Oil futures exposure through an ETF | 20% |
| BTC-USD | Bitcoin against the US dollar | 20% |

These instruments provide contrasting equity, bond, commodity and cryptocurrency exposures. Equal weights keep the comparison transparent and avoid fitting weights to the evaluation sample. The 2015-2025 period provides a long common history and includes the 2020 market shock.

**Data handling.** Yahoo Finance via `yfinance`, adjusted prices. SPY defines the calendar; prices are forward-filled for at most five rows; incomplete rows are dropped. Asset log returns are ln(P_t / P_{t-1}). The portfolio proxy is the **weighted sum of asset log returns**, which approximates, but is not exactly, the log return of a rebalanced portfolio. Bitcoin returns across stock-market closures span those calendar gaps. USO is a fund proxy with futures-roll effects, not a spot-oil investment.

### Evaluation settings

| Setting | Value |
| :-- | :-- |
| Aligned sample | 5 Jan 2015 to 30 Dec 2025; 2,764 rows |
| Forecast evaluation | 24 Dec 2018 to 30 Dec 2025; 1,764 dates |
| Training and refit | Trailing 1,000 rows; refit every 21 trading days |
| Risk and simulation | Lower-tail α = 0.01; 2,000 draws per model and forecast |
| Tails and seed | EVT tails beyond the 5th and 95th percentiles; seed 42 |

---

## Models Compared

| ID | Specification |
| :-- | :-- |
| **M0** | Historical simulation of the weighted return proxy |
| **M1** | AR-GARCH/EVT marginals and an attempted five-asset vine copula on raw returns (no wavelets) |
| **M2** | Wavelet components, scale-specific marginals and within-scale copulas (**no** cross-scale coupling) |
| **M3** | M2 plus Gaussian stress-rank coupling across scales |

M2 vs M3 isolates cross-scale coupling. M1 vs M2 differs in decomposition, marginals and reconstruction as well as dependence, so it does **not** isolate the effect of ignoring horizon dependence.

---

## Methodology

### 1. Wavelet components

`pywt.swt` with `db4`, 5 levels, `norm=True`, `trim_approx=True`; each detail and the smooth component is reconstructed separately with `iswt`:

```
r_t = D1_t + D2_t + D3_t + D4_t + D5_t + S5_t
```

Approximate bands: D1 1-2 d, D2 2-4 d, D3 4-8 d, D4 8-16 d, D5 16-32 d, S5 > 32 d. Inputs get 250 observations of **symmetric padding** on each side (plus right padding to a multiple of 32); components are cropped back to the original length. This reduces periodic wrap-around, but reflected boundary observations remain an approximation. Exact reconstruction of the training series does not establish correct out-of-sample reconstruction or equivalence of scales to investment holding periods.

### 2. Marginals

Each raw return series (M1) or reconstructed component (M2/M3) is fitted with **AR(1)-GARCH(1,1)** (Gaussian likelihood, EWMA λ = 0.94 fallback if the fit fails). Standardized residuals have an empirical centre and **generalized Pareto tails** beyond the 5th/95th percentiles. Stability choices: residuals clipped to [−15, 15], GPD shape to [−0.49, 0.49]. The PIT is applied to the fitted standardized residuals, consistent with the EVT fitting sample. Simulation inverts the fitted residual distribution and restores the stored forecast mean and volatility.

### 3. Dependence and simulation

- **Within a scale:** an R-vine fit with BIC controls is attempted via `pyvinecopulib`, with a regularized Gaussian copula fallback. Exceptions are suppressed and the run manifest does not record which backend was used, so **no claim is made that every archived fit used an R-vine or BIC selection.**
- **Across scales (M3):** each scale's PIT vector is summarized as a mean normal-score stress score; a Gaussian copula is fitted on the six scalar stress scores, and simulated scale rows are paired by rank while preserving within-scale scenarios. Summed components are weighted to form the portfolio proxy.

### 4. Tail-dependence estimator

For an asset pair and threshold q:

```
lambda_L(q) = count(U_i < q and U_j < q) / (n * q)
```

Primary q = 0.05 (sensitivity: 0.025 and 0.10). Under independent uniform PITs the reference value is q. These are **finite-threshold** coefficients, not limits as q → 0. The headline value is the mean of the ten pairwise coefficients. The bootstrap resamples time rows independently with 2,000 resamples, holds fitted transformations fixed, and uses paired resampling for scale-vs-D1 differences with Benjamini-Hochberg FDR correction. It ignores serial dependence, model-estimation uncertainty and boundary effects, so intervals should be treated as **exploratory**.

### 5. Forecast timing (what is updated between refits)

Every fitted window ends before its forecast origin and the wavelet decomposition occurs inside that window, so no post-origin observations enter a fit. However means and volatilities are stored at the refit date and **not updated on intervening days**: new simulation seeds are used daily, but distributions stay frozen for up to 21 trading days. These are dated out-of-sample forecasts under a periodic-refit design, not fully updated daily conditional GARCH forecasts.

### 6. Risk measures and validation

- VaR is the 1st percentile of the simulated return distribution; ES is the mean at or below it. Negative values are losses. A breach is a realized return below the forecast VaR. A calibrated 99% threshold gives about 17.64 breaches in 1,764 days.
- **Kupiec** unconditional coverage and **Christoffersen** independence tests.
- **Fissler-Ziegel FZ0 joint VaR/ES loss** (lower is better), with ES e < 0:

```
L_FZ0(y, v, e) = - I(y <= v) * (v - y) / (alpha * e) + v / e + ln(-e) - 1
```

> The **original repository code used a positive sign on the first term** (`src/backtests.py`, `term1`), which made larger breaches *reduce* the loss and invalidated the earlier scores and the M1 recommendation derived from them. See [Known Issues](#known-issues--reproducibility-status).

- **Selection rule:** a model is eligible only if both the Kupiec and Christoffersen p-values are ≥ 0.05; among eligible models, the lowest (corrected) FZ0 loss is preferred.

---

## Results

### Out-of-sample VaR validation (1,764 days)

| Model | Breaches | Rate | Mean VaR | Coverage p | Independence p |
| :-- | :-- | :-- | :-- | :-- | :-- |
| M0 | 21 | 1.19% | −3.17% | 0.435 | 0.249 |
| M1 | 24 | 1.36% | −2.94% | 0.149 | 0.416 |
| M2 | 136 | 7.71% | −1.39% | 4.23e-73 | 0.00155 |
| M3 | 127 | 7.20% | −1.46% | 5.85e-65 | 0.00448 |

Neither test rejects M0 or M1 at 5%; M2 and M3 fail both because their VaR thresholds lie too close to zero. Cross-scale coupling lowers the breach rate from 7.71% (M2) to 7.20% (M3), about 0.51 percentage points. Whether that improvement is statistically significant has not been tested, and M3 still breaches about seven times the nominal frequency.

### Expected Shortfall and corrected FZ0

| Model | Mean ES | Original score (sign bug) | Corrected FZ0 | Eligible |
| :-- | :-- | :-- | :-- | :-- |
| M0 | −5.36% | −3.8875 | **−2.8080** | Yes |
| M1 | −3.85% | −4.2653 | −2.7862 | Yes |
| M2 | −1.61% | −9.4896 | 0.8775 | No |
| M3 | −1.70% | −8.8381 | 0.3344 | No |

M0 has the lowest corrected loss among eligible models, so the rule selects **M0**. The small-magnitude ES of M2/M3 does **not** indicate lower risk: it accompanies severe VaR under-coverage and worse corrected scores. Average ES over all dates is not comparable to average loss on breach dates, which condition on different observations.

> The "corrected" values were obtained by rescoring the same saved forecasts per the report. The files currently in `results/` may still show the original scores until the pipeline is re-run with the sign fix applied.

### Tail dependence by scale (lower tail, q = 0.05, 2015-2019 sample)

The multiscale model is fitted on 1,257 observations; the saved pairwise table has 1,256 aligned PIT rows. Independence reference = 0.05.

| Scale | Mean λ_L | 95% interval | Difference from D1 |
| :-- | :-- | :-- | :-- |
| D1 | 0.0701 | [0.0462, 0.0971] | – |
| D2 | 0.0701 | [0.0462, 0.0971] | +0.0000 |
| D3 | 0.0541 | [0.0382, 0.0717] | −0.0159 |
| D4 | 0.0939 | [0.0669, 0.1258] | +0.0239 |
| D5 | 0.0334 | [0.0191, 0.0510] | **−0.0366** |
| S5 | 0.0764 | [0.0557, 0.0987] | +0.0064 |

D5 − D1 has paired 95% interval [−0.0685, −0.0080], supporting a difference between these two endpoint scales under the reported procedure. It does not establish a steady decrease with scale. (D1 and D2 report identical values and intervals in the saved table; this is unexplained and worth verifying.)

### Stress period (descriptive)

The stress comparison fits marginals to 2015-2019, carries them through January 2020, and applies them to the **104** aligned observations from February to June 2020. Wavelet components use data through June, so this is not a sequence of real-time stress forecasts. Coefficients rise at every scale at q = 0.10. Because stress-period PITs use pre-stress transformations, the changes combine altered marginal exceedance rates with joint co-movement and should **not** be read as pure copula-dependence changes.

### Robustness (D5 − D1, first forecast training window)

Common 252-date evaluation period 2018-12-24 to 2019-12-23; dependence comparison uses the first 1,000 training observations (ending 21 Dec 2018) with 500 resamples. This differs from the 2015-2019 sample and 2,000 resamples above.

| Case | Change | D5 − D1 | 95% interval | p-value |
| :-- | :-- | :-- | :-- | :-- |
| Baseline | – | −0.032 | [−0.069, 0.004] | 0.088 |
| A | `TAIL_QUANTILE = 0.90` | −0.044 | [−0.081, −0.010] | 0.016 |
| B | `TRAIN_WINDOW = 750` | −0.064 | [−0.108, −0.029] | 0.000 |
| C | `N_SIM = 1000` | −0.032 | [−0.069, 0.004] | 0.088 |
| D | `SEED = 99` | −0.032 | [−0.065, −0.002] | 0.044 |

All point estimates have D1 above D5. The baseline and Monte Carlo intervals include zero; the EVT, shorter-window and seed-99 intervals exclude it. The seed-only change illustrates bootstrap sensitivity. Changing the Monte Carlo count does not change the empirical dependence estimates, as expected. **Legacy robustness ES scores are excluded** because alternative-case daily forecasts were not saved for rescoring.

---

## Recommendation

For this portfolio proxy and the archived evaluation:

1. **Retain M0 historical simulation** as the primary risk-monitoring baseline, with **M1 as a comparator**. M0 satisfies the calibration filters, has the breach rate closest to 1%, and the lowest corrected FZ0 among eligible models. Treat M0's small loss edge over M1 as a conditional selection result.
2. **Use multiscale analysis as a diagnostic** to flag periods when diversification appears weaker, **not** to reduce risk limits. Current M2/M3 VaR and ES values are too small.
3. **Before reconsidering M2/M3 for operational forecasting:** update conditional states between refits, repair AR mean-parameter handling, verify and log the copula backend, and rerun the complete out-of-sample experiment with recorded dependencies. Support any calibration change with subsequent evaluation rather than accepting a model because it is more complex.

**Limits of the evidence.** Equal-weighted liquid proxies and a weighted-log-return approximation; liquidity, trading costs and capital requirements are not modelled. The ordinary bootstrap ignores serial dependence (a block bootstrap and common-sample comparison would strengthen inference). Stress has only 104 observations. Pairwise effects are heterogeneous. No significance test is provided for the corrected score differences. Conclusions describe the archived implementation and forecasts, not an idealized fully updated wavelet-vine model.

---

## Known Issues & Reproducibility Status

| Issue | Where | Status |
| :-- | :-- | :-- |
| **FZ0 sign error** | `src/backtests.py` (`term1` positive) | Documented in the report; correction rescored saved forecasts and changes the selected model from M1 to M0. **Apply the fix and regenerate** so `final_validation_summary.*`, `final_competition_comparison.csv` and generated recommendations agree with this README. |
| **AR mean parameters** | `src/marginals.py` (`_get_param` lookups `mu`/`constant`/`c`, `ar[1]`) | Candidate names omit `arch`'s usual labels (e.g. `Const`), so the fitted AR mean may not be restored in forecasts. Requires regenerating forecasts. |
| **Frozen state between refits** | `src/backtest.py`, `src/models.py` | Mean/volatility held up to 21 days. |
| **Copula backend not logged** | `src/copulas.py` | Exceptions suppressed; vine vs Gaussian fallback not recorded. |
| **Exploratory bootstrap** | `src/tail_dependence.py` | iid row resampling; no block bootstrap. |
| **Dependencies unpinned** | `requirements.txt` | Lower bounds only; exact library versions are not recorded. |
| **Manifest checksum** | `results/run_manifest.json` | The original data digest differs from a re-serialization of the cached file in the audit environment; use a byte-level checksum and a fixed environment for a fresh reproducible run. |

**What was independently checked (per the report):** date alignment; realized returns vs cached weighted-return series; recomputed breach counts and coverage/independence statistics; reconstructed M0 VaR/ES forecasts from cached returns using the implemented refit schedule (max absolute error < 1.1e-16); rescoring of all four models with the corrected FZ0; aggregate dependence point estimates vs the mean of saved pairwise event counts. Source commit: `c1310e8508da0a6a8e7dcb73f6c0f12e2f95c9ad`.

**Not independently regenerated:** the full M1-M3 model fits and the bootstrap confidence intervals (source analysis tables are preserved in the evidence pack). One-command end-to-end reproduction of the report remains to be confirmed after the remaining fixes.

---

## Installation & Usage

```bash
git clone https://github.com/imaadh-ifthi/saifa_tiramisu.git
cd saifa_tiramisu
pip install -r requirements.txt
```

| Command | Purpose |
| :-- | :-- |
| `python main.py --quick` | Smoke test (750-day window, 60 test days, 3 scales, 1,000 draws) |
| `python main.py` | **Archived configuration:** all out-of-sample days (1,764), 5 scales, 2,000 draws; also runs the horizon analysis and robustness suite |
| `python main.py --full` | 10,000 draws. **Not** the archived configuration |
| `python main.py --refresh` | Re-downloads data; may change the data snapshot |

The default run uses the included return cache in `data/portfolio_returns.csv`. Each run deletes known generated artifacts first and writes `results/run_manifest.json` (data hash, date range, parameters). A `tests/` folder (wavelet decomposition, cross-scale coupling, tail dependence, lower-tail PIT, ablation, final validation) is included; run with `pytest tests/` after installing `pytest`.

---

## CLI Reference

| Argument | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `--quick` | flag | off | Smoke test settings |
| `--full` | flag | off | 5 scales, all OOS days, 10,000 draws |
| `--levels` | int | `5` | Number of wavelet detail scales |
| `--test-days` | int | all days | Limit OOS days |
| `--simulations` | int | `2000` | Monte Carlo draws per forecast |
| `--refit-every` | int | `21` | Refit frequency (trading days) |
| `--train-window` | int | `1000` | Trailing training window |
| `--refresh` | flag | off | Force data re-download |

---

## Project Directory Layout

```text
saifa_tiramisu/
├── main.py                 # Entry point: validation, backtest, report, robustness
├── config.py               # Configuration
├── requirements.txt
├── README.md
├── src/
│   ├── data.py             # Yahoo Finance loader with CSV cache
│   ├── modwt.py            # Normalized SWT-based additive decomposition
│   ├── marginals.py        # AR(1)-GARCH(1,1) + EVT, PIT / inverse PIT
│   ├── copulas.py          # Vine copula with Gaussian fallback
│   ├── cross_scale.py      # Cross-scale Gaussian coupler
│   ├── models.py           # M0 historical sim, M1 heavy-tail, wavelet-vine (M2/M3)
│   ├── tail_dependence.py  # Empirical tail dependence, bootstrap, FDR, stress
│   ├── robustness.py       # Robustness cases
│   ├── risk.py             # VaR and ES
│   ├── backtests.py        # Kupiec, Christoffersen, FZ0, Diebold-Mariano
│   ├── backtest.py         # Rolling out-of-sample loop (M0-M3)
│   ├── report.py           # Summaries and figures
│   └── utils.py
├── tests/
├── data/                   # Cached portfolio_returns.csv
└── results/                # Generated outputs (CSVs, figures, run_manifest.json)
```

`recover.py` and `scratch_test.py` are development scripts; remove them before final submission if not needed.

---

## AI Disclosure

AI tools assisted with learning, report drafting, code review, debugging and evidence checks. The review identified the scoring-sign error and helped implement regression checks. The team remains responsible for the final code, assumptions, calculations and interpretation, and must be able to explain or modify the work live as required by the competition.
