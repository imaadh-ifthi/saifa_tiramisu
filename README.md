# Quant Edge 1.0 — Tiramisu_

## Risk Across Tails and Timescales

A reproducible research framework for testing whether portfolio tail dependence changes across wavelet timescales, and what happens to measured portfolio risk when that structure is ignored.

**Challenge:** SAIFA Quant Edge 1.0 — Initial Screening Challenge  
**Team:** Tiramisu_ — University of Peradeniya  
**Primary assets:** SPY, IEF, GLD, USO, BTC-USD  
**Risk measures:** 99% VaR and Expected Shortfall (ES)

---

##  Quickstart: One-Command Reproduction

Run the full pipeline or smoke test directly from your terminal:

```bash
# 1. Clone & install
git clone https://github.com/imaadh-ifthi/saifa_tiramisu.git
cd saifa_tiramisu
pip install -r requirements.txt

# 2. Fast 30-second smoke test
python main.py --quick

# 3. Full end-to-end experiment (1,764 OOS days + robustness + report)
python main.py

# 4. Run test suite (61 tests)
pytest tests/
```

---

## 1. What are we trying to answer?

> **Does tail dependence change with investment horizon, and what does ignoring it do to a portfolio's measured risk?**

The project separates this question into two experiments:

1. **Dependence experiment:** measure whether downside co-exceedances differ across wavelet scales.
2. **Forecasting experiment:** test whether adding increasingly rich dependence structure improves out-of-sample 99% VaR/ES forecasts.

This distinction is important. A model can detect interesting multiscale dependence without necessarily producing a better operational risk forecast.

Wavelet scales in this project are **frequency bands within daily returns**. They are not direct estimates of multi-day holding-period VaR.

---

## 2. The central idea

Financial losses are not driven only by the behaviour of individual assets. Assets can become more strongly connected in their downside tails, and that relationship can depend on the timescale of the return component being examined.

Our framework therefore moves through four increasingly structured models:

```text
M0  Historical simulation
        |
        v
M1  Heavy-tailed marginals + cross-asset dependence
        |
        v
M2  Wavelet decomposition + scale-specific dependence
        |
        v
M3  M2 + explicit cross-scale coupling
```

The purpose of M2 and M3 is not to assume that a more complicated model must win. Their purpose is to **test whether the additional structure earns its complexity out of sample**.

That is why the benchmark is kept in the experiment from beginning to end.

---

## 3. Main result at a glance

### Dependence changes across timescales

In the main 2015–2019 dependence sample, the mean finite-threshold lower-tail coefficient at `q = 0.05` is:

| Scale | Mean lower-tail coefficient |
|---|---:|
| D1 | 0.0701 |
| D2 | 0.0701 |
| D3 | 0.0557 |
| D4 | 0.0924 |
| **D5** | **0.0525** |
| S5 | 0.0764 |

The paired **D5 − D1 difference is −0.0175**, with ordinary-bootstrap 95% interval **[−0.0494, 0.0111]**. In this updated snapshot, the interval includes zero, showing that empirical horizon differences remain sample-sensitive.

### The richer forecasting models do not currently calibrate well enough

Across 1,764 out-of-sample observations (target: ~1% breaches):

| Model | Description | VaR breaches | Breach rate | Status |
|---|---|---:|---:|---|
| **M0** | Historical simulation | 21 | **1.19%** | Eligible (Selected) |
| **M1** | AR-GARCH/EVT + copula | 25 | **1.42%** | Eligible |
| M2 | Wavelet + within-scale copulas | 202 | **11.45%** | Ineligible |
| M3 | M2 + cross-scale coupling | 190 | **10.77%** | Ineligible |

The target for a 99% VaR is approximately **1%** breaches (17.64 expected).

M3 improves on M2 by about **0.68 percentage points** (breaches fall from 202 to 190), confirming that cross-scale interaction is present, but it remains far from operational calibration. Therefore the multiscale models are used as **diagnostic/research models**, not for production risk limits.

---

## 4. Why M0 can be selected even though it is the baseline

M0 is the benchmark, but it is **not automatically declared the winner**.

The implemented selection rule is:

1. Reject models that fail either the unconditional coverage or breach-independence diagnostic at the 5% level.
2. Among the remaining models, prefer the lower mean Fissler–Ziegel FZ0 loss.

Under that rule:

| Model | Coverage p | Independence p | Corrected mean FZ0 | Eligible? |
|---|---:|---:|---:|---|
| **M0** | **0.435** | **0.249** | **−2.8080** | **Yes (Selected)** |
| **M1** | 0.097 | 0.396 | **−2.7475** | Yes |
| M2 | < 10⁻¹⁰⁰ | < 10⁻¹⁵ | Severe penalty | No |
| M3 | < 10⁻¹⁰⁰ | < 10⁻¹³ | Severe penalty | No |

So M0 is selected because it is the **best-performing eligible model under the predefined rule**, not because it is the baseline.

The M0–M1 FZ0 difference is modest and no formal significance test of the difference is provided. The correct interpretation is therefore **conditional model selection**, not proof that historical simulation is universally superior.

---

## 5. Models compared

### M0 — Historical Simulation

Directly estimates the empirical distribution of the weighted portfolio return proxy.

**Purpose:** simple, transparent benchmark.

### M1 — Heavy-Tail Marginals + Cross-Asset Dependence

For each asset:

- AR(1)-GARCH(1,1) volatility filtering
- empirical central distribution
- generalized Pareto tails (EVT)
- probability integral transform (PIT)

The transformed asset returns are then modelled jointly using an attempted five-asset R-vine copula, with a regularized Gaussian fallback.

**Purpose:** test whether explicit heavy tails and cross-asset dependence improve over historical simulation without introducing wavelet scales.

### M2 — Wavelet + Within-Scale Dependence

Each asset is decomposed into five detail components and one smooth component:

```text
r_t = D1_t + D2_t + D3_t + D4_t + D5_t + S5_t
```

Each scale receives its own marginal model and within-scale dependence model.

**Purpose:** test whether modelling risk separately across timescales changes the resulting portfolio tail forecast.

### M3 — Full Cross-Scale Model

M3 extends M2 by adding a Gaussian cross-scale coupling layer.

Each scale is summarized by a scalar stress score derived from the mean normal-score PIT across assets. A Gaussian copula is then used to rank-pair simulated scale scenarios while preserving each scale's within-scale scenario structure.

**Purpose:** test whether dependence between timescales contains additional information beyond modelling each scale independently.

---

## 6. Methodology

### 6.1 Data

Five assets with equal 20% weights:

| Asset | Role |
|---|---|
| SPY | US equities |
| IEF | US intermediate Treasuries |
| GLD | Gold |
| USO | Oil exposure via ETF |
| BTC-USD | Bitcoin |

Data are downloaded through `yfinance` using adjusted prices. SPY defines the calendar; prices are forward-filled for at most five rows and incomplete rows are removed.

Asset log returns are:

```text
r_t = ln(P_t / P_{t-1})
```

The portfolio proxy is the weighted sum of asset log returns. This is a transparent approximation to a rebalanced portfolio log return, not an exact rebalanced-portfolio calculation.

### 6.2 Wavelet decomposition

The implementation uses PyWavelets stationary wavelet transforms with:

- wavelet: `db4`
- levels: `5`
- `norm=True`
- separate reconstruction of each detail and the smooth component
- symmetric boundary padding

This is a **normalized SWT/MODWT-equivalent implementation**. It is not a direct claim that the resulting bands are literal investment holding periods.

### 6.3 Marginal tail model

Each raw return series in M1, or each reconstructed wavelet component in M2/M3, is filtered using:

```text
AR(1) mean + GARCH(1,1) volatility
                    |
                    v
          standardized residuals
                    |
            empirical centre
              + GPD tails
                    |
                   PIT
```

The EVT tails begin beyond the 5th and 95th percentiles.

A failed GARCH fit falls back to EWMA with decay `0.94`.

Stability constraints in the implementation clip standardized residuals to `[-15, 15]` and fitted GPD shape to `[-0.49, 0.49]`.

### 6.4 Dependence model

Within each scale the framework attempts an R-vine fit with BIC controls. A regularized Gaussian copula is available as a fallback.

For M3, a separate cross-scale Gaussian coupler operates on six scalar stress scores.

The implementation therefore does **not** claim that every archived scale fit successfully used a full R-vine. The saved run should record the backend used for future audited runs.

### 6.5 Tail-dependence estimator

For an asset pair and threshold `q`:

```text
lambda_L(q) = count(U_i < q and U_j < q) / (n q)
```

The primary threshold is `q = 0.05`, with `q = 0.025` and `q = 0.10` used as sensitivity checks.

These are finite-threshold co-exceedance coefficients, not asymptotic tail-dependence limits as `q -> 0`.

### 6.6 Risk measures

For each model and forecast date:

- **VaR:** 1st percentile of the simulated portfolio return distribution
- **ES:** mean simulated return at or below the VaR threshold

Negative values correspond to losses.

---

## 7. Out-of-sample design

The common aligned sample contains **2,764 observations** from **5 January 2015 to 30 December 2025**.

The forecasting evaluation begins on **24 December 2018** and contains **1,764 dates**.

| Setting | Value |
|---|---:|
| Training window | 1,000 observations |
| Refit frequency | Every 21 trading days |
| Monte Carlo draws | 2,000 per forecast |
| VaR level | 99% |
| EVT thresholds | 5% / 95% |
| Main random seed | 42 |

Every fitting window ends before its forecast origin. Therefore post-origin observations are not used to fit the corresponding forecast.

**Important:** conditional means and volatilities are currently stored at refit time and held fixed between refits. The system is therefore a **periodic-refit out-of-sample framework**, not a fully state-updated daily GARCH implementation.

---

## 8. Validation

### Calibration

We use:

- **Kupiec unconditional coverage test**
- **Christoffersen independence diagnostic**

At 99% VaR, a calibrated model should produce about **17.64 breaches in 1,764 observations**.

### Joint VaR/ES scoring

The study also uses the return-based **FZ0 joint VaR/ES loss**. Lower average loss is preferred.

The sign in the original scoring implementation was corrected before the reported model-selection result was obtained. The correction changes the FZ scores and model selection, but **does not change the underlying VaR, ES or breach counts**.

---

## 9. Interpretation of the results

The project produces three separate findings.

### Finding 1 — Tail dependence is scale-dependent in the selected sample

The D1 and D5 lower-tail estimates differ in point estimate, but the updated bootstrap interval for D5 − D1 includes zero. The evidence for a difference is therefore not statistically conclusive in this snapshot.

This is evidence of **timescale-specific downside co-movement in this sample**.

It is not evidence of a universal monotonic horizon law.

### Finding 2 — Detecting multiscale dependence is easier than forecasting with it

M2 and M3 change the risk distribution substantially, but their 99% VaR thresholds are too close to zero and consequently understate the frequency of realized losses.

M3 is directionally better than M2, but the improvement is not enough for operational use.

### Finding 3 — The benchmark remains the safest operational choice for this tested setting

M0 passes the reported calibration diagnostics and has the lowest corrected FZ0 loss among eligible models.

Therefore the practical recommendation is:

> **Use M0 historical simulation as the primary operational risk-monitoring baseline for this tested portfolio. Use the multiscale framework as a dependence/stress diagnostic until its forecasting calibration is improved and revalidated.**

This is deliberately a conservative conclusion: a more sophisticated model is not accepted merely because it is more sophisticated.

---

## 10. Stress analysis

A separate 2020 stress comparison is included as a **descriptive diagnostic**.

Marginal transformations are fitted on 2015–2019 and carried into the 2020 stress period. Because the wavelet components use data through the stress period, this analysis is not a sequence of real-time forecasts.

The observed lower-tail co-movement rises across scales during the stress period, with the largest descriptive increase at S5.

These results should not be interpreted as pure copula-dependence changes because stress-period PIT probabilities are based on pre-stress marginal transformations.

---

## 11. Robustness

The D5 − D1 dependence difference remains negative across the tested specifications, but its reported interval changes with sample and bootstrap settings.

The baseline and Monte Carlo cases include zero, while some threshold/window/seed variants exclude zero.

This means the dependence result should be treated as **evidence in the selected sample, not a universal law**.

A block bootstrap and common-sample analysis would provide stronger time-series inference.

---

## 12. Limitations and audit status

This project intentionally separates what is supported by the saved evidence from what still requires a clean regenerated run.

### Important implementation limitations

- Conditional means and volatilities are frozen between 21-day refits.
- The AR mean-parameter lookup required correction to account for the parameter labels emitted by `arch` (for example `Const` and `y[1]`).
- Copula fallback decisions should be logged explicitly rather than silently suppressed.
- The bootstrap used for the main tail-dependence intervals resamples time rows independently and therefore does not fully capture serial dependence or estimation uncertainty.
- Wavelet boundary handling and scenario reconstruction can affect forecast dispersion.
- Library versions should be pinned for exact byte-for-byte reproduction.

### What has been independently checked

The audit of the archived evidence checked:

- date alignment;
- realized returns against the cached weighted return series;
- the four breach counts;
- coverage and independence statistics;
- reconstruction of the M0 forecasts;
- corrected FZ0 rescoring;
- aggregate tail-dependence point estimates against the saved pairwise event counts.

The full M1–M3 fits and bootstrap intervals in the archived evidence were not independently regenerated in the audit environment. Therefore the current report should be understood as a **validated analysis of the saved run**, not as proof that a fresh environment will reproduce every archived M1–M3 number bit-for-bit without rerunning the model.

Do not silently replace archived numbers with a newly generated run: record the new configuration and compare it explicitly.

---

## 13. Reproducibility

### Install

```bash
git clone https://github.com/imaadh-ifthi/saifa_tiramisu.git
cd saifa_tiramisu
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Run a smoke test

```bash
python main.py --quick
```

### Run the configured experiment

```bash
python main.py
```

This uses the repository's configured data cache and default experimental settings.

### Optional controls

```bash
python main.py --test-days 100
python main.py --simulations 1000
python main.py --refit-every 21
python main.py --train-window 1000
python main.py --refresh
```

`--refresh` downloads a fresh Yahoo Finance snapshot and may therefore produce a different data hash or numerical result from the archived run.

`--full` increases the Monte Carlo workload and is **not** the archived competition configuration.

### Tests

```bash
pytest tests/
```

The tests cover wavelet reconstruction, cross-scale coupling, tail dependence, lower-tail PIT behaviour, model ablation and final validation logic.

### Outputs

The main pipeline writes generated artefacts under `results/`, including:

```text
results/
├── final_validation_summary.csv
├── final_competition_comparison.csv
├── model_comparison.csv
├── tail_dependence_scale_comparison.csv
├── robustness_summary.csv
├── final_oos_var_comparison.png
├── final_breach_calibration.png
├── final_es_comparison.png
├── tail_dependence_by_horizon_ci.png
└── run_manifest.json
```

The run manifest records experiment parameters and data information so that a result can be traced back to its configuration.

---

## 14. Project structure

```text
saifa_tiramisu/
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── src/
│   ├── data.py
│   ├── modwt.py
│   ├── marginals.py
│   ├── copulas.py
│   ├── cross_scale.py
│   ├── models.py
│   ├── tail_dependence.py
│   ├── robustness.py
│   ├── risk.py
│   ├── backtests.py
│   ├── backtest.py
│   ├── report.py
│   └── utils.py
├── tests/
├── data/
└── results/
```

### Module map

| File | Purpose |
|---|---|
| `data.py` | Yahoo Finance loading and cached return preparation |
| `modwt.py` | normalized SWT/MODWT-equivalent decomposition |
| `marginals.py` | AR-GARCH/EVT marginals, PIT and inverse PIT |
| `copulas.py` | within-scale R-vine / Gaussian fallback |
| `cross_scale.py` | M3 cross-scale Gaussian stress-rank coupling |
| `models.py` | M0–M3 model implementations |
| `tail_dependence.py` | empirical tail dependence, bootstrap, FDR and stress analysis |
| `backtest.py` | rolling out-of-sample model loop |
| `backtests.py` | coverage, independence and FZ0 scoring |
| `report.py` | result tables and figures |

---

## 15. Practical recommendation

For the tested portfolio and evaluation period:

**Tomorrow's action for a risk manager:**

> Keep the historical-simulation VaR/ES process as the operational baseline. Use the multiscale model to identify potential changes in dependence and diversification, but do not use its current forecasts to reduce risk limits.

Before promoting M2 or M3 to an operational role, regenerate the complete out-of-sample experiment after the implementation fixes, verify the actual copula backend used, update conditional states between refits, and require the richer models to demonstrate calibrated performance on unseen data.

---

## 16. AI disclosure

AI tools were used for learning, report drafting, code review, debugging and evidence checks. AI-assisted review identified the FZ0 scoring-sign issue and helped develop regression checks.

The team remains responsible for the final code, assumptions, calculations, interpretation and any changes made to the submitted repository, and must be able to explain and modify the work during judging.

---

## 17. Final takeaway

This project does **not** assume that complexity wins.

It asks a more useful question:

> **Does the extra information about tails and timescales improve real-world risk measurement?**

Our current evidence says:

- tail dependence differs across selected wavelet scales;
- cross-scale coupling changes the forecast distribution;
- the current M2/M3 implementation is not yet calibrated well enough for operational 99% VaR;
- the simple benchmark remains the safest operational choice for this experiment.

That separation between **detecting a phenomenon** and **proving that a model can exploit it reliably** is the central lesson of the study.
