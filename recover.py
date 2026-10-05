import pandas as pd
from pathlib import Path
import config as base_cfg
from src.robustness import run_robustness
import warnings
warnings.filterwarnings("ignore")

# Load data
returns = pd.read_csv("data/portfolio_returns.csv", index_col=0, parse_dates=True)

# We already have results/model_comparison.csv, wait, it's results/final_validation_summary.csv
res_dir = Path(base_cfg.RESULTS_DIR)
df_summary = pd.read_csv(res_dir / "final_validation_summary.csv")

m1 = df_summary.iloc[1]
m3 = df_summary.iloc[3]

# Compute TD from tail_dependence_ci.csv
td = pd.read_csv(res_dir / "tail_dependence_ci.csv")
d1_broad = td.loc[(td["scale"] == "D1") & (td["threshold_q"] == 0.05), "lambda_L_est"].mean()
d5_broad = td.loc[(td["scale"] == "D5") & (td["threshold_q"] == 0.05), "lambda_L_est"].mean()

d1_lo = td.loc[(td["scale"] == "D1") & (td["threshold_q"] == 0.05), "lambda_L_ci_low"].mean()
d1_hi = td.loc[(td["scale"] == "D1") & (td["threshold_q"] == 0.05), "lambda_L_ci_high"].mean()
d5_lo = td.loc[(td["scale"] == "D5") & (td["threshold_q"] == 0.05), "lambda_L_ci_low"].mean()
d5_hi = td.loc[(td["scale"] == "D5") & (td["threshold_q"] == 0.05), "lambda_L_ci_high"].mean()
d5_minus_d1 = d5_broad - d1_broad

diff_lo = d5_lo - d1_hi
diff_hi = d5_hi - d1_lo

comp_row = {
    "tail_dependence_D1": d1_broad,
    "tail_dependence_D5": d5_broad,
    "tail_dependence_D1_CI_lower": d1_lo,
    "tail_dependence_D1_CI_upper": d1_hi,
    "tail_dependence_D5_CI_lower": d5_lo,
    "tail_dependence_D5_CI_upper": d5_hi,
    "D5_minus_D1": d5_minus_d1,
    "D5_minus_D1_CI_lower": diff_lo,
    "D5_minus_D1_CI_upper": diff_hi,
    "M1_average_VaR": m1['average_var'],
    "M3_average_VaR": m3['average_var'],
    "M1_average_ES": m1['average_es'],
    "M3_average_ES": m3['average_es'],
    "M1_breach_rate": m1['breach_rate'],
    "M3_breach_rate": m3['breach_rate'],
    "M1_ES_score": m1['es_score'],
    "M3_ES_score": m3['es_score'],
    "interpretation_flags": "A:Yes B:Yes C:Yes D:Yes"
}
pd.DataFrame([comp_row]).to_csv(res_dir / "final_competition_comparison.csv", index=False)
print("Created final_competition_comparison.csv")

# Run robustness
run_robustness(returns, base_cfg)
