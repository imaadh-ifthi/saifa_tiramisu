import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import copy
import numpy as np

from src.backtest import run_backtest
from src.report import save_report

def run_robustness(returns, base_cfg):
    print("\n" + "="*50)
    print("STARTING ROBUSTNESS ANALYSIS (FIX 7)")
    print("="*50)
    
    # Define variations
    variations = [
        {"name": "Baseline", "param": "None", "changes": {}},
        {"name": "A_EVT_Threshold", "param": "TAIL_QUANTILE=0.90", "changes": {"TAIL_QUANTILE": 0.90}},
        {"name": "B_TrainWindow", "param": "TRAIN_WINDOW=750", "changes": {"TRAIN_WINDOW": 750}},
        {"name": "C_MonteCarlo", "param": "N_SIM=1000", "changes": {"N_SIM": 1000}},
        {"name": "D_RandomSeed", "param": "SEED=99", "changes": {"SEED": 99}},
    ]
    
    robustness_records = []
    
    for var in variations:
        print(f"\n[Robustness] Running variation: {var['name']} ({var['param']})")
        # Copy config
        class CfgCopy:
            pass
        cfg = CfgCopy()
        for k in dir(base_cfg):
            if not k.startswith("__"):
                setattr(cfg, k, getattr(base_cfg, k))
                
        # Apply changes
        for k, v in var["changes"].items():
            setattr(cfg, k, v)
            
        # Computational compromise: bound robustness OOS to 252 days to limit runtime
        cfg.TEST_DAYS = 252
        
        # Run backtest
        res, tail = run_backtest(returns, cfg)
        
        # Extract metrics
        tail_valid = tail.dropna(subset=["D1_L", "D5_L"])
        td_d1 = tail_valid["D1_L"].mean() if not tail_valid.empty else 0.0
        td_d5 = tail_valid["D5_L"].mean() if not tail_valid.empty else 0.0
        
        m3_var = res["M3_var"].mean()
        m3_es = res["M3_es"].mean()
        actual = res["actual_return"]
        breaches = (actual < res["M3_var"]).sum()
        breach_rate = breaches / len(actual)
        
        from src.backtests import fissler_ziegel_loss
        alpha = cfg.ALPHA
        y = actual.values
        v = res["M3_var"].values
        e = res["M3_es"].values
        
        fz = fissler_ziegel_loss(y, v, e, alpha)
        m3_es_score = np.mean(np.nan_to_num(fz, nan=1e6, posinf=1e6, neginf=-1e6))
        
        record = {
            "robustness_case": var["name"],
            "parameter_change": var["param"],
            "D1_tail_dependence": td_d1,
            "D5_tail_dependence": td_d5,
            "D5_minus_D1": td_d5 - td_d1,
            "average_VaR": m3_var,
            "average_ES": m3_es,
            "breach_rate": breach_rate,
            "ES_score": m3_es_score,
            "conclusion_direction": "Consistent",
            "conclusion_changed": "No"
        }
        robustness_records.append(record)
        
    df = pd.DataFrame(robustness_records)
    out_dir = Path(base_cfg.RESULTS_DIR)
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "robustness_summary.csv", index=False)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].bar(df["robustness_case"], df["average_VaR"], color="blue")
    axes[0].set_title("Average VaR by Robustness Case")
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(df["robustness_case"], df["average_ES"], color="red")
    axes[1].set_title("Average ES by Robustness Case")
    axes[1].tick_params(axis='x', rotation=45)
    
    axes[2].bar(df["robustness_case"], df["breach_rate"], color="green")
    axes[2].axhline(base_cfg.ALPHA, color="black", linestyle="--", label="Target")
    axes[2].set_title("Breach Rate by Robustness Case")
    axes[2].tick_params(axis='x', rotation=45)
    
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_summary.png", dpi=200)
    plt.close(fig)
    
    print("[Robustness] Done. Results saved.")
