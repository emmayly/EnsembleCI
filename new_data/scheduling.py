import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
INPUT_COEFF = 0.000025
OUTPUT_COEFF = 0.000450
CACHE_COEFF = 0.0000010
# ---------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-factor carbon intensity scheduling.")
    parser.add_argument("--ci_file", type=str, required=True)
    parser.add_argument("--workload_file", type=str, required=True)
    parser.add_argument("--prediction_len", type=float, default=0, help="Custom lookahead in minutes")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Parse Data
    try:
        df_ci = pd.read_csv(args.ci_file)
        df_wl = pd.read_csv(args.workload_file, sep=None, engine="python")
    except Exception as e:
        print(f"Error reading input files: {e}", file=sys.stderr)
        sys.exit(1)

    df_ci.columns = [c.strip().lower() for c in df_ci.columns]
    time_col = [c for c in df_ci.columns if "start" in c or "time" in c][0]
    df_ci[time_col] = pd.to_datetime(df_ci[time_col])
    df_ci = df_ci.sort_values(time_col).reset_index(drop=True)

    # Metrics: MAPE and Trend
    try:
        actuals = df_ci["actual_t+0"]
        preds = df_ci["predicted_t+0"]
        mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
        trend = np.corrcoef(actuals, preds)[0, 1]
        print(f"Forecast MAPE: {mape:.2f}%", file=sys.stderr)
        print(f"Forecast Pattern Trend (Correlation): {trend:.4f}", file=sys.stderr)
    except KeyError:
        print("Warning: Could not calculate MAPE/Trend (missing actual_t+0 or predicted_t+0)", file=sys.stderr)

    df_wl.columns = [c.strip().lower().replace("_", " ") for c in df_wl.columns]
    wl_start_col = [c for c in df_wl.columns if "start" in c][0]
    wl_end_col = [c for c in df_wl.columns if "end" in c][0]
    in_col = [c for c in df_wl.columns if "input token" in c][0]
    out_col = [c for c in df_wl.columns if "output token" in c][0]
    cache_col = [c for c in df_wl.columns if "cache" in c][0]
    
    df_wl[wl_start_col] = pd.to_datetime(df_wl[wl_start_col])
    df_wl[wl_end_col] = pd.to_datetime(df_wl[wl_end_col])

    interval_dur = df_ci[time_col].iloc[1] - df_ci[time_col].iloc[0] if len(df_ci) > 1 else pd.Timedelta(minutes=15)
    run_steps = max(1, int(np.ceil((df_wl[wl_end_col].max() - df_wl[wl_start_col].min()) / interval_dur)))
    
    energy_factor = (pd.to_numeric(df_wl[in_col]).sum() * INPUT_COEFF) + \
                    (pd.to_numeric(df_wl[out_col]).sum() * OUTPUT_COEFF) + \
                    (pd.to_numeric(df_wl[cache_col]).sum() * CACHE_COEFF)
    
    max_t = max([int(c.split("+")[1]) for c in df_ci.columns if c.startswith("predicted_t+")])

    horizons = {"6h": 360, "12h": 720, "24h": 1440}
    results = []
    savings_stats = {h: [] for h in horizons.keys()}

    for idx in range(len(df_ci)):
        row = df_ci.iloc[idx]
        base_avg = np.mean([row[f"actual_t+{k}"] for k in range(run_steps) if f"actual_t+{k}" in row])
        base_emission = (base_avg * energy_factor) / 1000.0
        
        row_res = {"Start Time": row[time_col], "Original Emission (g)": base_emission}
        
        for name, mins in horizons.items():
            lookahead = int(mins / (interval_dur.total_seconds() / 60))
            best_s = 0
            min_f = float("inf")
            for s in range(lookahead + 1):
                if s + run_steps > max_t + 1: break
                f_avg = np.mean([row[f"predicted_t+{s+k}"] for k in range(run_steps) if f"predicted_t+{s+k}" in row])
                if f_avg < min_f: min_f, best_s = f_avg, s
            
            shifted_avg = np.mean([row[f"actual_t+{best_s+k}"] for k in range(run_steps) if f"actual_t+{best_s+k}" in row])
            shifted_emission = (shifted_avg * energy_factor) / 1000.0
            
            row_res[f"Emission ({name}) (g)"] = shifted_emission
            
            # Calculate savings %
            if base_emission > 0:
                pct_saving = ((base_emission - shifted_emission) / base_emission) * 100
                savings_stats[name].append(pct_saving)
            
        results.append(row_res)

    # Print Average Savings to stderr
    for h, vals in savings_stats.items():
        print(f"Average Carbon Saving ({h}): {np.mean(vals):.2f}%", file=sys.stderr)

    # 2. Output CSV
    df_out = pd.DataFrame(results)
    df_out.to_csv(sys.stdout, index=False)

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df_out["Start Time"], df_out["Original Emission (g)"], label="Original", linestyle='--')
    for h in horizons.keys():
        plt.plot(df_out["Start Time"], df_out[f"Emission ({h}) (g)"], label=f"Deadline {h}")
    plt.xlabel("Start Time")
    plt.ylabel("Carbon Emissions (g)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("carbon_emissions_plot.png")

    plt.figure(figsize=(10, 6))
    for h in horizons.keys():
        savings = df_out["Original Emission (g)"] - df_out[f"Emission ({h}) (g)"]
        plt.plot(df_out["Start Time"], savings, label=f"Savings ({h})")
    plt.xlabel("Start Time")
    plt.ylabel("Carbon Savings (g)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("carbon_savings_plot.png")

if __name__ == "__main__":
    main()