#!/usr/bin/env python3
"""Plot carbon intensity patterns for CISO.csv.

The script reads a timestamped CSV with a `carbon_intensity` column and produces
two views:

1. The full 15-minute time series with a 24-hour rolling mean.
2. The average carbon intensity by hour of day to highlight the daily pattern.

By default it reads `../new_data/CISO.csv` relative to this file and writes a
PNG next to the input file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and validate the CISO carbon intensity data."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    required_columns = {"timestamp", "carbon_intensity"}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise SystemExit(f"Missing required column(s): {missing_text}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[["timestamp", "carbon_intensity"]].dropna()
    return df


def build_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Create the time-series and daily pattern plots."""
    series = df.set_index("timestamp")["carbon_intensity"]

    rolling_window = 96  # 24 hours of 15-minute samples
    rolling_mean = series.rolling(window=rolling_window, min_periods=1).mean()

    hourly_pattern = df.assign(hour=df["timestamp"].dt.hour).groupby("hour")["carbon_intensity"].mean()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

    ax_top.plot(series.index, series.values, color="#5b6dee", linewidth=0.7, alpha=0.35, label="15-minute data")
    ax_top.plot(rolling_mean.index, rolling_mean.values, color="#d1495b", linewidth=2.0, label="24-hour rolling mean")
    ax_top.set_title("CISO Carbon Intensity Over Time")
    ax_top.set_ylabel("Carbon intensity")
    ax_top.legend(loc="upper right")

    ax_bottom.bar(hourly_pattern.index, hourly_pattern.values, color="#2a9d8f", width=0.8)
    ax_bottom.set_title("Average Carbon Intensity by Hour of Day")
    ax_bottom.set_xlabel("Hour of day")
    ax_bottom.set_ylabel("Average carbon intensity")
    ax_bottom.set_xticks(range(24))

    fig.suptitle("CISO Carbon Intensity Pattern", fontsize=16, fontweight="bold")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot carbon intensity patterns for CISO.csv.")
    default_input = Path(__file__).resolve().parent.parent / "new_data" / "CISO.csv"
    parser.add_argument("csv_file", nargs="?", default=str(default_input), help="Path to CISO.csv")
    parser.add_argument(
        "--output",
        help="Path for the saved plot image. Defaults to the input file name with _pattern.png.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_file).expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"File not found: {csv_path}")

    output_path = Path(args.output).expanduser().resolve() if args.output else csv_path.with_name(f"{csv_path.stem}_pattern.png")

    df = load_data(csv_path)
    build_plot(df, output_path)

    print(f"Saved plot to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())