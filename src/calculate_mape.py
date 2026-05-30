#!/usr/bin/env python3
"""Calculate row-wise forecast quality metrics from a forecast CSV.

The input CSV must contain a `forecast_start` column and paired columns like
`actual_t+0`/`predicted_t+0`, `actual_t+1`/`predicted_t+1`, etc.

For each row, the script computes:

    MAPE = mean(abs((actual - predicted) / actual)) * 100

It also computes fluctuation-pattern metrics:

    trend match rate = fraction of steps where actual and predicted changes
    have the same sign

    delta correlation = Pearson correlation between actual and predicted
    step-to-step changes

Zero actual values are ignored for that row to avoid division-by-zero. If all
actual values in a row are zero, that row's MAPE is reported as NaN.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


PAIR_PATTERN = re.compile(r"^actual_t\+(\d+)$")


def discover_pairs(fieldnames: List[str]) -> List[Tuple[str, str]]:
    """Return sorted (actual_col, predicted_col) pairs."""
    pairs: List[Tuple[int, str, str]] = []
    field_set = set(fieldnames)

    for name in fieldnames:
        match = PAIR_PATTERN.match(name)
        if not match:
            continue
        step = int(match.group(1))
        predicted_name = f"predicted_t+{step}"
        if predicted_name in field_set:
            pairs.append((step, name, predicted_name))

    pairs.sort(key=lambda item: item[0])
    return [(actual, predicted) for _, actual, predicted in pairs]


def row_mape(row: Dict[str, str], pairs: List[Tuple[str, str]]) -> float:
    """Compute row-wise MAPE as a percentage, skipping zero actual values."""
    errors: List[float] = []

    for actual_col, predicted_col in pairs:
        actual = float(row[actual_col])
        predicted = float(row[predicted_col])
        if actual == 0.0:
            continue
        errors.append(abs((actual - predicted) / actual))

    if not errors:
        return math.nan

    return sum(errors) / len(errors) * 100.0


def row_trend_match_rate(row: Dict[str, str], pairs: List[Tuple[str, str]]) -> float:
    """Compute the fraction of step changes whose direction matches."""
    actual_values = [float(row[actual_col]) for actual_col, _ in pairs]
    predicted_values = [float(row[predicted_col]) for _, predicted_col in pairs]

    if len(actual_values) < 2:
        return math.nan

    matches = 0
    total = 0
    for index in range(1, len(actual_values)):
        actual_delta = actual_values[index] - actual_values[index - 1]
        predicted_delta = predicted_values[index] - predicted_values[index - 1]
        if math.copysign(1.0, actual_delta) == math.copysign(1.0, predicted_delta):
            matches += 1
        total += 1

    return matches / total if total else math.nan


def row_delta_correlation(row: Dict[str, str], pairs: List[Tuple[str, str]]) -> float:
    """Compute Pearson correlation between actual and predicted deltas."""
    actual_values = [float(row[actual_col]) for actual_col, _ in pairs]
    predicted_values = [float(row[predicted_col]) for _, predicted_col in pairs]

    if len(actual_values) < 2:
        return math.nan

    actual_deltas = [actual_values[index] - actual_values[index - 1] for index in range(1, len(actual_values))]
    predicted_deltas = [predicted_values[index] - predicted_values[index - 1] for index in range(1, len(predicted_values))]

    if len(actual_deltas) < 2:
        return math.nan

    actual_mean = sum(actual_deltas) / len(actual_deltas)
    predicted_mean = sum(predicted_deltas) / len(predicted_deltas)

    actual_centered = [value - actual_mean for value in actual_deltas]
    predicted_centered = [value - predicted_mean for value in predicted_deltas]

    numerator = sum(a * p for a, p in zip(actual_centered, predicted_centered))
    actual_variance = sum(a * a for a in actual_centered)
    predicted_variance = sum(p * p for p in predicted_centered)

    if actual_variance == 0.0 or predicted_variance == 0.0:
        return math.nan

    return numerator / math.sqrt(actual_variance * predicted_variance)


def plot_metrics(
    timestamps: List[str],
    mape_scores: List[float],
    trend_scores: List[float],
    corr_scores: List[float],
    output_path: Path,
) -> None:
    """Save a three-panel plot for the per-row metric patterns."""
    valid_positions = list(range(len(timestamps)))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)

    axes[0].plot(valid_positions, mape_scores, color="#d1495b", linewidth=1.5)
    axes[0].set_ylabel("MAPE (%)")
    axes[0].set_title("Row-wise forecast quality patterns")

    axes[1].plot(valid_positions, trend_scores, color="#2a9d8f", linewidth=1.5)
    axes[1].set_ylabel("Trend match rate")
    axes[1].set_ylim(0.0, 1.05)

    axes[2].plot(valid_positions, corr_scores, color="#5b6dee", linewidth=1.5)
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[2].set_ylabel("Delta correlation")
    axes[2].set_xlabel("Forecast start")

    if len(timestamps) <= 12:
        axes[2].set_xticks(valid_positions)
        axes[2].set_xticklabels(timestamps, rotation=45, ha="right")
    else:
        tick_count = min(8, len(timestamps))
        tick_positions = [round(index) for index in list(
            (i * (len(timestamps) - 1) / (tick_count - 1) if tick_count > 1 else 0)
            for i in range(tick_count)
        )]
        axes[2].set_xticks(tick_positions)
        axes[2].set_xticklabels([timestamps[index] for index in tick_positions], rotation=45, ha="right")

    fig.suptitle("Forecast Metric Patterns", fontsize=16, fontweight="bold")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute row-wise MAPE from a forecast CSV.")
    parser.add_argument(
        "csv_file",
        nargs="?",
        default=str(Path(__file__).resolve().parent.parent / "new_data" / "CISO_forecasted.csv"),
        help="Path to a forecast CSV produced by carbonForecast.py.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a plot of per-row MAPE, trend match rate, and delta correlation.",
    )
    parser.add_argument(
        "--plot-output",
        help="Path for the saved plot image. Defaults to the input file name with _metrics.png.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_file).expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"File not found: {csv_path}")

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("CSV file is missing a header row")

        pairs = discover_pairs(reader.fieldnames)
        if not pairs:
            raise SystemExit("No actual/predicted column pairs were found in the CSV")

        row_mape_scores: List[float] = []
        row_trend_scores: List[float] = []
        row_corr_scores: List[float] = []
        row_timestamps: List[str] = []
        for index, row in enumerate(reader, start=1):
            mape_score = row_mape(row, pairs)
            trend_score = row_trend_match_rate(row, pairs)
            corr_score = row_delta_correlation(row, pairs)

            row_mape_scores.append(mape_score)
            row_trend_scores.append(trend_score)
            row_corr_scores.append(corr_score)

            timestamp = row.get("forecast_start", str(index))
            row_timestamps.append(timestamp)
            if math.isnan(mape_score):
                print(f"Row {index} ({timestamp}): MAPE = NaN")
            else:
                print(f"Row {index} ({timestamp}): MAPE = {mape_score:.4f}%")

            if math.isnan(trend_score):
                trend_text = "NaN"
            else:
                trend_text = f"{trend_score:.4f}"

            if math.isnan(corr_score):
                corr_text = "NaN"
            else:
                corr_text = f"{corr_score:.4f}"

            print(f"  Trend match rate = {trend_text}")
            print(f"  Delta correlation = {corr_text}")

    valid_mape_scores = [score for score in row_mape_scores if not math.isnan(score)]
    valid_trend_scores = [score for score in row_trend_scores if not math.isnan(score)]
    valid_corr_scores = [score for score in row_corr_scores if not math.isnan(score)]

    if not valid_mape_scores:
        print("Average MAPE: NaN (no valid rows)")
        return 0

    average_mape = sum(valid_mape_scores) / len(valid_mape_scores)
    average_trend = sum(valid_trend_scores) / len(valid_trend_scores) if valid_trend_scores else math.nan
    average_corr = sum(valid_corr_scores) / len(valid_corr_scores) if valid_corr_scores else math.nan

    print(f"Average MAPE: {average_mape:.4f}%")
    if math.isnan(average_trend):
        print("Average trend match rate: NaN")
    else:
        print(f"Average trend match rate: {average_trend:.4f}")

    if math.isnan(average_corr):
        print("Average delta correlation: NaN")
    else:
        print(f"Average delta correlation: {average_corr:.4f}")

    if args.plot:
        output_path = (
            Path(args.plot_output).expanduser().resolve()
            if args.plot_output
            else csv_path.with_name(f"{csv_path.stem}_metrics.png")
        )
        plot_metrics(row_timestamps, row_mape_scores, row_trend_scores, row_corr_scores, output_path)
        print(f"Saved plot to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
