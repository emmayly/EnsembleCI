import argparse
import os

import pandas as pd


def convert_csv(input_path, output_path, group_size=3):
	if not os.path.exists(input_path):
		raise FileNotFoundError(f"Input file not found: {input_path}")

	if group_size <= 0:
		raise ValueError("group_size must be a positive integer")

	df = pd.read_csv(input_path)

	required_columns = ["timestamp", "carbon_intensity"]
	missing_required = [column for column in required_columns if column not in df.columns]
	if missing_required:
		raise ValueError(f"Missing required columns: {', '.join(missing_required)}")

	usable_length = len(df) - (len(df) % group_size)
	if usable_length == 0:
		raise ValueError("Input file does not contain enough rows to form one group")

	if usable_length != len(df):
		print(
			f"Warning: dropping the last {len(df) - usable_length} row(s) because they do not form a complete group of {group_size}."
		)

	df = df.iloc[:usable_length].copy()

	sum_columns = [column for column in df.columns if column not in {"timestamp", "carbon_intensity"}]

	if sum_columns:
		df[sum_columns] = df[sum_columns].apply(pd.to_numeric, errors="coerce").fillna(0)

	df["carbon_intensity"] = pd.to_numeric(df["carbon_intensity"], errors="coerce")

	group_ids = df.index // group_size
	aggregated = df.groupby(group_ids, sort=False).agg(
		{"timestamp": "first", "carbon_intensity": "mean", **{column: "sum" for column in sum_columns}}
	)

	column_order = ["timestamp", "carbon_intensity", *sum_columns]
	aggregated = aggregated[column_order]

	aggregated.to_csv(output_path, index=False)
	print(f"Saved merged CSV to: {output_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Merge every 3 rows of a CSV into one row."
	)
	parser.add_argument("input_csv", help="Path to the input CSV file")
	parser.add_argument("output_csv", help="Path to the output CSV file")
	parser.add_argument(
		"--group-size",
		type=int,
		default=3,
		help="Number of rows to merge into one output row (default: 3)",
	)

	args = parser.parse_args()
	convert_csv(args.input_csv, args.output_csv, args.group_size)
