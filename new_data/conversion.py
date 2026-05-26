import argparse
import json
import os
import pandas as pd


def process_csv(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)

    # 1. Drop 'zone' and 'total_production' columns
    columns_to_drop = ["zone", "total_production"]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    # 2. Strip the timezone offset (-07:00, -08:00, etc.) from the timestamp
    if "timestamp" in df.columns:
        print("Removing timezone offsets from timestamp...")
        # Converts "2023-05-25T00:05:00-07:00" -> "2023-05-25T00:05:00"
        # We split by the '-' character, but since the date has hyphens,
        # we only split from the right side where the timezone offset lives.
        df["timestamp"] = df["timestamp"].astype(str).str.rsplit("-", n=1).str[0]

    # 3. Check if the production_mix column exists and expand it
    if "production_mix" in df.columns:
        print("Expanding 'production_mix' column...")

        # Safely parse the JSON string into a Python dictionary
        df["production_mix"] = df["production_mix"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )

        # Flatten the dictionary into distinct columns
        mix_expanded = pd.json_normalize(df["production_mix"])

        # Combine everything back together, dropping the old json column
        df = pd.concat([df.drop(columns=["production_mix"]), mix_expanded], axis=1)
    else:
        print("Warning: 'production_mix' column not found in the CSV.")

    # 4. Save the result
    output_file = f"processed_{os.path.basename(file_path)}"
    df.to_csv(output_file, index=False)
    print(f"Success! Saved processed data to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean CSV, remove timezone offsets, and expand JSON production mix data."
    )
    parser.add_argument(
        "csv_file", type=str, help="The path to the CSV file you want to process"
    )

    args = parser.parse_args()
    process_csv(args.csv_file)