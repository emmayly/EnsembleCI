from MultilabelPredictor import *
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from utility import *

############################## main ###########################


if __name__ == "__main__":
    # process command line arguments (original style)
    arguments = sys.argv[1:]
    if len(arguments) < 4:
        print("Usage: python3 carbonForecast.py <region> <d/l> <model> <l/t>")
        sys.exit(1)

    region_name = arguments[0]
    emission_type = "direct" if arguments[1] == 'd' else "lifecycle"
    model_used_to_train = arguments[2]
    isToLoad = True if arguments[3] == 'l' else False

    # new_data path (relative to this script)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    new_data_dir = os.path.join(script_dir, "..", "new_data")
    new_data_dir = os.path.normpath(new_data_dir)

    # prefer exact file like new_data/<region>.csv (e.g., CISO.csv, ISNE.csv)
    processed_file = None
    candidate = os.path.join(new_data_dir, f"{region_name}.csv")
    if os.path.isfile(candidate):
        processed_file = candidate
    else:
        # fallback: find any csv in new_data that contains the region name
        if os.path.isdir(new_data_dir):
            for fname in sorted(os.listdir(new_data_dir)):
                if fname.endswith('.csv') and region_name in fname:
                    processed_file = os.path.join(new_data_dir, fname)
                    break

    if processed_file is None:
        print(f'No data file found for region "{region_name}" in new_data; aborting')
        sys.exit(1)

    print(f"Using input file: {processed_file}")

    # Read processed data (no weather data expected)
    df_raw = pd.read_csv(processed_file, header=0, parse_dates=['timestamp'], infer_datetime_format=True)
    df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)
    df_raw = df_raw.set_index('timestamp')

    # infer sampling interval in minutes (mode of diffs)
    diffs = df_raw.index.to_series().diff().dropna().dt.total_seconds().div(60).round().astype(int)
    if len(diffs) == 0:
        print('Input file too small'); sys.exit(1)
    delta_min = int(diffs.mode().iloc[0])
    steps_per_day = int(24 * 60 // delta_min)
    print(f"Detected sampling interval: {delta_min} minutes -> {steps_per_day} steps per day")

    # Forecast horizon: 24 hours -> steps_per_day
    H = steps_per_day
    # Use past H rows (24 hours) as input
    P = steps_per_day

    # Columns available
    available_cols = list(df_raw.columns)
    # target is carbon_intensity
    if 'carbon_intensity' not in df_raw.columns:
        print('carbon_intensity column missing in processed file'); sys.exit(1)

    # choose which features to include for past history (use all numeric columns)
    feature_cols = [c for c in available_cols if c != 'carbon_intensity']
    # For past-history flattening include carbon_intensity + feature_cols
    past_cols = ['carbon_intensity'] + feature_cols

    new_rows = []
    for i in range(P, len(df_raw) - H):
        prev_vals = df_raw.iloc[i-P:i][past_cols].values.flatten()
        future_vals = df_raw.iloc[i:i+H][['carbon_intensity']].values.flatten()
        row = list(prev_vals) + list(future_vals)
        new_rows.append(row)

    past_column_names = [f'{col}_t-{P-j}' for j in range(P) for col in past_cols]
    predicted_columns = [f'carbon_intensity_t+{j}' for j in range(H)]
    final_columns = past_column_names + predicted_columns
    new_df = pd.DataFrame(new_rows, columns=final_columns)

    # Split: last 30 days as testing data
    test_size = H * 30
    if test_size >= len(new_df):
        print('Not enough data for 30 days test split'); sys.exit(1)

    train_df = new_df[:-test_size]
    test_df = new_df[-test_size:]

    targets = predicted_columns
    problem_types = ['regression' for _ in range(len(targets))]
    eval_metrics = ['mean_absolute_percentage_error' for _ in range(len(targets))]

    # build model path inside new_data for persistence (use model name + region)
    model_base = os.path.join(new_data_dir, 'models')
    os.makedirs(model_base, exist_ok=True)
    saved_model_name = os.path.join(model_base, f"{model_used_to_train}_{region_name}_{emission_type}")

    # Train or load depending on isToLoad
    if isToLoad and os.path.isdir(saved_model_name):
        multi_predictor = MultilabelPredictor.load(saved_model_name)
    else:
        multi_predictor = MultilabelPredictor(labels=targets, problem_types=problem_types, eval_metrics=eval_metrics, path=saved_model_name)
        # simple default training using specified model type when possible
        if model_used_to_train.upper() != 'AUTO':
            multi_predictor.fit(train_df, hyperparameters={model_used_to_train: {}}, hyperparameter_tune_kwargs='auto')
        else:
            multi_predictor.fit(train_df, hyperparameters={'GBM': {}}, hyperparameter_tune_kwargs='auto')

    # Prepare test features (drop targets)
    X_test = test_df.drop(columns=targets)
    y_test = test_df[targets].reset_index(drop=True)

    # Predict
    y_pred = multi_predictor.predict(X_test).reset_index(drop=True)

    # Compose output: one file with timestamps for forecast starts and actual/predicted arrays
    # Recover forecast start timestamps: for each sample i in new_df, forecast start corresponds to index i+P in original df_raw
    timestamps = []
    for i in range(P, len(df_raw) - H):
        timestamps.append(df_raw.index[i])
    timestamps = pd.Series(timestamps)
    test_timestamps = timestamps[-test_size:].reset_index(drop=True)

    out_df = pd.DataFrame()
    out_df['forecast_start'] = test_timestamps
    # expand actual and predicted columns
    for j in range(H):
        out_df[f'actual_t+{j}'] = y_test[f'carbon_intensity_t+{j}']
        out_df[f'predicted_t+{j}'] = y_pred[f'carbon_intensity_t+{j}']

    output_file = os.path.join(new_data_dir, os.path.basename(processed_file).replace('.csv', '') + '_forecasted.csv')
    out_df.to_csv(output_file, index=False)
    print(f'Forecasts saved to: {output_file}')
        
    