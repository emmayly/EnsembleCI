from MultilabelPredictor import *
import os
import sys
import pandas as pd
import numpy as np

from utility import *

############################## main ###########################


def predictor_dir(base_path, label):
    return os.path.join(base_path, f"Predictor_{label}")


def has_saved_predictor(path):
    return os.path.isfile(os.path.join(path, "predictor.pkl"))


def train_single_predictor(multi_predictor, label, train_data, model_used_to_train, hyperparameter_tune_kwargs='auto'):
    predictor = multi_predictor.get_predictor(label)
    if model_used_to_train.upper() != 'AUTO':
        predictor.fit(train_data=train_data, hyperparameters={model_used_to_train: {}}, hyperparameter_tune_kwargs=hyperparameter_tune_kwargs)
    else:
        predictor.fit(train_data=train_data, hyperparameters={'GBM': {}, 'FASTAI': {}, 'CAT': {}}, hyperparameter_tune_kwargs=hyperparameter_tune_kwargs)
    multi_predictor.predictors[label] = predictor.path


def load_or_train_models(multi_predictor, labels, train_df, saved_model_name, model_used_to_train, load_only=False):
    loaded_count = 0
    trained_count = 0
    total_count = len(labels)
    for i, label in enumerate(labels):
        label_path = predictor_dir(saved_model_name, label)
        if has_saved_predictor(label_path):
            multi_predictor.predictors[label] = label_path
            print(f"Loaded existing predictor for label: {label}")
            loaded_count += 1
            print(f"Progress: {loaded_count + trained_count}/{total_count} predictors ready")
            continue

        if load_only:
            raise FileNotFoundError(f"Missing saved predictor for label {label}: {label_path}")

        if multi_predictor.consider_labels_correlation:
            labels_to_drop = [labels[j] for j in range(i + 1, len(labels))]
        else:
            labels_to_drop = [l for l in labels if l != label]

        train_subset = train_df.drop(labels_to_drop, axis=1)
        print(f"Fitting TabularPredictor for label: {label} ...")
        train_single_predictor(multi_predictor, label, train_subset, model_used_to_train)
        trained_count += 1
        print(f"Progress: {loaded_count + trained_count}/{total_count} predictors ready")

    print(f"Resume summary: loaded {loaded_count}, trained {trained_count}, total {total_count}")


if __name__ == "__main__":
    # process command line arguments (original style)
    arguments = sys.argv[1:]
    if len(arguments) < 4:
        print("Usage: python3 carbonForecast.py <region> <model> <d/l> <t/l>")
        sys.exit(1)

    region_name = arguments[0]
    emission_type = "direct" if arguments[1] == 'd' else "lifecycle"
    model_used_to_train = arguments[2]
    mode = arguments[3].lower()
    if mode not in {'t', 'l'}:
        print("Fourth argument must be 't' to load if available or train otherwise, or 'l' to load only")
        sys.exit(1)

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

    # For 15-minute data, this becomes 96 rows per 24 hours.
    lookback_steps = steps_per_day
    forecast_steps = steps_per_day
    window_stride = 1

    # Forecast horizon: next 24 hours
    H = forecast_steps
    # Use past 24 hours as input
    P = lookback_steps

    # target is carbon_intensity
    if 'carbon_intensity' not in df_raw.columns:
        print('carbon_intensity column missing in processed file'); sys.exit(1)

    # choose which features to include for past history (use all numeric columns)
    numeric_cols = list(df_raw.select_dtypes(include=[np.number]).columns)
    feature_cols = [c for c in numeric_cols if c != 'carbon_intensity']
    # For past-history flattening include carbon_intensity + feature_cols
    past_cols = ['carbon_intensity'] + feature_cols

    new_rows = []
    for i in range(P, len(df_raw) - H + 1, window_stride):
        prev_vals = df_raw.iloc[i-P:i][past_cols].values.flatten()
        future_times = df_raw.index[i:i+H]
        future_time_vals = list(future_times)
        future_vals = df_raw.iloc[i:i+H][['carbon_intensity']].values.flatten()
        row = list(prev_vals) + future_time_vals + list(future_vals)
        new_rows.append(row)

    past_column_names = [f'{col}_t-{P-j}' for j in range(P) for col in past_cols]
    future_time_columns = [f'forecast_datetime_t+{j}' for j in range(H)]
    predicted_columns = [f'carbon_intensity_t+{j}' for j in range(H)]
    final_columns = past_column_names + future_time_columns + predicted_columns
    new_df = pd.DataFrame(new_rows, columns=final_columns)

    for column in future_time_columns:
        new_df[column] = pd.to_datetime(new_df[column])

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
    multi_predictor = MultilabelPredictor(labels=targets, problem_types=problem_types, eval_metrics=eval_metrics, path=saved_model_name)
    if mode == 'l' and not os.path.isdir(saved_model_name):
        print(f'No existing model found at {saved_model_name}; load-only mode requested')
        sys.exit(1)

    load_only = mode == 'l'
    load_or_train_models(multi_predictor, targets, train_df, saved_model_name, model_used_to_train, load_only=load_only)

    # Prepare test features (drop targets)
    X_test = test_df.drop(columns=targets)
    y_test = test_df[targets].reset_index(drop=True)

    # Predict
    y_pred = multi_predictor.predict(X_test).reset_index(drop=True)

    # Compose output: one file with timestamps for forecast starts and actual/predicted arrays
    # Recover forecast start timestamps: for each sample i in new_df, forecast start corresponds to index i+P in original df_raw
    timestamps = []
    for i in range(P, len(df_raw) - H + 1, window_stride):
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
        
    