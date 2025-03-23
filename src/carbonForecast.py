from MultilabelPredictor import *
import os.path
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from utility import *

############################## main ###########################


if __name__ == "__main__":
    # process command line arguments
    arguments = sys.argv[1:]
    if len(arguments) < 3:
        print("Usage: python3 carbonForecast.py <region> <d/l> <model> <l/t>")
        sys.exit(1) 
    
    data_folder = "../data/"
    model_folder = ""
    emission_file_path = "_emissions.csv"
    weather_file_path="_weather_forecast.csv"
    saved_model_name=""
    output_name="_predicted.csv"
    first_day_saved_model_name =""
    model_used_to_train= arguments[2]
    isToLoad = True if arguments[3] == 'l' else False
    cut2019 = False
    region_name = arguments[0]

    # Process the data path
    emission_type = "direct" if arguments[1] == 'd' else "lifecycle"
    weather_file_path = data_folder+region_name +"/"+region_name + weather_file_path
    emission_file_path = data_folder+region_name +"/"+region_name+"_"+emission_type + emission_file_path
    saved_model_name = model_folder + model_used_to_train + "_" + region_name + "_" + emission_type
    first_day_saved_model_name = model_folder + model_used_to_train + "_" + region_name + "_" + emission_type + "_first_day"
    output_name = data_folder + region_name + "/" + region_name + "_" + emission_type + output_name


    data, weather, dateTime = initialize(emission_file_path,weather_file_path, 2)
    if cut2019:
        data = data[8760:]
        weather = weather[8760:]
    data_copy = data.copy()

    columns_to_flatten = ['carbon_intensity']  # Columns for past 24 rows
    first_day_columns_to_flatten = ['carbon_intensity', 'biomass', 'coal', 'nat_gas', 'geothermal', 'hydro', 'nuclear', 'oil', 'solar', 'wind' , 'unknown', 'other']
    weather_columns = ['hour_sin','hour_cos', 'month_sin','month_cos','weekend', 'forecast_avg_wind_speed_wMean', 'forecast_avg_temperature_wMean', 'forecast_avg_dewpoint_wMean', 'forecast_avg_dswrf_wMean', 'forecast_avg_precipitation_wMean']
    new_rows = []

    # only flatten columns exist in the dataset
    columns_to_flatten = [col for col in columns_to_flatten if col in data.columns]
    first_day_columns_to_flatten = [col for col in first_day_columns_to_flatten if col in data.columns]
    print(first_day_columns_to_flatten)
    # Loop starting from the 25th row (index 24)
    for i in range(24, len(data)-24):
        previous_24_rows = data.iloc[i-24:i, data.columns.get_indexer(columns_to_flatten)].values.flatten()  # Shape will be (24 * 14,)
        future_24_rows = data.iloc[i:i+24, data.columns.get_indexer(['carbon_intensity'])].values.flatten()
        future_24_weather_rows = weather.iloc[i:i+24, weather.columns.get_indexer(weather_columns)].values.flatten()
        new_row = list(previous_24_rows) + list(future_24_rows) + list(future_24_weather_rows)
        new_rows.append(new_row)
    flattened_columns = [f'{col}_t-{24-j}' for j in range(24) for col in columns_to_flatten]  # Column names for past 24 rows
    weather_columns_names = [f'{col}_t+{j}' for j in range(24) for col in weather_columns]
    predicted_columns = [f'carbon_intensity_t+{j}' for j in range(24)]  # Column names for the current row
    final_columns = flattened_columns + predicted_columns + weather_columns_names  # Combine with the columns to keep
    print("columns", final_columns)
    new_df = pd.DataFrame(new_rows, columns=final_columns)

    # data preprocess first day (with past source production)
    new_rows = []
    for i in range(24, len(data_copy)-24):
        previous_24_rows = data_copy.iloc[i-24:i, data_copy.columns.get_indexer(first_day_columns_to_flatten)].values.flatten()  # Shape will be (24 * 14,)
        future_24_rows = data_copy.iloc[i:i+24, data_copy.columns.get_indexer(['carbon_intensity'])].values.flatten()
        future_24_weather_rows = weather.iloc[i:i+24, weather.columns.get_indexer(weather_columns)].values.flatten()
        new_row = list(previous_24_rows) + list(future_24_rows) + list(future_24_weather_rows)
        new_rows.append(new_row)
    first_day_flattened_columns = [f'{col}_t-{24-j}' for j in range(24) for col in first_day_columns_to_flatten]  # Column names for past 24 rows
    first_day_final_columns = first_day_flattened_columns + predicted_columns + weather_columns_names  # Combine with the columns to keep
    print("first day columns", first_day_final_columns)
    first_day_new_df = pd.DataFrame(new_rows, columns=first_day_final_columns)

    train_df = new_df[:-4344]
    first_day_train_df = first_day_new_df[:-4344]

    targets = predicted_columns 
    problem_types = ['regression' for _ in range(len(targets))]
    eval_metrics = ['mean_absolute_percentage_error' for _ in range(len(targets))]

    multi_predictor = ""
    first_day_multi_predictor = ""
    print(saved_model_name)
    if isToLoad and os.path.isdir(saved_model_name):
        multi_predictor = MultilabelPredictor.load(saved_model_name)
        first_day_multi_predictor =  MultilabelPredictor.load(first_day_saved_model_name)
    else:
        multi_predictor = MultilabelPredictor(labels=targets, problem_types=problem_types, eval_metrics=eval_metrics, path=saved_model_name)
        first_day_multi_predictor = MultilabelPredictor(labels=targets, problem_types=problem_types, eval_metrics=eval_metrics, path=first_day_saved_model_name)
        if model_used_to_train != 'AUTO' :
            multi_predictor.fit(train_df, hyperparameters={model_used_to_train:{}}, hyperparameter_tune_kwargs='auto')
            first_day_multi_predictor.fit(first_day_train_df, hyperparameters={model_used_to_train:{}}, hyperparameter_tune_kwargs='auto')
            # first_day_multi_predictor =  MultilabelPredictor.load(first_day_saved_model_name)

        else:
            # multi_predictor = MultilabelPredictor.load(saved_model_name)
            multi_predictor.fit(train_df, hyperparameters={'GBM': {}, 'FASTAI': {}, 'CAT': {}}, hyperparameter_tune_kwargs='auto')
            # first_day_multi_predictor =  MultilabelPredictor.load(first_day_saved_model_name)
            first_day_multi_predictor.fit(first_day_train_df, hyperparameters={'GBM': {}, 'FASTAI': {}, 'CAT': {}}, hyperparameter_tune_kwargs='auto')

    test_df = new_df[-4344:] # 181 days for testing
    first_day_test_df = first_day_new_df[-4344:] # 181 days for testing
    # 96h prediction
    predicted_datas_frames, actual_datas_96 = DayByDayPrediction(multi_predictor, first_day_multi_predictor, test_df, first_day_test_df, targets)

    predicted_datas_96 = []
    num_rows = len(predicted_datas_frames[0])
    for i in range(num_rows):
        row_concatenated = []
        for df in predicted_datas_frames:
            row_concatenated.extend(df.iloc[i].to_list())
        predicted_datas_96.extend(row_concatenated)
    df = pd.DataFrame({
        'actual': actual_datas_96,
        'predicted': predicted_datas_96,
    })

    # Save the DataFrame to a CSV file
    df.to_csv(output_name, index=False)

    print(getScore(actual_datas_96, predicted_datas_96))

    with open('output_results/fixed_new_design.txt', 'a') as file:
        file.write(f'{saved_model_name}\n')
        file.write(f'{getScore(actual_datas_96, predicted_datas_96)}\n')
        
    