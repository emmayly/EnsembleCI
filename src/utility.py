from MultilabelPredictor import *
import pandas as pd
import numpy as np
import tensorflow as tf

def addDateTimeFeatures(dataset, dateTime, startCol):
    dates = []
    hourList = []
    dayList=[]
    yearList = []
    hourSin, hourCos = [], []
    monthList = []
    monthSin, monthCos = [], []
    daySin, dayCos = [], []
    weekendList = []
    columns = dataset.columns
    secInDay = 24 * 60 * 60 # Seconds in day 
    secInYear = (365.25) * secInDay # Seconds in year 

    day = pd.to_datetime(dateTime[0])
    isWeekend = 0
    zero = 0
    one = 0
    for i in range(0, len(dateTime)):
        day = pd.to_datetime(dateTime[i])
        dates.append(day)
        hourList.append(day.hour)
        hourSin.append(np.sin(day.hour * (2 * np.pi / 24)))
        hourCos.append(np.cos(day.hour * (2 * np.pi / 24)))
        daySin.append(np.sin(day.day * (2 * np.pi / 31)))
        dayCos.append(np.cos(day.day * (2 * np.pi / 31)))
        dayList.append(day.day)
        monthList.append(day.month)
        yearList.append(day.year)
        monthSin.append(np.sin(day.timestamp() * (2 * np.pi / secInYear)))
        monthCos.append(np.cos(day.timestamp() * (2 * np.pi / secInYear)))
        if (day.weekday() < 5):
            isWeekend = 0
            zero +=1
        else:
            isWeekend = 1
            one +=1
        weekendList.append(day.weekday())        
    loc = startCol+1

    # hour of day feature
    dataset.insert(loc=loc, column="hour_sin", value=hourSin)
    dataset.insert(loc=loc+1, column="hour_cos", value=hourCos)
    # month of year feature
    dataset.insert(loc=loc+2, column="month_sin", value=monthSin)
    dataset.insert(loc=loc+3, column="month_cos", value=monthCos)
    # is weekend feature
    dataset.insert(loc=loc+4, column="weekend", value=weekendList)

    print(dataset.columns)
    return dataset


def initialize(inFileName, weatherInFileName, startCol):
    dataset = pd.read_csv(inFileName, header=0, infer_datetime_format=True, 
                            parse_dates=['UTC time'], index_col=['UTC time'])    
    dataset = dataset
    dateTime = dataset.index.values
    for i in range(startCol, len(dataset.columns.values)):
        col = dataset.columns.values[i]
        dataset[col] = dataset[col].astype(np.float64)

    forecastDataset = pd.read_csv(weatherInFileName, header=0, infer_datetime_format=True, 
                            parse_dates=['datetime'], index_col=['datetime']) # new data files in data
    forecastDataset = forecastDataset
    modifiedDataset = addDateTimeFeatures(forecastDataset, dateTime, startCol)
    forecastDataset = modifiedDataset
    # dataset.insert(column="datetime", value=pd.to_datetime(dateTime), loc=0)
    # forecastDataset.insert(column="datetime", value=pd.to_datetime(dateTime), loc=0)
    return dataset,forecastDataset, dateTime

def DayByDayPrediction(model, first_day_model, test_df, first_day_test_df, targets):
    my_test_df = test_df.iloc[::24]
    my_first_day_test_df = first_day_test_df.iloc[::24]

    actual_datas_96 = []
    for i in range(len(my_test_df) - 3):
        window_data = my_test_df.iloc[i:i+4][targets]
        actual_datas_96.extend(window_data.to_numpy().flatten())

    my_test_df_topredict = my_test_df.drop(columns=targets)
    first_day_my_test_df_topredict = my_first_day_test_df.drop(columns=targets)
    assert len(my_test_df_topredict) == len(first_day_my_test_df_topredict)

    predicted_datas_frames = []
    for iter in range(4):
        day_x_prediction = ""
        if iter == 0:
            day_x_prediction = first_day_model.predict(first_day_my_test_df_topredict[iter:iter+len(my_test_df_topredict)-3])
        else:
            day_x_prediction = model.predict(my_test_df_topredict[iter:iter+len(my_test_df_topredict)-3])
        predicted_datas_frames.append(day_x_prediction)
        past_columns = [f'carbon_intensity_t-{i}' for i in range(24, 0, -1)]  # ['t-24', 't-23', ..., 't-1']
        predicted_columns = [f'carbon_intensity_t+{i}' for i in range(24)]         # ['t+0', 't+1', ..., 't+23']

        # Replace real past value with predicted value
        if iter == 3:
            break
        j = 0
        for i in range(iter+1, iter+len(my_test_df_topredict)-2):
            my_test_df_topredict.loc[my_test_df_topredict.index[i], past_columns] = day_x_prediction.iloc[j].loc[predicted_columns].values
            j += 1
    return predicted_datas_frames, actual_datas_96


def getScore(acutalData, predictedData):
    PREDICTION_WINDOW_HOURS = 96
    mape = tf.keras.losses.MeanAbsolutePercentageError()
    rows, cols = len(acutalData)//PREDICTION_WINDOW_HOURS, PREDICTION_WINDOW_HOURS//24
    dailyMapeScore = np.zeros((rows, cols))
    for i in range(0, len(acutalData), PREDICTION_WINDOW_HOURS):
        for j in range(0, PREDICTION_WINDOW_HOURS, 24):
            mapeTensor =  mape(np.array(acutalData[i+j:i+j+24]), np.array(predictedData[i+j:i+j+24]))
            mapeScore = mapeTensor.numpy()
            dailyMapeScore[i//PREDICTION_WINDOW_HOURS][j//24] = mapeScore


    mapeTensor =  mape(np.array(acutalData), np.array(predictedData))
    mapeScore = mapeTensor.numpy()

    return mapeScore, np.mean(dailyMapeScore, axis=0)