import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from feature_engine.timeseries.forecasting import LagFeatures

pd.options.mode.chained_assignment = None
# handling cmd line vars

file_name = sys.argv[1]

forecast_start_date = sys.argv[2]
forecast_end_date = sys.argv[3]
train_start_date = pd.Timestamp(forecast_start_date)
train_end_date = pd.Timestamp(forecast_end_date)

variables = sys.argv[
    4
]  # Stopped using variables param as the loaded model already has the list of used features.

model = sys.argv[5]

# load in model
loaded_model = xgb.XGBRegressor()
loaded_model.load_model(model)
print("Loaded the model")

# read in dataset

df = pd.read_excel(file_name)
print("Successfully read the excel file")
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y %H:%M")
df["Date"] = df["Date"].apply(lambda x: x.round(freq="h"))


# split into df_2 & forecast

df_2 = df[(df["Date"] < forecast_start_date)]
df_forecast = df[
    (df["Date"] >= forecast_start_date) & (df["Date"] <= forecast_end_date)
]

# recursive forecasting

df_forecast["PriceSK"] = 0
df_forecast.reset_index(drop=True, inplace=True)
ref_date = df_forecast["Date"].iloc[0] - pd.Timedelta(days=60)
temp = df_2[df_2["Date"] >= ref_date]
combined_df = pd.concat([temp, df_forecast], ignore_index=True)
index_start = combined_df[combined_df["Date"] == df_forecast["Date"][0]].index[0]
num_of_forecast = df_forecast.shape[0]

pred_list = []
print("Initiating Forecasting")
for index in range(num_of_forecast):

    temp = combined_df.loc[:index_start].copy()

    # extract datetime features
    temp["hour"] = temp["Date"].dt.hour
    temp["week"] = temp["Date"].dt.dayofweek

    # create lag features
    lag_transformer = LagFeatures(
        variables=["PriceSK","GAS"], periods=[1, 2, 24, 24 * 7]
    )
    combined_df_lag = lag_transformer.fit_transform(temp)
    X = combined_df_lag[loaded_model.get_booster().feature_names].copy()
    X.dropna(inplace=True)

    # predict on test set
    pred = loaded_model.predict(np.reshape(X.iloc[-1, :].to_numpy(), (1, X.shape[1])))
    pred_list.append(pred[0])
    combined_df.loc[index_start, "PriceSK"] = pred[0]
    index_start += 1

output = pd.DataFrame()
output["Date"] = df_forecast["Date"]
output["forecast"] = pred_list
output.to_csv("forecast.csv", index=False)
print("Forecasting Completed & result is stored")