import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from feature_engine.timeseries.forecasting import LagFeatures
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import pyswarms as ps

pd.options.mode.chained_assignment = None

file_name = sys.argv[1]

train_start_date = sys.argv[2]
train_end_date = sys.argv[3]
train_start_date = pd.Timestamp(train_start_date)
train_end_date = pd.Timestamp(train_end_date)

variables = sys.argv[4]
variables = list(map(str.strip, variables.lstrip("[").rstrip("]").split(",")))
print(file_name)
print(train_start_date)
print(train_end_date)
print(variables)
# read in dataset
print("Reading Data")
df = pd.read_excel(file_name)
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y %H:%M")
df["Date"] = df["Date"].apply(lambda x: x.round(freq="h"))

# filter dates
df_train_test = df[(df["Date"] >= train_start_date) & (df["Date"] <= train_end_date)]
# extract datetime features
df_train_test["hour"] = df_train_test.loc[:, "Date"].dt.hour
df_train_test["week"] = df_train_test.loc[:, "Date"].dt.dayofweek
# split into train & test

n = int(np.round(len(df_train_test) / 100 * 95, 0))
df_test = df_train_test.iloc[n:, :]
df_train = df_train_test.iloc[:n, :]
print(
    f"Train DF Size: {df_train.shape[0]}\nTest DF Size {df_test.shape[0]}",
)

# create lag features
lag_transformer = LagFeatures(
    variables=["PriceSK"], periods=[1, 2, 24, 24 * 7]
)

# Add lag features in train data
df_train_lag = lag_transformer.fit_transform(df_train)
df_train_lag.dropna(inplace=True)

# create X & y
add_var = [
    "hour",
    "week",
    "PriceSK_lag_1",
    "PriceSK_lag_2",
    "PriceSK_lag_24",
    "PriceSK_lag_168",
]
variables.extend(add_var)
print("Selected Variables:", variables)
X_train = df_train_lag[variables]
y_train = df_train_lag["PriceSK"]

# Add lag features in test data
df_test_lag = lag_transformer.transform(df_test)
df_test_lag.dropna(inplace=True)

X_test = df_test_lag[variables]
y_test = df_test_lag["PriceSK"]


# Set up PSO optimizer
def optimize_xgb(params):
    n_particles = params.shape[0]
    rmse_list = []

    for i in range(n_particles):
        eta = params[i, 0]
        max_depth = int(params[i, 1])

        # Initialize XGBoost model with chosen parameters
        xgbr_ = xgb.XGBRegressor(
            colsample_bytree=0.7,
            random_state=42,
            early_stopping_rounds=10,
            eta=eta,
            max_depth=max_depth,
            tree_method="hist",
        )

        # Train the model with early stopping
        xgbr_.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Predict on the validation set (X_test, y_test)
        pred_list = xgbr_.predict(X_test)

        # Calculate RMSE for the validation set
        rmse = root_mean_squared_error(
            y_test,
            pred_list,
        )
        rmse_list.append(rmse)

    return np.array(rmse_list)


# Define the bounds for eta and max_depth
bounds = (np.array([0.01, 1]), np.array([1, 10]))
options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
optimizer = ps.single.GlobalBestPSO(
    n_particles=10, dimensions=2, options=options, bounds=bounds
)

# Perform optimization
cost, pos = optimizer.optimize(optimize_xgb, iters=80)

# Extract best parameters
best_eta, best_max_depth = pos
best_max_depth = int(round(best_max_depth))

# Print the best parameters found
print(f"Best parameters: eta={best_eta}, max_depth={best_max_depth}, RMSE={cost}")
print("PSO Completed Successfully")
# train best param model on entire dataset



# create lag features
lag_transformer = LagFeatures(
    variables=["PriceSK"], periods=[1, 2, 24, 24 * 7]
)
df_lag = lag_transformer.fit_transform(df_train_test)
df_lag.dropna(inplace=True)

# create X & y
X_train = df_lag[variables]
y_train = df_lag["PriceSK"]

# training
print("Training Hyperparameter Fine-Tuned model")
best_xgb = xgb.XGBRegressor(
    colsample_bytree=0.7,
    random_state=42,
    eta=best_eta,
    max_depth=best_max_depth,
    early_stopping_rounds=10,
    tree_method="hist",
)
best_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("Training Completed. Evaluation Started: ")

# Predict on the test set
y_pred = best_xgb.predict(X_test)

# Calculate RMSE, MAE, MAPE
rmse = root_mean_squared_error(
    y_test,
    y_pred,
)
mae = mean_absolute_error(y_test, y_pred)

print("root_mean_squared_error:", rmse)
print("mean_absolute_error:", mae)

print("Saving Model")
best_xgb.save_model("xgb_model.json")
print("Model saved as: xgb_model.json")

print("Imp")
xgb_fea_imp = pd.DataFrame(
    list(best_xgb.get_booster().get_fscore().items()), columns=["feature", "importance"]
)
xgb_fea_imp = xgb_fea_imp.sort_values("importance", ascending=False)
print(xgb_fea_imp)

print("Variables Used:", variables)
