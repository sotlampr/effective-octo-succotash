import pickle

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_predict

import pandas as pd


with open("features.pkl", "rb") as fp:
    features = pickle.load(fp)
    features = features.apply(pd.to_numeric)

with open("target.pkl", "rb") as fp:
    target = pickle.load(fp)

df_roll = features.interpolate().copy()
df_roll["target"] = target

for col in [0, 1, 2, "target"]:
    for t in [12, 24, 48]:
        df_roll[f"{col}_rmean_{t}"] = df_roll[col].rolling(t).mean().shift(1)
        df_roll[f"{col}_rmin_{t}"] = df_roll[col].rolling(t).min().shift(1)
        df_roll[f"{col}_rmax_{t}"] = df_roll[col].rolling(t).max().shift(1)

feat_cols = df_roll.columns.difference(["target"])
df_roll = df_roll.dropna()

y_pred = cross_val_predict(LinearRegression(), df_roll[feat_cols],
                           df_roll.target, cv=10, n_jobs=-1)
print("Using rolling means, min and max")
print(f"Mean absolute error: {mean_absolute_error(df_roll.target, y_pred):.4f}")

df_ar = features.interpolate().copy()
df_ar["target"] = target

for i in range(12):
    df_ar[f"target_lag_{i}"] = np.nan

lag_cols = [i for (i, x) in enumerate(df_ar.columns.values)
            if isinstance(x, str) and "lag" in x]

for i in range(12, len(df_ar)):
    df_ar.iloc[i, lag_cols] = df_ar.target.iloc[i-12:i].values

feat_cols = df_ar.columns.difference(["target"])
df_ar = df_ar.dropna()

y_pred = cross_val_predict(LinearRegression(), df_ar[feat_cols], df_ar.target,
                           cv=10, n_jobs=-1)
print("Using previous 12 target values")
print(f"Mean absolute error: {mean_absolute_error(df_ar.target, y_pred):.4f}")

df_comb = pd.merge(df_roll, df_ar)
feat_cols = df_comb.columns.difference(["target"])

y_pred = cross_val_predict(LinearRegression(), df_comb[feat_cols],
                           df_comb.target, cv=10, n_jobs=-1)
print("Using both")
print(f"Mean absolute error: {mean_absolute_error(df_comb.target, y_pred):.4f}")
