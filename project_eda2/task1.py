import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import math
import optuna
import lightgbm as lgb
import re
import plotly.express as px
import statsmodels.api as sm

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, RobustScaler

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
submission = pd.read_csv('data/sample_submission.csv')
original = pd.read_csv('data/machine_failure.csv')
df_1 = df.copy()
original_1 = original.copy()


def drop_missing_rows(df):
    if df.isnull().any().any():
        df = df.dropna(axis=0)
        dropped_rows = len(df) - len(df.dropna())
        print(f"Dropped {dropped_rows} rows.")
    else:
        print("No missing values.")

    return df

df = drop_missing_rows(df)
test = drop_missing_rows(test)


def count_duplicate_rows(df):
    duplicate_rows = df[df.duplicated()]
    if not duplicate_rows.empty:
        count = len(duplicate_rows)
        print(f"Number of : {count}.")
    else:
        print("O duplicates found.")

    return df

df = count_duplicate_rows(df)
test = count_duplicate_rows(test)


target_col = 'Machine failure'

num_cols = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

binary_cols = [
    'TWF',
    'HDF',
    'PWF',
    'OSF',
    'RNF'
]

cat_cols = 'Type'
cat_columns = ['Type']



df_1.drop(columns='id', inplace=True)
original_1.drop(columns='UDI', inplace=True)

test.drop(columns='RNF', inplace=True)
df_1.drop(columns='RNF', inplace=True)
original_1.drop(columns='RNF', inplace=True)


###FEAT ENGINEERING
data_all = pd.concat([df_1, original_1], ignore_index=True)
data_all['air_process_diff'] = abs(data_all['Air temperature [K]'] - data_all['Process temperature [K]'])
test['air_process_diff'] = abs(test['Air temperature [K]'] - test['Process temperature [K]'])

data_all['speed_power'] = data_all['Rotational speed [rpm]'] * (2 * np.pi / 60) / (data_all['Rotational speed [rpm]'] * (2 * np.pi / 60) * data_all['Torque [Nm]'])
test['speed_power'] = test['Rotational speed [rpm]'] * (2 * np.pi / 60) / (test['Rotational speed [rpm]'] * (2 * np.pi / 60) * test['Torque [Nm]'])

data_all['torque_power'] = data_all['Torque [Nm]'] / (data_all['Rotational speed [rpm]'] * (2 * np.pi / 60) * data_all['Torque [Nm]'])
test['torque_power'] = test['Torque [Nm]'] / (test['Rotational speed [rpm]'] * (2 * np.pi / 60) * test['Torque [Nm]'])

data_all["tool_process"]=data_all["Tool wear [min]"] * data_all["Process temperature [K]"]
test["tool_process"]=test["Tool wear [min]"] * test["Process temperature [K]"]

data_all["temp_ratio"] = data_all["Process temperature [K]"] / data_all["Air temperature [K]"]
test["temp_ratio"] = test["Process temperature [K]"] / test["Air temperature [K]"]

data_all["product_id_num"] = pd.to_numeric(data_all["Product ID"].str.slice(start=1))
test["product_id_num"] = pd.to_numeric(test["Product ID"].str.slice(start=1))


data_all.drop(columns='Product ID', inplace=True)
test.drop(columns='Product ID', inplace=True)

le = LabelEncoder()
for col in cat_columns:
    data_all['encoded_' + col] = le.fit_transform(data_all[col])
        
data_all.drop(cat_columns, axis=1, inplace=True)

for col in cat_columns:
    test['encoded_' + col] = le.transform(test[col])

test.drop(cat_columns, axis=1, inplace=True)

data_all.head()

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve

X = data_all.drop("Machine failure", axis=1)
y = data_all["Machine failure"]
X.columns = [re.sub(r"[^a-zA-Z0-9_]+", "_", col) for col in X.columns]
final_valid_predictions = []
oof_targets = []

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.85, random_state=42)

xgb = XGBClassifier(n_estimators=1500, 
                    objective='binary:logistic', 
                    eval_metric='auc', 
                    random_state=42, 
                    learning_rate = 0.08719877907815099, 
                    subsample = 0.9225106653522045, 
                    colsample_bytree = 0.212545425027345,
                    max_depth = 8
                    )

xgb.fit(X_train, y_train, early_stopping_rounds=100, eval_set=[(X_val, y_val)], verbose=100)
y_pred = xgb.predict_proba(X_val)[:, 1]

final_valid_predictions.extend(y_pred)
oof_targets.extend(y_val)

oof_preds = np.array(final_valid_predictions)
oof_targets = np.array(oof_targets)

feature_importance = xgb.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
features = X.columns

roc_score = roc_auc_score(oof_targets, oof_preds)
print(f"Final ROC AUC score: {roc_score}")
