import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import lightgbm as lgb

def create_multi_step_targets(df, target_col='log_return', horizon=6):
    for i in range(1, horizon + 1):
        df[f'target_t+{i}'] = df[target_col].shift(-i)
    df = df.dropna().reset_index(drop=True)
    return df

def select_features_rfe(df, target_col='log_return', n_features=10):
    df = df.dropna()
    df['target'] = df[target_col].shift(-1)
    df = df.dropna()
    X = df.drop(columns=['target', 'timestamp'])
    y = df['target']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

    lgb_estimator = lgb.LGBMRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rfe_selector = RFE(estimator=lgb_estimator, n_features_to_select=n_features, step=5)
    rfe_selector.fit(X_train, y_train)
    selected_features = X.columns[rfe_selector.support_].tolist()

    if 'close' not in selected_features:
        selected_features.append('close')

    return selected_features

def inverse_transform(scaler, data_scaled, feature_index=0):
    mean = scaler.mean_[feature_index]
    scale = scaler.scale_[feature_index]
    return data_scaled * scale + mean