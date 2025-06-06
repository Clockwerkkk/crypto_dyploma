import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import lightgbm as lgb
import ta

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

def add_technical_indicators(df):
    # Скользящие средние
    df['SMA_7'] = ta.trend.sma_indicator(df['close'], window=7)
    df['SMA_21'] = ta.trend.sma_indicator(df['close'], window=21)
    df['EMA_7'] = ta.trend.ema_indicator(df['close'], window=7)
    df['EMA_21'] = ta.trend.ema_indicator(df['close'], window=21)

    # MACD
    df['MACD'] = ta.trend.macd(df['close'])
    df['MACD_signal'] = ta.trend.macd_signal(df['close'])
    df['MACD_diff'] = ta.trend.macd_diff(df['close'])

    # RSI
    df['RSI_14'] = ta.momentum.rsi(df['close'], window=14)

    # Bollinger Bands
    bb_indicator = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BB_upper'] = bb_indicator.bollinger_hband()
    df['BB_middle'] = bb_indicator.bollinger_mavg()
    df['BB_lower'] = bb_indicator.bollinger_lband()

    # ATR (волатильность)
    df['ATR_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    # Стохастик
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['STOCH_slowk'] = stoch.stoch()
    df['STOCH_slowd'] = stoch.stoch_signal()

    # OBV (объем)
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])

    # Добавляем CCI (Commodity Channel Index)
    # Рассчитаем Typical Price
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
    # SMA от Typical Price
    n = 20
    df['TP_SMA'] = df['TP'].rolling(window=n).mean()
    # Среднее абсолютное отклонение (MAD)
    df['TP_MAD'] = df['TP'].rolling(window=n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    # CCI
    df['CCI_20'] = (df['TP'] - df['TP_SMA']) / (0.015 * df['TP_MAD'])

    # Производные признаки
    df['return_1h'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['range'] = df['high'] - df['low']
    df['close_open_diff'] = df['close'] - df['open']

    # Временные признаки
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek

    return df