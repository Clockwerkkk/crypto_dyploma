import pandas as pd
import os
# from data.load_from_binance import fetch_full_history
from model.utils import create_multi_step_targets
from model.utils import select_features_rfe
from model.utils import add_technical_indicators
from model.dataset import MultiStepDataset
from model.model import HybridPerformerBiLSTM
from model.train import train_model
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np


DATA_PATH = "service/app/data"
coin_files = [
    "BTC_USDT.csv",
    "ETH_USDT.csv",
    "SOL_USDT.csv",
    "XRP_USDT.csv",
    "TRX_USDT.csv"
]
horizon = 1
seq_len = 120
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Модель и признаки пока не знаем
model = None
feature_cols = None

for idx, filename in enumerate(coin_files):
    print(f"\n🔁 Этап {idx + 1}: обучение на {filename}")

    # 1. Загрузка и подготовка данных
    file_path = os.path.join(DATA_PATH, filename)
    df = pd.read_csv(file_path, parse_dates=["timestamp"])

    df = add_technical_indicators(df)
    df = create_multi_step_targets(df, target_col="log_return", horizon=horizon)
    df = df.dropna().reset_index(drop=True)

    if feature_cols is None:
        feature_cols = select_features_rfe(df, "log_return", n_features=10)
        if "close" not in feature_cols:
            feature_cols.append("close")

    target_cols = [f"target_t+{i}" for i in range(1, horizon + 1)]
    dataset = MultiStepDataset(df, feature_cols, target_cols, seq_len=seq_len)

    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.15)
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 2. Инициализация модели только на первом шаге
    if model is None:
        model = HybridPerformerBiLSTM(input_dim=len(feature_cols), horizon=horizon).to(device)
    else:
        model.to(device)

    # 3. Обучение (дообучение)
    train_model(model, train_loader, val_loader, epochs=7, lr=1e-4, device=device)

# 5. Финальное сохранение
torch.save(model.state_dict(), "models/best_hybrid_model.pth")
print("\n✅ Финальная модель сохранена.")
