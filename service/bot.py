import logging
import requests
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import os
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from app.model.model import HybridPerformerBiLSTM
from app.model.utils import add_technical_indicators
API_TOKEN = "MY_SECRET_KEY"
API_URL = "http://localhost:8000/predict"

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
logging.basicConfig(level=logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"
feature_cols = [
    'EMA_7', 'SMA_7', 'MACD', 'RSI_14', 'BB_middle',
    'ATR_14', 'STOCH_slowk', 'OBV', 'CCI_20', 'return_1h', 'close'
]

@dp.message_handler(commands=["predict"])
async def predict_handler(message: types.Message):
    try:
        parts = message.text.strip().split()
        if len(parts) != 2:
            await message.reply("❗ Формат: /predict <COIN>")
            return

        coin = parts[1].upper()
        file_path = os.path.join("data", f"{coin}.csv")
        if not os.path.exists(file_path):
            await message.reply(f"❌ Файл {coin}.csv не найден в /data")
            return

        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        df = add_technical_indicators(df).dropna().reset_index(drop=True)

        if len(df) < 120:
            await message.reply(f"⚠️ Недостаточно данных по {coin} (нужно минимум 120 строк)")
            return

        df_last = df[-120:].copy()
        scaler = StandardScaler()
        X = scaler.fit_transform(df_last[feature_cols])
        x_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

        model = HybridPerformerBiLSTM(input_dim=len(feature_cols), horizon=1).to(device)
        model.load_state_dict(torch.load("models/best_hybrid_model.pth", map_location=device))
        model.eval()

        with torch.no_grad():
            pred = model(x_tensor).cpu().numpy().flatten()[0]

        close_idx = feature_cols.index("close")
        pred_price = pred * scaler.scale_[close_idx] + scaler.mean_[close_idx]

        await message.reply(f"📈 Прогноз цены {coin} на t+1: **{pred_price:.2f} USDT**")

    except Exception as e:
        await message.reply(f"⚠️ Ошибка при предсказании: {e}")


def get_latest_data(coin: str) -> list[dict]:
    df = pd.read_csv(f"data/{coin}_prepared.csv")
    return df.tail(150).to_dict(orient="records")


@dp.message_handler(commands=["start"])
async def start(msg: types.Message):
    await msg.answer("Привет! Используй /forecast BTC")


@dp.message_handler(lambda msg: msg.text.lower().startswith("/forecast"))
async def forecast(msg: types.Message):
    try:
        coin = msg.text.split()[1].upper()
        raw_data = get_latest_data(coin)
        payload = {"coin": coin, "raw_data": raw_data}
        resp = requests.post(API_URL, json=payload)
        if resp.status_code != 200:
            await msg.answer(f"Ошибка: {resp.text}")
            return
        prices = resp.json()["predicted_prices"]
        txt = f"Прогноз цены {coin}:\n" + "\n".join([f"t+{i+1}: {p:.2f} USDT" for i, p in enumerate(prices)])
        await msg.answer(txt)
    except Exception as e:
        await msg.answer(f"Ошибка: {str(e)}")


@dp.callback_query_handler(lambda c: c.data.startswith("predict:"))
async def process_callback_predict(callback_query: types.CallbackQuery):
    coin = callback_query.data.split(":")[1]
    raw_data = get_latest_data(coin)
    if not raw_data:
        await bot.answer_callback_query(callback_query.id, text="Данные не найдены")
        return

    payload = {"coin": coin, "raw_data": raw_data}
    resp = requests.post(API_URL, json=payload)
    if resp.status_code != 200:
        await bot.answer_callback_query(callback_query.id, text="Ошибка на сервере")
        return

    result = resp.json()
    price = result.get("predicted_price", "?")
    text = f"Прогноз по {coin} на следующий час:\n📈 {price:.2f} USDT"
    await bot.send_message(callback_query.from_user.id, text)
    await bot.answer_callback_query(callback_query.id)


if __name__ == "__main__":
    executor.start_polling(dp)

