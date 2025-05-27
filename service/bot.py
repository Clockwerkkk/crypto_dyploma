import logging
import requests
import pandas as pd
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor

API_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
API_URL = "http://localhost:8000/predict"

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
logging.basicConfig(level=logging.INFO)

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

if __name__ == "__main__":
    executor.start_polling(dp)