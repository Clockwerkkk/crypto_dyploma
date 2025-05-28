import logging
import requests
import pandas as pd
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils import executor

# === Конфигурация ===
API_TOKEN = "8159072368:AAHN6sckoTYLXoh74yDbpgdUR-Pw2dFSwz0"
API_URL = "http://localhost:8000/predict"  # Адрес MCP API

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
logging.basicConfig(level=logging.INFO)

# === Поддерживаемые монеты ===
SUPPORTED_COINS = ["BTC", "ETH", "SOL", "XRP", "TRX"]
pending_coins = {}  # coin_name -> user_id

# === Команда /start ===
@dp.message_handler(commands=["start"])
async def start(msg: types.Message):
    await msg.answer("Привет! Используй команду /predict для прогноза или /addcoin <COIN> для добавления новой монеты.")

# === Команда /predict ===
@dp.message_handler(commands=["predict"])
async def handle_predict(msg: types.Message):
    keyboard = InlineKeyboardMarkup(row_width=2)
    buttons = [InlineKeyboardButton(text=coin, callback_data=f"predict:{coin}") for coin in SUPPORTED_COINS]
    keyboard.add(*buttons)
    await msg.answer("Выбери токен для прогноза:", reply_markup=keyboard)

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
    prices = resp.json()["predicted_prices"]
    text = f"Прогноз по {coin}:\n" + "\n".join([f"t+{i+1}: {p:.2f} USDT" for i, p in enumerate(prices)])
    await bot.send_message(callback_query.from_user.id, text)
    await bot.answer_callback_query(callback_query.id)

# === Команда /addcoin ===
@dp.message_handler(commands=["addcoin"])
async def add_coin(msg: types.Message):
    parts = msg.text.strip().split()
    if len(parts) != 2:
        await msg.reply("Формат: /addcoin <COIN>")
        return
    coin = parts[1].upper()
    pending_coins[coin] = msg.from_user.id
    await msg.reply(f"Валюта {coin} добавлена! После её обработки она станет доступна для выбора.")


# === Запуск бота ===
if __name__ == "__main__":
    executor.start_polling(dp)

