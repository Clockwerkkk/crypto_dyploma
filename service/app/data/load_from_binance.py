import ccxt as ccxt
from datetime import datetime
import pandas as pd
import time
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="service//.env")

binance = ccxt.binance({
    'apiKey': os.getenv(BINANCE_API_KEY), # здесь нужно тянуть из env переменных
    'secret': os.getenv(BINANCE_SECRET_KEY), # здесь нужно тянуть из env переменных
    'enableRateLimit': True
})

# Пошаговая загрузка за 5 лет
def fetch_full_history(symbol, timeframe='1h', since=None, max_batches=1000, batch_limit=1500, sleep_sec=1):
    all_ohlcv = []
    since = since or binance.parse8601('2019-01-01T00:00:00Z')

    for i in range(max_batches):
        try:
            print(f"[{symbol}] Батч {i + 1}, since = {datetime.utcfromtimestamp(since / 1000)}")
            ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=batch_limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            time.sleep(sleep_sec)
        except Exception as e:
            print(f"Ошибка: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df
