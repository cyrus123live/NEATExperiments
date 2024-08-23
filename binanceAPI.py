import requests
from datetime import datetime

url = 'https://api.binance.com/api/v3/klines'
params = {
    'symbol': 'BTCUSDT',
    'interval': '1m',
    'limit': '1000'  # Number of data points
}

response = requests.get(url, params=params)
data = response.json()

minute_data = []
for candle in data:
    time = datetime.utcfromtimestamp(candle[0] / 1000.0)
    open_price = candle[1]
    high_price = candle[2]
    low_price = candle[3]
    close_price = candle[4]
    volume = candle[5]
    minute_data.append([time, open_price, high_price, low_price, close_price, volume])

    