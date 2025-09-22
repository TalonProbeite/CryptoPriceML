from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd
import requests


def get_raw_candle_csv(
    file_name: str = "raw_candle.csv",
    save_path: str = "data\\raw_data\\samples\\raw\\",
    symbol: str = "BTCUSDT",
    interval: str = "30",
    category: str = "spot",
    days_ago: int = 20
) -> None:
    """
    Fetch historical candle data from Bybit API and save to CSV file.

    Args:
        file_name: Name of the output CSV file. Defaults to "raw_candle.csv".
        save_path: Directory path where the CSV file will be saved. 
                  Defaults to "data\\raw_data\\samples\\".
        symbol: Trading symbol. Defaults to "BTCUSDT".
        interval: Candle interval in minutes. Defaults to "30".
        category: Market category. Defaults to "spot".
        days_ago: Number of days to look back from current time. Defaults to 7.

    Returns:
        None: Saves data to CSV file.

    Raises:
        requests.exceptions.RequestException: If API request fails.
        KeyError: If expected data is missing from API response.
        ValueError: If API returns error response.

    Examples:
        >>> get_raw_candle_csv("btc_data.csv", "data/", "BTCUSDT", "60", "spot", 30)
        >>> get_raw_candle_csv()  # Uses all default parameters
    """
    # Calculate time range
    end_time = datetime.now(timezone.utc)  - timedelta(days=140)
    start_time = end_time - timedelta(days=days_ago)

    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    # API request
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "start": start_ts,
        "end": end_ts,
        "limit": 1000            
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()  # Raise exception for bad status codes
        data = resp.json()
        
        # Check API response for errors
        if data.get("retCode") != 0:
            raise ValueError(f"API error: {data.get('retMsg', 'Unknown error')}")
        
        if "result" not in data or "list" not in data["result"]:
            raise KeyError("Invalid API response structure")
            
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"API request failed: {e}") from e

    # Process data
    klines = data["result"]["list"]
    klines_ohclv = [row[:6] for row in klines]

    df = pd.DataFrame(klines_ohclv, columns=[
        "timestamp", "open", "high", "low", "close", "volume"
    ])
    
    # Convert data types
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype('int64'), unit="ms")
    df = df.astype({
        "open": float, "high": float, "low": float,
        "close": float, "volume": float
    })
    
    # Ensure save path ends with separator
    if not save_path.endswith(("\\", "/")):
        save_path += "\\"
    
    # Save to CSV
    full_path = save_path + file_name
    df.to_csv(full_path, index=False)
    
    print(f"Data successfully saved to: {full_path}")
    print(f"Records fetched: {len(df)}")










import requests
import pandas as pd
from datetime import datetime, timezone
import os
import math


def get_raw_candles_csv(
    save_path: str = "data\\raw_data\\samples\\raw\\",
    symbol: str = "BTCUSDT",
    interval: str = "30",
    category: str = "spot",
    total_candles: int = 40000,
    candles_per_file: int = 1000
) -> None:
    """
    Fetch exactly `total_candles` closed candles from Bybit API starting from now (backwards).
    Save in chunks to CSV files named raw_candle_{n}.csv.

    Args:
        save_path: Directory path where CSV files will be saved.
        symbol: Trading symbol (default: BTCUSDT).
        interval: Candle interval in minutes (default: "30").
        category: Market category (default: spot).
        total_candles: Total number of candles to fetch (default: 40000).
        candles_per_file: Max candles per file (default: 1000).
    """
    url = "https://api.bybit.com/v5/market/kline"

    # Ensure folder exists
    os.makedirs(save_path, exist_ok=True)

    # End time = now (UTC). API берет только закрытые свечи
    end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    fetched = 0
    file_index = 1

    while fetched < total_candles:
        limit = min(1000, total_candles - fetched)

        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "end": end_ts,
            "limit": limit
        }

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("retCode") != 0:
            raise ValueError(f"API error: {data.get('retMsg', 'Unknown error')}")

        klines = data["result"]["list"]
        if not klines:
            print("Нет больше данных для загрузки")
            break

        # Формируем DataFrame
        klines_ohclv = [row[:6] for row in klines]
        df = pd.DataFrame(klines_ohclv, columns=[
            "timestamp", "open", "high", "low", "close", "volume"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")
        df = df.astype({
            "open": float, "high": float, "low": float,
            "close": float, "volume": float
        })

        # Сохраняем кусками по candles_per_file
        for start in range(0, len(df), candles_per_file):
            chunk = df.iloc[start:start + candles_per_file]
            if chunk.empty:
                continue

            filename = f"raw_candle_{file_index}.csv"
            full_path = os.path.join(save_path, filename)
            chunk.to_csv(full_path, index=False)

            print(f"Сохранено {len(chunk)} свечей в {full_path}")
            file_index += 1

        fetched += len(df)
        # Сдвигаем конец выборки на самый ранний timestamp текущего ответа
        end_ts = int(klines[-1][0]) - 1  

    print(f"Всего собрано свечей: {fetched}")



get_raw_candles_csv()