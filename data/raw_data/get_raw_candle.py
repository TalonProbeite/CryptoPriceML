import requests
import pandas as pd
from datetime import datetime, timezone
import os
import math


def get_raw_candles_csv(
    save_path: str = "data\\raw_data\\samples\\",
    symbol: str = "BTCUSDT",
    interval: str = "30",
    category: str = "spot",
    total_candles: int = 5,
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

    # End time = now (UTC). API fetches only closed candles
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
            print("No more data to fetch")
            break

        # Create DataFrame
        klines_ohclv = [row[:6] for row in klines]
        df = pd.DataFrame(klines_ohclv, columns=[
            "timestamp", "open", "high", "low", "close", "volume"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")
        df = df.astype({
            "open": float, "high": float, "low": float,
            "close": float, "volume": float
        })

        # Save in chunks of candles_per_file
        for start in range(0, len(df), candles_per_file):
            chunk = df.iloc[start:start + candles_per_file]
            if chunk.empty:
                continue

            filename = f"raw_candle_{file_index}.csv"
            full_path = os.path.join(save_path, filename)
            chunk.to_csv(full_path, index=False)

            print(f"Saved {len(chunk)} candles to {full_path}")
            file_index += 1

        fetched += len(df)
        # Shift the end of the sample to the earliest timestamp of the current response
        end_ts = int(klines[-1][0]) - 1  

    print(f"Total candles collected: {fetched}")

