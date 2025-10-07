import pandas as pd
import ta
import numpy as np


def processing_data_c5(threshold: float = 0.041):
    # Загружаем сырые данные
    raw_dt = pd.read_csv("data\\raw_data\\samples\\raw_dataset_eth_5.csv", parse_dates=True, index_col=0)

    # Проверим наличие нужных колонок
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in raw_dt.columns:
            raise ValueError(f"Нет колонки {col} в датасете")

    df = pd.DataFrame()

    # --- Close (в процентах) ---
    df["Close_t"]     = raw_dt["close"].pct_change(1) * 100
    df["Close_(t-1)"] = raw_dt["close"].pct_change(2) * 100
    df["Close_(t-2)"] = raw_dt["close"].pct_change(3) * 100
    df["Close_(t-3)"] = raw_dt["close"].pct_change(4) * 100
    df["Close_(t-4)"] = raw_dt["close"].pct_change(5) * 100

    # --- Volume (в процентах) ---
    df["Volume_t"]     = raw_dt["volume"].pct_change(1) * 100
    df["Volume_(t-1)"] = raw_dt["volume"].pct_change(2) * 100
    df["Volume_(t-2)"] = raw_dt["volume"].pct_change(3) * 100
    df["Volume_(t-3)"] = raw_dt["volume"].pct_change(4) * 100
    df["Volume_(t-4)"] = raw_dt["volume"].pct_change(5) * 100

    # --- Derived features из свечей ---
    df["Body_size"] = (raw_dt["close"] - raw_dt["open"]).abs()
    df["Range"] = raw_dt["high"] - raw_dt["low"]
    df["Body/Range_ratio"] = df["Body_size"] / df["Range"].replace(0, 1)
    df["Upper_shadow"] = raw_dt["high"] - raw_dt[["open", "close"]].max(axis=1)
    df["Lower_shadow"] = raw_dt[["open", "close"]].min(axis=1) - raw_dt["low"]

    # --- RSI (14) ---
    df["RSI_14"] = ta.momentum.RSIIndicator(close=raw_dt["close"], window=14).rsi()

    # --- MACD (12, 26, 9) ---
    macd = ta.trend.MACD(close=raw_dt["close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"] = macd.macd_diff()

    # --- ATR (14) ---
    atr = ta.volatility.AverageTrueRange(
        high=raw_dt["high"],
        low=raw_dt["low"],
        close=raw_dt["close"],
        window=14
    )
    df["ATR_14"] = atr.average_true_range()

    # --- Bollinger Bands (20, 2) ---
    bb = ta.volatility.BollingerBands(close=raw_dt["close"], window=20, window_dev=2)
    df["BB_bbm"] = bb.bollinger_mavg()
    df["BB_bbh"] = bb.bollinger_hband()
    df["BB_bbl"] = bb.bollinger_lband()
    df["BB_bbw"] = bb.bollinger_wband()
    df["BB_percent"] = bb.bollinger_pband()

    # --- EMA (10, 50, 200) ---
    df["EMA_10"] = ta.trend.EMAIndicator(close=raw_dt["close"], window=10).ema_indicator()
    df["EMA_50"] = ta.trend.EMAIndicator(close=raw_dt["close"], window=50).ema_indicator()
    df["EMA_diff"] = df["EMA_10"] - df["EMA_50"]
    df["EMA_200"] = ta.trend.EMAIndicator(close=raw_dt["close"], window=200).ema_indicator()

    # --- Distance_to_swing (High/Low за 48 свечей ≈ 1 день) ---
    N = 48
    df["Distance_to_High"] = raw_dt["high"].rolling(N).max() - raw_dt["close"]
    df["Distance_to_Low"] = raw_dt["close"] - raw_dt["low"].rolling(N).min()

    # --- ATR_ratio ---
    df["ATR_ratio"] = df["ATR_14"] / df["ATR_14"].rolling(30).median()

    # --- OBV + slope ---
    obv = [0]  # OBV начальное значение
    for i in range(1, len(raw_dt)):
        if raw_dt["close"].iloc[i] > raw_dt["close"].iloc[i-1]:
            obv.append(obv[-1] + raw_dt["volume"].iloc[i])
        elif raw_dt["close"].iloc[i] < raw_dt["close"].iloc[i-1]:
            obv.append(obv[-1] - raw_dt["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    df["OBV_slope"] = df["OBV"].diff()

    # --- Целевая переменная (Up/Down/Flat) ---
    future_close = raw_dt["close"].shift(-1)
    change_pct = (future_close - raw_dt["close"]) / raw_dt["close"] * 100

    df["Up"]   = (change_pct > threshold).astype(int)
    df["Down"] = (change_pct < -threshold).astype(int)
    df["Flat"] = ((change_pct >= -threshold) & (change_pct <= threshold)).astype(int)

    # Убираем NaN
    df = df.dropna().reset_index(drop=True)

    # Сохраняем
    output_path = "data\\ready_data\\samples\\dataset_eth_5.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Таблица сохранена: {output_path}")

    # Статистика распределения классов
    total = len(df)
    up_pct = df["Up"].sum() / total * 100
    down_pct = df["Down"].sum() / total * 100
    flat_pct = df["Flat"].sum() / total * 100
    print(f"📊 Up: {up_pct:.2f}%, Down: {down_pct:.2f}%, Flat: {flat_pct:.2f}%")

    return df





processing_data_c5()