import pandas as pd
import ta

def processing_data(threshold: float = 0.1):
    # Загружаем сырые данные
    raw_dt = pd.read_csv("data\\raw_data\\samples\\raw_dataset.csv")

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

    # --- Volume (в процентах) ---
    df["Volume_t"]     = raw_dt["volume"].pct_change(1) * 100
    df["Volume_(t-1)"] = raw_dt["volume"].pct_change(2) * 100
    df["Volume_(t-2)"] = raw_dt["volume"].pct_change(3) * 100

    # --- Derived features из свечей ---
    df["Body_size"] = (raw_dt["close"] - raw_dt["open"]).abs()
    df["Range"] = raw_dt["high"] - raw_dt["low"]
    df["Body/Range_ratio"] = df["Body_size"] / df["Range"].replace(0, 1)  # защита от деления на 0
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
    df["ATR_14"] = ta.volatility.AverageTrueRange(
        high=raw_dt["high"],
        low=raw_dt["low"],
        close=raw_dt["close"],
        window=14
    ).average_true_range()

    # --- Bollinger Bands (20, 2) ---
    bb = ta.volatility.BollingerBands(close=raw_dt["close"], window=20, window_dev=2)
    df["BB_bbm"] = bb.bollinger_mavg()     # средняя линия
    df["BB_bbh"] = bb.bollinger_hband()    # верхняя граница
    df["BB_bbl"] = bb.bollinger_lband()    # нижняя граница
    df["BB_bbw"] = bb.bollinger_wband()    # ширина (сжатие/расширение)
    df["BB_percent"] = bb.bollinger_pband()  # позиция цены внутри полос

    # --- EMA (10) и EMA (50) ---
    df["EMA_10"] = ta.trend.EMAIndicator(close=raw_dt["close"], window=10).ema_indicator()
    df["EMA_50"] = ta.trend.EMAIndicator(close=raw_dt["close"], window=50).ema_indicator()
    df["EMA_diff"] = df["EMA_10"] - df["EMA_50"]  # сигнал для "золотого креста"

    # --- Целевая переменная (Up/Down/Flat) ---
    future_close = raw_dt["close"].shift(-1)
    change_pct = (future_close - raw_dt["close"]) / raw_dt["close"] * 100

    df["Up"]   = (change_pct > threshold).astype(int)
    df["Down"] = (change_pct < -threshold).astype(int)
    df["Flat"] = ((change_pct >= -threshold) & (change_pct <= threshold)).astype(int)

    # Убираем NaN (первые из-за pct_change и последние из-за shift)
    df = df.dropna().reset_index(drop=True)

    # Сохраняем
    output_path = "data\\ready_data\samples\dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Таблица сохранена: {output_path}")

    # Статистика распределения классов
    total = len(df)
    up_pct = df["Up"].sum() / total * 100
    down_pct = df["Down"].sum() / total * 100
    flat_pct = df["Flat"].sum() / total * 100
    print(f"📊 Up: {up_pct:.2f}%, Down: {down_pct:.2f}%, Flat: {flat_pct:.2f}%")

    return df





processing_data()