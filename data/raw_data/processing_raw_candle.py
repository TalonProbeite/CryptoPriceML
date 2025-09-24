import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


# def process_raw_candles(
#     raw_file: str,
#     processed_file: Optional[str] = None,
#     threshold_pct: float = 0.073,
#     sma_period: int = 12,
#     ema_period: int = 12,
#     wma_period: int = 12,
#     ema26_period: int = 26,
#     macd_signal_period: int = 9,
#     delet_incomplete_lines:bool = True,
#     delet_timestamp: bool = True 
# ) -> pd.DataFrame:
#     """
#     Прочитать raw CSV со столбцами (timestamp, open, high, low, close, volume),
#     посчитать тренды (up, down, float) и индикаторы:
#       - SMA_{sma_period}
#       - EMA_{ema_period}
#       - WMA_{wma_period}
#       - EMA_{ema26_period}
#       - MACD  = EMA_{ema_period} - EMA_{ema26_period}
#       - Signal = EMA of MACD (span = macd_signal_period)
#     Сохранить в processed_file (если None — создаётся рядом с raw_file с префиксом "processed_").
#     Возвращает pandas.DataFrame.
#     """
#     raw_path = Path(raw_file)
#     if not raw_path.is_file():
#         raise FileNotFoundError(f"File not found: {raw_file}")

#     df = pd.read_csv(raw_path)

#     # Проверим минимум необходимых колонок
#     if "timestamp" not in df.columns or "close" not in df.columns:
#         raise KeyError("CSV должен содержать как минимум колонки 'timestamp' и 'close'")

#     if df.empty:
#         raise ValueError("CSV пустой")

#     # datetime и сортировка
#     df["timestamp"] = pd.to_datetime(df["timestamp"])
#     df = df.sort_values("timestamp").reset_index(drop=True)

#     # Тренды: сравнение текущего close с next close
#     next_close = df["close"].shift(-1)
#     # берём абсолютное % изменение; защищаемся от деления на 0
#     with np.errstate(divide="ignore", invalid="ignore"):
#         pct_change = (next_close - df["close"]).abs() / df["close"] * 100

#     # инициализация
#     df["up"] = 0
#     df["down"] = 0
#     df["flat"] = 0

#     up_mask = (next_close > df["close"]) & (pct_change > threshold_pct)
#     down_mask = (next_close < df["close"]) & (pct_change > threshold_pct)
#     float_mask = ~(up_mask | down_mask)  # включает последний ряд с NaN next_close

#     df.loc[up_mask, "up"] = 1
#     df.loc[down_mask, "down"] = 1
#     df.loc[float_mask, "flat"] = 1

#     # Индикаторы
#     # SMA
#     df[f"SMA_{sma_period}"] = df["close"].rolling(window=sma_period).mean()
#     # EMA
#     df[f"EMA_{ema_period}"] = df["close"].ewm(span=ema_period, adjust=False).mean()
#     # WMA
#     def _wma(series: pd.Series, period: int) -> pd.Series:
#         weights = np.arange(1, period + 1)
#         return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
#     df[f"WMA_{wma_period}"] = _wma(df["close"], wma_period)

#     # EMA 26, MACD, Signal
#     df[f"EMA_{ema26_period}"] = df["close"].ewm(span=ema26_period, adjust=False).mean()
#     df["MACD"] = df[f"EMA_{ema_period}"] - df[f"EMA_{ema26_period}"]
#     df["Signal"] = df["MACD"].ewm(span=macd_signal_period, adjust=False).mean()
#     n = 14
#     df['L14'] = df['low'].rolling(window=n).min()
#     df['H14'] = df['high'].rolling(window=n).max()
#     df['%K'] = (df['close'] - df['L14']) / (df['H14'] - df['L14']) * 100
#     df['%D'] = df['%K'].rolling(window=3).mean()

#     # Переместим колонки трендов в конец
#     trend_cols = ["up", "down", "flat"]
#     cols = [c for c in df.columns if c not in trend_cols] + trend_cols
#     df = df[cols]

#     if delet_incomplete_lines:
#         df = df.dropna().reset_index(drop=True)
#     if delet_timestamp:
#         df = df.drop(columns=["timestamp"])



#     # Сохранение
#     if processed_file is None:
#         processed_file = raw_path.parent / f"processed_{raw_path.name}"
#     else:
#         processed_file = Path(processed_file)
#     df.to_csv(processed_file, index=False)

#     print(f"Сохранено: {processed_file} (записей: {len(df)})")

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

def process_raw_candles(
    raw_file: str,
    processed_file: Optional[str] = None,
    threshold_pct: float = 0.073,
    sma_period: int = 12,
    ema_period: int = 12,
    wma_period: int = 12,
    ema26_period: int = 26,
    macd_signal_period: int = 9,
    delet_incomplete_lines: bool = True,
    delet_timestamp: bool = True
) -> pd.DataFrame:
    """
    Прочитать raw CSV со столбцами (timestamp, open, high, low, close, volume),
    посчитать тренды (up, down, flat) и индикаторы:
      - SMA_{sma_period}
      - EMA_{ema_period}
      - WMA_{wma_period}
      - EMA_{ema26_period}
      - MACD  = EMA_{ema_period} - EMA_{ema26_period}
      - Signal = EMA of MACD (span = macd_signal_period)
      - Стохастик %K и %D
      - open/high/low/close/volume трёх предыдущих свечей
    Сохранить в processed_file (если None — создаётся рядом с raw_file с префиксом "processed_").
    Возвращает pandas.DataFrame.
    """
    raw_path = Path(raw_file)
    if not raw_path.is_file():
        raise FileNotFoundError(f"File not found: {raw_file}")

    df = pd.read_csv(raw_path)

    # Проверим минимум необходимых колонок
    required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        raise KeyError(f"CSV должен содержать колонки: {required_cols}")

    if df.empty:
        raise ValueError("CSV пустой")

    # datetime и сортировка
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Тренды: сравнение текущего close с next close
    next_close = df["close"].shift(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_change = (next_close - df["close"]).abs() / df["close"] * 100

    df["up"] = 0
    df["down"] = 0
    df["flat"] = 0

    up_mask = (next_close > df["close"]) & (pct_change > threshold_pct)
    down_mask = (next_close < df["close"]) & (pct_change > threshold_pct)
    float_mask = ~(up_mask | down_mask)

    df.loc[up_mask, "up"] = 1
    df.loc[down_mask, "down"] = 1
    df.loc[float_mask, "flat"] = 1

    # Индикаторы
    df[f"SMA_{sma_period}"] = df["close"].rolling(window=sma_period).mean()
    df[f"EMA_{ema_period}"] = df["close"].ewm(span=ema_period, adjust=False).mean()

    def _wma(series: pd.Series, period: int) -> pd.Series:
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    df[f"WMA_{wma_period}"] = _wma(df["close"], wma_period)

    df[f"EMA_{ema26_period}"] = df["close"].ewm(span=ema26_period, adjust=False).mean()
    df["MACD"] = df[f"EMA_{ema_period}"] - df[f"EMA_{ema26_period}"]
    df["Signal"] = df["MACD"].ewm(span=macd_signal_period, adjust=False).mean()

    # Стохастик
    n = 14
    df["L14"] = df["low"].rolling(window=n).min()
    df["H14"] = df["high"].rolling(window=n).max()
    df["%K"] = (df["close"] - df["L14"]) / (df["H14"] - df["L14"]) * 100
    df["%D"] = df["%K"].rolling(window=3).mean()

    # === Добавляем open, high, low, close, volume из 3 предыдущих свечей ===
    for i in range(1, 4):  # 1, 2, 3 свечи назад
        df[f"open_lag{i}"] = df["open"].shift(i)
        df[f"high_lag{i}"] = df["high"].shift(i)
        df[f"low_lag{i}"] = df["low"].shift(i)
        df[f"close_lag{i}"] = df["close"].shift(i)
        df[f"volume_lag{i}"] = df["volume"].shift(i)

    # Переместим тренды в конец
    trend_cols = ["up", "down", "flat"]
    cols = [c for c in df.columns if c not in trend_cols] + trend_cols
    df = df[cols]

    if delet_incomplete_lines:
        df = df.dropna().reset_index(drop=True)
    if delet_timestamp:
        df = df.drop(columns=["timestamp"])

    # Сохранение
    if processed_file is None:
        processed_file = raw_path.parent / f"processed_{raw_path.name}"
    else:
        processed_file = Path(processed_file)
    df.to_csv(processed_file, index=False)

    print(f"Сохранено: {processed_file} (записей: {len(df)})")

    return df



process_raw_candles(raw_file=f"data\\raw_data\samples\\raw\\raw_dataset.csv",processed_file=f"data\\raw_data\samples\processing\processing_candle.csv" )