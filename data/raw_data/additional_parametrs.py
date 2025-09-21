import pandas as pd
from pathlib import Path


def update_indicators_with_trends(
    file_path: str = "data\\raw_data\samples\processed_candle_1.csv"
) -> None:
    """
    Update processed_candle.csv with SMA_12, EMA_12, and WMA_12 indicators,
    ensuring up, down, float remain the last columns.

    Args:
        file_path (str): Path to the CSV file to be updated.
                        Defaults to "processed_candle.csv" (relative to current directory).

    Returns:
        None: Updates the CSV file with new indicator columns.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        KeyError: If the 'close' or trend columns (up, down, float) are missing.
        ValueError: If the input data is empty.
    """
    # Validate file existence
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read existing data
    df = pd.read_csv(file_path)

    # Validate required columns
    required_columns = ["close", "up", "down", "float"]
    if not all(col in df.columns for col in required_columns):
        raise KeyError(f"CSV must contain columns: {required_columns}")

    # Validate data
    if df.empty:
        raise ValueError("CSV is empty")

    # Convert timestamp to datetime and sort by time (optional, for consistency)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(by="timestamp").reset_index(drop=True)

    # Calculate indicators with 12-day period
    period = 12
    df["SMA_12"] = df["close"].rolling(window=period).mean()
    df["EMA_12"] = df["close"].ewm(span=period, adjust=False).mean()

    def wma(series, period):
        weights = list(range(1, period + 1))
        return series.rolling(period).apply(lambda x: (x * weights).sum() / sum(weights), raw=True)

    df["WMA_12"] = wma(df["close"], period)

    # Reorder columns to place trend columns (up, down, float) last
    trend_columns = ["up", "down", "float"]
    columns = [
        col for col in df.columns if col not in trend_columns] + trend_columns
    df = df[columns]

    # Save updated data back to the same CSV file
    df.to_csv(file_path, index=False)

    # Output success message
    print(f"Indicators added and saved to: {file_path}")
    print(f"Records processed: {len(df)}")


if __name__ == "__main__":
    update_indicators_with_trends()
