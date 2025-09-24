import pandas as pd
from pathlib import Path


def normalize_data(
    file_path: str = "data\\raw_data\samples\\ready\processing_candle (6).csv"
) -> None:
    """
    Normalize numerical columns in processed_candle_1.csv by removing rows with NaN
    and applying Min-Max scaling to open, high, low, close, volume, SMA_12, EMA_12, WMA_12,
    ensuring up, down, float remain unchanged.

    Args:
        file_path (str): Path to the CSV file to be normalized.
                        Defaults to "processed_candle_1.csv" (relative to current directory).

    Returns:
        None: Updates the CSV file with normalized data.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        KeyError: If required columns are missing.
    """
    # Validate file existence
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read existing data
    df = pd.read_csv(file_path)

    # Define columns to normalize
    columns_to_normalize = ['open', 'high', 'low',
                            'close', 'volume', 'SMA_12', 'EMA_12', 'WMA_12', 'EMA_26', 'MACD', 'Signal', 'L14', 'H14', '%K', '%D', 'open_lag1', 'high_lag1', 'low_lag1', 'close_lag1', 'volume_lag1', 'open_lag2', 'high_lag2', 'low_lag2', 'close_lag2', 'volume_lag2', 'open_lag3', 'high_lag3', 'low_lag3', 'close_lag3', 'volume_lag3']

    # Check for missing columns
    missing_cols = [
        col for col in columns_to_normalize if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns: {missing_cols}")

    # Remove rows with NaN in the columns to normalize
    df = df.dropna(subset=columns_to_normalize)

    # Apply Min-Max normalization
    for column in columns_to_normalize:
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val > min_val:  # Avoid division by zero
            df[column] = (df[column] - min_val) / (max_val - min_val)
        else:
            print(
                f"Column {column} has constant values, skipping normalization.")

    # Save updated data back to the same CSV file
    df.to_csv(file_path, index=False)

    # Output success message and statistics
    print(f"Data normalized and saved to: {file_path}")
    print(f"Normalized columns: {columns_to_normalize}")
    print(
        f"Rows before: {len(df) + df[columns_to_normalize].isna().any(axis=1).sum()}, Rows after: {len(df)}")
    print("Sample after normalization (first 5 rows):")
    print(df[columns_to_normalize].head())
    print("\nRange check (should be ~0 to 1):")
    for col in columns_to_normalize:
        print(f"{col}: min={df[col].min():.4f}, max={df[col].max():.4f}")


if __name__ == "__main__":
    normalize_data()
