import pandas as pd


def process_candle_data(
    input_file: str = "data\\raw_data\samples\\raw_candle_test.csv",
    output_file: str = "data\\raw_data\samples\\processed_candle_test.csv",
    threshold: float = 0.1
) -> None:
    """
    Process raw candle CSV data by adding up, down, and float columns based on price changes.

    Args:
        input_file (str): Path to the input CSV file containing raw candle data.
                          Defaults to "data\\raw_data\\samples\\raw_candle1.csv".
        output_file (str): Path to the output CSV file where processed data will be saved.
                           Defaults to "data\\raw_data\\samples\\processed_candle.csv".
        threshold (float): Percentage threshold for significant price change.
                           Defaults to 0.2 (0.2%).

    Returns:
        None: Saves processed data to CSV file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        pd.errors.EmptyDataError: If the input CSV is empty.
        ValueError: If required columns are missing in the input data.
        PermissionError: If unable to write to the output file.

    Examples:
        >>> process_candle_data("raw_data.csv", "processed_data.csv", 0.5)
        >>> process_candle_data()  # Uses all default parameters
    """
    try:
        # Read the input CSV file
        df = pd.read_csv(input_file)

        # Validate required columns
        required_columns = ['timestamp', 'close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(
                f"Input file must contain columns: {required_columns}")

    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_file}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("Input CSV file is empty")
    except Exception as e:
        raise ValueError(f"Error reading input file: {e}")

    # Convert timestamp to datetime and sort data chronologically
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Initialize new columns
    df['up'] = 0
    df['down'] = 0
    df['float'] = 0

    # Process each row except the last one
    for i in range(len(df) - 1):
        current_close = df.loc[i, 'close']
        next_close = df.loc[i + 1, 'close']

        # Calculate absolute percentage change
        percent_change = abs(next_close - current_close) / current_close * 100

        # Determine classification based on conditions
        if next_close > current_close and percent_change > threshold:
            df.loc[i, 'up'] = 1
        elif next_close < current_close and percent_change > threshold:
            df.loc[i, 'down'] = 1
        else:
            df.loc[i, 'float'] = 1

    try:
        # Save processed data to output CSV
        df.to_csv(output_file, index=False)

        # Output success message and statistics
        print('Сохранено')
        print(f"Processed records: {len(df)}")
        print(
            f"Up cases: {df['up'].sum()}, Down cases: {df['down'].sum()}, Float cases: {df['float'].sum()}")

    except PermissionError:
        raise PermissionError(
            f"Permission denied when writing to: {output_file}")
    except Exception as e:
        raise ValueError(f"Error saving output file: {e}")


process_candle_data()
