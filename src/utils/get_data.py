import pandas as pd


def get_data_from_csv(path="data\\raw_data\samples\processing\processing_candle1.csv"):
    data = pd.read_csv(path)
    X = data.iloc[:, :-3].values
    y = data.iloc[:, -3:].values
    return [X,y]