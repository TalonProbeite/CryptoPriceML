import pandas as pd

df = pd.read_csv('data\\raw_data\samples\\raw_candle1.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp').reset_index(drop=True)

df['up'] = 0
df['down'] = 0
df['float'] = 0
threshold = 0.2

for i in range(len(df) - 1):
    current_close = df.loc[i, 'close']
    next_close = df.loc[i + 1, 'close']

    percent_change = abs(next_close - current_close) / current_close * 100

    if next_close > current_close and percent_change > threshold:
        df.loc[i, 'up'] = 1
    elif next_close < current_close and percent_change > threshold:
        df.loc[i, 'down'] = 1
    else:
        df.loc[i, 'float'] = 1

df.to_csv('data\\raw_data\samples\\processed_candle.csv', index=False)
print('Сохранено')
