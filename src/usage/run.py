import sys 
import os
import requests
import pandas as pd
from datetime import datetime, timezone
import ta
from sklearn.preprocessing import MinMaxScaler
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = все сообщения, 1 = убрать INFO, 2 = убрать INFO+WARNING, 3 = убрать всё кроме ошибок

import keras

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


def main():
    def get_model_info(current_model):
        model = keras.models.load_model(current_model + "\\best_model.keras")
        parts = current_model.split('\\')
        model_type = parts[1].replace('\\','')
        model_name = parts[2].replace('\\','')
        print(f"Model:\nType: {model_type}\nName: {model_name}")
        model.summary()

    def get_start_model():
        for root, dirs, files in os.walk("models\MLP"):
            return "models\MLP\\"+dirs[0]  


    def switch_model():
        root, dirs, files = next(os.walk("models\\MLP"))
        dir_mlp = dirs  
        os.system("cls")
        print("select model number:")
        n = 1
        models = {}
        print("MLP:")
        if dir_mlp == []:
            print("model is missing")
        for el in dir_mlp:
            print(f"    {n}.{el}")
            models[n] = "models\MLP\\" + el 
            n+=1
        model = models[int(input())]
        return model
    

    def get_predict(current_model):
        def get_input_vector():
            end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
            url = "https://api.bybit.com/v5/market/kline"
            params = {
            "category": "spot",
            "symbol": "BTCUSDT",
            "interval": "30",
            "end": end_ts,
            "limit": 200}
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            klines = data["result"]["list"]
            klines_ohclv = [row[:6] for row in klines]
            raw_dt = pd.DataFrame(klines_ohclv, columns=[
                "timestamp", "open", "high", "low", "close", "volume"
            ])
            raw_dt["timestamp"] = pd.to_datetime(raw_dt["timestamp"].astype("int64"), unit="ms")
            raw_dt = raw_dt.astype({
                "open": float, "high": float, "low": float,
                "close": float, "volume": float
            })
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
            df = df.dropna().reset_index(drop=True)
            scaler = MinMaxScaler()
            df[df.columns] = scaler.fit_transform(df[df.columns])

            return df.iloc[0].tolist()
        
        x_input = get_input_vector()  
        x_input = np.array(x_input).reshape(1, -1)  
        model = keras.models.load_model(current_model+"\\best_model.keras")
        y_pred = model.predict(x_input)
        print(f"""prediction:
                  Up: {y_pred[0][0]}
                  Down: {y_pred[0][1]}
                  Flat: {y_pred[0][2]}""")


        


    run = True
    current_model= get_start_model()
    while run:
         answer = input("\nOptions:\n   completion of work: close\n   get information about the current model: model_info\n   switch model: switch_model\n    get a prediction: get_predict\n->")
         match answer:
            case 'close':
                run = False
            case "model_info":
                os.system('cls')
                get_model_info(current_model=current_model)
            case "clear":
                os.system("cls")
            case "switch_model":
                os.system('cls')
                current_model = switch_model()
                os.system("cls")
                print("model switched!")
            case "get_predict":
                 os.system("cls")
                 get_predict(current_model)
            



        

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass