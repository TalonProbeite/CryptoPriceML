from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import glob


def get_normalization_data():
    df = pd.read_csv("data/ready_data/samples/dataset_eth_5.csv")

    
    df_features = df.iloc[:, :-3]
    df_tail = df.iloc[:, -3:]

    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    
    df_norm = pd.DataFrame(X_scaled, columns=df_features.columns)
    df_final = pd.concat([df_norm, df_tail.reset_index(drop=True)], axis=1)

    
    df_final.to_csv("data/ready_data/samples/dataset_eth_5_norm.csv", index=False)



def merg_data():
    path = r"data\\ready_data\\samples"  # путь к папке с CSV
    files = glob.glob(os.path.join(path, "dataset_*_*_norm.csv"))

    frames = []
    total_up = 0
    total_down = 0
    total_flat = 0

    for f in files:
        parts = os.path.basename(f).split("_")
        asset = parts[1]
        tf = parts[2]

        df = pd.read_csv(f)

        up = df[df["Up"] == 1]
        down = df[df["Down"] == 1]
        flat = df[df["Flat"] == 1]

        # Берем до 20000 строк, если меньше — сколько есть
        up = up.sample(min(20000, len(up)), random_state=42)
        down = down.sample(min(20000, len(down)), random_state=42)
        flat = flat.sample(min(20000, len(flat)), random_state=42)

        subset = pd.concat([up, down, flat])
        subset["asset"] = asset
        subset["time_frame"] = tf

        frames.append(subset)

        total_up += len(up)
        total_down += len(down)
        total_flat += len(flat)

    if not frames:
        raise RuntimeError("Нет подходящих файлов для объединения.")

    final_df = pd.concat(frames)
    final_df.to_csv(os.path.join(path, "data.csv"), index=False)

    print("\n✅ Обработка завершена.")
    print(f"Всего строк с Up = 1:   {total_up}")
    print(f"Всего строк с Down = 1: {total_down}")
    print(f"Всего строк с Flat = 1: {total_flat}")

