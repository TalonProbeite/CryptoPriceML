import pandas as pd
import glob
import os



def combine_csv_files(directory_path, output_filename='raw_dataset.csv'):
    search_pattern = os.path.join(directory_path, '*.csv')
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        return "CSV файлы не найдены"
    
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    output_path = os.path.join(directory_path, output_filename)
    combined_df.to_csv(output_path, index=False)
    
    return f"Объединено {len(csv_files)} файлов в {output_path}"


combine_csv_files(directory_path="data\\raw_data\samples\\raw\\")