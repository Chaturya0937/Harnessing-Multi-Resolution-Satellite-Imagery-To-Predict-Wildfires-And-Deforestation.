import os
import glob
import pandas as pd

def combine_downloaded_csvs(folder_path, output_path="./wildfire_project_global_v3.csv"):
    import glob
    import pandas as pd
    csv_files = glob.glob(f"{folder_path}/*.csv")
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    dfs = [pd.read_csv(f) for f in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    print(f"Combined {len(csv_files)} files ({len(combined_df)} rows total) into {output_path}")

combine_downloaded_csvs(r"C:\Users\Vardhan\Desktop\Coding\Minor Project\experiments\occurence_data\wildfire_project_global_v3-20260907T040659Z-1-001\wildfire_project_global_v3")