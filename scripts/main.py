from data_processing import *
from data_processing.mnist_preprocess import clean_data, process_data, split_data, extract_game_price_features, extract_player_history_features
import pandas as pd
import os

# Data Preprocessing

# Gaming console datasets - each folder contains multiple CSV files
# Get the parent directory of the scripts folder for the datasets path
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")
datasets_path = os.path.join(script_dir, "datasets")

results_data = {}

platforms = ['steam', 'playstation', 'xbox']

# Process game price features
for platform in platforms:
    print(f"Processing game price features for {platform}")
    df = extract_game_price_features(datasets_path, platform, sample_size=10000)  # Sample to handle large data
    if not df.empty:
        target_col = 'usd'
        if target_col in df.columns:
            X, y = process_data(df, target_col)
            X_test, y_test, subsets = split_data(X, y)
            results_data[f"{platform}_price"] = {'X_test': X_test, 'y_test': y_test, 'subsets': subsets}
            print(f"Successfully prepared {platform}_price. Subset sizes: {[len(v[0]) for v in subsets.values()]}")
        else:
            print(f"Target column {target_col} not found in {platform} data.")

# Process player history features
for platform in platforms:
    print(f"Processing player history features for {platform}")
    df = extract_player_history_features(datasets_path, platform, sample_size=10000)
    if not df.empty:
        target_col = 'total_spend'  # or 'num_games'
        if target_col in df.columns:
            X, y = process_data(df, target_col)
            X_test, y_test, subsets = split_data(X, y)
            results_data[f"{platform}_history"] = {'X_test': X_test, 'y_test': y_test, 'subsets': subsets}
            print(f"Successfully prepared {platform}_history. Subset sizes: {[len(v[0]) for v in subsets.values()]}")
        else:
            print(f"Target column {target_col} not found in {platform} data.")

print("Feature extraction completed.")


# Model Training


# Model Evaluation Metrics (logs and plots)
