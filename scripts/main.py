from data_processing import *
from data_processing.mnist_preprocessing import clean_data, process_data, split_data, extract_game_price_features, extract_player_history_features
import pandas as pd
import os
from tensorflow import keras # type: ignore

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


# ================== MNIST DATASET ================== #
print("Processing MNIST dataset...")
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = keras.datasets.mnist.load_data()

# Flatten and scale
X_mnist = x_train_mnist.reshape(-1, 784).astype('float32') / 255.0
X_test_mnist = x_test_mnist.reshape(-1, 784).astype('float32') / 255.0

_, _, subsets_mnist = split_data(X_mnist, y_train_mnist)

# Plug it into the main results dictionary
results_data["mnist_digits"] = {
    'X_test': X_test_mnist, 
    'y_test': y_test_mnist, 
    'subsets': subsets_mnist
}
print(f"Successfully prepared mnist dataset sizes: {[len(v[0]) for v in subsets_mnist.values()]}")

# Model Training
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

experiment_results = []

for dataset_name, data in results_data.items():
    X_test = data['X_test']
    y_test = data['y_test']
    subsets = data['subsets']

    for size, (X_train, y_train) in subsets.items():
        print(f"\nDataset: {dataset_name}, Training size: {size}")

        for run in range(3):  # repeat 3 times
            # -------- Linear Regression --------
            lr = LinearRegression()
            lr.fit(X_train, y_train)

            train_pred_lr = lr.predict(X_train)
            test_pred_lr = lr.predict(X_test)

            train_error_lr = mean_squared_error(y_train, train_pred_lr)
            test_error_lr = mean_squared_error(y_test, test_pred_lr)

            experiment_results.append({
                "dataset": dataset_name,
                "model": "LinearRegression",
                "train_size": size,
                "run": run,
                "train_error": train_error_lr,
                "test_error": test_error_lr
            })

            # -------- MLP --------
            mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=None)

            mlp.fit(X_train, y_train)

            train_pred_mlp = mlp.predict(X_train)
            test_pred_mlp = mlp.predict(X_test)

            train_error_mlp = mean_squared_error(y_train, train_pred_mlp)
            test_error_mlp = mean_squared_error(y_test, test_pred_mlp)

            experiment_results.append({
                "dataset": dataset_name,
                "model": "MLP",
                "train_size": size,
                "run": run,
                "train_error": train_error_mlp,
                "test_error": test_error_mlp
            })

df_results = pd.DataFrame(experiment_results)

summary = df_results.groupby(["dataset", "model", "train_size"]).agg({
    "train_error": ["mean", "std"],
    "test_error": ["mean", "std"]
}).reset_index()

print(summary)


# Model Evaluation Metrics (logs and plots)
