import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import ast
import os
from datetime import datetime

def clean_data(df, name):
    """ Performs quality checks and handles missing values in the dataset. """
    #Sum of null values in each column
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        for col in df.columns:
            # Fill missing values with mode for categorical columns and mean for numerical columns
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                # For object/string columns, fill with mode if available
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    # If no mode available, fill with a placeholder
                    df[col] = df[col].fillna('unknown')
    print(f"Dataset '{name}' dimensions after cleaning: {df.shape}")
    return df


def load_sample_csv(file_path, sample_size=10000):
    """ Load a sample of the CSV file to handle large datasets. """
    try:
        # For large files, read first sample_size rows
        df = pd.read_csv(file_path, nrows=sample_size)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()


def extract_game_price_features(datasets_path, platform='steam', sample_size=10000):
    """ Extract features for game price prediction. """
    games_path = os.path.join(datasets_path, platform, 'games.csv')
    prices_path = os.path.join(datasets_path, platform, 'prices.csv')
    
    games_df = load_sample_csv(games_path, sample_size)
    prices_df = load_sample_csv(prices_path, sample_size)
    
    if games_df.empty or prices_df.empty:
        print("No data loaded for game price features.")
        return pd.DataFrame()
    
    # Merge on gameid
    df = pd.merge(games_df, prices_df, on='gameid', how='inner')
    
    # Clean data
    df = clean_data(df, f"{platform}_games_prices")
    
    # Feature engineering
    # Convert release_date to game age in years
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    current_year = datetime.now().year
    df['game_age'] = current_year - df['release_date'].dt.year
    df['game_age'] = df['game_age'].fillna(df['game_age'].mean())
    
    # Count supported languages
    df['num_languages'] = df['supported_languages'].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) and x.startswith('[') else 1)
    
    # One-hot encode genres (assuming it's a list)
    df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])
    genres_dummies = df['genres'].explode().str.get_dummies().groupby(level=0).sum()
    df = pd.concat([df, genres_dummies], axis=1)
    
    # Drop unnecessary columns
    drop_cols = ['gameid', 'title', 'developers', 'publishers', 'genres', 'supported_languages', 'release_date', 'date_acquired', 'eur', 'gbp', 'jpy', 'rub']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    # Target is usd
    if 'usd' not in df.columns:
        print("USD column not found.")
        return pd.DataFrame()
    
    df['usd'] = pd.to_numeric(df['usd'], errors='coerce')
    df = df.dropna(subset=['usd'])
    
    return df


def extract_player_history_features(datasets_path, platform='steam', sample_size=10000):
    """ Extract features for player purchase history. """
    players_path = os.path.join(datasets_path, platform, 'players.csv')
    purchased_path = os.path.join(datasets_path, platform, 'purchased_games.csv')
    prices_path = os.path.join(datasets_path, platform, 'prices.csv')
    
    players_df = load_sample_csv(players_path, sample_size)
    purchased_df = load_sample_csv(purchased_path, sample_size)
    # Load all prices to ensure price_dict is complete
    prices_df = pd.read_csv(prices_path)
    
    if players_df.empty or purchased_df.empty:
        print("No data loaded for player history features.")
        return pd.DataFrame()
    
    # Parse library
    purchased_df['library'] = purchased_df['library'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])
    
    # Merge players and purchased
    df = pd.merge(players_df, purchased_df, on='playerid', how='inner')
    
    # Feature engineering
    df['num_games'] = df['library'].apply(len)
    
    # Calculate total spend (load all prices)
    price_dict = prices_df.set_index('gameid')['usd'].to_dict()
    df['total_spend'] = df['library'].apply(lambda libs: sum(price_dict.get(gid, 0) for gid in libs if gid in price_dict))
    df['avg_price'] = df['total_spend'] / df['num_games'].replace(0, 1)
    
    # Account age from created
    if 'created' in df.columns:
        df['created'] = pd.to_datetime(df['created'], errors='coerce')
        current_time = pd.Timestamp.now()
        df['account_age_days'] = (current_time - df['created']).dt.days
        df['account_age_days'] = df['account_age_days'].fillna(df['account_age_days'].mean())
    
    # Country encoding (if exists)
    if 'country' in df.columns:
        df['country_encoded'] = LabelEncoder().fit_transform(df['country'].astype(str))
    
    # Drop unnecessary
    drop_cols = ['playerid', 'library', 'created', 'country']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    # Remove rows where total_spend is NaN or 0 if no valid prices
    df = df[df['total_spend'] > 0].dropna(subset=['total_spend'])
    
    # Target could be total_spend or num_games
    return df


def process_data(df, target_col):
    """ Processes the dataset for machine learning model training. """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target labels if they are strings
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Keep numeric features and only one-hot encode low-cardinality categorical features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    low_card_cats = [col for col in categorical_cols if X[col].nunique(dropna=False) <= 10]
    high_card_cats = [col for col in categorical_cols if col not in low_card_cats]

    if high_card_cats:
        print(f"Dropping high-cardinality categorical columns: {high_card_cats}")

    if low_card_cats:
        X = pd.get_dummies(X[numeric_cols + low_card_cats], drop_first=True)
    else:
        X = X[numeric_cols]

    if X.shape[1] == 0:
        raise ValueError("No valid feature columns remain after preprocessing. Add numeric features or low-cardinality categorical columns.")

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Processed dataset dimensions: X={X_scaled.shape}, y={y.shape}")
    print(f"Number of input features: {X_scaled.shape[1]}")
    print(f"Labels: {np.unique(y)}")
    return X_scaled, y

def split_data(X, y, train_sizes=[0.1, 0.3, 0.5, 1.0], test_size=0.2, random_state=42):
    """ Splits the dataset into training and testing sets based on varying training sizes. """
    n_samples = len(X)
    if n_samples < 2:
        raise ValueError("Dataset too small for splitting. Need at least 2 samples.")
    
    # Adjust test_size if necessary
    min_test = max(1, int(n_samples * 0.1))  # At least 1 for test
    if n_samples * test_size < 1:
        test_size = min_test / n_samples
    
    #Extract test set from training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    subsets = {}

    #Create subsets of the training data based on specified sizes
    for size in train_sizes:
        if size == 1.0:
            subsets[1.0] = (X_train, y_train)
        else:
            n_train_subset = int(len(X_train) * size)
            if n_train_subset < 1:
                print(f"Skipping subset {size}: would result in 0 training samples.")
                continue
            X_subset, _, y_subset, _ = train_test_split(X_train, y_train, train_size=size, random_state=random_state)
            subsets[size] = (X_subset, y_subset)

    print(f"Test set dimensions: X_test={X_test.shape}, y_test={y_test.shape}")
    for size, (X_sub, y_sub) in subsets.items():
        print(f"Train subset {size}: X_train={X_sub.shape}, y_train={y_sub.shape}")
    return X_test, y_test, subsets
