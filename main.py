import pandas as pd
import numpy as np
from src.data_processing import load_data, data_preparation, synthetic_data
from src.feature_engineering import features_X
from src.model import train_model

# Define features in main for easy editing
base = ['bpm', 'danceability', 'valence', 'energy', 'acousticness', 
        'instrumentalness', 'liveness', 'speechiness', 'streams', 
        'inspotifyplaylists', 'inspotifycharts', 'inappleplaylists', 'inapplecharts']

def main():
    print("=== Spotify Hit Predictor Pipeline ===")
    
    # 1. Load and preprocess
    print("1. Loading and preprocessing data...")
    raw_df = load_data('data/raw/spotify.csv')
    processed_df = data_preparation(raw_df, base)
    print(f"   Processed {len(processed_df)} songs")
    
    # 2. Create synthetic data directly
    print("2. Creating synthetic variations...")
    
    # Create synthetic variations
    medium_df = synthetic_data(processed_df, 0.7, 0.9)
    low_df = synthetic_data(processed_df, 0.3, 0.6)
    
    # Combine all datasets
    final_df = pd.concat([processed_df, medium_df, low_df], ignore_index=True)
    # Apply log transform to streams in the final dataset
    final_df['streams'] = np.log1p(final_df['streams'])
    print(f"   Final dataset: {len(final_df)} songs")
    
    # 3. Feature engineering
    print("3. Engineering features...")
    df_engineered = features_X(final_df)

    
    # 4. Define feature columns (exclude streams as it's the target)
    feature_cols = [col for col in base if col != 'streams'] + ['energy_x_danceability', 'valence_x_energy', 'mood_score']
    # 5. Train model
    print("4. Training model...")
    
    print("5. Training model...")
    weights, val_rmse, test_rmse, y_test, y_test_pred = train_model(df_engineered, feature_cols)

    print(f"6. Validation RMSE: {val_rmse:.3f}")
    print(f"7. Test RMSE: {test_rmse:.3f}")
   
   
    
if __name__ == "__main__":
    main()