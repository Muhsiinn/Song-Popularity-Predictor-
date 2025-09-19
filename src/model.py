import pandas as pd 
import numpy as np 

def train_test_split(df, train_ratio=0.6, val_ratio=0.2):
    n = len(df)
    n_val = int(n * val_ratio)
    n_test = int(n * val_ratio)
    n_train = n - n_val - n_test
    
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    
    df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
    df_val = df.iloc[idx[n_train:n_train+n_val]].reset_index(drop=True)
    df_test = df.iloc[idx[n_train+n_val:]].reset_index(drop=True)
    
    return df_train, df_val, df_test


def train_model(df, feature_cols, target_col='streams'):
    """Complete training pipeline from dataframe to results"""
    
    # Split data
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2) 
    n_train = n - n_val - n_test
    
    idx = np.arange(n)
    np.random.shuffle(idx)
    
    # Extract train/val/test
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    
    # Get matrices
    X_train = df.iloc[train_idx][feature_cols].values
    X_val = df.iloc[val_idx][feature_cols].values
    y_train = df.iloc[train_idx][target_col].values
    y_val = df.iloc[val_idx][target_col].values
    
    # Add bias
    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    X_val = np.column_stack([np.ones(len(X_val)), X_val])
    
    # Train
    XTX = X_train.T.dot(X_train)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X_train.T).dot(y_train)
    
    # Predict and evaluate
    y_pred = X_val.dot(w)
    rmse_score = np.sqrt(np.mean((y_val - y_pred) ** 2))
    
    return w, rmse_score