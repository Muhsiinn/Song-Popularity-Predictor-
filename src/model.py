import pandas as pd 
import numpy as np 


def train_model(df, feature_cols, target_col='streams', r=0.001):
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2) 
    n_train = n - n_val - n_test
    
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    
    
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    
    
    X_train = df.iloc[train_idx][feature_cols].values
    X_val = df.iloc[val_idx][feature_cols].values
    X_test = df.iloc[test_idx][feature_cols].values
    
    y_train = df.iloc[train_idx][target_col].values
    y_val = df.iloc[val_idx][target_col].values
    y_test = df.iloc[test_idx][target_col].values
    
    
    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    X_val = np.column_stack([np.ones(len(X_val)), X_val])
    X_test = np.column_stack([np.ones(len(X_test)), X_test])
    
    
    XTX = X_train.T.dot(X_train)
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X_train.T).dot(y_train)
    
    
    y_val_pred = X_val.dot(w)
    y_test_pred = X_test.dot(w)
    
    val_rmse = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
    test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
    
    return w, val_rmse, test_rmse, y_test, y_test_pred