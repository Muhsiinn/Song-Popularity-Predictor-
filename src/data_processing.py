import numpy as np 
import pandas as pd

def load_data(filepath,features=None):

    df = pd.read_csv(filepath, encoding='latin-1')
    df.columns = df.columns.str.lower().str.replace('_','').str.replace('%','')

    return df

base = ['bpm','danceability','valence','energy','acousticness','instrumentalness','liveness',               
'speechiness','streams','inspotifyplaylists','inspotifycharts','inappleplaylists','inapplecharts' ]

def data_preparation(df,base=[]):
    df = df.copy()
    df = df[base]
    # stream colums seems to be not in int format so we convert it
    df['streams'] = df['streams'].astype(str).str.replace(r'[^0-9]', '', regex=True)
    df['streams'] = pd.to_numeric(df['streams'], errors='coerce').fillna(0).astype('int64')

    return df

def synthetic_data(df, multiplier_min=0.6, multiplier_max=0.9):
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Apply multiplier 
    multiplier = np.random.uniform(multiplier_min, multiplier_max)
    df[numeric_cols] *= multiplier
    
    
    
    return df

def create_final_dataset(df):
    medium_df = synthetic_data(df, 0.7, 0.9)
    low_df = synthetic_data(df, 0.3, 0.6)
    
    # log of original streams too
    df['streams'] = np.log1p(df['streams'])
    
    final_df = pd.concat([df, medium_df, low_df], ignore_index=True)
    return final_df







