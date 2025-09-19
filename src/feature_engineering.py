
import pandas as pd 
def features_X(df):
    df = df.copy()
    df['energy_x_danceability'] = df['energy'] * df['danceability']
    df['valence_x_energy'] = df['valence'] * df['energy']
    df['mood_score'] = df['valence'] - df['acousticness']
    df = df.fillna(0)
    return df