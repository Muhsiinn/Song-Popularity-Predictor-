# Spotify Song Popularity Predictor

Predicts song popularity using audio features and playlist data. Built with linear regression on Spotify dataset.

## What it does

Takes song characteristics (tempo, energy, danceability, etc.) and predicts how many streams it'll get. Uses log-transformed stream counts because the original numbers are all over the place.

## Results

- Validation RMSE: 0.947  
- Test RMSE: 1.101
- Dataset: 2,859 songs (953 real + synthetic variations)

The notebook version gets around 1.47 RMSE. Production pipeline performs better, probably due to different data splits and regularization.

## Data Source

Built using the [Top Spotify Songs 2023](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023) dataset from Kaggle. Started with 953 real songs, then created synthetic variations by scaling features down to simulate less popular tracks. This gave us different popularity tiers for training.

The data is pretty scarce though. Would be cool to train this on way more real songs instead of having to make synthetic ones. Like maybe thousands of songs from different years and popularity levels. Right now I'm just working with what I got.

## Files

```
spotify-hit-predictor/
├── main.py                     # Runs everything
├── src/
│   ├── data_processing.py      # Loads data, creates synthetic variations
│   ├── feature_engineering.py # Makes interaction features  
│   └── model.py               # Linear regression training
├── data/raw/spotify_dataset.csv
└── playground/workbook.ipynb  # Original exploration
```

## Features

**Audio stuff**: BPM, danceability, valence, energy, acousticness, instrumentalness, liveness, speechiness

**Platform stuff**: Spotify playlists, Apple playlists, chart positions

**Engineered**: energy × danceability, valence × energy, mood score

Turns out playlist counts matter way more than the actual music features. Makes sense.

## How to run

```bash
python main.py
```

Needs pandas, numpy, seaborn, matplotlib.

## What happens

1. Loads the CSV file
2. Cleans up the data (log transforms, handles missing values)
3. Creates synthetic songs by scaling down features to simulate less popular tracks
4. Engineers some interaction features
5. Trains regularized linear regression with 60/20/20 train/val/test split
6. Spits out RMSE

## Dependencies

- pandas
- numpy  
- seaborn
- matplotlib

