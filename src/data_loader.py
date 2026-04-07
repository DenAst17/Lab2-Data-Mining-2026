import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.config import (
    RAW_DATA_PATH, PROCESSED_PATH, SCALED_PATH, BINNED_PATH,
    AUDIO_FEATURES, ASSOC_FEATURES, GENRE_COL,
    BIN_EDGES_01, BIN_LABELS_01,
    LOUDNESS_EDGES, LOUDNESS_LABELS, TOP_N_GENRES, RANDOM_STATE
)


def load_spotify_raw(path=None):
    """Load raw Spotify dataset from CSV. Auto-downloads via kagglehub if missing."""
    path = path or RAW_DATA_PATH
    if not os.path.exists(path):
        try:
            import kagglehub, shutil, glob
            print("Датасет не знайдено локально. Завантаження з Kaggle...")
            dl_path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
            for f in glob.glob(os.path.join(dl_path, "**", "dataset.csv"), recursive=True):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                shutil.copy(f, path)
                print(f"Завантажено та збережено: {path}")
                break
        except Exception as e:
            raise FileNotFoundError(
                f"Dataset not found at {path} and auto-download failed: {e}\n"
                "Download from: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset\n"
                "Place 'dataset.csv' into the data/ folder."
            )
    return pd.read_csv(path)


def preprocess(df):
    """Clean the Spotify dataset: drop duplicates, nulls, select columns."""
    cols_keep = ['track_id', 'track_name', 'artists', 'track_genre'] + AUDIO_FEATURES
    df = df[cols_keep].copy()
    n_before = len(df)
    df = df.dropna(subset=AUDIO_FEATURES + [GENRE_COL])
    df = df.drop_duplicates(subset='track_id', keep='first')
    n_after = len(df)
    print(f"Очищення: {n_before} → {n_after} записів "
          f"(видалено {n_before - n_after} дублікатів/пропусків)")
    return df.reset_index(drop=True)


def get_top_genres(df, n=TOP_N_GENRES):
    """Return list of top N genres by frequency."""
    return df[GENRE_COL].value_counts().head(n).index.tolist()


def create_binned(df, top_genres=None):
    """Discretize audio features + genre for association rule mining."""
    if top_genres is None:
        top_genres = get_top_genres(df)

    binned = pd.DataFrame()

    for feat in ASSOC_FEATURES:
        binned[feat] = pd.cut(
            df[feat], bins=BIN_EDGES_01, labels=BIN_LABELS_01, include_lowest=True
        )

    binned['loudness'] = pd.cut(
        df['loudness'], bins=LOUDNESS_EDGES, labels=LOUDNESS_LABELS, include_lowest=True
    )

    genre_mapped = df[GENRE_COL].where(df[GENRE_COL].isin(top_genres), 'Other')
    binned[GENRE_COL] = genre_mapped

    return binned


def create_scaled(df, features=None):
    """StandardScaler on audio features. Returns (scaled_df, scaler)."""
    features = features or AUDIO_FEATURES
    scaler = StandardScaler()
    scaled_arr = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled_arr, columns=features, index=df.index)
    return scaled_df, scaler


def subsample(df, n, seed=RANDOM_STATE):
    """Consistent random subsampling."""
    if n is None or n >= len(df):
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)
