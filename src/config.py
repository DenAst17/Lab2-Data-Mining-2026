import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

RAW_DATA_PATH = os.path.join(DATA_DIR, "dataset.csv")
PROCESSED_PATH = os.path.join(DATA_DIR, "data_processed.csv")
SCALED_PATH = os.path.join(DATA_DIR, "data_scaled.csv")
BINNED_PATH = os.path.join(DATA_DIR, "data_binned.csv")

TOP_PAIRS_PATH = os.path.join(RESULTS_DIR, "top_pairs.json")
APRIORI_TIME_PATH = os.path.join(RESULTS_DIR, "apriori_time.csv")

# --- Features ---
AUDIO_FEATURES = [
    'danceability', 'energy', 'valence', 'acousticness',
    'speechiness', 'loudness'
]
ASSOC_FEATURES = [
    'danceability', 'energy', 'valence', 'acousticness',
    'speechiness'
]
GENRE_COL = 'track_genre'

# --- Association Rules ---
TOP_N_GENRES = 15
MIN_SUPPORT = 0.03
MIN_LIFT = 1.0
N_TOP_PAIRS = 4

# Binning (0-1 features)
BIN_EDGES_01 = [0, 0.33, 0.66, 1.01]
BIN_LABELS_01 = ['Low', 'Medium', 'High']
# Loudness
LOUDNESS_EDGES = [-60, -20, -10, 5]
LOUDNESS_LABELS = ['Quiet', 'Medium', 'Loud']

# --- Clustering ---
K_RANGE = range(2, 11)
RANDOM_STATE = 42
SAMPLE_SIZES = {
    'kmeans': 20_000,
    'kmedians': 10_000,
    'hierarchical': 10_000,
    'knn': 10_000,
    'dbscan': 20_000,
}
DBSCAN_MIN_SAMPLES = 10
KNN_N_NEIGHBORS = 10

# --- Visualization ---
SCATTER_SAMPLE = 8_000
FIGSIZE_WIDE = (20, 6)
FIGSIZE_SQUARE = (8, 8)
PALETTE = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12',
           '#1abc9c', '#e67e22', '#34495e', '#e91e63', '#00bcd4']
