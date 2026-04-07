# Lab 2: Association Rules & Clustering — Spotify Tracks Dataset

## Dataset
[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) — ~114,000 tracks, 125 genres.

### Audio Features Used
| Feature | Range | Description |
|---------|-------|-------------|
| danceability | 0-1 | How suitable for dancing |
| energy | 0-1 | Intensity and activity |
| valence | 0-1 | Musical positivity (happiness) |
| acousticness | 0-1 | Acoustic vs electronic |
| speechiness | 0-1 | Presence of spoken words |
| instrumentalness | 0-1 | Vocal vs instrumental |
| tempo | BPM | Beats per minute |
| loudness | dB | Overall loudness |

## Project Structure
```
├── src/                    # Reusable Python modules
│   ├── config.py           # Constants and parameters
│   ├── data_loader.py      # Data loading, cleaning, binning, scaling
│   ├── visualization.py    # Plot functions (scatter, radar, heatmap, etc.)
│   ├── evaluation.py       # Clustering metrics and results I/O
│   └── clustering.py       # Algorithm wrappers (k-Means, k-Medians, etc.)
├── data/                   # Generated data files
├── results/                # Clustering results and association rules output
└── 01-09 notebooks         # Sequential analysis pipeline
```

## Notebooks
| # | File | Description |
|---|------|-------------|
| 1 | `01_preprocessing.ipynb` | Data loading, EDA, cleaning, discretization |
| 2 | `02_apriori.ipynb` | Apriori algorithm for association rules |
| 3 | `03_fp_growth.ipynb` | FP-Growth algorithm + comparison with Apriori |
| 4 | `04_kmeans.ipynb` | k-Means clustering |
| 5 | `05_kmedians.ipynb` | k-Medians clustering (pyclustering) |
| 6 | `06_hierarchical.ipynb` | Hierarchical Agglomerative clustering |
| 7 | `07_knn_clustering.ipynb` | KNN-graph based clustering |
| 8 | `08_dbscan.ipynb` | DBSCAN density-based clustering |
| 9 | `09_comparison.ipynb` | Comparative analysis of all algorithms |

## Setup
```bash
pip install -r requirements.txt
```

### Dataset Download
1. Download from [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
2. Place `dataset.csv` into the `data/` folder

Or via kagglehub:
```python
import kagglehub
kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
```

Run notebooks sequentially: `01_preprocessing.ipynb` first, then any others.
