import time
import os
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from src.config import RESULTS_DIR


def compute_metrics(X, labels):
    """Compute clustering quality metrics. Returns dict."""
    unique = set(labels)
    unique.discard(-1)
    if len(unique) < 2:
        return {
            'silhouette': np.nan,
            'davies_bouldin': np.nan,
        }
    mask = np.array(labels) != -1
    X_clean = np.array(X)[mask]
    labels_clean = np.array(labels)[mask]
    return {
        'silhouette': silhouette_score(X_clean, labels_clean),
        'davies_bouldin': davies_bouldin_score(X_clean, labels_clean),
    }


def timed_fit_predict(fit_predict_fn, X, **kwargs):
    """Time a clustering function. Returns (labels, elapsed_seconds)."""
    start = time.time()
    labels = fit_predict_fn(X, **kwargs)
    elapsed = time.time() - start
    return labels, elapsed


def find_best_k(X, k_range, fit_fn):
    """Find best k by Silhouette score.

    fit_fn(X, k) -> labels
    Returns (best_k, silhouette_scores, inertias_or_none).
    """
    scores = []
    for k in k_range:
        labels = fit_fn(X, k)
        s = silhouette_score(X, labels)
        scores.append(s)
    best_idx = np.argmax(scores)
    best_k = list(k_range)[best_idx]
    return best_k, scores


def build_results_row(algorithm, pair_name, k, metrics, elapsed):
    """Build standardized results dict."""
    return {
        'algorithm': algorithm,
        'pair': pair_name,
        'k': k,
        'silhouette': metrics['silhouette'],
        'davies_bouldin': metrics['davies_bouldin'],
        'time_sec': round(elapsed, 4),
    }


def save_results(results_list, filename):
    """Save results list to CSV in results/ directory."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    pd.DataFrame(results_list).to_csv(path, index=False)
    print(f"Результати збережено: {path}")
    return path


def load_all_results():
    """Load and concatenate all results_*.csv files."""
    frames = []
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.startswith('results_') and f.endswith('.csv'):
            frames.append(pd.read_csv(os.path.join(RESULTS_DIR, f)))
    if not frames:
        raise FileNotFoundError("No results files found in results/")
    return pd.concat(frames, ignore_index=True)
