import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics import silhouette_score
from src.config import RANDOM_STATE, DBSCAN_MIN_SAMPLES, KNN_N_NEIGHBORS


def run_kmeans(X, k, random_state=RANDOM_STATE):
    """Run k-Means. Returns (labels, model)."""
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return labels, model


def run_kmedians(X, k, random_state=RANDOM_STATE, max_iter=100):
    """Run k-Medians (L1/Manhattan distance, median centers). Returns (labels, medians)."""
    X_arr = np.array(X, dtype=float)
    rng = np.random.RandomState(random_state)

    centers = [X_arr[rng.randint(len(X_arr))]]
    for _ in range(1, k):
        dists = np.min([np.sum(np.abs(X_arr - c), axis=1) for c in centers], axis=0)
        probs = dists / dists.sum()
        centers.append(X_arr[rng.choice(len(X_arr), p=probs)])
    centers = np.array(centers)

    for _ in range(max_iter):
        # Assign to nearest center (Manhattan distance)
        dists = np.array([np.sum(np.abs(X_arr - c), axis=1) for c in centers])
        labels = np.argmin(dists, axis=0)
        # Update centers as median
        new_centers = np.array([np.median(X_arr[labels == i], axis=0)
                                if np.any(labels == i) else centers[i]
                                for i in range(k)])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    return labels, centers


def run_hierarchical(X, k, linkage='ward'):
    """Run Agglomerative Clustering. Returns (labels, None)."""
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(X)
    return labels, None


def run_knn_clustering(X, k, n_neighbors=KNN_N_NEIGHBORS):
    """KNN-graph based Agglomerative Clustering. Returns (labels, None)."""
    connectivity = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=False)
    model = AgglomerativeClustering(n_clusters=k, connectivity=connectivity, linkage='ward')
    labels = model.fit_predict(X)
    return labels, None


def run_dbscan(X, eps, min_samples=DBSCAN_MIN_SAMPLES):
    """Run DBSCAN. Returns (labels, model)."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels, model


def find_best_dbscan_eps(X, eps_range, min_samples=DBSCAN_MIN_SAMPLES):
    """Grid search eps for DBSCAN. Returns (best_eps, best_silhouette, all_results)."""
    results = []
    for eps in eps_range:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct = (labels == -1).sum() / len(labels) * 100

        if n_clusters >= 2:
            mask = labels != -1
            sil = silhouette_score(X[mask], labels[mask])
        else:
            sil = -1

        results.append({
            'eps': eps, 'n_clusters': n_clusters,
            'noise_pct': round(noise_pct, 1), 'silhouette': round(sil, 4)
        })

    best = max(results, key=lambda r: r['silhouette'])
    return best['eps'], best['silhouette'], pd.DataFrame(results)


def compute_k_distances(X, k=None):
    """Compute k-distances for DBSCAN eps selection."""
    if k is None:
        k = min(2 * X.shape[1], len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    return np.sort(distances[:, -1])[::-1]


def get_cluster_profiles(X_original, labels, feature_names):
    """Compute mean feature values per cluster (on original unscaled data).

    Returns DataFrame: (n_clusters x n_features) with cluster IDs as index.
    """
    df = pd.DataFrame(X_original, columns=feature_names)
    df['cluster'] = labels
    df = df[df['cluster'] != -1]
    return df.groupby('cluster')[feature_names].mean()


