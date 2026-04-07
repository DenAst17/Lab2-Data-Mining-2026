import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import SCATTER_SAMPLE, RANDOM_STATE, PALETTE

sns.set_style('whitegrid')


def _subsample_for_plot(X, labels, max_n=SCATTER_SAMPLE):
    """Subsample data for scatter plots to keep them readable."""
    if len(X) <= max_n:
        return X, labels
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X), max_n, replace=False)
    return X[idx] if isinstance(X, np.ndarray) else X.iloc[idx], labels[idx]


def plot_cluster_scatter_2d(X, labels, feature_names, title,
                            centers=None, ax=None, noise_label=-1):
    """2D scatter plot with cluster colors and optional centers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    X_arr = np.array(X)
    labels_arr = np.array(labels)
    X_plot, labels_plot = _subsample_for_plot(X_arr, labels_arr)

    unique_labels = sorted(set(labels_plot))
    for i, lab in enumerate(unique_labels):
        mask = labels_plot == lab
        if lab == noise_label:
            ax.scatter(X_plot[mask, 0], X_plot[mask, 1],
                       c='gray', alpha=0.2, s=8, label='Шум')
        else:
            color = PALETTE[i % len(PALETTE)]
            ax.scatter(X_plot[mask, 0], X_plot[mask, 1],
                       c=color, alpha=0.4, s=12,
                       label=f'Кластер {lab}')

    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1],
                   c='red', marker='X', s=150, edgecolors='black',
                   linewidths=1, zorder=10, label='Центри')

    ax.set_xlabel(feature_names[0], fontsize=11)
    ax.set_ylabel(feature_names[1], fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    return ax


def plot_elbow(k_range, inertias, title, ax=None):
    """Elbow curve for k selection."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('k', fontsize=11)
    ax.set_ylabel('Інерція (SSE)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(list(k_range))
    return ax


def plot_silhouette_curve(k_range, scores, title, ax=None):
    """Silhouette score vs k."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), scores, 'rs-', linewidth=2, markersize=8)
    best_k = list(k_range)[np.argmax(scores)]
    ax.axvline(x=best_k, color='green', linestyle='--', alpha=0.7,
               label=f'Найкраще k={best_k}')
    ax.set_xlabel('k', fontsize=11)
    ax.set_ylabel('Silhouette Score', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(list(k_range))
    ax.legend(fontsize=10)
    return ax


def plot_dendrogram(Z, title, ax=None, truncate_p=30):
    """Dendrogram visualization."""
    from scipy.cluster.hierarchy import dendrogram
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, truncate_mode='lastp', p=truncate_p, ax=ax,
               leaf_rotation=90, leaf_font_size=8)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Зразки', fontsize=11)
    ax.set_ylabel('Відстань', fontsize=11)
    return ax


def plot_k_distance(distances, title, ax=None):
    """K-distance graph for DBSCAN eps selection."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(distances)), np.sort(distances)[::-1], linewidth=1.5)
    ax.set_xlabel('Точки (відсортовані)', fontsize=11)
    ax.set_ylabel('k-відстань', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    return ax


def plot_comparison_heatmap(pivot_df, metric_name, cmap='RdYlGn', ax=None):
    """Heatmap of algorithms x pairs."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap=cmap,
                ax=ax, linewidths=0.5, cbar_kws={'label': metric_name})
    ax.set_title(f'{metric_name}: алгоритм × пара', fontsize=13, fontweight='bold')
    ax.set_ylabel('')
    return ax


def plot_comparison_bars(results_df, metric, title, ax=None):
    """Grouped bar chart for algorithm comparison."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    pivot = results_df.pivot(index='pair', columns='algorithm', values=metric)
    pivot.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='black', linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel(metric, fontsize=11)
    ax.set_xlabel('')
    ax.legend(title='Алгоритм', fontsize=9)
    ax.tick_params(axis='x', rotation=15)
    return ax


def plot_genre_distribution(genre_counts, title, ax=None, top_n=20):
    """Horizontal bar chart of genre frequencies."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    data = genre_counts.head(top_n)
    ax.barh(range(len(data)), data.values, color=sns.color_palette('viridis', len(data)))
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Кількість треків', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    return ax
