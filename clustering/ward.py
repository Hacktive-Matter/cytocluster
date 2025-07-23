import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from data import get_data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


def ward_cluster(X: np.ndarray, n_clusters: int, **kwargs):
    """
    Run Ward‑linkage agglomerative clustering.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    n_clusters : int
        Desired number of clusters (cut level).
    kwargs : passed straight to sklearn.cluster.AgglomerativeClustering
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", **kwargs)
    return model, model.fit_predict(X)


def plot_dendrogram(X: np.ndarray, truncate_mode: str | None = None, p: int = 12):
    """
    Draw a Ward dendrogram so you can pick a cut (distance threshold)
    or count how many clusters to keep.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    truncate_mode : {"lastp", "level", None}, optional
        Passed to `scipy.cluster.hierarchy.dendrogram`.
        Use "lastp" to show only the last `p` merges.
    p : int, default=12
        Number of links to show when `truncate_mode="lastp"`.
    """
    Z = linkage(X, method="ward")
    dendrogram(
        Z, truncate_mode=truncate_mode, p=p, color_threshold=None, no_labels=True
    )
    plt.title("Ward dendrogram")
    plt.xlabel("Merged sample index or (cluster size)")
    plt.ylabel("Linkage distance")
    plt.grid(alpha=0.3)
    plt.show()
