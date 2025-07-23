import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def ward_cluster(X: np.ndarray, n_clusters: int, **kwargs):
    """
    Fit Ward-linkage hierarchical clustering for a chosen number of clusters.
    Returns the fitted model and the cluster labels.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", **kwargs)
    return model, model.fit_predict(X)


def plot_silhouette_vs_k(X: np.ndarray, max_k: int = 10):
    """
    Compute & plot silhouette scores for k = 2 … max_k.
    Pick the k with the highest silhouette (or where it levels off).
    """
    ks = range(2, max_k + 1)
    sil_scores = []

    for k in ks:
        _, labels = ward_cluster(X, n_clusters=k)
        sil_scores.append(silhouette_score(X, labels))

    plt.plot(ks, sil_scores, marker="o")
    plt.title("Silhouette vs. Number of Clusters (Ward linkage)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.show()
