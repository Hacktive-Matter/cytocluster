import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def kmeans(X: np.ndarray, k: int, **kwargs):
    kmeans = KMeans(n_clusters=k, n_init=10, **kwargs)
    return kmeans, kmeans.fit_predict(X)


def plot_kmeans_elbow(X: np.ndarray, max_k: int = 10):
    inertias = []
    for k in range(1, max_k + 1):
        k_means, labels = kmeans(X, k)
        inertias.append(k_means.inertia_)

    plt.plot(range(1, max_k + 1), inertias, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.show()
