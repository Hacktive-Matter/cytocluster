import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def dbscan(X: np.ndarray, eps: float, min_samples: int = 5, **kwargs):
    """
    Fit DBSCAN with a user-chosen eps.
    Returns the fitted model and the cluster labels.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    return model, model.fit_predict(X)


def plot_k_distance(X: np.ndarray, k: int = 4):
    """
    Plot the sorted distance to the k-th nearest neighbour for every point.
    The 'knee' (sharp change in slope) is a good eps guess.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = nbrs.kneighbors(X)
    k_dists = np.sort(dists[:, k - 1])

    plt.plot(k_dists)
    plt.title(f"{k}-NN Distance Plot (choose eps at the knee)")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{k}-NN distance")
    plt.show()