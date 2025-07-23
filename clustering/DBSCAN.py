import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def dbscan(X: np.ndarray, eps: float, min_samples: int = 4, **kwargs):
    """
    Run DBSCAN and return the fitted estimator and labels.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    eps : float
        Radius of the neighborhood to form a core point.
    min_samples : int, default=4
        Minimum #points to form a dense region.
    kwargs : passed straight to sklearn.cluster.DBSCAN
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    return model, model.fit_predict(X)


def plot_k_distance(X: np.ndarray, min_samples: int = 4):
    """
    Plot the sorted k‑distance graph (a.k.a. "elbow for DBSCAN").
    The knee of the curve is a good guess for `eps`.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    min_samples : int, default=4
        k in the k‑distance plot (typically matches the DBSCAN min_samples).
    """
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)
    distances, _ = nbrs.kneighbors(X)

    k_distances = np.sort(distances[:, -1])  # distance to k‑th NN
    plt.plot(k_distances, marker=".")
    plt.title(f"{min_samples}‑NN distance plot\n(choose knee ≈ eps)")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"Distance to {min_samples}‑th NN")
    plt.grid(alpha=0.3)
    plt.show()
