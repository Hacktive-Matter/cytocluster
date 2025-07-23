import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def pca(X: np.ndarray, n_components=2):
    """
    Perform PCA on the input data.

    Parameters:
    -----------
    X : np.ndarray
        Input data (should be standardized)
    n_components : int
        Number of principal components to return

    Returns:
    --------
    np.ndarray : PCA transformed data
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca


def plot_explained_var(X: np.ndarray):
    """Perform PCA and return explained variance ratio plot."""

    pca = PCA()
    X_pca = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_

    # Plot explained variance
    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(
        range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
        "bo-",
    )
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot")

    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, len(pca.explained_variance_ratio_) + 1),
        np.cumsum(pca.explained_variance_ratio_),
        "ro-",
    )
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance")
    plt.axhline(y=0.95, color="k", linestyle="--", alpha=0.7, label="95%")
    plt.legend()

    plt.tight_layout()
    return fig
