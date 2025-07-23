import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler


def kpca(X: np.ndarray, n_components=2):
    """Perform Kernel PCA and return the transformed data."""
    kpca = KernelPCA(n_components=n_components, kernel="rbf", random_state=42)
    return kpca.fit_transform(X)


def plot_kpca_mse_vs_pc(X: np.ndarray):

    fig = plt.figure(figsize=(8, 5))

    mse_vals = []
    for i in range(15):
        kpca = KernelPCA(
            n_components=i + 1,
            kernel="rbf",
            fit_inverse_transform=True,
            random_state=42,
        )
        X_kpca = kpca.fit_transform(X)
        X_rec = kpca.inverse_transform(X_kpca)
        mse = np.mean((X - X_rec) ** 2)
        mse_vals.append(mse)

    plt.plot(range(1, 16), mse_vals, marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Reconstruction MSE")
    plt.title("Kernel PCA - MSE vs. Number of Components")
    plt.tight_layout()
    plt.show()
