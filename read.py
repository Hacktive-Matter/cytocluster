import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from dataset_4 import get_c2_data

if __name__ == "__main__":
    # Read the CSV file into a DataFrame
    from dataset_4 import get_c1_data
    
    df, X_scaled = get_c1_data()

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    print(f"Original dimensions: {X_scaled.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")

    # Plot explained variance
    plt.figure(figsize=(12, 4))

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
    plt.show()
