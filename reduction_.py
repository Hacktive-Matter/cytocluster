import numpy as np
from sklearn.manifold import TSNE, trustworthiness
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import time


def optimize_tsne_params(X, subset_size=500, param_grid=None, verbose=True):
    """
    Find optimal t-SNE parameters using trustworthiness score
    """
    if param_grid is None:
        param_grid = {
            "perplexity": [10, 30, 50, 100],
            "learning_rate": [200, 500, 1000],
            "n_iter": [1000, 2000],
        }

    # Use subset for efficiency
    n_samples = min(subset_size, len(X))
    X_subset = X[:n_samples]

    best_params = None
    best_score = -np.inf
    results = []

    if verbose:
        print(
            f"Testing {len(list(ParameterGrid(param_grid)))} parameter combinations..."
        )

    for i, params in enumerate(ParameterGrid(param_grid)):
        try:
            start_time = time.time()
            tsne = TSNE(n_components=2, random_state=42, **params)
            X_tsne_temp = tsne.fit_transform(X_subset)

            # Calculate trustworthiness score
            score = trustworthiness(
                X_subset, X_tsne_temp, n_neighbors=min(10, n_samples - 1)
            )

            elapsed_time = time.time() - start_time

            results.append({"params": params, "score": score, "time": elapsed_time})

            if score > best_score:
                best_score = score
                best_params = params

            if verbose:
                print(
                    f"  {i+1}: {params} -> Score: {score:.3f} (Time: {elapsed_time:.1f}s)"
                )

        except Exception as e:
            if verbose:
                print(f"  {i+1}: {params} -> Failed: {str(e)}")

    if verbose:
        print(f"\nBest parameters: {best_params}")
        print(f"Best trustworthiness score: {best_score:.3f}")

    return best_params, results


def _tSNE(X: np.ndarray, n_components=2, optimize_params=False, **kwargs):
    """
    Perform t-SNE with optional parameter optimization

    Parameters:
    -----------
    X : np.ndarray
        Input data (should be standardized)
    n_components : int
        Number of components for output
    optimize_params : bool
        Whether to run parameter optimization first
    **kwargs : additional parameters for TSNE or optimization

    Returns:
    --------
    np.ndarray : t-SNE transformed data
    """
    if optimize_params:
        print("Optimizing t-SNE parameters...")
        best_params, _ = optimize_tsne_params(X, **kwargs)
        # Remove optimization-specific kwargs
        tsne_params = {k: v for k, v in best_params.items()}
        tsne_params["n_components"] = n_components
        tsne_params["random_state"] = 42
    else:
        # Use default or provided parameters
        tsne_params = {
            "n_components": n_components,
            "perplexity": kwargs.get("perplexity", 30),
            "learning_rate": kwargs.get("learning_rate", 200),
            "n_iter": kwargs.get("n_iter", 1000),
            "random_state": 42,
        }

    tsne = TSNE(**tsne_params)
    X_tsne = tsne.fit_transform(X)
    return X_tsne


def _PCA(X: np.ndarray, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca


if __name__ == "__main__":
    # Load and prepare data
    df = pd.read_csv("data/BARCODE_Higher_filament_930am.csv")
    df.drop(columns=["Filename", "Channel", "Flags"], inplace=True, errors="ignore")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[df.columns])

    # Option 1: Use with parameter optimization
    print("Running t-SNE with parameter optimization...")
    X_tsne_optimized = _tSNE(X_scaled, optimize_params=True)
    print("Optimized t-SNE result shape:", X_tsne_optimized.shape)

    # Option 2: Use with default parameters
    print("\nRunning t-SNE with default parameters...")
    X_tsne_default = _tSNE(X_scaled)
    print("Default t-SNE result shape:", X_tsne_default.shape)

    # Option 3: Use with custom parameters
    print("\nRunning t-SNE with custom parameters...")
    X_tsne_custom = _tSNE(X_scaled, perplexity=50, learning_rate=500, n_iter=2000)
    print("Custom t-SNE result shape:", X_tsne_custom.shape)

    # PCA for comparison
    X_pca = _PCA(X_scaled)
    print("PCA result shape:", X_pca.shape)

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].scatter(X_tsne_optimized[:, 0], X_tsne_optimized[:, 1], alpha=0.7, s=20)
    axes[0, 0].set_title("t-SNE (Optimized Parameters)")

    axes[0, 1].scatter(X_tsne_default[:, 0], X_tsne_default[:, 1], alpha=0.7, s=20)
    axes[0, 1].set_title("t-SNE (Default Parameters)")

    axes[1, 0].scatter(X_tsne_custom[:, 0], X_tsne_custom[:, 1], alpha=0.7, s=20)
    axes[1, 0].set_title("t-SNE (Custom Parameters)")

    axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=20)
    axes[1, 1].set_title("PCA")

    for ax in axes.flat:
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    plt.tight_layout()
    plt.show()
