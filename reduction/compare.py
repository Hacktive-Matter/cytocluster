import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

# --------------------------------------------------------------------------- #
#                               metric helpers                                #
# --------------------------------------------------------------------------- #
def distance_spearman(X_hi: np.ndarray, X_emb: np.ndarray,
                      sample_size: int | None = None) -> float:
    """
    Spearman rank-correlation between pair‑wise distances in the original
    space and in the embedding.  If sample_size is given, subsample points
    to keep memory at O(sample_size²).
    """
    if sample_size and sample_size < X_hi.shape[0]:
        rng = np.random.default_rng(0)
        idx = rng.choice(X_hi.shape[0], size=sample_size, replace=False)
        X_hi, X_emb = X_hi[idx], X_emb[idx]

    D_hi   = pairwise_distances(X_hi, metric="euclidean")
    D_emb  = pairwise_distances(X_emb, metric="euclidean")

    # Flatten the upper triangles (excluding diagonal) for correlation
    triu_mask = np.triu_indices_from(D_hi, k=1)
    rho, _ = spearmanr(D_hi[triu_mask], D_emb[triu_mask])
    return float(rho)          # ρ ∈ [‑1, 1]; +1 is perfect


def knn_overlap(X_hi: np.ndarray, X_emb: np.ndarray, k: int = 10) -> float:
    """Average fractional overlap of k‑NN sets."""
    nn_hi  = NearestNeighbors(n_neighbors=k + 1).fit(X_hi)
    nn_emb = NearestNeighbors(n_neighbors=k + 1).fit(X_emb)

    idx_hi  = nn_hi.kneighbors(return_distance=False)[:, 1:]
    idx_emb = nn_emb.kneighbors(return_distance=False)[:, 1:]

    return np.mean([
        len(set(a).intersection(b)) / k
        for a, b in zip(idx_hi, idx_emb)
    ])


# --------------------------------------------------------------------------- #
#                             evaluation driver                               #
# --------------------------------------------------------------------------- #
def evaluate_embeddings(X_hi, X_pca, X_kpca, k: int = 10,
                        sample_size: int | None = 3000):
    """Return metrics & declare the ‘better’ embedding."""
    # -- distance preservation -------------------------------------------------
    rho_pca  = distance_spearman(X_hi, X_pca,  sample_size)
    rho_kpca = distance_spearman(X_hi, X_kpca, sample_size)

    # -- k‑NN overlap ----------------------------------------------------------
    knn_pca  = knn_overlap(X_hi, X_pca,  k=k)
    knn_kpca = knn_overlap(X_hi, X_kpca, k=k)

    # -- rank each metric (higher is better) -----------------------------------
    ranks_rho = {
        "PCA":  1 if rho_pca  > rho_kpca else (1.5 if rho_pca == rho_kpca else 2),
        "kPCA": 1 if rho_kpca > rho_pca  else (1.5 if rho_pca == rho_kpca else 2),
    }
    ranks_knn = {
        "PCA":  1 if knn_pca  > knn_kpca else (1.5 if knn_pca == knn_kpca else 2),
        "kPCA": 1 if knn_kpca > knn_pca  else (1.5 if knn_pca == knn_kpca else 2),
    }
    total_rank = {m: ranks_rho[m] + ranks_knn[m] for m in ["PCA", "kPCA"]}
    best = min(total_rank, key=total_rank.get)

    # -- report ----------------------------------------------------------------
    print("Distance preservation (Spearman ρ, ↑ better):")
    print(f"  PCA  : {rho_pca:.4f}")
    print(f"  kPCA : {rho_kpca:.4f}\n")

    print(f"K‑NN overlap (k={k}, ↑ better):")
    print(f"  PCA  : {knn_pca:.4f}")
    print(f"  kPCA : {knn_kpca:.4f}\n")

    if total_rank["PCA"] != total_rank["kPCA"]:
        print(f"🏆  **{best} embedding wins** (lower rank total)")
    else:                                 # exact tie → choose the one with higher ρ
        best = "PCA" if rho_pca >= rho_kpca else "kPCA"
        print("⚖️  Metrics tie exactly – prefer the one with higher distance correlation.")
        print(f"🏆  **{best} embedding wins**")

    return {
        "distance_spearman": {"PCA": rho_pca, "kPCA": rho_kpca},
        "knn_overlap":       {"PCA": knn_pca, "kPCA": knn_kpca},
        "best": best,
    }


# --------------------------------------------------------------------------- #
#                            example standalone run                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Replace these with your data
    # X_c1 = np.load("X_c1.npy")
    # X_c1_PCA = np.load("X_c1_PCA.npy")
    # X_c1_kPCA = np.load("X_c1_kPCA.npy")
    results = evaluate_embeddings(X_c1, X_c1_PCA, X_c1_kPCA,
                                  k=10, sample_size=3000)