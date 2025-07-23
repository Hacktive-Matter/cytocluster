from __future__ import annotations

import numpy as np
from sklearn.cluster import AffinityPropagation


def affinity(X: np.ndarray):

    model = AffinityPropagation(random_state=42).fit(X)
    return model, model.labels_

"""
Grid‑search helper for Affinity Propagation clustering.
Chooses the best (damping, preference) combination using multiple validity metrics.

Author: <your name> – 2025‑07‑22
"""


from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import euclidean_distances


def run_affinity_propagation(
    X: np.ndarray,
    *,
    damping: float,
    preference: float | None,
) -> Tuple[AffinityPropagation, np.ndarray]:
    """
    Fit an AffinityPropagation model and return it along with the labels.
    """
    model = AffinityPropagation(
        damping=damping,
        preference=preference,
        max_iter=500,
        convergence_iter=15,
        random_state=0,
    ).fit(X)

    return model, model.labels_


def _preference_grid(X: np.ndarray) -> List[float | None]:
    """
    Build a data‑driven grid of 'preference' values.

    The similarity matrix in AP is the *negative squared* Euclidean distance.
    We sample a few quantiles of that similarity distribution.
    """
    # pairwise squared distances, negate → similarities
    sim = -euclidean_distances(X, squared=True)
    flat_sim = sim[np.triu_indices_from(sim, k=1)]  # upper‑triangular (no diagonal)

    return [
        None,  # sklearn default: median of the similarities
        np.quantile(flat_sim, 0.10),
        np.quantile(flat_sim, 0.50),
        np.quantile(flat_sim, 0.90),
    ]


def optimize_ap_params(
    X: np.ndarray,
    grid_damping: List[float] | None = None,
    grid_preference: List[float | None] | None = None,
) -> Tuple[AffinityPropagation, np.ndarray]:
    """
    Exhaustive search over (damping, preference) and pick the model with the
    highest silhouette score, then Calinski‑Harabasz, then *lowest* Davies‑Bouldin.

    Returns
    -------
    best_model, best_labels
    """
    if grid_damping is None:
        grid_damping = [0.50, 0.70, 0.90, 0.95]
    if grid_preference is None:
        grid_preference = _preference_grid(X)

    print("Running grid search …\n")
    print(
        f"{'damp':<8}"
        f"{'pref':<12}"
        f"{'k':<5}"
        f"{'silhouette':<12}"
        f"{'calinski':<12}"
        f"{'davies':<12}"
    )

    results: List[Dict] = []

    for damp in grid_damping:
        for pref in grid_preference:
            model, labels = run_affinity_propagation(X, damping=damp, preference=pref)

            # # of clusters (excl. noise – AP has no noise label)
            k = len(set(labels))
            if k < 2 or k >= len(X):  # need at least 2 clusters, but not 1‑per‑sample
                continue

            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)

            print(
                f"{damp:<8.2f}"
                f"{str(round(pref, 3) if pref is not None else 'None'):<12}"
                f"{k:<5d}"
                f"{sil:<12.4f}"
                f"{ch:<12.2f}"
                f"{db:<12.4f}"
            )

            results.append(
                dict(
                    damping=damp,
                    preference=pref,
                    k=k,
                    silhouette=sil,
                    calinski=ch,
                    davies=db,
                )
            )

    if not results:
        raise RuntimeError("No parameter combination yielded ≥2 clusters.")

    # multi‑criteria sort: higher sil → higher CH → *lower* DB
    best = max(
        results,
        key=lambda r: (
            r["silhouette"],
            r["calinski"],
            -r["davies"],
        ),
    )

    best_params = {"damping": best["damping"], "preference": best["preference"]}

    print(
        "\nBest hyper‑parameters → "
        f"{best_params} | "
        f"silhouette={best['silhouette']:.4f}, "
        f"CH={best['calinski']:.1f}, "
        f"DB={best['davies']:.4f}, "
        f"k={best['k']}"
    )

    best_model, best_labels = run_affinity_propagation(X, **best_params)

    print("\n=== FINAL MODEL SUMMARY ===")
    print("Clusters found:", best["k"])
    cluster_sizes = np.bincount(best_labels)
    print("Cluster sizes:", cluster_sizes)

    return best_model, best_labels
