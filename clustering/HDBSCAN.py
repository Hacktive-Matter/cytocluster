# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# from hdbscan import HDBSCAN
# from hdbscan.validity import validity_index


# def hdbscan(
#     X: np.ndarray,
#     min_cluster_size: int,
#     min_samples: int,
# ):
#     model = HDBSCAN(
#         min_cluster_size=min_cluster_size,
#         min_samples=min_samples,
#         metric="euclidean",
#         gen_min_span_tree=True,
#         core_dist_n_jobs=1,
#     )
#     return model, model.fit_predict(X)


# def optimize_hdbscan_params(X: np.ndarray):

#     results = []
#     grid_min_cluster_size = [5, 10, 20, 30]
#     grid_min_samples = [None, 5, 10, 20]

#     print("Running grid search …\n")
#     for mcs in grid_min_cluster_size:
#         for ms in grid_min_samples:

#             model, labels = hdbscan(X, mcs, ms)

#             # require at least 2 non‑noise clusters
#             n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#             if n_clusters < 2:
#                 continue

#             # count noise points
#             noise_cnt = np.count_nonzero(labels == -1)

#             # --- 3a. Quality metrics ---
#             dbcv = validity_index(X, labels)  # ∈ [-1, 1], higher better
#             if np.isnan(dbcv):  # happens if all noise
#                 continue

#             rel_val = model.relative_validity_
#             # average (size‑weighted) persistence
#             sizes = np.bincount(labels[labels >= 0])
#             avg_persist = np.average(model.cluster_persistence_, weights=sizes)

#             print(
#                 "{:<8} {:<8} {:<6} {:<8.4f} {:<10.4f} {:<10.4f}".format(
#                     mcs, str(ms), noise_cnt, dbcv, avg_persist, rel_val
#                 )
#             )

#             results.append(
#                 {
#                     "min_cluster_size": mcs,
#                     "min_samples": ms,
#                     "dbcv": dbcv,
#                     "persistence": avg_persist,
#                     "rel_validity": rel_val,
#                     "noise_cnt": noise_cnt,
#                 }
#             )

#     best = sorted(
#         results, key=lambda r: (-r["dbcv"], -r["persistence"], -r["rel_validity"])
#     )[0]

#     best_params = dict(
#         min_cluster_size=best["min_cluster_size"], min_samples=best["min_samples"]
#     )

#     print("\nBest hyper‑parameters →", best_params)
#     print(
#         "DBCV: {:.4f} | Avg. persistence: {:.4f} | Rel. validity: {:.4f}".format(
#             best["dbcv"], best["persistence"], best["rel_validity"]
#         )
#     )
#     print("Noise points in this best run:", best["noise_cnt"])

#     model, labels = hdbscan(X, **best_params)

#     noise_final = np.count_nonzero(labels == -1)
#     unique_clusters = set(labels) - {-1}

#     print("\n=== FINAL MODEL SUMMARY ===")
#     print("Clusters found (excl. noise):", len(unique_clusters))
#     print("Noise points (label = ‑1):   ", noise_final)

#     return model, labels

"""
Hyper-parameter search for HDBSCAN with several cluster-quality metrics.
Author: <your name> – 2025-07-22
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from hdbscan import HDBSCAN
from hdbscan.validity import validity_index


def run_hdbscan(
    X: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int | None,
) -> Tuple[HDBSCAN, np.ndarray]:
    """
    Fit an HDBSCAN model and return both the model and the assigned labels.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples × n_features).
    min_cluster_size : int
        HDBSCAN `min_cluster_size` hyper-parameter.
    min_samples : int | None
        HDBSCAN `min_samples` hyper-parameter.

    Returns
    -------
    model : HDBSCAN
        The fitted HDBSCAN instance.
    labels : np.ndarray
        Cluster labels (-1 denotes noise).
    """
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        gen_min_span_tree=True,
        core_dist_n_jobs=1,
    ).fit(X)

    return model, model.labels_


def optimize_hdbscan_params(
    X: np.ndarray,
    grid_min_cluster_size: List[int] | None = None,
    grid_min_samples: List[int | None] | None = None,
) -> Tuple[HDBSCAN, np.ndarray]:
    """
    Grid-search over (min_cluster_size, min_samples) and pick the model with the
    highest DBCV, then average persistence, then relative validity.

    Returns
    -------
    best_model : HDBSCAN
    best_labels : np.ndarray
    """
    if grid_min_cluster_size is None:
        grid_min_cluster_size = [5, 10, 20, 30]
    if grid_min_samples is None:
        grid_min_samples = [None, 5, 10, 20]

    print("Running grid search …\n")
    print(
        f"{'mcs':<8}{'ms':<8}{'noise':<8}{'DBCV':<12}"
        f"{'avg-persist':<14}{'rel-valid':<12}"
    )

    results: List[Dict] = []

    for mcs in grid_min_cluster_size:
        for ms in grid_min_samples:
            model, labels = run_hdbscan(X, min_cluster_size=mcs, min_samples=ms)

            # number of clusters (excluding noise)
            n_clusters = len(set(labels)) - (-1 in labels)
            if n_clusters < 2:
                continue  # ignore runs with <2 clusters

            noise_cnt = np.count_nonzero(labels == -1)

            # --- quality metrics
            dbcv = validity_index(X, labels)  # ∈ [-1, 1]; NaN if all noise
            if np.isnan(dbcv):
                continue

            rel_val = model.relative_validity_
            sizes = np.bincount(labels[labels >= 0])
            avg_persist = np.average(model.cluster_persistence_, weights=sizes)

            print(
                f"{mcs:<8}{str(ms):<8}{noise_cnt:<8}"
                f"{dbcv:<12.4f}{avg_persist:<14.4f}{rel_val:<12.4f}"
            )

            results.append(
                dict(
                    min_cluster_size=mcs,
                    min_samples=ms,
                    dbcv=dbcv,
                    persistence=avg_persist,
                    rel_validity=rel_val,
                    noise_cnt=noise_cnt,
                )
            )

    if not results:
        raise RuntimeError("No valid parameter combination produced ≥2 clusters.")

    # sort on multiple keys (descending)
    best = max(
        results,
        key=lambda r: (r["dbcv"], r["persistence"], r["rel_validity"]),
    )

    best_params = {
        "min_cluster_size": best["min_cluster_size"],
        "min_samples": best["min_samples"],
    }

    print(
        "\nBest hyper-parameters → "
        f"{best_params} | "
        f"DBCV={best['dbcv']:.4f}, "
        f"Avg. persistence={best['persistence']:.4f}, "
        f"Rel. validity={best['rel_validity']:.4f}, "
        f"Noise={best['noise_cnt']}"
    )

    best_model, best_labels = run_hdbscan(X, **best_params)

    print("\n=== FINAL MODEL SUMMARY ===")
    print("Clusters found (excl. noise):", len(set(best_labels) - {-1}))
    print("Noise points (label = -1):   ", np.count_nonzero(best_labels == -1))

    return best_model, best_labels
