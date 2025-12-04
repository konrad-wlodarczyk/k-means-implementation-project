import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as SKKMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

from kmeans_project import KMeans

from scipy.optimize import linear_sum_assignment

def centroid_alignment_error(C1, C2):
    """Compute minimal centroid matchin errror between two sets of ceontroids
        using the Hungarian Algorithm.
    """    
    cost = np.linalg.norm(C1[:, None, :] - C2[None, :, :], axis=2)
    r, c = linear_sum_assignment(cost)
    
    return cost[r, c].sum()

def compare_kmeans(X, n_clusters=3, random_state=42):
    # Fit custom Kmeans
    custom_km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        random_state=random_state,
        max_iter=300,
    )
    custom_km.fit(X)
    
    # Fit sklearn Kmeans
    sk_km = SKKMeans(
        n_clusters=n_clusters,
        init="k-means++",
        random_state=random_state,
        max_iter=300,
        n_init=1,
    )
    sk_km.fit(X)
    
    # Metrics comparison
    intertia_diff = abs(custom_km.inertia_ - sk_km.inertia_)
    silhouette_custom = silhouette_score(X, custom_km.labels_)
    silhouette_sklearn = silhouette_score(X, sk_km.labels_)
    centroid_error = centroid_alignment_error(custom_km.centroids, sk_km.cluster_centers_)
    
    print("=== Numerical Comparison ===")
    print(f"Custom inertia:     {custom_km.inertia_:.4f}")
    print(f"sklearn inertia:    {sk_km.inertia_:.4f}")
    print(f"Inertia difference: {intertia_diff:.6f}")
    print()
    print(f"Custom silhouette:  {silhouette_custom:.4f}")
    print(f"sklearn silhouette: {silhouette_sklearn:.4f}")
    print()
    print(f"Centroid alignment error: {centroid_error:.6f}")
    print(f"Iterations (custom vs sklearn): {custom_km.n_iter_} vs {sk_km.n_iter_}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(X[:, 0], X[:, 1], c=custom_km.labels_, s=20, alpha=0.6)
    axes[0].scatter(custom_km.centroids[:, 0], custom_km.centroids[:, 1],
                    c="black", s=100, marker="X")
    axes[0].set_title("Custom KMeans")

    axes[1].scatter(X[:, 0], X[:, 1], c=sk_km.labels_, s=20, alpha=0.6)
    axes[1].scatter(sk_km.cluster_centers_[:, 0], sk_km.cluster_centers_[:, 1],
                    c="black", s=100, marker="X")
    axes[1].set_title("Scikit-learn KMeans")

    plt.show()
    
X, y_true = make_blobs(
    n_samples=1000,
    centers=45,
    cluster_std=0.60,
    random_state=42,
)

compare_kmeans(X, n_clusters=15, random_state=42)