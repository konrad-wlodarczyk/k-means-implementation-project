import numpy as np
import matplotlib.pyplot as plt
from kmeans_project import KMeans

rng = np.random.default_rng(42)

cluster_1 = rng.normal(loc=[-5, 5], scale=1.0, size=(100, 2))
cluster_2 = rng.normal(loc=[0, 5], scale=1.0, size=(100, 2))
cluster_3 = rng.normal(loc=[5, 2], scale=1.0, size=(100, 2))

X = np.vstack([cluster_1, cluster_2, cluster_3])

model = KMeans(n_clusters=3, init="k-means++", random_state=0)
model.fit(X)
labels = model.labels_
centroids = model.centroids

print("Centroids:\n", centroids)
print("Inertia:", model.inertia_)
print("Iterations:", model.n_iter_)

plt.figure(figsize=(6, 8))

plt.scatter(
    X[:, 0], X[:, 1],
    c = labels,
    s = 40,
    cmap="viridis",
    alpha=0.7,
    label = "Samples"
)

plt.scatter(
    centroids[:, 0], centroids[:, 1],
    c="red",
    s=200,
    marker="X",
    edgecolor="black",
    linewidth=2,
    label="Centroids"
)

plt.title("Clustering Result – Custom KMeans Implementation")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)

plt.show()