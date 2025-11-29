import numpy as np
import matplotlib.pyplot as plt
from kmeans_project import KMeans

rng = np.random.default_rng(42)

cluster_1 = rng.normal(loc=[-5, 5], scale=1.0, size=(100, 2))
cluster_2 = rng.normal(loc=[0, 5], scale=1.0, size=(100, 2))
cluster_3 = rng.normal(loc=[5, 2], scale=1.0, size=(100, 2))

X = np.vstack([cluster_1, cluster_2, cluster_3])