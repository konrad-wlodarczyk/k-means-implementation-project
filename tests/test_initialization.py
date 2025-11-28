import numpy as np 
from kmeans_project.kmeans import KMeans

def test_random_initialization_shape():
    X = np.random.rand(100, 4)
    km = KMeans(n_clusters=3, init="random", random_state=42)
    centroids = km._initialize_centroids(X)
    assert centroids.shape == (3, 4)
    for c in centroids:
        assert any((c == x). all() for x in X)
        
def test_random_initialization_reproducibility():
    X = np.random.rand(50, 2)
    km1 = KMeans(n_clusters=3, init="random", random_state=123)
    km2 = KMeans(n_clusters=3, init="random", random_state=123)
    c1 = km1._initialize_centroids(X)
    c2 = km2._initialize_centroids(X)
    assert np.allclose(c1, c2)
    
def test_kmeans_plus_plus_initialization():
    X = np.random.rand(50, 2)
    km = KMeans(n_clusters=3, init="k-means++", random_state=0)
    centroids = km._initialize_centroids(X)
    assert centroids.shape == (3, 2)
    for c in centroids:
        assert any((c == x).all() for x in X)