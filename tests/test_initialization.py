import numpy as np 
from kmeans_project.kmeans import KMeans

def test_random_initialization_shape():
    """
    Test that random initialization returns centroids of the correct shape
    and that all centroids exist in the original dataset.

    Args:
        None

    Raises:
        AssertionError: If the shape of centroids is incorrect
                        or if any centroid is not present in X.

    Returns:
        None
    """
    X = np.random.rand(100, 4)
    km = KMeans(n_clusters=3, init="random", random_state=42)
    centroids = km._initialize_centroids(X)
    assert centroids.shape == (3, 4)
    for c in centroids:
        assert any((c == x). all() for x in X)
        
def test_random_initialization_reproducibility():
    """
    Test that random initialization with the same random_state is reproducible.

    Args:
        None

    Raises:
        AssertionError: If centroids initialized with the same random_state
                        are not identical.

    Returns:
        None
    """
    X = np.random.rand(50, 2)
    km1 = KMeans(n_clusters=3, init="random", random_state=123)
    km2 = KMeans(n_clusters=3, init="random", random_state=123)
    c1 = km1._initialize_centroids(X)
    c2 = km2._initialize_centroids(X)
    assert np.allclose(c1, c2)
    
def test_kmeans_plus_plus_initialization():
    """
    Test that k-means++ initialization returns centroids of the correct shape
    and that all centroids exist in the original dataset.

    Args:
        None

    Raises:
        AssertionError: If the shape of centroids is incorrect
                        or if any centroid is not present in X.

    Returns:
        None
    """
    X = np.random.rand(50, 2)
    km = KMeans(n_clusters=3, init="k-means++", random_state=0)
    centroids = km._initialize_centroids(X)
    assert centroids.shape == (3, 2)
    for c in centroids:
        assert any((c == x).all() for x in X)
        
def test_kmeans_plus_plus_reproducibility():
    """
    Test that random initialization with the same random_state is reproducible.

    Args:
        None

    Raises:
        AssertionError: If centroids initialized with the same random_state
                        are not identical.

    Returns:
        None
    """
    X = np.random.rand(100, 3)
    km1 = KMeans(n_clusters=4, init="k-means++", random_state=123)
    km2 = KMeans(n_clusters=4, init="k-means++", random_state=123)
    c1 = km1._initialize_centroids(X)
    c2 = km2._initialize_centroids(X)
    assert np.allclose(c1, c2)
    
def test_initialize_centroids_single_cluster():
    """
    Test initialization when there is only one cluster (n_clusters=1).

    Args:
        None

    Raises:
        AssertionError: If the centroid shape is incorrect or
                        if the centroid does not match a sample from X.

    Returns:
        None
    """
    X = np.random.rand(20, 2)
    km = KMeans(n_clusters=1, init="random", random_state=0)
    centroids = km._initialize_centroids(X)
    assert centroids.shape == (1, 2)
    assert any(np.all(centroids[0] == x) for x in X)
    
def test_initialize_centroids_with_duplicates():
    """
    Test that initialization works correctly when X contains duplicate points.

    Args:
        None

    Raises:
        AssertionError: If centroids shape is incorrect or
                        if centroids do not match any samples in X.

    Returns:
        None
    """
    X = np.array([
        [1, 1],
        [1, 1],
        [2, 2],
        [3, 3]
    ])
    km = KMeans(n_clusters=2, init="random", random_state=42)
    centroids = km._initialize_centroids(X)
    assert centroids.shape == (2, 2)
    for c in centroids:
        assert any(np.all(c == x) for x in X)
        