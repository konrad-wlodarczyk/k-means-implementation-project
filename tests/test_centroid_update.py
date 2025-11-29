import numpy as np
from kmeans_project.kmeans import KMeans

def test_update_centroids_basic():
    """
    Test that centroid positions are correctly updated based on cluster assignments.
    
    Args:
        None
        
    Raises:
        AssertionError: If the updated centroids do not match the expected mean positions.
        
    Returns:
        None
    """
    X = np.array([
        [0, 0],
        [0, 2],
        [10, 10],
        [10, 12]
    ])
    
    labels = np.array([0, 0, 1, 1])
    km = KMeans(n_clusters=2)
    new_centroids = km._update_centroids(X, labels)
    expected = np.array([
        [0, 1],
        [10, 11]
    ])
    assert np.allclose(new_centroids, expected)
    
def test_update_centroids_handles_empty_cluster():
    """
    Test that empty clusters are properly handled by reinitializing their centroids.
    
    Args:
        None
        
    Raises:
        AssertionError: If the centroid for an empty cluster is not one of the dataset samples
        
    Returns:
        None
    """
    X = np.array([[1, 1], [2, 2], [3, 3]])
    labels = np.array([0, 0, 0])
    km = KMeans(n_clusters=2, random_state=0)
    new_centroids = km._update_centroids(X, labels)
    assert any(np.all(new_centroids[1] == x) for x in X)
    
def test_update_centroids_all_points_identical():
    """
    Test that centroid update works correctly when all points are identical.

    Args:
        None

    Raises:
        AssertionError: If the updated centroid does not match the identical point value.

    Returns:
        None
    """
    X = np.array([
        [5.0, 5.0],
        [5.0, 5.0],
        [5.0, 5.0]
    ])
    labels = np.array([0, 0, 0])
    km = KMeans(n_clusters=1)
    new_centroids = km._update_centroids(X, labels)
    assert np.allclose(new_centroids, np.array([[5.0, 5.0]]))
    
def test_update_centroids_empty_cluster_reproducibility():
    """
    Test that centroid reinitialization for empty clusters is reproducible
    when using the same random_state.

    Args:
        None

    Raises:
        AssertionError: If reinitialized centroids are not reproducible.

    Returns:
        None
    """
    X = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0]
    ])
    labels = np.array([0, 0, 0])
    km1 = KMeans(n_clusters=2, random_state=123)
    km2 = KMeans(n_clusters=2, random_state=123)
    c1 = km1._update_centroids(X, labels)
    c2 = km2._update_centroids(X, labels)
    assert np.allclose(c1, c2)