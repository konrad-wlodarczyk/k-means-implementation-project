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