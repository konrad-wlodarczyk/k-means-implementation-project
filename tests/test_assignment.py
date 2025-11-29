import numpy as np
from kmeans_project.kmeans import KMeans
import pytest

def test_assign_clusters_simple():
    """
    Test that each sample is assigned to the nearest centroid and distances are correct.
    
    Args:
        None
        
    Raises:
        AssertionError: If labels do not match expected cluster assignments
                        or if the distances array length is incorrect.
                        
    Returns:
        None
    """
    x = np.array([
        [0, 0],
        [0, 1],
        [10, 10],
        [11, 11]
    ])
    
    centroids = np.array([
        [0, 0],
        [10, 10]
    ])
    
    km = KMeans(n_clusters=2)
    labels, distances = km._assign_clusters(x, centroids)
    assert np.array_equal(labels, np.array([0, 0, 1, 1]))
    assert len(distances) == 4
    
def test_assign_clusters_label_range():
    """
    Test that cluster labels assigned by _assign_cluster are within the valid range
    [0, n_clusters-1]
    
    Args:
        None
        
    Raises:
        AssertionError: If any label is outside the range [0, n_clusters-1]
        
    Returns:
        None
    """
    X = np.random.rand(50, 2)
    centroids = np.array([[0, 0], [1, 1], [2, 2]])
    km = KMeans(n_clusters=3)
    labels, _ = km._assign_clusters(X, centroids)
    assert labels.min() >= 0
    assert labels.max() < 3
    
def test_assign_clusters_invalid_X_dimensions():
    """
    Test that _assign_clusters raises a ValueError when input X is not a 2D array.
    
    Args:
        None
        
    Raises:
        ValueError: Raised when X.ndim != 2
        
    Returns:
        None
    """
    X = np.random.rand(10)
    centroids = np.random.rand(3, 2)
    km = KMeans(n_clusters=3)
    with pytest.raises(ValueError):
        km._assign_clusters(X, centroids)
        
def test_assign_clusters_invalid_centroid_dimension():
    """
    Test that _assign_clusters raises a ValueError when centroids is not 2D
    
    Args:
        None
        
    Raises:
        ValueError: Raised when centroids.ndim != 2.
        
    Returns:
        None
    """
    X = np.random.rand(10, 2)
    centroids = np.random.rand(3)
    km = KMeans(n_clusters=3)
    with pytest.raises(ValueError):
        km._assign_clusters(X, centroids)
        
def test_assign_clusters_identical_centroids():
    """
    Test that _assign_clusters handles identical centroids correctly.
    
    Args:
        None
        
    Raises:
        AssertionError: If labels or distance shape is incorrect,
                        or if labels exceed n_clusters - 1.
                        
    Returns:
        None
    """
    X = np.array([
        [0, 0],
        [1, 1],
        [2, 2]
    ])
    centroids = np.array([
        [0, 0],
        [0, 0]
    ])
    km = KMeans(n_clusters=2)
    labels, distances = km._assign_clusters(X, centroids)
    assert labels.shape == (3,)
    assert distances.shape == (3,)
    assert labels.max() < 2
    
def test_assign_clusters_large_values():
    """
    Test that _assign_clusters handles extremely large values without overflow.
    
    Args:
        None
        
    Raises:
        AssertionError: If labels do not match expected assignments or distances array is incorrect.
        
    Returns:
        None
    """
    X = np.array([
        [1e12, 1e12],
        [1e12 + 1, 1e12 + 1],
        [1e12 + 100, 1e12 + 100]
    ])
    centroids = np.array([
        [1e12, 1e12],
        [1e12 + 100, 1e12 + 100]
    ])
    km = KMeans(n_clusters=2)
    labels, distances = km._assign_clusters(X, centroids)
    assert np.array_equal(labels, np.array([0, 0, 1]))
    assert distances.shape == (3,)
    