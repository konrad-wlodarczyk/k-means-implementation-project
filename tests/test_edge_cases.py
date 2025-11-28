import numpy as np
import pytest 
from kmeans_project.kmeans import KMeans

def test_all_points_identical():
    """
    Test KMeans behavior when all points in the dataset are identical.

    Ensures that all centroids are equal to the identical points and labels are valid.

    Args:
        None

    Raises:
        AssertionError: If centroids do not match the identical point or labels exceed expected range.

    Returns:
        None
    """
    X = np.array([
        [5.0, 5.0],
        [5.0, 5.0],
        [5.0, 5.0]
    ])
    km = KMeans(n_clusters=2, random_state=0)
    km.fit(X)
    for c in km.centroids:
        assert np.all(c == X[0])
        
    assert set(km.labels_) <= {0, 1}
    
def test_large_values():
    """
    Test KMeans with very large numerical values.

    Ensures centroids are computed without overflow and labels remain valid.

    Args:
        None

    Raises:
        AssertionError: If centroids shape is incorrect or labels exceed expected range.

    Returns:
        None
    """
    X = np.array([
        [1e12, 1e12],
        [1e12 + 1, 1e12 + 1],
        [1e12 + 100, 1e12 + 100]
    ])
    km = KMeans(n_clusters=2, random_state=0)
    km.fit(X)
    assert km.centroids.shape == (2, 2)
    assert set(km.labels_) <= {0, 1}
    
def test_small_values():
    """
    Test KMeans with very small numerical values.

    Ensures centroids are computed without underflow and labels remain valid.

    Args:
        None

    Raises:
        AssertionError: If centroids shape is incorrect or labels exceed expected range.

    Returns:
        None
    """
    X = np.array([
        [1e-12, 1e-12],
        [2e-12, 2e-12],
        [3e-12, 3e-12]
    ])
    km = KMeans(n_clusters=2, random_state=0)
    km.fit(X)
    assert km.centroids.shape == (2, 2)
    assert set(km.labels_) <= {0, 1}
    
def test_overlapping_centroids():
    """
    Test KMeans when multiple points are close or overlapping.

    Ensures that centroids after fitting are selected from actual points.

    Args:
        None

    Raises:
        AssertionError: If centroids do not correspond to one of the input points.

    Returns:
        None
    """
    X = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3]
    ])
    km = KMeans(n_clusters=3, random_state=0)
    km.fit(X)
        
    for c in km.centroids:
        assert any(np.all(c == x) for x in X)
            
def test_empty_cluster_after_initialization():
    """
    Test KMeans behavior when empty clusters may appear due to duplicate points.

    Ensures that empty clusters are properly initialized to an existing point.

    Args:
        None

    Raises:
        AssertionError: If centroids do not match one of the dataset points.

    Returns:
        None
    """
    X = np.array([
        [0, 0],
        [0, 0],
        [0, 0]
    ])
    km = KMeans(n_clusters=2, random_state=0)
    km.fit(X)
    
    for c in km.centroids:
        assert np.all(c == X[0])