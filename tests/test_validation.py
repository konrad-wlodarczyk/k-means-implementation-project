import pytest
from kmeans_project.kmeans import KMeans
import numpy as np

def test_invalid_n_clusters():
    """
    Test that KMeans raises a ValueError when n_clusters is not positive.
    
    Args:
        None
    
    Raises:
        ValueError: Raised when n_clusters <= 0
        
    Returns:
        None
    """
    with pytest.raises(ValueError):
        KMeans(n_clusters=0)
        
def test_invalid_init():
    """
    Test that KMeans raises a ValueError when 'init' parameter is invalid.
    
    Args:
        None
        
    Raises:
        ValueError: Raised when init is not 'random' or 'k-means++'
    
    Returns:
        None
    """
    with pytest.raises(ValueError):
        KMeans(n_clusters=3, init="unknown")
        
def test_invalid_max_iter():
    """
    Test that KMeans raises a ValueError when max_iter is <= 0.
    
    Args:
        None
        
    Raises:
        ValueError: Raised when max_iter <= 0
        
    Returns:
        None
    """
    with pytest.raises(ValueError):
        KMeans(n_clusters=3, max_iter=0)
        
def test_invalid_tol():
    """
    Test that KMeans raises a value error when tol is <= 0
    
    Args:
        None
        
    Raises:
        ValueError: Raised when tol <= 0
        
    Returns: 
        None
    """
    with pytest.raises(ValueError):
        KMeans(n_clusters=3, tol=0)
        
def test_initialize_centroids_invalid_dimension():
    """
    Test that the dimensions are 2D
    
    Args:
        None
        
    Raises:
        ValueError: Raised when dimensions != 2D
        
    Returns: 
        None
    """
    X = np.random.rand(10)
    km = KMeans(n_clusters=3)
    with pytest.raises(ValueError):
        km._initialize_centroids(X)
        
def test_fit_raises_on_invalid_shape():
    """
    Test that fit raises a ValueError when input array has invalid shape.

    Args:
        None

    Raises:
        ValueError: Raised when input X is not 2D

    Returns:
        None
    """
    km = KMeans(n_clusters=2)
    X = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        km.fit(X)
        
def test_fit_raises_when_too_many_clusters():
    """
    Test that fit raises a ValueError when n_clusters is greater than the number of samples.

    Args:
        None

    Raises:
        ValueError: Raised when n_clusters > number of samples in X

    Returns:
        None
    """
    km = KMeans(n_clusters=10)
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    with pytest.raises(ValueError):
        km.fit(X)
        
def test_predict_raises_before_fit():
    """
    Test that predict raises a ValueError if called before fit.

    Args:
        None

    Raises:
        ValueError: Raised when centroids have not been initialized

    Returns:
        None
    """
    km = KMeans(n_clusters=2)
    X = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        km.predict(X)
        
def test_fit_raises_on_nan_values():
    """
    Test that fit raises a ValueError when input contains NaN values.

    Args:
        None

    Raises:
        ValueError: Raised when X contains NaN values

    Returns:
        None
    """
    km = KMeans(n_clusters=2)
    X = np.array([
        [1.0, 2.0],
        [np.nan, 3.0]
    ])
    with pytest.raises(ValueError):
        km.fit(X)
        
def test_invalid_random_state():
    """
    Test that KMeans raises a ValueError when random_state is invalid (e.g., negative integer).

    Args:
        None

    Raises:
        ValueError: Raised when random_state is invalid

    Returns:
        None
    """
    with pytest.raises(ValueError):
        KMeans(n_clusters=3, random_state=-10)