import pytest
from kmeans_project.kmeans import KMeans

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