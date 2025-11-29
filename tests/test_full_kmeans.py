import numpy as np
import pytest
from kmeans_project.kmeans import KMeans

def test_fit_converges_simple_data():
    """
    Test that KMeans fit converges on a simple 2-cluster dataset.

    Args:
        None

    Raises:
        AssertionError: If centroids shape is incorrect, labels are invalid, or inertia is negative.

    Returns:
        None
    """
    X = np.array([
        [0, 0], [0, 1],
        [10, 10], [11, 11]
    ])
    km = KMeans(n_clusters=2, random_state=0)
    km.fit(X)
    assert km.centroids.shape == (2, 2)
    assert set(km.labels_) == {0, 1}
    assert km.inertia_ >= 0
    
def test_predict_after_fit():
    """
    Test that predict assigns new points to nearest clusters after fit.

    Args:
        None

    Raises:
        AssertionError: If predicted labels are not in the expected cluster range.

    Returns:
        None
    """
    X = np.array([[0, 0], [10, 10]])
    km = KMeans(n_clusters=2, random_state=0)
    km.fit(X)
    new_points = np.array([[1, 1], [9, 9]])
    labels = km.predict(new_points)
    assert len(labels) == 2
    assert all(l in {0, 1} for l in labels)
    
def test_converges_early_with_tol():
    """
    Test that KMeans can converge before reaching max_iter when tol is set.

    Args:
        None

    Raises:
        AssertionError: If number of iterations exceeds max_iter without convergence.

    Returns:
        None
    """
    X = np.array([
        [1, 1], [1.1, 1.1],
        [10, 10], [10.1, 10.1]
    ])
    km = KMeans(n_clusters=2, tol=1e-1, max_iter=100)
    km.fit(X)
    assert km.n_iter_ < 100
    
def test_stops_at_max_iter():
    """
    Test that KMeans stops exactly at max_iter if convergence is not reached.

    Args:
        None

    Raises:
        AssertionError: If n_iter_ is not equal to max_iter when early convergence is not reached.

    Returns:
        None
    """
    X = np.random.rand(50, 2)
    km = KMeans(n_clusters=3, max_iter=1, random_state=0)
    km.fit(X)
    assert km.n_iter_ == 1
    
def test_fit_single_cluster():
    """
    Test that KMeans correctly handles n_clusters=1.

    Args:
        None

    Raises:
        AssertionError: If centroids shape is incorrect or all labels are not zero.

    Returns:
        None
    """
    X = np.random.rand(10, 2)
    km = KMeans(n_clusters=1, random_state=42)
    km.fit(X)
    assert km.centroids.shape == (1, 2)
    assert np.all(km.labels_ == 0)
    
def test_fit_reproducibility():
    """
    Test that KMeans fit is reproducible with the same random_state.

    Args:
        None

    Raises:
        AssertionError: If centroids or labels differ between fits with same random_state.

    Returns:
        None
    """
    X = np.random.rand(30, 2)
    km1 = KMeans(n_clusters=3, random_state=42)
    km2 = KMeans(n_clusters=3, random_state=42)
    km1.fit(X)
    km2.fit(X)
    assert np.allclose(km1.centroids, km2.centroids)
    assert np.array_equal(km1.labels_, km2.labels_)
    
def test_inertia_decreases_over_iterations():
    """
    Test that KMeans inertia does not increase when re-fitting on the same data.

    Args:
        None

    Raises:
        AssertionError: If inertia increases between consecutive fits.

    Returns:
        None
    """
    X = np.array([
        [0, 0], [0, 1],
        [10, 10], [11, 11]
    ])
    km = KMeans(n_clusters=2, max_iter=10, random_state=0)
    km.fit(X)
    prev_inertia = km.inertia_
    for _ in range(3):
        km.fit(X)
        assert km.inertia_ <= prev_inertia
        prev_inertia = km.inertia_