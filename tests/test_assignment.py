import numpy as np
from kmeans_project.kmeans import KMeans

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