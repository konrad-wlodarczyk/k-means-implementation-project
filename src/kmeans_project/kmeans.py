import numpy as np
from typing import Optional

class KMeans:
    """Skeleton of a custom K-Means clustering class.

    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_
        NotImplementedError: _description_
    """    """_summary_
    """    
    def __init__(
        self,
        n_clusters: int,
        init: str = "random",
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids = None
        self.labels_ = None
        self.intertia_ = None
        self.n_iter_ = None

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Placeholder for centroid initialization logic.
        
        Args:
            X (np.ndarray): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            np.ndarray: _description_
        """        
        
        raise NotImplementedError("Centroid init not implemented yet.")
    
    def fit(self, X: np.ndarray):
        """Fit the K-Means model to the dataset.

        Args:
            X (np.ndarray): _description_

        Raises:
            NotImplementedError: _description_
        """        
        raise NotImplementedError("Fir method not implemented yet.")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict closest cluster for each sample.

        Args:
            X (np.ndarray): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            np.ndarray: _description_
        """        
        raise NotImplementedError("Predict method not implemented yet.")