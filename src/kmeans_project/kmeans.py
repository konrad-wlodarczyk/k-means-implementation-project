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
        
        if n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")
        
        if init not in ("random", "k-means++"):
            raise ValueError("init must be 'random' or 'k-means++'.")
        
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0.")
        
        if tol <= 0:
            raise ValueError("tol must be > 0.")

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
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        
        if self.init == "random":
            indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
            return X[indices].copy()
        
        elif self.init == "k-means++":
            # Choose first centroid randomly
            indices = []
            first_index = rng.choice(n_samples)
            indices.append(first_index)
            
            # Select the remianing k-1 centroids
            for _ in range(1, self.n_clusters):
                # Compute squared distances to nearest already-chosen centroid
                existing_centroids = X[indices]
                distances = np.min(
                    np.sum((X[:, np.newaxis, :] - existing_centroids)**2, axis=2),
                    axis=1,
                )
                
                # Probability proportional to squared distance
                if np.sum(distances) == 0:
                    # All of the points are identical - choose randomly
                    new_index = rng.choice(n_samples)
                else:
                    probabilities = distances / distances.sum()
                    new_index = rng.choice(n_samples, p=probabilities)
                    
                indices.append(new_index)
            
            return X[indices].copy()
        
        else:
            raise ValueError(
                "Unknown initialization method: "
                f"{self.init!r}. Supported options are 'random' and 'k-means++'."
            )
        
        
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
    
    def _euclidean_distance(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            X (np.ndarray): _description_
            centroids (np.ndarray): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            np.ndarray: _description_
        """        
        raise NotImplementedError("Distance computation not implemented yet.")
    
    def __repr__(self):
        return (
            f"KMeans(n_clusters={self.n_clusters})"
            f"init='{self.init}"
            f"max_iter={self.max_iter}"
            f"tol={self.tol}"
            f"random_state={self.random_state}"
        )