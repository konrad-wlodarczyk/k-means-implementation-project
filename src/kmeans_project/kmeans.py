import numpy as np
from typing import Optional

class KMeans:
    """Custom implementation of the K-Means clustering algorithm.
    
    This class provide a basic implementation of the k-Means clustering
    procedure, including support for random and k-means++ centroid initialization,
    an iterative refinement procedure, and computation of inertia and cluster labels.
    """    
    def __init__(
        self,
        n_clusters: int,
        init: str = "random",
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        """
        Initialize a KMeans instance.

        Args:
            n_clusters (int): The number of clusters to form.
            init (str, optional): Method for centroid initialization.
                Supported options are:
                - "random": choose initial centroids uniformly at random
                  from the data.
                - "k-means++": use the k-means++ initialization strategy.
                Defaults to "random".
            max_iter (int, optional): Maximum number of iterations of the
                K-Means algorithm for a single run. Defaults to 300.
            tol (float, optional): Tolerance for convergence. The algorithm
                stops when the Euclidean norm of the change in centroids
                between two consecutive iterations is less than this value.
                Defaults to 1e-4.
            random_state (Optional[int], optional): Seed for the random number
                generator, used to ensure reproducible initialization.
                Defaults to None.

        Raises:
            ValueError: If `n_clusters <= 0`.
            ValueError: If `init` is not "random" or "k-means++".
            ValueError: If `max_iter <= 0`.
            ValueError: If `tol <= 0`.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
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
        Initialize cluster centroids according to the selected strategy.
        
        Args:
            X (np.ndarray): Input data from which initial centorids will be selected

        Raises:
            ValueError: If X is not a 2D array.

        Returns:
            np.ndarray: The initialized centroids.
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
        
    def fit(self, X: np.ndarray):
          
        """Compute K-Means clustering on dataset X.

        Raises:
            ValueError: X must be a 2D arraty of shape (n_samples, n_features).

        Returns:
            self: object
        """        
        if X.ndim != 2:
            raise ValueError("X must be a 2D arraty of shape (n_samples, n_features).")
        
        centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iter):
            labels, distances = self._assign_clusters(X, centroids)
            
            new_centroids = self._update_centroids(X, labels)
            
            centroid_shift = np.linalg.norm(new_centroids - centroids)
            
            if centroid_shift < self.tol:
                centroids = new_centroids
                break
            
            centroids = new_centroids
            
        self.centroids = centroids
        self.labels_ = labels
        self.inertia_ = self._compute_inertia(X, labels, centroids)
        self.n_iter_ = iteration + 1
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample in X to the nearest cluster centroid.

        Args:
            X (np.ndarray): New data to assign to clusters.

        Raises:
            ValueError: If the model has not been fitted or if X is not a 2D array.

        Returns:
            labels (np.ndarray): Index of the closest centroid for each sample.
        """        
        if self.centroids is None:
            raise ValueError("Model has not been fitted. Call fit(X) before predict(X).")
        
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        
        labels, _ = self._assign_clusters(X, self.centroids)
        
        return labels
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each sample in X to the nearest centroid

        Args:
            X (np.ndarray): Input data
            centroids (np.ndarray): Current centroid positions.

        Raises:
            ValueError: X and centroids must be 2D arrays

        Returns:
            np.ndarray: Index of the closest centroid for each sample.
            labels: Index of the closest centroid for each sample.
         
        Notes:    
        This method computes squared Euclidean distance using fully
        vectorized numpy operations: distance(i, j) = || X[i] - centroids[j] ||^2
        The closest centroid is chosen for each sample.
        """        
        if X.ndim != 2 or centroids.ndim != 2:
            raise ValueError("X and centroids must be 2D arrays.")
        
        # compute squared distanec using broadcasting:
        # shape result: (n_sampled, n_clusters)
        distances_matrix = np.sum(
            (X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        
        #cluster labels = index of the nearest centroid
        labels = np.argmin(distances_matrix, axis=1)

        # squared distances to the assigned centroid
        min_distances = distances_matrix[np.arange(X.shape[0]), labels]
        
        return labels, min_distances
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute updated centroid positions based on cluster assignments

        Args:
            X (np.ndarray): Input data
            labels (np.ndarray): Cluster index assigned to each sample.
a
        Returns:
            new_centroids (np.ndarray): Updated centroid Positions.
        """        
        n_samples, n_features = X.shape
        new_centroids = np.zeros((self.n_clusters, n_features), dtype=X.dtype)
        rng = np.random.default_rng(self.random_state)
        
        for cluster_idx in range(self.n_clusters):
            cluster_points = X[labels == cluster_idx]
            
            if len(cluster_points) == 0:
                # empty cluster: reinitialize randomly
                new_centroids[cluster_idx] = X[rng.integers(0, n_samples)]
            else:
                new_centroids[cluster_idx] = cluster_points.mean(axis=0)
        
        return new_centroids
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Compute K-Means inertia (sum of squared distances to assigned centroids)

        Args:
            X (np.ndarray): ndarray of shape (n_samples, n_features)
            labels (np.ndarray): ndarray of shape (n_samples,)
            centroids (np.ndarray): ndarray of shape (n_clusters, n_features)

        Returns:
            float: Sum of squared distances of samples to their closest centroid.
        """        
        distances = np.sum((X - centroids[labels]) ** 2, axis=1)
        return float(distances.sum())
        
        
        
    def __repr__(self):
        return (
            f"KMeans(n_clusters={self.n_clusters}, "
            f"init='{self.init}', "
            f"max_iter={self.max_iter}, "
            f"tol={self.tol}, "
            f"random_state={self.random_state})"
        )