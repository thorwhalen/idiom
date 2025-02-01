"""Tools to study high-dimensional statistics."""


import numpy as np
from scipy.spatial.distance import pdist, squareform

class HighDimStats:
    def __init__(self, data=None, dim=100, n_points=1000, generator=np.random.uniform):
        """
        Initialize with data or generate random high-dimensional points.
        
        Args:
            data (numpy.ndarray): (n_points, dim) dataset.
            dim (int): Number of dimensions (if generating data).
            n_points (int): Number of points (if generating data).
            generator (callable): Random generator function (default: uniform).
        """
        if data is not None:
            self.data = np.array(data)
            self.dim = self.data.shape[1]
        else:
            self.dim = dim
            self.data = generator(-1, 1, size=(n_points, dim))  # Generate points in [-1,1]^dim
        
        self.n_points = self.data.shape[0]

    def pairwise_distances(self):
        """Compute pairwise Euclidean distances and report mean and variance."""
        distances = pdist(self.data, metric="euclidean")
        return {"mean_distance": np.mean(distances), "variance": np.var(distances)}

    def cosine_similarity_distribution(self):
        """Compute cosine similarity between random pairs of points."""
        norms = np.linalg.norm(self.data, axis=1, keepdims=True)
        normalized_data = self.data / norms  # Normalize to unit vectors
        cosine_sim = np.dot(normalized_data, normalized_data.T)
        np.fill_diagonal(cosine_sim, 0)  # Remove self-similarity (cosine=1)
        return {"mean_cosine": np.mean(cosine_sim), "variance": np.var(cosine_sim)}

    def norm_distribution(self):
        """Analyze the concentration of vector norms."""
        norms = np.linalg.norm(self.data, axis=1)
        return {"mean_norm": np.mean(norms), "variance": np.var(norms)}

    def hollow_ball_effect(self, threshold=0.95):
        """
        Computes the fraction of points near the hypersphere's surface.
        
        Args:
            threshold (float): Fraction of the radius where "near surface" is defined.
        
        Returns:
            Fraction of points whose norm is in the top (1 - threshold) of the radius.
        """
        norms = np.linalg.norm(self.data, axis=1)
        radius = np.max(norms)
        near_surface = np.sum(norms >= threshold * radius) / self.n_points
        return {"fraction_near_surface": near_surface}

    def corner_volume_fraction(self, threshold=0.95):
        """
        Computes the fraction of points near the corners of a hypercube.
        
        Args:
            threshold (float): Fraction of max distance from center to define "corner".
        
        Returns:
            Fraction of points that are near the corners.
        """
        norms = np.linalg.norm(self.data, axis=1)
        max_norm = np.sqrt(self.dim)  # Distance from center to a cube vertex
        near_corners = np.sum(norms >= threshold * max_norm) / self.n_points
        return {"fraction_near_corners": near_corners}


import pandas as pd

def run_high_dim_experiment(dims=(2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000), 
                            n_points=5000, generator=np.random.uniform):
    """
    Runs HighDimStats across multiple dimensions and returns results as a DataFrame.
    
    Args:
        dims (tuple): List of dimensions to evaluate.
        n_points (int): Number of points per dimension.
        generator (callable): Random generator function (default: uniform).
    
    Returns:
        pd.DataFrame: Contains statistics for each dimension.
    """
    results = []

    for dim in dims:
        print(f"Running experiment for dimension: {dim}")
        stats = HighDimStats(dim=dim, n_points=n_points, generator=generator)

        # Compute statistics
        distance_stats = stats.pairwise_distances()
        cosine_stats = stats.cosine_similarity_distribution()
        norm_stats = stats.norm_distribution()
        hollow_ball = stats.hollow_ball_effect()
        corner_volume = stats.corner_volume_fraction()

        # Collect all stats in a dict
        results.append({
            "dimension": dim,
            "mean_distance": distance_stats["mean_distance"],
            "distance_variance": distance_stats["variance"],
            "mean_cosine_similarity": cosine_stats["mean_cosine"],
            "cosine_similarity_variance": cosine_stats["variance"],
            "mean_norm": norm_stats["mean_norm"],
            "norm_variance": norm_stats["variance"],
            "fraction_near_surface": hollow_ball["fraction_near_surface"],
            "fraction_near_corners": corner_volume["fraction_near_corners"]
        })

    # Convert results to DataFrame
    df = pd.DataFrame(results)
    return df
