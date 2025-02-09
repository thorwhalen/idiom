"""Tools to study high-dimensional statistics."""

# -------------------------------------------------------------------------------------
# (Near-)Orthogonality in High Dimensions
# See https://github.com/thorwhalen/idiom/discussions/1#discussioncomment-12105176

import math
from itertools import combinations
from typing import Literal
from warnings import warn

import numpy as np
from scipy.stats import norm

AngleUnits = Literal['cosine', 'degrees', 'radians']


def convert_angle(
    angle: float,
    *,
    input_unit: AngleUnits = 'degrees',
    output_unit: AngleUnits = 'cosine',
) -> float:
    """
    Converts an angle between different units: degrees, radians, and cosine.

    :param angle: The angle to convert.
    :param input_unit: The unit of the input angle ('degrees', 'radians', 'cosine').
    :param output_unit: The unit to convert to ('degrees', 'radians', 'cosine').
    :return: The converted angle.

    >>> convert_angle(90, input_unit='degrees', output_unit='radians')  # doctest: +ELLIPSIS
    1.570796...
    >>> convert_angle(0.5, input_unit='radians', output_unit='degrees')  # doctest: +ELLIPSIS
    28.6478...
    >>> convert_angle(0, input_unit='degrees', output_unit='cosine')  # doctest: +ELLIPSIS
    1.0
    """
    if input_unit == output_unit:
        return angle

    if input_unit == 'degrees':
        radians = np.deg2rad(angle)
    elif input_unit == 'radians':
        radians = angle
    elif input_unit == 'cosine':
        radians = np.arccos(angle)
    else:
        raise ValueError(f"Unsupported input unit: {input_unit}")

    if output_unit == 'degrees':
        return np.rad2deg(radians)
    elif output_unit == 'radians':
        return radians
    elif output_unit == 'cosine':
        return np.cos(radians)
    else:
        raise ValueError(f"Unsupported output unit: {output_unit}")


def convert_epsilon_angle(
    angle: float,
    *,
    input_unit: AngleUnits = 'degrees',
    output_unit: AngleUnits = 'cosine',
) -> float:
    """
    Computes the absolute difference between a straight angle (180 degrees or π radians)
    and the input angle, converting it to the desired unit.

    :param angle: The input angle.
    :param input_unit: The unit of the input angle ('degrees', 'radians', 'cosine').
    :param output_unit: The unit to convert the epsilon angle to ('degrees', 'radians', 'cosine').
    :return: The converted epsilon angle.

    >>> convert_epsilon_angle(0, input_unit='degrees', output_unit='radians')  # doctest: +ELLIPSIS
    1.5707...
    """
    straight_angle = {'degrees': 90, 'radians': np.pi / 2, 'cosine': -1}[input_unit]
    epsilon_angle = abs(straight_angle - angle)

    return convert_angle(epsilon_angle, input_unit=input_unit, output_unit=output_unit)


def expected_dot_product_variance(n: int) -> float:
    """
    Computes the expected variance of the dot product between two random unit vectors
    in an n-dimensional space.

    :param n: Dimension of the space
    :return: Expected variance of the dot product

    >>> expected_dot_product_variance(10)
    0.1
    >>> expected_dot_product_variance(100)
    0.01
    """
    return 1 / n


def nearly_orthogonal_vectors_johnson_lindenstrauss(
    n: int, epsilon: float = 0.01, epsilon_unit: AngleUnits = 'cosine'
) -> float:
    """
    Estimates the number of nearly orthogonal vectors that can fit in an n-dimensional
    space given an orthogonality tolerance epsilon.

    See https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma#Statement

    :param n: Dimension of the space
    :param epsilon: Tolerance level for near-orthogonality
    :param epsilon_unit: The unit of epsilon ('cosine', 'degrees', or 'radians').
    :return: Estimated number of nearly orthogonal vectors


    # >>> nearly_orthogonal_vectors_johnson_lindenstrauss(100, 0.1)
    # 100
    # >>> nearly_orthogonal_vectors_johnson_lindenstrauss(1000, 0.1)
    # 1484
    # >>> nearly_orthogonal_vectors_johnson_lindenstrauss(10000, 0.1)
    # 51847055285871095840768
    """
    warn("Don't believe these numbers. Not sure I got the math right.")
    if n < 2:
        raise ValueError("The dimension must be at least 2.")
    epsilon = convert_epsilon_angle(
        epsilon, input_unit=epsilon_unit, output_unit='cosine'
    )
    return math.floor(
        max(n, np.exp((n * epsilon**2) / 2) / epsilon)
    )


def nearly_orthogonal_vectors_equiangular(
    n: int, epsilon: float, epsilon_unit: AngleUnits = 'cosine'
) -> float:
    """
    Estimates a (provable) lower bound for the maximum number N(ε, n) of unit vectors in an n-dimensional
    Euclidean space that can be chosen so that every pair of distinct vectors has an inner product (cosine
    of the angle between them) bounded by ε, i.e.

          |v_i · v_j| <= epsilon,   for all i ≠ j.

    Special cases:
      - If epsilon = 0, the vectors must be exactly orthogonal, so the maximum is exactly n.
      - If epsilon >= 1, every pair of unit vectors trivially satisfies |v_i · v_j| ≤ 1, so one may
        pack infinitely many such vectors.

    For 0 < epsilon < 1, one standard (Johnson–Lindenstrauss) argument shows one can pack roughly
          exp((epsilon**2 * n) / 8)
    vectors. However, more refined techniques from spherical coding theory and the Kabatjanskii–Levenstein
    bound yield a stronger lower bound of the form
          N(ε, n) >= exp((epsilon**2 * n) / 2).

    For example, with n ≈ 1500 and epsilon ≈ 0.1 the refined bound gives
          N(0.1, 1500) >= exp((0.1**2 * 1500) / 2) = exp(7.5) ≈ 1800.

    This function returns the larger of the trivial bound (n) and the refined bound exp((epsilon² * n) / 2).

    For more details on these results and their derivations, see, for example,
    "Equiangular lines" on Wikipedia:
      https://en.wikipedia.org/wiki/Equiangular_lines

    Note: This is an asymptotic (existential) lower bound and should be considered a heuristic estimate.

    :param n: Dimension of the Euclidean space (n >= 1).
    :param epsilon: Tolerance for “near-orthogonality” (0 <= epsilon < 1).
    :param epsilon_unit: The unit of epsilon ('cosine', 'degrees', or 'radians').
    :return: An estimate for the maximum number of nearly orthogonal unit vectors.

    # >>> nearly_orthogonal_vectors_equiangular(100, 0.1)
    # 100
    # >>> nearly_orthogonal_vectors_equiangular(1000, 0.1)
    # 1000
    # >>> nearly_orthogonal_vectors_equiangular(10000, 0.1)
    # 5184705528587109793792

    # With different epsilon units:

    # >>> nearly_orthogonal_vectors_equiangular(100000, 0, epsilon_unit='degrees')
    # 100000
    # >>> nearly_orthogonal_vectors_equiangular(1000, 5, epsilon_unit='degrees')
    
    """
    warn("Don't believe these numbers. Not sure I got the math right.")
    epsilon = convert_epsilon_angle(epsilon, input_unit=epsilon_unit, output_unit='cosine')
    # If epsilon is 1 or larger, every unit vector trivially satisfies the condition,
    # so the maximum number is unbounded.
    if epsilon >= 1:
        return float('inf')
    # For epsilon = 0, the only possibility is an orthonormal set.
    if epsilon == 0:
        return n
    # Using refined spherical coding / Kabatjanskii–Levenstein bound:
    # one can prove that for any fixed ε > 0, one may pack at least
    # N(ε, n) ≥ exp((ε² * n) / 2)
    # (see, e.g., https://en.wikipedia.org/wiki/Equiangular_lines for background)
    refined_bound = math.exp((epsilon**2 * n) / 2)
    # The trivial bound (an orthonormal set) gives n vectors.
    return math.floor(max(n, refined_bound))


def nearly_orthogonal_vectors(
    n: int, epsilon: float, epsilon_unit: AngleUnits = 'cosine'
):
    """
    Estimates the min number of nearly orthogonal vectors that can fit in an
    n-dimensional.

    """
    return max(
        [
            nearly_orthogonal_vectors_johnson_lindenstrauss(n, epsilon, epsilon_unit),
            nearly_orthogonal_vectors_equiangular(n, epsilon, epsilon_unit),
        ]
    )


def generate_random_unit_vectors(dim: int, num_vectors: int = 100) -> np.ndarray:
    """
    Generates `num_vectors` random unit vectors in `dim`-dimensional space.

    :param dim: Dimension of the space
    :param num_vectors: Number of vectors to generate
    :return: A (num_vectors, dim) numpy array of unit vectors

    >>> vectors = generate_random_unit_vectors(5, 3)
    >>> np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    True
    """
    vectors = np.random.randn(num_vectors, dim)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def measure_pairwise_angles(vectors: np.ndarray) -> np.ndarray:
    """
    Computes the pairwise angles (in degrees) between unit vectors.

    :param vectors: A numpy array of shape (num_vectors, dim)
    :return: A numpy array of pairwise angles in degrees

    >>> vecs = generate_random_unit_vectors(10, 5)
    >>> angles = measure_pairwise_angles(vecs)
    >>> (0 <= angles).all() and (angles <= 180).all()
    True
    """
    num_vectors = vectors.shape[0]
    angles = []
    for i, j in combinations(range(num_vectors), 2):
        dot_product = np.dot(vectors[i], vectors[j])
        angle = np.arccos(np.clip(dot_product, -1, 1)) * (180 / np.pi)
        angles.append(angle)
    return np.array(angles)


# -------------------------------------------------------------------------------------
# High Dimensional Statistics

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
            self.data = generator(
                -1, 1, size=(n_points, dim)
            )  # Generate points in [-1,1]^dim

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


def run_high_dim_experiment(
    dims=(2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000),
    n_points=5000,
    generator=np.random.uniform,
):
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
        results.append(
            {
                "dimension": dim,
                "mean_distance": distance_stats["mean_distance"],
                "distance_variance": distance_stats["variance"],
                "mean_cosine_similarity": cosine_stats["mean_cosine"],
                "cosine_similarity_variance": cosine_stats["variance"],
                "mean_norm": norm_stats["mean_norm"],
                "norm_variance": norm_stats["variance"],
                "fraction_near_surface": hollow_ball["fraction_near_surface"],
                "fraction_near_corners": corner_volume["fraction_near_corners"],
            }
        )

    # Convert results to DataFrame
    df = pd.DataFrame(results)
    return df
