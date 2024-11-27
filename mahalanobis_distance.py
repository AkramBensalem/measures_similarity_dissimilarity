"""
Mahalanobis distance metric.
"""
import numpy as np
from scipy.linalg import inv, LinAlgError
from distance_metrics import DistanceMetric


class MahalanobisDistance(DistanceMetric):
    """
    Mahalanobis distance metric.

    The Mahalanobis distance measures the distance between two points
    in a multivariate space while considering correlations between variables.
    """

    def __init__(self, covariance):
        """
        Initialize the Mahalanobis distance metric.

        Parameters:
        covariance (ndarray): Covariance matrix of the data. Must be positive definite.
        """
        if not isinstance(covariance, np.ndarray):
            raise TypeError("Covariance matrix must be a numpy array.")
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("Covariance matrix must be a square matrix.")
        try:
            # Check if the covariance matrix is invertible
            _ = inv(covariance)
        except LinAlgError:
            raise ValueError("Covariance matrix must be invertible.")

        self.covariance = covariance

    def compute(self, x_point, y_point):
        """
        Compute the Mahalanobis distance between two points.

        Parameters:
        x (ndarray): First point (1D array).
        y (ndarray): Second point (1D array).

        Returns:
        float: Mahalanobis distance between x and y.
        """
        x, y = np.asarray(x_point), np.asarray(y_point)
        if x.shape != y.shape:
            raise ValueError("Points x and y must have the same dimensions.")
        if x.ndim != 1:
            raise ValueError("Points x and y must be 1D arrays.")

        delta = x - y
        return np.sqrt(np.dot(np.dot(delta.T, inv(self.covariance)), delta))

if __name__ == "__main__":
    # Example dataset
    data = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 5.0],
        [4.0, 4.0]
    ])

    # Points
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 5.0])

    # Create MahalanobisDistance instance from data
    mahalanobis = MahalanobisDistance.from_data(np.array([x, y]))
    distance = mahalanobis.compute(x, y)
    print(f"Mahalanobis Distance (from data): {distance}")
