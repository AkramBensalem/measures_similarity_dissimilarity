"""
Minkowski distance metric.
"""

import numpy as np
from distance_metrics import DistanceMetric

class MinkowskiDistance(DistanceMetric):
    """
    Minkowski distance metric.
    Generalization of distance metrics such as Manhattan, Euclidean, and Chebyshev.
    """

    def __init__(self, p):
        """
        Initialize the Minkowski distance metric.

        Parameters:
        p (float): Order of the Minkowski metric. Must be >= 1.
        """
        if p < 1:
            raise ValueError("p must be greater than or equal to 1")
        self.p = p

    @staticmethod
    def manhattan_distance(x_point, y_point):
        """
        Factory method for Manhattan distance (Minkowski with p=1).
        """
        return MinkowskiDistance(p=1).compute(x_point, y_point)

    @staticmethod
    def euclidean(x_point, y_point):
        """
        Factory method for Euclidean distance (Minkowski with p=2).
        """
        return MinkowskiDistance(p=2).compute(x_point, y_point)

    @staticmethod
    def chebyshev(x_point, y_point):
        """
        Factory method for Chebyshev distance (Minkowski with p=infinity).
        """
        return MinkowskiDistance(p=float('inf')).compute(x_point, y_point)

    def compute(self, x_point, y_point):
        """
        Compute the Minkowski distance between two points x and y.

        Parameters:
        x (array-like): First point.
        y (array-like): Second point.

        Returns:
        float: Computed Minkowski distance.
        """
        if self.p == float('inf'):
            return np.max(np.abs(x_point - y_point))  # Special case for Chebyshev distance
        return np.sum(np.abs(x_point - y_point) ** self.p) ** (1 / self.p)

if __name__ == "__main__":
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    print(f'Manhattan distance: {MinkowskiDistance.manhattan_distance(x, y)}')
    print(f'Euclidean distance: {MinkowskiDistance.euclidean(x, y)}')
    print(f'Chebyshev distance: {MinkowskiDistance.chebyshev(x, y)}')
