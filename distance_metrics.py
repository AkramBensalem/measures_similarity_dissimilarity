"""
This module contains implementations of various distance metrics used in clustering algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class DistanceMetric(ABC):
    """
    Abstract base class for distance metrics.
    Provides a common interface for distance computation and visualization.
    """

    @abstractmethod
    def compute(self, x_point, y_point):
        """
        Compute the distance between two points x and y.

        Parameters:
        x (array-like): First point.
        y (array-like): Second point.

        Returns:
        float: Computed distance.
        """
        pass

    def visualize(self, reference_point, x_range = (-10, 10), y_range = (-10, 10), title=None, cmap="viridis", levels=20):
        """
        Visualize the distance field of the metric around a reference point.

        Parameters:
        x_range (tuple): Range of x-values for the visualization (min, max).
        y_range (tuple): Range of y-values for the visualization (min, max).
        reference_point (array-like): The reference point for distance computation.
        title (str, optional): Title of the plot. Defaults to None.
        cmap (str, optional): Colormap for the plot. Defaults to "viridis".
        levels (int, optional): Number of contour levels. Defaults to 20.
        """
        _x = np.linspace(x_range[0], x_range[1], 100)
        _y = np.linspace(y_range[0], y_range[1], 100)
        _X, _Y = np.meshgrid(_x, _y)
        distances = np.zeros_like(_X)

        for i in range(_X.shape[0]):
            for j in range(_X.shape[1]):
                try:
                    distances[i, j] = self.compute(reference_point, np.array([_X[i, j], _Y[i, j]]))
                except ValueError:
                    distances[i, j] = 0

        plt.contourf(_X, _Y, distances, levels=levels, cmap=cmap)
        plt.colorbar(label="Distance")
        plt.scatter(*reference_point, color="red", label="Reference Point")
        plt.legend()
        plt.title(title if title else "Distance Field")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

class PowerDistance(DistanceMetric):
    def __init__(self, p):
        self.p = p

    def compute(self, x, y):
        return 1 - np.dot(x, y) ** self.p / (np.linalg.norm(x) ** self.p * np.linalg.norm(y) ** self.p)

if __name__ == "__main__":
    # Generate some example data
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    S = np.cov(np.stack((x, y)), rowvar=False)

    # Create instances of distance metrics
    cosine_distance = CosineDistance()
    power_distance = PowerDistance(p=2)

    # Compute and print distances
    print("Cosine Distance:", cosine_distance.compute(x, y))
    print("Power Distance (p=2):", power_distance.compute(x, y))

    # Note: Mahalanobis Distance and Bregman Divergence require additional input and setup
    # Uncomment and modify based on your specific requirements
    # mahalanobis_distance = MahalanobisDistance(S=S)
    # print("Mahalanobis Distance:", mahalanobis_distance.compute(x, y))

    # f = some symbolic function defining the Bregman divergence
    # Example: f = lambda x1, x2: x1**2 + x2**2
    # variables = symbols('x1 x2')
    # bregman_divergence = BregmanDivergence(f=f, vars=variables)
    # print("Bregman Divergence:", bregman_divergence.compute(np.array([1,2]), np.array([3,4])))