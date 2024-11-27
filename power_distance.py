"""
Power Distance metric.
"""
import numpy as np
from distance_metrics import DistanceMetric

class PowerDistance(DistanceMetric):
    """
    Power Distance metric.

    This metric measures the distance between two vectors by comparing their
    dot product and norms, with both raised to a power p.

    Power Distance = 1 - ( (x · y)^p / (||x||^p * ||y||^p) )
    """
    @staticmethod
    def cosine_distance(x_point, y_point):
        """
        Factory method for cosine distance (PowerDistance with p=1).
        """
        return PowerDistance(p=1).compute(x_point, y_point)

    def __init__(self, p):
        """
        Initialize the Power Distance metric.

        Parameters:
        p (float): Power parameter (must be greater than 0).
        """
        if p <= 0:
            raise ValueError("Power parameter p must be greater than 0.")
        self.p = p

    def compute(self, x_point, y_point):
        """
        Compute the Power Distance between two points.

        Parameters:
        x_point (array-like): First vector.
        y_point (array-like): Second vector.

        Returns:
        float: Power Distance between x_point and y_point.
        """
        x_point = np.asarray(x_point)
        y_point = np.asarray(y_point)

        # Compute norms
        norm_x = np.linalg.norm(x_point)
        norm_y = np.linalg.norm(y_point)

        # Handle zero vectors
        if norm_x == 0 or norm_y == 0:
            raise ValueError("Input vectors must not be zero vectors.")

        # Compute dot product and norms raised to power p
        dot_product = np.dot(x_point, y_point)
        dot_product_p = dot_product ** self.p
        norms_p = (norm_x * norm_y) ** self.p

        # Compute Power Distance
        return 1 - (dot_product_p / norms_p)

if __name__=="__main__":
    # Define two vectors
    x = np.array([1, 2])
    y = np.array([4, 5])

    # Create a PowerDistance instance with p=2
    power_distance = PowerDistance(p=10)

    # Compute the Power Distance between x and y
    distance = power_distance.compute(x, y)
    print(f'Power Distance: {distance}')

    # Visualize the distance field
    power_distance.visualize(
        reference_point = np.array([1, 2]),
        title="Power Distance"
    )