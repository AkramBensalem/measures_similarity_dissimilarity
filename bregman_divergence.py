import numpy as np
from sympy import diff, lambdify
from distance_metrics import DistanceMetric
from sympy import symbols

class BregmanDivergence(DistanceMetric):
    """
    Bregman Divergence metric.

    The Bregman Divergence is a generalization of squared Euclidean distance
    that uses a convex function to compute distance in a non-Euclidean space.
    """

    def __init__(self, f, vars):
        """
        Initialize the Bregman Divergence metric.

        Parameters:
        f (sympy.Expr): A convex function defining the Bregman Divergence.
        vars (list of sympy.Symbol): The variables of the convex function.
        """
        self.f = f
        self.vars = vars
        # Precompute gradient functions for better performance
        self.grad_f = [
            lambdify(vars, diff(f, var), 'numpy') for var in vars
        ]
        self.f_func = lambdify(vars, f, 'numpy')

    def compute(self, x, y):
        """
        Compute the Bregman Divergence between two points.

        Parameters:
        x (array-like): First point.
        y (array-like): Second point.

        Returns:
        float: Bregman Divergence between x and y.
        """
        x, y = np.asarray(x), np.asarray(y)
        if len(x) != len(self.vars) or len(y) != len(self.vars):
            raise ValueError("Points x and y must match the number of variables in the function.")

        # Compute f(x) and f(y)
        f_x = self.f_func(*x)
        f_y = self.f_func(*y)

        # Compute gradient of f at x
        grad_f_x = np.array([grad(*x) for grad in self.grad_f])

        # Compute Bregman Divergence
        divergence = f_x - f_y - np.dot(grad_f_x, x - y)
        return divergence

if __name__=="__main__":
    # Define a convex function f(x1, x2) = x1^2 + x2^2
    x1, x2 = symbols('x1 x2')
    f = x1 ** 2 + x2 ** 2

    # Create a Bregman Divergence instance
    bregman = BregmanDivergence(f=f, vars=[x1, x2])

    # Points
    x = [1, 2]
    y = [3, 4]

    # Compute the Bregman Divergence
    divergence = bregman.compute(x, y)
    print(f"Bregman Divergence between {x} and {y}: {divergence}")