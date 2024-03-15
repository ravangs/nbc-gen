import numpy as np


def generate_darboux(num_points=500):
    np.random.seed(42)

    x = np.random.uniform(-2, 2, num_points)
    y = np.random.uniform(-2, 2, num_points)

    # safe and unsafe conditions
    xo_condition = (0 <= x) & (x <= 1) & (1 <= y) & (y <= 2)
    xu_condition = x + y ** 2 <= 0

    # Get labels
    labels = np.zeros(num_points)
    labels[xo_condition] = 1
    labels[xu_condition] = 2

    # Data
    X = np.vstack((x, y)).T
    Xo = np.vstack((x[xo_condition], y[xo_condition])).T
    Xu = np.vstack((x[xu_condition], y[xu_condition])).T

    return X, Xo, Xu, labels
