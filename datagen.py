import numpy as np


def generate_points_for_region(condition, num_points):
    points = []
    while len(points) < num_points:
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        if condition(x, y):
            points.append((x, y))
    return np.array(points)


def generate_darboux(num_points_per_region=500):

    Xo = generate_points_for_region(condition_o, num_points_per_region)
    Xu = generate_points_for_region(condition_u, num_points_per_region)
    Xn = generate_points_for_region(condition_n, num_points_per_region)

    X = np.vstack((Xo, Xu, Xn))

    labels = np.array([1] * len(Xo) + [2] * len(Xu) + [0] * len(Xn))

    return X, Xo, Xu, labels


def condition_o(x, y):
    return 0 <= x <= 1 <= y <= 2


def condition_u(x, y):
    return x + y ** 2 <= 0


def condition_n(x, y):
    return not (condition_o(x, y) or condition_u(x, y))
