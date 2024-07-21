import numpy as np

def generate_stochastic_matrix(n):
    n = int(n)
    matrix = np.random.rand(n, n)
    matrix = matrix / matrix.sum(axis=1)[:, None]
    return matrix

def generate_equal_spacing(delta, n):
    return np.array([i * float(delta) for i in range(int(n))])

def generate_asymptotic_spacing(delta, C, n):
    delta = float(delta)
    n= int(n)
    C = float(C)
    spacing = []
    current = np.random.uniform(0, C)
    spacing.append(current)
    for i in range(1, n):
        step = np.random.uniform(delta, C)
        current += step
        spacing.append(current)
    return np.array(spacing)

def generate_random_spacing(delta, n):
    delta = float(delta)
    n =  int(n)
    spacing = np.random.exponential(delta, n)
    return np.cumsum(spacing)