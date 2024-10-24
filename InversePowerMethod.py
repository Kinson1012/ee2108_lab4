import matplotlib.pyplot as plt
import numpy as np


def inverse_power_method(A, num_iterations=20):
    """
    Calculate the smallest eigenvalue and corresponding eigenvector of matrix A
    using the inverse power method.
    Parameters:
    - A: numpy array, square matrix
    - num_iterations: int, number of iterations to perform
    Returns:
    - smallest_eigenvalue: float, the smallest eigenvalue of A
    - corresponding_eigenvector: numpy array, the eigenvector corresponding to the
      smallest eigenvalue
    - convergence_metric: numpy array, convergence metric over iterations
    """
    if np.linalg.det(A) == 0:
        raise ValueError("Matrix A is singular and cannot be inverted.")

    # Invert matrix A
    B = np.linalg.inv(A)

    # Initialize variables
    shape = (A.shape[0], num_iterations)
    X = np.zeros(shape)  # To store eigenvectors at each iteration
    eigenvalues = np.zeros(num_iterations)  # To store eigenvalues at each iteration
    convergence_metric = np.zeros(num_iterations)  # To measure convergence

    # Initial eigenvector guess, normalized
    X0 = np.array([[2], [1]])
    X[:, 0] = X0.flatten() / np.linalg.norm(X0)

    # Initial eigenvalue approximation
    eigenvalues[0] = np.dot(X[:, 0].T, np.dot(B, X[:, 0]))

    # Initial convergence metric
    convergence_metric[0] = np.amax(np.abs(X[:, 0] - X0.flatten()))

    # Inverse power method iterations
    for i in range(1, num_iterations):
        X[:, i] = np.dot(B, X[:, i - 1]) / np.linalg.norm(np.dot(B, X[:, i - 1]))
        eigenvalues[i] = np.dot(X[:, i].T, np.dot(B, X[:, i])) / np.dot(X[:, i].T, X[:, i])
        convergence_metric[i] = np.amax(np.abs(X[:, i] - X[:, i - 1]))

    smallest_eigenvalue = 1 / eigenvalues[-1]
    corresponding_eigenvector = X[:, -1]

    return smallest_eigenvalue, corresponding_eigenvector, convergence_metric


# Inputs for matrix A and num_iterations
A = np.array([[-5, 2], [-7, 4]])
num_iterations = 20
smallest_eigenvalue, corresponding_eigenvector, convergence_metric = inverse_power_method(A, num_iterations)
print("The smallest eigenvalue is", smallest_eigenvalue)
print("The eigenvector corresponding to the smallest eigenvalue is", corresponding_eigenvector)

# Plotting
plt.figure()
plt.plot(np.arange(num_iterations), convergence_metric, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Convergence Metric')
plt.title('Convergence of Inverse Power Method')
plt.grid()
plt.show()