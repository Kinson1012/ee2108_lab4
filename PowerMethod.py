import matplotlib.pyplot as plt
import numpy as np


def power_method(A, num_iterations=20):
    """
    Calculate the dominant eigenvalue and corresponding eigenvector of matrix A
    using the power method.
    Parameters:
    - A: numpy array, square matrix
    - num_iterations: int, number of iterations to perform
    Returns:
    - dominant_eigenvalue: float, the dominant eigenvalue of A
    - dominant_eigenvector: numpy array, the eigenvector corresponding to the dominant
    eigenvalue
    - convergence_metric: numpy array, convergence metric over iterations
    """
    # Initialize variables
    shape = (A.shape[0], num_iterations)
    X = np.zeros(shape)  # To store eigenvectors at each iteration
    eigenvalues = np.zeros(num_iterations)  # To store eigenvalues at each iteration
    convergence_metric = np.zeros(num_iterations)  # To measure convergence

    # Initial eigenvector guess, normalized
    X0 = np.array([[2], [1]])
    X[:, 0] = X0.flatten() / np.linalg.norm(X0)

    # Initial eigenvalue approximation
    eigenvalues[0] = np.dot(X[:, 0].T, np.dot(A, X[:, 0]))

    # Initial convergence metric
    convergence_metric[0] = np.amax(np.abs(X[:, 0] - X0.flatten()))

    # Power method iterations
    for i in range(1, num_iterations):
        X[:, i] = np.dot(A, X[:, i - 1]) / np.linalg.norm(np.dot(A, X[:, i - 1]))
        eigenvalues[i] = np.dot(X[:, i].T, np.dot(A, X[:, i])) / np.dot(X[:, i].T, X[:, i])
        convergence_metric[i] = np.amax(np.abs(X[:, i] - X[:, i - 1]))

    dominant_eigenvalue = eigenvalues[-1]
    dominant_eigenvector = X[:, -1]

    return dominant_eigenvalue, dominant_eigenvector, convergence_metric


# Inputs for matrix A and num_iterations
A = np.array([[7, -2], [-2, 7]])
num_iterations = 20
dominant_eigenvalue, dominant_eigenvector, convergence_metric = power_method(A, num_iterations)
print("The largest eigenvalue is", dominant_eigenvalue)
print("The eigenvector corresponding to the largest eigenvalue is", dominant_eigenvector)

# Plotting
plt.figure()
plt.plot(np.arange(num_iterations), convergence_metric, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Convergence Metric')
plt.title('Convergence of Power Method')
plt.grid()
plt.show()
