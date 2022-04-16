import os
import math
import numpy as np
import pickle
from alive_progress import alive_bar

cwd = os.getcwd()

def isvd(matrix, k):
    threshold = 0.005
    m, n = matrix.shape

    B = np.matmul(matrix.T, matrix)

    # Allocate space
    U = np.zeros([m, k])
    SIGMA = np.zeros([k, k])
    V = np.zeros([n, k])

    # Calculate largest k eigenvectors of ATA
    with alive_bar(k) as bar:
        for i in range(k):
            # Generate random unit vector
            u = np.random.rand(n, 1)
            u = (1 / np.linalg.norm(u)) * u

            # Store last u vector for comparison
            u_last = np.zeros([n, 1])

            while np.linalg.norm(np.subtract(u, u_last)) > threshold:
                # Eigenvector algorithm
                u_last = u
                v = np.matmul(B, u)
                u = (1 / np.linalg.norm(v)) * v

                # Gram-Schmidt algorithm
                for j in range(i - 1):
                    u = np.subtract(u, np.dot(u.T, V[:, [j]]) * V[:, [j]])

                u = (1 / np.linalg.norm(u)) * u

            # Store eigenvector and compute singular value
            V[:, [i]] = u
            SIGMA[i, i] = math.sqrt(np.linalg.norm(np.matmul(B, u)) / np.linalg.norm(u))

            bar()

    # Populate U
    for i in range(k):
        U[:, [i]] = (1 / SIGMA[i, i]) * np.matmul(A, V[:, [i]])

    return U, SIGMA, V

# Load matrix
with open(cwd + "/matrices/A.pickle", "rb") as f:
    A = pickle.load(f)

# Compute SVD factorization
print("Calculating SVD factorization...")
U, S, V = isvd(A, 100)

# Save matrix factors
with open(cwd + "/matrices/U.pickle", "wb") as f:
    pickle.dump(U, f)
with open(cwd + "/matrices/SIGMA.pickle", "wb") as f:
    pickle.dump(S, f)
with open(cwd + "/matrices/V.pickle", "wb") as f:
    pickle.dump(V, f)
