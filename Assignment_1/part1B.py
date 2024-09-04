import numpy as np

def initialise_input(N, d):
    '''
    N: Number of vectors
    d: Dimension of vectors
    '''
    np.random.seed(0)
    U = np.random.randn(N, d)
    M1 = np.abs(np.random.randn(d, d))
    M2 = np.abs(np.random.randn(d, d))

    return U, M1, M2

def solve(N, d):
    U, M1, M2 = initialise_input(N, d)

    # Step 1: Compute X and Y
    X = np.dot(U, M1)
    Y = np.dot(U, M2)
    print("Matrix X:\n", X)
    print("\n")
    print("Matrix Y:\n", Y)
    print("\n")
    
    # Step 2: Modify X to create X_hat
    row_indices = np.arange(1, N + 1)
    X_hat = X + row_indices[:, np.newaxis]
    print("Matrix X_hat:\n", X_hat)
    print("\n")
    # Step 3: Compute Z and apply the sparsify operation
    Z = np.dot(X_hat, Y.T)
    print("Matrix Z:\n", Z)
    print("\n")
    # Apply sparsify operation
    mask = np.eye(Z.shape[0], dtype=bool)  # Create a mask for keeping diagonal elements
    Z_sparsified = np.where(mask, Z, 0)    # Apply mask to zero out non-diagonal elements
    print("Matrix Z_sparsified:\n",Z_sparsified)
    print("\n")
    # Step 4: Compute softmax for Z
    exp_Z = np.exp(Z_sparsified - np.max(Z_sparsified, axis=1, keepdims=True))  # for numerical stability
    softmax_Z = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    print("Matrix softmax_Z:\n", softmax_Z)
    print("\n")
    # Step 5: Get the index of the maximum probability in each row of softmax_Z
    max_indices = np.argmax(softmax_Z, axis=1)
    print("the maximum probability is: ", max_indices)
    print("\n")
    return max_indices

# Example usage
N = 4  # Number of column vectors
d = 3  # Dimension of each vector
solve(N, d)
