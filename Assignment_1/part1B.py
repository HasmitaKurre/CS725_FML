import numpy as np

def initialise_input(N, d):
  '''
  N: Number of vectors
  d: dimension of vectors
  '''
  np.random.seed(0)
  U = np.random.randn(N, d)
  M1 = np.abs(np.random.randn(d, d))
  M2 = np.abs(np.random.randn(d, d))

  return U, M1, M2

def solve(N, d):
  U, M1, M2 = initialise_input(N, d)

  '''
  Enter your code here for steps 1 to 6
  '''
  P = np.dot(U, M1)
 
  Q = np.dot(U, M2)
   
  P_hat = P + np.arange(N).reshape(-1, 1)
   
  R = np.dot(P_hat, Q.T)
  ##print(R)
  N = R.shape[0]
  i_indices = np.arange(N).reshape(-1, 1)  # Shape (N, 1)
  j_indices = np.arange(N).reshape(1, -1)  # Shape (1, N)

  mask = (i_indices % 2) == (j_indices % 2)

  R = R * mask  
  ##print(R)

  R_exp = np.exp(R - np.max(R, axis=1, keepdims=True))
  R_hat = R_exp / R_exp.sum(axis=1, keepdims=True)
   
  max_indices = np.argmax(R_hat, axis=1)

  return max_indices


N = int(input("Enter an Integer for N :"))
d = int(input("Enter an Integer for d :"))

print(solve(N,d))



