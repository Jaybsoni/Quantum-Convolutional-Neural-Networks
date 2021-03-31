import numpy as np
from numpy import linalg as LA


x = np.arange(225).reshape((15, 15))
print("Diagonalized x is: ", np.diag(x))

w, v = LA.eig(x)
index = np.where(w == np.amax(w))
print("Max eigenvalue is: ", w[index])
print("Corresponding max eigenvector is: ", v[index])

