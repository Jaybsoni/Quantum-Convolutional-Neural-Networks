import numpy as np
from hamiltonian import *



##### Tests
test = np.kron(np.kron(Z, I), I)
res = find_kron(Z, 1, 3)
np.testing.assert_array_equal(test, res)

test = np.kron(np.kron(I, Z), I)
res = find_kron(Z, 2, 3)
np.testing.assert_array_equal(test, res)

test = np.kron(np.kron(np.kron(I, I), I), Z)
res = find_kron(Z, 4, 4)
np.testing.assert_array_equal(test, res)

test = np.kron(np.kron(I, I), Z)
res = find_kron(Z, 3, 3)
np.testing.assert_array_equal(test, res)

test = np.kron(np.kron(I, I), X)
res = find_kron(X, 3, 3)
np.testing.assert_array_equal(test, res)