import numpy as np
from hamiltonian import *

X = np.array([[0, 1], [1, 0]], dtype=int)
Z = np.array([[1, 0], [0, -1]], dtype=int)
II = sparse.dia_matrix((np.ones(2), np.array([0])), dtype=int, shape=(2, 2))

##### Tests
test = np.kron(np.kron(Z, II), II)
res = find_kron_no_np(Z, 1, 3).toarray()
np.testing.assert_array_equal(test, res)

test = np.kron(np.kron(II, Z), II)
res = find_kron_no_np(Z, 2, 3).toarray()
np.testing.assert_array_equal(test, res)

test = np.kron(np.kron(np.kron(II, II), II), Z)
res = find_kron_no_np(Z, 4, 4).toarray()
np.testing.assert_array_equal(test, res)

test = np.kron(np.kron(II, II), Z)
res = find_kron_no_np(Z, 3, 3).toarray()
np.testing.assert_array_equal(test, res)

test = np.kron(np.kron(II, II), X)
res = find_kron_no_np(X, 3, 3).toarray()
np.testing.assert_array_equal(test, res)

# h1h2_old, old = read_eigenvectors('data/dataset_n=4_train_w_arrays.txt')
# h1h2_old, old = read_eigenvectors('data/dataset_n=4_train.txt')
# h1h2_new, new = read_eigenvectors('dataset_n=4_train.txt')


# print(np.array_equal(np.array(h1h2_old), np.array(h1h2_new)))
#
# print(old.shape)
# print(new.shape)
# print(np.allclose(old, new, 1))
# print(np.allclose(old, new, atol=1))
# i = 0
# for a, b in zip(old, new):
#     if not np.allclose(old, new, 1e2):
#         i += 1
#         print("-")
#         print(np.allclose(old, new, 1e2))
#         print(a, b)
# print(i)