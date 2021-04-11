import numpy as np
import scipy.sparse as sparse

import hamiltonian as H
import hamiltonianOld as HO
import hamiltonianVeryOld as HVO

X = np.array([[0, 1], [1, 0]], dtype=int)
Z = np.array([[1, 0], [0, -1]], dtype=int)
I = np.array([[1, 0], [0, 1]])
II = sparse.dia_matrix((np.ones(2), np.array([0])), dtype=int, shape=(2, 2))

##### Tests
test = np.kron(np.kron(Z, I), I)
res1 = H.find_kron_no_np(Z, 1, 3).toarray()
res2 = HO.find_kron_faster(Z, 1, 3)
print(np.array_equal(test, res1), np.array_equal(test, res2))

test = np.kron(np.kron(I, Z), I)
res1 = H.find_kron_no_np(Z, 2, 3).toarray()
res2 = HO.find_kron_faster(Z, 2, 3)
print(np.array_equal(test, res1), np.array_equal(test, res2))

test = np.kron(np.kron(np.kron(I, I), I), Z)
res1 = H.find_kron_no_np(Z, 4, 4).toarray()
res2 = HO.find_kron_faster(Z, 4, 4)
print(np.array_equal(test, res1), np.array_equal(test, res2))

test = np.kron(np.kron(I, I), Z)
res1 = H.find_kron_no_np(Z, 3, 3).toarray()
res2 = HO.find_kron_faster(Z, 3, 3)
print(np.array_equal(test, res1), np.array_equal(test, res2))

test = np.kron(np.kron(I, I), X)
res1 = H.find_kron_no_np(X, 3, 3).toarray()
res2 = HO.find_kron_faster(X, 3, 3)
print(np.array_equal(test, res1), np.array_equal(test, res2))

n=8
h1 = (0, 1.6)
h2 = (-1.6, 1.6)
H_Ham = H.Hamiltonian(n, "_trainNEW1", h1, h2)
HO_Ham = HO.HamiltonianOld(n, h1, h2)

H_Ham.get_first_term_faster()
HO_Ham.get_first_term()
print(np.array_equal(H_Ham.first_term, HO_Ham.first_term))

H_Ham.get_second_term()
HO_Ham.get_second_term()
print(np.array_equal(H_Ham.second_term, HO_Ham.second_term))

H_Ham.get_third_term_faster()
HO_Ham.get_third_term()
print(np.array_equal(H_Ham.third_term, HO_Ham.third_term))


vects_new = H_Ham.generate_data(64, 64)
vects_old  = HO_Ham.generate_train_data(64, 64)

for a, b in zip(vects_new, vects_old):
    eigenvalues_new, eigenvectors_new, HH_new = a
    eigenvalues_old, eigenvectors_old, HH_old = b
    print("+")
    # print(HH_new)
    # print(HH_old)
    print(np.array_equal(np.array(HH_old), np.array(HH_new)))
    vals_same = np.isclose(np.array(eigenvalues_old), np.array(eigenvalues_new), atol=1e-9)
    print(vals_same)
    if not vals_same:
        print("---")
        print(eigenvalues_old[0], eigenvalues_new[0])


    # print(np.array_equal(np.array(eigenvectors_old), np.array(eigenvectors_new)))

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