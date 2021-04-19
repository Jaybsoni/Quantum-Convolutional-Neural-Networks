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
def test_kron():
    """
    Calculates the correct kron with nothing but np.kron, directly. Then calls
    both the current method to calculate kron (H.find_kron()), and the previous
    example (HO.find_kron_faster) (no longer fastest method), and makes sure they
    all are equate to the absolute answer
    """
    test = np.kron(np.kron(Z, I), I)
    res1 = H.find_kron(Z, 1, 3).toarray()
    res2 = HO.find_kron_faster(Z, 1, 3)
    print(np.array_equal(test, res1), np.array_equal(test, res2))

    test = np.kron(np.kron(I, Z), I)
    res1 = H.find_kron(Z, 2, 3).toarray()
    res2 = HO.find_kron_faster(Z, 2, 3)
    print(np.array_equal(test, res1), np.array_equal(test, res2))

    test = np.kron(np.kron(np.kron(I, I), I), Z)
    res1 = H.find_kron(Z, 4, 4).toarray()
    res2 = HO.find_kron_faster(Z, 4, 4)
    print(np.array_equal(test, res1), np.array_equal(test, res2))

    test = np.kron(np.kron(I, I), Z)
    res1 = H.find_kron(Z, 3, 3).toarray()
    res2 = HO.find_kron_faster(Z, 3, 3)
    print(np.array_equal(test, res1), np.array_equal(test, res2))

    test = np.kron(np.kron(I, I), X)
    res1 = H.find_kron(X, 3, 3).toarray()
    res2 = HO.find_kron_faster(X, 3, 3)
    print(np.array_equal(test, res1), np.array_equal(test, res2))

def check_terms():
    """
    Creates an instance of the current version of Hamiltonian, and
    the old version, and confirms their three terms equate. Useful
    for fast development and catching mistakes in development of
    Hamiltonian
    """
    n=8
    h1 = (0, 1.6)
    h2 = (-1.6, 1.6)
    H_Ham = H.Hamiltonian(n, "_trainNEW1", h1, h2)
    HO_Ham = HO.HamiltonianOld(n, h1, h2)

    H_Ham.get_first_term()
    HO_Ham.get_first_term()
    print(np.array_equal(H_Ham.first_term, HO_Ham.first_term))

    H_Ham.get_second_term()
    HO_Ham.get_second_term()
    print(np.array_equal(H_Ham.second_term, HO_Ham.second_term))

    H_Ham.get_third_term()
    HO_Ham.get_third_term()
    print(np.array_equal(H_Ham.third_term, HO_Ham.third_term))


def test_dataset(n, dataset_file, tol):
    """
    Reads a datasets file and confirms the values for all lines (rows) are valid
    eigenvectors for the values of h1, h2, given at the beginning of the row.
    Uses np.eig to confirm, and is very slow at larger n values. This is just for
    testing to confirm our eigenvectors are valid energy states
    :param n: int - number of qbits
    :param dataset_file: str - location of file to check
    :param tol: float - absolute tolerance to accept
    :return:
    """
    H_Ham = H.Hamiltonian(n, (0, 1.6), (-1.6, 1.6), v=0)
    H_Ham.get_first_term()
    H_Ham.get_second_term()
    H_Ham.get_third_term()
    # H_Ham.calculate_terms()

    h1h2_ours, eigvecs = H.read_eigenvectors(dataset_file)
    print("Checking file ", dataset_file)
    misses = 0
    for i, (h1, h2) in enumerate(h1h2_ours):
        computed_hamiltonian = H_Ham.first_term + (H_Ham.second_term * h1) + (H_Ham.third_term * h2)

        eigenvalues, _ = H_Ham.find_eigval_with_np(computed_hamiltonian)
        test = (computed_hamiltonian @ eigvecs[i]) / eigenvalues
        real = np.array(eigvecs[i], dtype=complex)
        # print(misses)
        if not np.allclose(test, real, tol):
            misses += 1
    print("Total eigenvectors that are misses is:", misses)
