import os
import sys
import time
import psutil
import itertools
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from functools import lru_cache, wraps


def np_cache(*args, **kwargs):
    def decorator(function):
        @wraps(function)
        def wrapper(np_array, *args, **kwargs):
            hashable_array = array_to_tuple(np_array)
            # hashable_array = tuple(map(tuple, np_array))
            return cached_wrapper(hashable_array, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(hashable_array, *args, **kwargs):
            array = np.array(hashable_array)
            return function(array, *args, **kwargs)

        def array_to_tuple(np_array):
            try:
                return tuple(array_to_tuple(_) for _ in np_array)
            except TypeError:
                return np_array

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper
    return decorator


def read_eigenvectors(file):
    with open(file, 'r+') as f:
        textData = f.readlines()

        h_vals = []
        for i in range(len(textData)):
            h1h2, eigenvector = textData[i].split("_")

            h_vals.append(tuple(map(float, h1h2[1: -1].split(', '))))
            textData[i] = eigenvector

        return h_vals, np.loadtxt(textData, dtype=complex)


# @np_cache(maxsize=1024)
def find_kron_faster(array, index, n):
    before_w = psutil.virtual_memory().used / 1024 ** 2
    if np.array_equal(array, X):
        array = sparse.dia_matrix((np.array([np.ones(1)]), np.array([-1])), dtype=int, shape=(2, 2))
        array.setdiag(np.ones(1), 1)
    elif np.array_equal(array, Z):
        array = sparse.dia_matrix((np.array([1, -1]), np.array([0])), dtype=int, shape=(2, 2))

    assert index <= n  # n elements should always be larger than index for array
    t = sparse.dia_matrix((pow(2, n), pow(2, n)), dtype=int)

    # Creates a list of 1's setting the index value as 0 to represent the array parameter given
    order = np.ones(n)
    order[index-1] = 0

    for i in range(1, len(order)):
        # Sets next element to Identity if next element is a 1, if zero, then array
        current = array if order[i] == 0 else II

        # print(i, len(order))
        if i == 1:
            # First time - compute kron(j-1, j)
            last = array if order[i-1] == 0 else II
            t = sparse.kron(last, current)

        else:  # Computes kron of last element current matrix with next element
            t = sparse.kron(t, current)

    return t.toarray().copy()



class HamiltonianOld:
    def __init__(self, n=2, h1_metadata=(0, 1.6), h2_metadata=(-1.6, 1.6)):
        self.n = n
        self.h1_min, self.h1_max = h1_metadata
        self.h2_min, self.h2_max = h2_metadata

        self.size = pow(2, self.n)
        before_w = psutil.virtual_memory().used / 1024 ** 2  # MB
        self.first_term = np.zeros(shape=(self.size, self.size), dtype=float)
        self.second_term = np.zeros(shape=(self.size, self.size), dtype=float)
        self.third_term = np.zeros(shape=(self.size, self.size), dtype=float)

        # Delete the output file if exists so we can append to a fresh ones.
        self.filename = f'dataset_n={n}'

    def get_first_term(self):
        self.first_term = np.zeros(shape=(self.size, self.size), dtype=float)

        for i in range(self.n - 2):
            # print("first term", i)
            elem = i + 1  # math element is indexes at 1

            a_diag = np.diag(np.array(find_kron_faster(Z, elem, self.n)))
            A = sparse.dia_matrix((a_diag, np.array([0])), shape=(self.size, self.size))
            B = self.create_matrix_of_X(elem + 1)

            c_diag = np.diag(np.array(find_kron_faster(Z, elem + 2, self.n)))
            C = sparse.dia_matrix((c_diag, np.array([0])), shape=(self.size, self.size))

            d =  (A.dot(B)).dot(C)
            self.first_term -= d

            del A, B, C, a_diag, c_diag
            # print('get_first_term_fast after:', (psutil.virtual_memory().used / 1024 ** 2) - before_w, 'MB')
        return self.first_term

    def create_matrix_of_X(self, elem):
        full_array = np.array(find_kron_faster(X, elem, self.n))
        diag_index = np.where(full_array[0] == 1)[0][0]
        data = np.array([np.diag(full_array, -diag_index)])
        A = sparse.dia_matrix((data, np.array([-diag_index])), shape=(self.size, self.size))

        # Set second diagonal separately due to stupid indexing of sparse.dia_matrix
        A.setdiag(np.diag(full_array, diag_index), diag_index)
        assert np.array_equal(full_array, A.toarray())
        del full_array, diag_index, data  # Delete arrays since they're huge
        return A

    def get_second_term(self):
        self.second_term = np.zeros(shape=(self.size, self.size), dtype=float)

        for i in range(self.n):
            # print(f"second term {i}/{self.n}")
            self.second_term -= find_kron_faster(X, i+1, self.n)


    def get_third_term(self):
        self.third_term = np.zeros(shape=(self.size, self.size), dtype=float)

        for i in range(self.n - 1):  # This is actually 1 to N-2, python indexing has self.n-1
            # print(f"third term {i}/{self.n-1}")
            elem = i + 1  # math element is indexes at 1

            A = self.create_matrix_of_X(elem)
            B = self.create_matrix_of_X(elem + 1)
            self.third_term -= (A.dot(B)).toarray()

    def convert_sec(self, t):
        min = np.floor(t/60)
        sec = round(t % 60, 2)
        return "{}m-{:0.2f}s".format(int(min), sec)

    def calculate_time_remaining(self, n, t0, i):
        time_remaning = ((1 - (i/n)) * (time.time() - t0))  * (n/i)
        percentage = (i / n) * 100
        print("{:0.2f}% \tElapsed: {} \tRemaining: {}".format(percentage, self.convert_sec(time.time() - t0), self.convert_sec(time_remaning)))

    def generate_train_data(self, h1_range, h2_range):
        filename = self.filename + "_trainOLD1.txt"
        if os.path.isfile(filename): os.remove(filename)

        s = time.time()
        self.get_first_term()
        self.get_second_term()
        self.get_third_term()
        print(time.time() - s)

        s = time.time()
        i = 1
        vects = []
        for h1 in np.linspace(self.h1_min, self.h1_max, h1_range):
            for h2 in np.linspace(self.h2_min, self.h2_max, h2_range):
                H = self.first_term + (self.second_term * h1) + (self.third_term * h2)
                H = np.array(H)

                eigenvalues, eigenvectors = self.find_eigval(H)
                self.test_dataset(H, eigenvalues)
                vects.append([eigenvalues, eigenvectors, H])

                print(self.third_term)
                # Write to file each time to avoid saving to ram
                self.write_to_file(filename, h1, h2, eigenvectors)
                # i += 1
                # if i % 1 == 0:
                #     self.calculate_time_remaining(h1_range * h2_range, s, i)
        return vects

    def generate_test_data(self, h1_range):
        filename = self.filename + "_test.txt"
        if os.path.isfile(filename): os.remove(filename)

        self.get_first_term()
        self.get_second_term()
        self.get_third_term()

        for h1 in np.linspace(self.h1_min, self.h1_max, h1_range):
            H = self.first_term + (self.second_term * h1)  # h2 = 0, third term removed
            eigenvalues, eigenvectors = self.find_eigval(H)
            # self.test_dataset(H, eigenvalues)

            # Write to file each time to avoid saving to ram
            self.write_to_file(filename, h1, 0, eigenvectors)


    def write_to_file(self, filename, h1, h2, eigenvectors):
        with open(filename, 'a+') as f:
            f.write(f"{h1, h2}_")  # Append h1, h2 for reference
            for line in eigenvectors: f.write(str(line) + " ")
            f.write("\n")

    @staticmethod
    def find_eigval(H):
        b, c = sparse.linalg.eigs(H, k=1, which='SR', tol=1e-16)
        return b, c.flatten()

    def test_dataset(self, H, possible_eigenvalues):
        ww, vv = np.linalg.eig(H)  # Old method with linalg
        index = np.where(ww == np.amin(ww))
        npEigVal, npEigVec = ww[index], vv[:, index]

        """
        np.linalg.eig returns the eigenvalues and vectors of a matrix
        BUT, it returns a list of lists of lists, where the elements of
        each triple nested list is the first element of each eigenvector,
        not a list of eigenvectors like any sensical person would return.
        """ # np.linalg.eig is grade A stupid, change my mind...
        eigValsList = []  # Converts np.eig to an output that's actually usable
        for eigVal in range(len(npEigVal)):
            tempVec = []

            for eigVec in range(len(npEigVec)):
                tempVec.append(npEigVec[eigVec][0][eigVal])
            eigValsList.append(np.array(tempVec))

        # Test they're the same
        sum_vec = np.sum(eigValsList, axis=0)
        slowVectMag = sum_vec / np.linalg.norm(sum_vec)

        aa = (H @ slowVectMag) / possible_eigenvalues
        is_same = np.allclose(aa, np.array(slowVectMag, dtype=complex), 1e-9)

        if not is_same:
            print(possible_eigenvalues, npEigVal)
        assert is_same

X = np.array([[0, 1], [1, 0]], dtype=int)
Z = np.array([[1, 0], [0, -1]], dtype=int)
II = sparse.dia_matrix((np.ones(2), np.array([0])), dtype=int, shape=(2, 2))

if __name__ == '__main__':
    s = time.time()
    h1 = (0, 1.6)
    h2 = (-1.6, 1.6)
    H = HamiltonianOld(8, h1, h2)
    H.generate_train_data(64, 64)
    # H.generate_test_data(32)
    # print(find_kron_no_np.cache_info())

    print(f"Time for creating dataset was {time.time() - s} seconds")
