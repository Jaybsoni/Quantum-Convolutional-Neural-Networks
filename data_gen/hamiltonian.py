import os
import sys
import time
import tqdm
import psutil
import itertools
import numpy as np
import multiprocessing as mp
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


def find_kron_no_np(array, index, n):
    # before_w = psutil.virtual_memory().used / 1024 ** 2
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

    return t.copy()


class Hamiltonian:
    def __init__(self, n=2, filename="", h1_metadata=(0, 1.6), h2_metadata=(-1.6, 1.6)):
        self.n = n
        self.h1_min, self.h1_max = h1_metadata
        self.h2_min, self.h2_max = h2_metadata

        self.size = pow(2, self.n)
        before_w = psutil.virtual_memory().used / 1024 ** 2  # MB
        self.first_term = np.zeros(shape=(self.size, self.size), dtype=float)
        self.second_term = np.zeros(shape=(self.size, self.size), dtype=float)
        self.third_term = np.zeros(shape=(self.size, self.size), dtype=float)

        # Delete the output file if exists so we can append to a fresh ones.
        self.filename = f'dataset_n={n}_' + filename + ".txt"
        if os.path.isfile(self.filename): os.remove(self.filename)

    def get_first_term_faster(self):
        self.first_term = np.zeros(shape=(self.size, self.size), dtype=float)
        for i in range(self.n - 2):
            # print(f"first term {i}/{self.n - 2}")
            elem = i + 1  # math element is indexes at 1

            AA = find_kron_no_np(Z, elem, self.n)
            BB = find_kron_no_np(X, elem + 1, self.n)
            CC = find_kron_no_np(Z, elem + 2, self.n)

            # Instead of A.dot(B).dot(C), note that A and C are diagonal with 1's and -1's.
            # Convert A, C into vectors and just multiply them to B, a 45 000% speedup over dot product for n=14
            a_diag = AA.diagonal()[..., None]
            c_diag = CC.diagonal()[..., None]
            ss_fast = BB.multiply(a_diag).multiply(c_diag)

            self.first_term -= ss_fast.toarray()

    def get_second_term(self):
        self.second_term = np.zeros(shape=(self.size, self.size), dtype=float)
        for i in range(self.n):
            # print(f"second term {i}/{self.n}")
            self.second_term -= find_kron_no_np(X, i+1, self.n).toarray()

    def get_third_term_faster(self):
        self.third_term = np.zeros(shape=(self.size, self.size), dtype=float)
        """
        Honestly? This method is magic. It. Just. Works. Don't believe me? Try to make it faster....
        :return:
        """
        for i in range(self.n - 1):  # This is actually 1 to N-2, python indexing has self.n-1
            # print(f"third term {i}/{self.n-1}")
            
            elem = i + 1  # math element is indexes at 1

            B1 = find_kron_no_np(X, elem, self.n)
            B2 = find_kron_no_np(X, elem + 1, self.n)

            B1_rows, B1_cols = sparse.coo_matrix(B1, dtype=sparse.coo_matrix).nonzero()
            B2_rows = B1_cols
            B2_cols0 = []

            def extract_elem(elem):
                B2_cols0.append(elem)
            list(map(B2.getrow, filter(extract_elem, B2_rows)))

            # Good = []  # Temp!
            # for val in B2_rows:
            #     B2_col = str(B2.getrow(val)).split(")")[0].split(" ")[-1]
            #     # print(val, B2_cols[elem], B2_col)
            #     Good.append(int(B2_col))

            flat_list = []
            swaps = int(pow(2, self.n - 1 - elem))  # How many groups the list should be seperated into
            groups = [B2_cols0[i:i + swaps] for i in range(0, len(B2_cols0), swaps)]  # respective sections
            for i in range(int(len(groups) / 2)):  # Preform the swaps
                switch = groups[2 * i:(2 * i) + 2]
                flat_list.append([*switch[1], *switch[0]])
            # a = [[*groups[2 * i:(2 * i) + 2][1], *groups[2 * i:(2 * i) + 2][0]] for i in range(int(len(groups) / 2))]
            flat_list = list(itertools.chain(*flat_list))

            size = len(B1_rows)
            coo = sparse.coo_matrix((np.ones(size, dtype=int), (B1_rows, flat_list)), shape=(size, size))

            # Old method to compare to to confirm dot product is done correctly
            # ss_slow = (B1.dot(B2)).toarray()
            # print(f"dot vs new are equal?", np.array_equal(ss_slow, coo.toarray()))
            # assert np.array_equal(ss_slow, coo.toarray())
            self.third_term -= coo.toarray()

    def convert_sec(self, t):
        min = np.floor(t/60)
        sec = round(t % 60, 2)
        return "{}m-{:0.2f}s".format(int(min), sec)

    def calculate_time_remaining(self, n, t0, i):
        time_remaning = ((1 - (i/n)) * (time.time() - t0))  * (n/i)
        percentage = (i / n) * 100
        print("{:0.2f}% \tElapsed: {} \tRemaining: {}".format(percentage, self.convert_sec(time.time() - t0), self.convert_sec(time_remaning)))

    def generate_data(self, h1_range, h2_range):
        s = time.time()
        self.get_first_term_faster()
        self.get_second_term()
        self.get_first_term_faster()
        print(time.time() - s)

        h1h2 = [[h1, h2] for h1 in np.linspace(self.h1_min, self.h1_max, h1_range)
                for h2 in np.linspace(self.h2_min, self.h2_max, h2_range)]

        s = time.time()
        i = 1
        vects = []
        # for h1, h2 in h1h2:
        for h1 in np.linspace(self.h1_min, self.h1_max, h1_range):
            for h2 in np.linspace(self.h2_min, self.h2_max, h2_range):
                if h2_range == 1:
                    h2 = 0  # TODO Jay: Do I not need this?

                H = self.first_term + (self.second_term * h1) + (self.third_term * h2)
                eigenvalues, eigenvectors = self.find_eigval(H)
                if h1 == 0.0 and h2 == -1.6:
                    self.test_dataset(H, eigenvalues)  # SLOW!
                vects.append([eigenvalues, eigenvectors, H])

                # Write to file each time to avoid saving to ram
                with open(self.filename, 'a+') as f:
                    f.write(f"{h1, h2}_")  # Append h1, h2 for reference
                    for line in eigenvectors: f.write(str(line) + " ")
                    f.write("\n")

            # i += 1
            # if i % 10 == 0:
            #     self.calculate_time_remaining(h1_range * h2_range, s, i)

        return vects
    # def pool_func_for_mp(self, h1h2):
    #     h1, h2 = h1h2
    #     # print(h1, h2)
    #     H = self.first_term + (self.second_term * h1) + (self.third_term * h2)
    #     eigenvalues, eigenvectors = self.find_eigval(H)
    #     # self.test_dataset(H, eigenvalues)  # SLOW!
    #     return h1, h2, eigenvectors
    # def generate_train_data_with_mp(self, h1_range, h2_range):
    #     s = time.time()
    #     self.get_first_term_faster()
    #     self.get_second_term()
    #     self.get_first_term_faster()
    #     print(time.time() - s)
    #
    #     h1h2 = [[h1, h2] for h1 in np.linspace(self.h1_min, self.h1_max, h1_range)
    #             for h2 in np.linspace(self.h2_min, self.h2_max, h2_range)]
    #
    #     p = mp.Pool(mp.cpu_count())
    #     MS = list(tqdm.tqdm(p.imap(self.pool_func_for_mp, h1h2), total=len(h1h2)))
    #     # MS = p.map(self.pool_func_for_mp, h1h2)
    #     print(sys.getsizeof(MS))
    #     for h1, h2, eigenvector in MS:
    #         # Write to file each time to avoid saving to ram
    #         with open(self.filename, 'a+') as f:
    #             f.write(f"{h1, h2}_")  # Append h1, h2 for reference
    #             for line in eigenvector: f.write(str(line) + " ")
    #             f.write("\n")
    #     # gradient_mat = copy.deepcopy(self.params)

    @staticmethod
    def find_eigval(H):
        b, c = sparse.linalg.eigs(H, k=1, which='SR', tol=1e-4)
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


        # # sum_vec is the sum of all vectors returned by np.linalg.eig (the slow, 'right' one)
        b, c = sparse.linalg.eigs(H, k=1, which='SR', tol=1e-16)

        # Test they're the same
        sum_vec = np.sum(eigValsList, axis=0)

        slowVectMag = sum_vec / np.linalg.norm(sum_vec)
        aa = (H @ slowVectMag) / possible_eigenvalues
        assert np.allclose(aa, np.array(slowVectMag, dtype=complex), 1e-9)

        # Tests the inverse way too? TODO: JAAYYYYY
        fastVectMag = c.flatten() / np.linalg.norm(c.flatten())
        assert np.allclose((H @ fastVectMag) / npEigVal[0], np.array(fastVectMag, dtype=complex), 1e-9)




X = np.array([[0, 1], [1, 0]], dtype=int)
Z = np.array([[1, 0], [0, -1]], dtype=int)
II = sparse.dia_matrix((np.ones(2), np.array([0])), dtype=int, shape=(2, 2))

if __name__ == '__main__':
    s = time.time()

    filename = "test"

    h1 = (0, 1.6)
    h2 = (-1.6, 1.6)
    H = Hamiltonian(8, filename, h1, h2)
    H.generate_data(64, 32)

    filename = "train"
    H = Hamiltonian(8, filename, h1, h2)
    H.generate_data(40, 1)
    # print(find_kron_no_np.cache_info())

    print(f"Time for creating dataset was {time.time() - s} seconds")

    #
    # h1h2_old, old = read_eigenvectors('example_test_data_n8.txt')
    # h1h2_new, new = read_eigenvectors('dataset_n=8_test.txt')
    #
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

    # flipped_pred_mat = new.reshape((64, 64), order='F')
    # pred_mat = []
    # for row_index in np.arange(len(flipped_pred_mat) - 1, -1, -1):
    #     pred_mat.append(flipped_pred_mat[row_index])
    #
    # pred_mat = np.array(pred_mat)
