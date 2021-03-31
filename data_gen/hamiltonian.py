import time, os
import pickle
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as lng
from functools import lru_cache, wraps
from numba import jit

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



X = np.array([[0, 1], [1, 0]], dtype=int)
Z = np.array([[1, 0], [0, -1]], dtype=int)
II = sparse.dia_matrix((np.ones(2), np.array([0])), dtype=int, shape=(2, 2))


@np_cache(maxsize=2048)
def find_kron_faster(array, index, n):
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



class HamiltonianFast:
    def __init__(self, n=2, h1_min=0, h1_max=1.6, h2_min=-1.6, h2_max=1.6):
        self.n = n
        self.h1_min = h1_min
        self.h1_max = h1_max
        self.h1_range = 32
        self.h2_min = h2_min
        self.h2_max = h2_max
        self.h2_range = 64

        self.size = pow(2, self.n)
        self.first_term = np.zeros(shape=(self.size, self.size), dtype=float)
        self.second_term = np.zeros(shape=(self.size, self.size), dtype=float)
        self.third_term = np.zeros(shape=(self.size, self.size), dtype=float)

        # Delete the output file if exists so we can append to a fresh one.
        self.filename = f'dataset_n={n}.txt'
        if os.path.isfile(self.filename):
            os.remove(self.filename)

    def get_first_term_fast(self):
        for i in range(self.n - 2):
            elem = i + 1  # math element is indexes at 1

            a_diag = np.diag(np.array(find_kron_faster(Z, elem, self.n)))
            A = sparse.dia_matrix((a_diag, np.array([0])), shape=(self.size, self.size))

            B = self.create_matrix_of_X(elem + 1)

            c_diag = np.diag(np.array(find_kron_faster(Z, elem + 2, self.n)))
            C = sparse.dia_matrix((c_diag, np.array([0])), shape=(self.size, self.size))
            self.first_term -= ((A.dot(B)).dot(C)).toarray()

    def create_matrix_of_X(self, elem):
        full_array = np.array(find_kron_faster(X, elem, self.n))
        diag_index = np.where(full_array[0] == 1)[0][0]
        data = np.array([np.diag(full_array, -diag_index)])
        A = sparse.dia_matrix((data, np.array([-diag_index])), shape=(self.size, self.size))

        # Set second diagonal separately due to stupid indexing of sparse.dia_matrix
        A.setdiag(np.diag(full_array, diag_index), diag_index)
        del full_array  # Delete array since it's huge

        return A

    def get_second_term(self):
        self.second_term = np.zeros(shape=(self.size, self.size), dtype=float)
        for i in range(self.n):
            self.second_term -= find_kron_faster(X, i+1, self.n)


    def get_third_term_fast(self):
        for i in range(self.n - 1):  # This is actually 1 to N-2, python indexing has self.n-1
            elem = i + 1  # math element is indexes at 1

            # a is slightly more messy
            A = self.create_matrix_of_X(elem)
            B = self.create_matrix_of_X(elem + 1)
            self.third_term -= (A.dot(B)).toarray()

    def convert_sec(self, t):
        min = np.floor(t/60)
        sec = round(t % 60, 2)
        return "{}m-{:0.2f}s".format(int(min), sec)

    def calculate_time_remaining(self, t0, i):
        n = self.h1_range * self.h2_range

        if i % 10  == 0:
            time_remaning = ((1 - (i/n)) * (time.time() - t0))  * (n/i)
            percentage = (i / n) * 100
            print("{:0.2f}% \tElapsed: {} \tRemaining: {}".format(percentage, self.convert_sec(time.time() - t0), self.convert_sec(time_remaning)))

    def calculate_hamiltonian(self):
        s = time.time()
        self.get_first_term_fast()
        self.get_second_term()
        self.get_third_term_fast()
        print(time.time() - s)

        s = time.time()
        i = 1
        for h1 in np.linspace(self.h1_min, self.h1_max, self.h1_range):
            for h2 in np.linspace(self.h2_min, self.h2_max, self.h2_range):
                self.calculate_time_remaining(s, i)
                i += 1


                H = self.first_term + (self.second_term * h1) + (self.third_term * h2)
                eigenvectors = self.find_eigval(H.astype(complex))

                # Write to file each time to avoid saving to ram
                with open(self.filename, 'a+') as f:
                    for line in eigenvectors:
                        f.write(str(line) + " ")
                    f.write("\n")

        print(f"added all terms in {time.time() - s} seconds")

    @staticmethod
    # @jit(nopython=True)
    def find_eigval(H):
        # OLD
        ww, vv = np.linalg.eig(H)
        index = np.where(ww == np.amin(ww))
        # print(H.shape)
        # print(vv.shape)
        # print(index[0][0])
        # print(vv[index][0])


        # New
        # w, v = sparse.linalg.eigs(H, k=512, which="SR")
        # print(v.shape)
        # index = np.where(w == np.amin(w))
        # print(v[index][0])
        # print(len(vv[index][0]), len(v[index][0]))
        # print(vv[index][0])
        # print("_")
        # print(v[index][0])
        return vv[index][0]




# s = time.time()
# H = HamiltonianFast(8, 0, 1.6, -1.6, 1.6).calculate_hamiltonian()
# print(find_kron_faster.cache_info())
# print(f"Time for caching took {time.time() -s} seconds")


###
new_n8 = np.loadtxt('dataset_n=8.txt', dtype=complex)

with open('old_data_n8.pkl', 'rb') as file:
    old_n8 = pickle.load(file)  # Call load method to deserialze

if os.path.isfile('old_data_n8.txt'):
    os.remove('old_data_n8.txt')

with open('old_data_n8.txt', 'a+') as f:
    for line in old_n8:
        f.write(str(line[2]) + " ")
    f.write("\n")


for a, b in zip(old_n8[0][2][0], new_n8[0]):
    print(a, b)


