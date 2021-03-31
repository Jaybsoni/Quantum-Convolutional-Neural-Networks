import time, os
import pickle
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as lng
from functools import lru_cache, wraps


def np_cache(*args, **kwargs):
    """LRU cache implementation for functions whose FIRST parameter is a numpy array
    >>> array = np.array([[1, 2, 3], [4, 5, 6]])
    >>> @np_cache(maxsize=256)
    ... def multiply(array, factor):
    ...     print("Calculating...")
    ...     return factor*array
    >>> multiply(array, 2)
    Calculating...
    array([[ 2,  4,  6],
           [ 8, 10, 12]])
    >>> multiply(array, 2)
    array([[ 2,  4,  6],
           [ 8, 10, 12]])
    >>> multiply.cache_info()
    CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)

    """

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



I = np.array([[1, 0], [0, 1]], dtype=int)
X = np.array([[0, 1], [1, 0]], dtype=int)
Z = np.array([[1, 0], [0, -1]], dtype=int)



II = sparse.dia_matrix((np.ones(2), np.array([0])), dtype=int, shape=(2, 2))
XX = sparse.dia_matrix((np.array([np.ones(1)]), np.array([-1])), dtype=int, shape=(2, 2))
XX.setdiag(np.ones(1), 1)
ZZ = sparse.dia_matrix((np.array([1, -1]), np.array([0])), dtype=int, shape=(2, 2))





@np_cache(maxsize=2048)
def find_kron(array, index, n):
    assert index <= n  # n elements should always be larger than index for array

    # Creates a list of 1's setting the index value as 0 to represent the array parameter given
    order = np.ones(n)
    order[index-1] = 0

    t = np.zeros(shape=(pow(2, n), pow(2, n)), dtype=int)  # Initializes t even though it will be overwritten (PEP-8)
    for i in range(1, len(order)):

        # Sets next element to Identity if next element is a 1, if zero, then array
        current = array if order[i] == 0 else I
        # print(i, len(order))
        if i == 1:
            # First time - compute kron(j-1, j)
            last = array if order[i-1] == 0 else I
            t = np.kron(last, current)

        else:  # Computes kron of last element current matrix with next element
            t = np.kron(t, current)

    return t.copy()


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


class HamiltonianSlow:
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
        print(self.first_term.shape)
        # Delete the output file if exists so we can append to a fresh one.
        self.filename = f'dataset_n={n}.txt'
        if os.path.isfile(self.filename):
            os.remove(self.filename)

    def get_first_term(self):
        self.first_term = np.zeros(shape=(self.size, self.size), dtype=float)
        for i in range(self.n - 2):
            elem = i + 1  # math element is indexes at 1
            a = np.array(find_kron(Z, elem, self.n))
            b = np.array(find_kron(X, elem + 1, self.n))
            c = np.array(find_kron(Z, elem + 2, self.n))

            self.first_term -= np.matmul(np.matmul(a, b), c)


    def get_second_term(self):
        self.second_term = np.zeros(shape=(self.size, self.size), dtype=float)
        for i in range(self.n):
            self.second_term -= find_kron(X, i+1, self.n)

    def get_third_term(self):
        for i in range(self.n - 1):  # This is actually 1 to N-2, python indexing has self.n-1
            elem = i + 1  # math element is indexes at 1
            self.third_term -= np.matmul(np.array(find_kron(X, elem, self.n)),
                                         np.array(find_kron(X, elem + 1, self.n)))


    def convert_sec(self, t):
        min = np.floor(t/60)
        sec = round(t % 60, 2)
        return "{}m-{:0.2f}s".format(int(min), sec)

    def calculate_time_remaining(self, t0, i):
        n = self.h1_range * self.h2_range

        if i % 1  == 0:
            time_remaning = ((1 - (i/n)) * (time.time() - t0))  * (n/i)
            percentage = (i / n) * 100
            print("{:0.2f}% \tElapsed: {} \tRemaining: {}".format(percentage, self.convert_sec(time.time() - t0), self.convert_sec(time_remaning)))



    def calculate_hamiltonian(self):

        s = time.time()
        self.get_first_term()
        self.get_second_term()
        self.get_third_term()
        print(time.time() - s)


        s = time.time()
        i = 0
        for h1 in np.linspace(self.h1_min, self.h1_max, self.h1_range):
            for h2 in np.linspace(self.h2_min, self.h2_max, self.h2_range):
                i += 1
                self.calculate_time_remaining(s, i)


                H = self.first_term + (self.second_term * h1) + (self.third_term * h2)
                w, v = np.linalg.eig(H)
                index = np.where(w == np.amin(w))

                # Write to file each time to avoid saving to ram
                with open(self.filename, 'a+') as f:
                    eigenvectors = v[index][0]

                    f.write(f"({h1},{h2})-[")
                    for line in eigenvectors:
                        f.write(str(line) + ", ")
                    f.write(f"]\n")

        print(f"added all terms in {time.time() - s} seconds")
        return




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
        print(self.first_term.shape)
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
            self.second_term -= find_kron(X, i+1, self.n)


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

        if i % 1  == 0:
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
        i = 0
        for h1 in np.linspace(self.h1_min, self.h1_max, self.h1_range):
            for h2 in np.linspace(self.h2_min, self.h2_max, self.h2_range):
                i += 1
                self.calculate_time_remaining(s, i)


                H = self.first_term + (self.second_term * h1) + (self.third_term * h2)

                # OLD
                ww, vv = np.linalg.eig(H)
                index = np.where(ww == np.amin(ww))

                # New
                w, v = sparse.linalg.eigs(H, k=100, which="SM")
                index = np.where(w == np.amin(w))

                print(len(vv[index][0]), len(v[index][0]))
                print(vv[index][0])
                print("_")
                print(v[index][0])

                # Write to file each time to avoid saving to ram
                with open(self.filename, 'a+') as f:
                    eigenvectors = v[index][0]

                    f.write(f"({h1},{h2})-[")
                    for line in eigenvectors:
                        f.write(str(line) + ", ")
                    f.write(f"]\n")

        print(f"added all terms in {time.time() - s} seconds")
        return

#######################

# s = time.time()
# find_kron_faster(X, 6, 15)
# print(time.time() - s)


s = time.time()
H = HamiltonianFast(10, 0, 1.6, -1.6, 1.6).calculate_hamiltonian()
print(find_kron.cache_info())
print(f"Time for caching took {time.time() -s} seconds")