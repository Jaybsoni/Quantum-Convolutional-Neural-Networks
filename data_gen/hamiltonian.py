import time
import pickle
import numpy as np

I = np.array([[1, 0], [0, 1]], dtype=int)
X = np.array([[0, 1], [1, 0]], dtype=int)
Z = np.array([[1, 0], [0, -1]], dtype=int)

def kron_helper(t, max, i=0):
    if i >= max:
        return t
    else:
        return kron_helper(np.kron(t, I), max, i + 1)

def find_fast_kron(array, index, n):
    assert index <= n  # n elements should always be larger than index for array

    if index == 1:  # If array is in first element
        return kron_helper(array, n-index)
    else: # If array is elsewhere in the operation
        before = index - 1  # -1 since index is the location of the array element
        res = kron_helper(I, before-1)  # -1 since we start with I already
        return kron_helper(np.kron(res, array), n-index)  # Repeat for elements after with nested array


def find_kron(array, index, n):
    assert index <= n  # n elements should always be larger than index for array

    # Creates a list of 1's setting the index value as 0 to represent the array parameter given
    order = np.ones(n)
    order[index-1] = 0

    t = np.empty(shape=(pow(2, n), pow(2, n)), dtype=int)  # Initializes t even though it will be overwritten (PEP-8)
    for i in range(len(order)):
        if i == 0:
            continue  # Skip first loop through

        # Sets next element to Identity if next element is a 1, if zero, then array
        current = array if order[i] == 0 else I

        if i == 1:
            # First time - compute kron(j-1, j)
            last = array if order[i-1] == 0 else I
            t = np.kron(last, current)

        else:  # Computes kron of last element current matrix with next element
            t = np.kron(t, current)

    return t


class Hamiltonian:
    def __init__(self, n=2, h1=1, h2=2, j=1):
        self.n = n
        self.h1 = h1
        self.h2 = h2
        self.j = j

        self.size = pow(2, self.n)
        self.term = np.zeros(shape=(self.size, self.size), dtype=float)

    def first_term(self):
        for i in range(self.n - 2):
            elem = i + 1  # math element is indexes at 1

            self.term += -self.j * np.dot(np.dot(find_kron(Z, elem, self.n), find_kron(X, elem + 1, self.n)),
                                          find_kron(Z, elem + 2, self.n))

    def second_term(self):
        for i in range(self.n):
            self.term += -self.h1 * find_kron(X, i+1, self.n)


    def third_term(self):
        for i in range(self.n - 1):  # This is actually 1 to N-2, python indexing has self.n-1
            elem = i + 1  # math element is indexes at 1
            self.term += -self.h2 * np.dot(find_kron(X, elem, self.n), find_kron(X, elem + 1, self.n))

    def calculate_hamiltonian(self):
        self.first_term()
        self.second_term()
        self.third_term()

        return self.term

from resource import getrusage, RUSAGE_SELF
import gc

gc.collect()
startMem = getrusage(RUSAGE_SELF).ru_maxrss
s = time.time()
find_fast_kron(Z, 6, 15)
print(f"Time for old took {time.time() -s} seconds")
print((getrusage(RUSAGE_SELF).ru_maxrss - startMem) / (1024 * 1024))



# i = 1
# total = 64*32
# data = []
# for h2 in np.linspace(-1.6, 1.6, 64):
#     for h1 in np.linspace(0, 1.6, 32):
#         H = Hamiltonian(6, h1, h2).calculate_hamiltonian()
#
#         w, v = np.linalg.eig(H)
#
#         index = np.where(w == np.amin(w))
#         data.append([h1, h2, v[index]])
#         # print("Corresponding max eigenvector is: ", v[index])
#         if i%10 == 0:
#             print(f"Finished {i/total*100}%")
#
#         i += 1
#
# # Open a file and use dump()
# with open('data_n.pkl', 'wb') as file:
#     pickle.dump(data, file)
#