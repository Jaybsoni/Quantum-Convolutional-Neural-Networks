import os
import time
import tqdm
import itertools
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg


def read_eigenvectors(file):
    """
    Takes a dataset and returns the h1h2 values that
    are associated for the eigenvector for each line
    :param file: str - file location
    :return: tuple of list & np.array
    """
    with open(file, 'r+') as f:
        text_data = f.readlines()

        h_vals = []
        for i in range(len(text_data)):
            h1h2, eigenvector = text_data[i].split("_")

            h_vals.append(tuple(map(float, h1h2[1: -1].split(', '))))
            text_data[i] = eigenvector

        return h_vals, np.loadtxt(text_data, dtype=complex)


def find_kron(array, index, q_bits):
    """

    :param array: sparse.dia_matrix - Tensor (X or Z)
    :param index: int - location of X or Z in identities
    :param q_bits: int - number of qbits
    :return:
    """
    order = np.ones(q_bits)
    order[index-1] = 0  # Sets index as 0 to represent the array parameter given
    assert index <= q_bits  # n elements should always be larger than index for array
    t = sparse.dia_matrix((pow(2, q_bits), pow(2, q_bits)), dtype=int)

    for i in range(1, len(order)):
        # Sets next element to Identity if next element is a 1, else array (Z or X)
        current = array if order[i] == 0 else II

        if i == 1:  # First time - compute kron(j-1, j)
            t = array if order[i-1] == 0 else II

        t = sparse.kron(t, current)

    return t.copy()


class Hamiltonian:
    def __init__(self, qbits=4, h1_metadata=(0, 1.6), h2_metadata=(-1.6, 1.6), v=1):
        self.qbits = qbits
        self.verbose = v
        self.h1_min, self.h1_max = h1_metadata
        self.h2_min, self.h2_max = h2_metadata

        self.size = pow(2, self.qbits)
        self.first_term = np.zeros(shape=(self.size, self.size), dtype=float)
        self.second_term = np.zeros(shape=(self.size, self.size), dtype=float)
        self.third_term = np.zeros(shape=(self.size, self.size), dtype=float)

    def get_first_term(self):
        self.first_term = np.zeros(shape=(self.size, self.size), dtype=float)

        for i in range(self.qbits - 2):
            elem = i + 1  # math element is indexes at 1
            if self.verbose: print(f"first term {elem}/{self.qbits - 2}")

            a = find_kron(Z, elem, self.qbits)
            b = find_kron(X, elem + 1, self.qbits)
            c = find_kron(Z, elem + 2, self.qbits)

            # Instead of A.dot(B).dot(C), note that A and C are diagonal with 1's and -1's.
            # Convert A, C into vectors and just multiply them to B, a 45 000% speedup over dot product for n=14
            a_diag = a.diagonal()[..., None]
            c_diag = c.diagonal()[..., None]
            combined = b.multiply(a_diag).multiply(c_diag)

            self.first_term -= combined.toarray()

    def get_second_term(self):
        self.second_term = np.zeros(shape=(self.size, self.size), dtype=float)
        for i in range(self.qbits):
            elem = i + 1  # math element is indexes at 1
            if self.verbose: print(f"second term {elem}/{self.qbits}")
            self.second_term -= find_kron(X, elem, self.qbits).toarray()

    def get_third_term(self):
        self.third_term = np.zeros(shape=(self.size, self.size), dtype=float)
        """
        Honestly? This method is magic. It. Just. Works. Don't believe me? Try to make it faster...
        :return:
        """
        for i in range(self.qbits - 1):  # This is actually 1 to N-2, python indexing has self.n-1
            elem = i + 1  # math element is indexes at 1
            if self.verbose: print(f"third term {elem}/{self.qbits-1}")

            """
            We want to take the dot product of B1 and B2, and subtract that value to 
            self.third term for each value of elem. Problem? The dot product is 
            a very inefficient operation for arrays that are sparse, both np.dot as 
            well as sparse.dot(). So we had to find a way to do this faster!
            """
            b1 = find_kron(X, elem, self.qbits)
            b2 = find_kron(X, elem + 1, self.qbits)
            """
            This is where the really clever symmetry starts. It only works for this use 
            case, but by god we need it. If you truly want to understand this trick, I 
            recommend creating 4x4 arrays on paper to understand why this property works. 
            Consider you have two square 4x4 matricies, that have the special property 
            that each row or column has only a sigle 1, the rest are 0's. Almost like a 
            sudoku with 1's in a matrix of zeros, and all 1's should bu unique to that row 
            or column. If this property holds true for both matricies, then if you take 
            the dot product of these two matricies, the locations of the 1's on the 
            resultant 4x4 matrix is at the locations of the rows of the first matrix 
            with the column of the second matrix that has a 1 populated at the rowID 
            that is the column of the 1 on the first matrix. Yeah. Hard to explain. 
            """
            # Maybe
            b1_rows, _ = sparse.coo_matrix(b1, dtype=sparse.coo_matrix).nonzero()
            # _, b2_cols = sparse.coo_matrix(b2, dtype=sparse.coo_matrix).nonzero()

            # Original
            b1_rows, b1_cols = sparse.coo_matrix(b1, dtype=sparse.coo_matrix).nonzero()
            b2_rows, wrongly_ordered_b2_cols = b1_cols, []
            def extract_elem(elem):
                wrongly_ordered_b2_cols.append(elem)

            list(map(b2.getrow, filter(extract_elem, b2_rows)))

            """
            There are clearer ways to calculate the B2_Cols, (below) but they are
            slower because they require calling .getrow() self.size times. This scales
            with the size of the hamiltonian, and is just as slow as using sparse.dot(),
            which is sadly too slow. Therefore, I'm using a property of the symmetry of
            B2 (the X tensor) under transformation to find what the values of B2_Cols are.
            Honestly, I know this works through over 6 hours of studying how B2 evolves,
            and working like crazy to find a more efficient method to compute the dot
            product instead of using sparse.dot() (which is pretty darn efficient already)
            """
            # This commented out section is what I'm trying to mimic with the above map and the below for loop
            # b2_cols_slow = []  # Temp!
            # for val in b2_rows:
            #     # b2.getrow(val) is what makes this too slow
            #     b2_col = str(b2.getrow(val)).split(")")[0].split(" ")[-1]
            #     b2_cols_slow.append(int(b2_col))
            """
            If you were to print wrongly_ordered_B2_cols, and compare it to the B2_cols_slow
            in the commented out section above, you'd they'd be contain the same row numbers
            (makes sense since theres the same number of rows and each only has a single 1
            in them), but the orders are different. This is because wrongly_ordered_B2_cols
            doesn't account for the swapping nature that kron applies to X (print B2 and
            notice how the 1's alternate on two diagonals (those diagonals change too for
            a different n). If you print multiple different B2's for different qbits (n),
            you'll see the number of 1's that repeat on a diagonal before there's 0's changes
            for the number of qbits. This is exactly the property that causes
            wrongly_ordered_B2_cols to contain the rows, but in an incorrect order from
            B2_cols_slow. wrongly_ordered_B2_cols needs to have whole chunks of numbers
            swapped, but the size of the chunks and quantity of swaps depends on the
            iteration we are on (the value of elem, which summation we're on). So, the
            swaps variable below calculates how many blocks wrongly_ordered_B2_cols should
            be divided into, and the variable groups splits up wrongly_ordered_B2_cols into
            that many chunks, evenly and without altering them.

            An example of what wrongly_ordered_B2_cols would be compared to B2_cols_slow is:
                wrongly_ordered_B2_cols = [a, b, c, d, e, f, g, h, i, j, k, l]
                B2_cols_slow =  [d, e, f, a, b, c, j, k, l, g, h, i]
            Notice how changes are in chunks of three, so a, b, c was swapped with d, e, f;
            but depending on the iteration of the summation (elem) we're on, it may be
                wrongly_ordered_B2_cols = [a, b, c, d, e, f, g, h, i, j, k, l]
                B2_cols_slow =  [c, d, a, b, g, h, e, f, k, l, i, j]
            which is in chunks of two, or
                wrongly_ordered_B2_cols = [a, b, c, d, e, f, g, h, i, j, k, l]
                B2_cols_slow = [b, a, d, c, f, e, h, g, j, i, l, k]
            which is in chunks of 1. So we need to find out how big these chunks are that need
            to be swapped, and swap them.
            """
            b2_cols = []
            # How many groups the list should be separated into. (how big to make the chunks)
            swaps = int(pow(2, self.qbits - 1 - elem))
            groups = [wrongly_ordered_b2_cols[i:i + swaps] for i in range(0, len(wrongly_ordered_b2_cols), swaps)]
            """
            Ref to above two lines. First, we need to find how big the chunks are, and
            separate wrongly_ordered_B2_cols into those chunks. For example, if:
                wrongly_ordered_B2_cols = [a, b, c, d, e, f, g, h, i, j, k, l]
            and we want to break this into chunks of three, we would want to create:
                groups = [[a, b, c], [d, e, f], [g, h, i], [j, k, l]]
            or if we wanted chunks of two:
                groups = [[a, b], [c, d], [e, f], [g, h], [i, j], [k, l]]
            so that we can swap them to
                groups = [[d, e, f], [a, b, c], [j, k, l], [g, h, i]]
                and
                groups = [[c, d], [a, b], [g, h], [e, f], [k, l], [i, j]]
            respectively. If the chunks were smaller, and we had:
                [[a], [b], [c], [d], [e], [f]]
            then we would want to swap groups to get
                [[b], [a], [d], [c], [f], [e]]
            So the below for loops performs this swap for a list of lists.
            """
            for j in range(int(len(groups) / 2)):  # Preform the swaps
                switch = groups[2 * j:(2 * j) + 2]
                b2_cols.append([*switch[1], *switch[0]])

            # [[b, a], [d, c]] -> [b, a, d, c]  # np.flatten() for lists
            b2_cols = list(itertools.chain(*b2_cols))

            # combined = B1.dot(B2), but now with the data we need for where all those 1's are.
            combined = sparse.coo_matrix((np.ones(self.size, dtype=int), (b1_rows, b2_cols)),
                                         shape=(self.size, self.size))
            """
            Old method to compare to confirm dot product is done correctly. 
            This one line is what we replace with the above code. But the above 
            code is *much* faster for larger qbits, as it minimizes the 
            information we need to use to preform the dot product, and does so 
            smartly without calling sparse.getrow() self.size times, which sadly 
            was also inefficient. Also, we are not performing this calculation 
            explicitly, as you clearly see. So the above method is much longer, 
            but much faster too. A trade off we desperately needed for high qbits 
            (anything larger than n=12, really)
            """
            # The above code is over hundreds of times faster than using dot product (for n=10 I believe), don't
            # have the energy
            # slow_method = B1.dot(B2)
            # assert np.array_equal(slow_method.toarray(), coo.toarray())
            self.third_term -= combined.toarray()

    def generate_data(self, h1_range, h2_range, name):
        """
        Given a filename, and h1 + h2 ranges, calculate the three terms used to
        construct the hamiltonian in advance (unchanging between values of h1, h2)
        and start to iterate through h1 and h2, appending to a text file each time
        to avoid storing the huge dataset and saving once.

        :param h1_range: float - # of steps to get from self.h1_min to self.h1_max
        :param h2_range: float - # of steps to get from self.h2_min to self.h2_max
        :param filename: filename start appending outputs to in streaming mode
        :return:
        """
        t0 = time.time()
        self.get_first_term()
        self.get_second_term()
        self.get_third_term()
        print(f"{round(time.time() - t0, 4)}s elapsed to calculate term")

        # Delete the output file if exists so we can append to a fresh ones.
        filename = f'dataset_n={self.qbits}_' + name + ".txt"
        if os.path.isfile(filename): os.remove(filename)

        # Create a list of h1 and h2 values to loop over
        h1h2 = [[h1, h2] for h1 in np.linspace(self.h1_min, self.h1_max, h1_range)
                for h2 in np.linspace(self.h2_min, self.h2_max, h2_range)]
        for h1, h2 in tqdm.tqdm(h1h2):

            if name == "train": h2 = 0  # If in training mode, h2 should be 0!

            h = self.first_term + (self.second_term * h1) + (self.third_term * h2)
            eigenvalue, eigenvector = self.find_eigval_with_sparse(h)
            # self.test_dataset(h, eigenvalue)  # SLOW! Compares np.eig with sparse.eig

            # Write to file each time to avoid saving to ram
            with open(filename, 'a+') as f:
                f.write(f"{h1, h2}_")  # Append h1, h2 for reference
                for line in eigenvector:
                    f.write(str(line) + " ")
                f.write("\n")

    @staticmethod
    def find_eigval_with_sparse(h):
        """
        Uses an approximation to find minimum eigenvalue and corresponding
        eigenvector, works well for sparse Hamiltonians (Valid for this class)
        :param h: np.array - 2D hamiltonian
        :return: np.ndarray, np.ndarray - Minimum EigVal and EigVec
        """
        b, c = sparse.linalg.eigs(h, k=1, which='SR', tol=1e-16)
        return b, c.flatten()

    @staticmethod
    def find_eigval_with_np(h):
        """
        Uses the much slower way to find the minimized eigenvalue and corresponding
        eigenvector of a hamiltonian. MUCH slower and very inefficient for large H
        :param h: np.array - 2D hamiltonian
        :return: float64, np.array - EigVal and EigVec
        """
        ww, vv = np.linalg.eig(h)  # Old method with linalg
        index = np.where(ww == np.amin(ww))  # Find lowest eigenvalue
        np_eig_vec, np_eig_val = vv[:, index], ww[index]  # Use index to find lowest eigenvector

        """
        np.linalg.eig returns the eigenvalues and vectors of a matrix
        BUT, it returns a list of lists of lists, where the elements of
        each triple nested list is the first element of each eigenvector,
        not a list of eigenvectors like any sensical person would return.
        """  # np.linalg.eig is grade A stupid, change my mind...
        eig_vect_list = []
        for eigVal in range(len(np_eig_val)):
            temp_vec = []

            for eigVec in range(len(np_eig_vec)):
                temp_vec.append(np_eig_vec[eigVec][0][eigVal])
            eig_vect_list.append(np.array(temp_vec))

        sum_vec = np.sum(eig_vect_list, axis=0)

        return np_eig_val[0], sum_vec / np.linalg.norm(sum_vec)

    def test_dataset(self, h, possible_eigenvalue):
        """
        Computes the explicit eigenvector for a given Hamiltonian
        to check if the eigenvalue is a valid eigenvalue
        :param h: np.array - Hamiltonian
        :param possible_eigenvalue: np.array
        :return: N/A
        """
        _, np_eig_vec = self.find_eigval_with_np(h)
        magnitude = (h @ np_eig_vec) / possible_eigenvalue

        assert np.allclose(magnitude, np.array(np_eig_vec, dtype=complex), 1e-9)


II = sparse.dia_matrix((np.ones(2), np.array([0])), dtype=int, shape=(2, 2))
Z = sparse.dia_matrix((np.array([1, -1]), np.array([0])), dtype=int, shape=(2, 2))
X = sparse.dia_matrix((np.array([np.ones(1)]), np.array([-1])), dtype=int, shape=(2, 2))
X.setdiag(np.ones(1), 1)

if __name__ == '__main__':
    s = time.time()

    # n represents the number of qbits in the system. The larger the value,
    # the more complicated and slower the calculations. Note that computation
    # scales by 2^n, so anything larger than 9 or 10 starts to become
    # exponentially long
    n = 9

    # Create the hamiltonian and generate both train and test data sets
    H = Hamiltonian(n)

    # Train data only requires 40x1 resolution
    H.generate_data(40, 1, "train")

    # Testing data is a 64x64 grid, as defined in the paper
    H.generate_data(64, 64, "test")
    print(f"Time for creating dataset was {time.time() - s} seconds")