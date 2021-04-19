import copy
import random
import numpy as np

class TetrisData():
    def __init__(self, dimension):
        # Create general shapes of tetris blocks
        self.line = np.array([[1], [1], [1]])
        self.cw = np.array([[1, 1],
                            [1, 0]])

        # Create canvas array
        self.dim = dimension
        self.grid = np.zeros((dimension, dimension))

        # Create empty arrays where tetris block rotations will be put into
        self.rotations_list = []
        self.lines_list = []
        self.train_dataset, self.validate_dataset,self.test_dataset = [], [], []

    # Comment
    def add_array_to_grid(self, shape, shape_list, i, j):
        """
        Given a shape 'shape', and index points i, j; add_array_to_grid tried to overlay
        shape to a copy of self.grid to see if it will fit. If it does, it will add it to
        grid and then append the result to shape_list, a list of unique locations
        that shape has fit in the grid size.
        :param shape: tuple - shape of matrix
        :param shape_list: list - list to append values to
        :param i: int - i element of grid shape is being added to
        :param j: int - j element of grid shape is being added to
        :return:
        """
        grid = copy.deepcopy(self.grid)  # Necessary to do a deep copy!

        try:  # Find the shape of the object to know how much to shift when inserting into grid
            i_shift, j_shift = shape.shape
        except ValueError:  # (3, ) corresponds to horizontal line, should be i_shift, j_shift = 1, 3
            i_shift, j_shift = 1, 3

        # Try to add shape to grid
        try:  # Some combinations may fail due to indexing overflow
            grid[i:i + i_shift, j:j + j_shift] = shape  # Insert into grid
        except ValueError:  # Indexing overflow
            pass  # Ignore example (not valid location on grid)
        else:
            is_array_in_list = next((True for elem in shape_list if np.array_equal(elem, grid)), False)
            if not is_array_in_list: # If unique entry
                shape_list.append(grid)  # Append to list of combinations

    def generate_combinations_of_tetris_blocks(self):
        """
        Itterates through all the I and L combinations and their rotations and calls
        add_array_to_grid to them.
        """
        self.rotations_list, self.lines_list = [], []

        # Find all combinations of cw and line shape in the canvas (grid)
        for i in range(self.dim):  # Iterate through column index
            for j in range(self.dim):  # Iterate through row index

                for rot in range(4):  # Iterate through all rotations of 2x2 cw array
                    rotation_of_l = np.rot90(self.cw, rot)
                    self.add_array_to_grid(rotation_of_l, self.rotations_list, i, j)

                self.add_array_to_grid(self.line, self.lines_list, i, j)
                self.add_array_to_grid(self.line.T[0], self.lines_list, i, j)

    @staticmethod
    def split_data(list_in, train, validate, test):
        """
        Given a list of values, randomize the list in groups that roughly match
        the ratios of train, validate, and test
        :param list_in:
        :param train: float - value between 0 and 1
        :param validate: float - value between 0 and 1
        :param test: float - value between 0 and 1
        :return: 3 x list of vals - properly sizes lists
        """
        assert train + validate + test == 1  # Ratio should equal 1
        random.shuffle(list_in) # Randomized input list to help training set

        # Calculate the bounds for the data split step
        bound1 = int(np.ceil(len(list_in) * train))
        bound2 = int(np.ceil(len(list_in) * (train + validate)))

        # Split the list into respectively sized lists
        d_train, d_val, d_test = list_in[: bound1], list_in[bound1:bound2], list_in[bound2:]
        return d_train, d_val, d_test

    def package_data(self):
        """
        Takes self.lines_list and self.rotations_list and does the necessary transformations
        to make them the correct dimensions and types for training.
        """
        # Separate lines_list and rotations_list into training, validation, and testing sets
        lines_train, lines_validate, lines_test = self.split_data(self.lines_list, 0.5, 0.3, 0.2)
        rots_train, rots_validate, rots_test = self.split_data(self.rotations_list, 0.5, 0.3, 0.2)

        # Extract the lines and rotations lists together
        self.x_train = np.array([*lines_train, *rots_train])
        self.y_train = np.array([*[[1, 0]] * len(lines_train), *[[0, 1]] * len(rots_train)])

        self.x_validate = np.array([*lines_validate, *rots_validate])
        self.y_validate = np.array([*[[1, 0]] * len(lines_validate), *[[0, 1]] * len(rots_validate)])

        self.x_test = np.array([*lines_test, *rots_test])
        self.y_test = np.array([*[[1, 0]] * len(lines_test), *[[0, 1]] * len(rots_test)])


if __name__ == '__main__':
    data = TetrisData(4)
    data.generate_combinations_of_tetris_blocks()
    data.package_data()
