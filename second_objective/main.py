import copy
import random
import numpy as np

# Create canvas array
dim = 3
grid = np.zeros((dim, dim))

# Create general shapes of tetris blocks
line = np.array([[1], [1], [1]])
cw = np.array([[1, 1],
               [1, 0]])

# Create empty arrays where tetris block rotations will be put into
rotations_list = []
lines_list = []


# Comment
def add_array_to_grid(grid, shape, shape_list, i, j, i_shift, j_shift):
    try:  # Try to add shape to grid, some combinations may fail due to indexing overflow
        grid[i:i + i_shift, j:j + j_shift] =  shape # Insert into grid
    except ValueError:  # Indexing overflow, ignore example (not valid location on grid)
        pass
    else:
        is_array_in_list = next((True for elem in shape_list if np.array_equal(elem, grid)), False)
        if not is_array_in_list:
            shape_list.append(grid)  # Append to list of rotation combinations


# Find all combinations of cw and line shape in the canvas (grid)
for i in range(dim):  # Iterate through column index
    for j in range(dim):  # Iterate through row index

        for rot in range(4):  # Iterate through all rotations of 2x2 cw array
            shape = np.rot90(cw, rot)
            grid_copy = copy.deepcopy(grid)  # Make a copy of the grid
            add_array_to_grid(grid_copy, shape, rotations_list, i, j, 2, 2)

        grid_copy = copy.deepcopy(grid)  # Make a copy of the grid
        add_array_to_grid(grid_copy, line.T[0], lines_list, i, j, 1, 3)

        grid_copy = copy.deepcopy(grid)  # Make a copy of the grid
        add_array_to_grid(grid_copy, line, lines_list, i, j, 3, 1)


# # print(rotations_list)
# for elem in lines_list:
#     print(elem)
# #
# for elem in rotations_list:
#     print(elem)


def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

x = partition(lines_list, 2)
for i in x:
    print("_")
    for elem in i:
        print(elem)