import os
import sys
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt 

# sys.path.insert(0, "../qml_src/")
# from qml import qcnn as q
# from qml import layers as cl  # custom layers
import generate_tetris_blocks as gtb

def read_eigenvectors(dim):
    
    data = gtb.TetrisData(dim)
    data.generate_combinations_of_tetris_blocks()
    data.package_data()

    # Wavefunctions:---------------------------------------------------------

    train_len = data.x_train.shape[0]       # Number of matrices in the array.
    train_dim = data.x_train.shape[1]**2    # Assuming square matrix, this is the nxn dimension
    test_len = data.x_test.shape[0]         # Same idea for test...
    test_dim = data.x_test.shape[1]**2      # Same idea for test...

    wavefunc_train = np.zeros((train_len, train_dim))
    wavefunc_test = np.zeros((test_len, test_dim))

    for ii in range(train_len):
    	wavefunc_train[ii, :] = data.x_train[ii].flatten() / np.linalg.norm(data.x_train[ii].flatten())

    for jj in range(test_len):
    	wavefunc_test[jj, :] = data.x_test[jj].flatten() / np.linalg.norm(data.x_test[jj].flatten())


    # Labels:----------------------------------------------------------------
    label_train = []
    label_test = []

    for row in range(len(data.y_train)):
    	if (data.y_train[row][0] == 1):
    		label_train.append(0)

    	else:
    		label_train.append(1)

    for row in range(len(data.y_test)):
    	if (data.y_test[row][0] == 1):
    		label_test.append(0)

    	else:
    		label_test.append(1) #-------------------------------------------

    return wavefunc_train, wavefunc_test, np.array(label_train), np.array(label_test)