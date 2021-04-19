import os
import sys
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt 

sys.path.insert(0, "../qml_src/")
from qml import qcnn as q
from qml import layers as cl  # custom layers
import generate_tetris_blocks as gtb

def wavefuncs_labels(dim):
    
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


def run_qcnn(train_data, labels, my_qcnn, unique_name):

    # my_qcnn = q.Qcnn(num_qubits)

    # # Add Custom layer:
    # legacy_fully_connected_layer = cl.get_legacy_fc_layer(num_qubits // 3)
    # my_qcnn.Layers[legacy_fully_connected_layer.name] = legacy_fully_connected_layer

    # custom_fully_connected_layer = cl.get_custom_conv_layer(3)
    # my_qcnn.Layers[custom_fully_connected_layer.name] = custom_fully_connected_layer

    # # def structure:
    # my_qcnn.add_layer(my_qcnn.Layers["legacy_conv4_layer"], kwargs={"label": "C4_1"})
    # my_qcnn.add_layer(my_qcnn.Layers["custom_conv_layer_n3"], kwargs={})
    # my_qcnn.add_layer(my_qcnn.Layers["custom_conv_layer_n3"], kwargs={"start_index": 1})
    # my_qcnn.add_layer(my_qcnn.Layers["custom_conv_layer_n3"], kwargs={"start_index": 2})
    # my_qcnn.add_layer(my_qcnn.Layers["legacy_pool_layer"], kwargs={"label": "P1",
    #                                                                "update_active_qubits": {"group_len": 3,
    #                                                                                         "target": 1}})

    # my_qcnn.add_layer(my_qcnn.Layers[legacy_fully_connected_layer.name], kwargs={})

    # Initialize parameters:
    my_qcnn.initialize_params(random=True)
    initial_params = copy.deepcopy(my_qcnn.params)

    # Learning as described in paper:
    learning_rate = 100000  # intial value was 10 but this quantity doesn't learn fast enough !
    successive_loss = 1.0  # initialize to arbitrary value > 10^-5
    loss_lst = []  # initialize
    iteration_num = 1

    while (abs(successive_loss) > 1e-5) and (iteration_num < 3):
        pred = my_qcnn.forward(train_data, my_qcnn.params.copy())
        loss = my_qcnn.mse_loss(pred, labels)

        print("---- Iteration : {}, Loss {} ----------------------".format(iteration_num, loss))

        if iteration_num == 1:
            pass
        else:
            successive_loss = loss - loss_lst[-1]
            if successive_loss < 0:
                learning_rate *= 1.05  # if loss decreases, increase learning rate by 5%
            else:
                learning_rate /= 2  # if it gets bigger, decrease learning rate by 50%

        grad_mat = my_qcnn.compute_grad(train_data, labels)
        # grad_mat = my_qcnn.compute_grad_w_mp(train_data, labels)  # with multi processing
        my_qcnn.update_params(grad_mat, learning_rate)

        loss_lst.append(loss)
        iteration_num += 1
    # model end --------------------------------------------------------------------------------------------------------

    if (os.path.isdir("results/" + unique_name)):
        # os.remove(f"./results/{unique_name}model.pkl")
        pass

    else:
        os.mkdir("results/" + unique_name)

    # os.mkdir("results/" + unique_name)

    optimal_params = copy.deepcopy(my_qcnn.params)
    my_qcnn.export_params(my_qcnn.structure, optimal_params, fname=f'./results/{unique_name}model_optimal.pkl')
    my_qcnn.export_params(my_qcnn.structure, initial_params, fname=f'./results/{unique_name}model_initial.pkl')

    # # Using model on test data (graph visualization) :
    # predictions = my_qcnn.forward(test_data, my_qcnn.params.copy())
    # pred_mat = predictions.reshape((64, 64), order='F')
    # plot_heat_map(pred_mat, f'{unique_name}results_optimal_params.png')

    # initial_predictions = my_qcnn.forward(test_data, initial_params)
    # pred_mat = initial_predictions.reshape((64, 64), order='F')
    # plot_heat_map(pred_mat, f'{unique_name}results_initial_params.png')

    # Loss plot:
    x_axis = range(len(loss_lst))
    plt.plot(x_axis, loss_lst)
    plt.title('Training Loss over Epoches')
    plt.xlabel('Epoches')
    plt.ylabel("Loss")
    plt.savefig(f"./results/{unique_name}loss.png")

    return my_qcnn


def test_qcnn(test_input, test_labels, my_qcnn):

    pred = my_qcnn.forward(test_input, my_qcnn.params.copy())

    mse_loss = my_qcnn.mse_loss(pred, test_labels)
    acc = 0

    for prediction, label in zip(pred, test_labels):

        if (np.ceil(prediction) == float(label)):
            acc += 1

    acc = acc / len(pred)

    return mse_loss, acc, pred

def new_model(num_qubits):

    my_qcnn = q.Qcnn(num_qubits)

    # Add Custom layer:
    legacy_fully_connected_layer = cl.get_legacy_fc_layer(num_qubits // 3)
    my_qcnn.Layers[legacy_fully_connected_layer.name] = legacy_fully_connected_layer

    custom_fully_connected_layer = cl.get_custom_conv_layer(3)
    my_qcnn.Layers[custom_fully_connected_layer.name] = custom_fully_connected_layer

    # def structure:
    my_qcnn.add_layer(my_qcnn.Layers["legacy_conv4_layer"], kwargs={"label": "C4_1"})
    my_qcnn.add_layer(my_qcnn.Layers["custom_conv_layer_n3"], kwargs={})
    my_qcnn.add_layer(my_qcnn.Layers["custom_conv_layer_n3"], kwargs={"start_index": 1})
    my_qcnn.add_layer(my_qcnn.Layers["custom_conv_layer_n3"], kwargs={"start_index": 2})
    my_qcnn.add_layer(my_qcnn.Layers["legacy_pool_layer"], kwargs={"label": "P1",
                                                                   "update_active_qubits": {"group_len": 3,
                                                                                            "target": 1}})

    my_qcnn.add_layer(my_qcnn.Layers[legacy_fully_connected_layer.name], kwargs={})

    return my_qcnn

def OG_model(num_qubits):

    my_qcnn = q.Qcnn(num_qubits)
    
    # Add Custom layer:
    legacy_fully_connected_layer = cl.get_legacy_fc_layer(num_qubits // 3)
    my_qcnn.Layers[legacy_fully_connected_layer.name] = legacy_fully_connected_layer
    
    # def structure:
    my_qcnn.add_layer(my_qcnn.Layers["legacy_conv4_layer"], kwargs={"label": "C4_1"})
    my_qcnn.add_layer(my_qcnn.Layers["legacy_conv_layer"], kwargs={"label": "C2"})
    my_qcnn.add_layer(my_qcnn.Layers["legacy_conv_layer"], kwargs={"label": "C3", "start_index": 1})
    my_qcnn.add_layer(my_qcnn.Layers["legacy_conv_layer"], kwargs={"label": "C4", "start_index": 2})
    my_qcnn.add_layer(my_qcnn.Layers["legacy_pool_layer"], kwargs={"label": "P1",
                                                                   "update_active_qubits": {"group_len": 3,
                                                                                            "target": 1}})
    
    my_qcnn.add_layer(my_qcnn.Layers[legacy_fully_connected_layer.name], kwargs={})

    return my_qcnn


def main():

    num_qubits = 4
    # num_qubits = 12
    # training_fname = f"../data_gen/tetris_dataset_n={num_qubits}_train.txt"
    # test_fname = f"../data_gen/tetris_dataset_n={num_qubits}_test.txt"
    unique_name = f"n{num_qubits}_2itterations_tetris_qcnn_train/"

    wavefunc_train, wavefunc_test, label_train, label_test = wavefuncs_labels(4)

    # print(type(wavefunc_train))
    # print(type(label_train))
    # print(wavefunc_train)
    # print(label_train)

    og_qcnn = OG_model(num_qubits)
    # new_qcnn = new_model(num_qubits)

    # run_qcnn(num_qubits, unique_name, training_fname, test_fname)
    trained_qcnn = run_qcnn(wavefunc_train, label_train, og_qcnn, "og" + unique_name)
    mse_loss, acc, pred = test_qcnn(wavefunc_test, label_test, trained_qcnn)

    print(mse_loss)
    print(acc)
    print(pred)
    # run_qcnn(wavefunc_train, label_train, new_qcnn, unique_name + "new")
    print(f"* * * * * * * * * * * * * * *Finished {num_qubits}, qbits! * * * * * * * * * * * * * * *")


if __name__ == "__main__":
    main()
# # main

# wavefunc_train, wavefunc_test, label_train, label_test = wavefuncs_labels(4)

# print("Training Wavefunction:\n", wavefunc_train)
# print("Testing Wavefunction:\n", wavefunc_test)
# print("Training Label:\n", label_train)
# print("Testing Label:\n", label_test)




