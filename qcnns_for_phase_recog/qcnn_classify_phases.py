import os
import sys
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt 

sys.path.insert(0, "../qml_src/")
from qml import qcnn as q
from qml import layers as cl  # custom layers


def read_eigenvectors(file):
    with open(file, 'r+') as f:
        textData = f.readlines()

        h_vals = []
        for i in range(len(textData)):
            h1h2, eigenvector = textData[i].split("_")

            h_vals.append(tuple(map(float, h1h2[1: -1].split(', '))))
            textData[i] = eigenvector

        return h_vals, np.loadtxt(textData, dtype=complex)


def plot_heat_map(pred_mat, fname_save, title="2-D Heat Map", save_path="./results/"):
    fig = plt.figure()
    h1_vals = np.linspace(0, 1.6, 64)
    h2_vals = np.linspace(-1.6, 1.6, 64)

    # plot heat map
    heat_map = plt.pcolormesh(h1_vals, h2_vals, pred_mat, cmap='autumn')
    plt.colorbar(heat_map, label="Exp_val X")
    plt.title(title)
    plt.xlabel('h1/J')
    plt.ylabel('h2/J')

    # plot phase boundaries
    h1_vals = [0.1000, 0.2556, 0.4111, 0.5667, 0.7222, 0.8778, 1.0333, 1.1889, 1.3444, 1.5000]
    anti_ferro_mag_boundary = [-1.004, -1.0009, -1.024, -1.049, -1.079, -1.109, -1.154, -1.225, -1.285, -1.35]
    para_mag_boundary = [0.8439, 0.6636, 0.5033, 0.3631, 0.2229, 0.09766, -0.02755, -0.1377, -0.2479, -0.3531]

    plt.plot(h1_vals, anti_ferro_mag_boundary, "g--*", label="Antiferromagnetics")
    plt.plot(h1_vals, para_mag_boundary, "b--*", label="Paramagnetic")
    plt.legend()
    plt.savefig(save_path + fname_save)
    plt.close(fig)
    return


def run_qcnn(num_qubits, unique_name, training_fname, test_fname):
    h1h2_train, train_data = read_eigenvectors(training_fname)
    h1h2_test, test_data = read_eigenvectors(test_fname)

    labels = np.zeros(40)
    for index, h1h2 in enumerate(h1h2_train):

        h1, h2 = h1h2
        if h1 <= 1:
            labels[index] = 1.0

    # Model ------------------------------------------------------------------------
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

    ## Initialize parameters:
    my_qcnn.initialize_params(random=True)
    initial_params = copy.deepcopy(my_qcnn.params)

    ## Learning as described in paper:
    learning_rate = 100000  # intial value was 10 but this quantity doesn't learn fast enough !
    successive_loss = 1.0  # initialize to arbitrary value > 10^-5
    loss_lst = []  # initialize
    iteration_num = 1

    while (abs(successive_loss) > 1e-5) and (iteration_num < 250):
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

        # grad_mat = my_qcnn.compute_grad(train_data, labels)
        grad_mat = my_qcnn.compute_grad_w_mp(train_data, labels)  # with multi processing
        my_qcnn.update_params(grad_mat, learning_rate)

        loss_lst.append(loss)
        iteration_num += 1
    # model end --------------------------------------------------------------------------------------------------------

    os.mkdir("results/" + unique_name)

    optimal_params = copy.deepcopy(my_qcnn.params)
    my_qcnn.export_params(my_qcnn.structure, my_qcnn.params, fname=f'./results/{unique_name}model.pkl')

    ## Using model on test data (graph visualization) :
    predictions = my_qcnn.forward(test_data, my_qcnn.params.copy())
    pred_mat = predictions.reshape((64, 64), order='F')
    plot_heat_map(pred_mat, f'{unique_name}results_optimal_params.png')


    initial_predictions = my_qcnn.forward(test_data, initial_params)
    pred_mat = initial_predictions.reshape((64, 64), order='F')
    plot_heat_map(pred_mat, f'{unique_name}results_initial_params.png')

    ## Loss plot:
    x_axis = range(len(loss_lst))
    plt.plot(x_axis, loss_lst)
    plt.title('Training Loss over Epoches')
    plt.xlabel('Epoches')
    plt.ylabel("Loss")
    plt.savefig(f"./results/{unique_name}loss.png")
    return


def main():
    for num_qubits in [9]:
        training_fname = f"../data_gen/dataset_n={num_qubits}_train.txt"
        test_fname = f"../data_gen/dataset_n={num_qubits}_test.txt"
        unique_name = f"n{num_qubits}_250itterations_with_proper_train/"

        run_qcnn(num_qubits, unique_name, training_fname, test_fname)
        print(f"* * * * * * * * * * * * * * *Finished {num_qubits}, qbits! * * * * * * * * * * * * * * *")

if __name__ == "__main__":
    main()
