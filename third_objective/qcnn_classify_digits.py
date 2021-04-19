import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt 

sys.path.insert(0, "../qml_src/")
from qml import qcnn as q
from qml import layers as cl  # custom layers


def get_wfs_from_images(num_qubits):
    x = num_qubits
    wf_train = []
    wf_test = []
    labels_train = []
    labels_test = []

    return wf_train, wf_test, labels_train, labels_test


def run_qcnn(train_data, labels, my_qcnn, unique_name):
    # Initialize parameters:
    my_qcnn.initialize_params(random=True)
    initial_params = copy.deepcopy(my_qcnn.params)

    batch_size = 100
    num_batches = len(train_data) // batch_size
    num_epoches = 250
    loss_lst = []  # initialize

    for batch_index in range(1, num_batches+1):
        batched_data = train_data[(batch_index - 1)*batch_size: batch_index*batch_size]
        batched_labels = labels[(batch_index - 1)*batch_size: batch_index*batch_size]
        learning_rate = 100000  # intial value was 10 but this quantity doesn't learn fast enough !

        for epoch in range(1, num_epoches + 1):
            pred = my_qcnn.forward(batched_data, my_qcnn.params.copy())
            loss = my_qcnn.mse_loss(pred, batched_labels)

            if epoch != 1:
                successive_loss = loss - loss_lst[-1]
                if successive_loss < 0:
                    learning_rate *= 1.05  # if loss decreases, increase learning rate by 5%
                else:
                    learning_rate /= 2  # if it gets bigger, decrease learning rate by 50%

            grad_mat = my_qcnn.compute_grad(train_data, labels)
            # grad_mat = my_qcnn.compute_grad_w_mp(train_data, labels)  # with multi processing
            my_qcnn.update_params(grad_mat, learning_rate)

            loss_lst.append(loss)
    # model end --------------------------------------------------------------------------------------------------------

    if not os.path.isdir("results/" + unique_name):
        os.mkdir("results/" + unique_name)

    optimal_params = copy.deepcopy(my_qcnn.params)
    my_qcnn.export_params(my_qcnn.structure, optimal_params, fname=f'./results/{unique_name}model_optimal.pkl')
    my_qcnn.export_params(my_qcnn.structure, initial_params, fname=f'./results/{unique_name}model_initial.pkl')

    # Loss plot:
    x_axis = range(len(loss_lst))
    plt.plot(x_axis, loss_lst)
    plt.title('Training Loss over Epoches')
    plt.xlabel('Epoches')
    plt.ylabel("Loss")
    plt.savefig(f"./results/{unique_name}loss.png")

    return my_qcnn


def test_qcnn(test_input, test_labels, my_qcnn, fpath):
    file = open(fpath + 'loss_accuracy.txt', 'w')
    pred = my_qcnn.forward(test_input, my_qcnn.params.copy())

    mse_loss = my_qcnn.mse_loss(pred, test_labels)
    acc = 0

    for prediction, label in zip(pred, test_labels):
        if np.ceil(prediction) == float(label):
            acc += 1
    acc = acc / len(pred)

    file.write('MSE Loss = {}\n'.format(mse_loss))
    file.write('Accuracy = {}\n'.format(acc))
    file.write('Prediction:\n{}'.format(pred))
    file.close()

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


def og_model(num_qubits):
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
    num_qubits = 8
    unique_name = f"n{num_qubits}_digit_qcnn_train/"

    wavefuncs_train, wavefuncs_test, labels_train, labels_test = get_wfs_from_images(num_qubits)

    og_qcnn = og_model(num_qubits)
    # new_qcnn = new_model(num_qubits)

    trained_og_qcnn = run_qcnn(wavefuncs_train, labels_train, og_qcnn, "og_" + unique_name)
    # trained_new_qcnn = run_qcnn(wavefuncs_train, labels_train, new_qcnn, "new" + unique_name)

    mse_loss, acc, pred = test_qcnn(wavefuncs_test, labels_test, trained_og_qcnn, "og_" + unique_name)
    # mse_loss, acc, pred = test_qcnn(wavefuncs_test, labels_test, trained_new_qcnn)

    print(mse_loss)
    print(acc)
    print(pred)
    print(f"* * * * * * * * * * * * * * *Finished {num_qubits}, qbits! * * * * * * * * * * * * * * *")
    return


if __name__ == "__main__":
    main()
