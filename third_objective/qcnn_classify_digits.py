import os
import sys
import copy
import tqdm
import numpy as np
import matplotlib.pyplot as plt 

sys.path.insert(0, "../qml_src/")
from qml import qcnn as q
from qml import layers as cl  # custom layers
from scraper import Data


def get_wfs_from_images():
    data = Data()

    wfs_train = [vect / np.linalg.norm(vect) for vect in data.x_train]
    wfs_test = [vect / np.linalg.norm(vect) for vect in data.x_test]

    padding = np.zeros(256 - 196)

    wfs_train = [np.append(wf, padding) for wf in wfs_train]
    wfs_test = [np.append(wf, padding) for wf in wfs_test]

    label_train = [int(i == 4) for i in data.y_train]
    label_test = [int(i == 4) for i in data.y_test]

    return wfs_train, wfs_test, label_train, label_test


def run_qcnn(train_data, labels, my_qcnn, unique_name):
    # Initialize parameters:
    my_qcnn.initialize_params(random=True)
    initial_params = copy.deepcopy(my_qcnn.params)

    if not os.path.isdir("results/" + unique_name):
        os.mkdir("results/" + unique_name)

    # Create backup files
    loss_file = f"./results/{unique_name}/Loss.txt"
    param_file = f"./results/{unique_name}/Parameters.txt"
    acc_file = f"./results/{unique_name}/TrainingAccuracy.txt"
    if os.path.isfile(loss_file): os.remove(loss_file)
    if os.path.isfile(acc_file): os.remove(acc_file)
    if os.path.isfile(param_file): os.remove(param_file)

    batch_size = 50
    num_batches = 10 #len(train_data) // batch_size
    num_epoches = 100
    loss_lst = []  # initialize
    acc_lst = []

    for batch_index in range(1, num_batches + 1):
        batched_data = train_data[(batch_index - 1)*batch_size: batch_index*batch_size]
        batched_labels = labels[(batch_index - 1)*batch_size: batch_index*batch_size]
        learning_rate = 100000  # intial value was 10 but this quantity doesn't learn fast enough !
        accuracy = 0

        for epoch in tqdm.tqdm(range(1, num_epoches + 1)):
            pred = my_qcnn.forward(batched_data, my_qcnn.params.copy())
            loss = my_qcnn.mse_loss(pred, batched_labels)

            if epoch != 1:
                successive_loss = loss - loss_lst[-1]
                if successive_loss < 0:
                    learning_rate *= 1.05  # if loss decreases, increase learning rate by 5%
                else:
                    learning_rate /= 2  # if it gets bigger, decrease learning rate by 50%

            # grad_mat = my_qcnn.compute_grad(train_data, labels)
            grad_mat = my_qcnn.compute_grad_w_mp(batched_data, batched_labels)  # with multi processing

            my_qcnn.update_params(grad_mat, learning_rate)
            loss_lst.append(loss)

            # Write to file each time to avoid saving to ram
            with open(loss_file, 'a+') as f:
                f.write(str(loss) + "\n")

        # Write to file each time to avoid saving to ram
        with open(param_file, 'w+') as f:
            tempCopy = copy.deepcopy(my_qcnn.params)
            f.write(str(tempCopy) + "\n")

        for prediction, label in zip(pred, batched_labels):
            if np.round(prediction) == float(label):
                accuracy += 1

        acc_lst.append(accuracy / batch_size)
        with open(acc_file, 'a+') as f:
            f.write(str(accuracy/batch_size) + "\n")

    # model end --------------------------------------------------------------------------------------------------------


    optimal_params = copy.deepcopy(my_qcnn.params)
    my_qcnn.export_params(my_qcnn.structure, optimal_params, fname=f'./results/{unique_name}/model_optimal.pkl')
    my_qcnn.export_params(my_qcnn.structure, initial_params, fname=f'./results/{unique_name}/model_initial.pkl')

    # Loss plot:
    x_axis = range(len(loss_lst))
    plt.plot(x_axis, loss_lst)
    plt.title('Training Loss over Epoches')
    plt.xlabel('Epoches')
    plt.ylabel("Loss")
    plt.savefig(f"./results/{unique_name}/loss.png")
    plt.close()

    # Accuracy plot:
    x_axis = range(len(acc_lst))
    plt.plot(x_axis, acc_lst)
    plt.title('Training Accuracy over Batches')
    plt.xlabel('Batches')
    plt.ylabel("Accuracy")
    plt.savefig(f"./results/{unique_name}/acc.png")
    plt.close()

    return my_qcnn


def qcnn_test(test_input, test_labels, my_qcnn, fpath):
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
    unique_name = f"n{num_qubits}_digit_qcnn_train"

    wavefuncs_train, wavefuncs_test, labels_train, labels_test = get_wfs_from_images()

    wavefuncs_train_subset = wavefuncs_train[:5000]
    labels_train_subset = labels_train[:5000]

    og_qcnn = og_model(num_qubits)
    # new_qcnn = new_model(num_qubits)

    trained_og_qcnn = run_qcnn(wavefuncs_train_subset, labels_train_subset, og_qcnn, "og_" + unique_name)
    # trained_new_qcnn = run_qcnn(wavefuncs_train_subset, labels_train_subset, new_qcnn, "new" + unique_name)

    mse_loss, acc, pred = qcnn_test(wavefuncs_test, labels_test, trained_og_qcnn, "og_" + unique_name)
    # mse_loss, acc, pred = test_qcnn(wavefuncs_test, labels_test, trained_new_qcnn)

    print(mse_loss)
    print(acc)
    print(pred)
    print(f"* * * * * * * * * * * * * * *Finished {num_qubits}, qbits! * * * * * * * * * * * * * * *")
    return


if __name__ == "__main__":
    main()
