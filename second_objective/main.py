import time, math, pickle
import json, sys
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.optim as optim

import nn
import generate_tetris_blocks as gtb


sys.path.append('src')

def plot_results(obj_vals, cross_vals, filename):
    assert len(obj_vals) == len(cross_vals), 'Length mismatch between the curves'
    num_epochs = len(obj_vals)

    # Plot saved in results folder
    plt.plot(range(num_epochs), obj_vals, label="Training loss", color="blue")
    plt.plot(range(num_epochs), cross_vals, label="Test loss", color="green")
    plt.legend()
    plt.savefig(filename + '.pdf')
    plt.close()

def train_model(param, model, data):
    # Define an optimizer and the loss function
    optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
    loss = torch.nn.BCELoss(reduction='mean')

    obj_vals = []
    cross_vals = []
    num_epochs = int(param['num_epochs'])

    # Training loop
    for epoch in range(1, num_epochs + 1):
        obj_vals.append(model.backprop(data, loss, optimizer))
        cross_vals.append(model.test(data, loss))

        # High verbosity report in output stream
        if v >= 2 and not ((epoch + 1) % param['display_epochs']):
            train_cor, train_tot = model.accuracy(data.x_train, data.y_train)
            test_cor, test_tot1 = model.accuracy(data.x_test, data.y_test)

            t = time.time() - start_time
            print('Epoch [{}/{}] - {}m{:.2f}s -'.format(epoch + 1, num_epochs, math.floor(t / 60), t % 60) +
                  '\tTraining Loss: {:.4f}  - '.format(obj_vals[-1]) +
                  'Training Accuracy: {}/{} ({:.2f}%)  -'.format(train_cor, train_tot, 100 * train_cor/train_tot) +
                  'Test Loss: {:.4f}  - '.format(cross_vals[-1]) +
                  'Test Accurecy: {}/{} ({:.2f}%)'.format(test_cor, test_tot1, 100 * test_cor/test_tot1))

    # Low verbosity final report
    if v >= 1:
        print('Final training loss: {:.4f} \t Final test loss: {:.4f}'.format(obj_vals[-1], cross_vals[-1]))

    return obj_vals, cross_vals



if __name__ == '__main__':
    start_time = time.time()
    v = 2
    filename = "temp"

    # Hyperparameters from json file are loaded
    with open("param.json") as paramfile:
        param = json.load(paramfile)

    # Create an instance of the model and initialize the data
    model = nn.Net()

    data = gtb.TetrisData(4)
    data.generate_combinations_of_tetris_blocks()
    data.package_data()

    if v:
        print(f"\n Learning rate given as {param['learning_rate']}, "
              f"with {param['num_epochs']} Epochs, and a verbosity of {v}\n"
              f"No model given, training new model")

    obj_vals, cross_vals = train_model(param, model, data)

    # Save the model and the picked data
    torch.save(model.state_dict(), filename + ".pth")

    # Save the training and test losses
    with open(filename + ".pkl", 'wb') as pfile:
        pickle.dump((obj_vals, cross_vals), pfile)

    # if v:
    #     print(f"\n Learning rate given as {param['exec']['learning_rate']}, "
    #           f"with {param['exec']['num_epochs']} Epochs, and a verbosity of {args.v}\n"
    #           f"Model given, attempting to load it")
    #
    # model.load_state_dict(torch.load(filename + ".pth"))
    # with open(filename + ".pkl", 'rb') as pfile:
    #     obj_vals, cross_vals = pickle.load(pfile)

    plot_results(obj_vals, cross_vals, filename)

    # # Print final loss/acceptances
    # if v:
    #     t = time.time() - start_time
    #     correct, total = model.accuracy(data.x_test, data.y_test)
    #     print("\nAfter training, we find the model has a test loss of {:.3f} ".format(cross_vals[-1]) +
    #           "and an accuracy of {}/{}, or {:.3f}%\n".format(correct, total, (correct/total) * 100) +
    #           "This was completed in {} minutes and {:.2f} seconds".format(math.floor(t / 60), t % 60))