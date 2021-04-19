import numpy as np

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=2, padding=1, stride=2)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=(2, 1), stride=1)

        self.struct = nn.Sequential(self.conv1,
                                    nn.ReLU(),
                                    self.conv2,
                                    nn.ReLU(),
                                    self.conv3,
                                    nn.Sigmoid())

        # self.struct2 = nn.Sequential(nn.Linear(4, 2),
        #                              nn.Sigmoid())


    # Feedforward function
    def forward(self, x):

        x11 = self.struct(x)
        # x1 = torch.flatten(x11)
        x2 = torch.reshape(x11, (x.shape[0], 2))
        return x2

    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()

    # Backpropagation function
    def backprop(self, data, loss, optimizer):
        self.train()

        batch_size = data.x_train.shape[0]
        inputs = torch.from_numpy(np.reshape(data.x_train, (batch_size, 1, 4, 4))).float()
        targets = torch.from_numpy(data.y_train).float()
        # An alternative to what you saw in the jupyter notebook is to
        # flatten the output tensor. This way both the targets and the model
        # outputs will become 1-dim tensors.


        obj_val = loss(self.forward(inputs), targets)
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()

    # Test function. Avoids calculation of gradients.
    def test(self, data, loss):
        self.eval()
        with torch.no_grad():
            test_size = data.x_test.shape[0]
            inputs = torch.from_numpy(np.reshape(data.x_test, (test_size, 1, 4, 4))).float()
            targets = torch.from_numpy(data.y_test).float()

            cross_val = loss(self.forward(inputs), targets)
        return cross_val.item()

    def accuracy(self, x, y):
        self.eval()

        with torch.no_grad():
            # Convert the input data to a tensor and pass into the model
            x_new = torch.from_numpy(np.reshape(x, (x.shape[0], 1, 4, 4))).float()
            y_new = torch.from_numpy(y).float()

            output = self(x_new)
            prediction = np.round(output.numpy())  # Round to closest values

            differences = np.abs(y_new - prediction)
            num_of_wrong = np.count_nonzero(np.count_nonzero(differences, axis=1))

        return len(x_new) - num_of_wrong, len(x_new)