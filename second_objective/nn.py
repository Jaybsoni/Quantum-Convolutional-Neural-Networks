import numpy as np

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=2, padding=1, stride=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=2, stride=1)

        self.struct = nn.Sequential(self.conv1, nn.ReLU(),
                                    self.conv2, nn.ReLU(),
                                    self.conv3, nn.Sigmoid())

    # Feedforward function
    def forward(self, x):
        import time
        h = self.struct(x)
        # print(x.shape)
        # h = self.conv1(x)
        # print(h.shape)
        # time.sleep(1)
        # h = nn.ReLU()
        # time.sleep(1)

        # h = func.relu(self.conv2(h))
        # h = torch.sigmoid(self.conv3(h))
        return h

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
            inputs = torch.from_numpy(data.x_test)
            targets = torch.from_numpy(data.y_test)
            cross_val = loss(self.forward(inputs).reshape(-1), targets)
        return cross_val.item()

    def accuracy(self, x, y):
        self.eval()

        with torch.no_grad():
            # Convert the input data to a tensor and pass into the model
            output = self(torch.from_numpy(x))
            prediction = np.round(output.numpy())  # Round to closest values

            differences = np.abs(y - prediction)
            num_of_wrong = np.count_nonzero(np.count_nonzero(differences, axis=1))

        return len(x) - num_of_wrong, len(x)