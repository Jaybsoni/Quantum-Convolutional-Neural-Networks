import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 12)
        self.fc2 = nn.Linear(12, 8)
        self.fc3 = nn.Linear(8, 2)

    # Feedforward function
    def forward(self, x):
        h = func.relu(self.fc1(x))
        h = func.relu(self.fc2(h))
        h = torch.sigmoid(self.fc3(h))
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
        inputs = torch.from_numpy(data.x_train)
        targets = torch.from_numpy(data.y_train)
        # An alternative to what you saw in the jupyter notebook is to
        # flatten the output tensor. This way both the targets and the model
        # outputs will become 1-dim tensors.
        obj_val = loss(self.forward(inputs).reshape(-1), targets)
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