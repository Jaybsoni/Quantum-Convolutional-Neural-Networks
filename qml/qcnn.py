# qcnn
import copy
import numpy as np
from layers import legacy_conv4_layer, legacy_conv_layer, legacy_pool_layer
from qiskit import QuantumCircuit
from qiskit import quantum_info as qi
import itertools
import multiprocessing as mp


class QcnnStruct:

    Layers = {legacy_conv4_layer.name: legacy_conv4_layer,
              legacy_conv_layer.name: legacy_conv_layer,
              legacy_pool_layer.name: legacy_pool_layer}

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.active_qubits = list(range(num_qubits))
        self.structure = []
        self.params = []

    def add_layer(self, layer, kwargs={}):
        self.structure.append((layer.name, kwargs))
        return

    def initialize_params(self, specific_params=None, random=False):

        if specific_params is not None:
            self.params = specific_params
            return

        params = []
        for layer_info in self.structure:
            layer = self.Layers[layer_info[0]]
            layer_param_shape = layer.shape_params

            if random:
                layer_params = np.random.uniform(0, 2*np.pi, layer_param_shape)
            else:
                layer_params = np.ones(layer_param_shape)

            params.append(layer_params)

        self.params = params
        return

    @staticmethod
    def get_updated_active_qubits(active_qubits, group_len, target):
        num_groups = len(active_qubits) // group_len
        update_qubits = []

        for i in range(num_groups):
            index = i * group_len + target
            update_qubits.append(active_qubits[index])

        return update_qubits

    def update_active_qubits(self, group_len, target):
        self.active_qubits = self.get_updated_active_qubits(self.active_qubits, group_len, target)
        return

    def reset_active_qubits(self):
        self.active_qubits = list(range(self.num_qubits))
        return

    def generate_circ(self, params, draw=False):
        circ = QuantumCircuit(self.num_qubits)

        for index, layer_info in enumerate(self.structure):
            layer_name, kwargs = layer_info
            layer = self.Layers[layer_name]

            layer.apply_layer(circ, params[index], self.active_qubits, kwargs)

            if "update_active_qubits" in kwargs:
                update_params_dict = kwargs["update_active_qubits"]
                group_len = update_params_dict["group_len"]
                target = update_params_dict["target"]

                self.update_active_qubits(group_len, target)

        self.reset_active_qubits()
        circ = circ.reverse_bits()  # swap to the convention found in textbooks

        if draw:
            circ.draw(reverse_bits=True)

        return circ

    def get_final_state_active_qubits(self):
        active_qubits = self.active_qubits.copy()

        for index, layer_info in enumerate(self.structure):
            layer_name, kwargs = layer_info

            if "update_active_qubits" in kwargs:
                update_params_dict = kwargs["update_active_qubits"]
                group_len = update_params_dict["group_len"]
                target = update_params_dict["target"]

                active_qubits = self.get_updated_active_qubits(active_qubits, group_len, target)

        return active_qubits

    @staticmethod
    def embedding(wf):
        q_state = qi.Statevector(wf)
        return q_state

    @staticmethod
    def get_operator(circ):
        operator = qi.Operator(circ)
        return operator


class Qcnn(QcnnStruct):

    def __init__(self, num_qubits):
        super().__init__(num_qubits)

    def forward(self, input_wfs, params):
        circ = self.generate_circ(params)
        predictions = np.zeros(len(input_wfs))

        for index, wf in enumerate(input_wfs):
            state = self.embedding(wf)
            state = state.evolve(circ)

            predictions[index] = self.middle_qubit_exp_value(state)

        return predictions

    def compute_grad(self, input_wfs, labels, epsilon=0.0001):
        original_params = copy.deepcopy(self.params)
        gradient_mat = []

        for layer_index, layer_params in enumerate(self.params):
            layer_grad = np.zeros(len(layer_params))

            for param_index, _ in enumerate(layer_params):
                grad = 0
                for i in [1, -1]:
                    self.params[layer_index][param_index] += i * epsilon  # shift params
                    grad += i * self.mse_loss(self.forward(input_wfs, self.params.copy()), labels)
                    self.params = copy.deepcopy(original_params)  # reset params to original values
                layer_grad[param_index] = grad / 2 * epsilon

            gradient_mat.append(layer_grad)

        return gradient_mat

    def pool_func_for_mp(self, indexes, input_wfs, labels, epsilon):
        i, j = indexes
        grad = 0

        for k in [1, -1]:
            params = copy.deepcopy(self.params)
            params[i][j] += k * epsilon  # shift params
            grad += k * self.mse_loss(self.forward(input_wfs, params), labels)

        layer_grad = grad / 2 * epsilon
        return i, j, layer_grad

    def compute_grad_w_mp(self, input_wfs, labels, epsilon=0.0001):
        indexes = [(i, j) for i, val in enumerate(self.params) for j, _ in enumerate(val)]

        p = mp.Pool(mp.cpu_count())
        grad_tuple = p.starmap(self.pool_func_for_mp, zip(indexes,
                                                          itertools.repeat(input_wfs),
                                                          itertools.repeat(labels),
                                                          itertools.repeat(epsilon)))

        gradient_mat = copy.deepcopy(self.params)
        for i, j, val in grad_tuple:
            gradient_mat[i][j] = val

        return gradient_mat

    def update_params(self, gradient_mat, learning_rate):
        for param_layer, grad_layer in zip(self.params, gradient_mat):
            param_layer -= learning_rate * grad_layer  # step in direction of -grad with size learning rate
        return

    def load_model(self, model_struct, specific_params):
        self.params = specific_params
        self.structure = model_struct
        return

    @staticmethod
    def mse_loss(predictions, labels):
        num_entries = len(predictions)  # should be the same as len(labels)
        loss = np.sum(np.power((labels - predictions), 2))

        return loss / num_entries

    def middle_qubit_exp_value(self, state_vect):
        final_active_qubits = self.get_final_state_active_qubits()
        middle_qbit = final_active_qubits[len(final_active_qubits) // 2]  # this is the index of the middle qubit

        probability_vector = (np.abs(np.array(state_vect.data))) ** 2

        all_binary_combs = list(map(list, itertools.product([0, 1], repeat=self.num_qubits)))
        new_lst = np.array([elem for elem, val in enumerate(all_binary_combs) if val[middle_qbit] == 1])
        sums = np.sum(probability_vector[new_lst])

        return (-1 * sums) + (1 * (1 - sums))


def main():
    return


if __name__ == "__main__":
    main()
