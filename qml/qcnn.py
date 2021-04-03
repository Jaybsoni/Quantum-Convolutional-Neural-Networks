# qcnn
import numpy as np
from layers import legacy_conv4_layer, legacy_conv_layer, legacy_pool_layer
from qiskit import QuantumCircuit
from qiskit import quantum_info as qi


class QcnnStruct:

    Layers = {legacy_conv4_layer.name: legacy_conv4_layer,
              legacy_conv_layer.name: legacy_conv_layer,
              legacy_pool_layer.name: legacy_pool_layer}

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.active_qubits = list(range(num_qubits))
        self.structure = []
        self.params = []

    def add_layer(self, layer, kwargs):
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
        assert len(active_qubits) % group_len == 0
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

    def generate_circ(self):
        circ = QuantumCircuit(self.num_qubits)

        for index, layer_info in enumerate(self.structure):
            layer_name, kwargs = layer_info
            layer = self.Layers[layer_name]

            layer.apply_layer(circ, self.params[index], self.active_qubits, kwargs)

            if "update_active_qubits" in kwargs:
                update_params_dict = kwargs["update_active_qubits"]
                group_len = update_params_dict["group_len"]
                target = update_params_dict["target"]

                self.update_active_qubits(group_len, target)

        self.reset_active_qubits()
        return circ.reverse_bits()  # swap to the convention found in textbooks

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

    def forward(self, input_wfs):
        circ = self.generate_circ()
        predictions = np.zeros(len(input_wfs))

        for index, wf in enumerate(input_wfs):
            state = self.embedding(wf)
            state = state.evolve(circ)

            predictions[index] = self.middle_qubit_exp_value(state)

        return predictions

    def compute_grad(self, input_wfs, labels, epsilon=0.0001):
        original_params = self.params.copy()
        gradient_mat = []

        for layer_index, layer_params in self.params:
            layer_grad = np.zeros(len(layer_params))
            for param_index, param in enumerate(layer_params):
                self.params[layer_index][param_index] += epsilon   # shift param by epsilon
                plus_epsilon_pred = self.forward(input_wfs)
                plus_epsilon_loss = self.mse_loss(plus_epsilon_pred, labels)
                self.params = original_params.copy()  # reset params to original values

                self.params[layer_index][param_index] -= epsilon   # shift param by epsilon
                minus_epsilon_pred = self.forward(input_wfs)
                minus_epsilon_loss = self.mse_loss(minus_epsilon_pred, labels)
                self.params = original_params.copy()  # reset params to original values

                grad = (plus_epsilon_loss - minus_epsilon_loss) / 2*epsilon
                layer_grad[param_index] = grad

            gradient_mat.append(layer_grad)

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
        num_qubits = len(final_active_qubits)
        middle_qubit_index = num_qubits // 2

        middle_qubit = final_active_qubits[middle_qubit_index]  # this is the index of the middle qubit

        operator_circ = QuantumCircuit(self.num_qubits)  # define the circuit for the operator we want to computer
        operator_circ.x(middle_qubit_index)              # expectation value for
        operator_circ = operator_circ.reverse_bits()

        operator = qi.Operator(operator_circ)  # operator of interest
        exp_value = state_vect.expectation_value(operator)  # expectation value of X operator on middle qubit

        return exp_value


def main():
    return


if __name__ == "__main__":
    main()
