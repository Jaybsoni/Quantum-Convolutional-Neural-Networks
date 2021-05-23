import copy
import itertools
import multiprocessing as mp
import pickle

import numpy as np
from qiskit import QuantumCircuit
from qiskit import quantum_info as qi

from qml import layers


class QcnnStruct:
    """
    The base class which our QCNN will inherit core functionality from.
    """

    Layers = {layers.legacy_conv4_layer.name: layers.legacy_conv4_layer,   # A dict of layers which is used to simplify
              layers.legacy_conv_layer.name: layers.legacy_conv_layer,     # the construction of a Qcnn
              layers.legacy_pool_layer.name: layers.legacy_pool_layer}

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.active_qubits = list(range(num_qubits))   # pooling layers reduce the number of 'active qubits'
        self.structure = []  # this will contain a summary of the structure of the model as an ordered list of layers
        self.params = []  # this will contain the trainable parameters of the model and their current values

    def add_layer(self, layer, kwargs={}):
        """
        Adds a layer to the structure of the model along with the layer
        specific keyword params
        :param layer: layers.Layer, a specific layer
        :param kwargs: dict, a dictionary containing keyword params
        :return: None
        """
        self.structure.append((layer.name, kwargs))
        return

    def initialize_params(self, specific_params=None, random=False):
        """
        Once the structure of the model is set, this method is called
        to initialize the params of the model. If random=True, then the method
        initializes random parameters according to the structure of the model. Else
        the parameters of the model are set to 1.

        :param specific_params: A list of np.arrays, corresponding to the parameters of this model
        :param random: Bool, defaul=False,
        :return: None
        """

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
        """
        A method which takes a list (active qubits), and then
        generates a new list containing a subset of the active qubits.

        This subset is constructed by assuming a multi-controlled single target pooling scheme
        which considers the target qubits as the resultant active qubits.

        Eg: active_qubits = [0, 1, 2, 3, 4, 5], group_len=3, target=2
        then the active qubits are grouped together in groups of 3 where the 2 index qubit is the
        active qubit post pooling. Thus update_qubits = [2, 5]

        :param active_qubits: list containing ints, of active qubits indicies
        :param group_len: int, the grouping size (must be <= len(active_qubits))
        :param target: int, the index of the target qubit (must be < group_len)
        :return: list (update_qubits),  of the new active qubits
        """
        num_groups = len(active_qubits) // group_len
        update_qubits = []

        for i in range(num_groups):
            index = i * group_len + target
            update_qubits.append(active_qubits[index])

        return update_qubits

    def update_active_qubits(self, group_len, target):
        """
        Wrapper function which calls the static method and updates the
        instance variable active_qubits

        :param group_len: int, the grouping size (must be <= len(active_qubits))
        :param target: int, the index of the target qubit (must be < group_len)
        :return: None
        """
        self.active_qubits = self.get_updated_active_qubits(self.active_qubits, group_len, target)
        return

    def reset_active_qubits(self):
        """
        resets the active_qubits instance variable
        :return:
        """
        self.active_qubits = list(range(self.num_qubits))
        return

    def generate_circ(self, params, draw=False):
        """
        A core method of this class, this function generates a qiskit Quantum Circuit
        object from the structure of the model and the model parameters provided

        :param params: list of np.arrays, containing parameter values corresponding to the model structure
        :param draw: Bool, default=False, If true it will plot circuit
        :return: circ, Qiskit QuantumCircuit object based on model structure and parameters
        """
        circ = QuantumCircuit(self.num_qubits)  # initialize circuit object

        for index, layer_info in enumerate(self.structure):  # iterate through the structure
            layer_name, kwargs = layer_info
            layer = self.Layers[layer_name]

            layer.apply_layer(circ, params[index], self.active_qubits, kwargs)  # apply each layer in order

            if "update_active_qubits" in kwargs:
                update_params_dict = kwargs["update_active_qubits"]
                group_len = update_params_dict["group_len"]
                target = update_params_dict["target"]

                self.update_active_qubits(group_len, target)  # update active qubits as required

        self.reset_active_qubits()  # once the circ object is generated, reset the active qubits
        circ = circ.reverse_bits()  # swap to the convention found in textbooks

        if draw:
            circ.draw(reverse_bits=True)

        return circ

    def get_final_state_active_qubits(self):
        """
        Since qcnns can have multiple convolutional layers and pooling layers
        it is valuable to know which qubits remain 'active' at the end of the model
        so that we can measure only those qubits.

        This method uses the structure of the model to produce a list of 'active qubits'
        :return: active_qubits, list of ints, a list of the indicies of the active qubits
        """
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
        """
        Takes a list of complex coefficients corresponding to a wave function and generates
        a qiskit Statevector object.

        :param wf: np.array of complex values, must be normalized
        :return: q_state, qiskit Statevector object corresponding to the given wave function
        """
        q_state = qi.Statevector(wf)
        return q_state

    @staticmethod
    def get_operator(circ):
        """
        given a qiskit QuantumCircuit object, get qiskit Operator object
        :param circ: qiskit QuantumCircuit object
        :return: operator, qiskit Operator object
        """
        operator = qi.Operator(circ)
        return operator


class Qcnn(QcnnStruct):
    """
    This is the class we use to generate the QCNN
    """

    def __init__(self, num_qubits):
        super().__init__(num_qubits)

    def forward(self, input_wfs, params):
        """
        Uses the input wave functions as the initial state of the system
        and evolves using the qcnn model structure with provided parameters
        in order to generate the final predictions of the model.

        :param input_wfs: np.array of complex, the initial wavefunctions
        :param params: parameters of the model
        :return: predictions of the model
        """
        circ = self.generate_circ(params)       # generate circuit for given parameters
        predictions = np.zeros(len(input_wfs))  # initialize the predictions

        for index, wf in enumerate(input_wfs):
            state = self.embedding(wf)
            state = state.evolve(circ)          # evolve the wf according to the circuit

            predictions[index] = self.middle_qubit_exp_value(state)  # get predictions as expt_val of X of middle qubit

        return predictions

    def compute_grad(self, input_wfs, labels, epsilon=0.0001):
        """
        We compute the gradiant manually using finite differences.
        eg: f'(x)|_a = f(a + epsilon) - f(a - epsilon) / 2*epsilon

        :param input_wfs: np.array of complex, the initial wavefunctions
        :param labels: list of correct labels which the predictions will be compared against in the loss function
        :param epsilon: float, the parameter shift in the finite differenc calculation
        :return: gradient_mat, list of np.arrays containing the rate of change of each parameter independantly
        """
        original_params = copy.deepcopy(self.params)  # copy original params to prevent potential loss of information
        gradient_mat = []

        for layer_index, layer_params in enumerate(self.params):  # iterate over each layer
            layer_grad = np.zeros(len(layer_params))

            for param_index, _ in enumerate(layer_params):  # iterate over each parameter in the layer
                grad = 0                                    # initialize gradient
                for i in [1, -1]:
                    self.params[layer_index][param_index] += i * epsilon  # shift params by epsilon
                    grad += i * self.mse_loss(self.forward(input_wfs, self.params.copy()), labels)
                    self.params = copy.deepcopy(original_params)  # reset params to original values
                layer_grad[param_index] = grad / 2 * epsilon      # compute final grad and append to layer_grad

            gradient_mat.append(layer_grad)   # append all the layers into the grad matrix

        return gradient_mat

    def pool_func_for_mp(self, indexes, input_wfs, labels, epsilon):
        """
        Compute the rate of change of the i,j th parameter in the params array. This function will allow us
        to compute the gradient using multi-threading.

        :param indexes: the coordinates of the parameter which we are computing the rate of change for
        :param input_wfs: np.array of complex, the initial wavefunctions
        :param labels: list of correct labels which the predictions will be compared against in the loss function
        :param epsilon: float, the parameter shift in the finite differenc calculation
        :return: tuple (int, int, float), containing the indexes and the final gradient
        """
        i, j = indexes
        grad = 0

        for k in [1, -1]:
            params = copy.deepcopy(self.params)
            params[i][j] += k * epsilon  # shift params
            grad += k * self.mse_loss(self.forward(input_wfs, params), labels)

        final_grad = grad / 2 * epsilon
        return i, j, final_grad

    def compute_grad_w_mp(self, input_wfs, labels, epsilon=0.0001):
        """
        Wrapper function for pool_func_for_mp which now computes the entire gradient matrix using
        multi-threading.

        :param input_wfs: np.array of complex, the initial wavefunctions
        :param labels: list of correct labels which the predictions will be compared against in the loss function
        :param epsilon: float, the parameter shift in the finite differenc calculation
        :return: gradient_mat, list of np.arrays containing the rate of change of each parameter independantly
        """
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
        """
        updates the parameters of the model using the gradient matrix and the learning rate
        :param gradient_mat: list of np.arrays containing the rate of change of each parameter
        :param learning_rate: float, a scale factor used to determine how much to update the parameters of the model
        :return: None
        """
        for param_layer, grad_layer in zip(self.params, gradient_mat):
            param_layer -= learning_rate * grad_layer  # step in direction of -grad with size learning rate
        return

    def load_model(self, model_struct, specific_params):
        """
        Given a specific model structure and specific model parameters, it will load that model
        in this instance of qcnn.

        :param model_struct: list of tuples containing information about the layers and arguments for the structure
        :param specific_params: list of np.arrays, the trained parameters of the model
        :return: None
        """
        self.params = specific_params
        self.structure = model_struct
        return

    @staticmethod
    def mse_loss(predictions, labels):
        """
        local implementation of mean squared error:
        eg: mse = (len(x))^-1 * sum_i((x_label_i - x_pred_i)^2)

        :param predictions: list of floats, predictions from the model
        :param labels: list of floats, the correct values for those given inputs
        :return: mean_squared_error of the predicitons
        """
        num_entries = len(predictions)  # should be the same as len(labels)
        loss = np.sum(np.power((labels - predictions), 2))

        return loss / num_entries

    def middle_qubit_exp_value(self, state_vect):
        """
        Computes the expectation value of X on the middle qubit. This quantity is used for final predictions of the
        model.
        :param state_vect: Qiskit Statevector object, contains the probability densities for the final state of the sys
        :return: the expectation value of X in the middle qubit
        """
        final_active_qubits = self.get_final_state_active_qubits()
        middle_qbit = final_active_qubits[len(final_active_qubits) // 2]  # this is the index of the middle qubit

        probability_vector = (np.abs(np.array(state_vect.data))) ** 2  # probability of each outcome

        all_binary_combs = list(map(list, itertools.product([0, 1], repeat=self.num_qubits)))
        new_lst = np.array([elem for elem, val in enumerate(all_binary_combs) if val[middle_qbit] == 1])
        sums = np.sum(probability_vector[new_lst])

        return (-1 * sums) + (1 * (1 - sums))

    @staticmethod
    def export_params(qcnn_struct, params, fname="model.pkl"):
        """
        Export the structure of the qcnn and the trained parameters into a pkl file
        :param qcnn_struct: ordered list of tuples containing the layers and arguments for the structure of the model
        :param params: list of np.arrays, the trained parameters of the model
        :param fname: str, name of the file, default='model.pkl'
        :return: None
        """
        with open(fname, 'wb') as file:
            pickle.dump((qcnn_struct, params), file)  # Save data as pickle

    @staticmethod
    def import_params(fname="model.pkl"):
        """
        Read exported model pickle file and extract the structure and parameters of the model.
        :param fname: str, file name
        :return: qcnn_struct, params
        """
        with open(fname, 'rb') as file:
            qcnn_struct, params = pickle.load(file)  # Call load method to deserialze

        return qcnn_struct, params


def main():
    return


if __name__ == "__main__":
    main()
