import numpy as np
import scipy.linalg as la
from qiskit import quantum_info as qi


# Helper Functions ################################################
def b_mat(i, j, n):
    """
    Generates an n x n matrix of 0s with the i,j th entry is a one.
    This is the i,j th basis vector on the space of n x n real matricies

    :param i: int, row index (must be < n)
    :param j: int, column index (must be < n)
    :param n: int, dimension of the matrices
    :return: np.array of floats, shape (n,n)
    """
    basis_matrix = np.zeros((n, n), dtype=np.float32)
    basis_matrix[i, j] = 1.0

    return basis_matrix


def generate_gell_mann(order):
    """
    Generates a list of np.arrays which represent Gell Mann matricies of order 'order'.
    eg: order = 2
    lst_of_gm_matricies = [ [[0,  1],
                             [1,  0]] ,

                            [[0, -i]
                             [i,  0]] ,

                            [[1,  0],
                             [0, -1]] ]
    :param order: int, the order of Gell Mann matricies
    :return: list of np.arrays, each array has shape (order, order), there are order^2 - 1 such elements in the lst
    """

    lst_of_gm_matricies = []
    for k in range(order):
        j = 0
        while j < k:
            sym_mat = b_mat(j, k, order) + b_mat(k, j, order)
            anti_sym_mat = complex(0.0, -1.0) * (b_mat(j, k, order) - b_mat(k, j, order))

            lst_of_gm_matricies.append(sym_mat), lst_of_gm_matricies.append(anti_sym_mat)
            j += 1

        if k < (order - 1):
            n = k + 1
            coeff = np.sqrt(2 / (n*(n+1)))

            sum_diag = b_mat(0, 0, order)
            for i in range(1, k+1):
                sum_diag += b_mat(i, i, order)

            diag_mat = coeff * (sum_diag - n*(b_mat(k+1, k+1, order)))
            lst_of_gm_matricies.append(diag_mat)

    return lst_of_gm_matricies


def get_conv_op(mats, parms):
    """
    The convolutional operator is parameterized according to
    gell mann matricies scaled by trainable parameters, this method generates the operator as
    defined in the paper.

    eg. Convolutional operator = exp(-i * Sum_j(GM_j * theta_j) )
    :param mats: lst of np.arrays which contain the self adjoint matricies used in the parameterization
    :param parms: lst of floats which are the scale parameters
    :return: np.array which represents the final convolutional operator
    """
    final = np.zeros(mats[0].shape, dtype=np.complex128)
    for mat, parm in zip(mats, parms):  # sum over the gm matricies scaled by the parameters
        final += parm * mat

    return la.expm(complex(0, -1) * final)  # get the matrix exponential of the final matrix


def controlled_pool(mat):
    """
    Generate the matrix corresponding the controlled - mat operator.

    :param mat: np.array, shape (2x2) for the controlled operator
    :return: np.array, the final controlled-mat operator
    """
    i_hat = np.array([[1.0, 0.0],
                      [0.0, 0.0]])
    j_hat = np.array([[0.0, 0.0],
                      [0.0, 1.0]])
    identity = i_hat + j_hat

    return np.kron(i_hat, identity) + np.kron(j_hat, mat)


def generate_uniformly_controlled_rotation(circ, params, control_qubit_indicies,
                                           target_qubit_index, axis='z', label=""):
    """
    This function implements a circuit for performing a multi-controlled rotation about a specified axis. The
    specifics can be found at: https://arxiv.org/pdf/quant-ph/0407010.pdf

    :param circ: qiskit QuantumCircuit object, the circuit we wish to augment with the unif controlled rotation
    :param params: np.array of real valued float, contains parameters which specify the amount of rotation
    :param control_qubit_indicies: list of ints, a list containing the indicies of the control qubits in the circuit
    :param target_qubit_index: int, index of the target qubit in the circuit
    :param axis: str, one of 'x', 'y', or 'z' which determine which axis the rotations will occur around
    :param label: str, custom name for the circuit (not fully implemented as of yet)
    :return: None
    """
    num_control_qubits = len(control_qubit_indicies)

    divisors = range(num_control_qubits - 1, -1, -1)   # starts from largest divisor to smallest
    divisors = [2**i for i in divisors]

    for iteration_num, theta in zip(range(1, 2**num_control_qubits + 1), params):
        if axis == 'z':
            circ.rz(theta, target_qubit_index)
        elif axis == 'y':
            circ.ry(theta, target_qubit_index)
        else:
            circ.rx(theta, target_qubit_index)

        for divisor in divisors:
            if iteration_num % divisor == 0:
                control_element = int((num_control_qubits - 1) - np.log2(divisor))
                circ.cx(control_qubit_indicies[control_element], target_qubit_index)
                break
    return


# Layer Implement ################################################

def legacy_conv4_layer_func(circ, params, active_qubits, barrier=True, kwargs={}):
    """
    This function takes a qiskit QuantumCircuit object and applies the
    4 qubit convolutional layer as described in the paper.

    This layer takes a group of four qubits, and considers each possible pair of qubits in the group. For each pair, it
    applies a two qubit parameterized operation defined by the Gell Mann matricies and the trained parameters.

    :param circ: qiskit QuantumCircuit object, the circuit to which the layer must be added
    :param params: list of np.arrays, containing the learnable parameters used in the convolutional layer (15 params)
    :param active_qubits: a list of ints, containing the indicies of the active qubits
    :param barrier: Bool, if true, plot a barrier to make visualization of circuit nicer
    :param kwargs: dict, contains args used in the layer implementation
    :return: augmented quantum circuit
    """
    conv_operators = generate_gell_mann(4)  # 2 qubit gell mann matricies
    u_conv = qi.Operator(get_conv_op(conv_operators, params))  # parameterized conv operator

    if "start_index" in kwargs:
        index = kwargs["start_index"]  # apply the convolutional operator on adjacent sets of 4 qubits starting here
    else:
        index = 0

    if "label" in kwargs:  # name of the layer for easy of
        label = kwargs["label"]
    else:
        label = 'lc4'

    while index + 3 < len(active_qubits):
        q_index = active_qubits[index]
        q_index_1 = active_qubits[index + 1]
        q_index_2 = active_qubits[index + 2]
        q_index_3 = active_qubits[index + 3]

        circ.unitary(u_conv, [q_index_2, q_index_3], label=label)
        circ.unitary(u_conv, [q_index, q_index_1], label=label)
        circ.unitary(u_conv, [q_index, q_index_3], label=label)
        circ.unitary(u_conv, [q_index, q_index_2], label=label)
        circ.unitary(u_conv, [q_index_1, q_index_3], label=label)
        circ.unitary(u_conv, [q_index_1, q_index_2], label=label)
        circ.barrier()

        if index == 0:
            index += 2
        else:
            index += 3

    if barrier:
        circ.barrier()

    return circ


def legacy_conv_layer_func(circ, params, active_qubits, barrier=True, kwargs={}):
    """
    This function takes a qiskit QuantumCircuit object and applies the generalized
    convolutional layer as described in the original paper on QCNNs.

    This layer takes a group of 3 qubits and performs the parameterized 3 qubit operation defined
    by the Gell Mann matrices and learnable parameters

    :param circ: qiskit QuantumCircuit object, the circuit to which the layer must be added
    :param params: list of np.arrays, containing the learnable parameters used in the convolutional layer (63 params)
    :param active_qubits: a list of ints, containing the indicies of the active qubits
    :param barrier: Bool, if true, plot a barrier to make visualization of circuit nicer
    :param kwargs: dict, contains args used in the layer implementation
    :return: augmented quantum circuit
    """
    conv_operators = generate_gell_mann(8)  # 3 qubit operators
    u_conv = qi.Operator(get_conv_op(conv_operators, params))

    if "start_index" in kwargs:
        index = kwargs["start_index"]
    else:
        index = 0

    if "label" in kwargs:
        label = kwargs["label"]
    else:
        label = 'lc'

    while index + 2 < len(active_qubits):
        q_index = active_qubits[index]
        q_index_1 = active_qubits[index + 1]
        q_index_2 = active_qubits[index + 2]

        circ.unitary(u_conv, [q_index, q_index_1, q_index_2], label=label)
        index += 3

    if barrier:
        circ.barrier()

    return circ


def legacy_pool_layer_func(circ, params, active_qubits, barrier=True, kwargs={}):
    """
    This function takes a qiskit QuantumCircuit object and applies the pooling layer
    as described in the original paper on QCNNs.

    This layer takes a group of 3 qubits, measures two of them and uses each measurement result to perform a
    controlled operation onto the remaining qubit, the measured qubits are then untouched for the remainder of
    the QCNN algorithm (effective reduction in number of required parameters)

    :param circ: qiskit QuantumCircuit object, the circuit to which the layer must be added
    :param params: list of np.arrays, containing the learnable parameters used in the pool layer (3 x 2 = 6 parameters)
    :param active_qubits: a list of ints, containing the indicies of the active qubits
    :param barrier: Bool, if true, plot a barrier to make visualization of circuit nicer
    :param kwargs: dict, contains args used in the layer implementation
    :return: augmented quantum circuit
    """
    pool_operators = generate_gell_mann(2)  # 1 qubit operators
    v1 = get_conv_op(pool_operators, params[:3])  # first 3 parameters for V1, last 3 for V2
    v2 = get_conv_op(pool_operators, params[3:])
    v1_pool = qi.Operator(controlled_pool(v1))
    v2_pool = qi.Operator(controlled_pool(v2))

    if "start_index" in kwargs:
        index = kwargs["start_index"]
    else:
        index = 0

    if "label" in kwargs:
        label = kwargs["label"]
    else:
        label = 'lp'

    while index + 2 < len(active_qubits):
        q_index = active_qubits[index]        # control index 1
        q_index_1 = active_qubits[index + 1]  # target index
        q_index_2 = active_qubits[index + 2]  # control index 2

        circ.h(q_index)
        circ.unitary(v1_pool, [q_index, q_index_1], label=label+'(1)')
        circ.h(q_index_2)
        circ.unitary(v2_pool, [q_index_2, q_index_1], label=label+'(2)')
        index += 3

    if barrier:
        circ.barrier()

    return circ


def legacy_fc_layer_fun(circ, params, active_qubits, barrier=True, kwargs={}):
    """
    This function takes a qiskit QuantumCircuit object and applies a fully connected layer which in this case
    is equivalent to a convolution layer over all of the active qubits. (convolve them all togehter)

    :param circ: qiskit QuantumCircuit object, the circuit to which the layer must be added
    :param params: list of np.arrays, containing the learnable parameters used in the fc layer (2^n - 1 params)
    :param active_qubits: a list of ints, containing the indicies of the active qubits
    :param barrier: Bool, if true, plot a barrier to make visualization of circuit nicer
    :param kwargs: dict, contains args used in the layer implementation
    :return: augmented quantum circuit
    """
    num_active_qubits = len(active_qubits)
    fully_connected_mats = generate_gell_mann(2**num_active_qubits)  # num active qubits operator
    fully_connected_operator = get_conv_op(fully_connected_mats, params)

    if "start_index" in kwargs:
        index = kwargs["start_index"]  # the fully connected layer acts on all active qubits so this isnt used
    else:
        index = 0

    if "label" in kwargs:
        label = kwargs["label"]
    else:
        label = 'fc'

    circ.unitary(fully_connected_operator, active_qubits, label=label)

    if barrier:
        circ.barrier()

    return circ


def custom_conv_layer_fun(circ, params, active_qubits, barrier=True, kwargs={}):
    """
    This function takes a qiskit QuantumCircuit object and applies a custom convolutional layer.

    This layer differs from the legacy version described in the original paper because it uses a
    different parameterization. In the legacy version we used the Gell Mann matricies, in this case
    we use parameterized uniformally controlled rotations. They have been shown to use much fewer parameters
    than the Gell Mann parameterization while still having the ability to express any arbitrary state. 

    :param circ: qiskit QuantumCircuit object, the circuit to which the layer must be added
    :param params: list of np.arrays, containing the parameters used in the custom conv layer (2^(n+2) - 5 params)
    :param active_qubits: a list of ints, containing the indicies of the active qubits
    :param barrier: Bool, if true, plot a barrier to make visualization of circuit nicer
    :param kwargs: dict, contains args used in the layer implementation
    :return: augmented quantum circuit
    """

    if "start_index" in kwargs:
        index = kwargs["start_index"]
    else:
        index = 0

    if "label" in kwargs:
        label = kwargs["label"]
    else:
        label = 'cc'

    if "group_size" in kwargs:
        group_size = kwargs["group_size"]
    else:
        group_size = 3

    while index + (group_size - 1) < len(active_qubits):
        param_pointer = 0
        lst_indicies = range(index, index + group_size)

        # z,y ascending loop
        for axis in ['z', 'y']:
            split_index = group_size - 1
            while split_index > 0:
                control_indicies = lst_indicies[:split_index]
                control_qubit_indicies = [active_qubits[i] for i in control_indicies]
                target_qubit_index = active_qubits[lst_indicies[split_index]]

                num_local_params = 2**(len(control_qubit_indicies))
                local_params = params[param_pointer:param_pointer + num_local_params]
                param_pointer += num_local_params

                generate_uniformly_controlled_rotation(circ, local_params, control_qubit_indicies,
                                                       target_qubit_index, axis=axis, label=label)

                split_index -= 1

            if axis == 'z':
                circ.rz(params[param_pointer], active_qubits[lst_indicies[split_index]])
            else:
                circ.ry(params[param_pointer], active_qubits[lst_indicies[split_index]])
            param_pointer += 1

        # descending loop
        for axis in ['y', 'z']:
            split_index = 1

            if axis == 'z':
                circ.rz(params[param_pointer], active_qubits[lst_indicies[split_index-1]])
                param_pointer += 1

            while split_index < group_size:
                control_indicies = lst_indicies[:split_index]
                control_qubit_indicies = [active_qubits[i] for i in control_indicies]
                target_qubit_index = active_qubits[lst_indicies[split_index]]

                num_local_params = 2**(len(control_qubit_indicies))
                local_params = params[param_pointer:param_pointer + num_local_params]
                param_pointer += num_local_params

                generate_uniformly_controlled_rotation(circ, local_params, control_qubit_indicies,
                                                       target_qubit_index, axis=axis, label=label)

                split_index += 1

        index += group_size

    if barrier:
        circ.barrier()

    return


# Layer class ######################################################################
class Layer:
    """
    A class to wrap up the required fields of a layer. This layer object will then be used
    in the QCNN class to build a quantum machine learning model.
    """

    def __init__(self, name, func, param_shape):
        self.name = name  # a str which labels the layer
        self.func = func  # a callable function which acts on a quantum circuit to apply the layer
        self.shape_params = param_shape  # the size of parameters required for this layer, used for initialization
        return                           # and learning of parameters

    def apply_layer(self, circ, params, active_qubits, kwargs={}):
        inst = self.func(circ, params, active_qubits, kwargs=kwargs)  # each gate has its own unique label
        return inst


# Functions to get customizable layers  ###############################################
def get_legacy_fc_layer(num_active_qubits):
    """
    Since the fully connected layer has a variable number of parameters based on the
    number of remaining active qubits once the model has been generated. For this reason we
    need a method that allows user to get a fully connected layer based on the number of active qubits

    :param num_active_qubits: int, the number of active qubits you will have left at the end of the model
    :return: a Layer instance, the fully connected layer
    """
    layer_name = "legacy_fc_layer_n{}".format(num_active_qubits)
    fc_layer = Layer(layer_name, legacy_fc_layer_fun, (2**num_active_qubits - 1,))
    return fc_layer


def get_custom_conv_layer(group_size):
    """
    This custom convolutional layer implementation is general enough to allow
    users to choose their own group size and thus requires its own get method.

    :param group_size: int, the number of qubits grouped together in each convolution for a single layer
    :return: a Layer instance, the custom convolutional layer
    """
    num_params = 0
    for q in range(group_size):
        num_params += 2 ** q
    num_params = (num_params * 2 - 1) * 2 + 1
    # ^^ this is determined from the paper: https://arxiv.org/pdf/quant-ph/0407010.pdf

    layer_name = "custom_conv_layer_n{}".format(group_size)
    cc_layer = Layer(layer_name, custom_conv_layer_fun, (num_params,))
    return cc_layer


# Base Legacy Layers #################################################################################################
legacy_conv4_layer = Layer("legacy_conv4_layer", legacy_conv4_layer_func, (15,))
legacy_conv_layer = Layer("legacy_conv_layer", legacy_conv_layer_func, (63,))
legacy_pool_layer = Layer("legacy_pool_layer", legacy_pool_layer_func, (6,))

# These (^^) layers are explicitly initialized here so that they can be easily imported and used in the qcnn module


def main():
    return


if __name__ == "__main__":
    main()
