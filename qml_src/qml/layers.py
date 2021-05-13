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
            # print('iteration_num: {}, divisor: {}'.format(iteration_num, divisor))
            if iteration_num % divisor == 0:
                # print('----------')
                control_element = int((num_control_qubits - 1) - np.log2(divisor))
                circ.cx(control_qubit_indicies[control_element], target_qubit_index)
                break
    return


# Layer Implement ################################################

def legacy_conv4_layer_func(circ, params, active_qubits, barrier=True, kwargs={}):
    """
    15 params, (one per gell mann mat)
    :param circ:
    :param params:
    :param active_qubits:
    :param barrier
    :param kwargs
    :return:
    """
    conv_operators = generate_gell_mann(4)  # 2 qubits operators
    u_conv = qi.Operator(get_conv_op(conv_operators, params))  # unitary operators

    if "start_index" in kwargs:
        index = kwargs["start_index"]
    else:
        index = 0

    if "label" in kwargs:
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
    :param circ:
    :param params: 63 parameters
    :param active_qubits:
    :param barrier:
    :param kwargs:
    :return:
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
    :param circ:
    :param params: 3 x 2 parameters (6)
    :param active_qubits:
    :param barrier:
    :param kwargs:
    :return:
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
    :param circ:
    :param params:  num_active_qubits ^2 - 1  (same as the number of gm mats)
    :param active_qubits:
    :param barrier:
    :param kwargs:
    :return:
    """
    num_active_qubits = len(active_qubits)
    fully_connected_mats = generate_gell_mann(2**num_active_qubits)
    fully_connected_operator = get_conv_op(fully_connected_mats, params)

    if "start_index" in kwargs:
        index = kwargs["start_index"]
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

    def __init__(self, name, func, param_shape):
        self.name = name
        self.func = func
        self.shape_params = param_shape
        return

    def apply_layer(self, circ, params, active_qubits, kwargs={}):
        inst = self.func(circ, params, active_qubits, kwargs=kwargs)  # each gate has its own unique label
        return inst


# Temp functions to get customizable layers, im not sure how to make this cleaner right now ##########################
def get_legacy_fc_layer(num_active_qubits):
    layer_name = "legacy_fc_layer_n{}".format(num_active_qubits)
    fc_layer = Layer(layer_name, legacy_fc_layer_fun, (2**num_active_qubits - 1,))
    return fc_layer


def get_custom_conv_layer(group_size):
    num_params = 0
    for q in range(group_size):
        num_params += 2 ** q
    num_params = (num_params * 2 - 1) * 2 + 1

    layer_name = "custom_conv_layer_n{}".format(group_size)
    cc_layer = Layer(layer_name, custom_conv_layer_fun, (num_params,))
    return cc_layer


# Base Legacy Layers #################################################################################################
legacy_conv4_layer = Layer("legacy_conv4_layer", legacy_conv4_layer_func, (15,))
legacy_conv_layer = Layer("legacy_conv_layer", legacy_conv_layer_func, (63,))
legacy_pool_layer = Layer("legacy_pool_layer", legacy_pool_layer_func, (6,))


def main():
    return


if __name__ == "__main__":
    main()
