import numpy as np
import scipy.linalg as la
from qiskit import quantum_info as qi


# Helper Functions ################################################
def b_mat(i, j, n):
    basis_matrix = np.zeros((n, n), dtype=np.float32)
    basis_matrix[i, j] = 1.0

    return basis_matrix


def generate_gell_mann(order):
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
    final = np.zeros(mats[0].shape, dtype=np.complex128)
    for mat, parm in zip(mats, parms):
        final += parm * mat

    return la.expm(complex(0, -1) * final)


def controlled_pool(mat):
    i_hat = np.array([[1.0, 0.0],
                      [0.0, 0.0]])
    j_hat = np.array([[0.0, 0.0],
                      [0.0, 1.0]])
    identity = i_hat + j_hat

    return np.kron(i_hat, identity) + np.kron(j_hat, mat)


# Layer Implement ################################################

def legacy_conv4_layer_func(circ, params, active_qubits, barrier=True, kwargs={}, label='lc4'):
    """
    15 params, (one per gell mann mat)
    :param circ:
    :param params:
    :param active_qubits:
    :param barrier
    :param kwargs
    :param label
    :return:
    """
    conv_operators = generate_gell_mann(4)  # 2 qubits operators
    u_conv = qi.Operator(get_conv_op(conv_operators, params))  # unitary operators

    if "start_index" in kwargs:
        index = kwargs["start_index"]
    else:
        index = 0

    while index + 3 < len(active_qubits):
        q_index = active_qubits[index]
        q_index_1 = active_qubits[index + 1]
        q_index_2 = active_qubits[index + 2]
        q_index_3 = active_qubits[index + 3]

        circ.unitary(u_conv, [q_index_1, q_index_2], label=label)
        circ.unitary(u_conv, [q_index_1, q_index_3], label=label)
        circ.unitary(u_conv, [q_index,   q_index_2], label=label)
        circ.unitary(u_conv, [q_index,   q_index_3], label=label)
        circ.unitary(u_conv, [q_index,   q_index_1], label=label)
        circ.unitary(u_conv, [q_index_2, q_index_3], label=label)

        if index == 0:
            index += 2
        else:
            index += 3

    if barrier:
        circ.barrier()

    return circ


def legacy_conv_layer_func(circ, params, active_qubits, barrier=True, kwargs={}, label='lc'):
    """
    :param circ:
    :param params: 63 parameters
    :param active_qubits:
    :param barrier:
    :param kwargs:
    :param label:
    :return:
    """
    conv_operators = generate_gell_mann(8)  # 3 qubit operators
    u_conv = qi.Operator(get_conv_op(conv_operators, params))

    if "start_index" in kwargs:
        index = kwargs["start_index"]
    else:
        index = 0

    while index + 2 < len(active_qubits):
        q_index = active_qubits[index]
        q_index_1 = active_qubits[index + 1]
        q_index_2 = active_qubits[index + 2]

        circ.unitary(u_conv, [q_index, q_index_1, q_index_2], label=label)
        index += 3

    if barrier:
        circ.barrier()

    return circ


def legacy_pool_layer_func(circ, params, active_qubits, barrier=True, kwargs={}, label='lp'):
    """
    :param circ:
    :param params: 3 x 2 parameters
    :param active_qubits:
    :param barrier:
    :param kwargs:
    :param label:
    :return:
    """
    pool_operators = generate_gell_mann(2)  # 1 qubit operators
    v1 = get_conv_op(pool_operators, params[0])
    v2 = get_conv_op(pool_operators, params[1])
    v1_pool = qi.Operator(controlled_pool(v1))
    v2_pool = qi.Operator(controlled_pool(v2))

    if "start_index" in kwargs:
        index = kwargs["start_index"]
    else:
        index = 0

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


# Layer class ######################################################################
class Layer:

    def __init__(self, name, func):
        self.name = name
        self.func = func
        return

    def apply_layer(self, circ, params, active_qubits, kwargs={}, label=None):
        if label is not None:
            new_circ = self.func(circ, params, active_qubits, kwargs=kwargs, label=label)
        else:
            new_circ = self.func(circ, params, active_qubits, kwargs=kwargs)  # each gate has its own unique label
        return new_circ


legacy_conv4_layer = Layer("legacy_conv4_layer", legacy_conv4_layer_func)
legacy_conv_layer = Layer("legacy_conv_layer", legacy_conv_layer_func)
legacy_pool_layer = Layer("legacy_pool_layer", legacy_pool_layer_func)


def main():
    return


if __name__ == "__main__":
    main()
