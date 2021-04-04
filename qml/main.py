import numpy as np
from qiskit import QuantumCircuit
from qiskit import quantum_info as qi
import matplotlib.pyplot as plt
import scipy.linalg as la
from pprint import pprint
"""
In order to better understand the structure requred for the classes for implementing general QCNNs, I am going to code 
out the particular example as described in the paper and then use it as a reference when designing the abstract classes. 

The mess of sphaghetti code you see below is my quick and dirty attempt at this: 
"""


def b_mat(i, j, n):
    basis_matrix = np.zeros((n, n), dtype=np.complex128)
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


def update_active_qubits(active_qubits, group_len, target):
    assert len(active_qubits) % group_len == 0
    num_groups = int(len(active_qubits) / group_len)
    update_qubits = []

    for i in range(num_groups):
        index = i * group_len + target
        update_qubits.append(active_qubits[index])

    return update_qubits


def get_conv_op(mats, parms):
    final = np.zeros(mats[0].shape, dtype=np.complex128)
    for mat, parm in zip(mats, parms):
        # print(parm * mat)
        final += parm * mat

    return la.expm(complex(0, -1) * final)


def controlled_pool(mat):
    i_hat = np.array([[1.0, 0.0],
                      [0.0, 0.0]])
    j_hat = np.array([[0.0, 0.0],
                      [0.0, 1.0]])
    identity = i_hat + j_hat

    return np.kron(i_hat, identity) + np.kron(j_hat, mat)


def run_qcnn(num_qubits, parameters, wf):
    circ = QuantumCircuit(num_qubits)
    active_qubits = range(num_qubits)

    # Embedding layer:
    # wave_func = qi.Statevector(wf)
    # circ.initialize(wave_func.data, list(active_qubits))
    # circ.barrier()

    # 1st convolution layer:
    group_size = 4
    apply_on_index = 3
    conv_params = parameters[0]
    conv_operators = generate_gell_mann(4)  # 2 qubit operators

    index = 0
    while index + 3 < len(active_qubits):
        U_conv = qi.Operator(get_conv_op(conv_operators, conv_params))
        circ.unitary(U_conv, [index + 1, index + 2], label='U_1')
        circ.unitary(U_conv, [index + 1, index + 3], label='U_1')
        circ.unitary(U_conv, [index + 0, index + 2], label='U_1')
        circ.unitary(U_conv, [index + 0, index + 3], label='U_1')
        circ.unitary(U_conv, [index + 0, index + 1], label='U_1')
        circ.unitary(U_conv, [index + 2, index + 3], label='U_1')
        circ.barrier()

        if index == 0:
            index += 2
        else:
            index += 3

    # 2nd - 4th convolution layers:
    reg_conv_operators = generate_gell_mann(8)  # 3 qubit operators
    reg_conv_parameters = [parameters[1], parameters[2], parameters[3]]
    num_act_on_qubits = 3

    for start_index in range(3):
        comb_conv_opt = get_conv_op(reg_conv_operators, reg_conv_parameters[start_index])
        u_conv = qi.Operator(comb_conv_opt)
        working_index = start_index
        while working_index + 2 < len(active_qubits):
            # print('start_index: {}, working_index: {}'.format(start_index, working_index))
            # print(working_index)
            circ.unitary(u_conv, [working_index, working_index + 1, working_index + 2],
                         label='U_{}'.format(start_index+2))
            working_index += 3
        circ.barrier()

    # 1st Pooling layer:
    pool_operators = generate_gell_mann(2)  # 1 qubit operators
    pooling_params = [parameters[4], parameters[5]]
    working_index = 0
    v1 = get_conv_op(pool_operators, pooling_params[0])
    v2 = get_conv_op(pool_operators, pooling_params[1])
    v1_pool = qi.Operator(controlled_pool(v1))
    v2_pool = qi.Operator(controlled_pool(v2))

    while working_index + 2 < len(active_qubits):
        circ.h(working_index)
        circ.unitary(v1_pool, [working_index, working_index + 1], label='V1')
        circ.h(working_index + 2)
        circ.unitary(v2_pool, [working_index + 2, working_index + 1], label='V2')
        working_index += 3

    circ = circ.reverse_bits()
    # Embedding input wf
    # wave_func = qi.Statevector(wf)
    # print(wave_func.data)
    # wave_func = wave_func.evolve(circ)

    return circ


def get_parms(num_params):
    params = np.random.uniform(0, np.pi*2, num_params)
    return params


def compare_unitary(mat):
    new_mat = np.matrix(mat)
    c_t_mat = new_mat.H

    resultant_mat = c_t_mat @ new_mat
    # pprint('{}'.format(resultant_mat))
    return


def main():
    gm_mats = generate_gell_mann(4)
    gm_mats8 = generate_gell_mann(8)
    gm_mats2 = generate_gell_mann(2)
    num_2mats = len(gm_mats2)
    num_8mats = len(gm_mats8)
    params = [get_parms(len(gm_mats))]
    params2 = get_parms(num_8mats)
    params3 = get_parms(num_8mats)
    params4 = get_parms(num_8mats)
    params5 = get_parms(num_2mats)
    params6 = get_parms(num_2mats)
    params.append(params2)
    params.append(params3)
    params.append(params4)
    params.append(params5)
    params.append(params6)

    # print(params)
    wf = np.zeros(2**15)
    wf[0] = 1

    new_wf = run_qcnn(15, params, wf)
    # circ.draw()

    return


if __name__ == "__main__":
    main()
