import numpy as np
from qiskit import QuantumCircuit
from qiskit import quantum_info as qi
import matplotlib.pyplot as plt
import scipy.linalg as la

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
        print(parm * mat)
        final += parm * mat

    return la.expm(complex(0, -1) * final)


def run_qcnn(num_qubits, parameters, wf):
    circ = QuantumCircuit(num_qubits)
    active_qubits = range(num_qubits)

    # 1st convolution layer:
    group_size = 4
    apply_on_index = 3
    conv_params = parameters[0]
    conv_operators = generate_gell_mann(4)  # 2 qubit operators

    for index, qubit_index in enumerate(active_qubits):
        if ((index+1) % apply_on_index == 0) and (index+1 < len(active_qubits)):
            U_conv = qi.Operator(get_conv_op(conv_operators, conv_params))
            circ.unitary(U_conv, [index + 1, index + 2], label='U_1')
            circ.unitary(U_conv, [index + 1, index + 3], label='U_1')
            circ.unitary(U_conv, [index + 0, index + 2], label='U_1')
            circ.unitary(U_conv, [index + 0, index + 3], label='U_1')
            circ.unitary(U_conv, [index + 0, index + 1], label='U_1')
            circ.unitary(U_conv, [index + 2, index + 3], label='U_1')
            # circ.barrier()

    # 2nd - 4th convolution layers:

    wave_func = qi.Statevector(wf)
    wave_func.evolve(circ)
    return wave_func


def get_parms(num_params):
    params = np.random.uniform(0, np.pi*2, num_params)
    return params


def main():
    gm_mats = generate_gell_mann(4)
    params = [get_parms(len(gm_mats))]
    print(params)
    wf = np.zeros(15)
    wf[0] = 1

    circ = run_qcnn(15, params, wf)
    # circ.draw()
    return circ


if __name__ == "__main__":
    main()
