# qcnn
import numpy as np
from qiskit import QuantumCircuit
from qiskit import quantum_info as qi


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


def update_active_qubits(active_qubits, group_len, target):
    assert len(active_qubits) % group_len == 0
    num_groups = int(len(active_qubits) / group_len)
    update_qubits = []

    for i in range(num_groups):
        index = i*group_len + target
        update_qubits.append(active_qubits[index])

    return update_qubits


class QcnnStruct:

    def __init__(self, num_qubits):
        self.inp = num_qubits
        self.active_qubits = range(num_qubits)

    def conv_layer(self, circ, params):
        return

    def conv_spec4_layer(self, circ, params):
        return

    def pooling_layer(self, circ, group, target_ind, params):
        assert self.inp % group == 0  # ensure that we can pool properly
        return

    @staticmethod
    def embedding(wf):
        q_state = qi.Statevector(wf)
        return q_state

    @staticmethod
    def get_operator(circ):
        operator = qi.Operator(circ)
        return operator


def main():
    # lst_gm_mat = generate_gell_mann(3)
    # for i in lst_gm_mat:
    #     print(i)
    #     print('\n')

    # active_qs = [0, 1, 2, 3, 4, 5, 6, 7]
    # group = 4
    # target = 2
    # print(active_qs)
    # print(update_active_qubits(active_qs, group, target))
    return


if __name__ == "__main__":
    main()
