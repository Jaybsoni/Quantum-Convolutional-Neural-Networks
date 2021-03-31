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
        self.inp = num_qubits
        self.active_qubits = range(num_qubits)
        self.structure = []

    def add_layer(self, layer, ):
        return

    def get_params(self):
        return

    @staticmethod
    def embedding(wf):
        q_state = qi.Statevector(wf)
        return q_state

    @staticmethod
    def get_operator(circ):
        operator = qi.Operator(circ)
        return operator

    @staticmethod
    def update_active_qubits(active_qubits, group_len, target):
        assert len(active_qubits) % group_len == 0
        num_groups = int(len(active_qubits) / group_len)
        update_qubits = []

        for i in range(num_groups):
            index = i * group_len + target
            update_qubits.append(active_qubits[index])

        return update_qubits


def main():
    return


if __name__ == "__main__":
    main()
