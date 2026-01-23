import torch
import numpy as np


class HamiltonianParams:
    c_vector: torch.Tensor
    c_direction: np.ndarray

    def __init__(self, cv, cd):
        self.c_vector = cv
        self.c_direction = cd


single_dual_params = HamiltonianParams(
    cv=torch.tensor([[-0.5, -0.15, 1.1, 0.33, 0.75, 0.3, 0.3, -0.8, -0.75]]),
    cd=np.array(
        [
            [
                -0.08657043,
                0.10342339,
                0.22536035,
                0.39073649,
                0.01098348,
                -0.44329002,
                0.1645999,
                -0.08833236,
                -0.21927355,
            ]
        ]
    ),
)
h = 1.5
J = 1
ising_hamiltonian_2 = HamiltonianParams(
    cv=None, cd=5 * np.array([[0, 0, h - J, -J - h, 0, 0, 0, 0, -h, J - h, 0, 0]])
)
