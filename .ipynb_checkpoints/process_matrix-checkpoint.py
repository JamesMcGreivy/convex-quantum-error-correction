"""
File: process_matrix.py
Author: James McGreivy
Date: 04/29/2024
Description: Containts a pytorch module to wrap the process matrices (X_C or X_R) as well as utility functions for
constructing the process matrices. 
"""
import numpy as np
import torch
import opt_einsum
import itertools
import utils

# Constructs X_E, the error process matrix, given a set of 2x2 Krauss operators describing a single-qubit error channel.
def krauss_to_X_E(K, q_c):
    E = torch.zeros(size=q_c*[len(K)] + 2*[2**q_c], dtype = torch.complex128, device = K[0].device)

    for i in itertools.product(*[range(0,len(K)) for _ in range(0, q_c)]):
        K_tot = K[i[0]]
        for j in range(1, q_c):
            K_tot = torch.kron(K_tot, K[i[j]])
        E[i] = K_tot

    X_E = opt_einsum.contract("".join([str(i) for i in range(q_c)]) + "lm," + "".join([str(i) for i in range(q_c)]) + "gs->lmgs", E, E.conj())
    return X_E

# Initializes an arbitrary process matrix according to the required constraints.
def initialize_process_matrix(n_1, n_2, device = "cpu"):
    X = torch.normal(mean=0, std=1/(n_1**2 * n_2**2), size=(n_2, n_1, n_2, n_1), dtype=torch.complex128, device=device)
    X = utils.make_PSD(X)
    X = utils.make_sum_to_identity(X)
    return X

class ProcessMatrix(torch.nn.Module):
    def __init__(self, q_1, q_2, device = "cpu"):
        super().__init__()

        self.n_1 = 2**q_1
        self.n_2 = 2**q_2
        
        self.X = torch.nn.Parameter(initialize_process_matrix(self.n_1, self.n_2, device))

    # Returns the process matrices for the coding, error, and recovery channel
    def forward(self):
        return self.X



