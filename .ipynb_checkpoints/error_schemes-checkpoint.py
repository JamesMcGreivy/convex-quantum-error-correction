"""
File: error_schemes.py
Author: James McGreivy
Date: 04/29/2024
Description: This file is used to instantiate several quantum error channels, such as the
generalized amplitude damping channel
"""
import torch
import numpy as np
import opt_einsum
import itertools

# Constructs an error process matrix, given a set of 2x2 Krauss operators describing a single-qubit error channel.
def error_to_X_E(K, q):
    E = torch.zeros(size=q*[len(K)] + 2*[2**q], dtype = torch.complex128, device = K[0].device)

    for i in itertools.product(*[range(0,len(K)) for _ in range(0, q)]):
        K_tot = K[i[0]]
        for j in range(1, q):
            K_tot = torch.kron(K_tot, K[i[j]])
        E[i] = K_tot

    X = opt_einsum.contract("".join([str(i) for i in range(q)]) + "lm," + "".join([str(i) for i in range(q)]) + "gs->lmgs", E, E.conj())
    return X

# Generalized Amplitude Damping (GAD) Krauss Operators and Process Matrix
def GAD(q, l):
    g = l
    N = l
    
    K_E_1 = torch.tensor([[np.sqrt(1 - N), 0],[0, np.sqrt(1 - N) * np.sqrt(1 - g)]])
    K_E_2 = torch.tensor([[0,np.sqrt(g*(1-N))],[0,0]])
    K_E_3 = torch.tensor([[np.sqrt(N)*np.sqrt(1-g), 0],[0,np.sqrt(N)]])
    K_E_4 = torch.tensor([[0,0],[np.sqrt(g * N), 0]])
    
    K_E = [K_E_1, K_E_2, K_E_3, K_E_4]
    X_E = error_to_X_E(K_E, q)
    return X_E