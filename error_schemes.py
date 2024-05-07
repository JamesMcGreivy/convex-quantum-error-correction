"""
File: error_schemes.py
Author: James McGreivy
Date: 04/29/2024
Description: This file is used to instantiate several quantum error channels, such as the
generalized amplitude damping channel
"""
import torch
import numpy as np
import itertools

alphabet = ["A", "B", "C", "D", "E", "F" , "G", "H" , "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",]

# Constructs an error process matrix, given a set of 2x2 Krauss operators describing a single-qubit error channel.
def error_to_X_E(K, q):
    E = torch.zeros(size=q*[len(K)] + 2*[2**q], dtype=torch.complex64, device=K[0].device)

    for i in itertools.product(*[range(0,len(K)) for _ in range(0, q)]):
        K_tot = K[i[0]]
        for j in range(1, q):
            K_tot = torch.kron(K_tot, K[i[j]])
        E[i] = K_tot

    X = torch.einsum("".join([alphabet[i] for i in range(q)]) + "lm," + "".join([alphabet[i] for i in range(q)]) + "gs->lmgs", E, E.conj())
    return X

# Amplitude Damping (AD) Krauss Operators and Process Matrix
def AD(q, l, device = "cpu"):
    K_E_1 = torch.tensor([[1,0],[0,np.sqrt(1 - l)]], device=device)
    K_E_2 = torch.tensor([[0,np.sqrt(l)],[0, 0]], device=device)
    
    K_E = [K_E_1, K_E_2]
    X_E = error_to_X_E(K_E, q)
    return X_E

# Depolarizing channel Krauss Operators and Process Matrix
def depolarizing(q, l, device="cpu"):

    # Pauli Matrices
    I = torch.tensor([[1,0],[0,1]], dtype=torch.complex64, device=device)
    X = torch.tensor([[0,1],[1,0]], dtype=torch.complex64, device=device)
    Y = torch.tensor([[0,-1j],[1j,0]], dtype=torch.complex64, device=device)
    Z = torch.tensor([[1,0],[0,-1]], dtype=torch.complex64, device=device)
    
    K_E_1 = np.sqrt(1 - 3*l/4) * I
    K_E_2 = np.sqrt(l / 4) * X
    K_E_3 = np.sqrt(l / 4) * Y
    K_E_4 = np.sqrt(l / 4) * Z
    
    K_E = [K_E_1, K_E_2, K_E_3, K_E_4]
    X_E = error_to_X_E(K_E, q)
    return X_E

def dephasing(q, l, device="cpu"):

    # Pauli Matrices
    I = torch.tensor([[1,0],[0,1]], dtype=torch.complex64, device=device)
    X = torch.tensor([[0,1],[1,0]], dtype=torch.complex64, device=device)
    Y = torch.tensor([[0,-1j],[1j,0]], dtype=torch.complex64, device=device)
    Z = torch.tensor([[1,0],[0,-1]], dtype=torch.complex64, device=device)

    K_E_1 = np.sqrt(1 - l) * I
    K_E_2 = np.sqrt(l) * torch.tensor([[1,0],[0,0]], dtype = torch.complex64)
    K_E_3 = np.sqrt(l) * torch.tensor([[0,0],[0,1]], dtype = torch.complex64)

    K_E = [K_E_1, K_E_2, K_E_3]
    X_E = error_to_X_E(K_E, q)
    return X_E