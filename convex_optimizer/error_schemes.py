"""
File: error_schemes.py
Author: James McGreivy
Date: 04/29/2024
Description: This file is used to instantiate several quantum error channels, such as the
generalized amplitude damping channel
"""
import torch
import numpy as np
import utils
import itertools

alphabet = ["A", "B", "C", "D", "E", "F" , "G", "H" , "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",]

#~~ Qubit Error Channels ~~#

# Constructs an error process matrix, given a set of 2x2 Krauss operators describing a single-qubit error channel.
def multi_krauss_error(K, q):
    E = []
    for i in itertools.product(*[range(0,len(K)) for _ in range(0, q)]):
        K_tot = K[i[0]]
        for j in range(1, q):
            K_tot = torch.kron(K_tot, K[i[j]])
        E.append(K_tot)
    return E

# Amplitude Damping (AD) Krauss Operators and Process Matrix
def AD_krauss(n, l, device="cpu"):
    q = int(np.log2(n))
    K_E_1 = torch.tensor([[1,0],[0,np.sqrt(1 - l)]], dtype=torch.complex64, device=device)
    K_E_2 = torch.tensor([[0,np.sqrt(l)],[0, 0]], dtype=torch.complex64, device=device)
    E = multi_krauss_error([K_E_1, K_E_2], q)
    return E
    
def AD(n, l, device="cpu"):
    return utils.krauss_to_X(AD_krauss(n, l, device))

# Depolarizing channel Krauss Operators and Process Matrix
def depolarizing_krauss(n, l, device="cpu"):
    q = int(np.log2(n))
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
    E = multi_krauss_error(K_E, q)
    return E

def depolarizing(n, l, device="cpu"):
    return utils.krauss_to_X(depolarizing_krauss(n, l, device=device))

def dephasing_krauss(n, l, device="cpu"):
    q = int(np.log2(n))
    # Pauli Matrices
    I = torch.tensor([[1,0],[0,1]], dtype=torch.complex64, device=device)
    X = torch.tensor([[0,1],[1,0]], dtype=torch.complex64, device=device)
    Y = torch.tensor([[0,-1j],[1j,0]], dtype=torch.complex64, device=device)
    Z = torch.tensor([[1,0],[0,-1]], dtype=torch.complex64, device=device)

    K_E_1 = np.sqrt(1 - l) * I
    K_E_2 = np.sqrt(l) * torch.tensor([[1,0],[0,0]], dtype = torch.complex64)
    K_E_3 = np.sqrt(l) * torch.tensor([[0,0],[0,1]], dtype = torch.complex64)

    K_E = [K_E_1, K_E_2, K_E_3]
    return multi_krauss_error(K_E, q)

def dephasing(n, l, device="cpu"):
    return utils.krauss_to_X(dephasing_krauss(n, l, device))

#~~ Qutrit Error Channels ~~#

def depolarizing_qutrit_krauss(n, l, device="cpu"):
    qt = int(np.emath.logn(3, n))
    I = torch.eye(3, dtype=torch.complex64, device=device)
    Y = torch.tensor([[0,1,0],[0,0,1],[1,0,0]], dtype=torch.complex64, device=device)
    w = np.exp(2j/3 * np.pi)
    Z = torch.tensor([[1,0,0],[0,w,0],[0,0,w*w]], dtype=torch.complex64, device=device)

    E0 = np.sqrt(1-l) * I 
    E1 = np.sqrt(l/8) * Y
    E2 = np.sqrt(l/8) * Z 
    E3 = np.sqrt(l/8) * Y @ Y 
    E4 = np.sqrt(l/8) * Y @ Z 
    E5 = np.sqrt(l/8) * Y @ Y @ Z 
    E6 = np.sqrt(l/8) * Y @ Z @ Z 
    E7 = np.sqrt(l/8) * Y @ Y @ Z @ Z
    E8 = np.sqrt(l/8) * Z @ Z

    K = [E0, E1, E2, E3, E4, E5, E6, E7, E8]
    return multi_krauss_error(K, qt)

def depolarizing_qutrit(n, l, device="cpu"):
    return utils.krauss_to_X(depolarizing_qutrit_krauss(n, l, device))

def AD_qutrit_krauss(n, l, device="cpu"):
    qt = int(np.emath.logn(3, n))
    K_1 = torch.tensor([[1,0,0],[0,np.sqrt(1-l),0],[0,0,np.sqrt(1-l)]], dtype=torch.complex64, device=device)
    K_2 = torch.tensor([[0,np.sqrt(l),0],[0,0,0],[0,0,0]], dtype=torch.complex64, device=device)
    K_3 = torch.tensor([[0,0,np.sqrt(l)],[0,0,0],[0,0,0]], dtype=torch.complex64, device=device)
    K = [K_1, K_2, K_3]
    return multi_krauss_error(K, qt)

def AD_qutrit(n, l, device="cpu"):
    return utils.krauss_to_X(AD_qutrit_krauss(n, l, device))

def dephasing_qutrit_krauss(n, l, device="cpu"):
    qt = int(np.emath.logn(3, n))
    K_1 = np.sqrt(1-l) * torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=torch.complex64, device=device)
    K_2 = np.sqrt(l) * torch.tensor([[1,0,0],[0,0,0],[0,0,0]], dtype=torch.complex64, device=device)
    K_3 = np.sqrt(l) * torch.tensor([[0,0,0],[0,1,0],[0,0,0]], dtype=torch.complex64, device=device)
    K_4 = np.sqrt(l) * torch.tensor([[0,0,0],[0,0,0],[0,0,1]], dtype=torch.complex64, device=device)
    K = [K_1, K_2, K_3, K_4]
    return multi_krauss_error(K, qt)

def dephasing_qutrit(n, l, device="cpu"):
    return utils.krauss_to_X(dephasing_qutrit_krauss(n, l, device))