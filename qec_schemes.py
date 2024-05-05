
"""
File: qec_schemes.py
Author: James McGreivy
Date: 04/29/2024
Description: This file is used to instantiate several quantum error correcting (qec) schemes, such as the
3 qubit repetition code as well as the convex optimizer code.
"""
import torch
import opt_einsum
import numpy as np
import cvxpy_caller
import process_matrix
import utils
from utils import Dag

def convex_optimizer_qec(q_s, q_c, X_E):
    n_s = 2**q_s
    n_c = 2**q_c
    
    fids = []
    X_Cs = []
    X_Ds = []
    for _ in range(5):
    
        X_C = process_matrix.initialize_process_matrix(n_s, n_c)
        X_D = process_matrix.initialize_process_matrix(n_c, n_s)

        cutoff = 1e-3
        delta = np.inf

        prev_f_avg, i = 0, 0
        while (delta > cutoff):
            W_C = (1/n_s**2) * opt_einsum.contract("iljg,lmgs->misj", X_D, X_E)
            _, X_C = cvxpy_caller.maximize_fidelity(X_C, W_C)

            W_D = (1/n_s**2) * opt_einsum.contract("misj,lmgs->iljg", X_C, X_E)
            f_avg, X_D = cvxpy_caller.maximize_fidelity(X_D, W_D)

            delta = (f_avg - prev_f_avg)

            prev_f_avg = f_avg
            i += 0
            
        fids.append(f_avg)
        X_Cs.append(X_C)
        X_Ds.append(X_D)
    
    best = fids.index(max(fids))
    
    X_C = X_Cs[best]
    X_D = X_Ds[best]
    C = utils.X_to_krauss(X_C)
    D = utils.X_to_krauss(X_D)

    R = []
    for c in range(len(C)):
        for d in range(len(D)):
            R.append(C[c] @ D[d])
    X_R = utils.krauss_to_X(R, 3, 3)
    R = utils.X_to_krauss(X_R)

    return {"C" : C, "R" : R, "D" : D, "X_C" : X_C, "X_R" : X_R, "X_D" : X_D}

# Pauli Matrices
I = torch.tensor([[1,0],[0,1]], dtype = torch.complex128)
X = torch.tensor([[0,1],[1,0]], dtype = torch.complex128)
Z = torch.tensor([[1,0],[0,-1]], dtype = torch.complex128)

# States
Psi0 = torch.tensor([[1,0],], dtype = torch.complex128)
Psi1 = torch.tensor([[0,1],], dtype = torch.complex128)

def three_qubit_bitflip_qec():
    # ~~ Recovery krauss operators ~~
    U0 = torch.kron(I, torch.kron(I, I))
    P0 = torch.kron(Dag(Psi0)@Psi0, torch.kron(Dag(Psi0)@Psi0, Dag(Psi0)@Psi0)) + torch.kron(Dag(Psi1)@Psi1, torch.kron(Dag(Psi1)@Psi1, Dag(Psi1)@Psi1))
    R0 = U0 @ P0

    U1 = torch.kron(X, torch.kron(I, I))
    P1 = torch.kron(Dag(Psi1)@Psi1, torch.kron(Dag(Psi0)@Psi0, Dag(Psi0)@Psi0)) + torch.kron(Dag(Psi0)@Psi0, torch.kron(Dag(Psi1)@Psi1, Dag(Psi1)@Psi1))
    R1 = U1 @ P1

    U2 = torch.kron(I, torch.kron(X, I))
    P2 = torch.kron(Dag(Psi0)@Psi0, torch.kron(Dag(Psi1)@Psi1, Dag(Psi0)@Psi0)) + torch.kron(Dag(Psi1)@Psi1, torch.kron(Dag(Psi0)@Psi0, Dag(Psi1)@Psi1))
    R2 = U2 @ P2

    U3 = torch.kron(I, torch.kron(I, X))
    P3 = torch.kron(Dag(Psi0)@Psi0, torch.kron(Dag(Psi0)@Psi0, Dag(Psi1)@Psi1)) + torch.kron(Dag(Psi1)@Psi1, torch.kron(Dag(Psi1)@Psi1, Dag(Psi0)@Psi0))
    R3 = U3 @ P3

    R = [R0, R1, R2, R3]
    X_R = utils.krauss_to_X(R, 3, 3)

    # ~~ Encoding krauss operators ~~
    C0 = Dag(torch.kron(Psi0, torch.kron(Psi0, Psi0))) @ Psi0 + Dag(torch.kron(Psi1, torch.kron(Psi1, Psi1))) @ Psi1
    C = [C0]
    X_C = utils.krauss_to_X(C, 1, 3)

    D = []
    for c in range(len(C)):
        for r in range(len(R)):
            D.append(Dag(C[c]) @ R[r])
    X_D = utils.krauss_to_X(D, 3, 1)
    D = utils.X_to_krauss(X_D)

    return {"C" : C, "R" : R, "D" : D, "X_C" : X_C, "X_R" : X_R, "X_D" : X_D}