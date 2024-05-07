
"""
File: qec_schemes.py
Author: James McGreivy
Date: 04/29/2024
Description: This file is used to instantiate several quantum error correcting (qec) schemes, such as the
3 qubit repetition code as well as the convex optimizer code.
"""
import torch
import numpy as np
import cvxpy_caller
import process_matrix
import utils
from utils import Dag

def convex_optimizer_qec(q_s, q_c, X_Es, X_C = None, X_D = None, device = "cpu"):
    n_s = 2**q_s
    n_c = 2**q_c
    
    if X_C is None:
        X_C = process_matrix.initialize_process_matrix(n_s, n_c, device = device)
    if X_D is None:
        X_D = process_matrix.initialize_process_matrix(n_c, n_s, device = device)
    optimizer_C = None
    optimizer_D = None

    cutoff = 1e-10
    delta = np.inf
    prev_f_avg = 0
    while (delta > cutoff):
        W_Cs = [(1/n_s**2) * torch.einsum("iljg,lmgs->misj", X_D, X_E) for X_E in X_Es]
        if optimizer_C is None:
            optimizer_C = cvxpy_caller.CVXPY_Caller(X_C, W_Cs)
        X_C, _ = optimizer_C(X_C, W_Cs)

        W_Ds = [(1/n_s**2) * torch.einsum("misj,lmgs->iljg", X_C, X_E) for X_E in X_Es]
        if optimizer_D is None:
            optimizer_D = cvxpy_caller.CVXPY_Caller(X_D, W_Ds)
        X_D, f_avg = optimizer_D(X_D, W_Ds)

        print(f_avg)
        delta = (f_avg - prev_f_avg)
        prev_f_avg = f_avg

    return {"X_C" : X_C, "X_D" : X_D}

# Pauli Matrices
I = torch.tensor([[1,0],[0,1]], dtype = torch.complex128)
X = torch.tensor([[0,1],[1,0]], dtype = torch.complex128)
Y = torch.tensor([[0,-1j],[1j,0]], dtype = torch.complex128)
Z = torch.tensor([[1,0],[0,-1]], dtype = torch.complex128)

# States
Psi0 = torch.tensor([[1,0],], dtype = torch.complex128).T
Psi1 = torch.tensor([[0,1],], dtype = torch.complex128).T

def stabilizer_projector(stabilizers, result):
    bigI = torch.eye(stabilizers[0].shape[0], dtype = torch.complex128)
    projector = torch.eye(stabilizers[0].shape[0], dtype = torch.complex128)
    for i, res in enumerate(result):
        projector = projector @ (bigI + (-1)**res * stabilizers[i])/2.0
    return projector

def three_qubit_bitflip_qec():
    # ~~ Recovery krauss operators ~~
    U0 = torch.kron(I, torch.kron(I, I))
    P0 = torch.kron(Psi0@Dag(Psi0), torch.kron(Psi0@Dag(Psi0), Psi0@Dag(Psi0))) + torch.kron(Psi1@Dag(Psi1), torch.kron(Psi1@Dag(Psi1), Psi1@Dag(Psi1)))
    R0 = U0 @ P0

    U1 = torch.kron(X, torch.kron(I, I))
    P1 = torch.kron(Psi1@Dag(Psi1), torch.kron(Psi0@Dag(Psi0), Psi0@Dag(Psi0))) + torch.kron(Psi0@Dag(Psi0), torch.kron(Psi1@Dag(Psi1), Psi1@Dag(Psi1)))
    R1 = U1 @ P1

    U2 = torch.kron(I, torch.kron(X, I))
    P2 = torch.kron(Psi0@Dag(Psi0), torch.kron(Psi1@Dag(Psi1), Psi0@Dag(Psi0))) + torch.kron(Psi1@Dag(Psi1), torch.kron(Psi0@Dag(Psi0), Psi1@Dag(Psi1)))
    R2 = U2 @ P2

    U3 = torch.kron(I, torch.kron(I, X))
    P3 = torch.kron(Psi0@Dag(Psi0), torch.kron(Psi0@Dag(Psi0), Psi1@Dag(Psi1))) + torch.kron(Psi1@Dag(Psi1), torch.kron(Psi1@Dag(Psi1), Psi0@Dag(Psi0)))
    R3 = U3 @ P3

    R = [R0, R1, R2, R3]
    X_R = utils.krauss_to_X(R, 3, 3)

    # ~~ Encoding krauss operators ~~
    C0 = torch.kron(Psi0, torch.kron(Psi0, Psi0)) @ Dag(Psi0) + torch.kron(Psi1, torch.kron(Psi1, Psi1)) @ Dag(Psi1)
    C = [C0]
    X_C = utils.krauss_to_X(C, 1, 3)

    D = []
    for c in range(len(C)):
        for r in range(len(R)):
            D.append(Dag(C[c]) @ R[r])
    X_D = utils.krauss_to_X(D, 3, 1)
    D = utils.X_to_krauss(X_D)

    return {"C" : C, "R" : R, "D" : D, "X_C" : X_C, "X_R" : X_R, "X_D" : X_D}


def five_qubit_qec():
    # ~~ Recovery krauss operators ~~
    stabilizers = [torch.kron(X, torch.kron(Z, torch.kron(Z, torch.kron(X, I)))),
                   torch.kron(I, torch.kron(X, torch.kron(Z, torch.kron(Z, X)))),
                   torch.kron(X, torch.kron(I, torch.kron(X, torch.kron(Z, Z)))),
                   torch.kron(Z, torch.kron(X, torch.kron(I, torch.kron(X, Z))))]

    P0 = stabilizer_projector(stabilizers, [0,0,0,1])
    U0 = torch.kron(X, torch.kron(I, torch.kron(I, torch.kron(I, I))))
    R0 = U0 @ P0

    P1 = stabilizer_projector(stabilizers, [1,0,0,0])
    U1 = torch.kron(I, torch.kron(X, torch.kron(I, torch.kron(I, I))))
    R1 = U1 @ P1

    P2 = stabilizer_projector(stabilizers, [1,1,0,0])
    U2 = torch.kron(I, torch.kron(I, torch.kron(X, torch.kron(I, I))))
    R2 = U2 @ P2

    P3 = stabilizer_projector(stabilizers, [0,1,1,0])
    U3 = torch.kron(I, torch.kron(I, torch.kron(I, torch.kron(X, I))))
    R3 = U3 @ P3

    P4 = stabilizer_projector(stabilizers, [0,0,1,1])
    U4 = torch.kron(I, torch.kron(I, torch.kron(I, torch.kron(I, X))))
    R4 = U4 @ P4

    P5 = stabilizer_projector(stabilizers, [1,0,1,0])
    U5 = torch.kron(Z, torch.kron(I, torch.kron(I, torch.kron(I, I))))
    R5 = U5 @ P5

    P6 = stabilizer_projector(stabilizers, [0,1,0,1])
    U6 = torch.kron(I, torch.kron(Z, torch.kron(I, torch.kron(I, I))))
    R6 = U6 @ P6

    P7 = stabilizer_projector(stabilizers, [0,0,1,0])
    U7 = torch.kron(I, torch.kron(I, torch.kron(Z, torch.kron(I, I))))
    R7 = U7 @ P7

    P8 = stabilizer_projector(stabilizers, [1,0,0,1])
    U8 = torch.kron(I, torch.kron(I, torch.kron(I, torch.kron(Z, I))))
    R8 = U8 @ P8

    P9 = stabilizer_projector(stabilizers, [0,1,0,0])
    U9 = torch.kron(I, torch.kron(I, torch.kron(I, torch.kron(I, Z))))
    R9 = U9 @ P9

    P10 = stabilizer_projector(stabilizers, [1,0,1,1])
    U10 = torch.kron(Y, torch.kron(I, torch.kron(I, torch.kron(I, I))))
    R10 = U10 @ P10

    P11 = stabilizer_projector(stabilizers, [1,1,0,1])
    U11 = torch.kron(I, torch.kron(Y, torch.kron(I, torch.kron(I, I))))
    R11 = U11 @ P11

    P12 = stabilizer_projector(stabilizers, [1,1,1,0])
    U12 = torch.kron(I, torch.kron(I, torch.kron(Y, torch.kron(I, I))))
    R12 = U12 @ P12

    P13 = stabilizer_projector(stabilizers, [1,1,1,1])
    U13 = torch.kron(I, torch.kron(I, torch.kron(I, torch.kron(Y, I))))
    R13 = U13 @ P13

    P14 = stabilizer_projector(stabilizers, [0,1,1,1])
    U14 = torch.kron(I, torch.kron(I, torch.kron(I, torch.kron(I, Y))))
    R14 = U14 @ P14

    P15 = stabilizer_projector(stabilizers, [0,0,0,0])
    U15 = torch.kron(I, torch.kron(I, torch.kron(I, torch.kron(I, I))))
    R15 = U15 @ P15

    R = [R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15]
    X_R = utils.krauss_to_X(R, 5, 5)

    # ~~ Encoding krauss operators ~~
    Logical0 = 1/4 * (torch.kron(Psi0, torch.kron(Psi0, torch.kron(Psi0, torch.kron(Psi0, Psi0)))) +
        torch.kron(Psi1, torch.kron(Psi0, torch.kron(Psi0, torch.kron(Psi1, Psi0)))) +
        torch.kron(Psi0, torch.kron(Psi1, torch.kron(Psi0, torch.kron(Psi0, Psi1)))) + 
        torch.kron(Psi1, torch.kron(Psi0, torch.kron(Psi1, torch.kron(Psi0, Psi0)))) +
        torch.kron(Psi0, torch.kron(Psi1, torch.kron(Psi0, torch.kron(Psi1, Psi0)))) -
        torch.kron(Psi1, torch.kron(Psi1, torch.kron(Psi0, torch.kron(Psi1, Psi1)))) -
        torch.kron(Psi0, torch.kron(Psi0, torch.kron(Psi1, torch.kron(Psi1, Psi0)))) -
        torch.kron(Psi1, torch.kron(Psi1, torch.kron(Psi0, torch.kron(Psi0, Psi0)))) -
        torch.kron(Psi1, torch.kron(Psi1, torch.kron(Psi1, torch.kron(Psi0, Psi1)))) - 
        torch.kron(Psi0, torch.kron(Psi0, torch.kron(Psi0, torch.kron(Psi1, Psi1)))) -
        torch.kron(Psi1, torch.kron(Psi1, torch.kron(Psi1, torch.kron(Psi1, Psi0)))) -
        torch.kron(Psi0, torch.kron(Psi1, torch.kron(Psi1, torch.kron(Psi1, Psi1)))) -
        torch.kron(Psi1, torch.kron(Psi0, torch.kron(Psi0, torch.kron(Psi0, Psi1)))) -
        torch.kron(Psi0, torch.kron(Psi1, torch.kron(Psi1, torch.kron(Psi0, Psi0)))) -
        torch.kron(Psi1, torch.kron(Psi0, torch.kron(Psi1, torch.kron(Psi1, Psi1)))) +
        torch.kron(Psi0, torch.kron(Psi0, torch.kron(Psi1, torch.kron(Psi0, Psi1)))))

    Logical1 = 1/4 * (torch.kron(Psi1, torch.kron(Psi1, torch.kron(Psi1, torch.kron(Psi1, Psi1)))) +
        torch.kron(Psi0, torch.kron(Psi1, torch.kron(Psi1, torch.kron(Psi0, Psi1)))) +
        torch.kron(Psi1, torch.kron(Psi0, torch.kron(Psi1, torch.kron(Psi1, Psi0)))) +
        torch.kron(Psi0, torch.kron(Psi1, torch.kron(Psi0, torch.kron(Psi1, Psi1)))) +
        torch.kron(Psi1, torch.kron(Psi0, torch.kron(Psi1, torch.kron(Psi0, Psi1)))) -
        torch.kron(Psi0, torch.kron(Psi0, torch.kron(Psi1, torch.kron(Psi0, Psi0)))) -
        torch.kron(Psi1, torch.kron(Psi1, torch.kron(Psi0, torch.kron(Psi0, Psi1)))) -
        torch.kron(Psi0, torch.kron(Psi0, torch.kron(Psi1, torch.kron(Psi1, Psi1)))) -
        torch.kron(Psi0, torch.kron(Psi0, torch.kron(Psi0, torch.kron(Psi1, Psi0)))) -
        torch.kron(Psi1, torch.kron(Psi1, torch.kron(Psi1, torch.kron(Psi0, Psi0)))) -
        torch.kron(Psi0, torch.kron(Psi0, torch.kron(Psi0, torch.kron(Psi0, Psi1)))) -
        torch.kron(Psi1, torch.kron(Psi0, torch.kron(Psi0, torch.kron(Psi0, Psi0)))) -
        torch.kron(Psi0, torch.kron(Psi1, torch.kron(Psi1, torch.kron(Psi1, Psi0)))) -
        torch.kron(Psi1, torch.kron(Psi0, torch.kron(Psi0, torch.kron(Psi1, Psi1)))) -
        torch.kron(Psi0, torch.kron(Psi1, torch.kron(Psi0, torch.kron(Psi0, Psi0)))) +
        torch.kron(Psi1, torch.kron(Psi1, torch.kron(Psi0, torch.kron(Psi1, Psi0)))))
            
    C = [Logical0 @ Dag(Psi0) + Logical1 @ Dag(Psi1)]
    X_C = utils.krauss_to_X(C, 1, 3)

    D = []
    for c in range(len(C)):
        for r in range(len(R)):
            D.append(Dag(C[c]) @ R[r])
    X_D = utils.krauss_to_X(D, 3, 1)
    D = utils.X_to_krauss(X_D)

    return {"C" : C, "R" : R, "D" : D, "X_C" : X_C, "X_R" : X_R, "X_D" : X_D}


def nothing():

    C = [I]
    X_C = utils.krauss_to_X(C, 1, 1)

    D = [I]
    X_D = utils.krauss_to_X(D, 1, 1)

    return {"X_C" : X_C, "X_D": X_D}