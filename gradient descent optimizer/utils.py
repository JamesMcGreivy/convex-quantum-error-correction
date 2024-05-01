"""
File: utils.py
Author: James McGreivy
Date: 04/29/2024
Description: Some utility functions for the optimization.
"""
import numpy as np
import torch
import opt_einsum
import itertools

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

# Regularization term equal to zero if X, the process matrix, sums up to the identity as required of a quantum channel.
def sums_to_identity(X):
    d = torch.eye(X.shape[-1], device = X.device)
    normalization = (opt_einsum.contract("ijik->jk",X) - d)
    normalization = (normalization.conj() * normalization).real
    return opt_einsum.contract("jk->", normalization)

# Regularization term equal to zero if X, the process matrix, is PSD
def positive_eigenvalues(X):
    eig = torch.linalg.eigvals(X.flatten(start_dim=0,end_dim=1).flatten(start_dim=1,end_dim=2)).real
    return torch.sum((eig * (eig < 0)).abs())

# Forces a tensor to be PSD
def make_PSD(X):
    X_flat = X.flatten(start_dim=2,end_dim=3).flatten(start_dim=0,end_dim=1)
    
    # Forces the tensor to be hermitian
    X_flat = (X_flat + X_flat.conj().T)/2

    # Performs eigenvalue decomposition and sets all negative eigenvalues to zero
    eig = torch.linalg.eig(X_flat)
    eigenvalues = eig.eigenvalues
    eigenvectors = eig.eigenvectors
    eigenvalues = eigenvalues * (eigenvalues.real >= 0)

    X_PSD_flat = eig.eigenvectors @ torch.diag(eigenvalues) @ eig.eigenvectors.conj().T
    X_PSD = X_PSD_flat.unflatten(1,X.shape[0:2]).unflatten(0,X.shape[0:2])
    return X_PSD

# Forces a process tensor to sum to the identity as required of a quantum channel
def make_sum_to_identity(X):
    partial_diag = opt_einsum.contract("kikj->ij",X)
    for k in range(X.shape[0]):
        X[k, range(X.shape[1]), k, range(X.shape[1])] /= partial_diag[range(X.shape[1]), range(X.shape[1])]
        X[k, range(X.shape[1]), k, range(X.shape[1])] += (partial_diag[range(X.shape[1]),range(X.shape[1])] / X.shape[0])
        X[k, range(X.shape[1]), k, :] -= (partial_diag[range(X.shape[1]),:] / X.shape[0])
    return X