"""
File: utils.py
Author: James McGreivy
Date: 04/29/2024
Description: Some general utility functions.
"""
import numpy as np
import torch
import itertools

# ~~ general utility functions ~~ #

def Dag(M):
    return M.T.conj()

def compute_fidelity(X_C, X_E, X_D):
    return (1/X_C.shape[1]**2) * torch.einsum("iljg,misj,lmgs->", X_D, X_C, X_E).real

# ~~ utility functions for converting between krauss operators and process matrices ~~ #

def X_to_krauss(X):
    X_flat = X.flatten(start_dim = 0, end_dim=1).flatten(start_dim=1, end_dim=2)
    V, S, _ = torch.svd(X_flat)
    V = V.unflatten(dim=0, sizes=[X.shape[0], X.shape[1]])
    
    K = []
    for r in range(V.shape[-1]):
        K.append(np.sqrt(S[r]) * V[:,:,r])
    return K

def krauss_to_X(K, q_1, q_2):
    n_1 = 2**q_1
    n_2 = 2**q_2
    return sum([torch.einsum("il,jk->iljk", k, k.conj()) for k in K])

def apply_krauss(rho, K):
    return sum([k @ rho @ Dag(k) for k in K])

def rank_X(X):
    X_flat = X.flatten(start_dim=0,end_dim=1).flatten(start_dim=1,end_dim=2)
    return torch.linalg.matrix_rank(X_flat, atol=1e-6)

# ~~ utility functions for convex optimization over process matrices ~~ #

# Regularization term equal to zero if X, the process matrix, sums up to the identity as required of a quantum channel.
def sums_to_identity(X):
    d = torch.eye(X.shape[-1], device = X.device)
    normalization = (torch.einsum("ijik->jk",X) - d)
    normalization = (normalization.conj() * normalization).real
    return torch.einsum("jk->", normalization)

def positive_eigenvalues(X):
    eig = torch.linalg.eigvals(X.flatten(start_dim=0,end_dim=1).flatten(start_dim=1,end_dim=2)).real
    return torch.sum((eig * (eig < 0)).abs())

# Regularization term equal to zero if X, the process matrix, is PSD
def is_PSD(X):
    positive = positive_eigenvalues(X)
    X = X.flatten(start_dim=0, end_dim=1).flatten(start_dim=1, end_dim=2)
    hermitian = (X - X.T.conj()).abs().max()
    return positive, hermitian

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
    partial_diag = torch.einsum("kikj->ij",X)
    for k in range(X.shape[0]):
        X[k, range(X.shape[1]), k, range(X.shape[1])] /= partial_diag[range(X.shape[1]), range(X.shape[1])]
        X[k, range(X.shape[1]), k, range(X.shape[1])] += (partial_diag[range(X.shape[1]),range(X.shape[1])] / X.shape[0])
        X[k, range(X.shape[1]), k, :] -= (partial_diag[range(X.shape[1]),:] / X.shape[0])
    return X

