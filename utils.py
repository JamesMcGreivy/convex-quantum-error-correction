"""
File: utils.py
Author: James McGreivy
Date: 04/29/2024
Description: Some general utility functions.
"""
import numpy as np
import torch
import itertools
import opt_einsum

# ~~ general utility functions ~~ #

def Dag(M):
    return M.T.conj()

def compute_fidelity(X_C, X_E, X_D):
    return (1/X_C.shape[1]**2) * torch.einsum("iljg,misj,lmgs->", X_D, X_C, X_E).real

def compute_fidelity_krauss(C, E, D):
    D = np.stack(D)
    E = np.stack(E)
    C = np.stack(C)
    comb = opt_einsum.contract("qij,rjk,skl->qrsil", D, E, C)
    return (1/C[0].shape[1]**2) * (np.abs(np.trace(comb, axis1=3, axis2=4))**2).sum()

def compute_fidelity_krauss_2(C, D):
    return (1/C[0].shape[1]**2) * sum([torch.trace(d @ c).abs()**2 for d in D for c in C])

# ~~ utility functions for converting between krauss operators and process matrices ~~ #

def X_to_krauss(X):
    X_flat = X.flatten(start_dim = 0, end_dim=1).flatten(start_dim=1, end_dim=2)
    V, S, _ = torch.svd(X_flat)
    V = V.unflatten(dim=0, sizes=[X.shape[0], X.shape[1]])
    
    K = []
    for r in range(V.shape[-1]):
        if S[r] < 1e-5:
            continue
        K.append(np.sqrt(S[r]) * V[:,:,r])
    return K

def krauss_to_X(K):
    return sum([torch.einsum("il,jk->iljk", k, k.conj()) for k in K])

def apply_krauss(rho, K):
    return sum([k @ rho @ Dag(k) for k in K])

def rank_X(X):
    X_flat = X.flatten(start_dim=0,end_dim=1).flatten(start_dim=1,end_dim=2)
    return torch.linalg.matrix_rank(X_flat, atol=1e-6)

# ~~ utility functions for the recursive composition of multiple QEC encoding schemes ~~ #

def tensor_prod(lst):
    if len(lst) == 1:
        return lst[0]
    else:
        return torch.kron(lst[0], tensor_prod(lst[1:]))

def compose_c(C_a, C_b, base=2):
    C = []
    for c_a in C_a:
        for c_b in C_b:
            if base == 3:
                c_b = torch.column_stack((c_b, torch.zeros(c_b.shape[0])))
            C.append(tensor_prod([c_b for _ in range(int(np.emath.logn(base, c_a.shape[0])))]) @ c_a)
    return C

def compose_d(D_a, D_b, base=2):
    q_c = int(np.emath.logn(base, D_a[0].shape[1]))
    D = []
    for d_a in D_a:
        for idxs in itertools.product(*[range(len(D_b)) for _ in range(q_c)]):
            if base == 3:
                D.append(d_a @ tensor_prod([torch.vstack((D_b[i],torch.zeros(D_b[i].shape[1]))) for i in idxs]))
            if base == 2:
                D.append(d_a @ tensor_prod([D_b[i] for i in idxs]))
    return D

def compose_qec(qec1, qec2, base=2):
    C_a = qec1["C"]
    C_b = qec2["C"]
    D_a = qec1["D"]
    D_b = qec2["D"]

    C = compose_c(C_a, C_b, base=base)
    D = compose_d(D_a, D_b, base=base)

    return {"C" : C, "D" : D}


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

def calculate_rank(X):
    X_flat = X.flatten(start_dim=2,end_dim=3).flatten(start_dim=0,end_dim=1)
    return torch.linalg.matrix_rank(X_flat)

def lowrank_approx(X, rank):
    X_flat = X.flatten(start_dim=0,end_dim=1).flatten(start_dim=1,end_dim=2)
    U, S, Vh = torch.linalg.svd(X_flat)
    S[rank : ] = 0
    X_lowrank = (U @ torch.diag(S).type(torch.complex64) @ Vh).unflatten(1, (X.shape[2], X.shape[3])).unflatten(0, (X.shape[0], X.shape[1]))
    return X_lowrank


