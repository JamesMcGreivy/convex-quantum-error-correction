"""
File: cvxpy_caller.py
Author: James McGreivy
Date: 04/29/2024
Description: Functions wrapping CVXPY to perform convex optimization of a process matrix
"""
import cvxpy as cp
import torch

def maximize_fidelity(X, W):
    W_flat = W.flatten(start_dim=0, end_dim=1).flatten(start_dim=1, end_dim=2)
    X_var = cp.Variable((X.shape[0]*X.shape[1], X.shape[2]*X.shape[3]), hermitian = True)
    
    constraints = [X_var >> 0, cp.partial_trace(X_var, dims=(X.shape[0], X.shape[1]), axis=0) == torch.eye(X.shape[1])]
    prob = cp.Problem(cp.Maximize(cp.real(cp.sum(cp.multiply(X_var, W_flat)))), constraints)
    
    prob.solve()
    return prob.value, torch.tensor(X_var.value).unflatten(1, (X.shape[2], X.shape[3])).unflatten(0, (X.shape[0], X.shape[1]))