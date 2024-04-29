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

# Average fidelity loss function
def avg_fidelity(X_C, X_E, X_R):
    return (1 / (X_C.shape[0]**2)) * opt_einsum.contract("iljg,misj,lmgs->", X_R, X_C, X_E).real

# Regularization term equal to zero if X, the process matrix, sums up to the identity as required of a quantum channel.
def sums_to_identity(X):
    d = torch.eye(X.shape[-1])
    normalization = (opt_einsum.contract("ijik->jk",X) - d)
    normalization = (normalization.conj() * normalization).real
    return opt_einsum.contract("jk->", normalization)