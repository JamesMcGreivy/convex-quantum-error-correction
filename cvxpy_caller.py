"""
File: cvxpy_caller.py
Author: James McGreivy
Date: 04/29/2024
Description: Functions wrapping CVXPY to perform convex optimization of a process matrix
"""
import cvxpy as cp
import torch
import numpy as np
import utils

class CVXPY_Caller:

    def __init__(self, X, Ws):
        self.X_var = cp.Variable((X.shape[0]*X.shape[1], X.shape[2]*X.shape[3]), PSD=True)
        self.constraints = [self.X_var >> 0, cp.partial_trace(self.X_var, dims=(X.shape[0], X.shape[1]), axis=0) == cp.Constant(np.identity(X.shape[1]).astype(np.complex64))]
        self.Ws_param = [cp.Parameter((W.shape[0]*W.shape[1], W.shape[2]*W.shape[3]), PSD=True) for W in Ws]

        problem = 0
        for i, W_param in enumerate(self.Ws_param):
            problem += cp.real(cp.sum(cp.multiply(self.X_var, self.Ws_param[i])))
        self.prob = cp.Problem(cp.Maximize(problem / (i + 1)), self.constraints)

    def __call__(self, X, Ws):
        for i, W_param in enumerate(self.Ws_param):
            W = Ws[i].flatten(start_dim=0, end_dim=1).flatten(start_dim=1, end_dim=2).cpu().numpy()
            W_param.value = W_param.project(W)
        self.prob.solve(warm_start=True, ignore_dpp = True)

        f_avg = float(self.prob.value)
        X_new = torch.tensor(self.X_var.value, dtype=torch.complex128)
        return X_new.unflatten(1, (X.shape[2], X.shape[3])).unflatten(0, (X.shape[0], X.shape[1])), f_avg