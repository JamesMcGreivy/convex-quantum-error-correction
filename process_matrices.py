"""
File: QuantumErrorCorrector.py
Author: James McGreivy
Date: 04/29/2024
Description: A pytorch module which wraps the process matrices X_C, X_E, and X_R. 
"""
import numpy as np
import torch
import opt_einsum
import itertools

class ProcessMatrices(torch.nn.Module):
    def __init__(self, q_s, q_c, K = None, X_E = None, device = "cpu"):
        super().__init__()
        self.q_s = q_s
        self.q_c = q_c

        self.n_s = 2**q_s
        self.n_c = 2**q_c

        if (K is None) == (X_E is None):
            raise ValueError("Please provide either K or X_E")
        if X_E is None:
            K = [k.to(device) for k in K]
            self.X_E = self.krauss_to_X_E(K)
        else:
            self.X_E = X_E.to(device)
        self.X_E.requires_grad = False
        
        self.X_C = torch.nn.Parameter(torch.normal(mean=0, std=1/(self.n_c**2 * self.n_s**2),size=(self.n_c,self.n_s,self.n_c,self.n_s),dtype=torch.complex128,device=device))
        
        self.X_R = torch.nn.Parameter(torch.normal(mean=0, std=1/(self.n_c**2 * self.n_s**2),size=(self.n_s,self.n_c,self.n_s,self.n_c),dtype=torch.complex128,device=device))

    # Returns the process matrices for the coding, error, and recovery channel
    def forward(self):
        X_C = self.X_C
        X_E = self.X_E
        X_R = self.X_R
        return X_C, X_E, X_R 

    # Constructs X_E, the error process matrix, given a set of 2x2 Krauss operators describing a single-qubit error channel.
    def krauss_to_X_E(self, K):
        E = torch.zeros(size=self.q_c*[4] + 2*[self.n_c], dtype = torch.complex128, device = K[0].device)

        for i in itertools.product(*[range(0,len(K)) for _ in range(0, self.q_c)]):
            K_tot = K[i[0]]
            for j in range(1, self.q_c):
                K_tot = torch.kron(K_tot, K[i[j]])
            E[i] = K_tot

        X_E = opt_einsum.contract("".join([str(i) for i in range(self.q_c)]) + "lm," + "".join([str(i) for i in range(self.q_c)]) + "gs->lmgs", E, E.conj())
        return X_E

