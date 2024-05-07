"""
File: process_matrix.py
Author: James McGreivy
Date: 04/29/2024
Description: Containts a pytorch module to wrap the process matrices (X_C or X_R) as well as utility functions for
constructing the process matrices. 
"""
import numpy as np
import torch
import itertools
import utils

# Initializes an arbitrary process matrix according to the required constraints.
def initialize_process_matrix(n_1, n_2, device = "cpu"):
    X = torch.rand(size=(n_2*n_1, n_2*n_1), dtype=torch.complex128)
    X = X.conj().T @ X
    return X.unflatten(1, (n_2, n_1)).unflatten(0, (n_2, n_1))

class ProcessMatrix(torch.nn.Module):
    def __init__(self, q_1, q_2, device = "cpu"):
        super().__init__()

        self.n_1 = 2**q_1
        self.n_2 = 2**q_2
        
        self.X = torch.nn.Parameter(initialize_process_matrix(self.n_1, self.n_2, device))

    # Returns the process matrices for the coding, error, and recovery channel
    def forward(self):
        return self.X



