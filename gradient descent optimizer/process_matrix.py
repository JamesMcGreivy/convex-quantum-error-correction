"""
File: process_matrix.py
Author: James McGreivy
Date: 04/29/2024
Description: A pytorch module which wraps the process matrices (X_C or X_R). 
"""
import numpy as np
import torch
import opt_einsum
import itertools

class ProcessMatrix(torch.nn.Module):
    def __init__(self, q_1, q_2, device = "cpu"):
        super().__init__()

        self.n_1 = 2**q_1
        self.n_2 = 2**q_2
        
        self.X = torch.nn.Parameter(torch.normal(mean=0, std=1/(self.n_1**2 * self.n_2**2),size=(self.n_2,self.n_1,self.n_2,self.n_1),dtype=torch.complex128,device=device))

    # Returns the process matrices for the coding, error, and recovery channel
    def forward(self):
        return self.X



