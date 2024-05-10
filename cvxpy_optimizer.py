import numpy as np
import torch
import opt_einsum
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from matplotlib import rc
rc('font',**{'family':'normal','size':26,'weight':'bold'})
rc('text', usetex=True)

import utils
import cvxpy_caller
import process_matrix
import error_schemes
import qec_schemes
import qiskit

import importlib
importlib.reload(utils)
importlib.reload(process_matrix)
importlib.reload(cvxpy_caller)
importlib.reload(error_schemes)
importlib.reload(qec_schemes)

device = "cpu"

# Load stabilizer codes
unprotected = qec_schemes.nothing(device)
bitflip_3 = qec_schemes.three_qubit_bitflip_qec(device)
phaseflip_3 = qec_schemes.three_qubit_phaseflip_qec(device)
stabilizer_5 = qec_schemes.five_qubit_qec(device)

# Train qubit convex QEC
q = 3

l_AD = 0.25
l_depolarizing = 0
l_dephasing = 0

error_model = [error_schemes.AD(2**q, l=0.0, device=device)] + [error_schemes.AD(2**q, l=l) for l in np.arange(0.01, l_AD + 1e-6, 0.01)] + [error_schemes.depolarizing(2**q, l=l) for l in np.arange(0.01, l_depolarizing + 1e-6, 0.01)] + [error_schemes.dephasing(2**q, l=l) for l in np.arange(0.01, l_dephasing + 1e-6, 0.01)]
cvx = qec_schemes.convex_optimizer_qec(2**1, 2**q, error_model, device=device)

# Train qutrit convex QEC
qt = 2

l_AD = 0.25
l_depolarizing = 0
l_dephasing = 0

error_model = [error_schemes.AD_qutrit(3**qt, l=0.0, device=device)] + [error_schemes.AD_qutrit(3**qt, l=l) for l in np.arange(0.01, l_AD + 1e-6, 0.01)] + [error_schemes.depolarizing_qutrit(3**qt, l=l) for l in np.arange(0.01, l_depolarizing + 1e-6, 0.01)] + [error_schemes.dephasing_qutrit(3**qt, l=l) for l in np.arange(0.01, l_dephasing + 1e-6, 0.01)]
cvx_qt = qec_schemes.convex_optimizer_qec(2**1, 3**qt, error_model, device=device)

# Concatenate codes
cvx_concat = utils.compose_qec(cvx, cvx_qt, base = 2)

def fidelity_vs_error(qec, error_scheme):
    fids = []
    errors = np.linspace(0, 0.5, 10)
    for e in errors:
        E = error_scheme(qec["C"][0].shape[0], e)
        fid = utils.compute_fidelity_krauss(C=qec["C"], E=E, D=qec["D"]).item()
        fids.append(fid)
        del E
    return errors, np.array(fids)

fig, axs = plt.subplots(3,1, figsize = (5,9))

axs[0].plot(*fidelity_vs_error(unprotected, error_schemes.AD_krauss), label = "Single Qubit", color = "black")
axs[0].plot(*fidelity_vs_error(stabilizer_5, error_schemes.AD_krauss), label = "5 qubit stabilizer", color = "orange")
axs[0].plot(*fidelity_vs_error(cvx_qt, error_schemes.AD_qutrit_krauss), label = "Convex Optimized QEC, qutrit", color = "blue", linestyle = "dashed")
axs[0].plot(*fidelity_vs_error(cvx, error_schemes.AD_krauss), label = "Convex Optimized QEC, qubit", color = "pink", linestyle = "dashed")
axs[0].plot(*fidelity_vs_error(cvx_concat, error_schemes.AD_qutrit_krauss), label = "Convex Optimized QEC, composed", color = "red", linestyle = "dashed")

axs[0].set_xlim(0, 0.5)
axs[0].set_ylim(0.4, 1.05)
axs[0].set_xlabel("$f_{avg}$")
axs[0].set_ylabel("$\lambda$")
axs[0].set_title("Avg fidelity on amplitude damping channel")
axs[0].legend()

axs[1].plot(*fidelity_vs_error(unprotected, error_schemes.depolarizing_krauss), label = "Single Qubit", color = "black")
axs[1].plot(*fidelity_vs_error(stabilizer_5, error_schemes.depolarizing_krauss), label = "5 qubit stabilizer", color = "orange")
axs[1].plot(*fidelity_vs_error(cvx_qt, error_schemes.depolarizing_qutrit_krauss), label = "Convex Optimized QEC, qutrit", color = "blue", linestyle = "dashed")
axs[1].plot(*fidelity_vs_error(cvx, error_schemes.depolarizing_krauss), label = "Convex Optimized QEC, qubit", color = "pink", linestyle = "dashed")
axs[1].plot(*fidelity_vs_error(cvx_concat, error_schemes.depolarizing_qutrit_krauss), label = "Convex Optimized QEC, composed", color = "red", linestyle = "dashed")

axs[1].set_xlim(0, 0.5)
axs[1].set_ylim(0.4, 1.05)
axs[1].set_xlabel("$f_{avg}$")
axs[1].set_ylabel("$\lambda$")
axs[1].set_title("Avg fidelity on depolarizing channel")
axs[1].legend()

axs[2].plot(*fidelity_vs_error(unprotected, error_schemes.dephasing_krauss), label = "Single Qubit", color = "black")
axs[2].plot(*fidelity_vs_error(stabilizer_5, error_schemes.dephasing_krauss), label = "5 qubit stabilizer", color = "orange")
axs[2].plot(*fidelity_vs_error(cvx_qt, error_schemes.dephasing_qutrit_krauss), label = "Convex Optimized QEC, qutrit", color = "blue", linestyle = "dashed")
axs[2].plot(*fidelity_vs_error(cvx, error_schemes.dephasing_krauss), label = "Convex Optimized QEC, qubit", color = "pink", linestyle = "dashed")
axs[2].plot(*fidelity_vs_error(cvx_concat, error_schemes.dephasing_qutrit_krauss), label = "Convex Optimized QEC, composed", color = "red", linestyle = "dashed")

axs[2].set_xlim(0, 0.5)
axs[2].set_ylim(0.4, 1.05)
axs[2].set_xlabel("$f_{avg}$")
axs[2].set_ylabel("$\lambda$")
axs[2].set_title("Avg fidelity on dephasing channel")
axs[2].legend()

plt.tight_layout()
plt.savefig("temp.png", dpi=600)