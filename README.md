## This is the repository of my [final project](https://github.com/JamesMcGreivy/convex-quantum-error-correction/blob/main/8_371_Final_Project_Paper.pdf) for 8.371 - Quantum Information Science & Quantum Computing II @ MIT.

In this final project I describe a procedure to reframe quantum error correction as a bi-convex optimization problem. I then implement a solver using Python, and explicitly solve for quantum error correcting codes which are optimal against error ensembles constructed from the amplitude damping, depolarizing, and dephasing noise models. I also test this formalism in the case of one qubit embedded into a three qutrit Hilbert space, as well as in the case of concatenating two quantum error correcting codes. I find that if errors are able to be well characterized, these optimal codes can yield better error correcting performance – measured using average channel fidelity – than a generic Stabilizer code. This comes with a tradeoff, however, of being less robust against errors which were not optimizer for. The key results are highlighted below -- the left column was trained using an equiprobabile error ensemble, the right column was trained using a damping dominant error ensemble.

<img width="600" alt="spectra" src="https://github.com/JamesMcGreivy/convex-quantum-error-correction/blob/main/assets/results.png">
