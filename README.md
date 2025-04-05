# Tensor network renormalization approach to antiferromagnetic 6-state clock model on the Union Jack lattice

This repository contains sample code for the research paper arXiv:2403.17309, titled "Tensor network renormalization approach to antiferromagnetic 6-state clock model on the Union Jack lattice".

## Overview
This project implements the tensor network renormalization (TNR) approach for the antiferromagnetic 6-state clock model on the Union Jack lattice. The goal is to provide tools for performing numerical simulations and analysis based on the TNR method.
Here, TNR utilizes the nuclear norm regularized (NNR) loop opmization technique. For its detail, see Phys. Rev. Research 6, 043102 and https://github.com/KenjiHomma/NNR-TNR.

The repository includes Python scripts for:

- NNR-TNR_one_impurity.py: Scripts that computes the one-point function using NNR-TNR.

- NNR-TNR_L4_lobcg.py: Scripts that computes the central charge, scaling dimension and OPE coefficients with the column length l_x=4 transfer matrix.

- LOBPCG_transfer_L4.py: Eigenvalue solvers used in the simulations.

- contraction.py: A module for tensor contraction operations.

- hamiltonian.py: Contains Hamiltonian definitions for the 6-state clock model.

## Requirements
Python 3.x

Necessary Python libraries (e.g., NumPy, SciPy) for tensor operations and linear algebra.

## Usage
To run the simulations,

 ```
python3 NNR-TNR_one_impurity.py
 ```


## Citation
If you use this code in your research, please cite the original paper:
Kenji Homma, Satoshi Morita, Naoki Kawashima, "Tensor network renormalization approach to antiferromagnetic 6-state clock model on the Union Jack lattice", arXiv:2403.17309.
