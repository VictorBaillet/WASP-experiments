import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import block_diag, vstack, csr_matrix
from scipy.optimize import linprog
import time
import pickle

import os

def call_lp_solver(A_eq, b_eq, cvec, bounds, maxiters=10000, solver='highs'):
    """
    Solve an LP problem using a specified solver.

    Parameters:
    - A_ub: Coefficient matrix for equality constraints.
    - b_ub: Right-hand side vector for equality constraints.
    - cvec: Coefficients of the linear objective function to be minimized.
    - maxiters: Maximum number of iterations.
    - solver: The LP solver to use.

    Returns:
    - xsol: Solution vector x.
    - output: A dictionary containing output information such as solve time.
    """

    start_time = time.time()
    res = linprog(cvec, A_eq=A_eq, b_eq=b_eq, bounds=bounds, options={'disp': False}, method=solver)
    end_time = time.time()

    xsol = res.x if res.success else None
    output = {
        'status': res.status,
        'message': res.message,
        'solve_time': end_time - start_time
    }

    return xsol, output

def read_pickle_files(dd, nsub, ndim, base_path='data/sub-sampling'):
    """Read pickle files corresponding to the data."""
    chr = 'mu1' if ndim == 1 else 'mu2'
    margMat = []
    for kk in range(1, nsub + 1):
        file_name = f"sample_{kk-1}.pkl" 
        file_path = os.path.join(base_path, file_name)
        with open(file_path, 'rb') as file:  
            margMat.append(pickle.load(file))
    return transform_dict_to_array(margMat)

def calculate_distances(margMat, nsub):
    """Calculate the pairwise squared Euclidean distances."""
    subsetPost = np.array([m[np.random.randint(0, len(m), 100), :] for m in margMat])
    lbd1, ubd1 = min(m[:, 0].min() for m in subsetPost), max(m[:, 0].max() for m in subsetPost)
    lbd2, ubd2 = min(m[:, 1].min() for m in subsetPost), max(m[:, 1].max() for m in subsetPost)

    overallPost = np.array(np.meshgrid(np.linspace(lbd1, ubd1, 30), np.linspace(lbd2, ubd2, 30))).T.reshape(-1, 2)
    distMatCell = [cdist(overallPost, sp, 'sqeuclidean') for sp in subsetPost]
    return overallPost, distMatCell, subsetPost

def transform_dict_to_array(margMat):
    transformed_data = []
    for m in margMat:
        ## The example focus on 'mu' but any other variable can be used
        flattened_data = m['mu'][:, 0].reshape(m['mu'].shape[0], -1)
        """
        flattened_data = np.concatenate([
            m['mu'].reshape(m['mu'].shape[0], -1),  # Flatten 'mu'
            m['cov'].reshape(m['cov'].shape[0], -1),  # Flatten 'cov'
            m['prob']  # 'prob' is already 2D but may need reshaping depending on its use
        ], axis=1)
        """
        transformed_data.append(flattened_data)
    
    return np.array(transformed_data)

def setup_lp_problem(D):
    """
    Setup for WASP LP problem.
    
    Parameters:
    D (list of np.ndarray): List of distance matrices D_j.
    
    Returns:
    A_eq (csr_matrix): Equality constraint matrix.
    b_eq (np.array): Equality constraint vector.
    c (np.array): Cost vector.
    bounds (list of tuple): Bounds for each variable.
    """
    g = D[0].shape[0] 
    k = len(D)
    s = [m.shape[1] for m in D]
    N = np.sum(s)
    ## Cost vector
    c = np.hstack([np.zeros(g)] + [D[j].flatten() for j in range(k)])
    rows_eq = []
    ## Constraint: 1^T a = 1
    rows_eq.append(np.hstack([np.ones(g), np.zeros(g*N)]))
    b_eq = [1]
    
    ## Constraint: T_j 1_sj = a and T_j^T 1_s = 1_sj / sj
    offset = g
    for j, sj in enumerate(s):
        ## T_j 1_sj = a
        for i in range(g):
            row = np.zeros(g + g*N)
            row[i] = -1  ## Corresponding a_i
            row[offset + i*sj : offset + (i+1)*sj] = 1  ## Corresponding elements in T_j
            rows_eq.append(row)
            b_eq.append(0)
        
        ## T_j^T 1_s = 1_sj / sj
        for i in range(sj):
            row = np.zeros(g + g*N)
            row[offset + i : offset + g*sj : sj] = 1  ## Corresponding elements in T_j^T
            rows_eq.append(row) 
            b_eq.append(1/sj)
        
        offset += g * sj
    
    A_eq = csr_matrix(np.array(rows_eq))
    
    return A_eq, b_eq, c

def main(dd, nsub, ndim):
    ## Setup
    margMat = read_pickle_files(dd, nsub, ndim)
    overallPost, distMatCell, subsetPost = calculate_distances(margMat, nsub)
    A_ub, b_ub, cvec = setup_lp_problem(distMatCell)
        
    ## Solve the LP problem
    xsol, output = call_lp_solver(A_ub, b_ub, cvec, bounds=(0, 1), solver='highs-ds')
    print("Solution status:", output['status'])
    print("Message:", output['message'])
    print("Solve time:", output['solve_time'])

    runtime = output['solve_time']
    return xsol, output, overallPost, subsetPost, cvec