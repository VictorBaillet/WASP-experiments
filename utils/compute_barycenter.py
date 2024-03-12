import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import block_diag, vstack, csr_matrix
from scipy.optimize import linprog
import time

import os

def call_lp_solver(A_ub, b_ub, cvec, maxiters=1000, tolx=1e-4, solver='highs'):
    """
    Solve an LP problem using a specified solver.

    Parameters:
    - A_ub: Coefficient matrix for inequality constraints (upper bound).
    - b_ub: Right-hand side vector for inequality constraints.
    - cvec: Coefficients of the linear objective function to be minimized.
    - maxiters: Maximum number of iterations (applicable to certain solvers).
    - tolx: Tolerance for termination (not directly used here, shown for compatibility).
    - solver: The LP solver to use. Options include 'highs', 'interior-point', 'revised simplex', etc.

    Returns:
    - xsol: Solution vector x.
    - output: A dictionary containing output information such as solve time.
    """

    # Solve the LP problem
    start_time = time.time()
    res = linprog(cvec, A_ub=A_ub, b_ub=b_ub, method=solver, options={'maxiter': maxiters, 'disp': False})
    end_time = time.time()

    # Prepare output
    xsol = res.x if res.success else None
    output = {
        'status': res.status,
        'message': res.message,
        'solve_time': end_time - start_time
    }

    return xsol, output

def read_csv_files(dd, nsub, ndim, base_path='data'):
    """Read CSV files corresponding to the data."""
    chr = 'mu1' if ndim == 1 else 'mu2'
    margMat = []
    for kk in range(1, nsub + 1):
        file_name = f'samp_cv_{dd}_nsub_{kk}_k10_{chr}.csv'
        margMat.append(pd.read_csv(os.path.join(base_path, file_name), header=None).to_numpy())
    return margMat

def calculate_distances(margMat, nsub):
    """Calculate the pairwise squared Euclidean distances."""
    subsetPost = [m[np.random.randint(0, len(m), 100), :] for m in margMat]
    lbd1, ubd1 = min(m[:, 0].min() for m in subsetPost), max(m[:, 0].max() for m in subsetPost)
    lbd2, ubd2 = min(m[:, 1].min() for m in subsetPost), max(m[:, 1].max() for m in subsetPost)

    overallPost = np.array(np.meshgrid(np.linspace(lbd1, ubd1, 60), np.linspace(lbd2, ubd2, 60))).T.reshape(-1, 2)
    distMatCell = [cdist(overallPost, sp, 'sqeuclidean') for sp in subsetPost]
    return overallPost, distMatCell

def setup_lp_problem(distMatCell, overallPost):
    """Set up the LP problem matrices."""
    N = overallPost.shape[0]
    K = len(distMatCell)
    A_ub = block_diag([-np.ones((1, N)) for _ in range(K)]).toarray()
    b_ub = np.array([-1] * K)
    cvec = np.concatenate([[0] * N, np.hstack(distMatCell).flatten()])
    return A_ub, b_ub, cvec

def main(dd, nsub, ndim):
    margMat = read_csv_files(dd, nsub, ndim)
    overallPost, distMatCell = calculate_distances(margMat, nsub)
    A_ub, b_ub, cvec = setup_lp_problem(distMatCell, overallPost)

    # Solve the LP problem
    xsol, output = call_lp_solver(A_ub, b_ub, cvec, solver='highs')
    print("Solution status:", output['status'])
    print("Message:", output['message'])
    print("Solve time:", output['solve_time'])

    # Save results - Adjust these lines according to how you want to save your outputs
    runtime = output['solve_time']
    # Add saving mechanism as needed
    return output