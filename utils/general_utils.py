import numpy as np

import pickle
import os
    

def read_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def write_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def cov2cor(covariance_matrix):
    v = np.sqrt(np.diag(covariance_matrix))
    outer_v = np.outer(v, v)
    correlation = covariance_matrix / outer_v
    correlation[covariance_matrix == 0] = 0
    return correlation

def process_data(mtd, id, base_path="/Shared/ssrivastva/wasp/mixtures/result"):
    cvs = np.tile(np.arange(1, 11), 2)
    subs = np.repeat(np.arange(1, 3), 10)
    cid = cvs[id]
    sid = subs[id]

    subdensRho = None
    tmp = []

    if sid == 1:
        nsub = 5
        subdensRho = np.zeros((2, 1000, nsub))
    else:
        nsub = 10
        subdensRho = np.zeros((2, 1000, nsub))

    for kk in range(1, nsub + 1):
        sub_folder = f"sub{nsub}"
        fname = os.path.join(base_path, f"comp/{sub_folder}/comp_cv_{cid}_nsub_{kk}_k{sid*5}.pkl")
        samp = read_pickle(fname)
        ppp = np.mean(samp['prob'], axis=0)

        for ii in range(1000):
            cov_matrix = samp['cov'][np.argmin(ppp), :, :, ii] if ppp[0] < ppp[1] else samp['cov'][np.argmax(ppp), :, :, ii]
            subdensRho[:, ii, kk-1] = np.diag(cov2cor(cov_matrix))

        tmp.append(samp['time'])

    # Placeholder for consensusMCindep and semiparamDPE equivalent
    scottDens, xingDens = np.transpose(subdensRho), "Not Implemented"

    fname1 = os.path.join(base_path, f"cons/cons_rho_cv_{cid}_k{sid*5}.pkl")
    fname2 = os.path.join(base_path, f"xing/xing_rho_cv_{cid}_k{sid*5}.pkl")

    write_pickle({'dens': scottDens, 'time': np.mean(tmp)}, fname1)
    write_pickle({'dens': xingDens, 'time': np.mean(tmp)}, fname2)