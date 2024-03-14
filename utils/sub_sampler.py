import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pymc as pm
from tqdm import tqdm

def mvn_wasp_mix(data_mat, ncomp=2, nrep=10, niter=10000, nburn=5000, nthin=5):
    nobs, ndim = data_mat.shape
    alpha = np.repeat(1. / ncomp, ncomp)

    # Initial sampling of mixture probabilities and component assignments
    probs = np.random.dirichlet(alpha)
    z_mat = np.random.multinomial(1, probs, size=nobs)
    
    # Initialize matrices and arrays for densities, means, and covariances
    dens_mat = np.zeros((nobs, ncomp))
    mu_mat = np.zeros((ncomp, ndim))
    sig_arr = np.array([np.eye(ndim) for _ in range(ncomp)])
    
    # Hyperparameters for the prior distributions
    kap0 = 0.01
    m0 = np.zeros(ndim)
    nu0 = 2
    s0 = 2 * nu0 * np.eye(ndim)
    
    # Initialize storage for sampled probabilities, means, and covariances
    sample_shape = ((niter - nburn) // nthin, ncomp, ndim, ndim)
    probs_samp = np.zeros(((niter - nburn) // nthin, ncomp))
    mu_mat_samp = np.zeros(sample_shape[:3])
    sig_mat_samp = np.zeros(sample_shape)
    
    cts = 0  # Counter for the number of samples stored

    for ii in tqdm(range(niter + 1)):
        # Update component assignments and calculate number of observations per component
        ns = np.array([np.sum(z_mat[:, j]) for j in range(ncomp)])
        
        # Sample new mixture probabilities
        probs = np.random.dirichlet(ns * nrep + alpha)

        # Update means and covariances for each component
        for jj in range(ncomp):
            idx = np.where(z_mat[:, jj] == 1)[0]
            if len(idx) > 0:
                dat_mean = np.mean(data_mat[idx, :], axis=0)
                
                # Update component mean
                mu_cov = sig_arr[jj] / (kap0 + ns[jj] * nrep)
                mu_mean = (kap0 * m0 + ns[jj] * nrep * dat_mean) / (kap0 + ns[jj] * nrep)
                mu_mat[jj] = np.random.multivariate_normal(mu_mean, mu_cov)
                
                # Update component covariance
                mat1 = (kap0 * ns[jj] * nrep / (kap0 + ns[jj] * nrep)) * np.outer(dat_mean - m0, dat_mean - m0)
                cent_mat = data_mat[idx, :] - dat_mean
                mat2 = np.dot(cent_mat.T, cent_mat) * nrep
                cov_scl_mat = mat1 + mat2 + s0
                cov_df = ns[jj] * nrep + nu0 + 1
                sig_arr[jj] = stats.invwishart.rvs(cov_df, cov_scl_mat)
                
                # Update density matrix
                dens_mat[:, jj] = stats.multivariate_normal.logpdf(data_mat, mu_mat[jj], sig_arr[jj])
        
        # Update component assignment probabilities for each observation
        lprobs = dens_mat + np.log(probs)
        eprobs = np.exp(lprobs - lprobs.max(axis=1, keepdims=True))
        eprobs /= eprobs.sum(axis=1, keepdims=True)

        # Resample component assignments
        z_mat = np.array([np.random.multinomial(1, eprobs[i, :]) for i in range(nobs)])

        # Store samples after burn-in and according to thinning interval
        if ii > nburn and (ii % nthin == 0):
            cts_idx = (ii - nburn) // nthin - 1
            probs_samp[cts_idx] = probs
            mu_mat_samp[cts_idx] = mu_mat
            sig_mat_samp[cts_idx] = sig_arr

    return {
        'mu': mu_mat_samp,
        'cov': sig_mat_samp,
        'prob': probs_samp,
    }

def wasp_logistic(data_mat, ncomp=2, nrep=10, niter=10000, nburn=5000, nthin=5):
    X = data_mat[:, :2]
    y = data_mat[:, 2:].T[0]

    with pm.Model() as logistic_model:
        # Priors
        weights = pm.Normal('weights', mu=0, sigma=10, shape=X.shape[1])
        bias = pm.Normal('bias', mu=0, sigma=10)
        
        # Linear model and logistic function
        logits = pm.math.dot(X, weights) + bias
        p = pm.math.sigmoid(logits)
        
        bernoulli_rv = pm.Bernoulli.dist(p=p)
        observed = pm.Potential('observed', nrep * pm.logp(bernoulli_rv, y))
    
    with logistic_model:
        trace = pm.sample(niter, tune=nburn, target_accept=0.95, chains=1)
    
    return {
        'weights': trace.posterior['weights'].values[0],
        'bias': trace.posterior['bias'].values[0],
    }
    