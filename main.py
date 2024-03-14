import numpy as np
import yaml
import argparse

import pickle
import os

from utils.generate_data import *
from utils.sub_sampler import *
from utils.compute_barycenter import *
from utils.general_utils import *

def run_experiment(config, parameter_projector, generate_data=True, options=None, data_path="data"):
    print("Running experiment : ", config['general']['name'])

    ## Setup
    np.random.seed(config['general']['seed'])
    folder_path = os.path.join(data_path, config['general']['folder_name'])
    input_folder = os.path.join(folder_path, 'input')
    sampling_folder = os.path.join(folder_path, 'sub-sampling')
    results_folder = os.path.join(folder_path, 'results')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        os.makedirs(input_folder)
        os.makedirs(sampling_folder)
        os.makedirs(results_folder)

    if generate_data:
        ## Generate data
        if config['model']['type'] == "Gaussian mixture":
            labels = config['model']['labels']
            mu_list = {}
            sig_mat_list = {}
            for label in labels:
                param = config['model'][f'gaussian_{label}']
                mu_list[label] = param['mu']
                sig_mat_list[label] = param['sigma']
                
            nrep = config['n_rep']
            reps = []
            nobs = config['n_observations']
            for r in range(nrep):
                data, clusters = gen_data(nobs=nobs)
                reps.append({'data': data, 'clusters': clusters})
        if config['model']['type'] == "Logistic":
            weights = config['model']['weights']
            bias = config['model']['bias']
            noise_std = config['model']['noise_std']

            nrep = config['n_rep']
            reps = []
            nobs = config['n_observations']
            for r in range(nrep):
                data = gen_data_logistic(weights=weights, bias=bias, nobs=nobs, noise_std=noise_std)
                reps.append({'data': data})
    
        ## Save data and create subsets
        pd.to_pickle(reps, os.path.join(input_folder, "full_data.pkl"))
    
        nclust = config['subsets']['n_clusters']
        for npart in config['subsets']['n_subset']:
            parts = partition_data(reps, npart, nclust)
            pd.to_pickle(parts, os.path.join(input_folder, f"subset_k{npart}.pkl"))
    
        if config['model']['type'] == "Gaussian mixture":
            sampling = lambda x: mvn_wasp_mix(x)
        elif config['model']['type'] == "Logistic":
            sampling = lambda x: wasp_logistic(x)
        else:
            raise ValueError("Unsupported model type: {}".format(config['model']['type']))
        
        full_data_sampling = sampling(np.array(reps[0]['data'])) ## Full data
        ## Sampling
        # Full data
        full_data_sampling = mvn_wasp_mix(np.array(reps[0]['data']), nrep=1)
        filename = os.path.join(sampling_folder, f"full_dataset.pkl")
        with open(filename, 'wb') as file: 
            pickle.dump(full_data_sampling, file)
        
        # Subsets
        for npart in config['subsets']['n_subset']:
            part_path = os.path.join(input_folder, f"subset_k{npart}.pkl")
            # Read the saved file
            parts = pd.read_pickle(part_path)
            x_res = []
            for i, cluster in enumerate(parts[0][0]):
                cluster = np.array(cluster)
                x_res.append(mvn_wasp_mix(cluster, nrep=npart))
    
        ## Save sampling
        for i, res in enumerate(x_res):
            filename = os.path.join(sampling_folder, f"sample_{i}.pkl")
            with open(filename, 'wb') as file: 
                pickle.dump(res, file)
    
    ## Compute barycenter and save results
    for npart in config['subsets']['n_subset']:
        xsol, output, overallPost, subsetPost, cvec = main(exp_name=config['general']['folder_name'], nsub=5, f=parameter_projector)
        file_path_sol = os.path.join(results_folder, f'xsol_k{npart}.pkl')
        file_path_support = os.path.join(results_folder, f'support_k{npart}.pkl')
        with open(file_path_sol, "wb") as file:
            pickle.dump(xsol, file)
        with open(file_path_support, "wb") as file:
            pickle.dump(overallPost, file)

    ## Compute f over the full dataset and over the subsets
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=('Scalable Bayes via Barycenter in Wasserstein Space'))
    parser.add_argument('--config_file', type=str,)
    parser.add_argument('--parameter_projector', type=str,)
    parser.add_argument('--generate_data', type=str,)
    args = parser.parse_args()

    config_path = os.path.join('config', args.config_file)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader) 

    proj = args.parameter_projector
    if proj == 'mean_sum':
        f = lambda m: m['mu'][:, 0].reshape(m['mu'].shape[0], -1) + m['mu'][:, 1].reshape(m['mu'].shape[0], -1)

    if proj == 'rho_1':
        def f(m):
            eps = 10e-3
            # Extract the elements of the covariance matrices
            sigma_0_0 = m['cov'][:, 0, 0, 0]  # Covariance matrix 1, element (0, 0)
            sigma_0_1 = m['cov'][:, 0, 0, 1]  # Covariance matrix 1, element (0, 1) = (1, 0)
            sigma_1_1 = m['cov'][:, 0, 1, 1]  # Covariance matrix 1, element (1, 1)
            sigma_2_0 = m['cov'][:, 1, 0, 0]  # Covariance matrix 2, element (0, 0)
            sigma_2_1 = m['cov'][:, 1, 0, 1]  # Covariance matrix 2, element (0, 1) = (1, 0)
            sigma_3_1 = m['cov'][:, 1, 1, 1]  # Covariance matrix 2, element (1, 1)
        
            # Compute correlation coefficients
            corr1 = sigma_0_1 / np.sqrt(np.maximum(eps, sigma_0_0 * sigma_1_1))
            corr2 = sigma_2_1 / np.sqrt(np.maximum(eps, sigma_2_0 * sigma_3_1))
        
            # Combine the results
            return np.stack((corr1, corr2), axis=-1)

    run_experiment(config, parameter_projector=f, generate_data=(args.generate_data=="True"))

    


