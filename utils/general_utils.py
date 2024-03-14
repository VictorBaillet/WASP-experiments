import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

import pickle
import os
    
def process_results(config_file, proj, npart, data_path='data'):
    ## Define the projection function of the parameters
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
    ## Setup paths
    config_path = os.path.join('config', config_file)
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader) 
    
    folder_path = os.path.join(data_path, config['general']['folder_name'])
    input_folder = os.path.join(folder_path, 'input')
    sampling_folder = os.path.join(folder_path, 'sub-sampling')
    results_folder = os.path.join(folder_path, 'results')
    file_path_sol = os.path.join(results_folder, f'xsol_k{npart}.pkl')
    file_path_support = os.path.join(results_folder, f'support_k{npart}.pkl')

    ## Load data
    with open(file_path_sol, 'rb') as file:
        xsol = pickle.load(file)
    with open(file_path_support, 'rb') as file:
        overallPost = pickle.load(file)


    filename = os.path.join(sampling_folder, f"full_dataset.pkl")

    with open(filename, 'rb') as file:
        full_data_sampling = pickle.load(file)

    x_sub_sampling = []
    for i in range(5):
        filename = os.path.join(sampling_folder, f"sample_{i}.pkl")
        with open(filename, 'rb') as file:
            x_sub_sampling.append(pickle.load(file))

    ## Plot results
    # First plot
    plt.figure(figsize=(8, 6))
    for sub_sampling in x_sub_sampling:
        f_m = f(sub_sampling)
        plt.scatter(f_m[:, 0], f_m[:, 1], c='#b4deb8')
    
    f_m = f(full_data_sampling)
    plt.scatter(f_m[:, 0], f_m[:, 1], c='#ffaba9')
    
    subset_patch = mpatches.Patch(color='#b4deb8', label='Subsets')
    full_data_patch = mpatches.Patch(color='#ffaba9', label='Full dataset')
    plt.legend(handles=[subset_patch, full_data_patch])
    plt.title('Samples inferred from the full dataset vs subsets') 
    plt.xlabel('$\sigma_1$')
    plt.ylabel('$\sigma_2$')
    plt.plot()

    # Second plot 
    plt.figure(figsize=(8, 6))
    plt.scatter(overallPost[:, 0], overallPost[:, 1], c=xsol[:30*30])
    plt.title('Barycenter distribution and its support') 
    plt.xlabel('$\sigma_1$')
    plt.ylabel('$\sigma_2$')
    plt.plot()
    
    # Third plot
    plt.figure(figsize=(8, 6))
    
    f_m = f(full_data_sampling)
    x = f_m[:, 0]
    y = f_m[:, 1]
    xx, yy, zz = smooth_equidensity_line(x, y)
    plt.contour(xx, yy, zz, levels=5, colors='#ffaba9')  # contour lines
    
    for sub_sampling in x_sub_sampling:
        f_m = f(sub_sampling)
        x = f_m[:, 0]
        y = f_m[:, 1]
        xx, yy, zz = smooth_equidensity_line(x, y)
        plt.contour(xx, yy, zz, levels=5, colors='#b4deb8')
    
    x, y = overallPost[:, 0], overallPost[:, 1]
    z = xsol[:30*30]  # Your density or value for each point
    
    # Create grid
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate z values on this grid
    zi = griddata((x, y), z, (xi, yi), method='linear')
    zi = gaussian_filter(zi, sigma=2) + 10e-8
    
    # Plot
    plt.contour(xi, yi, zi, levels=5, colors='#9db0d9')  # Adjust levels as needed
    
    subset_patch = mpatches.Patch(color='#b4deb8', label='Subsets')
    full_data_patch = mpatches.Patch(color='#ffaba9', label='Full dataset')
    barycenter_patch = mpatches.Patch(color='#9db0d9', label='WASP')
    plt.legend(handles=[subset_patch, full_data_patch, barycenter_patch])
    
    plt.title('Equidensity Levels')
    plt.xlabel('$\sigma_1$')
    plt.ylabel('$\sigma_2$')
    plt.plot();

    output = {'folder path': folder_path, 'input folder': input_folder, 'sampling folder': sampling_folder,
              'results folder': results_folder, 'file path solution': file_path_sol, 'file path support': file_path_support,
              'solution': xsol, 'overallPost': overallPost, 'full data sampling': full_data_sampling, 'subsampling': x_sub_sampling,
              'projection': f}
    return output
        
def smooth_equidensity_line(x, y):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    
    # Create a grid to evaluate KDE
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    grid_points = np.vstack([xx.ravel(), yy.ravel()])
    zz = gaussian_kde(xy)(grid_points).reshape(xx.shape)

    return xx, yy, zz

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