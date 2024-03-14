import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state

import pickle
import os

def gen_data(mu_list={'1': np.array([1, 2]), '2': np.array([7, 8])},
             sig_mat_list={'1': np.array([[1, 0.5], [0.5, 2]]),
                          '2': np.array([[1, 0.5], [0.5, 2]])},
             probs=np.array([0.3, 0.7]),
             nobs=1000):
    data = np.zeros((nobs, 2))
    clusters = np.zeros(nobs)
    previous_idx = 0
    for i, (label, mu) in enumerate(mu_list.items()):
        size = int(nobs * probs[i])
        data[previous_idx: previous_idx + size, :] = multivariate_normal.rvs(mean=mu, cov=sig_mat_list[label], size=size)
        clusters[previous_idx:previous_idx +size] = i
        previous_idx = previous_idx + size
    return pd.DataFrame(data, columns=['X1', 'X2']), clusters

def partition_data(reps, npart, nclust, random_state=12345):
    """
    Partition the data based on k-means clustering and random assignment within clusters.

    Parameters:
    - reps: List of dictionaries, where each dictionary contains 'data': DataFrame of the data points.
    - npart: Number of partitions to create.
    - nclust: Number of clusters to use in k-means.
    - random_state: Seed for the random number generator.

    Returns:
    - A tuple of (parts, partsIdx), where `parts` is a list of lists of DataFrames representing the partitions,
      and `partsIdx` is a list of arrays representing the partition indices for each data point.
    """
    random_state = check_random_state(random_state)
    parts = []
    partsIdx = []
    
    for rep in reps:
        data = rep['data']
        kmns = KMeans(n_clusters=nclust, random_state=random_state).fit(data)
        cluster_labels = kmns.labels_
        
        part_indices = random_state.choice(range(npart), size=data.shape[0], replace=True)
        rep_parts = [data.iloc[np.where((cluster_labels == i) & (part_indices == j))[0]] for j in range(npart) for i in range(nclust)]
        
        # Organize partitions by cluster then by part
        organized_parts = [[] for _ in range(npart)]
        for idx, df in enumerate(rep_parts):
            part_idx = idx % npart
            organized_parts[part_idx].append(df)
        
        # Combine DataFrames within each part and across clusters
        combined_parts = [pd.concat(organized_parts[i], ignore_index=True) for i in range(npart)]
        
        parts.append(combined_parts)
        partsIdx.append(part_indices)
    
    return parts, partsIdx