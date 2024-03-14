# WASP
Experiments built on "Scalable Bayes via Barycenter in Wasserstein Space" by Srivastava et al. 2017.

# WASP Experiments Repository

This repository contains code and experiments related to bayesian inferrence for tall datasets. The experiments focus on the method presented in the paper "Scalable Bayes via Barycenter in Wasserstein Space" by Srivastava et al. 2017.

## Table of Contents

1. [Introduction](#introduction)
2. [Contents](#contents)
3. [Usage](#usage)
4. [Dataset Descriptions](#dataset-descriptions)
5. [Experiment Configurations](#experiment-configurations)
6. [Results](#results)

## Introduction

Going through `results.ipynb` is an easy and fast way to explore our project's capabilities. 

## Contents

- **Utils**: The `utils` directory contains Python scripts with the main functions used in the experiments. It includes code for generating datasets, sampling posterior distributions and computing barycenter of distributions. 

- **Datasets**: The `data` directory contains the datasets used in the experiments. The data for each experiment is stored in a folder, each of this folder contains 3 sub-folders, `input` contains the raw data, the observations, and the subsets, `sub-sampling` contains the sample of the posterior for each subset and the whole dataset, `result` contains the barycenter of the posteriors.

- **Experiments**: Each of our experiments is parameterized by a config filer, in `config` folder, the experiments are then run in the bash using `main.py`.

- **Experiments Results** : `results.ipynb` display the results of our experiments, the plots are saved in `results`.

## Usage

To reproduce or extend the experiments, follow these steps:

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/VictorBaillet/WASP-experiments.git
   cd WASP-experiments
   ```

2. Install the required dependencies by creating a virtual environment and using `pip`:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Run an experiment with.

## Dataset Descriptions

In this work we experiment with two type of datasets :

- **Gaussian mixtures**
- **Logarithm regression**

Refer to the config files for the parameters used. 

## Experiment Configurations

The `config` folder contains the config files, defining the number of observations, the parameters of the distributions, the number of experiments to run. To launch

## Results

The experiments show the same properties as presented in the original paper, we also illustrate some limitations.



