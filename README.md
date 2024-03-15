# WASP Experiments Repository

This repository contains code and experiments related to bayesian inferrence for tall datasets. It builds upon the methodology introduced in "Scalable Bayes via Barycenter in Wasserstein Space" by Srivastava et al. in 2017.

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

- **Datasets**: Stored in the `data` directory, our experimental datasets are organized into folders. Each folder has three subdirectories: `input` for the raw data and observations, `sub-sampling` for posterior samples from subsets and the entire dataset, and `result` for the posterior barycenters.

- **Experiments**: Defined in the `config` directory through configuration files, our experiments are executed via `main.py` script in the command line.

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

3. Run an experiment with :
   ```bash
   python3 main.py --config_file experience_1.yaml --parameter_projector rho_1 --generate_data True
   ```
Refer to the [Experiment Configurations](#experiment-configurations) section for more details.
## Dataset Descriptions

In this work we experiment with two type of datasets :

- **Gaussian mixtures**
- **Logistic regression**

Refer to the config files for the parameters used. 

## Experiment Configurations

The `config` folder contains the config files, defining the number of observations, distribution parameters, and the total number of experiments. To start an experiment, run the following:
```bash
   python3 main.py --config_file experience_1.yaml --parameter_projector rho_1 --generate_data True
```
Where config_file is the name of the file in the folder `config`, parameter_projector is the function f (see paper) and generate_data allows to skip the data generation and sampling once it has been done.

## Results

Our findings reaffirm the original paper's conclusions and further explore some of its limitations.





