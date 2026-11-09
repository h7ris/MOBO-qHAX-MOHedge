# Print paper citation
# q-HAX-alpha and MO-Hedge for Multi-Objective Bayesian Optimization
# To Cite: Haris Moazam Sheikh, 'Simple Extensions of Single-Objective Acquisition Functions and Hedge Strategies for Multi-Objective Bayesian Optimization', 40th Conference on Neural Information Processing Systems, 2026

#==========================================#
# Library Imports
#==========================================#

# Standard libraries
import sys
import time
import copy
import glob
import warnings
from os import remove, path
from shutil import copyfile
from typing import List, Union, Dict, Any, Tuple, Optional

# Scientific computing and plotting
import numpy as np
import scipy
from scipy.stats import qmc, norm
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from tqdm import tqdm

# PyMOO libraries
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

# PyTorch and BoTorch libraries
import torch
import botorch
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf
from botorch.exceptions.errors import CandidateGenerationError
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.pathwise.posterior_samplers import get_matheron_path_model
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning, NondominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.box_decomposition_list import BoxDecompositionList
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement, qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.parego import qLogNParEGO as BoTorchParEGO
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import qHypervolumeKnowledgeGradient
from botorch.acquisition.multi_objective.joint_entropy_search import qLowerBoundMultiObjectiveJointEntropySearch
from botorch.acquisition.multi_objective.max_value_entropy_search import qLowerBoundMultiObjectiveMaxValueEntropySearch
from botorch.acquisition.multi_objective.predictive_entropy_search import qMultiObjectivePredictiveEntropySearch
from botorch.acquisition.multi_objective.utils import sample_optimal_points, compute_sample_box_decomposition, random_search_optimizer

#==========================================#
# Multi-objective benchmark functions
#==========================================#
# Note on input/output conventions across all benchmark functions:
# 1. Input x is always in the normalized unit hypercube [0, 1]^d with shape [n_points, dim].
# 2. Each function internally rescales x from [0, 1]^d to its native physical domain [left_lim, right_lim].
# 3. args[0][i] specifies the zero-mean Gaussian observation noise std for objective i.
# 4. All functions return objectives negated as needed so that all objectives are maximized.

def TestZDT2(x, args, m=2):
    # Test function: ZDT2 (Zitzler et al., 2000)
    # Input x: [n_points, dim] in [0, 1]^d. Native domain: [0, 1]^d (Global Pareto set: x_1 in [0, 1], x_2:d = 0)
    dim = x.shape[1]
    n_points = x.shape[0]

    # Map x from [0, 1]^d to native bounds
    left_lim = np.zeros(dim)
    right_lim = np.ones(dim)
    x = normalize(x, left_lim, right_lim, flag=1)

    # Cost function array initialization
    y = np.zeros((n_points, 2))

    # Evaluate objectives
    for j in range(n_points):
        y[j, 0] = x[j, 0]
        gx = 1 + (9 / (dim - 1)) * np.sum(np.squeeze(x[j, 1:]))
        y[j, 1] = np.squeeze(gx * (1 - (y[j, 0] / gx)**2))

    # Add observation noise
    for i in range(m):
        noise = np.random.normal(loc=0.0, scale=args[0][i], size=(x.shape[0]))
        y[:, i] = y[:, i] + noise

    # Negate for maximization
    return -1 * y


def TestZDT4(x, args, m=2):
    # Test function: ZDT4 (Zitzler et al., 2000)
    # Input x: [n_points, dim] in [0, 1]^d. Native domain: x_1 in [0, 1], x_2:d in [-5, 5] (Global Pareto set: x_1 in [0, 1], x_2:d = 0)
    dim = x.shape[1]
    n_points = x.shape[0]

    # Map x from [0, 1]^d to native bounds
    left_lim = -5 * np.ones(dim)
    right_lim = 5 * np.ones(dim)
    left_lim[0] = 0
    right_lim[0] = 1
    x = normalize(x, left_lim, right_lim, flag=1)

    # Cost function array initialization
    y = np.zeros((n_points, 2))

    # Evaluate objectives
    for j in range(n_points):
        y[j, 0] = x[j, 0]
        gx = 1 + 10 * (dim - 1) + np.sum(np.squeeze((x[j, 1:]**2) - (10 * np.cos(4 * np.pi * x[j, 1:]))))
        y[j, 1] = np.squeeze(gx * (1 - (y[j, 0] / gx)**0.5))

    # Add observation noise
    for i in range(m):
        noise = np.random.normal(loc=0.0, scale=args[0][i], size=(x.shape[0]))
        y[:, i] = y[:, i] + noise

    # Negate for maximization
    return -1 * y


def TestZDT6(x, args, m=2):
    # Test function: ZDT6 (Zitzler et al., 2000)
    # Input x: [n_points, dim] in [0, 1]^d. Native domain: [0, 1]^d (Global Pareto set: x_1 in [0, 1], x_2:d = 0)
    dim = x.shape[1]
    n_points = x.shape[0]

    # Map x from [0, 1]^d to native bounds
    left_lim = np.zeros(dim)
    right_lim = np.ones(dim)
    x = normalize(x, left_lim, right_lim, flag=1)

    # Cost function array initialization
    y = np.zeros((n_points, 2))

    # Evaluate objectives
    for j in range(n_points):
        y[j, 0] = 1 - (np.exp(-4 * x[j, 0]) * (np.sin(6 * np.pi * x[j, 0]))**6)
        gx = 1 + 9 * (np.sum(np.squeeze(x[j, 1:])) / (dim - 1))**0.25
        y[j, 1] = np.squeeze(gx * (1 - (y[j, 0] / gx)**2))

    # Add observation noise
    for i in range(m):
        noise = np.random.normal(loc=0.0, scale=args[0][i], size=(x.shape[0]))
        y[:, i] = y[:, i] + noise

    # Negate for maximization
    return -1 * y


def TestDTLZ1(x, args, m=3):
    # Test function: DTLZ1 (Deb et al., 2002)
    # Input x: [n_points, dim] in [0, 1]^d. Native domain: [0, 1]^d (Global Pareto set: x_i = 0.5 for i >= m)
    dim = x.shape[1]
    n_points = x.shape[0]

    # Map x from [0, 1]^d to native bounds
    left_lim = np.zeros(dim)
    right_lim = np.ones(dim)
    x = normalize(x, left_lim, right_lim, flag=1)

    # Cost function array initialization
    y = np.zeros((n_points, m))

    # Evaluate objectives (first m-1 variables control position; remaining k variables control distance gx)
    for j in range(n_points):
        k = dim - m + 1
        x_m = x[j, m-1:]
        gx = 100 * (k + np.sum((x_m - 0.5)**2 - np.cos(20 * np.pi * (x_m - 0.5))))

        for i in range(m):
            if i == 0:
                prod = np.prod(x[j, 0:m-1])
            else:
                prod = np.prod(x[j, 0:m-1-i]) * (1 - x[j, m-1-i])
            y[j, i] = 0.5 * (1 + gx) * prod

    # Add observation noise
    for i in range(m):
        noise = np.random.normal(loc=0.0, scale=args[0][i], size=(x.shape[0]))
        y[:, i] = y[:, i] + noise

    # Negate for maximization
    return -1 * y


def TestDTLZ2(x, args, m=3):
    # Test function: DTLZ2 (Deb et al., 2002)
    # Input x: [n_points, dim] in [0, 1]^d. Native domain: [0, 1]^d (Global Pareto set: x_i = 0.5 for i >= m)
    dim = x.shape[1]
    n_points = x.shape[0]

    # Map x from [0, 1]^d to native bounds
    left_lim = np.zeros(dim)
    right_lim = np.ones(dim)
    x = normalize(x, left_lim, right_lim, flag=1)

    # Cost function array initialization
    y = np.zeros((n_points, m))

    # Evaluate objectives (spherical Pareto front with unimodal distance gx)
    for j in range(n_points):
        x_m = x[j, m-1:]
        gx = np.sum((x_m - 0.5)**2)

        for i in range(m):
            n_cos = m - 1 - i
            val = (1 + gx)
            if n_cos > 0:
                val *= np.prod(np.cos(x[j, :n_cos] * np.pi / 2))
            if i > 0:
                val *= np.sin(x[j, n_cos] * np.pi / 2)
            y[j, i] = val

    # Add observation noise
    for i in range(m):
        noise = np.random.normal(loc=0.0, scale=args[0][i], size=(x.shape[0]))
        y[:, i] = y[:, i] + noise

    # Negate for maximization
    return -1 * y


def TestDTLZ3(x, args, m=3):
    # Test function: DTLZ3 (Deb et al., 2002)
    # Input x: [n_points, dim] in [0, 1]^d. Native domain: [0, 1]^d (Global Pareto set: x_i = 0.5 for i >= m)
    dim = x.shape[1]
    n_points = x.shape[0]

    # Map x from [0, 1]^d to native bounds
    left_lim = np.zeros(dim)
    right_lim = np.ones(dim)
    x = normalize(x, left_lim, right_lim, flag=1)

    # Cost function array initialization
    y = np.zeros((n_points, m))

    # Evaluate objectives (spherical Pareto front with multimodal distance gx)
    for j in range(n_points):
        k = dim - m + 1
        x_m = x[j, m-1:]
        gx = 100 * (k + np.sum((x_m - 0.5)**2 - np.cos(20 * np.pi * (x_m - 0.5))))

        for i in range(m):
            n_cos = m - 1 - i
            val = (1 + gx)
            if n_cos > 0:
                val *= np.prod(np.cos(x[j, :n_cos] * np.pi / 2))
            if i > 0:
                val *= np.sin(x[j, n_cos] * np.pi / 2)
            y[j, i] = val

    # Add observation noise
    for i in range(m):
        noise = np.random.normal(loc=0.0, scale=args[0][i], size=(x.shape[0]))
        y[:, i] = y[:, i] + noise

    # Negate for maximization
    return -1 * y

def TestDTLZ7(x, args, m=3):
    # Test function: DTLZ7 (Deb et al., 2002)
    # Input x: [n_points, dim] in [0, 1]^d. Native domain: [0, 1]^d (Global Pareto set: x_i = 0 for i >= m)
    dim = x.shape[1]
    n_points = x.shape[0]
    left_lim = np.zeros(dim)
    right_lim = np.ones(dim)
    x = normalize(x, left_lim, right_lim, flag=1)
    y = np.zeros((n_points, m))
    for j in range(n_points):
        k = dim - m + 1
        x_m = x[j, m-1:]
        gx = 1 + (9 / k) * np.sum(x_m)
        y[j, :m-1] = x[j, :m-1]
        h = m - np.sum((y[j, :m-1] / (1 + gx)) * (1 + np.sin(3 * np.pi * y[j, :m-1])))
        y[j, m-1] = (1 + gx) * h
    for i in range(m):
        noise = np.random.normal(loc=0.0, scale=args[0][i], size=(x.shape[0]))
        y[:, i] = y[:, i] + noise
    return -1 * y

def TestBraninCurrin(x, args, m=2):
    # Test function: Branin-Currin (Balandat et al., 2020)
    # Input x: [n_points, 2] in [0, 1]^2 (rescaled internally to Branin/Currin domains)
    dim = x.shape[1]
    n_points = x.shape[0]

    # Map x from [0, 1]^2
    left_lim = np.zeros(dim)
    right_lim = np.ones(dim)
    x = normalize(x, left_lim, right_lim, flag=1)

    # Cost function array initialization
    y = np.zeros((n_points, 2))

    # Evaluate objectives
    for j in range(n_points):
        x0 = x[j, 0]
        x1 = x[j, 1]

        # Objective 1: Branin (rescaled to [-5, 10] x [0, 15])
        y[j, 0] = ((15 * x1 - 5.1 * (15 * x0 - 5)**2 / (4 * np.pi**2) + 5 * (15 * x0 - 5) / np.pi - 6)**2 + (10 - 10 / (8 * np.pi)) * np.cos(15 * x0 - 5) + 10)

        # Objective 2: Currin (x1 clipped away from 0 for numerical stability)
        x1_safe = max(x1, 1e-9)
        y[j, 1] = ((1 - np.exp(-1 / (2 * x1_safe))) * (2300 * x0**3 + 1900 * x0**2 + 2092 * x0 + 60) / (100 * x0**3 + 500 * x0**2 + 4 * x0 + 20))

    # Add observation noise
    for i in range(m):
        noise = np.random.normal(loc=0.0, scale=args[0][i], size=(x.shape[0]))
        y[:, i] = y[:, i] + noise

    # Negate for maximization
    return -1 * y


def TestVehicleCrash(x, args, m=3):
    # Test function: Vehicle Crash Safety (Liao et al., 2008; Tanabe & Ishibuchi, 2020)
    # Input x: [n_points, 5] in [0, 1]^5. Native domain: [1, 3]^5 (mapped internally)
    dim = x.shape[1]
    n_points = x.shape[0]

    # Map x from [0, 1]^5 to native bounds [1, 3]^5
    left_lim = np.ones(dim) * 1.0
    right_lim = np.ones(dim) * 3.0
    x = normalize(x, left_lim, right_lim, flag=1)

    # Cost function array initialization
    y = np.zeros((n_points, 3))

    # Evaluate objectives
    for j in range(n_points):
        x1, x2, x3, x4, x5 = x[j, 0], x[j, 1], x[j, 2], x[j, 3], x[j, 4]

        # Objective 1: Vehicle mass
        y[j, 0] = (1640.2823 + 2.3573285 * x1 + 2.3220035 * x2 + 4.5688768 * x3 + 7.7213635 * x4 + 4.4559504 * x5)

        # Objective 2: Acceleration integration (crash severity)
        y[j, 1] = (6.5856075 - 1.15 * x1 - 1.0427 * x2 + 0.9738 * x3 + 0.1365 * x4 - 1.05041 * x1 * x2 + 0.06792 * x1 * x3 + 0.54246 * x1 * x4 + 1.0357 * x2 * x3 + 0.8456 * x2 * x4 + 1.4127 * x3 * x4)

        # Objective 3: Toe-board intrusion
        y[j, 2] = (0.21886928 + 0.00569696 * x1 * x4 + 0.01086522 * x2 * x3 + 0.0128098 * x3 * x4)

    # Add observation noise
    for i in range(m):
        noise = np.random.normal(loc=0.0, scale=args[0][i], size=(x.shape[0]))
        y[:, i] = y[:, i] + noise

    # Negate for maximization
    return -1 * y


def TestWeldedBeam(x, args, m=2):
    # Test function: Unconstrained Welded Beam (Coello Coello, 2002; Blank & Deb, 2020)
    # Input x: [n_points, 4] in [0, 1]^4. Native domain: h in [0.125, 2], l in [0.1, 10], t in [0.1, 10], b in [0.1, 2]
    dim = x.shape[1]
    n_points = x.shape[0]

    # Map x from [0, 1]^4 to native physical bounds
    left_lim = np.array([0.125, 0.1, 0.1, 0.1])
    right_lim = np.array([2.0, 10.0, 10.0, 2.0])
    x = normalize(x, left_lim, right_lim, flag=1)

    # Cost function array initialization
    y = np.zeros((n_points, 2))

    # Evaluate objectives
    for j in range(n_points):
        x1, x2, x3, x4 = x[j, 0], x[j, 1], x[j, 2], x[j, 3]

        # Objective 1: Fabrication cost
        y[j, 0] = 1.10471 * x1**2 * x2 + 0.04811 * x3 * x4 * (14.0 + x2)

        # Objective 2: Beam end deflection
        y[j, 1] = 2.1952 / (x3**3 * x4)

    # Add observation noise
    for i in range(m):
        noise = np.random.normal(loc=0.0, scale=args[0][i], size=(x.shape[0]))
        y[:, i] = y[:, i] + noise

    # Negate for maximization
    return -1 * y


def TestAirfoilDesign(x, args, m=2):
    # Test function: Airfoil Design-by-Morphing / AirDbM (Sheikh et al., 2023; Lee & Sheikh, 2026)
    # Input x: [n_points, 12] in [0, 1]^12. Native domain: morphing weights in [-1, 1]^12 (mapped internally)
    from airdbm_api import TestAirfoils

    dim = x.shape[1]
    n_points = x.shape[0]

    # Map x from [0, 1]^12 to morphing weight domain [-1, 1]^12
    left_lim = -1 * np.ones(dim)
    right_lim = np.ones(dim)
    x = normalize(x, left_lim, right_lim, flag=1)

    # Cost function array initialization
    y = np.zeros((n_points, m))

    # Evaluate airfoil geometries in XFOIL (returns (Cl/Cd)_max and stall margin for maximization)
    results = TestAirfoils(x, args={'dbm_normalization': 'ABS_SUM'})
    for i in range(n_points):
        y[i, :] = results[i].objectives

    # Add observation noise
    for i in range(m):
        noise = np.random.normal(loc=0.0, scale=args[0][i], size=(x.shape[0]))
        y[:, i] = y[:, i] + noise

    return y


def TestPD1CIFAR100(x, args, m=2):
    # Test function: PD1 CIFAR-100 Wide ResNet HPO (Wang et al., 2024)
    # Input x: [n_points, 4] in [0, 1]^4 (mapped internally via linear/log-linear transforms to hyperparameter space)
    import numpy as np
    from mfpbench.pd1.benchmark import PD1Config
    from mfpbench.pd1.benchmarks.cifar100 import PD1cifar100_wideresnet_2048

    x = np.asarray(x, dtype=float)
    n_points = x.shape[0]

    # Initialize tabular surrogate benchmark once
    if not hasattr(TestPD1CIFAR100, "_benchmark"):
        TestPD1CIFAR100._benchmark = PD1cifar100_wideresnet_2048()
    benchmark = TestPD1CIFAR100._benchmark

    # Cost function array initialization
    y = np.zeros((n_points, m), dtype=float)

    # Map [0, 1]^4 inputs to hyperparameter space and query surrogate at 199 epochs
    for j in range(n_points):
        lr_decay_factor = 0.010093 + x[j, 0] * (0.989012 - 0.010093)
        lr_initial = np.exp(np.log(0.000010) + x[j, 1] * (np.log(9.779176) - np.log(0.000010)))
        lr_power = 0.100708 + x[j, 2] * (1.999376 - 0.100708)
        opt_momentum = np.exp(np.log(0.000059) + x[j, 3] * (np.log(0.998993) - np.log(0.000059)))

        config = PD1Config(lr_decay_factor=float(lr_decay_factor), lr_initial=float(lr_initial), lr_power=float(lr_power), opt_momentum=float(opt_momentum))
        result = benchmark.query(config, at=199)

        # Negate validation error and training cost for maximization
        y[j, 0] = -float(result.error)
        y[j, 1] = -float(result.cost)

    # Add observation noise
    for i in range(m):
        noise = np.random.normal(loc=0.0, scale=args[0][i], size=n_points)
        y[:, i] = y[:, i] + noise

    return y


def TestPenicillin(x, args, m=3):
    # Test function: Penicillin Production (Liang & Lai, 2021; Balandat et al., 2020)
    # Input x: [n_points, 7] in [0, 1]^7 (mapped internally to native simulator bounds)
    from botorch.test_functions.multi_objective import Penicillin

    x = np.asarray(x, dtype=float)
    n_points = x.shape[0]

    # Initialize BoTorch simulator once
    if not hasattr(TestPenicillin, "_problem"):
        TestPenicillin._problem = Penicillin(noise_std=None, negate=False, dtype=torch.double)
    problem = TestPenicillin._problem

    # Map [0, 1]^7 inputs to native engineering bounds
    native_bounds = problem.bounds.detach().cpu().numpy()
    x_native = native_bounds[0] + x * (native_bounds[1] - native_bounds[0])
    x_torch = torch.from_numpy(x_native).to(torch.double)

    # Evaluate native simulator ([-yield, CO2, time]) and negate to maximize [yield, -CO2, -time]
    with torch.no_grad():
        y_native = problem.evaluate_true(x_torch)
    y = -y_native.detach().cpu().numpy()

    # Add observation noise
    for i in range(m):
        noise = np.random.normal(loc=0.0, scale=args[0][i], size=n_points)
        y[:, i] = y[:, i] + noise

    return y
    
#==========================================#
# Utility and plotting functions
#==========================================#

def init_sampling(initial_sample, n_states_qual, n_states_quant, n_cont, data_states, left_lim_cont, right_lim_cont, algo):
    # Generate initial design points across discrete (if any) and continuous dimensions
    dim = len(n_states_qual) + len(n_states_quant) + n_cont
    n_disc = len(n_states_qual) + len(n_states_quant)
    n_states = n_states_qual + n_states_quant

    # Initialize sample array
    data_init = np.zeros((initial_sample, dim), dtype=float)

    # Sample categorical and integer dimensions
    for i in range(n_disc):
        data_init[:, i] = np.random.randint(n_states[i], high=None, size=(initial_sample), dtype='l')

    # Map discrete indices to state values
    for i in range(initial_sample):
        for j in range(n_disc):
            data_init[i, j] = data_states[j][np.int64(data_init[i, j])]

    # Uniform random sampling for continuous dimensions
    if algo == 'uniform':
        data_init[:, n_disc:dim] = np.random.uniform(low=left_lim_cont, high=right_lim_cont, size=(initial_sample, n_cont))

    # Scrambled Sobol sequence sampling for continuous dimensions
    if algo == 'Sobol':
        sampler = qmc.Sobol(d=n_cont, scramble=True)
        data_init[:, n_disc:dim] = sampler.random(n=initial_sample)
        for i in range(n_cont):
            data_init[:, n_disc+i] = (data_init[:, n_disc+i] * (right_lim_cont[i] - left_lim_cont[i])) + left_lim_cont[i]

    # Scrambled Halton sequence sampling for continuous dimensions
    if algo == 'Halton':
        sampler = qmc.Halton(d=n_cont, scramble=True)
        data_init[:, n_disc:dim] = sampler.random(n=initial_sample)
        for i in range(n_cont):
            data_init[:, n_disc+i] = (data_init[:, n_disc+i] * (right_lim_cont[i] - left_lim_cont[i])) + left_lim_cont[i]

    return data_init


def file_writer(x, file):
    # Append 2D array rows to text file
    with open(file, 'ab') as f:
        for j in range(x.shape[0]):
            np.savetxt(f, x[j, :], fmt='%5f', delimiter=' ', newline=' ')
            f.write(b'\n')


def normalize(data, left_limit, right_limit, flag=0):
    # flag=0: normalize [left_limit, right_limit] -> [0, 1]; flag=1: unnormalize [0, 1] -> [left_limit, right_limit]
    left = np.array(left_limit)
    right = np.array(right_limit)
    if flag == 1:
        return (data * (right - left)) + left
    else:
        return (data - left) / (right - left)


def standardize(data, flag=0, file='Standardize.txt'):
    # flag=0: standardize objectives to zero mean and unit std, saving stats to file
    # flag=1: read most recent mean/std from file (last two rows) and revert data to original scale
    if isinstance(data, torch.Tensor):
        data = data.detach().numpy()

    if flag == 1:
        stats = np.loadtxt(file)
        mean = stats[-2, :]
        std = stats[-1, :]
        return (data * std) + mean
    else:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std < 10**-12] = 1.0
        file_writer(np.array([mean, std]), file)
        return (data - mean) / std


class MaximizeWrappedProblem(Problem):
    # Wraps a maximization benchmark function for PyMOO minimization
    def __init__(self, func, dim, m, args):
        super().__init__(n_var=dim, n_obj=m, n_constr=0, xl=np.zeros(dim), xu=np.ones(dim), elementwise_evaluation=False)
        self.func = func
        self.m = m
        self.args = args

    def _evaluate(self, X, out, *args, **kwargs):
        F_max = self.func(X, self.args, m=self.m)
        out["F"] = -F_max


def nsga2_optimize_and_pareto(spec, pop_size=200, n_gen=300, seed=1, verbose=False):
    # Approximate reference Pareto front for (func, dim, m) using NSGA-II
    func, dim, m = spec
    args = [np.zeros(m)]
    problem = MaximizeWrappedProblem(func=func, dim=dim, m=m, args=args)
    algorithm = NSGA2(pop_size=pop_size)
    termination = get_termination("n_gen", n_gen)
    res = minimize(problem, algorithm, termination, seed=seed, verbose=verbose)
    F_pareto_max = -res.F
    X_pareto = res.X
    return X_pareto, F_pareto_max, res


def non_dominated_filter(F, tol=0.0):
    # Filter non-dominated objective vectors under maximization convention
    F = np.asarray(F, dtype=float)
    F = F[np.all(np.isfinite(F), axis=1)]
    F = np.unique(F, axis=0)
    n = F.shape[0]
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if dominated[i]:
            continue
        dominates_i = np.all(F >= F[i] - tol, axis=1) & np.any(F > F[i] + tol, axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            dominated[i] = True

    return F[~dominated]


def plot_results(sample_file, time_file, test_function, dim, q, list_af, epoch_total):
    # Plot single-benchmark trajectories and runtime bar plot across list_af
    n_algorithms = len(list_af)
    af_names = ["[" + ", ".join([f.__name__ if hasattr(f, "__name__") else str(f) for f in af]) + "]" if isinstance(af, list) else (af.__name__ if hasattr(af, "__name__") else str(af)) for af in list_af]

    # Load runtime results and determine number of completed test runs
    time_raw = np.loadtxt(time_file)
    time_data = np.reshape(time_raw, (-1, n_algorithms))
    n_tests_run = time_data.shape[0]

    # Load sample trajectories and infer initial sample size and objective count
    data = np.loadtxt(sample_file)
    results_dim = data.shape[1]
    n_samples_per_run = int(data.shape[0] // (n_tests_run * n_algorithms))
    init_samples = n_samples_per_run - q * epoch_total
    nOuts_run = results_dim - dim

    arr = data.reshape(n_tests_run, n_algorithms, n_samples_per_run, results_dim)
    data = np.transpose(arr, (2, 3, 1, 0))
    x_data = data[:, :-nOuts_run, :, :]
    y_stored = data[:, -nOuts_run:, :, :]

    n_plot_epochs = epoch_total + 1
    epoch_ind = np.arange(n_plot_epochs)
    plot_std_scale = 1
    colors = plt.cm.Set1(np.linspace(0, 1, n_algorithms))

    # Check if benchmark is an application-derived problem without a known reference Pareto front
    app_benchmarks = {"TestAirfoilDesign", "TestPD1CIFAR100", "TestPenicillin"}
    is_app = test_function.__name__ in app_benchmarks

    if is_app:
        # Application-derived benchmark: compute Negative Relative HV Improvement directly from stored objectives
        hv_rel_out = np.zeros((n_plot_epochs, n_algorithms, n_tests_run))

        for l in range(n_tests_run):
            y_init_ref = np.reshape(y_stored[:init_samples, :, 0, l], (-1, nOuts_run))
            mins_init = np.min(y_init_ref, axis=0)
            ref_point = mins_init - 0.05 * np.abs(mins_init) - 0.01
            hv_ind = HV(ref_point=-1 * ref_point)

            for k in range(n_algorithms):
                y_data = np.reshape(y_stored[:, :, k, l], (-1, nOuts_run))
                hv_init = hv_ind(-1 * y_data[:init_samples, :])

                for j in range(n_plot_epochs):
                    current_y = y_data[:init_samples + j * q, :]
                    hv_curr = hv_ind(-1 * current_y)
                    hv_rel_out[j, k, l] = (hv_init - hv_curr) / hv_init if np.abs(hv_init) > 1e-12 else np.nan

        mean_hv_rel = np.nanmean(hv_rel_out, axis=2)
        std_hv_rel = np.nanstd(hv_rel_out, axis=2)

        # Plot Negative Relative HV Improvement trajectory
        plt.figure(figsize=(12, 7))
        for r, name in enumerate(af_names):
            plt.plot(epoch_ind, mean_hv_rel[:, r], color=colors[r], label=name, linewidth=2)
            plt.fill_between(epoch_ind, mean_hv_rel[:, r] - plot_std_scale * std_hv_rel[:, r], mean_hv_rel[:, r] + plot_std_scale * std_hv_rel[:, r], color=colors[r], alpha=0.15)
        plt.xlabel('Epochs After Initial Sampling', fontsize=12)
        plt.ylabel('Neg. Rel. HV Improve.', fontsize=12)
        plt.tick_params(labelsize=11)
        plt.legend(fontsize=12, loc='best', framealpha=0.9)
        plt.title('Acquisition Function Comparison - Relative Hypervolume Improvement', fontsize=14)
        plt.tight_layout()
        plt.show()

    else:
        # Synthetic/engineering benchmark: compute reference Pareto front via NSGA-II and log-normalized HV/IGD gaps
        X_pf, F_pf, _ = nsga2_optimize_and_pareto((test_function, dim, nOuts_run), pop_size=200, n_gen=500, seed=42, verbose=False)
        all_observed_y = []
        for l_tmp in range(n_tests_run):
            for k_tmp in range(n_algorithms):
                y_tmp = np.reshape(test_function(x_data[:, :, k_tmp, l_tmp], [np.zeros(nOuts_run)], nOuts_run), (-1, nOuts_run))
                all_observed_y.append(y_tmp)
        F_pf = non_dominated_filter(np.vstack([F_pf, np.vstack(all_observed_y)]))

        hv_out = np.zeros((n_plot_epochs, n_algorithms, n_tests_run))
        igd_out = np.zeros((n_plot_epochs, n_algorithms, n_tests_run))
        igd_ind = IGD(-1 * F_pf)
        igd_opt = 0.0

        for l in range(n_tests_run):
            y_init_ref = np.reshape(test_function(x_data[:init_samples, :, 0, l], [np.zeros(nOuts_run)], nOuts_run), (-1, nOuts_run))
            mins_init = np.min(y_init_ref, axis=0)
            ref_point = mins_init - 0.05 * np.abs(mins_init) - 0.01
            hv_ind = HV(ref_point=-1 * ref_point)
            hv_opt = hv_ind(-1 * F_pf) + 1e-6

            for k in range(n_algorithms):
                y_data = np.reshape(test_function(x_data[:, :, k, l], [np.zeros(nOuts_run)], nOuts_run), (-1, nOuts_run))
                hv_init = hv_ind(-1 * y_data[:init_samples, :])
                igd_init = igd_ind(-1 * y_data[:init_samples, :])
                hv_den = hv_opt - hv_init
                igd_den = igd_init - igd_opt

                for j in range(n_plot_epochs):
                    current_y = y_data[:init_samples + j * q, :]
                    hv_gap = (hv_opt - hv_ind(-1 * current_y)) / hv_den if np.abs(hv_den) > 1e-12 else np.nan
                    igd_gap = (igd_ind(-1 * current_y) - igd_opt) / igd_den if np.abs(igd_den) > 1e-12 else np.nan
                    hv_out[j, k, l] = np.log(hv_gap) if (np.isfinite(hv_gap) and hv_gap > 0) else np.nan
                    igd_out[j, k, l] = np.log(igd_gap) if (np.isfinite(igd_gap) and igd_gap > 0) else np.nan

        mean_hv = np.nanmean(hv_out, axis=2)
        std_hv = np.nanstd(hv_out, axis=2)
        mean_igd = np.nanmean(igd_out, axis=2)
        std_igd = np.nanstd(igd_out, axis=2)

        # Plot Hypervolume trajectory
        plt.figure(figsize=(12, 7))
        for r, name in enumerate(af_names):
            plt.plot(epoch_ind, mean_hv[:, r], color=colors[r], label=name, linewidth=2)
            plt.fill_between(epoch_ind, mean_hv[:, r] - plot_std_scale * std_hv[:, r], mean_hv[:, r] + plot_std_scale * std_hv[:, r], color=colors[r], alpha=0.15)
        plt.xlabel('Epochs After Initial Sampling', fontsize=12)
        plt.ylabel('Log Normalized HV Gap', fontsize=12)
        plt.tick_params(labelsize=11)
        plt.legend(fontsize=12, loc='best', framealpha=0.9)
        plt.title('Acquisition Function Comparison - Hypervolume', fontsize=14)
        plt.tight_layout()
        plt.show()

        # Plot IGD trajectory
        plt.figure(figsize=(12, 7))
        for r, name in enumerate(af_names):
            plt.plot(epoch_ind, mean_igd[:, r], color=colors[r], label=name, linewidth=2)
            plt.fill_between(epoch_ind, mean_igd[:, r] - plot_std_scale * std_igd[:, r], mean_igd[:, r] + plot_std_scale * std_igd[:, r], color=colors[r], alpha=0.15)
        plt.xlabel('Epochs After Initial Sampling', fontsize=12)
        plt.ylabel('Log Normalized IGD Gap', fontsize=12)
        plt.tick_params(labelsize=11)
        plt.legend(fontsize=12, loc='best', framealpha=0.9)
        plt.title('Acquisition Function Comparison - Inverted Generational Distance', fontsize=14)
        plt.tight_layout()
        plt.show()

    # Plot Runtime bar chart for all members of list_af
    mean_time = np.mean(time_data, axis=0)
    std_time = np.std(time_data, axis=0)
    plt.figure(figsize=(12, 5))
    for r, name in enumerate(af_names):
        plt.bar(r, mean_time[r], yerr=std_time[r], capsize=4, color=colors[r], label=name)
    plt.yscale("log")
    plt.xticks(np.arange(n_algorithms), af_names, rotation=45, ha="right")
    plt.ylabel("Runtime (s)", fontsize=12)
    plt.title("Mean Runtime per Algorithm (±1 std over tests)", fontsize=14)
    plt.legend(fontsize=10, loc='best', framealpha=0.9)
    plt.tight_layout()
    plt.show()
    
#==========================================#
# BoTorch-based acquisition functions
#==========================================#
# All acquisition functions optimize over normalized domain [0, 1]^d and return a [q, dim] candidate array.

def qEHVI(dim, model, ref_point_np, left_lim_cont, right_lim_cont, train_x, nOuts, q=1, ACQF_NUM_RESTARTS=10, ACQF_RAW_SAMPLES=500, ACQF_OPTIONS={"batch_limit": 5, "maxiter": 100}):
    # q-Expected Hypervolume Improvement (Daulton et al., 2020)
    ref_point_torch = torch.from_numpy(ref_point_np).to(train_x)
    bounds = torch.stack([torch.from_numpy(left_lim_cont), torch.from_numpy(right_lim_cont)]).to(train_x)

    # Partition non-dominated objective space using current GP posterior mean
    with torch.no_grad():
        pred = model.posterior(train_x).mean
    partitioning = FastNondominatedPartitioning(ref_point=ref_point_torch, Y=pred)

    # Jointly optimize q-batch via L-BFGS-B (sequential=False)
    qehvi_obj = qExpectedHypervolumeImprovement(model=model, ref_point=ref_point_torch, partitioning=partitioning)
    candidates, _ = optimize_acqf(acq_function=qehvi_obj, bounds=bounds, q=q, num_restarts=ACQF_NUM_RESTARTS, raw_samples=ACQF_RAW_SAMPLES, options=ACQF_OPTIONS, sequential=False)

    return np.reshape(candidates.detach().cpu().numpy(), (-1, dim))


def qNEHVI(dim, model, ref_point_np, left_lim_cont, right_lim_cont, train_x, nOuts, q=1, ACQF_NUM_RESTARTS=10, ACQF_RAW_SAMPLES=500, ACQF_OPTIONS={"batch_limit": 5, "maxiter": 100}):
    # q-Noisy Expected Hypervolume Improvement (Daulton et al., 2021)
    ref_point_torch = torch.from_numpy(ref_point_np).to(train_x)
    bounds = torch.stack([torch.from_numpy(left_lim_cont), torch.from_numpy(right_lim_cont)]).to(train_x)

    # Jointly optimize q-batch integrating over observation noise at X_baseline
    qnehvi_obj = qNoisyExpectedHypervolumeImprovement(model=model, ref_point=ref_point_torch, X_baseline=train_x, prune_baseline=True)
    candidates, _ = optimize_acqf(acq_function=qnehvi_obj, bounds=bounds, q=q, num_restarts=ACQF_NUM_RESTARTS, raw_samples=ACQF_RAW_SAMPLES, options=ACQF_OPTIONS, sequential=False)

    return np.reshape(candidates.detach().cpu().numpy(), (-1, dim))


def qLEHVI(dim, model, ref_point_np, left_lim_cont, right_lim_cont, train_x, nOuts, q=1, ACQF_NUM_RESTARTS=10, ACQF_RAW_SAMPLES=500, ACQF_OPTIONS={"batch_limit": 5, "maxiter": 100}):
    # q-Log Expected Hypervolume Improvement (Daulton et al., 2020)
    ref_point_torch = torch.from_numpy(ref_point_np).to(train_x)
    bounds = torch.stack([torch.from_numpy(left_lim_cont), torch.from_numpy(right_lim_cont)]).to(train_x)

    # Partition non-dominated objective space using current GP posterior mean
    with torch.no_grad():
        pred = model.posterior(train_x).mean
    partitioning = FastNondominatedPartitioning(ref_point=ref_point_torch, Y=pred)

    # Jointly optimize log-transformed EHVI q-batch
    qlehvi_obj = qLogExpectedHypervolumeImprovement(model=model, ref_point=ref_point_torch, partitioning=partitioning)
    candidates, _ = optimize_acqf(acq_function=qlehvi_obj, bounds=bounds, q=q, num_restarts=ACQF_NUM_RESTARTS, raw_samples=ACQF_RAW_SAMPLES, options=ACQF_OPTIONS, sequential=False)

    return np.reshape(candidates.detach().cpu().numpy(), (-1, dim))


def qLNEHVI(dim, model, ref_point_np, left_lim_cont, right_lim_cont, train_x, nOuts, q=1, ACQF_NUM_RESTARTS=10, ACQF_RAW_SAMPLES=500, ACQF_OPTIONS={"batch_limit": 5, "maxiter": 100}):
    # q-Log Noisy Expected Hypervolume Improvement (Daulton et al., 2020; Daulton et al., 2021)
    ref_point_torch = torch.from_numpy(ref_point_np).to(train_x)
    bounds = torch.stack([torch.from_numpy(left_lim_cont), torch.from_numpy(right_lim_cont)]).to(train_x)

    # Jointly optimize log-transformed noisy EHVI q-batch
    qlnehvi_obj = qLogNoisyExpectedHypervolumeImprovement(model=model, ref_point=ref_point_torch, X_baseline=train_x, prune_baseline=True)
    candidates, _ = optimize_acqf(acq_function=qlnehvi_obj, bounds=bounds, q=q, num_restarts=ACQF_NUM_RESTARTS, raw_samples=ACQF_RAW_SAMPLES, options=ACQF_OPTIONS, sequential=False)

    return np.reshape(candidates.detach().cpu().numpy(), (-1, dim))


def qLogNParEGO(dim, model, ref_point_np, left_lim_cont, right_lim_cont, train_x, nOuts, q=1, ACQF_NUM_RESTARTS=10, ACQF_RAW_SAMPLES=500, ACQF_OPTIONS={"batch_limit": 5, "maxiter": 100}):
    # q-Log Noisy ParEGO with augmented Chebyshev scalarization (Knowles, 2006; Daulton et al., 2020)
    bounds = torch.stack([torch.from_numpy(left_lim_cont), torch.from_numpy(right_lim_cont)]).to(train_x)

    # Jointly optimize scalarized log-noisy EI q-batch
    qlogparego_obj = BoTorchParEGO(model, train_x)
    candidates, _ = optimize_acqf(acq_function=qlogparego_obj, bounds=bounds, q=q, num_restarts=ACQF_NUM_RESTARTS, raw_samples=ACQF_RAW_SAMPLES, options=ACQF_OPTIONS, sequential=False)

    return np.reshape(candidates.detach().cpu().numpy(), (-1, dim))


def qHVKG(dim, model, ref_point_np, left_lim_cont, right_lim_cont, train_x, nOuts, q=1, N=500, num_fantasies=10, num_pareto=10, seed=None):
    # q-Hypervolume Knowledge Gradient one-step lookahead acquisition (Daulton et al., 2023)
    device = train_x.device
    dtype = train_x.dtype

    ref_point = torch.as_tensor(ref_point_np, device=device, dtype=dtype)
    bounds = torch.stack([torch.as_tensor(left_lim_cont, device=device, dtype=dtype), torch.as_tensor(right_lim_cont, device=device, dtype=dtype)], dim=0)

    # Fixed Sobol-QMC base samples for deterministic sample-average approximation (SAA) fantasies
    m = model.num_outputs
    if seed is None:
        seed = 0
    samplers = [SobolQMCNormalSampler(sample_shape=torch.Size([num_fantasies]), seed=seed + j) for j in range(m)]
    fantasy_sampler = ListSampler(*samplers)

    # Instantiate HVKG using posterior-mean terminal utility
    acq_func = qHypervolumeKnowledgeGradient(model=model, ref_point=ref_point, num_fantasies=num_fantasies, num_pareto=num_pareto, sampler=fantasy_sampler, objective=None, use_posterior_mean=True)
    options = {"maxiter": 100, "batch_limit": 5, "seed": seed}

    # One-shot optimization (return_full_tree=False extracts only the q design candidates)
    # Falls back to Sobol candidate search if inner one-shot initializer encounters numerical failure
    try:
        X_best, _ = optimize_acqf(acq_function=acq_func, bounds=bounds, q=q, num_restarts=1, raw_samples=N, options=options, return_full_tree=False)
    except CandidateGenerationError as e:
        q_fallback = q
        X_cand = draw_sobol_samples(bounds=bounds, n=min(2000, max(250, N)), q=q_fallback, seed=seed).squeeze(1)
        with torch.no_grad():
            vals = acq_func(X_cand.unsqueeze(1)) if q == 1 else acq_func(X_cand)
            best = torch.argmax(vals)
        X_best = X_cand[best : best + 1]

    return X_best.detach().cpu().numpy()


def qLB_MOJES(dim, model, ref_point_np, left_lim_cont, right_lim_cont, train_x, nOuts, q=3, num_pareto_samples=20, num_pareto_points=40, N_sobol=500, ACQF_NUM_RESTARTS=10, ACQF_RAW_SAMPLES=500, ACQF_OPTIONS={"batch_limit": 5, "maxiter": 100}):
    # q-Lower-Bound Multi-Objective Joint Entropy Search (Tu et al., 2022)
    model.eval()

    num_scalarizations = 8
    jes_num_samples = 1000
    estimation_type = "LB2"

    device, dtype = train_x.device, train_x.dtype
    ref_point_torch = torch.as_tensor(ref_point_np, device=device, dtype=dtype)
    bounds = torch.stack([torch.as_tensor(left_lim_cont, device=device, dtype=dtype), torch.as_tensor(right_lim_cont, device=device, dtype=dtype)], dim=0)

    pareto_sets_list, pareto_fronts_list = [], []

    # Stage 1: Sample GP posterior paths via Matheron's rule and extract approximate Pareto sets/fronts
    for _ in range(num_pareto_samples):
        path_model = get_matheron_path_model(model)
        X_candidates = []

        # Optimize random Dirichlet linear scalarizations on the sampled path
        for _ in range(num_scalarizations):
            weights = torch.distributions.Dirichlet(torch.ones(model.num_outputs, device=device, dtype=dtype)).sample()

            def scalar_objective(X, _w=weights):
                Y = path_model.posterior(X).mean

                if Y.ndim == 2:
                    Y = Y.unsqueeze(-2)
                elif Y.ndim == 4 and Y.shape[-2] == 1:
                    Y = Y.squeeze(-2)
                elif Y.ndim == 4 and Y.shape[-3] == 1:
                    Y = Y.squeeze(-3)

                if Y.ndim != 3:
                    raise RuntimeError(f"Unexpected posterior mean shape: {Y.shape}")

                val = (Y * _w).sum(dim=-1)
                out = val.sum(dim=-1)

                expected = X.shape[:-2]
                if out.shape != expected:
                    out2 = out.squeeze()
                    if out2.shape == expected:
                        out = out2
                    else:
                        raise RuntimeError("Unexpected scalar objective output shape.")
                return out

            Xopt, _ = optimize_acqf(acq_function=scalar_objective, bounds=bounds, q=1, num_restarts=ACQF_NUM_RESTARTS, raw_samples=ACQF_RAW_SAMPLES, options=ACQF_OPTIONS)
            X_candidates.append(Xopt)

        # Augment path optima with Sobol samples and filter non-dominated points
        X_candidates.append(draw_sobol_samples(bounds=bounds, n=N_sobol, q=1).squeeze(1))

        X_all = torch.cat(X_candidates, dim=0)
        Y_all = path_model.posterior(X_all).mean

        if Y_all.ndim == 3 and Y_all.shape[-2] == 1:
            Y_all = Y_all.squeeze(-2)
        if Y_all.ndim != 2:
            raise RuntimeError(f"Unexpected Y_all shape for Pareto filtering: {Y_all.shape}")

        mask = is_non_dominated(Y_all).view(-1).bool()
        Xp, Yp = X_all[mask], Y_all[mask]

        if Xp.shape[0] == 0:
            raise RuntimeError("Empty Pareto set from GP path.")

        # Pad or truncate to fixed size P = num_pareto_points for tensor stacking
        P = num_pareto_points
        if Xp.shape[0] > P:
            Xp, Yp = Xp[:P], Yp[:P]
        elif Xp.shape[0] < P:
            extra = P - Xp.shape[0]
            idx = torch.randint(0, Xp.shape[0], (extra,), device=device)
            Xp = torch.cat([Xp, Xp[idx]], dim=0)
            Yp = torch.cat([Yp, Yp[idx]], dim=0)

        pareto_sets_list.append(Xp)
        pareto_fronts_list.append(Yp)

    pareto_sets = torch.stack(pareto_sets_list, dim=0)
    pareto_fronts = torch.stack(pareto_fronts_list, dim=0)
    S, _, M = pareto_fronts.shape

    # Stage 2: Construct dominated hypercell decompositions (using BoxDecompositionList when M > 2)
    if M <= 2:
        part = DominatedPartitioning(ref_point=ref_point_torch, Y=pareto_fronts)
        hb = part.get_hypercell_bounds()
    else:
        parts = [DominatedPartitioning(ref_point=ref_point_torch, Y=pareto_fronts[i]) for i in range(S)]
        bd_list = BoxDecompositionList(*parts)
        hb = bd_list.get_hypercell_bounds()

    # Reshape hypercell bounds to [S, 2, J, M] expected by BoTorch JES
    if hb.ndim == 4:
        if hb.shape[0] == 2:
            hypercell_bounds = hb.permute(1, 0, 2, 3).contiguous()
        elif hb.shape[1] == 2:
            hypercell_bounds = hb.contiguous()
        else:
            raise RuntimeError(f"Unexpected hypercell bounds shape: {hb.shape}")
    elif hb.ndim == 3:
        if hb.shape[0] == 2:
            hypercell_bounds = hb.unsqueeze(0).contiguous()
        else:
            raise RuntimeError(f"Unexpected hypercell bounds shape: {hb.shape}")
    else:
        raise RuntimeError(f"Unexpected hypercell bounds shape: {hb.shape}")

    # Stage 3: Initialize and jointly optimize qLB-MOJES acquisition
    acqf = qLowerBoundMultiObjectiveJointEntropySearch(model=model, pareto_sets=pareto_sets, pareto_fronts=pareto_fronts, hypercell_bounds=hypercell_bounds, estimation_type=estimation_type, num_samples=jes_num_samples)
    candidates, _ = optimize_acqf(acq_function=acqf, bounds=bounds, q=q, num_restarts=ACQF_NUM_RESTARTS, raw_samples=ACQF_RAW_SAMPLES, options={**ACQF_OPTIONS, "nonnegative": False}, sequential=False)

    return np.reshape(candidates.detach().cpu().numpy(), (-1, dim))
    
#==========================================#
# qHAX-alpha acquisition functions (Sheikh, 2026)
#==========================================#
# Two-stage framework (Algorithm 1 in paper):
# Stage 1: Form vector acquisition alpha_t(x) from per-objective SO acquisitions and find Pareto set C_t^0 via NSGA-II.
# Stage 2: Greedily select q candidates from C_t^0 maximizing posterior-mean hypervolume contribution.

def qHAX_EI(dim, model, ref_point_np, left_lim_cont, right_lim_cont, train_x, nOuts, q=1, pop_size_ga=300, gen_n_ga=450):
    # qHAX-EI: Hypervolume-ranked Expected Improvement (Sheikh, 2026; Jones et al., 1998)
    # Extract posterior mean at training inputs and reference best point y_opt
    y_data = model.posterior(train_x).mean.detach().numpy()
    y_opt_ind = np.sum(y_data, axis=1)
    y_opt = y_data[np.argmax(np.squeeze(y_opt_ind)), :]

    # EI exploration parameter (xi = 0.01)
    epsilon = 0.01

    class AF_Opt(Problem):
        def __init__(self):
            super().__init__(n_var=dim, n_obj=nOuts, n_ieq_constr=0, xl=left_lim_cont, xu=right_lim_cont)

        def _evaluate(self, x, out, *args, **kwargs):
            x_torch = torch.from_numpy(x)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Evaluate GP posterior mean and std across objectives
                posterior = model.posterior(x_torch)
                mu_x = posterior.mean.detach().numpy()
                sigma_x = posterior.stddev.detach().numpy()

                # Compute analytical Expected Improvement per objective
                valid_mask = sigma_x > 1e-4
                z = np.divide((mu_x - y_opt - epsilon), sigma_x, out=np.zeros_like(mu_x), where=valid_mask)
                ei_values = (z * sigma_x) * norm.cdf(z) + sigma_x * norm.pdf(z)
                fit_score = np.where(valid_mask, ei_values, 0.0)

                # Negate acquisition values since PyMOO minimizes out["F"]
                out["F"] = -fit_score

    # Stage 1: Generate Pareto-optimal acquisition candidate pool x_pareto via NSGA-II
    problem = AF_Opt()
    algorithm = NSGA2(pop_size=pop_size_ga, eliminate_duplicates=True)
    res = minimize(problem, algorithm, ('n_gen', gen_n_ga), verbose=False)

    x_pareto = np.copy(res.X)
    candidate_af = np.zeros((q, dim))

    # Evaluate posterior mean objectives for candidate pool and set HV reference point
    # Note: PyMOO HV indicator assumes minimization, so objectives and reference point are negated
    y_pareto = np.reshape(model.posterior(torch.from_numpy(x_pareto)).mean.detach().numpy(), (-1, nOuts))
    ref_point_pareto = np.min(np.concatenate((y_data, y_pareto), axis=0), axis=0)
    ind = HV(ref_point=-1.01 * ref_point_pareto)

    # Stage 2: Greedy sequential hypervolume-ranked batch selection for q candidates
    for k in range(q):
        if y_pareto.shape[0] == 0:
            candidate_af[k, :] = np.random.rand(dim)
            continue

        # Compute posterior-mean HV of (y_data U {candidate i}) for all remaining candidates
        hv_contr = []
        for i in range(y_pareto.shape[0]):
            hv_check_arr = np.concatenate((y_data, np.reshape(y_pareto[i, :], (1, nOuts))), axis=0)
            hv_ind = ind(-hv_check_arr)
            hv_contr.append(hv_ind)

        # Select candidate with largest HV contribution, append to y_data, and remove from pool
        top_index = np.argmax(hv_contr)
        candidate_af[k, :] = x_pareto[top_index, :]
        y_data = np.concatenate((y_data, np.reshape(y_pareto[top_index, :], (1, -1))), axis=0)
        x_pareto = np.delete(x_pareto, top_index, axis=0)
        y_pareto = np.delete(y_pareto, top_index, axis=0)

    return candidate_af


def qHAX_UCB(dim, model, ref_point_np, left_lim_cont, right_lim_cont, train_x, nOuts, q=1, pop_size_ga=300, gen_n_ga=450):
    # qHAX-UCB: Hypervolume-ranked Upper Confidence Bound (Sheikh, 2026; Srinivas et al., 2010; Hoffman et al., 2011)
    y_data = model.posterior(train_x).mean.detach().numpy()
    y_opt_ind = np.sum(y_data, axis=1)
    y_opt = y_data[np.argmax(np.squeeze(y_opt_ind)), :]

    # GP-UCB exploration schedule parameters (nu = 0.2, delta = 0.1)
    nu = 0.2
    delta = 0.1
    test_count = y_data.shape[0]
    z_ucb = 2 * np.log(((test_count ** (dim / 2 + 2)) * np.pi ** 2) / (3 * delta))
    kappa = (nu * z_ucb) ** 0.5

    class AF_Opt(Problem):
        def __init__(self):
            super().__init__(n_var=dim, n_obj=nOuts, n_ieq_constr=0, xl=left_lim_cont, xu=right_lim_cont)

        def _evaluate(self, x, out, *args, **kwargs):
            x_torch = torch.from_numpy(x)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Evaluate GP posterior mean and std across objectives
                posterior = model.posterior(x_torch)
                mu_x = posterior.mean.detach().numpy()
                sigma_x = posterior.stddev.detach().numpy()

                # Compute Upper Confidence Bound per objective
                valid_mask = sigma_x > 1e-4
                ucb_values = mu_x + kappa * sigma_x
                fit_score = np.where(valid_mask, ucb_values, mu_x)

                # Negate acquisition values since PyMOO minimizes out["F"]
                out["F"] = -fit_score

    # Stage 1: Generate Pareto-optimal acquisition candidate pool x_pareto via NSGA-II
    problem = AF_Opt()
    algorithm = NSGA2(pop_size=pop_size_ga, eliminate_duplicates=True)
    res = minimize(problem, algorithm, ('n_gen', gen_n_ga), verbose=False)

    x_pareto = np.copy(res.X)
    candidate_af = np.zeros((q, dim))

    # Evaluate posterior mean objectives for candidate pool and set HV reference point
    # Note: PyMOO HV indicator assumes minimization, so objectives and reference point are negated
    y_pareto = np.reshape(model.posterior(torch.from_numpy(x_pareto)).mean.detach().numpy(), (-1, nOuts))
    ref_point_pareto = np.min(np.concatenate((y_data, y_pareto), axis=0), axis=0)
    ind = HV(ref_point=-1.01 * ref_point_pareto)

    # Stage 2: Greedy sequential hypervolume-ranked batch selection for q candidates
    for k in range(q):
        if y_pareto.shape[0] == 0:
            candidate_af[k, :] = np.random.rand(dim)
            continue

        # Compute posterior-mean HV of (y_data U {candidate i}) for all remaining candidates
        hv_contr = []
        for i in range(y_pareto.shape[0]):
            hv_check_arr = np.concatenate((y_data, np.reshape(y_pareto[i, :], (1, nOuts))), axis=0)
            hv_ind = ind(-hv_check_arr)
            hv_contr.append(hv_ind)

        # Select candidate with largest HV contribution, append to y_data, and remove from pool
        top_index = np.argmax(hv_contr)
        candidate_af[k, :] = x_pareto[top_index, :]
        y_data = np.concatenate((y_data, np.reshape(y_pareto[top_index, :], (1, -1))), axis=0)
        x_pareto = np.delete(x_pareto, top_index, axis=0)
        y_pareto = np.delete(y_pareto, top_index, axis=0)

    return candidate_af


def random(dim, model, ref_point_np, left_lim_cont, right_lim_cont, train_x, nOuts, q=1):
    # Uniform random sampling baseline within [left_lim_cont, right_lim_cont]
    candidate_af = np.zeros((q, dim))
    for i in range(q):
        random_sample = np.random.rand(dim)
        candidate_af[i, :] = random_sample * (right_lim_cont - left_lim_cont) + left_lim_cont

    return np.reshape(candidate_af, (-1, dim))
    
#==========================================#
# Multi-objective hedge strategies (Sheikh, 2026)
#==========================================#
# Portfolio selection rules (Algorithm 2 and Appendix D.2 ablations):
# - candidate_list has shape [len(AF), dim], corresponding to the q-th batch slot across all K portfolio strategies.
# - Historical nominees for strategy i are stored in <AF[i].__name__>_AF.txt.
# - PyMOO HV assumes minimization, so standardized posterior means and ref_point (-3, ..., -3) are negated.

def MO_Hedge(candidate_list, model, ref_point, AF, nOuts, train_x, dim, q=1):
    # MO-Hedge (Algorithm 2): Recalibrates HV reward over full candidate history under current GP posterior (Sheikh, 2026)
    gains = np.zeros(len(AF))
    prob = np.ones(len(AF)) / len(AF)

    y_data = model.posterior(train_x).mean.detach().numpy()

    # Re-evaluate cumulative candidate history H_t^k under updated GP posterior mean mu_t
    for i in range(len(AF)):
        if path.exists(AF[i].__name__ + "_AF.txt") == True:
            x_AF_old = np.reshape(np.loadtxt(AF[i].__name__ + "_AF.txt"), (-1, dim))
            pred_mean = model.posterior(torch.from_numpy(x_AF_old)).mean.detach().numpy()
            ind = HV(ref_point=-1 * ref_point)
            if nOuts == 1:
                gains[i] = np.sum(pred_mean)
            else:
                hv_check_arr = np.reshape(pred_mean, (-1, nOuts))
                gains[i] = ind(-1 * hv_check_arr)

    print('Candidates: ', candidate_list)
    print("Gains: ", gains)

    # Compute exponentiated-weights selection probabilities with time-varying learning rate eta_t
    test_count = train_x.shape[0]
    eta = (8 * np.log(len(AF)) / test_count)**0.5

    for i in range(len(AF)):
        prob[i] = np.exp((eta * gains[i]) - scipy.special.logsumexp(eta * gains))
    print("Hedge Probabilities:", prob)

    # Sample winning acquisition strategy for current batch slot
    choose_AF = np.arange(len(AF))
    arg_AF = np.random.choice(choose_AF, p=prob)
    output = np.reshape(candidate_list[arg_AF, :], (1, -1))

    return output


def HedgeMORandom(candidate_list, model, ref_point, AF, nOuts, train_x, dim, q=1):
    # MO-Hedge variant used for separate logging of random-inclusive portfolios
    gains = np.zeros(len(AF))
    prob = np.ones(len(AF)) / len(AF)

    y_data = model.posterior(train_x).mean.detach().numpy()

    # Re-evaluate cumulative candidate history H_t^k under updated GP posterior mean mu_t
    for i in range(len(AF)):
        if path.exists(AF[i].__name__ + "_AF.txt") == True:
            x_AF_old = np.reshape(np.loadtxt(AF[i].__name__ + "_AF.txt"), (-1, dim))
            pred_mean = model.posterior(torch.from_numpy(x_AF_old)).mean.detach().numpy()
            ind = HV(ref_point=-1 * ref_point)
            if nOuts == 1:
                gains[i] = np.sum(pred_mean)
            else:
                hv_check_arr = np.reshape(pred_mean, (-1, nOuts))
                gains[i] = ind(-1 * hv_check_arr)

    print('Candidates: ', candidate_list)
    print("Gains: ", gains)

    # Compute exponentiated-weights selection probabilities
    test_count = train_x.shape[0]
    eta = (8 * np.log(len(AF)) / test_count)**0.5

    for i in range(len(AF)):
        prob[i] = np.exp((eta * gains[i]) - scipy.special.logsumexp(eta * gains))
    print("Hedge Probabilities:", prob)

    # Sample winning acquisition strategy for current batch slot
    choose_AF = np.arange(len(AF))
    arg_AF = np.random.choice(choose_AF, p=prob)
    output = np.reshape(candidate_list[arg_AF, :], (1, -1))

    return output


def HedgeMOFixedPredictions(candidate_list, model, ref_point, AF, nOuts, train_x, dim, q=1):
    # MO-Hedge Ablation (Appendix D.2 - Frozen Predictions):
    # Freezes each nominee's predicted objective at the iteration it was evaluated (without re-querying updated GP)
    gains = np.zeros(len(AF))
    prob = np.ones(len(AF)) / len(AF)

    test_count = train_x.shape[0]

    # Guard file ensures previous batch is recorded only once per epoch (on first q_candidate call)
    update_history = True
    if path.exists('HedgeMOFixedPredictionsCount_AF.txt') == True:
        previous_count = np.loadtxt('HedgeMOFixedPredictionsCount_AF.txt')
        if previous_count == test_count:
            update_history = False

    # Evaluate only the newly added batch [-q:, :] and save unstandardized predictions to file
    if update_history == True:
        for i in range(len(AF)):
            if path.exists(AF[i].__name__ + "_AF.txt") == True:
                x_AF_history = np.reshape(np.loadtxt(AF[i].__name__ + "_AF.txt"), (-1, dim))
                x_AF_old = x_AF_history[-q:, :]
                pred_mean = model.posterior(torch.from_numpy(x_AF_old)).mean.detach().numpy()

                stats = np.reshape(np.loadtxt('StandardizeHedge.txt'), (-1, nOuts))
                mean = stats[-2, :]
                std = stats[-1, :]
                pred_mean_original = (pred_mean * std) + mean

                file_writer(np.reshape(pred_mean_original, (-1, nOuts)), AF[i].__name__ + "_FixedPredictions_AF.txt")

        np.savetxt('HedgeMOFixedPredictionsCount_AF.txt', np.array([test_count]))

    # Rescale stored predictions to current epoch's standardization and compute cumulative HV gain
    for i in range(len(AF)):
        if path.exists(AF[i].__name__ + "_FixedPredictions_AF.txt") == True:
            pred_mean_original = np.reshape(np.loadtxt(AF[i].__name__ + "_FixedPredictions_AF.txt"), (-1, nOuts))

            stats = np.reshape(np.loadtxt('StandardizeHedge.txt'), (-1, nOuts))
            mean = stats[-2, :]
            std = stats[-1, :]
            pred_mean = (pred_mean_original - mean) / std

            ind = HV(ref_point=-1 * ref_point)
            if nOuts == 1:
                gains[i] = np.sum(pred_mean)
            else:
                hv_check_arr = np.reshape(pred_mean, (-1, nOuts))
                gains[i] = ind(-1 * hv_check_arr)

    print('Candidates: ', candidate_list)
    print("Gains: ", gains)

    # Compute exponentiated-weights selection probabilities
    eta = np.sqrt(8 * np.log(len(AF)) / test_count)
    for i in range(len(AF)):
        prob[i] = np.exp((eta * gains[i]) - scipy.special.logsumexp(eta * gains))
    print("Hedge Probabilities:", prob)

    # Sample winning acquisition strategy for current batch slot
    choose_AF = np.arange(len(AF))
    arg_AF = np.random.choice(choose_AF, p=prob)
    output = np.reshape(candidate_list[arg_AF, :], (1, -1))

    return output


def HedgeMOFrozen(candidate_list, model, ref_point, AF, nOuts, train_x, dim, q=1):
    # MO-Hedge Ablation (Appendix D.2 - Frozen Rewards):
    # Accumulates incremental step rewards g_t^k = g_{t-1}^k + HV(mu_t(C_{t-1}^k)) without recalibrating past rewards
    gains = np.zeros(len(AF))
    rewards = np.zeros(len(AF))
    prob = np.ones(len(AF)) / len(AF)

    test_count = train_x.shape[0]

    # Load cumulative frozen gains from previous epochs
    if path.exists('HedgeMOFrozenGains_AF.txt') == True:
        gains = np.reshape(np.loadtxt('HedgeMOFrozenGains_AF.txt'), (-1))

    # Guard file ensures incremental reward is added only once per epoch (on first q_candidate call)
    update_gains = True
    if path.exists('HedgeMOFrozenCount_AF.txt') == True:
        previous_count = np.loadtxt('HedgeMOFrozenCount_AF.txt')
        if previous_count == test_count:
            update_gains = False

    # Compute HV of only the immediately preceding batch [-q:, :] and add to cumulative gains
    if update_gains == True:
        for i in range(len(AF)):
            if path.exists(AF[i].__name__ + "_AF.txt") == True:
                x_AF_history = np.reshape(np.loadtxt(AF[i].__name__ + "_AF.txt"), (-1, dim))
                x_AF_old = x_AF_history[-q:, :]
                pred_mean = model.posterior(torch.from_numpy(x_AF_old)).mean.detach().numpy()

                ind = HV(ref_point=-1 * ref_point)
                if nOuts == 1:
                    rewards[i] = np.sum(pred_mean)
                else:
                    hv_check_arr = np.reshape(pred_mean, (-1, nOuts))
                    rewards[i] = ind(-1 * hv_check_arr)

        gains = gains + rewards
        np.savetxt('HedgeMOFrozenGains_AF.txt', np.reshape(gains, (1, -1)))
        np.savetxt('HedgeMOFrozenCount_AF.txt', np.array([test_count]))

    print('Candidates: ', candidate_list)
    print("Gains: ", gains)

    # Compute exponentiated-weights selection probabilities
    eta = np.sqrt(8 * np.log(len(AF)) / test_count)
    for i in range(len(AF)):
        prob[i] = np.exp((eta * gains[i]) - scipy.special.logsumexp(eta * gains))
    print("Hedge Probabilities:", prob)

    # Sample winning acquisition strategy for current batch slot
    choose_AF = np.arange(len(AF))
    arg_AF = np.random.choice(choose_AF, p=prob)
    output = np.reshape(candidate_list[arg_AF, :], (1, -1))

    return output


def HedgeMOImmediate(candidate_list, model, ref_point, AF, nOuts, train_x, dim, q=1):
    # MO-Hedge Ablation (Appendix D.2 - One-Step Rewards):
    # Computes HV reward using only the immediately preceding batch [-q:, :], discarding earlier history
    gains = np.zeros(len(AF))
    prob = np.ones(len(AF)) / len(AF)

    # Evaluate only the previous iteration's proposed batch [-q:, :] under current GP posterior
    for i in range(len(AF)):
        if path.exists(AF[i].__name__ + "_AF.txt") == True:
            x_AF_history = np.reshape(np.loadtxt(AF[i].__name__ + "_AF.txt"), (-1, dim))
            x_AF_old = x_AF_history[-q:, :]
            pred_mean = model.posterior(torch.from_numpy(x_AF_old)).mean.detach().numpy()

            ind = HV(ref_point=-1 * ref_point)
            if nOuts == 1:
                gains[i] = np.sum(pred_mean)
            else:
                hv_check_arr = np.reshape(pred_mean, (-1, nOuts))
                gains[i] = ind(-1 * hv_check_arr)

    print('Candidates: ', candidate_list)
    print("Gains: ", gains)

    # Compute exponentiated-weights selection probabilities
    test_count = train_x.shape[0]
    eta = np.sqrt(8 * np.log(len(AF)) / test_count)
    for i in range(len(AF)):
        prob[i] = np.exp((eta * gains[i]) - scipy.special.logsumexp(eta * gains))
    print("Hedge Probabilities:", prob)

    # Sample winning acquisition strategy for current batch slot
    choose_AF = np.arange(len(AF))
    arg_AF = np.random.choice(choose_AF, p=prob)
    output = np.reshape(candidate_list[arg_AF, :], (1, -1))

    return output
    
#==========================================#
# Main MOBO Test Loops
#==========================================#
def main():
    # Problem configuration
    dim = 9 # Input decision space dimensionality (d)
    test_function = TestZDT2 # Benchmark function from Cell 3
    nOuts = 2 # Number of objectives (M)
    n_tests = 3 # Number of independent paired trials
    q = 3 # Batch size (Q) evaluated per BO epoch (q=1 for sequential, q=3 for batch)

    # Portfolio and evaluation budget configuration
    # Note: out_AF_list holds the portfolios to benchmark; use a single-item list (e.g., [[qHAX_UCB]]) to run a standalone acquisition
    # Note: Number of BO epochs is epoch_total. For q batch this yields epoch_total * q evaluations
    out_AF_list = [[qHAX_UCB], [qHAX_EI], [random, qLogNParEGO, qHAX_UCB], [qLNEHVI, qHVKG, qHAX_EI, qHAX_UCB], [qLB_MOJES], [qHVKG], [qLogNParEGO], [qLNEHVI]]
    noise_percent = 0.01 # Observation noise std as a fraction (1%) of mean initial objective value
    hedge_strategy = MO_Hedge # Hedge selection rule from Cell 7
    initial_sample_size = 30 # Number of initial Halton design points |D_0|
    epoch_total = 5 # Total number of BO epochs after initial_sample_size

    # Normalized unit-cube search domain [0, 1]^d and double precision setup
    n_cont = dim
    n_evals = epoch_total + initial_sample_size # Sets n_evals = epoch_total + initial_sample_size
    left_lim_cont = np.zeros(n_cont)
    right_lim_cont = np.ones(n_cont)
    torch.set_default_dtype(torch.float64)

    # Clean up previous output files before starting new benchmark run
    for old_file in ['StandardizeHedge.txt', 'HedgeSamplePointResults.txt', 'TimeResults.txt']:
        if path.exists(old_file) == True:
            remove(old_file)
    for filename in glob.glob('*_AF.txt'): 
        remove(filename)

    # Execute independent paired runs
    for test in range(n_tests):
        print('Test Number: ', test + 1)

        # Generate shared initial Halton design D_0 in [0, 1]^d and compute objective noise variance
        train_x_orig = init_sampling(initial_sample_size, [], [], n_cont, [], left_lim_cont, right_lim_cont, algo='Halton')
        noise_var = (noise_percent * np.squeeze(np.mean(test_function(train_x_orig, [np.zeros(nOuts)], nOuts), axis=0)))**2
        train_y_orig = test_function(train_x_orig, [np.sqrt(noise_var)], nOuts)

        # Evaluate each acquisition/portfolio starting from the identical initial dataset D_0
        for out_AF in range(len(out_AF_list)):
            AF = out_AF_list[out_AF]
            print('AF Name: ', AF)
            train_x = copy.deepcopy(train_x_orig)
            train_y = copy.deepcopy(train_y_orig)

            epoch_ind = [0]
            time_init = time.time()

            # Sequential / batch MOBO loop over T = (n_evals - initial_sample_size) epochs
            for epoch in range(n_evals - initial_sample_size):
                print('Epoch: ', epoch + 1)
                epoch_ind.append(epoch + 1)

                # Standardize current observations to zero mean and unit variance (saving mean/std to StandardizeHedge.txt)
                train_y = standardize(train_y, flag=0, file='StandardizeHedge.txt')
                ref_point_np = np.min(train_y, axis=0) - 0.01
                train_y = torch.from_numpy(train_y).to(torch.double)
                train_x = torch.from_numpy(train_x).to(torch.double)

                # Fit independent SingleTaskGP surrogate to each standardized objective
                models = []
                for i in range(train_y.shape[-1]):
                    train_obj = train_y[:, i:i+1]
                    models.append(SingleTaskGP(train_x, train_obj))
                model = ModelListGP(*models).to(torch.double)
                mll = SumMarginalLogLikelihood(model.likelihood, model)
                botorch.fit.fit_gpytorch_mll(mll)

                # Propose q-candidate batch independently from each acquisition function in portfolio AF
                candidate_list = np.zeros((q, len(AF), dim))
                for af_i in range(len(AF)):
                    candidate_AF = AF[af_i](dim, model, ref_point_np, np.zeros(n_cont), np.ones(n_cont), train_x, nOuts, q)
                    candidate_list[:, af_i, :] = candidate_AF

                # Select q evaluation candidates across portfolio using MO-Hedge (reference point r^hedge = [-3, ..., -3])
                candidate_x = np.zeros((q, dim))
                for q_candidate in range(q):
                    candidate_x[q_candidate, :] = hedge_strategy(np.reshape(candidate_list[q_candidate, :, :], (len(AF), dim)), model, -3 * np.ones(nOuts), AF, nOuts, train_x, dim, q)

                # Append all proposed nominees to each acquisition's history file (<AF_name>_AF.txt) for future hedge updates
                for af_i in range(len(AF)):
                    file_writer(np.reshape(candidate_list[:, af_i, :], (q, dim)), AF[af_i].__name__ + "_AF.txt")

                # Append selected batch to training inputs and evaluate black-box test function
                train_x = np.concatenate((train_x.detach().numpy(), np.reshape(candidate_x, (-1, dim))), axis=0)
                print('Candidates generated: ', np.squeeze(train_x[-q:, :]))
                candidate_y = test_function(np.reshape(train_x[-q:, :], (-1, dim)), [np.sqrt(noise_var)], nOuts)

                # Unstandardize existing observations back to original scale and append newly evaluated observations
                train_y = standardize(train_y, flag=1, file='StandardizeHedge.txt')
                train_y = np.concatenate((train_y, np.reshape(candidate_y, (-1, nOuts))), axis=0)

                if path.exists('StandardizeHedge.txt') == True:
                    remove('StandardizeHedge.txt')

            # Remove temporary candidate history and hedge state files (*_AF.txt) before next run
            for filename in glob.glob('*_AF.txt'):
                remove(filename)

            # Save full trajectory of evaluated inputs/objectives and wall-clock runtime
            data_tot = np.concatenate((train_x, train_y), axis=1)
            file_writer(data_tot, 'HedgeSamplePointResults.txt')

            time_exec = np.array([[time.time() - time_init]])
            file_writer(time_exec, 'TimeResults.txt')

    # Plot HV, IGD trajectories and runtime bar chart
    plot_results('HedgeSamplePointResults.txt', 'TimeResults.txt', test_function, dim, q, out_AF_list, epoch_total)


if __name__ == "__main__":
    # Print paper citation
    print("q-HAX-alpha and MO-Hedge for Multi-Objective Bayesian Optimization")
    print("To Cite: Haris Moazam Sheikh, 'Simple Extensions of Single-Objective Acquisition Functions and Hedge Strategies for Multi-Objective Bayesian Optimization', 40th Conference on Neural Information Processing Systems, 2026")
    main()
