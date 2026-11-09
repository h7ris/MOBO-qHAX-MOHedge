# Simple Extensions of Single-Objective Acquisition Functions and Hedge Strategies for Multi-Objective Bayesian Optimization

Official open-source implementation for the **NeurIPS 2026** paper:

> **Simple Extensions of Single-Objective Acquisition Functions and Hedge Strategies for Multi-Objective Bayesian Optimization**
>
> Haris Moazam Sheikh
>
> *Department of Aeronautical and Astronautical Engineering, University of Southampton, UK*
>
> *40th Conference on Neural Information Processing Systems (NeurIPS 2026)*

## Overview

This repository provides two lightweight, modular frameworks that extend classical single-objective Bayesian optimization (BO) strategies directly to the non-preferential multi-objective setting:

1. $\text{qHAX-}\alpha$ **(Parallel Hypervolume-ranked Acquisition Extension):**
   Decouples per-objective exploration-exploitation from cross-objective trade-off selection in two stages:
   * **Stage 1 (Pareto Candidate Generation):** Evaluates standard single-objective acquisitions (e.g., UCB or EI) independently for each objective to form a vector-valued acquisition $\alpha_t(x) \in \mathbb{R}^M$, and generates a Pareto-optimal candidate pool $\mathcal{C}_t^0$ via NSGA-II.
   * **Stage 2 (Greedy Hypervolume Ranking):** Selects a batch of $q$ evaluation points from $\mathcal{C}_t^0$ by greedily maximizing posterior-mean hypervolume improvement relative to the denoised Pareto front.

2. $\text{MO-Hedge}$ **(Multi-Objective Hedge Strategy):**
   An online portfolio strategy that adaptively selects among competing MOBO acquisition functions over time. At each iteration, historical candidate nominees from each constituent strategy are recalibrated under the current Gaussian Process (GP) posterior mean, and selection probabilities are updated via exponentiated weights based on their dominated hypervolume contributions.

## Implemented Algorithms & Benchmarks

### Acquisition Functions & Portfolios

* **Proposed** $\text{qHAX-}\alpha$ **Variants:**
  * `qHAX_UCB`: Hypervolume-ranked Upper Confidence Bound ($\text{qHAX-UCB}$)
  * `qHAX_EI`: Hypervolume-ranked Expected Improvement ($\text{qHAX-EI}$)
* **Proposed** $\text{MO-Hedge}$ **& Ablation Variants:**
  * `MO_Hedge`: Default posterior-recalibrated hypervolume reward over cumulative candidate history (Algorithm 2)
  * `HedgeMOFrozen`: Incremental frozen-reward accumulation ablation (Appendix D.2)
  * `HedgeMOImmediate`: One-step immediate-reward ablation (Appendix D.2)
  * `HedgeMOFixedPredictions`: Frozen historical prediction ablation (Appendix D.2)
* **BoTorch Baselines:**
  * `qLNEHVI`: $q$-Log Noisy Expected Hypervolume Improvement (Daulton et al., 2020, 2021)
  * `qLogNParEGO`: $q$-Log Noisy ParEGO with augmented Chebyshev scalarization (Knowles, 2006; Daulton et al., 2020)
  * `qHVKG`: $q$-Hypervolume Knowledge Gradient (Daulton et al., 2023)
  * `qLB_MOJES`: $q$-Lower-Bound Multi-Objective Joint Entropy Search (Tu et al., 2022)
  * `qEHVI`, `qNEHVI`, `qLEHVI`, and `random` (Uniform Random Sampling)

### Multi-Objective Benchmark Suite (13 Problems)

All benchmarks accept normalized inputs $x \in [0, 1]^d$, internally map them to their native domains, add independent Gaussian observation noise, and return objectives formulated for **maximization**:

| Category | Function Name | Description | Paper Config $(d, M)$ | Scaling Configs $(d, M)$ |
| ----- | ----- | ----- | ----- | ----- |
| **Synthetic** | `TestZDT2` | ZDT2 (non-convex Pareto front) | $(9, 2)$ | — |
| **Synthetic** | `TestZDT4` | ZDT4 (highly multimodal distance) | $(6, 2)$ | — |
| **Synthetic** | `TestZDT6` | ZDT6 (non-uniform Pareto front) | $(6, 2)$ | — |
| **Synthetic** | `TestDTLZ1` | DTLZ1 (linear front, multimodal) | $(8, 3)$ | $d \in \{10, 15, 20\}, M \in \{2, 3, 4\}$ |
| **Synthetic** | `TestDTLZ2` | DTLZ2 (spherical front, unimodal) | $(6, 3)$ | — |
| **Synthetic** | `TestDTLZ3` | DTLZ3 (spherical front, multimodal) | $(4, 2)$ | $d \in \{10, 15, 20\}, M \in \{2, 3, 4\}$ |
| **Synthetic** | `TestDTLZ7` | DTLZ7 (disconnected Pareto regions) | — | $d \in \{6, 10, 15, 20\}, M \in \{2, 3, 4\}$ |
| **Synthetic** | `TestBraninCurrin` | Branin-Currin bi-objective benchmark | $(2, 2)$ | — |
| **Engineering** | `TestWeldedBeam` | Unconstrained Welded Beam design | $(4, 2)$ | — |
| **Engineering** | `TestVehicleCrash` | Vehicle Crash Safety design | $(5, 3)$ | — |
| **Application** | `TestPD1CIFAR100` | Wide ResNet CIFAR-100 HPO (PD1) | $(4, 2)$ | — |
| **Application** | `TestAirfoilDesign` | AirDbM aerodynamic shape optimization | $(12, 2)$ | — |
| **Application** | `TestPenicillin` | Penicillin fermentation process | $(7, 3)$ | — |

## Installation

### Core Dependencies

Requires **Python 3.10+**:

```bash
pip install numpy scipy matplotlib tqdm torch gpytorch botorch pymoo
```

### Optional Application-Benchmark Dependencies

All synthetic benchmarks, engineering design problems, and `TestPenicillin` run out of the box with the core dependencies above. For the remaining two application-derived benchmarks:

* **HPO (`TestPD1CIFAR100`):** Install `mfpbench` and download the PD1 surrogate data from [automl/mf-prior-bench](https://github.com/automl/mf-prior-bench).
* **Airfoil Design (`TestAirfoilDesign`):** Clone [AirDbM-Bench](https://github.com/jun9303/AirDbM-Bench/tree/main) (`airdbm_api`) and ensure the `XFOIL` solver binary is on your system path.

## Quick Start & Usage

Run the complete benchmarking and plotting pipeline via:

```bash
python main.py
```

### Configuring an Experiment

In `main()`, configure the benchmark problem, batch size, evaluation budget, and portfolio list `out_AF_list`:

```python
# Problem configuration
dim = 9                       # Input decision space dimensionality (d)
test_function = TestZDT2      # Benchmark function handle
nOuts = 2                     # Number of objectives (M)
n_tests = 20                  # Number of independent paired runs (set to 3 in demo)
q = 3                         # Batch size Q per BO epoch (q=1 for sequential, q=3 for batch)

# Portfolio and evaluation budget configuration
out_AF_list = [
    [qHAX_UCB],                               # Standalone qHAX-UCB
    [qHAX_EI],                                # Standalone qHAX-EI
    [random, qLogNParEGO, qHAX_UCB],          # MO-Hedge 1 (mixed-strength portfolio)
    [qLNEHVI, qHVKG, qHAX_EI, qHAX_UCB],      # MO-Hedge 2 (strong portfolio)
    [qLB_MOJES],                              # Standalone qLB-MOJES
    [qHVKG],                                  # Standalone qHVKG
    [qLogNParEGO],                            # Standalone qLNParEGO
    [qLNEHVI]                                 # Standalone qLNEHVI
]
noise_percent = 0.01          # Observation noise std (1% of mean initial objective value)
hedge_strategy = MO_Hedge     # Hedge reward/selection rule
initial_sample_size = 30      # Number of initial Halton design points |D_0|
epoch_total = 25              # Number of BO epochs T after initial sampling (set to 5 in demo)
```

* **Standalone vs. Portfolio Execution:** Single-item lists in `out_AF_list` (e.g., `[qHAX_UCB]`) run that acquisition standalone; multi-item lists (e.g., `[random, qLogNParEGO, qHAX_UCB]`) run `hedge_strategy` over the specified portfolio.
* **Reproducing Paper Budgets:**
  * **Parallel Batch Mode ($Q = 3$):** Set `q = 3`, `initial_sample_size = 30`, `epoch_total = 25` ($75$ additional evaluations), and `n_tests = 20`.
  * **Sequential Mode ($Q = 1$):** Set `q = 1`, `initial_sample_size = 30`, `epoch_total = 75` ($75$ additional evaluations), and `n_tests = 20`.

## Outputs & Visualization

Each run outputs two summary files in the working directory:

* `HedgeSamplePointResults.txt`: Evaluated decision vectors $x \in [0, 1]^d$ and noisy objective observations $y \in \mathbb{R}^M$ across all runs and algorithms.
* `TimeResults.txt`: Wall-clock execution time (in seconds) per algorithm and run.

At the end of `main()`, `plot_results(...)` automatically plots:

1. **Convergence Trajectories:**
   * **Synthetic & Engineering Problems:** Computes a reference Pareto front via NSGA-II and plots the **Log Normalized HV Gap** and **Log Normalized IGD Gap** (mean $\pm 1$ std shaded).
   * **Application-Derived Problems:** Plots the **Negative Relative HV Improvement** directly from stored observations.
2. **Runtime Comparison:** A log-scale bar chart of the mean wall-clock runtime ($\pm 1$ std) for each strategy in `out_AF_list`.

## Citation

If you use $\text{qHAX-}\alpha$, $\text{MO-Hedge}$, or this codebase in your research, please cite:

```bibtex
@inproceedings{sheikh2026qhax,
  title     = {Simple Extensions of Single-Objective Acquisition Functions and Hedge Strategies for Multi-Objective Bayesian Optimization},
  author    = {Sheikh, Haris Moazam},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {40},
  year      = {2026}
}
```

## Contact

For questions or requests regarding the code or benchmarking data, please open an issue on GitHub or contact **Haris Moazam Sheikh** at `h.m.sheikh@soton.ac.uk`.