# unibas-nimo-pipeline

Reproducible pipeline from the University of Basel for benchmarking NIMO(https://arxiv.org/abs/2506.05059) against strong baselines on synthetic and real tabular datasets. The project includes:

- Synthetic scenario generator with ground-truth support and interaction metadata
- Unified loaders and robust sampling/rebalancing utilities
- End-to-end experiment runners for synthetic and real datasets
- NIMO baselines (adaptive ridge + MLP modulator) and transformer-based NIMO-T variants
- Classic baselines: Lasso, LassoNet, Random Forest, and a tuned Neural Network
- Results export + ready-made plotting/analysis notebooks and scripts


## Table of contents
- Introduction
- Repository structure
- Installation
- Data preparation
- Running experiments
- Methods overview
- Results and analysis
- Reproducibility notes
- Citation


## Introduction
This repository investigates feature selection and predictive performance for binary classification on tabular data. We compare:

- Linear sparse methods: Lasso
- Sparse neural method: LassoNet
- Tree ensemble: Random Forest
- Tuned feed-forward Neural Network
- NIMO family:
  - NIMO_MLP: Adaptive ridge logistic regression with a masked MLP modulator (self-leak guarded)
  - NIMO_T: Transformer-enhanced NIMO with strict no-self masking, optional pair-aware attention, and compliance diagnostics

Synthetic scenarios come with true supports and interaction graphs so we can study selection fidelity in addition to predictive quality.


## Repository structure

```text
data/
  real/                  # Preprocessed real datasets (npy/json) and raw CSVs
  synthetic/             # Generated synthetic scenarios (A–F) with npy/json artifacts
notebooks/
  loaders.py             # Unified loaders for real and synthetic data
  sampling.py            # Stratified sampling with replacement and rebalancing
  generate_synthetic_data.py
  real_runner.py         # Run all methods on real datasets only
  synthetic_runner.py    # Run all methods on synthetic datasets only
  methods/
    lasso.py             # L1-logistic baseline
    lasso_Net.py         # LassoNet baseline
    random_forest.py     # RandomForest baseline
    neural_net.py        # Tuned feed-forward NN
    nimo_variants/
      baseline.py        # NIMO_MLP (adaptive ridge + masked MLP modulator)
      nimo.py            # NIMO_T (strict transformer-based NIMO)
      nimoTransformer_NN.py  # NIMO_T (newer training pipeline + variants)
results/
  real/                  # CSV results + experiment metadata for real datasets
  synthetic/             # CSV results + experiment metadata for synthetic datasets
scripts/
  prepare_*.py           # One-off preprocessing for real datasets (Boston, Diabetes, Housing, Moon)
```

Plotting/analysis scripts live under `notebooks/plots/` and include statistical testing, weight analysis, and final figures.


## Installation

Tested with Python 3.10+ on macOS/Linux. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas scikit-learn scipy matplotlib seaborn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install lightning
pip install lassonet
```

Notes:
- If you have CUDA, install the appropriate `torch` package for your GPU.
- On Apple Silicon you can rely on CPU or MPS; Lightning will select the accelerator.


## Data preparation

### Synthetic data
Generate scenarios (C–F are currently enabled in the generator) into `data/synthetic/`:

```bash
python notebooks/generate_synthetic_data.py
```

This produces for each scenario `scenario_<ID>_X_full.npy`, `y_full.npy`, `idx_pool.npy`, `idx_test_big.npy`, and `scenario_<ID>_metadata.json` with true support and interaction edges.

### Real datasets
Raw CSVs should be placed under `data/real/<dataset>/raw/`. Preprocessors write standardized NumPy arrays and metadata to `data/real/<dataset>/processed/`.

- Boston Housing (binary via median split):

```bash
python scripts/prepare_boston.py
```

- Diabetes progression (binary via median split):

```bash
python scripts/prepare_diabetes.py
```

- California Housing (binary via median split):

```bash
python scripts/prepare_housing.py
```

Moon dataset is already provided in `data/real/moon/` in this repo snapshot.


## Running experiments

Two entry points run all methods end-to-end and save results to `results/`.

### Synthetic-only
```bash
python notebooks/synthetic_runner.py
```

Key defaults inside `synthetic_runner.main`:
- Iterations per dataset: 20–30 (set via `n_iterations`)
- Per-iteration sampling: robust stratified train/val from a fixed pool
- Rebalancing: undersample/oversample options via `rebalance_config`
- Methods: Lasso, LassoNet, NIMO_MLP, NIMO_T (+ optional NIMO_T variants), RF, NN
- Output: `results/synthetic/experiment_results.csv` and `experiment_metadata.json`

The runner auto-discovers available scenarios in `data/synthetic/`.

### Real-only
```bash
python notebooks/real_runner.py
```

Key defaults inside `real_runner.main`:
- Iterations per dataset: configurable (default 20–30)
- Per-iteration sampling: stratified with replacement from a pre-defined pool
- Optional rebalancing of the training set
- Methods: Lasso, LassoNet, NIMO_MLP, NIMO_T, RF, NN
- Output: `results/real/experiment_results.csv` and `experiment_metadata.json`

To exclude specific datasets, edit `EXCLUDE_DATASETS` in `notebooks/real_runner.py`.


## Methods overview

- Lasso (`notebooks/methods/lasso.py`):
  - Standardized features (fit on train), 3-fold CV over C grid, fixed 0.5 probability threshold
  - Returns standardized-space coefficients, flat metrics, and selection mask

- LassoNet (`notebooks/methods/lasso_Net.py`):
  - Standardization, 5-fold CV; extracts boolean feature selection from the learned network

- Random Forest (`notebooks/methods/random_forest.py`):
  - Standardization for fair comparison; uses OOB, selects features via importance threshold

- Neural Network (`notebooks/methods/neural_net.py`):
  - Two architectures (ImprovedNN / AdvancedNN) with small CV grid, early stopping, GPU support
  - Threshold selected on validation; fixed 0.5 fallback to avoid test leakage

- NIMO_MLP (`notebooks/methods/nimo_variants/baseline.py`):
  - Adaptive ridge logistic regression with masked MLP modulator, epoch-wise stabilized IRLS
  - Self-feature leak prevention via CO mask and scenario-aware options
  - Temperature scaling on validation; strong sparsity controls; artifact saving for weight analysis

- NIMO_T (`notebooks/methods/nimo_variants/nimoTransformer_NN.py` and `.../nimo.py`):
  - Transformer-based per-feature correction with strict no-self masking and optional pair-aware attention
  - Alternating IRLS updates for sparse β with correction/residual learning; compliance diagnostics
  - Scenario-specific training configs and artifact saving for interpretability/plots


## Results and analysis

After running, consolidated CSV files appear in:
- `results/synthetic/experiment_results.csv`
- `results/real/experiment_results.csv`

Metadata with configurations and timing is stored alongside as `experiment_metadata.json`.

Plotting and statistical testing utilities are available under `notebooks/plots/`, e.g.:
- Final F1 plots per scenario/dataset: `notebooks/plots/final/*.py`, `notebooks/plots/finalReal/*.py`
- Wilcoxon tests and LaTeX tables: `notebooks/plots/*/wilcoxon_tables/`
- Weight magnitude analysis for interpretability: `notebooks/plots/weight_magnitude_analysis/*`


## Reproducibility notes

- Seeds are set per iteration; stratified sampling uses deterministic seeds.
- Thresholds are chosen on validation splits only (no leakage). Fixed 0.5 is used when no validation is available.
- Standardization is always fitted on training data only and applied to val/test.
- Runners print timing summaries and save experiment metadata including start/end timestamps, durations, dataset registry, and configurations.


## Citation not necessary

If you use this code or the NIMO variants in your research, if you wisch you can cite the repository and the corresponding NIMO/NIMO-T methodology as appropriate. A generic BibTeX entry template:

```bibtex
@misc{unibas_nimo_pipeline,
  title  = {NIMO Pipeline: Benchmarking Neural-Interaction Models on Tabular Data},
  author = {Koca, Erdem and collaborators},
  year   = {2025},
  howpublished = {GitHub repository},
  note   = {University of Basel}
}
```


## Quick start checklist

1) Install dependencies (see Installation).
2) Generate synthetic data: `python notebooks/generate_synthetic_data.py`.
3) Prepare real data with `scripts/prepare_*.py` as needed.
4) Run experiments: `python notebooks/synthetic_runner.py` and/or `python notebooks/real_runner.py`.
5) Inspect `results/*/*.csv` and use `notebooks/plots/*` for figures and stats.
