#!/usr/bin/env python3
"""
Real Dataset Experiment Runner
Runs all methods on real datasets only (boston, housing)
"""

import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Import our modules
from datasets import DATASETS
from loaders import load_any
from sampling import stratified_with_replacement, rebalance_train_indices, get_class_distribution

# Import methods
from methods.lasso import run_lasso
from methods.lasso_Net import run_lassonet
from methods.nimo_variants import run_nimo, run_nimo_baseline, run_nimo_transformer
from methods.random_forest import run_random_forest
from methods.neural_net import run_neural_net

# =============================================================================
# EXCLUSION LIST - Add dataset IDs here to exclude them from running
# =============================================================================
# Example: EXCLUDE_DATASETS = ["boston", "breast_cancer"]  # Exclude these datasets
# Example: EXCLUDE_DATASETS = []  # Run all datasets
EXCLUDE_DATASETS = []  # Exclude diabetes and moon, run boston and housing

def get_real_datasets(exclude_datasets=None):
    """
    Get only real datasets from the registry, optionally excluding some.

    Args:
        exclude_datasets: List of dataset IDs to exclude from running

    Returns:
        list: List of real dataset configurations (excluding specified ones)
    """
    if exclude_datasets is None:
        exclude_datasets = []
    
    real_datasets = [d for d in DATASETS if d["kind"] == "real"]
    
    # Filter out excluded datasets
    if exclude_datasets:
        real_datasets = [d for d in real_datasets if d["id"] not in exclude_datasets]
        print(f"Excluding datasets: {exclude_datasets}")
    
    print(f"Found {len(real_datasets)} real datasets: {[d['id'] for d in real_datasets]}")
    return real_datasets


def run_all_methods(X_tr, y_tr, X_va, y_va, X_te, y_te, seed, feature_names, dataset_info=None):
    """
    Run all methods on the given data splits.

    Args:
        X_tr, y_tr: Training data
        X_va, y_va: Validation data
        X_te, y_te: Test data
        seed: Random seed
        feature_names: List of feature names
        dataset_info: Optional dataset metadata

    Returns:
        list: Results from all methods
    """
    methods = [
        ("Lasso", run_lasso),
        ("LassoNet", run_lassonet),
        ("NIMO_MLP", run_nimo_baseline),
        ("NIMO_T", run_nimo_transformer),
        ("RF", run_random_forest),
        ("NN", run_neural_net),
        # ("sparse_neural_net", run_sparse_neural_net),
        # ("sparse_linear_baseline", run_sparse_linear_baseline)
    ]

    results = []

    for method_name, method_func in methods:
        try:
            print(f"    Running {method_name}...")
            start_time = time.time()

            # Run method with consistent interface
            result = method_func(
                X_tr, y_tr, X_te, y_te,
                iteration=0,  # Will be set by caller
                randomState=seed,
                X_columns=feature_names,
                X_val=X_va,
                y_val=y_va
            )

            # Add timing info
            result["training_time"] = time.time() - start_time

            # Add dataset info if provided
            if dataset_info:
                result.update(dataset_info)

            results.append(result)
            print(f"      ✓ {method_name} completed in {result['training_time']:.2f}s")

        except Exception as e:
            print(f"      ✗ Error running {method_name}: {e}")
            # Create error result
            error_result = {
                "model_name": method_name,
                "iteration": 0,  # Will be set by caller
                "random_seed": seed,
                "error": str(e),
                "training_time": 0.0
            }
            if dataset_info:
                error_result.update(dataset_info)
            results.append(error_result)

    return results


def main(n_iterations=30, rebalance_config=None, output_dir="../results/real", exclude_datasets=None):
    """
    Main experiment runner for real datasets only.

    Args:
        n_iterations: Number of iterations per dataset
        rebalance_config: Rebalancing configuration dict
        output_dir: Output directory for results
        exclude_datasets: List of dataset IDs to exclude from running
    """
    # Record pipeline start time
    pipeline_start_time = time.time()
    pipeline_start_datetime = datetime.now()
    
    # Default rebalancing config
    if rebalance_config is None:
        rebalance_config = {"mode": "undersample", "target_pos": 0.5}

    # Get only real datasets (excluding specified ones)
    all_datasets = get_real_datasets(exclude_datasets)

    print("=" * 80)
    print("REAL DATASET EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Pipeline started at: {pipeline_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets: {len(all_datasets)} (real only)")
    print(f"Iterations per dataset: {n_iterations}")
    print(f"Rebalancing: {rebalance_config}")
    print(f"NIMO variants: nimo_transformer (updated), nimo_transformer_old (original)")
    print(f"Output directory: {output_dir}")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    start_time = time.time()

    for dataset_idx, entry in enumerate(all_datasets):
        print(f"\n{'=' * 20} DATASET {entry['id']} ({entry['kind']}) {'=' * 20}")
        print(f"Description: {entry.get('desc', 'No description')}")

        try:
            # Load dataset
            X, y, idx_test, idx_pool, meta = load_any(entry)
            X_te, y_te = X[idx_test], y[idx_test]
            feature_names = meta["feature_names"]

            print(f"Data shape: {X.shape}")
            print(f"Test set: {len(idx_test)} samples")
            print(f"Pool: {len(idx_pool)} samples")
            print(f"Features: {len(feature_names)}")

            # Dataset info for results
            dataset_info = {
                "data_type": entry["kind"],
                "dataset_id": entry["id"],
                "dataset_description": entry.get("desc", ""),
                "n_features_total": X.shape[1],
                "n_test_samples": len(idx_test),
                "n_pool_samples": len(idx_pool)
            }

            # Add unified scenario-like metadata for plotting compatibility
            scenario_descriptions = {
                'breast_cancer': 'Breast Cancer Wisconsin (569 samples, 30 features)',
                'pima': 'Pima Indians Diabetes (768 samples, 8 features)',
                'bank_marketing': 'Bank Marketing (11,161 samples, 50 features)',
                'adult_income': 'Adult Income (32,561 samples, 14 features)'
            }

            # Add scenario-like fields for unified plotting
            dataset_info.update({
                "scenario": entry["id"],  # Use dataset_id as scenario
                "scenario_description": scenario_descriptions.get(entry["id"], entry.get("desc", "")),
                "scenario_title": f"Dataset {entry['id']}: {scenario_descriptions.get(entry['id'], entry.get('desc', ''))}"
            })

            # For real datasets, add empty true support info for plotting compatibility
            dataset_info.update({
                "n_true_features": 0,
                "true_support": json.dumps([]),
                "beta_true": json.dumps([]),
                "b0_true": 0.0
            })

            # Run iterations
            for it in range(n_iterations):
                print(f"\n  --- Iteration {it + 1}/{n_iterations} ---")

                seed = 42 + 1000 * it

                # Sample train/val with replacement
                idx_tr, idx_va = stratified_with_replacement(
                    y, idx_pool,
                    entry["n_train"], entry["n_val"],
                    seed
                )

                # Rebalance training set if requested
                if rebalance_config["mode"] != "none":
                    idx_tr = rebalance_train_indices(
                        y, idx_tr,
                        mode=rebalance_config["mode"],
                        target_pos=rebalance_config["target_pos"],
                        seed=seed
                    )

                # Get data splits
                X_tr, y_tr = X[idx_tr], y[idx_tr]
                X_va, y_va = X[idx_va], y[idx_va]

                # Print class distributions
                train_dist = get_class_distribution(y, idx_tr)
                val_dist = get_class_distribution(y, idx_va)
                print(f"    Train: {train_dist['n_total']} samples ({train_dist['pos_proportion']:.1%} pos)")
                print(f"    Val: {val_dist['n_total']} samples ({val_dist['pos_proportion']:.1%} pos)")

                # Run all methods
                iteration_results = run_all_methods(
                    X_tr, y_tr, X_va, y_va, X_te, y_te,
                    seed, feature_names, dataset_info
                )

                # Add iteration info to all results
                for result in iteration_results:
                    result["iteration"] = it
                    result["random_seed"] = seed

                all_results.extend(iteration_results)

        except Exception as e:
            print(f"✗ Error loading dataset {entry['id']}: {e}")
            continue

    # Record pipeline end time and calculate duration
    pipeline_end_time = time.time()
    pipeline_end_datetime = datetime.now()
    pipeline_duration_seconds = pipeline_end_time - pipeline_start_time
    pipeline_duration_minutes = pipeline_duration_seconds / 60

    # Save results
    df = pd.DataFrame(all_results)
    output_file = os.path.join(output_dir, "experiment_results.csv")
    df.to_csv(output_file, index=False)

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_start_time": pipeline_start_datetime.isoformat(),
        "pipeline_end_time": pipeline_end_datetime.isoformat(),
        "pipeline_duration_seconds": pipeline_duration_seconds,
        "pipeline_duration_minutes": pipeline_duration_minutes,
        "n_iterations": n_iterations,
        "rebalance_config": rebalance_config,
        "n_datasets": len(all_datasets),
        "n_results": len(all_results),
        "datasets": all_datasets,
        "total_time": time.time() - start_time
    }

    with open(os.path.join(output_dir, "experiment_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("REAL DATASET EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Pipeline started at: {pipeline_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Pipeline finished at: {pipeline_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Pipeline duration: {pipeline_duration_minutes:.2f} minutes ({pipeline_duration_seconds:.1f} seconds)")
    print(f"Total results: {len(all_results)}")
    print(f"Datasets processed: {len(all_datasets)}")
    print(f"Iterations per dataset: {n_iterations}")
    print(f"Results saved to: {output_file}")

    # Print per-dataset summary
    if len(all_results) > 0:
        print(f"\nPer-dataset summary:")
        for dataset_id in df['dataset_id'].unique():
            dataset_results = df[df['dataset_id'] == dataset_id]
            if 'error' in df.columns:
                n_success = len(dataset_results[dataset_results['error'].isna()])
            else:
                n_success = len(dataset_results)
            n_total = len(dataset_results)
            print(f"  {dataset_id}: {n_success}/{n_total} successful runs")

    # Print per-method summary
    print(f"\nPer-method summary:")
    for method_name in df['model_name'].unique():
        method_results = df[df['model_name'] == method_name]
        if 'error' in df.columns:
            n_success = len(method_results[method_results['error'].isna()])
            successful_results = method_results[method_results['error'].isna()]
        else:
            n_success = len(method_results)
            successful_results = method_results
        n_total = len(method_results)
        avg_f1 = successful_results['f1'].mean() if n_success > 0 and 'f1' in successful_results.columns else 0
        print(f"  {method_name}: {n_success}/{n_total} successful runs, avg F1: {avg_f1:.3f}")

    print(f"\n✓ Real dataset experiment completed successfully!")
    print(f"✓ Total pipeline time: {pipeline_duration_minutes:.2f} minutes")
    return df


if __name__ == "__main__":
    # Run with default settings and exclusion list
    df = main(n_iterations=20, exclude_datasets=EXCLUDE_DATASETS)  # Start with 1 iteration for testing
