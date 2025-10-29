"""
Real Dataset Weight Magnitude Analysis

This script generates weight magnitude visualizations for NIMO-MLP and NIMO-Transformer
models trained on real datasets (boston, housing, diabetes, moon).

All plots are automatically saved to the specified output directory.

Usage:
    python real_weight_analysis.py [--dataset DATASET] [--output-dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directories to path for imports
current_dir = Path(__file__).parent
notebooks_dir = current_dir.parent.parent
sys.path.append(str(notebooks_dir))

from datasets import DATASETS
from loaders import load_any
from sampling import stratified_with_replacement, rebalance_train_indices
from methods.nimo_variants import run_nimo_baseline, run_nimo_transformer
from weight_utils import (
    mlp_first_fc_magnitudes, 
    transformer_first_token_bias_magnitudes,
    create_feature_names,
    plot_mlp_weights,
    plot_transformer_weights
)


def get_real_datasets(exclude_datasets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Get real datasets from the registry, optionally excluding some."""
    if exclude_datasets is None:
        exclude_datasets = []
    
    real_datasets = [d for d in DATASETS if d["kind"] == "real"]
    
    # Filter out excluded datasets
    if exclude_datasets:
        real_datasets = [d for d in real_datasets if d["id"] not in exclude_datasets]
    
    # Fix paths for our current working directory
    for dataset in real_datasets:
        # Convert relative path from notebooks/ to notebooks/plots/weight_magnitude_analysis/
        if dataset["path"].startswith("../data/"):
            dataset["path"] = dataset["path"].replace("../data/", "../../../data/")
    
    return real_datasets


def load_real_dataset(dataset_config: Dict[str, Any]) -> tuple:
    """Load a real dataset using the existing loader infrastructure."""
    dataset_id = dataset_config["id"]
    print(f"Loading dataset: {dataset_id}")
    
    # Use the existing load_any function with the full dataset config
    X, y, idx_test, idx_pool, meta = load_any(dataset_config)
    
    # Ensure y is binary for classification
    if len(np.unique(y)) > 2:
        # Convert to binary classification (median split)
        median_val = np.median(y)
        y = (y > median_val).astype(int)
        print(f"  Converted to binary classification (median split at {median_val:.3f})")
    
    print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Class distribution: {np.bincount(y)}")
    
    # Merge metadata with dataset config
    dataset_config.update(meta)
    
    return X, y, dataset_config


def train_models_and_extract_weights(dataset_id: str, X: np.ndarray, y: np.ndarray, 
                                   dataset_config: Dict[str, Any],
                                   n_train: int = 500, n_val: int = 200, 
                                   n_test: int = 1000) -> Dict[str, Any]:
    """Train both NIMO models and extract weight information."""
    
    print(f"Training models for dataset {dataset_id}...")
    
    # Prepare data splits
    n_samples = len(X)
    
    # Use stratified sampling for real datasets
    # For real datasets, we'll use all available indices as the pool
    idx_pool = np.arange(n_samples)
    train_indices, val_indices = stratified_with_replacement(y, idx_pool, n_train, n_val, 42)
    
    # For test set, use remaining data
    remaining_indices = np.setdiff1d(np.arange(n_samples), np.concatenate([train_indices, val_indices]))
    if len(remaining_indices) >= n_test:
        test_indices = remaining_indices[:n_test]
    else:
        test_indices = remaining_indices
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    # Rebalance training data
    train_indices_balanced = rebalance_train_indices(y, train_indices)
    X_train_balanced = X[train_indices_balanced]
    y_train_balanced = y[train_indices_balanced]
    
    results = {
        'dataset_id': dataset_id,
        'dataset_config': dataset_config,
        'X_test': X_test,
        'y_test': y_test
    }
    
    # Train NIMO-MLP (baseline)
    print("  Training NIMO-MLP...")
    try:
        mlp_result = run_nimo_baseline(
            X_train_balanced, y_train_balanced, X_test, y_test,
            iteration=0, randomState=42,
            X_val=X_val, y_val=y_val,
            return_model_bits=True
        )
        results['mlp'] = mlp_result
        print(f"    MLP F1: {mlp_result.get('f1', 'N/A'):.3f}")
    except Exception as e:
        print(f"    MLP training failed: {e}")
        results['mlp'] = None
    
    # Train NIMO-Transformer
    print("  Training NIMO-Transformer...")
    try:
        transformer_result = run_nimo_transformer(
            X_train_balanced, y_train_balanced, X_test, y_test,
            iteration=0, randomState=42,
            X_val=X_val, y_val=y_val,
            return_model_bits=True
        )
        results['transformer'] = transformer_result
        print(f"    Transformer F1: {transformer_result.get('f1', 'N/A'):.3f}")
    except Exception as e:
        print(f"    Transformer training failed: {e}")
        results['transformer'] = None
    
    return results


def create_weight_magnitude_plots(dataset_id: str, results: Dict[str, Any], 
                                 feature_names: Optional[List[str]] = None,
                                 output_dir: str = "real_weight_analysis_plots") -> None:
    """Create weight magnitude plots for both model types."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create feature names if not provided
    if feature_names is None:
        input_dim = results['X_test'].shape[1]
        # Try to get feature names from dataset config
        dataset_config = results.get('dataset_config', {})
        if 'feature_names' in dataset_config:
            feature_names = dataset_config['feature_names']
        else:
            feature_names = create_feature_names(input_dim)
    
    print(f"\nCreating weight magnitude plots for dataset {dataset_id}...")
    
    # MLP plots
    if results['mlp'] is not None and '_plot_bits' in results['mlp']:
        print("  Creating MLP weight magnitude plots...")
        mlp_weights = results['mlp']['_plot_bits']
        mlp_mags = mlp_first_fc_magnitudes(mlp_weights)
        
        save_path = output_dir / f"{dataset_id}_mlp_weights"
        plot_mlp_weights(
            mlp_mags['W_feat'], 
            mlp_mags['col_L2'],
            feature_names=feature_names,
            title_prefix=f"NIMO-MLP ({dataset_id})",
            save_path=str(save_path)
        )
        print(f"    MLP plots saved to: {save_path}")
    else:
        print("  Skipping MLP plots (no model weights available)")
    
    # Transformer plots
    if results['transformer'] is not None and '_plot_bits' in results['transformer']:
        print("  Creating Transformer weight magnitude plots...")
        transformer_weights = results['transformer']['_plot_bits']
        transformer_mags = transformer_first_token_bias_magnitudes(transformer_weights)
        
        save_path = output_dir / f"{dataset_id}_transformer_weights"
        plot_transformer_weights(
            transformer_mags['bias'],
            transformer_mags['row_L2'],
            feature_names=feature_names,
            title_prefix=f"NIMO-Transformer ({dataset_id})",
            save_path=str(save_path)
        )
        print(f"    Transformer plots saved to: {save_path}")
    else:
        print("  Skipping Transformer plots (no model weights available)")


def main():
    parser = argparse.ArgumentParser(description='Generate weight magnitude plots for real datasets')
    parser.add_argument('--dataset', type=str, default=None, 
                       help='Specific dataset to analyze (e.g., boston). If None, analyze all.')
    parser.add_argument('--output-dir', type=str, default='real_weight_analysis_plots',
                       help='Directory to save plots (default: real_weight_analysis_plots)')
    parser.add_argument('--exclude', nargs='*', default=[],
                       help='Datasets to exclude from analysis')
    parser.add_argument('--n-train', type=int, default=500,
                       help='Number of training samples')
    parser.add_argument('--n-val', type=int, default=200,
                       help='Number of validation samples')
    parser.add_argument('--n-test', type=int, default=1000,
                       help='Number of test samples')
    
    args = parser.parse_args()
    
    # Get real datasets
    datasets = get_real_datasets(exclude_datasets=args.exclude)
    if not datasets:
        print("No real datasets found!")
        return
    
    if args.dataset:
        datasets = [d for d in datasets if d["id"] == args.dataset]
        if not datasets:
            print(f"Dataset {args.dataset} not found. Available: {[d['id'] for d in get_real_datasets()]}")
            return
    
    print(f"Found datasets: {[d['id'] for d in datasets]}")
    
    # Process each dataset
    for dataset_config in datasets:
        dataset_id = dataset_config["id"]
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_id}")
        print(f"{'='*60}")
        
        try:
            # Load data
            X, y, dataset_config = load_real_dataset(dataset_config)
            
            # Train models and extract weights
            results = train_models_and_extract_weights(
                dataset_id, X, y, dataset_config,
                n_train=args.n_train, 
                n_val=args.n_val, 
                n_test=args.n_test
            )
            
            # Create weight magnitude plots
            create_weight_magnitude_plots(
                dataset_id, results, 
                output_dir=args.output_dir
            )
            
        except Exception as e:
            print(f"Error processing dataset {dataset_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("Weight magnitude analysis complete!")
    print(f"All plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
