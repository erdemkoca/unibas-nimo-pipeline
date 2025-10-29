"""
Synthetic Dataset Weight Magnitude Analysis

This script generates weight magnitude visualizations for NIMO-MLP and NIMO-Transformer
models trained on synthetic datasets (scenarios A, B, C, D, E).

All plots are automatically saved to the specified output directory.

Usage:
    python synthetic_weight_analysis.py [--scenario SCENARIO] [--output-dir OUTPUT_DIR]
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


def discover_synthetic_scenarios(synthetic_data_path="../../../data/synthetic"):
    """Discover available synthetic scenarios."""
    data_path = Path(synthetic_data_path)
    if not data_path.exists():
        print(f"Warning: Synthetic data path {data_path} does not exist")
        return []
    
    pattern_files = list(data_path.glob("scenario_*_X_full.npy"))
    scenarios = []
    
    for pattern_file in pattern_files:
        scenario_name = pattern_file.stem.replace("_X_full", "")
        scenarios.append(scenario_name)
    
    return sorted(scenarios)


def load_synthetic_data(scenario_name: str, synthetic_data_path: str = "../../../data/synthetic"):
    """Load synthetic dataset for a given scenario."""
    data_path = Path(synthetic_data_path)
    
    X_file = data_path / f"{scenario_name}_X_full.npy"
    y_file = data_path / f"{scenario_name}_y_full.npy"
    metadata_file = data_path / f"{scenario_name}_metadata.json"
    
    if not all(f.exists() for f in [X_file, y_file, metadata_file]):
        raise FileNotFoundError(f"Missing files for scenario {scenario_name}")
    
    X = np.load(X_file)
    y = np.load(y_file)
    
    # Load metadata
    import json
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return X, y, metadata


def train_models_and_extract_weights(scenario_name: str, X: np.ndarray, y: np.ndarray, 
                                   n_train: int = 500, n_val: int = 200, 
                                   n_test: int = 1000) -> Dict[str, Any]:
    """Train both NIMO models and extract weight information."""
    
    print(f"Training models for scenario {scenario_name}...")
    
    # Prepare data splits
    n_samples = len(X)
    train_indices = np.random.choice(n_samples, n_train, replace=False)
    remaining_indices = np.setdiff1d(np.arange(n_samples), train_indices)
    val_indices = np.random.choice(remaining_indices, n_val, replace=False)
    test_indices = np.setdiff1d(remaining_indices, val_indices)[:n_test]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    # Rebalance training data
    train_indices_balanced = rebalance_train_indices(y, train_indices)
    X_train_balanced = X[train_indices_balanced]
    y_train_balanced = y[train_indices_balanced]
    
    results = {}
    
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
    
    # Store test data for analysis
    results['X_test'] = X_test
    results['y_test'] = y_test
    
    return results


def create_weight_magnitude_plots(scenario_name: str, results: Dict[str, Any], 
                                 feature_names: Optional[List[str]] = None,
                                 output_dir: str = "weight_analysis_plots") -> None:
    """Create weight magnitude plots for both model types."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create feature names if not provided
    if feature_names is None:
        input_dim = results['X_test'].shape[1]
        feature_names = create_feature_names(input_dim)
    
    print(f"\nCreating weight magnitude plots for scenario {scenario_name}...")
    
    # MLP plots
    if results['mlp'] is not None and '_plot_bits' in results['mlp']:
        print("  Creating MLP weight magnitude plots...")
        mlp_weights = results['mlp']['_plot_bits']
        mlp_mags = mlp_first_fc_magnitudes(mlp_weights)
        
        save_path = output_dir / f"{scenario_name}_mlp_weights"
        plot_mlp_weights(
            mlp_mags['W_feat'], 
            mlp_mags['col_L2'],
            feature_names=feature_names,
            title_prefix=f"NIMO-MLP ({scenario_name})",
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
        
        save_path = output_dir / f"{scenario_name}_transformer_weights"
        plot_transformer_weights(
            transformer_mags['bias'],
            transformer_mags['row_L2'],
            feature_names=feature_names,
            title_prefix=f"NIMO-Transformer ({scenario_name})",
            save_path=str(save_path)
        )
        print(f"    Transformer plots saved to: {save_path}")
    else:
        print("  Skipping Transformer plots (no model weights available)")


def main():
    parser = argparse.ArgumentParser(description='Generate weight magnitude plots for synthetic scenarios')
    parser.add_argument('--scenario', type=str, default=None, 
                       help='Specific scenario to analyze (e.g., scenario_A). If None, analyze all.')
    parser.add_argument('--output-dir', type=str, default='weight_analysis_plots',
                       help='Directory to save plots (default: weight_analysis_plots)')
    parser.add_argument('--data-path', type=str, default='../../../data/synthetic',
                       help='Path to synthetic data directory')
    parser.add_argument('--n-train', type=int, default=500,
                       help='Number of training samples')
    parser.add_argument('--n-val', type=int, default=200,
                       help='Number of validation samples')
    parser.add_argument('--n-test', type=int, default=1000,
                       help='Number of test samples')
    
    args = parser.parse_args()
    
    # Discover scenarios
    scenarios = discover_synthetic_scenarios(args.data_path)
    if not scenarios:
        print("No synthetic scenarios found!")
        return
    
    if args.scenario:
        if args.scenario in scenarios:
            scenarios = [args.scenario]
        else:
            print(f"Scenario {args.scenario} not found. Available: {scenarios}")
            return
    
    print(f"Found scenarios: {scenarios}")
    
    # Process each scenario
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Processing scenario: {scenario}")
        print(f"{'='*60}")
        
        try:
            # Load data
            X, y, metadata = load_synthetic_data(scenario, args.data_path)
            print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Train models and extract weights
            results = train_models_and_extract_weights(
                scenario, X, y, 
                n_train=args.n_train, 
                n_val=args.n_val, 
                n_test=args.n_test
            )
            
            # Create weight magnitude plots
            create_weight_magnitude_plots(
                scenario, results, 
                output_dir=args.output_dir
            )
            
        except Exception as e:
            print(f"Error processing scenario {scenario}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("Weight magnitude analysis complete!")
    print(f"All plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
