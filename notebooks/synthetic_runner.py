#!/usr/bin/env python3
"""
Synthetic Experiment Runner
Runs all methods on synthetic datasets only (scenarios A, B, C, D, E, etc.)
"""

import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import our modules
from datasets import DATASETS
from loaders import load_any
from sampling import stratified_with_replacement, rebalance_train_indices, get_class_distribution
from generate_synthetic_data import stratified_train_val_from_pool

# Import methods
from methods.lasso import run_lasso
from methods.lasso_Net import run_lassonet
from methods.nimo_variants import run_nimo, run_nimo_baseline, run_nimo_baseline_scenario, run_nimo_transformer, run_nimo_transformer_scenario
from methods.random_forest import run_random_forest
from methods.neural_net import run_neural_net

# Import NIMO-Transformer variants
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from src.models.nimo_t import make_nimo_t


def run_nimo_transformer_variant(X_tr, y_tr, X_te, y_te, iteration, randomState, X_columns=None, *, X_val=None, y_val=None, variant="strict", scenario_id="unknown"):
    """
    Run a specific NIMO-Transformer variant using the same sophisticated pipeline as the original NIMO_T.
    
    Args:
        X_tr, y_tr: Training data
        X_te, y_te: Test data
        iteration: Iteration number
        randomState: Random seed
        X_columns: Feature names
        X_val, y_val: Validation data
        variant: NIMO-T variant name ("strict", "exploratory", "relaxed", "plus")
        scenario_id: Scenario ID for configuration
        
    Returns:
        dict: Results from the variant
    """
    try:
        # Import the sophisticated training function
        from notebooks.methods.nimo_variants.nimoTransformer_NN import (
            _train_single, TrainingConfig, get_scenario_config
        )
        
        # Get scenario-specific configuration (same as single NIMO_T)
        d = X_tr.shape[1]
        scenario_config = get_scenario_config(scenario_id, d)
        
        # Create config with scenario-specific settings (no variant-specific flags)
        config_dict = scenario_config.__dict__.copy()
        
        # Create final config (variant-specific flags will be handled in model creation)
        config = TrainingConfig(**config_dict)
        
        print(f"🎯 Running NIMO-T {variant} for scenario {scenario_id} with config: "
              f"scenario_name={config.scenario_name}, embed_dim={config.embed_dim}, "
              f"out_scale={config.out_scale}")
        
        # Use the same sophisticated training pipeline as single NIMO_T
        # But with variant-specific model creation
        result = _train_single_with_variant(
            config,
            variant=variant,
            X_train=X_tr,
            y_train=y_tr,
            X_test=X_te,
            y_test=y_te,
            iteration=iteration,
            randomState=randomState,
            X_columns=X_columns,
            X_val=X_val,
            y_val=y_val,
            return_model_bits=False,
            save_artifacts=False,  # Don't save artifacts for variants
            scenario_name=scenario_id,
            artifact_tag=f"{variant}_{iteration}",
            save_if="never",  # Don't save for variants
            cache_policy="reuse",
            artifact_dtype="float32",
            cfg_label=variant
        )
        
        # Add variant information to result
        result["variant"] = variant
        result["model_name"] = f"NIMO_T_{variant.upper()}"
        
        return result
        
    except Exception as e:
        print(f"❌ Error running NIMO-T {variant}: {e}")
        import traceback
        traceback.print_exc()  # Add full traceback
        # Return error result
        return {
            "model_name": f"NIMO_T_{variant.upper()}",
            "iteration": iteration,
            "random_seed": randomState,
            "error": str(e),
            "training_time": 0.0,
            "variant": variant,
            "f1": 0.0,
            "val_f1": 0.0
        }


def _train_single_with_variant(
    cfg,
    *,
    variant: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    iteration: int,
    randomState: int,
    X_columns,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    return_model_bits: bool = False,
    save_artifacts: bool = False,
    artifact_dir: str = "artifacts/nimo_transformer",
    scenario_name: Optional[str] = None,
    artifact_tag: Optional[str] = None,
    save_if: str = "better",
    cache_policy: str = "reuse",
    artifact_dtype: str = "float32",
    cfg_label: Optional[str] = None,
):
    """
    Wrapper around _train_single that patches the model creation to use variants.
    """
    import time
    
    # Import the sophisticated training function
    from notebooks.methods.nimo_variants.nimoTransformer_NN import (
        _train_single, NIMOTransformer
    )
    from src.models.nimo_t import make_nimo_t
    
    # Store original NIMOTransformer class
    original_nimo_transformer = NIMOTransformer
    
    # Set variant-specific flags on the config object
    variant_configs = {
        "strict": {
            "hard_no_self_mask": True,
            "use_companion_shortlist": False,
            "two_hop_attention": False,
            "topk_beta_aware": False,
            "use_residual_head": False,
            "residual_orthogonalize": True,
            "no_harm_margin": 0.005,
        },
        "exploratory": {
            "hard_no_self_mask": True,
            "use_companion_shortlist": False,
            "two_hop_attention": False,
            "topk_beta_aware": False,
            "use_residual_head": False,  # can flip to true in experiments
            "residual_orthogonalize": True,
            "no_harm_margin": 0.005,
        },
        "relaxed": {
            "hard_no_self_mask": False,
            "use_companion_shortlist": False,
            "two_hop_attention": False,
            "topk_beta_aware": False,
            "use_residual_head": False,
            "residual_orthogonalize": False,
            "no_harm_margin": 0.0,
        },
        "plus": {
            "hard_no_self_mask": True,
            "use_companion_shortlist": True,
            "two_hop_attention": True,
            "topk_beta_aware": True,
            "use_residual_head": True,
            "residual_orthogonalize": True,
            "no_harm_margin": 0.01,
        }
    }
    
    # Apply variant-specific configuration
    variant_config = variant_configs.get(variant, variant_configs["strict"])
    for key, value in variant_config.items():
        setattr(cfg, key, value)

    # Create a variant-aware NIMOTransformer factory
    # Use the SAME sophisticated NIMOTransformer class but with variant-specific configurations
    def create_variant_model(d, **kwargs):
        # Define variant-specific configurations
        variant_configs = {
            "strict": {
                "hard_no_self_mask": True,
                "use_companion_shortlist": False,
                "two_hop_attention": False,
                "topk_beta_aware": False,
                "use_residual_head": False,
                "residual_orthogonalize": True,
                "no_harm_margin": 0.005,
            },
            "exploratory": {
                "hard_no_self_mask": True,
                "use_companion_shortlist": False,
                "two_hop_attention": False,
                "topk_beta_aware": False,
                "use_residual_head": False,  # can flip to true in experiments
                "residual_orthogonalize": True,
                "no_harm_margin": 0.005,
            },
            "relaxed": {
                "hard_no_self_mask": False,
                "use_companion_shortlist": False,
                "two_hop_attention": False,
                "topk_beta_aware": False,
                "use_residual_head": False,
                "residual_orthogonalize": False,
                "no_harm_margin": 0.0,
            },
            "plus": {
                "hard_no_self_mask": True,
                "use_companion_shortlist": True,
                "two_hop_attention": True,
                "topk_beta_aware": True,
                "use_residual_head": True,
                "residual_orthogonalize": True,
                "no_harm_margin": 0.01,
            }
        }
        
        # Get variant-specific config
        variant_config = variant_configs.get(variant, variant_configs["strict"])
        
        # Use the original NIMOTransformer class with variant-specific configurations
        model = NIMOTransformer(
            d,
            embed_dim=kwargs.get("embed_dim", 64),
            num_heads=kwargs.get("num_heads", 4),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.1),
            out_scale=kwargs.get("out_scale", 0.4),
            residual_scale=kwargs.get("residual_scale", 0.3),
            use_binary_context=kwargs.get("use_binary_context", True),
            fast_preset=kwargs.get("fast_preset", False),
            use_residual_head=variant_config["use_residual_head"],
            hard_no_self_mask=variant_config["hard_no_self_mask"],
            use_companion_shortlist=variant_config["use_companion_shortlist"],
            two_hop_attention=variant_config["two_hop_attention"],
            topk_beta_aware=variant_config["topk_beta_aware"],
            residual_orthogonalize=variant_config["residual_orthogonalize"],
        )
        
        return model
    
    # Temporarily replace NIMOTransformer with our variant factory
    import notebooks.methods.nimo_variants.nimoTransformer_NN as nimo_module
    nimo_module.NIMOTransformer = create_variant_model
    
    try:
        # Start timing
        start_time = time.time()
        
        # Call the original function with our patched model creation
        result, val_metrics = _train_single(
            cfg,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            iteration=iteration,
            randomState=randomState,
            X_columns=X_columns,
            X_val=X_val,
            y_val=y_val,
            return_model_bits=return_model_bits,
            save_artifacts=save_artifacts,
            artifact_dir=artifact_dir,
            scenario_name=scenario_name,
            artifact_tag=artifact_tag,
            save_if=save_if,
            cache_policy=cache_policy,
            artifact_dtype=artifact_dtype,
            cfg_label=cfg_label
        )
        
        # Add timing information
        training_time = time.time() - start_time
        result["training_time"] = training_time
        
        return result
        
    finally:
        # Restore original NIMOTransformer class
        nimo_module.NIMOTransformer = original_nimo_transformer


def discover_synthetic_scenarios(synthetic_data_path="../data/synthetic", n_train=4000, n_val=1000):
    """
    Dynamically discover all available synthetic scenarios by scanning the data directory.

    Args:
        synthetic_data_path: Path to synthetic data directory
        n_train: Number of training samples per iteration
        n_val: Number of validation samples per iteration

    Returns:
        list: List of discovered synthetic dataset configurations
    """
    synthetic_scenarios = []
    data_path = Path(synthetic_data_path)

    if not data_path.exists():
        print(f"Warning: Synthetic data path {data_path} does not exist")
        return []

    # Look for scenario_* directories or files
    pattern_files = list(data_path.glob("scenario_*_X_full.npy"))

    for pattern_file in pattern_files:
        # Extract scenario ID from filename (e.g., "scenario_A_X_full.npy" -> "A", "scenario_GBX_PERM_X_full.npy" -> "GBX_PERM")
        # Remove "scenario_" prefix and "_X_full" suffix, then split on "_" and take everything except the last part
        name_parts = pattern_file.stem.split("_")  # ["scenario", "A", "X", "full"] or ["scenario", "GBX", "PERM", "X", "full"]
        if len(name_parts) >= 3:
            scenario_id = "_".join(name_parts[1:-2])  # Take everything between "scenario" and "X_full"
        else:
            scenario_id = name_parts[1]  # Fallback for simple cases like "scenario_A_X_full"

        # Check if all required files exist for this scenario
        required_files = [
            f"scenario_{scenario_id}_X_full.npy",
            f"scenario_{scenario_id}_y_full.npy",
            f"scenario_{scenario_id}_idx_pool.npy",
            f"scenario_{scenario_id}_idx_test_big.npy",
            f"scenario_{scenario_id}_metadata.json"
        ]

        all_files_exist = all((data_path / f).exists() for f in required_files)

        if all_files_exist:
            # Create dataset configuration
            scenario_config = {
                "kind": "synthetic",
                "id": scenario_id,
                "path": str(data_path),
                "n_train": n_train,
                "n_val": n_val,
                "desc": f"Synthetic scenario {scenario_id} (auto-discovered)"
            }
            synthetic_scenarios.append(scenario_config)
            print(f"  ✓ Discovered synthetic scenario: {scenario_id}")
        else:
            print(f"  ✗ Incomplete scenario {scenario_id}: missing required files")

    # Sort by scenario ID for consistent ordering
    synthetic_scenarios.sort(key=lambda x: x["id"])

    print(f"Discovered {len(synthetic_scenarios)} synthetic scenarios: {[s['id'] for s in synthetic_scenarios]}")
    return synthetic_scenarios




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
    ]

    # Fix 1: Set scenario_id once at the top
    scenario_id = (dataset_info or {}).get("scenario", "unknown")
    
    results = []

    for method_name, method_func in methods:
        try:
            print(f"    Running {method_name}...")
            start_time = time.time()

            # Run method with consistent interface
            if method_name == "NIMO_MLP":
                # Use scenario-specific configuration for optimal performance
                result = run_nimo_baseline_scenario(
                    X_tr, y_tr, X_te, y_te,
                    scenario_id=scenario_id,
                    iteration=0,  # Will be set by caller
                    randomState=seed,
                    X_columns=feature_names,
                    X_val=X_va,
                    y_val=y_va,
                    save_artifacts=True,
                    scenario_name=scenario_id,
                    save_if="better",
                    cache_policy="reuse"
                )
            elif method_name == "NIMO_T":
                result = run_nimo_transformer_scenario(
                    X_tr, y_tr, X_te, y_te,
                    scenario_id=scenario_id,
                    iteration=0,  # Will be set by caller
                    randomState=seed,
                    X_columns=feature_names,
                    X_val=X_va,
                    y_val=y_va,
                    save_artifacts=True,
                    scenario_name=scenario_id,
                    save_if="better",
                    cache_policy="reuse"
                )
            elif method_name.startswith("NIMO_T_"):
                # Handle NIMO-T variants
                variant = method_name.split("_")[-1].lower()
                result = run_nimo_transformer_variant(
                    X_tr, y_tr, X_te, y_te,
                    iteration=0,  # Will be set by caller
                    randomState=seed,
                    X_columns=feature_names,
                    X_val=X_va,
                    y_val=y_va,
                    variant=variant,
                    scenario_id=scenario_id
                )
            else:
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


def main(n_iterations=30, rebalance_config=None, output_dir="../results/synthetic", n_train=4000, n_val=1000):
    """
    Main experiment runner for synthetic datasets only.

    Args:
        n_iterations: Number of iterations per dataset
        rebalance_config: Rebalancing configuration dict
        output_dir: Output directory for results
        n_train: Number of training samples per iteration
        n_val: Number of validation samples per iteration
    """
    # Record pipeline start time
    pipeline_start_time = time.time()
    pipeline_start_datetime = datetime.now()
    
    # Default rebalancing config
    if rebalance_config is None:
        rebalance_config = {"mode": "undersample", "target_pos": 0.5}

    # Get only synthetic datasets
    all_datasets = discover_synthetic_scenarios(n_train=n_train, n_val=n_val)

    # Build banner from actual methods list
    methods = [
        ("Lasso", run_lasso),
        ("NIMO_MLP", run_nimo_baseline),
        ("NIMO_T", run_nimo_transformer),
        ("NIMO_T_STRICT", lambda *args, **kwargs: run_nimo_transformer_variant(*args, **kwargs, variant="strict")),
        ("NIMO_T_EXPLORATORY", lambda *args, **kwargs: run_nimo_transformer_variant(*args, **kwargs, variant="exploratory")),
        ("NIMO_T_RELAXED", lambda *args, **kwargs: run_nimo_transformer_variant(*args, **kwargs, variant="relaxed")),
        ("NIMO_T_PLUS", lambda *args, **kwargs: run_nimo_transformer_variant(*args, **kwargs, variant="plus")),
        ("RF", run_random_forest),
        ("NN", run_neural_net),
    ]
    method_names = [name for name, _ in methods]
    
    print("=" * 80)
    print("SYNTHETIC EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Pipeline started at: {pipeline_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets: {len(all_datasets)} (synthetic only)")
    print(f"Iterations per dataset: {n_iterations}")
    print(f"Sample sizes: {n_train} train, {n_val} val")
    print(f"Rebalancing: {rebalance_config}")
    print(f"Methods: {', '.join(method_names)}")
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
                'A': 'Linear (low-dim, 20 features)',
                'B': 'Linear (high-dim, 200 features)',
                'C': 'Linear + univariate nonlinearity (low-dim, 20 features)',
                'D': 'Linear + interactions + nonlinearity (high-dim, 200 features)',
                'E': 'Purely nonlinear (medium-dim, 50 features)',
                'F': 'High-dimensional with four interactions',
                'G': 'Medium-dimensional with complex interactions',
                'H': 'Low-dimensional with noise',
                'I': 'High-dimensional with sparsity'
            }

            # Add scenario-like fields for unified plotting
            dataset_info.update({
                "scenario": entry["id"],  # Use dataset_id as scenario
                "scenario_description": scenario_descriptions.get(entry["id"], entry.get("desc", "")),
                "scenario_title": f"Scenario {entry['id']}: {scenario_descriptions.get(entry['id'], entry.get('desc', ''))}"
            })

            # Add true support info for synthetic data
            true_support = meta.get("true_support", [])
            beta_nonzero = meta.get("beta_nonzero", {})
            n_features = meta.get("p", len(true_support))
            
            # Create full-length beta_true vector
            beta_true_full = [0.0] * n_features
            for idx, val in beta_nonzero.items():
                if 0 <= int(idx) < n_features:
                    beta_true_full[int(idx)] = float(val)
            
            dataset_info.update({
                "n_true_features": len(true_support),
                "true_support": json.dumps(true_support),
                "beta_true": json.dumps(beta_true_full),
                "b0_true": meta.get("b0", 0.0)
            })

            # Run iterations
            for it in range(n_iterations):
                print(f"\n  --- Iteration {it + 1}/{n_iterations} ---")

                seed = 42 + 1000 * it

                # Use robust stratified splitter with guards against empty splits
                target_pos = rebalance_config.get("target_pos", 0.5) if rebalance_config["mode"] != "none" else None
                if target_pos is not None:
                    # Use robust splitter with rebalancing
                    idx_tr, idx_va = stratified_train_val_from_pool(
                        y, idx_pool,
                        n_train=entry["n_train"],
                        n_val=entry["n_val"],
                        target_pos=target_pos,
                        seed=seed
                    )
                else:
                    # Use robust splitter without rebalancing (preserve natural distribution)
                    idx_tr, idx_va = stratified_train_val_from_pool(
                        y, idx_pool,
                        n_train=entry["n_train"],
                        n_val=entry["n_val"],
                        target_pos=None,  # Will use natural distribution
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
    print("SYNTHETIC EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Pipeline started at: {pipeline_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Pipeline finished at: {pipeline_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Pipeline duration: {pipeline_duration_minutes:.2f} minutes ({pipeline_duration_seconds:.1f} seconds)")
    print(f"Total results: {len(all_results)}")
    print(f"Datasets processed: {len(all_datasets)}")
    print(f"Iterations per dataset: {n_iterations}")
    print(f"Results saved to: {output_file}")

    # Print per-dataset summary
    if len(all_results) > 0 and len(df) > 0:
        print(f"\nPer-dataset summary:")
        for dataset_id in df['dataset_id'].unique():
            dataset_results = df[df['dataset_id'] == dataset_id]
            if 'error' in df.columns:
                n_success = len(dataset_results[dataset_results['error'].isna()])
            else:
                n_success = len(dataset_results)
            n_total = len(dataset_results)
            print(f"  {dataset_id}: {n_success}/{n_total} successful runs")
    else:
        print("No results to summarize (0 datasets or empty results).")

    # Print per-method summary
    if len(all_results) > 0 and len(df) > 0:
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
    else:
        print("No results to summarize (0 datasets or empty results).")

    print(f"\n✓ Synthetic experiment completed successfully!")
    print(f"✓ Total pipeline time: {pipeline_duration_minutes:.2f} minutes")
    return df


if __name__ == "__main__":
    # Run with default settings
    df = main(n_iterations=20, n_train=1400, n_val=600)  # Start with 1 iteration for testing