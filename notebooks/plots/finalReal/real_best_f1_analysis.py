#!/usr/bin/env python3
"""
Real Dataset Best F1 Analysis
Extracts and displays the best F1 scores for each method across all real datasets
"""

import pandas as pd
import numpy as np
import os

def get_canonical_dataset_key(dataset_description):
    """Map dataset description to canonical short key."""
    mapping = {
        "Two-Moon Dataset (binary classification)": "moon",
        "Boston Housing (binary classification)": "boston", 
        "California Housing (binary classification)": "housing",
        "Diabetes Progression (binary classification)": "diabetes",
    }
    return mapping.get(dataset_description, dataset_description.lower())

def get_dataset_title(dataset_key):
    """Get display title for dataset."""
    titles = {
        "moon": "Two-Moons Dataset",
        "boston": "Boston Housing Dataset", 
        "housing": "California Housing Dataset",
        "diabetes": "Diabetes Progression Dataset",
    }
    return titles.get(dataset_key, dataset_key)

def analyze_best_f1_scores(df):
    """Analyze best F1 scores for each method across all real datasets."""
    print("=== Real Dataset Best F1 Analysis ===")
    print("Extracting best F1 scores for each method across all real datasets")
    print()
    
    # Get unique real datasets
    real_datasets = sorted(df['dataset_description'].unique().tolist())
    df_real = df[df['dataset_description'].isin(real_datasets)].copy()
    
    print(f"Found {len(real_datasets)} real datasets:")
    for dataset in real_datasets:
        dataset_key = get_canonical_dataset_key(dataset)
        dataset_title = get_dataset_title(dataset_key)
        print(f"  - {dataset_key}: {dataset_title}")
    print()
    
    # Get all unique methods
    all_methods = sorted(df_real['model_name'].unique())
    print(f"Found {len(all_methods)} methods: {all_methods}")
    print()
    
    # Create results storage
    results = []
    
    # Analyze each dataset
    for dataset in real_datasets:
        dataset_key = get_canonical_dataset_key(dataset)
        dataset_title = get_dataset_title(dataset_key)
        
        print(f"=== {dataset_title} ({dataset_key}) ===")
        
        # Filter data for this dataset
        dataset_data = df_real[df_real['dataset_description'] == dataset].copy()
        
        if dataset_data.empty:
            print(f"No data found for {dataset_key}")
            continue
        
        # Get best F1 for each method in this dataset
        dataset_results = []
        
        for method in all_methods:
            method_data = dataset_data[dataset_data['model_name'] == method]
            
            if method_data.empty:
                print(f"  {method}: No data available")
                dataset_results.append({
                    'dataset': dataset_key,
                    'dataset_title': dataset_title,
                    'method': method,
                    'best_f1': None,
                    'best_iteration': None,
                    'total_iterations': 0
                })
                continue
            
            # Check if F1 column exists and has valid data
            if 'f1' not in method_data.columns:
                print(f"  {method}: No F1 column found")
                dataset_results.append({
                    'dataset': dataset_key,
                    'dataset_title': dataset_title,
                    'method': method,
                    'best_f1': None,
                    'best_iteration': None,
                    'total_iterations': len(method_data)
                })
                continue
            
            # Find best F1 score
            valid_f1_mask = method_data['f1'].notna()
            if not valid_f1_mask.any():
                print(f"  {method}: No valid F1 scores")
                dataset_results.append({
                    'dataset': dataset_key,
                    'dataset_title': dataset_title,
                    'method': method,
                    'best_f1': None,
                    'best_iteration': None,
                    'total_iterations': len(method_data)
                })
                continue
            
            # Get best F1 and iteration
            best_idx = method_data['f1'].idxmax()
            best_row = method_data.loc[best_idx]
            best_f1 = best_row['f1']
            best_iteration = best_row['iteration']
            total_iterations = len(method_data)
            
            print(f"  {method}: Best F1 = {best_f1:.4f} (iteration {best_iteration}/{total_iterations})")
            
            dataset_results.append({
                'dataset': dataset_key,
                'dataset_title': dataset_title,
                'method': method,
                'best_f1': best_f1,
                'best_iteration': best_iteration,
                'total_iterations': total_iterations
            })
        
        # Find best method for this dataset
        valid_results = [r for r in dataset_results if r['best_f1'] is not None]
        if valid_results:
            best_method_result = max(valid_results, key=lambda x: x['best_f1'])
            print(f"  🏆 Best method: {best_method_result['method']} (F1 = {best_method_result['best_f1']:.4f})")
        else:
            print(f"  ⚠️  No valid F1 scores found for any method")
        
        results.extend(dataset_results)
        print()
    
    # Create summary DataFrame
    results_df = pd.DataFrame(results)
    
    # Create summary table by method
    print("=" * 80)
    print("SUMMARY BY METHOD")
    print("=" * 80)
    
    for method in all_methods:
        method_results = results_df[results_df['method'] == method]
        valid_results = method_results[method_results['best_f1'].notna()]
        
        print(f"\n{method}:")
        if valid_results.empty:
            print("  No valid F1 scores found")
            continue
        
        # Show best F1 for each dataset
        for _, row in valid_results.iterrows():
            print(f"  {row['dataset']}: F1 = {row['best_f1']:.4f} (iter {row['best_iteration']})")
        
        # Calculate statistics
        f1_scores = valid_results['best_f1'].values
        print(f"  Statistics: Mean = {np.mean(f1_scores):.4f}, Std = {np.std(f1_scores):.4f}")
        print(f"  Best dataset: {valid_results.loc[valid_results['best_f1'].idxmax(), 'dataset']} (F1 = {valid_results['best_f1'].max():.4f})")
        print(f"  Worst dataset: {valid_results.loc[valid_results['best_f1'].idxmin(), 'dataset']} (F1 = {valid_results['best_f1'].min():.4f})")
    
    # Create summary table by dataset
    print("\n" + "=" * 80)
    print("SUMMARY BY DATASET")
    print("=" * 80)
    
    for dataset in real_datasets:
        dataset_key = get_canonical_dataset_key(dataset)
        dataset_title = get_dataset_title(dataset_key)
        dataset_results = results_df[results_df['dataset'] == dataset_key]
        valid_results = dataset_results[dataset_results['best_f1'].notna()]
        
        print(f"\n{dataset_title} ({dataset_key}):")
        if valid_results.empty:
            print("  No valid F1 scores found")
            continue
        
        # Sort by F1 score (descending)
        valid_results = valid_results.sort_values('best_f1', ascending=False)
        
        for _, row in valid_results.iterrows():
            print(f"  {row['method']}: F1 = {row['best_f1']:.4f} (iter {row['best_iteration']})")
        
        # Show winner
        winner = valid_results.iloc[0]
        print(f"  🏆 Winner: {winner['method']} (F1 = {winner['best_f1']:.4f})")
    
    # Create overall ranking
    print("\n" + "=" * 80)
    print("OVERALL METHOD RANKING")
    print("=" * 80)
    
    method_stats = []
    for method in all_methods:
        method_results = results_df[results_df['method'] == method]
        valid_results = method_results[method_results['best_f1'].notna()]
        
        if valid_results.empty:
            continue
        
        f1_scores = valid_results['best_f1'].values
        wins = 0
        for dataset in real_datasets:
            dataset_key = get_canonical_dataset_key(dataset)
            dataset_results = results_df[results_df['dataset'] == dataset_key]
            dataset_valid = dataset_results[dataset_results['best_f1'].notna()]
            if not dataset_valid.empty:
                best_method = dataset_valid.loc[dataset_valid['best_f1'].idxmax(), 'method']
                if best_method == method:
                    wins += 1
        
        method_stats.append({
            'method': method,
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'wins': wins,
            'datasets': len(f1_scores)
        })
    
    # Sort by mean F1 score
    method_stats = sorted(method_stats, key=lambda x: x['mean_f1'], reverse=True)
    
    print("\nRanking by average F1 score across all datasets:")
    for i, stats in enumerate(method_stats, 1):
        print(f"{i}. {stats['method']}: Mean F1 = {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f} "
              f"(wins: {stats['wins']}/{stats['datasets']} datasets)")
    
    # Save results to CSV
    os.makedirs("best_f1_analysis", exist_ok=True)
    
    # Save detailed results
    detailed_file = "best_f1_analysis/real_best_f1_detailed.csv"
    results_df.to_csv(detailed_file, index=False)
    print(f"\n✓ Detailed results saved to {detailed_file}")
    
    # Save summary by method
    method_summary = []
    for method in all_methods:
        method_results = results_df[results_df['method'] == method]
        valid_results = method_results[method_results['best_f1'].notna()]
        
        if valid_results.empty:
            continue
        
        f1_scores = valid_results['best_f1'].values
        wins = 0
        for dataset in real_datasets:
            dataset_key = get_canonical_dataset_key(dataset)
            dataset_results = results_df[results_df['dataset'] == dataset_key]
            dataset_valid = dataset_results[dataset_results['best_f1'].notna()]
            if not dataset_valid.empty:
                best_method = dataset_valid.loc[dataset_valid['best_f1'].idxmax(), 'method']
                if best_method == method:
                    wins += 1
        
        method_summary.append({
            'method': method,
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'min_f1': np.min(f1_scores),
            'max_f1': np.max(f1_scores),
            'wins': wins,
            'datasets': len(f1_scores)
        })
    
    method_summary_df = pd.DataFrame(method_summary)
    method_summary_df = method_summary_df.sort_values('mean_f1', ascending=False)
    
    summary_file = "best_f1_analysis/real_best_f1_summary.csv"
    method_summary_df.to_csv(summary_file, index=False)
    print(f"✓ Summary results saved to {summary_file}")
    
    print(f"\n🎉 Best F1 analysis completed!")
    print("Results saved in best_f1_analysis/ directory")

def main():
    """Main function to run best F1 analysis for all real datasets."""
    # Load data
    results_path = '../../../results/real/experiment_results.csv'
    df = pd.read_csv(results_path)
    
    print(f"Loaded {len(df)} total experiments")
    
    # Run analysis
    analyze_best_f1_scores(df)

if __name__ == "__main__":
    main()
