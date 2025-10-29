"""
Utility functions for extracting and analyzing weight magnitudes from NIMO models.

This module provides functions to extract weight information from both NIMO-MLP 
(AdaptiveRidgeLogisticRegression) and NIMO-Transformer models for visualization.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple


def mlp_first_fc_magnitudes(model_weights: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Extract first FC layer weight magnitudes for NIMO-MLP (AdaptiveRidgeLogisticRegression).
    
    Args:
        model_weights: Dictionary containing 'fc1_weight', 'input_dim', 'n_bits'
        
    Returns:
        Dictionary with 'W_feat', 'col_L2', 'col_L1' arrays
    """
    W = model_weights['fc1_weight']  # (h1, p+n_bits)
    p = model_weights['input_dim']
    
    # Debug: check if W is a list or numpy array
    if isinstance(W, list):
        W = np.array(W)
    
    W_feat = W[:, :p]  # only real features
    col_L2 = np.linalg.norm(W_feat, axis=0)  # per-feature magnitude
    col_L1 = np.mean(np.abs(W_feat), axis=0)  # alternative metric
    
    return {
        "W_feat": W_feat,
        "col_L2": col_L2, 
        "col_L1": col_L1
    }


def transformer_first_token_bias_magnitudes(model_weights: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Extract first token bias magnitudes for NIMO-Transformer.
    
    Args:
        model_weights: Dictionary containing 'feature_embed', 'binary_proj_weight', 'binary_codes'
        
    Returns:
        Dictionary with 'bias', 'row_L2', 'row_L1' arrays
    """
    E = model_weights['feature_embed']  # (d, emb)
    
    # Debug: check if arrays are lists and convert
    if isinstance(E, list):
        E = np.array(E)
    
    if model_weights['binary_proj_weight'] is not None and model_weights['binary_codes'] is not None:
        codes = model_weights['binary_codes']  # (d, n_bits)
        P = model_weights['binary_proj_weight']  # (emb, n_bits)
        
        if isinstance(codes, list):
            codes = np.array(codes)
        if isinstance(P, list):
            P = np.array(P)
            
        B = codes @ P.T  # (d, emb)
    else:
        B = np.zeros_like(E)
    
    bias = E + B  # per-feature bias into encoder
    row_L2 = np.linalg.norm(bias, axis=1)  # magnitude per feature
    row_L1 = np.mean(np.abs(bias), axis=1)
    
    return {
        "bias": bias,
        "row_L2": row_L2,
        "row_L1": row_L1
    }


def transformer_activation_importance(model, X_np: np.ndarray) -> np.ndarray:
    """
    Compute mean absolute correction per feature on a batch (activation-based view).
    
    Args:
        model: NIMOTransformer model instance
        X_np: Input data as numpy array
        
    Returns:
        Per-feature average absolute correction
    """
    with torch.no_grad():
        X = torch.tensor(X_np, dtype=torch.float32)
        g = model.corrections(X, detach=True).cpu().numpy()  # (N, d)
        return np.mean(np.abs(g), axis=0)  # per-feature avg |correction|


def create_feature_names(input_dim: int, feature_names: Optional[list] = None) -> list:
    """
    Create feature names for plotting, including self-features if applicable.
    
    Args:
        input_dim: Number of input features
        feature_names: Optional list of base feature names
        
    Returns:
        List of feature names for plotting
    """
    if feature_names is not None and len(feature_names) == input_dim:
        return feature_names
    
    # If we have self-features, we need to determine the base feature count
    # Common self-feature patterns: ["x2", "sin", "tanh", "arctan"]
    # This is a heuristic - in practice, you'd want to track this during data generation
    base_features = input_dim // 5 if input_dim > 10 else input_dim  # rough estimate
    
    if feature_names is not None and len(feature_names) == base_features:
        # Expand with self-feature suffixes
        expanded_names = []
        for name in feature_names:
            expanded_names.append(name)  # original
            expanded_names.append(f"{name}_x2")  # squared
            expanded_names.append(f"{name}_sin")  # sin
            expanded_names.append(f"{name}_tanh")  # tanh
            expanded_names.append(f"{name}_arctan")  # arctan
        return expanded_names[:input_dim]
    
    # Fallback: generic names
    return [f"feature_{i}" for i in range(input_dim)]


def plot_mlp_weights(W_feat: np.ndarray, col_L2: np.ndarray, 
                    feature_names: Optional[list] = None,
                    title_prefix: str = "NIMO-MLP",
                    save_path: Optional[str] = None) -> None:
    """
    Create plots for MLP first FC layer weights.
    
    Args:
        W_feat: Feature weight matrix (h1, p)
        col_L2: Per-feature L2 norms
        feature_names: Optional feature names for x-axis
        title_prefix: Prefix for plot titles
        save_path: Optional path to save plots
    """
    import matplotlib.pyplot as plt
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(W_feat.shape[1])]
    
    # Stem plot of per-feature magnitudes
    plt.figure(figsize=(12, 4))
    try:
        plt.stem(col_L2, use_line_collection=True)
    except TypeError:
        # Fallback for older matplotlib versions
        plt.stem(col_L2)
    plt.title(f"{title_prefix}: First FC Weight Magnitudes")
    plt.xlabel("Feature")
    plt.ylabel("L2 Magnitude")
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_stem.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Heatmap of weight magnitudes
    plt.figure(figsize=(10, 6))
    im = plt.imshow(np.abs(W_feat), aspect='auto', cmap='viridis')
    plt.colorbar(im, label='|weight|')
    plt.title(f"{title_prefix}: Weight Magnitude Heatmap")
    plt.xlabel("Feature")
    plt.ylabel("Hidden Unit")
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.subplots_adjust(bottom=0.15, right=0.85)
    
    if save_path:
        plt.savefig(f"{save_path}_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_transformer_weights(bias: np.ndarray, row_L2: np.ndarray,
                           feature_names: Optional[list] = None,
                           title_prefix: str = "NIMO-Transformer",
                           save_path: Optional[str] = None) -> None:
    """
    Create plots for Transformer token bias magnitudes.
    
    Args:
        bias: Per-feature bias matrix (d, emb)
        row_L2: Per-feature L2 norms
        feature_names: Optional feature names for x-axis
        title_prefix: Prefix for plot titles
        save_path: Optional path to save plots
    """
    import matplotlib.pyplot as plt
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(row_L2))]
    
    # Stem plot of per-feature bias magnitudes
    plt.figure(figsize=(12, 4))
    try:
        plt.stem(row_L2, use_line_collection=True)
    except TypeError:
        # Fallback for older matplotlib versions
        plt.stem(row_L2)
    plt.title(f"{title_prefix}: Token Bias Magnitudes")
    plt.xlabel("Feature")
    plt.ylabel("L2 Magnitude")
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_stem.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Heatmap of bias magnitudes
    plt.figure(figsize=(10, 6))
    im = plt.imshow(np.abs(bias), aspect='auto', cmap='viridis')
    plt.colorbar(im, label='|bias|')
    plt.title(f"{title_prefix}: |feature_embed + binary_proj|")
    plt.xlabel("Feature j")
    plt.ylabel("Embedding dimension")
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.subplots_adjust(bottom=0.15, right=0.85)
    
    if save_path:
        plt.savefig(f"{save_path}_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_activation_importance(activation_importance: np.ndarray,
                             feature_names: Optional[list] = None,
                             title_prefix: str = "NIMO-Transformer",
                             save_path: Optional[str] = None) -> None:
    """
    Create plot for activation-based feature importance.
    
    Args:
        activation_importance: Per-feature mean absolute corrections
        feature_names: Optional feature names for x-axis
        title_prefix: Prefix for plot title
        save_path: Optional path to save plot
    """
    import matplotlib.pyplot as plt
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(activation_importance))]
    
    plt.figure(figsize=(12, 4))
    try:
        plt.stem(activation_importance, use_line_collection=True)
    except TypeError:
        # Fallback for older matplotlib versions
        plt.stem(activation_importance)
    plt.title(f"{title_prefix}: Mean |correction| per feature (activation-based)")
    plt.xlabel("Feature index")
    plt.ylabel("Mean |correction|")
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_activation.png", dpi=300, bbox_inches='tight')
    plt.show()
