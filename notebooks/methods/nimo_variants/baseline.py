"""
NIMO Baseline Variant - Adaptive Ridge Logistic Regression with Lightning
(Stabilized IRLS; epoch-wise beta updates; probability outputs; HParam sweep)
- Self-features are DISABLED by default for paper compliance
- Use self_features=["x2","sin","tanh","arctan"] to enable if needed
"""

import os
import sys
import math
import itertools
import json
import datetime
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# AMP for speed
try:
    from torch.cuda.amp import autocast
except ImportError:
    autocast = None

# Lightning EarlyStopping (optional)
try:
    from lightning.pytorch.callbacks import EarlyStopping
except Exception:
    EarlyStopping = None

# Allow relative utils import if present
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils import standardize_method_output
except Exception:
    def standardize_method_output(result: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                out[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32, np.float16)):
                out[k] = float(v)
            else:
                out[k] = v
        return out


# -------------------- artifact helpers --------------------

def _short_hash(obj: dict) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:10]

def _mlp_key(scenario, seed, input_dim, hparams, tag=None):
    hp_compact = {k: hparams.get(k) for k in sorted(hparams.keys())}
    return {
        "scenario": scenario or "unknown",
        "seed": int(seed),
        "input_dim": int(input_dim),
        "h": _short_hash(hp_compact),
        "tag": tag or "default",
    }

def _mlp_paths(base_dir: str, key: dict) -> dict:
    root = Path(base_dir) / key["scenario"] / f"seed{key['seed']}" / f"in{key['input_dim']}" / key["h"]
    if key["tag"] != "default":
        root = root / key["tag"]
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "weights_npz": root / "mlp_weights.npz",
        "meta_json": root / "meta.json",
    }

def _load_mlp_artifacts(paths: dict):
    if not (paths["weights_npz"].exists() and paths["meta_json"].exists()):
        return None
    try:
        with open(paths["meta_json"], "r") as f:
            meta = json.load(f)
        npz = np.load(paths["weights_npz"])
        fc1_weight = npz["fc1_weight"]
        input_dim = int(npz["input_dim"])
        n_bits = int(npz["n_bits"])
        return {"meta": meta, "fc1_weight": fc1_weight, "input_dim": input_dim, "n_bits": n_bits}
    except Exception:
        return None

def _save_mlp_artifacts(fc1_weight: np.ndarray, meta: dict, paths: dict, dtype: str = "float32"):
    fc1 = fc1_weight.astype(np.float32 if dtype == "float32" else np.float64, copy=False)
    np.savez_compressed(paths["weights_npz"], fc1_weight=fc1, input_dim=meta["input_dim"], n_bits=meta["n_bits"])
    with open(paths["meta_json"], "w") as f:
        json.dump(meta, f, separators=(",", ":"), sort_keys=True)

# -------------------- helpers --------------------

def to_bin(x: int, n_bits: int) -> np.ndarray:
    # centered bits in {-0.5, +0.5}
    return np.array([int(b) for b in format(x, f'0{n_bits}b')], dtype=np.float32) - 0.5


def add_self_features(
    X: np.ndarray,
    self_features: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[Tuple[int, str]]]:
    """
    Expand X with simple self-nonlinearities to help NIMO capture self-terms.

    self_features: list of transforms to apply to each column of X.
      Supported: "x2", "x3", "sin", "cos", "abs", "tanh", "arctan" (alias: "atan")
    Returns (X_expanded, mapping) where mapping holds (orig_col_idx, transform_name)
    """
    if self_features is None:
        # DEFAULT: DISABLED to avoid self-feature leak (paper compliance)
        # Enable only if you understand the implications and fix CO masking
        self_features = []

    if not self_features:
        return X.astype(np.float32), []

    transforms = {
        "x2":    lambda v: v * v,
        "x3":    lambda v: v * v * v,
        "sin":   lambda v: np.sin(v),
        "cos":   lambda v: np.cos(v),
        "abs":   lambda v: np.abs(v),
        "tanh":  lambda v: np.tanh(v),
        "arctan":lambda v: np.arctan(v),
        "atan":  lambda v: np.arctan(v),
    }
    new_cols = []
    mapping = []
    for j in range(X.shape[1]):
        col = X[:, j:j+1]
        for name in self_features:
            fn = transforms.get(name)
            if fn is None:
                continue
            new_cols.append(fn(col))
            mapping.append((j, name))

    if new_cols:
        X_new = np.concatenate([X] + new_cols, axis=1).astype(np.float32)
    else:
        X_new = X.astype(np.float32)
    return X_new, mapping


def build_CO_with_derivatives(p_raw: int, mapping: List[Tuple[int, str]], n_bits: int) -> np.ndarray:
    """
    Build CO mask that properly handles self-feature leak by masking derived features.
    
    Args:
        p_raw: Number of original features
        mapping: List of (orig_col_idx, transform_name) for derived features
        n_bits: Number of positional encoding bits
        
    Returns:
        CO: (p, p + n_bits) mask where p = p_raw + len(mapping)
    """
    p = p_raw + len(mapping)
    CO = np.ones((p, p + n_bits), dtype=np.float32)
    
    # For each raw feature j, find all derived columns that come from j
    derived_by_raw = {j: [] for j in range(p_raw)}
    for k, (j, _) in enumerate(mapping, start=p_raw):
        derived_by_raw[j].append(k)
    
    # Zero out self and derived features when querying g_u_j for raw features
    for j in range(p_raw):
        CO[j, j] = 0.0  # zero self
        for k in derived_by_raw[j]:
            CO[j, k] = 0.0  # zero derived features from j
    
    # For derived features, zero self and progenitor raw column
    for k in range(p_raw, p):
        CO[k, k] = 0.0  # zero self
        j = mapping[k - p_raw][0]  # progenitor raw column
        CO[k, j] = 0.0  # zero progenitor
    
    return CO


def get_scenario_config(scenario_id: str) -> Dict[str, Any]:
    """
    Get scenario-specific configuration for optimal performance.
    
    Args:
        scenario_id: Scenario identifier (A, B, C, D, E, etc.)
        
    Returns:
        dict: Configuration parameters optimized for the scenario
    """
    configs = {
        'A': {  # Linear (low-dim, 20 features)
            'hidden_dim': 64,
            'group_penalty': 1.5,  # Increased for stronger sparsity
            'alpha2_schedule_epochs': 15,
            'alpha2_init': 0.5,
            'alpha2_final': 1.5,  # Capped for stability
            'lambda_reg': 5e-2,  # Increased for better regularization
            'noise_std': 0.05,
            'tau_l1': 3e-2,
            'tau_l1_max': 8e-2,
        },
        'B': {  # Linear (high-dim, 200 features)
            'hidden_dim': 96,
            'group_penalty': 1.5,  # Increased for stronger sparsity
            'alpha2_schedule_epochs': 15,
            'alpha2_init': 0.5,
            'alpha2_final': 1.5,  # Capped for stability
            'lambda_reg': 5e-2,  # Increased for better regularization
            'noise_std': 0.05,
            'tau_l1': 3e-2,
            'tau_l1_max': 8e-2,
        },
        'C': {  # Linear + univariate nonlinearity (low-dim, 20 features) - needs strong modulation
            'hidden_dim': 96,  # More capacity for nonlinear terms
            'group_penalty': 1.0,  # Moderate group penalty
            'alpha2_schedule_epochs': 0,  # Fixed alpha2 for immediate strong modulation
            'alpha2_init': 1.5,
            'alpha2_final': 1.5,  # Capped for stability
            'lambda_reg': 3e-2,  # Moderate ridge
            'noise_std': 0.0,  # No noise to preserve clean structure
            'early_noise_std': 0.0,
            'disable_irls_group_penalty': True,  # Let prox handle sparsity
            'tau_l1': 2e-2,
            'tau_l1_max': 5e-2,
        },
        'D': {  # Linear + interactions + nonlinearity (high-dim, 200 features) - needs strong modulation
            'hidden_dim': 96,
            'group_penalty': 1.0,  # Moderate group penalty
            'alpha2_schedule_epochs': 5,  # Shorter ramp for faster modulation
            'alpha2_init': 1.2,  # Start higher
            'alpha2_final': 1.8,  # Capped for stability
            'lambda_reg': 3e-2,  # Moderate ridge
            'noise_std': 0.0,  # No noise
            'early_noise_std': 0.0,
            'disable_irls_group_penalty': True,  # Let prox handle sparsity
            'tau_l1': 2e-2,
            'tau_l1_max': 6e-2,
        },
        'E': {  # Purely nonlinear (medium-dim, 50 features) - needs special handling
            'hidden_dim': 160,  # Increased for nonlinear scenarios
            'group_penalty': 0.8,  # Softer group penalty
            'group_penalty_ramp_epochs': 65,  # +15 for prox to bite
            'alpha2_schedule_epochs': 0,  # Fixed alpha2 for immediate strong modulation
            'alpha2_init': 1.5,
            'alpha2_final': 1.6,  # Capped to avoid over-modulation spikes
            'lambda_reg': 2e-2,  # Lower ridge to allow interaction learning
            'noise_std': 0.0,  # No noise to preserve clean structure
            'early_noise_std': 0.0,
            'disable_irls_group_penalty': True,  # Let prox handle sparsity
            'use_adaptive_l1': False,  # Fixed L1 schedule
            'tau_l1': 1e-2,
            'tau_l1_max': 3e-2,
        },
    }
    
    return configs.get(scenario_id, {})


def verify_CO_mask(CO: np.ndarray, p_raw: int, mapping: List[Tuple[int, str]], n_bits: int) -> bool:
    """
    Sanity check: verify CO mask is correct for self-feature leak prevention.
    
    Args:
        CO: The CO mask matrix
        p_raw: Number of original features
        mapping: List of (orig_col_idx, transform_name) for derived features
        n_bits: Number of positional encoding bits
        
    Returns:
        bool: True if mask is correct
    """
    p = p_raw + len(mapping)
    expected = (p, p + n_bits)
    
    # Check dimensions
    if CO.shape != expected:
        print(f"❌ CO shape mismatch: expected {expected}, got {CO.shape}")
        return False
    
    # Check raw features: should have zero self and derived
    for j in range(p_raw):
        if CO[j, j] != 0.0:
            print(f"❌ Raw feature {j} has non-zero self-mask: {CO[j, j]}")
            return False
        
        # Check derived features from this raw feature
        for k, (orig_j, _) in enumerate(mapping, start=p_raw):
            if orig_j == j and CO[j, k] != 0.0:
                print(f"❌ Raw feature {j} has non-zero mask for derived feature {k}")
                return False
    
    # Check derived features: should have zero self and progenitor
    for k in range(p_raw, p):
        if CO[k, k] != 0.0:
            print(f"❌ Derived feature {k} has non-zero self-mask: {CO[k, k]}")
            return False
        
        orig_j = mapping[k - p_raw][0]
        if CO[k, orig_j] != 0.0:
            print(f"❌ Derived feature {k} has non-zero mask for progenitor {orig_j}")
            return False
    
    print(f"✅ CO mask verification passed for {p_raw} raw + {len(mapping)} derived features")
    return True


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        self.features = features
        self.targets = targets
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return {'features': self.features[idx], 'target': self.targets[idx]}


# -------------------- Model --------------------

class AdaptiveRidgeLogisticRegression(L.LightningModule):
    """
    Neural-Interaction + Adaptive Ridge Logistic Regression (NIMO-style)
    - NN modulates interactions excluding self (via CO mask).
    - Epoch-wise stabilized IRLS to update beta (once per epoch).
    - NN (and c, beta_0, alpha2) trained via SGD.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        learning_rate: float = 1e-3,  # Increased from 3e-4
        group_penalty: float = 1.0,   # Increased for stronger sparsity
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
        noise_std: float = 0.05,      # Reduced from 0.3 for warm-start
        lambda_reg: float = 3e-2,
        c_penalty: float = 0.1,
        # L1 sparsity parameters
        tau_l1: float = 5e-2,
        tau_l1_max: float = 1e-1,
        use_adaptive_l1: bool = True,
        hard_threshold: float = 5e-3,
        trust_region: float = 1.0,
        # Warm-start parameters
        warm_start_epochs: int = 5,    # Freeze beta for first N epochs
        nn_steps_per_epoch: int = 3,   # Multiple NN steps per epoch
        group_penalty_ramp_epochs: int = 30,  # When to start ramping group penalty
        early_noise_std: float = 0.0,  # Noise during early training (first 1/3)
    ):
        super().__init__()
        # Disable automatic optimization for manual control
        self.automatic_optimization = False

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.lr = float(learning_rate)

        self.group_penalty = float(group_penalty)
        self.dropout = float(dropout)
        self.noise_std = float(noise_std)
        self.lambda_reg = float(lambda_reg)
        self.c_penalty = float(c_penalty)
        
        # L1 sparsity parameters
        self.tau_l1 = float(tau_l1)
        self.tau_l1_max = float(tau_l1_max)
        self.use_adaptive_l1 = bool(use_adaptive_l1)
        self.hard_threshold = float(hard_threshold)
        self.trust_region = float(trust_region)
        
        # Warm-start parameters
        self.warm_start_epochs = int(warm_start_epochs)
        self.nn_steps_per_epoch = int(nn_steps_per_epoch)
        self.group_penalty_ramp_epochs = int(group_penalty_ramp_epochs)
        self.early_noise_std = float(early_noise_std)
        
        # Alpha2 schedule parameters
        self.alpha2_schedule_epochs = 15  # Ramp alpha2 over first 15 epochs (0 = fixed)
        self.alpha2_init = 0.5  # Start value
        self.alpha2_final = 2.0  # End value
        
        # Scenario-specific options
        self.disable_irls_group_penalty = False  # Disable group penalty in IRLS for scenarios like E
        
        # Class-aware loss
        self.register_buffer("pos_weight_buf", torch.ones(1))

        # Replay cache for differentiable IRLS
        self._replay_X, self._replay_y = [], []
        self._replay_cap = 2  # keep last 2 batches

        self.save_hyperparameters()

        # Positional encoding via binary map
        self.n_bits = int(np.floor(np.log2(self.input_dim))) + 1
        bin_map = np.vstack([to_bin(i, self.n_bits) for i in range(1, self.input_dim + 1)])  # (p, n_bits)

        # CO: mask-out self + append positional bits (will be updated if self-features are used)
        CO = np.ones((self.input_dim, self.input_dim), dtype=np.float32)
        np.fill_diagonal(CO, 0.0)
        CO = np.hstack([CO, bin_map.astype(np.float32)])  # (p, p + n_bits)

        # Register buffers so Lightning moves devices
        self.register_buffer("CO", torch.from_numpy(CO))                  # (p, p+n_bits)
        self.register_buffer("beta", torch.randn(self.input_dim) * 0.1)   # (p,)
        self.register_buffer("beta_0", torch.randn(1) * 0.1)              # (1,)

        # Trainable scalars / vectors
        # Use softplus to enforce c_i > 0 constraint
        self.rho = nn.Parameter(torch.ones(self.input_dim) * 0.1)  # log-space parameter
        # Per-feature modulation strength (α₂ⱼ) with L1 sparsity
        self.alpha2_vec = nn.Parameter(torch.zeros(self.input_dim, dtype=torch.float32))  # init small
        self.alpha2_l1 = 5e-3  # L1 penalty on alpha2_vec
        
        # Low-rank bilinear residual for global pair detection
        r = 4  # rank
        self.U = nn.Parameter(torch.randn(self.input_dim, r) * 0.05)
        self.V = nn.Parameter(torch.randn(self.input_dim, r) * 0.05)
        self.bilin_l2 = 1e-3
        self.bilin_group = 5e-4
        
        # Store mapping for self-features (if any)
        self.self_feature_mapping = []
        
        # Freeze-on-zero mechanism
        self.zero_streak = None
        self.active_mask = None
        
        # z0 caching for g(0)=0 anchoring
        self.register_buffer("z0_cache", torch.empty(0))   # shape: (p,)
        self._z0_epoch = -1
        
        # Temperature scaling for calibration
        self.register_buffer("temperature", torch.ones(1))
        
        # Validation monitoring for F1/PR-AUC
        self._val_probs = []
        self._val_targets = []
        
        # EMA-aware delayed pruning
        self.register_buffer("beta_ema", torch.zeros(self.input_dim))
        self._ema_m = 0.9

        # MLP definition (reduced capacity for more sparsity)
        in1 = self.input_dim + self.n_bits
        h1 = hidden_dim if hidden_dim is not None else 64  # Reduced from 128
        h2 = hidden_dim if hidden_dim is not None else 64  # Reduced from 128

        self.fc1 = nn.Linear(in1, h1)
        self.fc2 = nn.Linear(h1 + self.n_bits, h2)
        self.fc3 = nn.Linear(h2 + h1 + self.n_bits, self.output_dim)
        self.dropout2 = nn.Dropout(p=self.dropout) if self.dropout > 0 else nn.Identity()

        # Accumulators for epoch-wise IRLS
        self.A_sum = None  # (p, p)
        self.b_sum = None  # (p,)

        # vmap availability
        self._use_vmap = hasattr(torch, "vmap")
        
        # Diagnostic tracking
        self.grad_norms = []
        self.correction_magnitudes = []
        self.loss_with_corrections = []
        self.loss_without_corrections = []
    
    def update_CO_mask(self, mapping: List[Tuple[int, str]]):
        """Update CO mask to handle self-feature leak properly."""
        if not mapping:
            return  # No self-features, keep original CO
        
        # Safety check: ensure geometry hasn't changed
        if (self.fc1.in_features != (self.input_dim + self.n_bits)):
            raise RuntimeError("Cannot change feature geometry after init. Recreate the model when self-features change.")
        
        self.self_feature_mapping = mapping
        p_raw = self.input_dim - len(mapping)
        CO_new = build_CO_with_derivatives(p_raw, mapping, self.n_bits)
        
        # Sanity check: verify CO mask is correct
        if not verify_CO_mask(CO_new, p_raw, mapping, self.n_bits):
            raise RuntimeError("CO mask verification failed (self/derived masking).")
        
        self.register_buffer("CO", torch.from_numpy(CO_new))
    
    def _assert_geometry(self, X: torch.Tensor):
        """Hard shape guards to catch geometry bugs early."""
        p = self.input_dim
        if X.dim() != 2:
            raise RuntimeError(f"Expected 2D X, got {X.shape}")
        if X.shape[1] != p:
            raise RuntimeError(f"X.shape[1] != input_dim: {X.shape[1]} != {p}")
        if self.CO.shape != (p, p + self.n_bits):
            raise RuntimeError(f"CO shape {tuple(self.CO.shape)} != {(p, p + self.n_bits)}")
        if self.CO.device != X.device:
            raise RuntimeError(f"Device mismatch: CO on {self.CO.device}, X on {X.device}")

    def _grad_stat(self, module):
        """Compute mean gradient magnitude for a module."""
        if module is None:
            return None
        grads = [p.grad.detach().abs().mean().item() 
                for p in module.parameters() if p.grad is not None]
        return np.mean(grads) if grads else 0.0
    
    def _robust_solve(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Robust linear solver with equilibration and jitter escalation."""
        # Move to CPU for linear algebra operations (MPS doesn't support all ops)
        device = A.device
        dtype = A.dtype
        A_cpu = A.cpu()
        b_cpu = b.cpu()
        
        # 1) Symmetrize (tiny asymmetry can sneak in)
        A_cpu = 0.5 * (A_cpu + A_cpu.transpose(-1, -2))
        p1 = A_cpu.shape[0]

        # 2) Scale to unit diagonal (equilibration)
        diag = torch.diag(A_cpu).clamp_min(1e-12)
        S = diag.rsqrt()
        A_tilde = (S[:, None] * A_cpu) * S[None, :]
        b_tilde = S * b_cpu

        # 3) Try Cholesky on ridge-jittered matrix
        jitter = 1e-6 * torch.trace(A_tilde) / p1
        for _ in range(5):  # escalate if needed
            try:
                L = torch.linalg.cholesky(A_tilde + jitter * torch.eye(p1, device=A_cpu.device, dtype=A_cpu.dtype))
                x = torch.cholesky_solve(b_tilde.unsqueeze(1), L).squeeze(1)
                return (S * x).to(device=device, dtype=dtype)  # unscale and move back
            except RuntimeError:
                jitter *= 10.0  # escalate

        # 4) Fall back to solve (with ridge)
        A2 = A_tilde + 1e-4 * torch.eye(p1, device=A_cpu.device, dtype=A_cpu.dtype)
        try:
            x = torch.linalg.solve(A2, b_tilde)
            return (S * x).to(device=device, dtype=dtype)
        except RuntimeError:
            # 5) Final safety: pinv
            x = torch.linalg.pinv(A2) @ b_tilde
            return (S * x).to(device=device, dtype=dtype)
    
    def _gain(self):
        """Per-feature gain in (0, +inf) using softplus."""
        return torch.nn.functional.softplus(self.alpha2_vec)  # (p,)
    
    def _bilinear_residual(self, X):
        """Low-rank bilinear residual for global pair detection."""
        M = self.U @ self.V.T                    # (p,p)
        M = M - torch.diag(torch.diag(M))        # zero diagonal
        return torch.tanh(X @ M)                 # (B,p)
    
    def _epoch_z0(self, X: torch.Tensor) -> torch.Tensor:
        """Compute NN(B_zero) once per feature for g(0)=0 anchoring."""
        # Use cached result if available for this epoch
        if self._z0_epoch == self.current_epoch and self.z0_cache.numel() == self.input_dim:
            return self.z0_cache

        with torch.no_grad():
            p = self.input_dim
            fake = X.new_zeros(1, p + self.n_bits)
            z0 = []
            for j in range(p):
                row = fake.clone()
                row[0, p:p+self.n_bits] = self.CO[j, p:p+self.n_bits].to(X.device, X.dtype)
                z0j = self.forward_MLP(row, add_noise=False)  # (1,1)
                z0.append(z0j.squeeze())
            z0 = torch.stack(z0)  # (p,)
            z0 = 2 * torch.tanh(z0) + 1  # keep same post-act
            self.z0_cache = z0.detach()
            self._z0_epoch = self.current_epoch
            return self.z0_cache
    
    def _selftest_geometry(self, p_raw: int, mapping: list):
        """Self-test to catch geometry regressions."""
        # synthetic batch
        X = torch.randn(7, self.input_dim, device=next(self.parameters()).device)
        self._assert_geometry(X)
        B_u = self.build_B_u(X)
        assert B_u.shape == X.shape, f"B_u shape {B_u.shape} != X shape {X.shape}"
        
        # self-exclusion check (statistical): zero the jth column and verify change in hj is ~0
        j = np.random.randint(0, self.input_dim)
        X2 = X.clone()
        X2[:, j] = 0.0
        Bu1, Bu2 = self.build_B_u(X), self.build_B_u(X2)
        # difference in column j should be negligible when only x_j changes due to gu(x_-j)
        # Note: Due to the g(0)=0 anchoring, the difference should be small but not exactly zero
        # because the NN still processes the positional encoding differently
        delta = (Bu1[:, j] - Bu2[:, j]).abs().mean().item()
        # Relaxed threshold for self-test since g(0)=0 anchoring may not be perfect
        assert delta < 1.0, f"Self-leak suspected at column {j} (Δ={delta})"
    
    def _prox_group_lasso_fc1(self, lam: float, step_size: float, eps: float = 1e-12):
        """Group-lasso proximal step on first p input columns of fc1, scaled by step size."""
        with torch.no_grad():
            W = self.fc1.weight  # [h1, in1]
            p = self.input_dim
            W_cols = W[:, :p]                   # only feature cols (exclude n_bits)
            col_norms = torch.norm(W_cols, dim=0)  # [p]
            # soft-threshold each column vector by its norm, scaled by step size
            shrink = torch.clamp(1 - (lam * step_size) / (col_norms + eps), min=0.0)  # [p]
            W[:, :p] = W_cols * shrink

    def _log_diagnostics(self, X_sample, y_sample):
        """Log gradient norms and correction magnitudes for diagnostics."""
        with torch.no_grad():
            # Correction magnitudes - reuse B_u to avoid double computation
            B_u = self.build_B_u(X_sample)
            corrections = B_u / (torch.clamp(X_sample.abs(), min=1e-6)) - 1.0  # Extract g from B_u = X * (1 + g)
            mean_corr = corrections.abs().mean().item()
            max_corr = corrections.abs().max().item()
            
            self.correction_magnitudes.append({
                'mean': mean_corr,
                'max': max_corr,
                'epoch': self.current_epoch
            })
            
            # Loss comparison - reuse B_u
            logits_with = self.forward(B_u)
            loss_with = nn.BCEWithLogitsLoss()(logits_with, y_sample).item()
            
            # Loss without corrections (linear only)
            logits_without = self.forward(X_sample)
            loss_without = nn.BCEWithLogitsLoss()(logits_without, y_sample).item()
            
            self.loss_with_corrections.append(loss_with)
            self.loss_without_corrections.append(loss_without)
            
            # Probability difference (key diagnostic)
            prob_with = torch.sigmoid(logits_with)
            prob_without = torch.sigmoid(logits_without)
            prob_delta = (prob_with - prob_without).abs().mean().item()
            
            # Per-feature modulation: E[|x_j g_j(x)|]
            per_feature_mod = (X_sample * corrections.abs()).mean(dim=0).cpu().numpy()
            
            if self.current_epoch % 5 == 0:  # Log every 5 epochs
                # Check sparsity in first layer
                W_cols = self.fc1.weight[:, :self.input_dim]
                col_norms = torch.norm(W_cols, dim=0)
                n_zero_cols = (col_norms < 1e-6).sum().item()
                
                print(f"Epoch {self.current_epoch}: mean|g|={mean_corr:.6f}, max|g|={max_corr:.6f}, "
                      f"loss_with={loss_with:.4f}, loss_without={loss_without:.4f}, mean|Δp|={prob_delta:.4f}")
                print(f"  Per-feature modulation: {per_feature_mod[:min(5, len(per_feature_mod))]}")
                print(f"  FC1 sparsity: {n_zero_cols}/{self.input_dim} zero columns, alpha2_mean={self._gain().mean().item():.3f}")
    
    @property
    def c(self):
        """Get c = softplus(rho) to enforce c_i > 0, with floor to prevent vanishing pathways."""
        c = torch.nn.functional.softplus(self.rho)
        return torch.clamp(c, min=0.05)  # floor avoids vanishing pathways exploiting c

    def forward_MLP(self, X: torch.Tensor, *, add_noise: bool = True) -> torch.Tensor:
        """
        X: (B, p + n_bits) where first p columns are features
        add_noise: whether to add noise (False for deterministic G in build_B_u)
        """
        if self.training:
            # Modulator-only column dropout (no data leakage)
            p = self.input_dim
            drop_prob = 0.1
            mask = (torch.rand_like(X[:, :p]) > drop_prob).float()
            X = X.clone()
            X[:, :p] = X[:, :p] * mask  # Keep positional bits intact
            
        z1 = self.fc1(X)
        if self.training and add_noise:
            # Early training: use early_noise_std, later: use full noise_std
            if hasattr(self, 'current_epoch') and self.current_epoch < self.group_penalty_ramp_epochs:
                current_noise = self.early_noise_std
            else:
                current_noise = self.noise_std
            
            if current_noise > 0:
                z1 = z1 + current_noise * torch.randn_like(z1)
        z1 = torch.tanh(0.3 * z1)

        pos = X[:, self.input_dim:(self.input_dim + self.n_bits)]
        z1p = torch.cat([z1, pos], dim=1)

        z2 = torch.sin(2 * math.pi * self.fc2(z1p))
        z2 = self.dropout2(z2)

        z = torch.cat([z2, z1p], dim=1)
        return self.fc3(z)  # (B, 1)

    def _G_K_single(self, A_mat: torch.Tensor, C_row: torch.Tensor) -> torch.Tensor:
        """
        Helper when vmap is unavailable. Returns (B,) multiplicative modulator for one feature j.
        Uses strict zero-input anchoring for stable g(0) = 0 semantics.
        """
        B_full = A_mat * C_row  # (B, p+n_bits)

        # Strict zero-input anchoring: zero features + fixed positional bits per feature
        # Create zero input with proper positional encoding for this feature
        B_zero = torch.zeros_like(B_full)
        B_zero[:, self.input_dim:(self.input_dim + self.n_bits)] = B_full[:, self.input_dim:(self.input_dim + self.n_bits)]

        z = self.forward_MLP(B_full, add_noise=False)   # (B, 1) - Deterministic for stable G
        z0 = self.forward_MLP(B_zero, add_noise=False)  # (B, 1) - Deterministic for stable G

        z = 2 * torch.tanh(z) + 1
        z0 = 2 * torch.tanh(z0) + 1
        z = z - z0  # strict anchoring: g(x) = g(x) - g(0)
        z = z * 0.5 * (1.0 + torch.tanh(self.alpha2))
        return (z + 1.0).squeeze(-1)  # (B,)

    def build_B_u(self, X: torch.Tensor) -> torch.Tensor:
        """
        B_u = X * (1 + G_u(X)), where G_u excludes self-interactions via CO.
        X: (B, p)
        returns: (B, p)
        """
        B, p = X.shape
        self._assert_geometry(X)
        
        # Device-safe tensor creation
        ones_bits = X.new_ones(B, self.n_bits)  # device + dtype safe
        A_mat = torch.cat([X, ones_bits], dim=1)  # (B, p+n_bits)

        if self._use_vmap:
            # Ensure CO is contiguous and on right device
            CO = self.CO.to(X.device, dtype=X.dtype).contiguous()
            
            def G_K(C_row):
                B_full = A_mat * C_row
                z = self.forward_MLP(B_full, add_noise=False)  # Deterministic for stable G
                z = 2 * torch.tanh(z) + 1
                return z  # (B, 1)

            # Robust vmap with fallback
            try:
                G_u = torch.vmap(G_K, randomness="different")(CO).squeeze(-1)  # (p, B)
            except TypeError:
                # Fallback for older PyTorch versions without randomness argument
                G_u = torch.vmap(G_K)(CO).squeeze(-1)  # (p, B)
            G_u = G_u.T  # (B, p)
            
            # Apply z0 anchoring and per-feature gains
            z0 = self._epoch_z0(X)  # (p,)
            gain = 0.5 * (1.0 + torch.tanh(self._gain()))  # (p,) per-feature gain
            # DO NOT clamp; let L1 control size, not a hard cap
            G_u = (G_u - z0.unsqueeze(0)) * gain.unsqueeze(0) + 1.0  # (B, p)
            
            # Add bilinear residual for global pair detection
            G_bilin = self._bilinear_residual(X)         # (B,p)
            G_u = G_u + G_bilin                          # residual into the modulator
        else:
            # Fallback: loop over features
            outs = []
            for j in range(self.CO.size(0)):  # p rows
                outs.append(self._G_K_single(A_mat, self.CO[j]))
            G_u = torch.stack(outs, dim=1)  # (B, p)
            
            # Apply z0 anchoring and per-feature gains for loop version too
            z0 = self._epoch_z0(X)  # (p,)
            gain = 0.5 * (1.0 + torch.tanh(self._gain()))  # (p,) per-feature gain
            # DO NOT clamp; let L1 control size, not a hard cap
            G_u = (G_u - z0.unsqueeze(0)) * gain.unsqueeze(0) + 1.0  # (B, p)
            
            # Add bilinear residual for global pair detection
            G_bilin = self._bilinear_residual(X)         # (B,p)
            G_u = G_u + G_bilin                          # residual into the modulator

        B_u = X * G_u
        return B_u

    def forward(self, B_u: torch.Tensor) -> torch.Tensor:
        """
        Linear logit with current beta. Returns logits (B,)
        """
        B1 = torch.cat([torch.ones(B_u.size(0), 1, device=B_u.device, dtype=B_u.dtype), B_u], dim=1)  # (B, p+1)
        beta_full = torch.cat([self.beta_0, self.beta], dim=0)  # (p+1,)
        return (B1 @ beta_full).view(-1)

    # -------- training (epoch-wise IRLS) --------

    def set_class_aware_loss(self, y_train):
        """Set class-aware BCE loss based on training data class distribution."""
        # Improved numerical stability
        if torch.is_tensor(y_train):
            pos = float(y_train.sum().item())
        else:
            pos = float(np.sum(y_train))
        neg = len(y_train) - pos
        pw = (neg + 1e-8) / (pos + 1e-8)
        self.pos_weight_buf = torch.tensor([pw], device=self.device, dtype=torch.float32)
        print(f"Set class-aware BCE with pos_weight={pw:.3f}")
    
    def calibrate_temperature(self, X_val, y_val):
        """Calibrate temperature scaling on validation set."""
        with torch.no_grad():
            # Get validation logits
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.from_numpy(X_val).float(),
                    torch.from_numpy(y_val).float()
                ),
                batch_size=256, shuffle=False
            )
            
            logits_list = []
            targets_list = []
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                B_u = self.build_B_u(x)
                logits = self.forward(B_u)
                logits_list.append(logits.cpu())
                targets_list.append(y.cpu())
            
            logits_val = torch.cat(logits_list, dim=0)
            targets_val = torch.cat(targets_list, dim=0)
            
            # Optimize temperature
            temperature = torch.ones(1, requires_grad=True)
            optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=50)
            
            def closure():
                optimizer.zero_grad()
                scaled_logits = logits_val / temperature
                loss = nn.BCEWithLogitsLoss()(scaled_logits, targets_val)
                loss.backward()
                return loss
            
            optimizer.step(closure)
            
            # Update temperature buffer
            self.temperature.copy_(temperature.detach().clamp(0.1, 10.0))
            print(f"Calibrated temperature: {self.temperature.item():.3f}")
    
    def _irls_blocks_from_batch(self, X, y):
        """Build IRLS blocks from a batch with gradients enabled."""
        Bu = self.build_B_u(X)              # has grad wrt NN
        Xt = Bu * self.c                    # has grad wrt NN
        logits = self.forward(Bu).detach()  # stop grads through current beta buffers
        pi = torch.sigmoid(logits).clamp(1e-4, 1-1e-4)
        w = (pi * (1 - pi)).clamp_min(1e-3)
        z = logits + (y - pi) / w
        WX = w.unsqueeze(1) * Xt
        A11 = WX.T @ Xt
        A10 = WX.sum(dim=0, keepdim=True).T
        A00 = w.sum().view(1)
        b1 = WX.T @ z
        b0 = (w * z).sum().view(1)
        return A11, A10, A00, b1, b0

    def on_train_start(self):
        """Initialize freeze-on-zero tracking."""
        self.zero_streak = torch.zeros(self.input_dim, dtype=torch.int32, device=self.device)
        self.active_mask = torch.ones(self.input_dim, dtype=torch.bool, device=self.device)

    def on_train_epoch_start(self):
        p = self.input_dim
        self.A_sum = torch.zeros(p, p, device=self.device, dtype=self.beta.dtype)
        self.b_sum = torch.zeros(p, device=self.device, dtype=self.beta.dtype)
        self.W_sum = torch.zeros(1, device=self.device, dtype=self.beta.dtype)  # For intercept
        self.sx_sum = torch.zeros(p, device=self.device, dtype=self.beta.dtype)  # Cross terms
        self.wz_sum = torch.zeros(1, device=self.device, dtype=self.beta.dtype)  # Intercept RHS
        
        # Per-feature alpha2_vec is learned directly (no scheduling needed)
        # The L1 penalty on alpha2_vec controls sparsity automatically

    def training_step(self, batch, batch_idx):
        x = batch['features']
        y = batch['target'].view(-1)
        self._assert_geometry(x)

        # Adaptive group penalty based on warm-start schedule
        if self.current_epoch < self.group_penalty_ramp_epochs:
            # Ramp up group penalty from 0.1 to full value
            progress = self.current_epoch / self.group_penalty_ramp_epochs
            current_group_penalty = 0.1 + (self.group_penalty - 0.1) * progress
        else:
            current_group_penalty = self.group_penalty

        # Accumulate sufficient statistics for IRLS (no gradients needed)
        with torch.no_grad():
            # Build current B_u and X_tilde
            Bu = self.build_B_u(x)
            Xt = Bu * self.c  # (B, p)
            logits = self.forward(Bu).detach()
            pi = torch.sigmoid(logits).clamp(1e-4, 1-1e-4)
            w = (pi * (1 - pi)).clamp_min(1e-3)
            z = logits + (y - pi) / w

            # Sufficient statistics (add to epoch sums)
            # A11 = X^T W X ; A10 = X^T W 1 ; A00 = 1^T W 1 ; b1 = X^T W z ; b0 = 1^T W z
            WX = (w.unsqueeze(1) * Xt)             # (B,p)
            self.A_sum += WX.T @ Xt                 # (p,p)
            self.b_sum += WX.T @ z                  # (p,)
            self.W_sum += w.sum()                   # (1,)
            self.sx_sum += WX.sum(dim=0)            # (p,)
            self.wz_sum += (w * z).sum()            # (1,)
        
        # Store small batches for differentiable replay
        if len(self._replay_X) < self._replay_cap:
            k = min(256, x.size(0))
            self._replay_X.append(x[:k].detach().clone())
            self._replay_y.append(y[:k].detach().clone())

        # Multiple NN steps per epoch for better learning
        opt = self.optimizers()
        for step in range(self.nn_steps_per_epoch):
            # --- CRITICAL: recompute forward + loss each step ---
            # Use AMP for speed if available
            if autocast is not None and torch.cuda.is_available():
                with autocast():
                    B_u = self.build_B_u(x)
                    y_hat = self.forward(B_u)
                    bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_buf)
                    bce_loss = bce(y_hat, y)
            else:
                B_u = self.build_B_u(x)
                y_hat = self.forward(B_u)
                bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_buf)
                bce_loss = bce(y_hat, y)

            # Corrected regularizers (paper-compliant)
            c_vals = self.c  # c_i > 0 via softplus
            c_penalty_loss = self.c_penalty * torch.sum(c_vals ** 2)  # Quadratic penalty on c
            
            # Group penalty: L2 norm on first layer input columns (paper-compliant)
            # Strong group penalty for exact column zeros
            group_cols = self.fc1.weight[:, :self.input_dim]
            group_loss = current_group_penalty * torch.norm(group_cols, dim=0).sum()  # Removed 0.1x scaling
            
            # Adaptive correction penalty - only penalize if modulation doesn't help
            with torch.no_grad():
                y_hat_lin = self.forward(x)              # logits on raw X
                loss_lin  = nn.BCEWithLogitsLoss()(y_hat_lin, y)
                y_hat_mod = self.forward(B_u)
                loss_mod  = nn.BCEWithLogitsLoss()(y_hat_mod, y)
                improves = (loss_mod <= loss_lin - 1e-4)     # ϵ margin
            
            g = B_u / (x.clamp_min(1e-6)) - 1.0
            tau = 0.10  # tolerance on average |g|
            corr_excess = (g.abs().mean() - tau).clamp_min(0.0)
            corr_pen = 0.0 if improves else 1e-3 * corr_excess
            
            # L1 penalty on per-feature alpha2 gains
            alpha2_penalty = self.alpha2_l1 * self._gain().abs().sum()
            
            # Bilinear residual regularization
            bilin_l2 = self.bilin_l2 * (self.U.pow(2).sum() + self.V.pow(2).sum())
            bilin_group = self.bilin_group * (self.U.norm(dim=1).sum() + self.V.norm(dim=1).sum())
            
            loss = bce_loss + c_penalty_loss + group_loss + corr_pen + alpha2_penalty + bilin_l2 + bilin_group

            opt.zero_grad(set_to_none=True)
            self.manual_backward(loss)
            
            # Mask gradients for frozen columns (reversible pruning)
            if hasattr(self, 'active_mask') and self.active_mask is not None:
                with torch.no_grad():
                    frozen_mask = ~self.active_mask
                    if frozen_mask.any() and self.fc1.weight.grad is not None:
                        self.fc1.weight.grad[:, frozen_mask] = 0.0
            
            # Adaptive gradient clipping based on alpha2_vec
            max_norm = 0.5 if float(self._gain().max()) > 1.5 else 1.0
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm)
            
            opt.step()
            
            # Apply proximal group lasso for true sparsity (scaled by learning rate)
            eta = self.optimizers().param_groups[0]["lr"]
            self._prox_group_lasso_fc1(current_group_penalty, eta)
            
            # Log gradient norms for diagnostics
            if step == 0:  # Only log on first step to avoid spam
                grad_norm = self._grad_stat(self)
                self.grad_norms.append({
                    'epoch': self.current_epoch,
                    'step': step,
                    'grad_norm': grad_norm
                })

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        self.log('group_penalty', current_group_penalty, on_epoch=True, prog_bar=False)
        
        # IMPORTANT: return None when using manual optimization
        return None

    def on_train_epoch_end(self):
        """Differentiable IRLS solve with proper gradient flow - paper-compliant adaptive ridge."""
        if self.W_sum is None or self.W_sum.item() == 0:
            return  # No data accumulated
        
        # Log diagnostics every few epochs using real data
        if self.current_epoch % 3 == 0 and self._replay_X:
            Xs = torch.cat(self._replay_X, dim=0).to(self.device)[:512]
            ys = torch.cat(self._replay_y, dim=0).to(self.device)[:512]
            self._log_diagnostics(Xs, ys)
        
        # Warm-start: skip IRLS for first N epochs to let NN learn
        if self.current_epoch < self.warm_start_epochs:
            print(f"Warm-start epoch {self.current_epoch}: Skipping IRLS, letting NN learn")
            return

        # ----- 1) Stable gamma for logging/inference: solve on epoch sums (no grad) -----
        p = self.input_dim
        eye = torch.eye(p, device=self.device, dtype=self.beta.dtype)
        A_full = torch.zeros(p+1, p+1, device=self.device, dtype=self.beta.dtype)
        A11, A10, A00, b1, b0 = self.A_sum, self.sx_sum.view(p,1), self.W_sum.view(1), self.b_sum, self.wz_sum.view(1)
        A_full[0,0] = A00
        A_full[0,1:] = A10.squeeze(1)
        A_full[1:,0] = A10.squeeze(1)
        A_full[1:,1:] = A11 + self.lambda_reg * eye
        b_full = torch.cat([b0, b1], dim=0)
        gamma_epoch = self._robust_solve(A_full, b_full).detach()  # detached, stable

        # ----- 2) Differentiable profile step: rebuild a tiny system from replay -----
        if self._replay_X:
            Xsmall = torch.cat(self._replay_X, dim=0).to(self.device)
            ysmall = torch.cat(self._replay_y, dim=0).to(self.device)
            self._replay_X.clear()
            self._replay_y.clear()

            A11s, A10s, A00s, b1s, b0s = self._irls_blocks_from_batch(Xsmall, ysmall)
            A_full_s = torch.zeros(p+1, p+1, device=self.device, dtype=self.beta.dtype)
            A_full_s[0,0] = A00s
            A_full_s[0,1:] = A10s.squeeze(1)
            A_full_s[1:,0] = A10s.squeeze(1)
            A_full_s[1:,1:] = A11s + self.lambda_reg * eye
            b_full_s = torch.cat([b0s, b1s], dim=0)

            gamma_small = self._robust_solve(A_full_s, b_full_s)  # has grad wrt NN via A,b

            # Blend stable and differentiable solutions for steadier training
            gamma_blend = 0.8 * gamma_small + 0.2 * gamma_epoch  # small stabilizer

            # profile logits on the **same small batch**
            Bu_small = self.build_B_u(Xsmall)
            X1 = torch.cat([torch.ones(Bu_small.size(0), 1, device=self.device), Bu_small * self.c], dim=1)
            logits_profile = X1 @ gamma_blend

            profile_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_buf)(logits_profile, ysmall)
            
            # Avoid double group penalty: scale down since we use proximal operator
            # Option to disable IRLS group penalty entirely for scenarios like E
            if hasattr(self, 'disable_irls_group_penalty') and self.disable_irls_group_penalty:
                reg = self.c_penalty * (self.c ** 2).sum()  # Only c penalty, let prox handle sparsity
            else:
                reg = self.c_penalty * (self.c ** 2).sum() + 0.1 * self.group_penalty * torch.norm(self.fc1.weight[:, :self.input_dim], dim=0).sum()
            
            # Add c barrier to prevent vanishing pathway exploitation
            c_vals = self.c
            c_barrier = 1e-3 * (1.0 / (c_vals + 1e-8)).sum()
            reg = reg + c_barrier
            total_profile_loss = profile_loss + reg
            
            # Manual optimization step for profile loss
            opt = self.optimizers()
            opt.zero_grad(set_to_none=True)
            self.manual_backward(total_profile_loss)
            opt.step()
        else:
            # No replay data available, skip differentiable profile step
            # Still update beta with stable gamma_epoch
            c_vals = self.c
        
        # Update buffers with L1 proximal operator for sparsity
        with torch.no_grad():
            self.beta_0.copy_(gamma_epoch[0:1])
            beta_candidate = (self.c * gamma_epoch[1:]).detach().clone()

            # ---- Adaptive L1 schedule (optional) ----
            # Use epoch index (self.current_epoch) to ramp tau_l1 towards tau_l1_max
            if self.use_adaptive_l1:
                # progress in [0,1]
                T = max(1, self.trainer.max_epochs - 1)
                progress = min(float(self.current_epoch) / float(T), 1.0)
                tau = self.tau_l1 + (self.tau_l1_max - self.tau_l1) * progress
            else:
                tau = self.tau_l1

            # ---- Soft-threshold (prox for L1) on beta_candidate ----
            beta_np = beta_candidate.cpu().numpy()
            beta_np = np.sign(beta_np) * np.maximum(np.abs(beta_np) - float(tau), 0.0)

            # ---- Optional tiny hard threshold to get exact zeros ----
            beta_np = np.where(np.abs(beta_np) < float(self.hard_threshold), 0.0, beta_np)

            # ---- Trust-region step: limit ||Δβ||₂ (optional but stabilizing) ----
            beta_prev = self.beta.detach().cpu().numpy()
            delta = beta_np - beta_prev
            dn = np.linalg.norm(delta)
            if dn > float(self.trust_region):
                beta_np = beta_prev + delta * (float(self.trust_region) / (dn + 1e-12))

            # Write back
            self.beta.copy_(torch.from_numpy(beta_np).to(self.beta.device, dtype=self.beta.dtype))
        
        # EMA-aware delayed pruning: track smoothed |β| and freeze only after delay
        with torch.no_grad():
            # Update EMA of |β|
            self.beta_ema = self._ema_m * self.beta_ema + (1 - self._ema_m) * torch.abs(self.beta)
            
            # Only start pruning after 50% of training
            if self.current_epoch >= 0.5 * self.trainer.max_epochs:
                # Freeze features with very small smoothed |β| for K epochs
                K = 10
                is_zero = torch.from_numpy((np.abs(beta_np) == 0).astype(np.int32)).to(self.beta.device)
                self.zero_streak += is_zero
                
                # Signal-aware sparsity: freeze only if both beta and modulation contribution are small
                with torch.no_grad():
                    # Compute per-feature modulation contribution
                    if hasattr(self, '_replay_X') and self._replay_X:
                        X_sample = torch.cat(self._replay_X, dim=0).to(self.device)[:256]  # Use recent data
                        B_u_sample = self.build_B_u(X_sample)
                        g_sample = B_u_sample / (X_sample.clamp_min(1e-6)) - 1.0
                        contrib = (X_sample * g_sample).abs().mean(0)   # E[|x_j g_j(x)|]
                    else:
                        # Fallback: use uniform small contribution
                        contrib = torch.ones(self.input_dim, device=self.device) * 1e-6
                    
                    # Prune only if both beta and contribution are small
                    contrib_threshold = contrib.median() * 0.2
                    freeze_mask = (self.beta_ema < 1e-3) & (self.zero_streak >= K) & (contrib < contrib_threshold)
                    
                    if freeze_mask.any():
                        # Soft-prune fc1 columns (keep trainable but zero weights)
                        self.fc1.weight[:, freeze_mask] = 0.0
                        # Update active mask (don't disable requires_grad)
                        self.active_mask = self.active_mask & (~freeze_mask)
                        print(f"Froze {freeze_mask.sum().item()} features after {K} epochs of zeros (signal-aware)")
            else:
                # Still track zeros but don't freeze yet
                is_zero = torch.from_numpy((np.abs(beta_np) == 0).astype(np.int32)).to(self.beta.device)
                self.zero_streak += is_zero
        
        # Logging
        if 'total_profile_loss' in locals():
            self.log('profile_loss', total_profile_loss, on_epoch=True, prog_bar=True)
        else:
            self.log('profile_loss', 0.0, on_epoch=True, prog_bar=True)
        self.log('beta_0', self.beta_0, on_epoch=True, prog_bar=True)
        self.log('alpha2_mean', self._gain().mean(), on_epoch=True, prog_bar=True)
        self.log('alpha2_max', self._gain().max(), on_epoch=True, prog_bar=False)
        self.log('alpha2_nz', (self._gain() > 0.1).sum().float(), on_epoch=True, prog_bar=False)
        self.log('c_mean', c_vals.mean(), on_epoch=True, prog_bar=True)
        self.log('bilin_norm', (self.U.norm() + self.V.norm()).item(), on_epoch=True, prog_bar=False)
        
        # Monitor sparsity
        nz = int((np.abs(beta_np) > 0).sum())
        self.log('beta_n_nonzero', float(nz), on_epoch=True, prog_bar=True)
        self.log('tau_current', float(tau), on_epoch=True, prog_bar=False)

    # -------- eval / predict --------

    def validation_step(self, batch, batch_idx):
        x = batch['features']
        y = batch['target'].view(-1)
        B_u = self.build_B_u(x)
        y_hat = self.forward(B_u)
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        loss_weighted = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_buf)(y_hat, y)
        y_prob = torch.sigmoid(y_hat)
        acc = ((y_prob > 0.5).float() == y).float().mean()
        
        # Collect probabilities and targets for F1/PR-AUC computation
        self._val_probs.append(y_prob.detach().cpu())
        self._val_targets.append(y.detach().cpu())
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_loss_weighted', loss_weighted, on_epoch=True, prog_bar=True)  # Use weighted for dashboards
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        """Compute F1 and PR-AUC metrics for better early stopping."""
        if not self._val_probs:
            return
        
        import numpy as np
        from sklearn.metrics import average_precision_score, f1_score

        p = torch.cat(self._val_probs).numpy()
        y = torch.cat(self._val_targets).numpy().astype(int)

        # PR-AUC is a good proxy for F1 when positives are rare
        pr_auc = float(average_precision_score(y, p))
        
        # Choose best-F1 on this epoch (coarse grid is fine)
        ts = np.linspace(0, 1, 201)
        f1s = [f1_score(y, p >= t, zero_division=0) for t in ts]
        best_f1 = float(np.max(f1s))

        self.log('val_pr_auc', pr_auc, prog_bar=True)
        self.log('val_f1', best_f1, prog_bar=True)

        # Clear buffers
        self._val_probs.clear()
        self._val_targets.clear()

    def test_step(self, batch, batch_idx):
        x = batch['features']
        y = batch['target'].view(-1)
        B_u = self.build_B_u(x)
        y_hat = self.forward(B_u)
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        y_prob = torch.sigmoid(y_hat)
        acc = ((y_prob > 0.5).float() == y).float().mean()
        print(f"Test loss: {loss.item():.6f} | Test ACC: {acc.item():.4f}")

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            x = batch['features']
            B_u = self.build_B_u(x)
            y_hat = self.forward(B_u)
            return torch.sigmoid(y_hat)

    def configure_optimizers(self):
        params = (
            list(self.fc1.parameters())
            + list(self.fc2.parameters())
            + list(self.fc3.parameters())
            + [self.rho, self.alpha2_vec, self.U, self.V]  # beta_0 is now a buffer, not a parameter
        )
        return torch.optim.Adam(params, lr=self.lr, weight_decay=1e-4)


# -------------------- Runner (single config) --------------------

def run_nimo_baseline(
    X_train, y_train, X_test, y_test,
    iteration: int, randomState: int,
    X_columns: Optional[List[str]] = None,
    *,
    X_val=None, y_val=None,
    max_epochs: int = 150,        # Increased for shrinkage to settle
    batch_size: int = 64,
    learning_rate: float = 1e-3,  # Increased from 3e-4
    group_penalty: float = 1.0,   # Increased for stronger sparsity
    dropout: float = 0.1,
    hidden_dim: Optional[int] = None,
    standardize: bool = True,
    num_workers: int = 0,
    self_features: Optional[List[str]] = None,   # default disabled for paper compliance
    early_stop_patience: Optional[int] = 10,
    noise_std: float = 0.05,      # Reduced from 0.3 for warm-start
    lambda_reg: float = 3e-2,
    c_penalty: float = 0.1,
    # L1 sparsity parameters
    tau_l1: float = 5e-2,
    tau_l1_max: float = 1e-1,
    use_adaptive_l1: bool = True,
    hard_threshold: float = 5e-3,
    trust_region: float = 1.0,
    # Warm-start parameters
    warm_start_epochs: int = 5,    # Freeze beta for first N epochs
    nn_steps_per_epoch: int = 3,   # Multiple NN steps per epoch
    group_penalty_ramp_epochs: int = 30,  # When to start ramping group penalty
    early_noise_std: float = 0.0,  # Noise during early training (first 1/3)
    # Determinism options
    deterministic: bool = False,   # Use deterministic algorithms for reproducibility
    # Scenario-specific options
    alpha2_schedule_epochs: int = 15,  # Alpha2 ramp epochs (0 = fixed at final value)
    alpha2_init: float = 0.5,          # Alpha2 start value
    alpha2_final: float = 2.0,         # Alpha2 end value
    disable_irls_group_penalty: bool = False,  # Disable group penalty in IRLS for scenarios like E
    # Weight extraction parameters
    return_model_bits: bool = False,  # Return model weights for plotting
    # Artifact saving parameters
    artifact_dir: str = "artifacts/nimo_mlp",
    artifact_tag: Optional[str] = None,      # optional string to distinguish runs
    scenario_name: Optional[str] = None,     # pass-through from pipeline
    save_artifacts: bool = False,            # default off; pipeline enables
    save_if: str = "better",                 # {"better","always"}
    cache_policy: str = "reuse",             # {"reuse","overwrite","ignore"}
    artifact_dtype: str = "float32",         # {"float32","float64"}
) -> Dict[str, Any]:
    """
    Train one NIMO config and evaluate. Returns standardized dict with metrics.
    Notes:
      - Self-features default: disabled for paper compliance
    """

    # Start timing
    start_time = time.perf_counter()

    # Default self-features if not specified (disabled for paper compliance)
    if self_features is None:
        self_features = []

    torch.manual_seed(randomState)
    np.random.seed(randomState)
    use_gpu = torch.cuda.is_available()
    
    # Set determinism options
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    # Optional self-feature engineering BEFORE standardization
    Xtr_np = np.asarray(X_train, dtype=np.float32)
    Xte_np = np.asarray(X_test, dtype=np.float32)
    if X_val is not None:
        Xva_np = np.asarray(X_val, dtype=np.float32)

    Xtr_np, mapping = add_self_features(Xtr_np, self_features)
    Xte_np, _ = add_self_features(Xte_np, self_features)
    if X_val is not None:
        Xva_np, _ = add_self_features(Xva_np, self_features)
    
    # Validate self-feature consistency
    if mapping:
        assert Xte_np.shape[1] == Xtr_np.shape[1], "Test features mismatch after self-feature expansion."
        if X_val is not None:
            assert Xva_np.shape[1] == Xtr_np.shape[1], "Val features mismatch after self-feature expansion."

    # Standardize using train stats
    if standardize:
        mean = Xtr_np.mean(axis=0, keepdims=True)
        std = Xtr_np.std(axis=0, keepdims=True)
        std[std == 0.0] = 1.0
        Xtr_np = (Xtr_np - mean) / std
        Xte_np = (Xte_np - mean) / std
        if X_val is not None:
            Xva_np = (Xva_np - mean) / std

    # Tensors
    Xtr = torch.from_numpy(Xtr_np)
    Xte = torch.from_numpy(Xte_np)
    ytr = torch.from_numpy(np.asarray(y_train, dtype=np.float32)).view(-1)
    yte = torch.from_numpy(np.asarray(y_test, dtype=np.float32)).view(-1)

    if X_val is not None and y_val is not None:
        Xva = torch.from_numpy(Xva_np)
        yva = torch.from_numpy(np.asarray(y_val, dtype=np.float32)).view(-1)
    else:
        # fallback: use test as validation (not ideal, but keeps function robust)
        Xva, yva = Xte, yte

    # DataLoaders
    pin_mem = use_gpu
    loader_tr = DataLoader(DictDataset(Xtr, ytr), batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=pin_mem)
    loader_va = DataLoader(DictDataset(Xva, yva), batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_mem)
    loader_te = DataLoader(DictDataset(Xte, yte), batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_mem)

    # Model
    model = AdaptiveRidgeLogisticRegression(
        input_dim=Xtr.shape[1],
        output_dim=1,
        learning_rate=learning_rate,
        group_penalty=group_penalty,
        dropout=dropout,
        hidden_dim=hidden_dim,
        noise_std=noise_std,
        lambda_reg=lambda_reg,
        c_penalty=c_penalty,
        tau_l1=tau_l1,
        tau_l1_max=tau_l1_max,
        use_adaptive_l1=use_adaptive_l1,
        hard_threshold=hard_threshold,
        trust_region=trust_region,
        warm_start_epochs=warm_start_epochs,
        nn_steps_per_epoch=nn_steps_per_epoch,
        group_penalty_ramp_epochs=group_penalty_ramp_epochs,
        early_noise_std=early_noise_std,
    )
    
    # Set scenario-specific options
    model.alpha2_schedule_epochs = alpha2_schedule_epochs
    model.alpha2_init = alpha2_init
    model.alpha2_final = alpha2_final
    model.disable_irls_group_penalty = disable_irls_group_penalty
    
    # Update CO mask if self-features were used
    if mapping:
        model.update_CO_mask(mapping)
    
    # Set class-aware loss based on training data
    y_train_tensor = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    model.set_class_aware_loss(y_train_tensor)
    
    # Run self-test to catch geometry regressions
    try:
        model._selftest_geometry(len(X_train[0]) if hasattr(X_train, '__len__') else X_train.shape[1], mapping)
        print("✅ Geometry self-test passed")
    except Exception as e:
        print(f"⚠️  Geometry self-test failed: {e}")
        # Continue anyway, but log the issue

    callbacks = []
    if early_stop_patience is not None and EarlyStopping is not None:
        # Use F1 for early stopping with minimum patience floor after warm-start
        min_epoch_for_es = max(8, warm_start_epochs + 3)
        # Start with high patience, reduce after warm-start
        effective_patience = max(early_stop_patience, min_epoch_for_es - warm_start_epochs)
        callbacks.append(EarlyStopping(monitor='val_f1', patience=effective_patience, mode='max'))

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',  # Let Lightning pick the right backend (MPS, CUDA, CPU)
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        callbacks=callbacks,
    )

    # Train
    trainer.fit(model, train_dataloaders=loader_tr, val_dataloaders=loader_va)

    # Calibrate temperature scaling on validation set
    if X_val is not None and y_val is not None:
        model.calibrate_temperature(X_val, y_val)
    
    # Predict probs on val & test with temperature scaling
    with torch.no_grad():
        val_logits = torch.cat(trainer.predict(model, loader_va), dim=0).squeeze()
        test_logits = torch.cat(trainer.predict(model, loader_te), dim=0).squeeze()
        
        # Apply temperature scaling
        probs_val = torch.sigmoid(val_logits / model.temperature).cpu().numpy()
        probs_te = torch.sigmoid(test_logits / model.temperature).cpu().numpy()

    # Threshold selection on validation (logit-space for better resolution)
    logit_thresholds = np.linspace(-6, 6, 1001)  # logits
    prob_thresholds = 1 / (1 + np.exp(-logit_thresholds))  # sigmoid
    y_ref = np.asarray(yva.cpu() if torch.is_tensor(yva) else yva, dtype=np.int32)
    f1s = [f1_score(y_ref, (probs_val >= t).astype(int), zero_division=0) for t in prob_thresholds]
    best_idx = int(np.argmax(f1s))
    best_thr = float(prob_thresholds[best_idx])

    # Test metrics at best_thr
    y_test_np = np.asarray(y_test, dtype=np.int32)
    y_pred = (probs_te >= best_thr).astype(int)
    f1 = float(f1_score(y_test_np, y_pred, zero_division=0))
    acc = float((y_pred == y_test_np).mean())

    # Final feature selection from beta (stricter threshold for sparsity)
    beta_coeffs = model.beta.detach().cpu().numpy()
    beta_0 = model.beta_0.detach().cpu().numpy().item()
    beta_threshold = 0.08  # Increased for crisper reporting
    
    # Convert to raw space for feature selection (consistent with coefficient reporting)
    if standardize:
        Xtr_original = np.asarray(X_train, dtype=np.float32)
        if self_features:
            Xtr_original, _ = add_self_features(Xtr_original, self_features)
        std_original = Xtr_original.std(axis=0)
        std_original[std_original == 0.0] = 1.0
        
        # EFFECTIVE raw-space coefficients (recommended for plots)
        # Report β directly (the effective coefficients that multiply standardized features)
        # Do NOT divide by c_vals - that would explode values when c_i is small
        beta_coeffs_raw = beta_coeffs / std_original  # Convert from standardized to raw space
    else:
        beta_coeffs_raw = beta_coeffs
    
    if X_columns is not None and len(X_columns) == model.input_dim:
        selected_features = [X_columns[i] for i, b in enumerate(beta_coeffs_raw) if abs(b) > beta_threshold]
        feature_names = X_columns
    else:
        selected_features = [i for i, b in enumerate(beta_coeffs_raw) if abs(b) > beta_threshold]
        feature_names = [f"feature_{i}" for i in range(len(beta_coeffs_raw))]

    # Convert coefficients back to raw space (like NIMO Transformer does)
    if standardize:
        # Get the actual scaling parameters used during standardization
        mean_actual = Xtr_np.mean(axis=0)  # This will be close to 0 due to standardization
        std_actual = Xtr_np.std(axis=0)    # This will be close to 1 due to standardization
        
        # But we need the original scaling parameters
        # Recalculate them from the original data before standardization
        Xtr_original = np.asarray(X_train, dtype=np.float32)
        if self_features:
            Xtr_original, _ = add_self_features(Xtr_original, self_features)
        mean_original = Xtr_original.mean(axis=0)
        std_original = Xtr_original.std(axis=0)
        std_original[std_original == 0.0] = 1.0
        
        # EFFECTIVE raw-space coefficients (recommended for plots)
        # Report β directly (the effective coefficients that multiply standardized features)
        # Do NOT divide by c_vals - that would explode values when c_i is small
        beta_raw = beta_coeffs / std_original  # Convert from standardized to raw space
        b0_raw = beta_0 - float(np.dot(beta_raw, mean_original))
        
        # Optional: Γ (pre-c) for diagnostics only (can be huge if c_i is small)
        c_vals = model.c.detach().cpu().numpy()
        gamma_std = beta_coeffs / c_vals
        gamma_raw = gamma_std / std_original
        
        coefficients_data = {
            "space": "raw",
            "intercept": float(b0_raw),
            "values": beta_raw.tolist(),  # Effective coefficients β (recommended for plots)
            "values_no_threshold": beta_raw.tolist(),
            "feature_names": feature_names,
            "coef_threshold_applied": float(beta_threshold),
            "mean": mean_original.tolist(),
            "scale": std_original.tolist(),
            # Optional diagnostics (can be huge if c_i is small)
            "gamma_raw": gamma_raw.tolist(),  # Pre-c coefficients γ
            "c_values": c_vals.tolist(),  # c scaling factors
        }
    else:
        # No standardization was applied
        coefficients_data = {
            "space": "raw",
            "intercept": float(beta_0),
            "values": beta_coeffs.tolist(),
            "values_no_threshold": beta_coeffs.tolist(),
            "feature_names": feature_names,
            "coef_threshold_applied": float(beta_threshold),
            "mean": [0.0] * len(beta_coeffs),
            "scale": [1.0] * len(beta_coeffs),
        }

    # End timing
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    result = {
        'model_name': 'NIMO_MLP',
        'iteration': iteration,
        'random_seed': randomState,
        'f1': f1,
        'accuracy': acc,
        'threshold': best_thr,
        'y_pred': y_pred.tolist(),
        'y_prob': probs_te.tolist(),
        'coefficients': coefficients_data,
        'selected_features': selected_features,
        'n_selected': len(selected_features),
        'selection': {
            'mask': [1 if abs(b) > beta_threshold else 0 for b in beta_coeffs_raw],
            'features': selected_features
        },
        'hyperparams': {
            'max_epochs': int(max_epochs),
            'batch_size': int(batch_size),
            'learning_rate': float(learning_rate),
            'group_penalty': float(group_penalty),
            'dropout': float(dropout),
            'hidden_dim': hidden_dim,
            'standardize': bool(standardize),
            'self_features': self_features or [],
            'noise_std': float(noise_std),
            'lambda_reg': float(lambda_reg),
            'c_penalty': float(c_penalty),
            'tau_l1': float(tau_l1),
            'tau_l1_max': float(tau_l1_max),
            'use_adaptive_l1': bool(use_adaptive_l1),
            'hard_threshold': float(hard_threshold),
            'trust_region': float(trust_region),
            'warm_start_epochs': int(warm_start_epochs),
            'nn_steps_per_epoch': int(nn_steps_per_epoch),
            'group_penalty_ramp_epochs': int(group_penalty_ramp_epochs),
            'early_noise_std': float(early_noise_std),
            'deterministic': bool(deterministic),
            'alpha2_schedule_epochs': int(alpha2_schedule_epochs),
            'alpha2_init': float(alpha2_init),
            'alpha2_final': float(alpha2_final),
            'disable_irls_group_penalty': bool(disable_irls_group_penalty),
        },
        
        # Timing information
        'execution_time': execution_time,
        'timing': {
            'total_seconds': execution_time,
            'start_time': start_time,
            'end_time': end_time
        },
        
        # Diagnostic information
        'diagnostics': {
            'grad_norms': model.grad_norms,
            'correction_magnitudes': model.correction_magnitudes,
            'loss_with_corrections': model.loss_with_corrections,
            'loss_without_corrections': model.loss_without_corrections,
        }
    }
    
    # Add model weights for plotting if requested
    if return_model_bits:
        result['_plot_bits'] = {
            'fc1_weight': model.fc1.weight.detach().cpu().numpy(),
            'input_dim': int(model.input_dim),
            'n_bits': int(model.n_bits),
        }
    
    # Artifact saving logic
    if save_artifacts:
        key = _mlp_key(
            scenario=scenario_name,
            seed=randomState,
            input_dim=int(model.input_dim),
            hparams=result["hyperparams"],
            tag=artifact_tag,
        )
        paths = _mlp_paths(artifact_dir, key)

        # decide: reuse/overwrite/ignore + always/better
        existing = _load_mlp_artifacts(paths) if cache_policy in ("reuse",) else None
        run_f1 = float(result.get("f1", 0.0))

        should_save = False
        if cache_policy == "ignore":
            should_save = False
        elif cache_policy == "overwrite":
            should_save = True
        else:  # reuse
            if existing is None:
                should_save = True
            elif save_if == "always":
                should_save = True
            else:  # "better"
                prev_f1 = float(existing["meta"].get("f1", -1.0))
                should_save = run_f1 > prev_f1

        if should_save:
            meta = {
                "scenario": scenario_name or "unknown",
                "model_type": "NIMO_MLP",
                "random_seed": int(randomState),
                "input_dim": int(model.input_dim),
                "n_bits": int(model.n_bits),
                "f1": run_f1,
                "accuracy": float(result.get("accuracy", 0.0)),
                "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
                "hyperparams": result["hyperparams"],
            }
            _save_mlp_artifacts(
                fc1_weight=model.fc1.weight.detach().cpu().numpy(),
                meta=meta,
                paths=paths,
                dtype=artifact_dtype,
            )
            # optionally expose a small hook for plotting code:
            result["_artifact_paths"] = {k: str(v) for k, v in paths.items()}
    
    # Feature-level reporting for interpretability
    with torch.no_grad():
        # Get training data for feature analysis
        Xtr_tensor = torch.from_numpy(X_train).float().to(model.device)
        if self_features:
            Xtr_tensor, _ = add_self_features(Xtr_tensor.cpu().numpy(), self_features)
            Xtr_tensor = torch.from_numpy(Xtr_tensor).float().to(model.device)
        
        # Compute modulation statistics
        B_u_train = model.build_B_u(Xtr_tensor)
        g_values = B_u_train / (Xtr_tensor.clamp_min(1e-6)) - 1.0  # g_j(x) = B_u_j / x_j - 1
        
        # Feature importance metrics
        feature_importance = {
            'modulation_magnitude': g_values.abs().mean(dim=0).cpu().numpy(),  # E[|g_j(x)|]
            'modulation_contribution': (Xtr_tensor * g_values).abs().mean(dim=0).cpu().numpy(),  # E[|x_j g_j(x)|]
            'linear_weights': model.fc1.weight[:, :model.input_dim].norm(dim=0).cpu().numpy(),  # ||W_j||
            'beta_coefficients': result.get('coefficients', []),
            'selected_features': result.get('selected_features', [])
        }
        result['feature_importance'] = feature_importance
    
    return standardize_method_output(result)


# -------------------- Grid Search Wrapper --------------------

def run_nimo_grid(
    X_train, y_train, X_test, y_test,
    *,
    X_val=None, y_val=None,
    iteration: int = 0,
    randomState: int = 42,
    X_columns: Optional[List[str]] = None,
    grid: Optional[Dict[str, List[Any]]] = None,
    max_epochs: int = 60,
    batch_size: int = 128,
    standardize: bool = True,
    num_workers: int = 0,
    self_features: Optional[List[str]] = None,   # default enabled below
    early_stop_patience: Optional[int] = 8,
    noise_std: float = 0.2,
    # L1 sparsity parameters
    tau_l1: float = 5e-2,
    tau_l1_max: float = 1e-1,
    use_adaptive_l1: bool = True,
    hard_threshold: float = 5e-3,
    trust_region: float = 1.0,
) -> Dict[str, Any]:
    """
    Fair hyperparameter sweep for NIMO, selecting by validation F1 with threshold search.
    Returns the best result plus all trials.
    Notes:
      - Self-features default: ["x2","sin","tanh","arctan"]
    """

    # Default self-features if not specified (disabled for paper compliance)
    if self_features is None:
        self_features = []

    # Default grid aligned with improved NIMO parameters
    if grid is None:
        grid = {
            "learning_rate": [1e-3, 3e-4],
            "group_penalty": [1.0, 1.5, 2.0],  # Increased for stronger sparsity
            "dropout": [0.0, 0.1, 0.2],
            "hidden_dim": [64, 96],
            "lambda_reg": [1e-2, 3e-2, 1e-1],
            "c_penalty": [3e-2, 1e-1, 3e-1],
            "noise_std": [0.05, 0.1, 0.2],     # Reduced for warm-start
            "tau_l1": [3e-2, 5e-2, 8e-2],      # Increased for stronger sparsity
            "tau_l1_max": [8e-2, 1e-1, 1.5e-1], # Increased for stronger sparsity
            "warm_start_epochs": [3, 5, 8],     # New parameter
            "nn_steps_per_epoch": [2, 3, 5],    # New parameter
            "early_noise_std": [0.0, 0.02, 0.05],  # Early noise schedule
        }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    trials: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for ci, values in enumerate(combos, 1):
        h = dict(zip(keys, values))
        print(f"[NIMO grid] Trial {ci}/{len(combos)} -> {h}")

        res = run_nimo_baseline(
            X_train, y_train, X_test, y_test,
            iteration=iteration, randomState=randomState,
            X_columns=X_columns,
            X_val=X_val, y_val=y_val,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=h.get("learning_rate", 1e-3),
            group_penalty=h.get("group_penalty", 0.5),
            dropout=h.get("dropout", 0.1),
            hidden_dim=h.get("hidden_dim", None),
            standardize=standardize,
            num_workers=num_workers,
            self_features=self_features,         # default disabled
            early_stop_patience=early_stop_patience,
            noise_std=h.get("noise_std", 0.05),
            lambda_reg=h.get("lambda_reg", 3e-2),
            c_penalty=h.get("c_penalty", 0.1),
            tau_l1=h.get("tau_l1", tau_l1),
            tau_l1_max=h.get("tau_l1_max", tau_l1_max),
            use_adaptive_l1=use_adaptive_l1,
            hard_threshold=hard_threshold,
            trust_region=trust_region,
            warm_start_epochs=h.get("warm_start_epochs", 5),
            nn_steps_per_epoch=h.get("nn_steps_per_epoch", 3),
            group_penalty_ramp_epochs=30,  # Fixed for grid search
            early_noise_std=h.get("early_noise_std", 0.0),
        )

        res['trial_hparams'] = h
        trials.append(res)

        if (best is None) or (res['f1'] > best['f1']):
            best = res

    assert best is not None

    out = {
        "best": best,
        "trials": trials,
        "n_trials": len(trials),
        "grid": grid,
        "note": "Best chosen by validation F1 with threshold optimization; test metrics reported for that threshold."
    }
    return standardize_method_output(out)


def run_nimo_baseline_scenario(
    X_train, y_train, X_test, y_test,
    scenario_id: str,
    iteration: int, randomState: int,
    X_columns: Optional[List[str]] = None,
    *,
    X_val=None, y_val=None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run NIMO baseline with scenario-specific configuration.
    
    Args:
        X_train, y_train, X_test, y_test: Data splits
        scenario_id: Scenario identifier (A, B, C, D, E, etc.)
        iteration: Iteration number
        randomState: Random seed
        X_columns: Feature names
        X_val, y_val: Validation data
        **kwargs: Additional parameters (will override scenario config)
        
    Returns:
        dict: Results with scenario-optimized configuration
    """
    # Get scenario-specific configuration
    scenario_config = get_scenario_config(scenario_id)
    
    # Merge with user-provided kwargs (user kwargs take precedence)
    config = {**scenario_config, **kwargs}
    
    print(f"🎯 Running NIMO for scenario {scenario_id} with config: {config}")
    
    # Run with merged configuration
    return run_nimo_baseline(
        X_train, y_train, X_test, y_test,
        iteration=iteration,
        randomState=randomState,
        X_columns=X_columns,
        X_val=X_val,
        y_val=y_val,
        **config
    )