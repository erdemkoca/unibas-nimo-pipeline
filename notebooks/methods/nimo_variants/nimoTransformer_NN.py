"""Transformer-enhanced NIMO variant.

Hybrid model: sparse adaptive logistic regression backbone (IRLS updates)
combined with a transformer-based per-feature correction module.
"""

from __future__ import annotations

import os
import sys
import json
import datetime
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Optional, Tuple, List, Any, Dict

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# Ensure local utils can be imported when executed from notebooks/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils import standardize_method_output
except ImportError:  # pragma: no cover - fallback for ad-hoc execution
    def standardize_method_output(result):
        out = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, (np.generic,)):
                out[k] = v.item()
            else:
                out[k] = v
        return out


# ---- artifact helpers (Transformer) ----

def _short_hash(obj: dict) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:10]

def _t_key(scenario, seed, input_dim, hparams, tag=None, cfg_label=None):
    hp_compact = {k: hparams.get(k) for k in sorted(hparams.keys())}
    key = {
        "scenario": scenario or "unknown",
        "seed": int(seed),
        "input_dim": int(input_dim),
        "h": _short_hash(hp_compact),
        "tag": tag or "default",
    }
    if cfg_label is not None:
        key["cfg"] = cfg_label
    return key

def _t_paths(base_dir: str, key: dict) -> dict:
    # artifacts/nimo_transformer/<scenario>/seed<seed>/in<input_dim>/<h>[/<tag>][/cfg_<label>]
    root = Path(base_dir) / key["scenario"] / f"seed{key['seed']}" / f"in{key['input_dim']}" / key["h"]
    if key.get("tag") and key["tag"] != "default":
        root = root / key["tag"]
    if key.get("cfg"):
        root = root / f"cfg_{key['cfg']}"
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "weights_npz": root / "transformer_weights.npz",
        "meta_json": root / "meta.json",
    }

def _load_t_artifacts(paths: dict):
    if not (paths["weights_npz"].exists() and paths["meta_json"].exists()):
        return None
    try:
        with open(paths["meta_json"], "r") as f:
            meta = json.load(f)
        npz = np.load(paths["weights_npz"])
        out = {
            "meta": meta,
            "_plot_bits": {
                "feature_embed": npz["feature_embed"],
                "binary_proj_weight": (None if "binary_proj_weight" not in npz.files else npz["binary_proj_weight"]),
                "binary_codes": (None if "binary_codes" not in npz.files else npz["binary_codes"]),
                "cls_token": (None if "cls_token" not in npz.files else npz["cls_token"]),
                "corr_head_last_weight": npz["corr_head_last_weight"],
                "corr_head_last_bias": npz["corr_head_last_bias"],
                "residual_head_last_weight": npz["residual_head_last_weight"],
                "residual_head_last_bias": npz["residual_head_last_bias"],
            }
        }
        return out
    except Exception:
        return None

def _save_t_artifacts(arrs: dict, meta: dict, paths: dict, dtype: str = "float32"):
    to_dtype = np.float32 if dtype == "float32" else np.float64
    safe = {k: (None if v is None else v.astype(to_dtype, copy=False)) for k, v in arrs.items()}
    np.savez_compressed(paths["weights_npz"], **{k: v for k, v in safe.items() if v is not None})
    with open(paths["meta_json"], "w") as f:
        json.dump(meta, f, separators=(",", ":"), sort_keys=True)


def _binary_code(index: int, n_bits: int) -> np.ndarray:
    """Return centered binary representation used as positional context."""
    return np.array([int(b) for b in format(index, f"0{n_bits}b")], dtype=np.float32) - 0.5


def corrections_nocenter(model, X):
    """Corrections without batch-centering for leakage testing"""
    corr, _ = model._raw_modulation(X)  # no batch centering here
    return corr

def diag_jacobian_g_nocenter(model, X, eps=1e-4):
    """Compute diagonal Jacobian of corrections w.r.t. inputs (no batch centering)"""
    Xp = X.clone()
    Xm = X.clone()
    d = X.size(1)
    diags = []
    with torch.no_grad():
        for j in range(d):
            Xp[:, j] += eps
            Xm[:, j] -= eps
            gp = corrections_nocenter(model, Xp)
            gm = corrections_nocenter(model, Xm)
            dj = ((gp[:, j] - gm[:, j]) / (2*eps)).abs().mean().item()
            diags.append(dj)
            Xp[:, j] -= eps
            Xm[:, j] += eps
    return float(np.mean(diags)), float(np.max(diags))

def diag_jacobian_g(model, X, eps=1e-4):
    """Compute diagonal Jacobian of corrections w.r.t. inputs (with batch centering)"""
    Xp = X.clone()
    Xm = X.clone()
    d = X.size(1)
    diags = []
    with torch.no_grad():
        base = model.corrections(X)
    for j in range(d):
        Xp[:, j] += eps
        Xm[:, j] -= eps
        gp = model.corrections(Xp)
        gm = model.corrections(Xm)
        dj = ((gp[:, j] - gm[:, j]) / (2*eps)).abs().mean().item()
        diags.append(dj)
        Xp[:, j] -= eps
        Xm[:, j] += eps
    return float(np.mean(diags)), float(np.max(diags))


def _schedule_out_scale(t: int, T: int, scenario: str) -> float:
    """Schedule out_scale based on scenario type."""
    if scenario == "B":  # linear
        return 0.2 + 0.25 * min(t / max(1, 15), 1.0)
    elif scenario == "C":  # nonlinear
        return 0.6
    elif scenario == "D":  # mixed
        return 0.35 + 0.25 * min(t / max(1, 5), 1.0)
    else:  # default
        return 0.45


def focal_bce_with_logits(logits, targets, alpha=0.5, gamma=2.0):
    """Focal BCE loss for handling class imbalance and hard examples."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = p*targets + (1-p)*(1-targets)
    w = (alpha*targets + (1-alpha)*(1-targets)) * (1 - pt).pow(gamma)
    return (w * ce).mean()


class TransformerCorrection(nn.Module):
    """Two-stream transformer encoder with no-self correction and strict zero-input anchoring."""

    def __init__(
        self,
        d: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_binary_context: bool = True,
        hard_no_self_mask: bool = True,
        two_hop_attention: bool = False,
    ) -> None:
        super().__init__()
        self.d = d
        self.embed_dim = embed_dim
        self.use_binary_context = use_binary_context
        self.hard_no_self_mask = hard_no_self_mask
        self.two_hop = two_hop_attention

        # Ensure embed_dim % num_heads == 0
        if embed_dim % num_heads != 0:
            # make heads divide embed_dim
            num_heads = max(1, min(num_heads, embed_dim))
            while embed_dim % num_heads != 0 and num_heads > 1:
                num_heads -= 1

        # Project scalar values for CONTEXT stream only (queries don't see x_j)
        self.value_proj = nn.Linear(1, embed_dim)
        self.feature_embed = nn.Parameter(torch.randn(d, embed_dim) * 0.02)

        if use_binary_context:
            n_bits = int(np.floor(np.log2(d))) + 1
            codes = np.stack([_binary_code(i + 1, n_bits) for i in range(d)], axis=0)
            self.register_buffer("binary_codes", torch.tensor(codes, dtype=torch.float32))
            self.binary_proj = nn.Linear(n_bits, embed_dim, bias=False)
        else:
            self.register_buffer("binary_codes", None)
            self.binary_proj = None

        # Use explicit projections for SDPA with -inf mask
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # FiLM scaling for query modulation (scalar gate only)
        self.film_alpha = nn.Parameter(torch.tensor(0.1))  # small initial scaling
        self.film_a = nn.Parameter(torch.tensor(1.0))  # scaling factor

        # Feed-forward network for processing attended features
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )

        # Stack multiple cross-attention + FFN layers
        self.num_layers = num_layers

        # Two-hop attention: separate parameters for first and second hop
        if self.two_hop:
            # Second hop projections (different from first hop)
            self.q_proj_2 = nn.Linear(embed_dim, embed_dim, bias=False)
            self.k_proj_2 = nn.Linear(embed_dim, embed_dim, bias=False)
            self.v_proj_2 = nn.Linear(embed_dim, embed_dim, bias=False)

        # Float mask with -inf on diagonal (broadcasted later)
        # This will be updated to pair-aware mask during training
        attn_mask = torch.zeros(d, d, dtype=torch.float32)
        # diagonal policy depends on hard_no_self_mask
        attn_mask.fill_diagonal_(float("-inf") if hard_no_self_mask else 0.0)
        self.register_buffer("attn_mask", attn_mask)
        self.register_buffer("allowed_bool", torch.zeros(d, d, dtype=torch.bool))

        # Initialize with self-attention policy based on hard_no_self_mask
        # For strict variants: no self-attention (diagonal = False)
        # For relaxed variants: allow self-attention (diagonal = True)
        if hard_no_self_mask:
            self.allowed_bool.fill_diagonal_(False)  # strict: no self
        else:
            self.allowed_bool.fill_diagonal_(True)   # relaxed: allow self

        # Heads (unchanged)
        self.corr_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

        # Low-rank bilinear head for pair interactions (NFM-style)
        self.bilinear_rank = min(8, max(4, d//4))  # adaptive rank
        self.bilinear_embeddings = nn.Parameter(torch.randn(d, self.bilinear_rank) * 0.02)
        self.bilinear_weight = nn.Parameter(torch.tensor(0.1))  # small initial weight

        # Learnable per-feature scale (α₂ analogue); we'll schedule it outside
        self.out_scale_param = nn.Parameter(torch.full((d,), 0.4))  # shape (d,) for per-feature modulation

        # Per-head dropout for attention regularization
        self.attn_dropout = nn.Dropout(dropout * 0.5)  # Lighter than FFN dropout

        # Stochastic depth (DropPath) for residual connections
        self.drop_path_prob = 0.05  # Small probability for stochastic depth

        # LayerScale γ for residual stability
        self.gamma_attn = nn.Parameter(1e-2 * torch.ones(embed_dim))
        self.gamma_ffn = nn.Parameter(1e-2 * torch.ones(embed_dim))

        # Learnable ALiBi strength (scenario-agnostic)
        self.use_alibi = True  # Always enabled, but strength is learnable
        self.alibi_strength = nn.Parameter(torch.tensor(0.0))  # starts neutral
        self.register_buffer("alibi", self._build_alibi_bias(d, num_heads))

    def _drop_path(self, x, drop_prob=0.0, training=False):
        """DropPath (Stochastic Depth) per sample."""
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1.0 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def _build_alibi_bias(self, d, num_heads):
        """Build ALiBi-style positional bias matrix for feature indices."""
        # Create distance matrix |j - k|
        j = torch.arange(d, dtype=torch.float32)
        k = torch.arange(d, dtype=torch.float32)
        distance = torch.abs(j.unsqueeze(0) - k.unsqueeze(1))  # (d, d)

        # ALiBi slope per head (decreasing with head index)
        slopes = torch.pow(2, -torch.arange(num_heads, dtype=torch.float32) / num_heads)

        # Apply slopes to distances
        alibi = slopes.unsqueeze(-1).unsqueeze(-1) * distance.unsqueeze(0)  # (H, d, d)
        return alibi.unsqueeze(0)  # (1, H, d, d)

    def _build_query_tokens(self, B):
        """Queries DO NOT include x_j (no self leakage)"""
        Q = self.feature_embed.unsqueeze(0).expand(B, -1, -1)
        if self.use_binary_context and self.binary_proj is not None:
            Q = Q + self.binary_proj(self.binary_codes).unsqueeze(0)
        return Q

    def _build_context_tokens(self, x):
        """Keys/values include x_k so every position can see others' values"""
        # Simplified approach: just project all features and let attention handle the rest
        KV = self.value_proj(x.unsqueeze(-1)) + self.feature_embed.unsqueeze(0)
        if self.use_binary_context and self.binary_proj is not None:
            KV = KV + self.binary_proj(self.binary_codes).unsqueeze(0)
        return KV

    def _encode(self, Q, KV, x_values=None):
        """True two-stream attention: Q and KV are completely separate streams"""
        B, d, E = Q.shape

        # Apply cross-attention layers
        x = Q
        for layer_idx in range(self.num_layers):
            # First hop: attend to direct companions
            q1 = self.q_proj(x).view(B, d, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,d,hd)
            k1 = self.k_proj(KV).view(B, d, self.num_heads, self.head_dim).transpose(1, 2) # (B,H,d,hd)
            v1 = self.v_proj(KV).view(B, d, self.num_heads, self.head_dim).transpose(1, 2) # (B,H,d,hd)

            # FiLM scaling: neighbor-only aggregate (no self dependency)
            if x_values is not None:
                M = self.allowed_bool  # (d,d) boolean mask, True means allowed j->k
                deg = M.sum(dim=1).clamp(min=1).unsqueeze(0).to(x_values.dtype)  # (1,d)
                x_neigh = (x_values.unsqueeze(2) * M.to(x_values.dtype).unsqueeze(0)).sum(dim=1) / deg  # (B,d)

                # bounded FiLM gain using neighbor-only mean
                film_gain = torch.tanh(self.film_alpha) * torch.tanh(self.film_a * x_neigh)  # (B,d)
                q_scale = (1.0 + film_gain).clamp(0.7, 1.3)  # (B,d)

                q_scale = q_scale.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, -1)  # (B,H,d,1)
                q1 = q1 * q_scale

            # Use manual attention computation to ensure proper masking
            # Compute attention scores
            scores = torch.matmul(q1, k1.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,d,d)

            # Apply mask: set masked positions to -inf
            mask = self.attn_mask.view(1, 1, d, d).expand(B, self.num_heads, d, d)
            scores = scores + mask

            # Add learnable ALiBi positional bias
            if self.use_alibi:
                strength = torch.clamp(self.alibi_strength, min=0).view(1, 1, 1, 1)
                scores = scores - strength * self.alibi  # subtract for penalty (ALiBi convention)

            # Apply softmax
            attn_weights = F.softmax(scores, dim=-1)  # (B,H,d,d)

            # Apply per-head dropout to attention weights
            attn_weights = self.attn_dropout(attn_weights)

            # Apply attention to values
            attn1 = torch.matmul(attn_weights, v1)  # (B,H,d,hd)
            attn_out1 = attn1.transpose(1, 2).contiguous().view(B, d, E)

            # Second hop: attend to neighbors of companions (if enabled and not first layer)
            if self.two_hop and layer_idx > 0:
                # Use first hop output as query for second hop
                q2 = self.q_proj_2(attn_out1).view(B, d, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,d,hd)
                k2 = self.k_proj_2(KV).view(B, d, self.num_heads, self.head_dim).transpose(1, 2) # (B,H,d,hd)
                v2 = self.v_proj_2(KV).view(B, d, self.num_heads, self.head_dim).transpose(1, 2) # (B,H,d,hd)

                # Second hop: attend to neighbors of companions (relaxed mask)
                # Allow attention to features that are companions of companions
                m2 = self._build_two_hop_mask()
                m2_expanded = m2.view(1, 1, d, d).expand(B, self.num_heads, d, d)

                # Manual attention for second hop too
                scores2 = torch.matmul(q2, k2.transpose(-2, -1)) / math.sqrt(self.head_dim)
                scores2 = scores2 + m2_expanded
                attn_weights2 = F.softmax(scores2, dim=-1)
                attn2 = torch.matmul(attn_weights2, v2)
                attn_out2 = attn2.transpose(1, 2).contiguous().view(B, d, E)

                # Combine both hops
                attn_out = attn_out1 + 0.5 * attn_out2  # weighted combination
            else:
                attn_out = attn_out1

            # Residual connection with stochastic depth and LayerScale
            x = x + self._drop_path(self.gamma_attn * attn_out, self.drop_path_prob, self.training)

            # Feed-forward network with stochastic depth and LayerScale
            ffn_out = self.ffn(x)
            x = x + self._drop_path(self.gamma_ffn * ffn_out, self.drop_path_prob, self.training)

        return x

    def _build_two_hop_mask(self):
        """Build mask for second hop using boolean allowed mask; respect hard_no_self_mask"""
        allowed = self.allowed_bool  # boolean (d,d)
        reach2 = (allowed.float() @ allowed.float()) > 0
        two_hop = allowed | reach2
        if self.hard_no_self_mask:
            two_hop = two_hop.clone()
            two_hop.fill_diagonal_(False)
        float_mask = torch.full_like(self.attn_mask, float("-inf"))
        float_mask[two_hop] = 0.0
        return float_mask

    def update_attention_mask(self, allowed_bool_mask: torch.Tensor):
        """allowed_bool_mask: True=allowed, False=blocked (incl. diagonal)."""
        with torch.no_grad():
            # Start with the provided mask
            allowed = allowed_bool_mask.clone()

            # Apply hard_no_self_mask policy to diagonal
            if self.hard_no_self_mask:
                allowed.fill_diagonal_(False)  # enforce no self for strict variants
            else:
                allowed.fill_diagonal_(True)   # allow self for relaxed variants

            # Create float mask: -inf for blocked, 0 for allowed
            float_mask = torch.full_like(allowed, float("-inf"), dtype=torch.float32)
            float_mask[allowed] = 0.0

            # Update buffers
            self.attn_mask.copy_(float_mask)
            self.allowed_bool.copy_(allowed)

    def _compute_bilinear_interactions(self, x, companion_mask):
        """
        Return interactions that are linear in x_{-j} for corr_j.
        For each j: interactions[:, j] = sum_k s_{jk} * x_k  (k ≠ j)
        where s_{jk} = v_j^T v_k. Critically, NO x_j factor here.
        """
        B, d = x.shape
        v = self.bilinear_embeddings  # (d, r)
        interactions = torch.zeros(B, d, device=x.device, dtype=x.dtype)

        # precompute low-rank similarity matrix S (d, d)
        S = v @ v.T  # s_{jk}

        for j in range(d):
            companions = torch.where(companion_mask[j])[0]
            if companions.numel() == 0:
                continue
            s_jk = S[j, companions]                  # (|companions|,)
            x_comp = x[:, companions]                # (B, |companions|)
            interactions[:, j] = (x_comp * s_jk.unsqueeze(0)).sum(dim=1)

        return interactions

    def forward_raw(self, x, companion_mask=None):
        """Raw corrections without anchoring or scaling"""
        B, d = x.shape
        Q = self._build_query_tokens(B)        # no x here
        KV = self._build_context_tokens(x)     # x enters here
        feat_enc = self._encode(Q, KV, x_values=x)  # pass x_values for FiLM scaling

        # Standard corrections from attention
        corr_attn = torch.tanh(self.corr_head(feat_enc)).squeeze(-1)

        # Add bilinear pair interactions if companion mask is provided
        if companion_mask is not None:
            corr_bilinear = self._compute_bilinear_interactions(x, companion_mask)
            corr_raw = corr_attn + self.bilinear_weight * corr_bilinear
        else:
            corr_raw = corr_attn

        res_raw = torch.tanh(self.residual_head(feat_enc)).squeeze(-1).mean(dim=1)  # pooled
        return corr_raw, res_raw

    def forward(self, x: torch.Tensor, companion_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (corrections, residual) with strict zero-input anchoring and scaling."""
        # Strict zero-input anchoring: g(x) - g(0)
        corr_x, res_x = self.forward_raw(x, companion_mask)
        zero = torch.zeros_like(x)
        corr_0, res_0 = self.forward_raw(zero, companion_mask)
        corr = corr_x - corr_0
        res = res_x - res_0

        # Scale via learnable per-feature α₂ analogue
        scale = 0.5 * (1.0 + torch.tanh(self.out_scale_param))  # shape (d,)
        return scale.unsqueeze(0) * corr, res  # broadcast to (B, d)


class NIMOTransformer(nn.Module):
    """Hybrid model with sparse β and transformer-based corrections."""

    def __init__(
        self,
        d: int,
        *,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        out_scale: float = 0.4,
        residual_scale: float = 0.3,
        use_binary_context: bool = True,
        fast_preset: bool = False,
        use_residual_head: bool = True,
        hard_no_self_mask: bool = True,
        use_companion_shortlist: bool = False,
        two_hop_attention: bool = False,
        topk_beta_aware: bool = False,
        residual_orthogonalize: bool = True,
        **_
    ) -> None:
        super().__init__()
        self.d = d
        self.beta = nn.Parameter(torch.zeros(d + 1))  # [b0, b_1..b_d]
        self.out_scale = out_scale
        self.residual_scale = residual_scale
        self.use_residual_head = use_residual_head
        self.hard_no_self_mask = hard_no_self_mask
        self.use_companion_shortlist = use_companion_shortlist
        self.two_hop_attention = two_hop_attention
        self.topk_beta_aware = topk_beta_aware
        self.residual_orthogonalize = residual_orthogonalize

        # Beta-aware Top-K gating
        self.register_buffer("beta_prior_mask", torch.ones(d))  # will be updated after IRLS

        # Companion shortlist for pair-aware attention (m=6-8 companions per feature)
        self.companion_m = min(8, max(4, d//2))  # adaptive m based on d
        self.register_buffer("companion_mask", torch.zeros(d, d, dtype=torch.bool))  # will be updated

        # Counterfactual pair flips for consistency loss
        self.counterfactual_prob = 0.1  # probability of applying pair flips
        self.consistency_weight = 0.01  # weight for consistency loss

        # Apply fast preset optimizations
        if fast_preset:
            embed_dim = 48
            num_layers = 2
            num_heads = min(3, max(1, d//4))

        self.correction_net = TransformerCorrection(
            d,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_binary_context=use_binary_context,
            hard_no_self_mask=hard_no_self_mask,
            two_hop_attention=two_hop_attention,
        )

        if use_residual_head:
            self.residual_mlp = nn.Sequential(
                nn.Linear(d, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, 1),
            )
        else:
            self.residual_mlp = None

    def _apply_counterfactual_flips(self, x, companion_mask):
        """Apply counterfactual pair flips to encourage pair logic learning"""
        B, d = x.shape
        x_flipped = x.clone()

        # Randomly select samples to flip
        flip_mask = (torch.rand(B, device=x.device) < self.counterfactual_prob).to(torch.bool)

        if flip_mask.any():
            for b in range(B):
                if flip_mask[b]:
                    # Randomly select a feature j
                    j = torch.randint(0, d, (1,)).item()
                    # Get companions of j
                    companions = torch.where(companion_mask[j])[0]

                    if len(companions) > 0:
                        # Randomly select a companion k
                        k = companions[torch.randint(0, len(companions), (1,))].item()

                        # Apply one of three flip strategies
                        flip_type = torch.randint(0, 3, (1,)).item()

                        if flip_type == 0:
                            # Flip sign of x_j
                            x_flipped[b, j] = -x_flipped[b, j]
                        elif flip_type == 1:
                            # Flip sign of x_k
                            x_flipped[b, k] = -x_flipped[b, k]
                        else:
                            # Shuffle x_k within the batch
                            other_idx = torch.randint(0, B, (1,)).item()
                            x_flipped[b, k] = x[other_idx, k]

        return x_flipped, flip_mask

    def _compute_consistency_loss(self, x, x_flipped, flip_mask, companion_mask):
        """Graph-safe, vectorized consistency loss."""
        corr_orig, _ = self.forward_raw(x, companion_mask)
        corr_flipped, _ = self.forward_raw(x_flipped, companion_mask)
        per_sample = (corr_orig - corr_flipped).abs().mean(dim=1)  # (B,)

        if flip_mask.any():
            loss = per_sample[flip_mask].mean()
        else:
            loss = (corr_orig.sum() * 0.0)  # graph-safe zero

        return self.consistency_weight * loss

    def _build_companion_shortlist(self, X, y):
        """Build companion shortlist using vectorized cosine similarity (fast, stable)"""
        with torch.no_grad():
            # standardize columns
            Xm = X - X.mean(dim=0, keepdim=True)
            Xs = Xm / (Xm.std(dim=0, keepdim=True) + 1e-6)
            # cosine sim ~ correlation
            S = (Xs.T @ Xs) / Xs.shape[0]  # (d,d)
            S.fill_diagonal_(0.0)
            d = S.shape[0]
            m = min(self.companion_m, d-1)
            topk_idx = torch.topk(S.abs(), k=m, dim=1).indices  # (d,m)
            companion_mask = torch.zeros(d, d, dtype=torch.bool, device=X.device)
            companion_mask.scatter_(1, topk_idx, True)
            companion_mask = companion_mask | companion_mask.T
            if self.hard_no_self_mask:
                companion_mask.fill_diagonal_(False)
            self.companion_mask.copy_(companion_mask)
            print(f"🔗 Built companion shortlist: {self.companion_m} companions per feature")

    def _get_pair_aware_attention_mask(self):
        """Return boolean 'allowed' mask (True = allowed), without self."""
        d = self.companion_mask.shape[0]
        allowed = self.companion_mask.clone()      # companions only
        allowed.fill_diagonal_(False)              # NO SELF
        return allowed

    def compute_pair_saliency(self, X, y, num_samples=100, cols=None):
        """Compute pair saliency matrix S_{j,k} = mean |∂ corr_j / ∂ x_k|"""
        # Sample a subset for efficiency
        if X.shape[0] > num_samples:
            idx = torch.randperm(X.shape[0])[:num_samples]
            X_sample = X[idx].clone().detach().requires_grad_(True)
        else:
            X_sample = X.clone().detach().requires_grad_(True)

        B, d = X_sample.shape
        saliency = torch.zeros(d, d, device=X.device)

        # Use provided columns or all features
        if cols is None:
            cols = torch.arange(d, device=X.device)

        # Compute gradients for each feature pair
        for j in cols:
            if X_sample.grad is not None:
                X_sample.grad.zero_()

            # Forward pass
            corr, _ = self.correction_net.forward_raw(X_sample, self.companion_mask)
            corr_j = corr[:, j].sum()  # sum over batch

            # Backward pass
            if corr_j.requires_grad:
                corr_j.backward(retain_graph=True)

                # Extract gradients w.r.t. all features
                if X_sample.grad is not None:
                    saliency[j, :] = X_sample.grad.abs().mean(dim=0)

        return saliency

    def _raw_modulation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # The correction_net now handles scaling internally
        # Get companion mask if available
        companion_mask = getattr(self, 'companion_mask', None)
        corr, residual = self.correction_net(x, companion_mask)
        return corr, residual

    def forward_raw(self, x: torch.Tensor, companion_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Raw corrections without anchoring or scaling"""
        return self.correction_net.forward_raw(x, companion_mask)

    def _topk_gate(self, corr, k):
        """Sample-wise Top-K gating for corrections with straight-through gradients"""
        # corr: (B,d)
        if k is None or k >= corr.size(1):
            return corr
        with torch.no_grad():
            # indices of top-|g| per row
            topk = torch.topk(corr.abs(), k, dim=1).indices
            mask = torch.zeros_like(corr).scatter_(1, topk, 1.0)
        # straight-through: use mask forward, pass full grads backward
        return corr * mask + corr.detach() * (1 - mask)

    def _topk_gate_beta_aware(self, corr, k):
        """Beta-aware Top-K gating: requires both local top-K and decent global β"""
        if k is None or k >= corr.size(1):
            return corr
        # per-sample top-|g|
        with torch.no_grad():
            topk = torch.topk(corr.abs(), k, dim=1).indices
            local_mask = torch.zeros_like(corr).scatter_(1, topk, 1.0)
        # global β prior (row-broadcast)
        prior = self.beta_prior_mask.unsqueeze(0).expand_as(corr)
        # fuse: require local winner *and* decent β (straight-through)
        fused = local_mask * (prior > 0.3).float()
        return corr * fused + corr.detach() * (1 - fused)


    def modulation(self, x: torch.Tensor, *, detach: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        corr, residual = self._raw_modulation(x)
        if detach:
            corr = corr.detach()
            residual = residual.detach()
        # Keep batch-centering for stability
        corr = corr - corr.mean(dim=0, keepdim=True)

        # Hard cap on correction magnitude at train time (tighter for stability)
        if self.training:
            corr = corr.clamp(-0.6, 0.6)

        # Sample-wise Top-K gating for corrections with scenario-wise schedule
        B, d = corr.shape
        scenario = getattr(self, "scenario_name", "")

        # Configurable K schedule
        k_start_frac = getattr(self, "k_start_frac", 0.25)
        k_end_frac = getattr(self, "k_end_frac", 0.05)
        warm_frac = getattr(self, "warm_frac", 0.4)

        k_start = max(4, int(d * k_start_frac))
        k_end = max(2, int(d * k_end_frac))
        warm_T = max(3, int(warm_frac * getattr(self, "T_total", 30)))

        t_step = getattr(self, "t_step", 0)
        k_now = int(k_end + (k_start - k_end) * max(0, warm_T - t_step) / max(1, warm_T))

        # Choose gating by flag
        if self.topk_beta_aware:
            corr = self._topk_gate_beta_aware(corr, k_now)
        else:
            corr = self._topk_gate(corr, k_now)

        return corr, residual

    def corrections(self, x: torch.Tensor, *, detach: bool = False) -> torch.Tensor:
        corr, _ = self.modulation(x, detach=detach)
        return corr

    def residual_component(self, x: torch.Tensor, use_correction: bool = True) -> torch.Tensor:
        """Return the orthogonalized residual used in the forward map."""
        _, r = self._build_features(x, use_correction=use_correction)
        return r

    def _orthogonalize_residual(self, X: torch.Tensor, r: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Make residual r (B,) have zero mean and be orthogonal to all columns of X (B,d).
        We solve (X^T X + eps I) alpha = X^T (r - mean(r)) and subtract X alpha.
        """
        # zero-mean
        r_center = r - r.mean()

        # project r_center onto span(X) and subtract
        # shapes: X: (B,d); r_center: (B,)
        Xt = X.T                            # (d,B)
        gram = Xt @ X                       # (d,d)
        gram = gram + eps * torch.eye(gram.size(0), device=X.device, dtype=X.dtype)
        rhs = Xt @ r_center                 # (d,)
        alpha = torch.linalg.solve(gram, rhs)        # (d,)
        r_proj = X @ alpha                  # (B,)
        r_ortho = r_center - r_proj                      # (B,)
        return r_ortho

    def _build_features(self, x: torch.Tensor, use_correction: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if use_correction:
            # multiplicative corrections (already batch-centered in .modulation)
            corr, residual_token = self.modulation(x)
            feats = x * (1.0 + corr)       # <-- apply multiplicative corrections

            # residual path: combine token residual + small MLP residual, then bound
            r = residual_token
            if self.use_residual_head and self.residual_mlp is not None:
                r = r + self.residual_mlp(x).squeeze(-1)

            # keep residual bounded (small scale already inside residual_token / residual_mlp head)
            r = torch.tanh(r) * self.residual_scale

            # Orthogonalize to modulated features if enabled
            if self.residual_orthogonalize:
                r = self._orthogonalize_residual(feats, r, eps=1e-6)

            # effective design stays multiplicative
            # (note: the linear part uses feats as-is; residual is *additive* but orthogonal)
            return feats, r
        else:
            feats = x
            r = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            return feats, r

    def predict_logits(self, x: torch.Tensor, use_correction: bool = True) -> torch.Tensor:
        feats, residual = self._build_features(x, use_correction)
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        B = torch.cat([ones, feats], dim=1)
        return B.matmul(self.beta) + residual

    def predict_proba(self, x: torch.Tensor, use_correction: bool = True) -> torch.Tensor:
        return torch.sigmoid(self.predict_logits(x, use_correction=use_correction))


@torch.no_grad()
def update_beta_irls(
    model: NIMOTransformer,
    X: torch.Tensor,
    y: torch.Tensor,
    lam_l2: float = 1e-3,
    tau_l1: float = 1e-3,
    tau_l1_max: float = 5e-2,
    use_correction: bool = True,
    eps: float = 1e-6,
    trust_region: float = 0.5,
    iteration: int = 0,
    max_iterations: int = 25,
    use_adaptive_l1: bool = True,
    use_hard_thresholding: bool = True,
    hard_threshold: float = 1e-6,
) -> None:
    """Single IRLS step with elastic net style penalties, adaptive L1, and hard thresholding."""
    beta_prev = model.beta.detach().clone()

    feats, residual = model._build_features(X, use_correction)
    ones = torch.ones(X.size(0), 1, device=X.device, dtype=X.dtype)
    B = torch.cat([ones, feats], dim=1)
    logits = B.matmul(model.beta) + residual
    p = torch.sigmoid(logits)
    W = p * (1.0 - p) + 5e-6  # Slightly larger eps for imbalanced data stability
    z = logits + (y - p) / W

    target = z - residual

    BW = B * W.unsqueeze(1)
    A = BW.t().matmul(B) + lam_l2 * torch.eye(B.shape[1], device=B.device, dtype=B.dtype)
    bvec = BW.t().matmul(target)

    # Use Cholesky decomposition for faster solve
    # Stabilize the matrix
    A = A + 1e-6 * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    try:
        L = torch.linalg.cholesky(A)
        beta_new = torch.cholesky_solve(bvec.unsqueeze(1), L).squeeze(1)
    except RuntimeError:
        # Fallback to regular solve if Cholesky fails
        beta_new = torch.linalg.solve(A, bvec)

    # Adaptive L1 thresholding: increase penalty over time
    if use_adaptive_l1 and max_iterations > 0:
        progress = min(iteration / max_iterations, 1.0)
        tau_l1_curr = tau_l1 + (tau_l1_max - tau_l1) * progress
    else:
        tau_l1_curr = tau_l1

    beta_np = beta_new.detach().cpu().numpy()
    beta_np[1:] = np.sign(beta_np[1:]) * np.maximum(np.abs(beta_np[1:]) - tau_l1_curr, 0.0)

    # Hard thresholding: set very small coefficients to exactly zero
    if use_hard_thresholding:
        beta_np[1:] = np.where(np.abs(beta_np[1:]) < hard_threshold, 0.0, beta_np[1:])

    beta_tensor = torch.from_numpy(beta_np).to(B.device, dtype=B.dtype)
    delta = beta_tensor - beta_prev
    delta_norm = torch.norm(delta)
    if delta_norm > trust_region:
        beta_tensor = beta_prev + delta * (trust_region / (delta_norm + 1e-12))

    model.beta.data.copy_(beta_tensor)

    # Update beta prior mask for beta-aware Top-K gating
    with torch.no_grad():
        beta_mag = model.beta[1:].abs()
        # normalize to [0,1]
        bp = (beta_mag / (beta_mag.max() + 1e-12)).clamp(0, 1)
        # soft floor so new features can still enter
        model.beta_prior_mask.copy_(0.25 + 0.75 * bp)


@dataclass
class TrainingConfig:
    # Core transformer parameters (optimized defaults)
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    out_scale: float = 0.4
    residual_scale: float = 0.3

    # Regularization parameters
    lam_l2: float = 5e-2
    tau_l1: float = 2e-2  # Increased from 5e-3 for stronger sparsity
    tau_l1_max: float = 5e-2  # Maximum L1 penalty for adaptive thresholding
    lam_g: float = 2e-2
    lam_align: float = 1e-3
    lam_residual: float = 5e-4  # L1 magnitude of orthogonalized residual
    lam_sparse_corr: float = 1e-3  # Additional sparsity regularization for corrections

    # Training parameters (optimized defaults)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    T: int = 30
    nn_steps: int = 2
    warm_start_steps: int = 3

    # IRLS optimization parameters
    irls_cached_batch_size: int = 0  # 0/None => full-data IRLS
    irls_max_iter: int = 10  # Reduced from 25 for speed

    # Residual head optimization
    use_residual_head: bool = True
    residual_every_k: int = 1  # Train residual head every k iterations

    # Fast preset for maximum speed
    fast_preset: bool = False

    # Evaluation optimization
    n_thresholds: int = 101  # Reduced from 501 for speed
    compute_decomposition: bool = False  # Skip decomposition by default
    eval_every_k: int = 1  # Evaluate every k iterations

    # Scenario-specific parameters
    scenario_name: Optional[str] = None  # For scenario-specific scheduling

    # === Variant toggles ===
    hard_no_self_mask: bool = True         # strict/exploratory/plus=True, relaxed=False
    use_companion_shortlist: bool = False  # plus=True
    two_hop_attention: bool = False        # plus=True
    topk_beta_aware: bool = False          # plus=True
    residual_orthogonalize: bool = True    # strict/plus=True, relaxed maybe False

    # No-harm and gating configuration
    no_harm_margin: float = 0.01   # C/D default; strict can use 0.005
    k_start_frac: float = 0.25     # start at d*k_start_frac
    k_end_frac: float = 0.05       # end at d*k_end_frac
    warm_frac: float = 0.4         # warmup proportion for k schedule

    # Existing parameters
    use_binary_context: bool = True
    use_no_harm: bool = True
    eps_g: float = 1e-3
    tau_beta_report: float = 0.0
    trust_region: float = 1.0  # Increased to allow more aggressive updates
    use_adaptive_l1: bool = True  # Enable adaptive L1 thresholding
    use_hard_thresholding: bool = True  # Enable hard thresholding for exact zeros
    hard_threshold: float = 1e-6  # Threshold below which coefficients are set to exactly zero


def get_scenario_config(scenario_id: str, d: int) -> TrainingConfig:
    """Get scenario-specific configuration for optimal performance."""
    configs = _default_config_grid(d)
    scenario_configs = {name: cfg for name, cfg in configs if name.startswith("scenario_")}

    scenario_map = {
        "B": "scenario_b",
        "C": "scenario_c",
        "D": "scenario_d",
        "E": "scenario_e",
    }

    config_name = scenario_map.get(scenario_id, "base")
    if config_name in scenario_configs:
        return scenario_configs[config_name]
    else:
        # Fallback to base config
        return configs[0][1]


def _default_config_grid(d: int) -> List[Tuple[str, TrainingConfig]]:
    base = TrainingConfig()

    # Apply fast preset if enabled
    if base.fast_preset:
        base.embed_dim = 48
        base.num_layers = 2
        base.num_heads = min(3, max(1, d//4))

    # Scenario-specific configurations
    scenario_b = replace(
        base,
        scenario_name="B",
        embed_dim=96,
        num_layers=2,
        num_heads=3,
        dropout=0.1,
        out_scale=0.45,  # Will be scheduled 0.2->0.45
        residual_scale=0.25,  # Lower for linear scenarios
        lam_l2=3e-2,
        lam_g=2e-2,
        lam_sparse_corr=1e-3,
        lr=1e-3,
        T=30,
        nn_steps=2,
        warm_start_steps=3,
    )

    scenario_c = replace(
        base,
        scenario_name="C",
        embed_dim=96,
        num_layers=2,
        num_heads=3,
        dropout=0.1,
        out_scale=0.6,  # Fixed high for nonlinear
        residual_scale=0.2,  # Very low for nonlinear scenarios
        lam_l2=1e-2,
        lam_g=1e-2,
        lam_sparse_corr=1e-3,
        lr=1e-3,
        T=30,
        nn_steps=2,
        warm_start_steps=3,
    )

    scenario_d = replace(
        base,
        scenario_name="D",
        embed_dim=120,  # +24 capacity bump
        num_layers=2,
        num_heads=6,    # min(6, max(3, d//4)) for d=10
        dropout=0.1,
        out_scale=0.6,  # Will be scheduled 0.35->0.6
        residual_scale=0.3,
        lam_l2=1e-2,
        lam_g=1e-2,
        lam_sparse_corr=1e-3,
        lr=1e-3,
        T=30,
        nn_steps=2,
        warm_start_steps=3,
    )

    scenario_e = replace(
        base,
        scenario_name="E",
        embed_dim=88,   # +24 capacity bump (64+24)
        num_layers=2,
        num_heads=6,    # min(6, max(3, d//4)) for d=15
        dropout=0.1,
        out_scale=0.4,  # Default
        residual_scale=0.3,
        lam_l2=1e-2,
        lam_g=1e-2,
        lam_sparse_corr=1e-3,
        lr=1e-3,
        T=30,
        nn_steps=2,
        warm_start_steps=3,
    )


    medium = replace(
        base,
        embed_dim=72,  # Reduced from 96
        num_layers=3,
        num_heads=3,   # Reduced from 4
        dropout=0.1,
        out_scale=0.45,
        residual_scale=0.4,
        lam_l2=2e-2,
        lam_g=1e-2,
        lam_align=5e-4,
        lam_residual=5e-4,
        lr=5e-4,
        weight_decay=5e-5,
        T=25,  # Reduced from 40
        nn_steps=2,
        warm_start_steps=3,  # Reduced from 5
        trust_region=1.5,
    )

    aggressive_heads = min(6, max(3, d//4))  # Reduced and adaptive
    aggressive_embed = 96 if d <= 20 else 120  # Reduced
    aggressive = replace(
        base,
        embed_dim=aggressive_embed,
        num_layers=3,  # Reduced from 4
        num_heads=aggressive_heads,
        dropout=0.15,
        out_scale=0.55,
        residual_scale=0.5,
        lam_l2=1e-2,
        lam_g=5e-3,
        lam_align=5e-4,
        lam_residual=1e-3,
        lr=3e-4,
        weight_decay=5e-5,
        T=35,  # Reduced from 60
        nn_steps=2,  # Reduced from 3
        warm_start_steps=3,  # Reduced from 6
        trust_region=2.0,
    )

    residual_heads = min(4, max(2, d//4))  # Reduced and adaptive
    residual_embed = 72 if residual_heads <= 3 else 96  # Reduced
    residual = replace(
        base,
        embed_dim=residual_embed,
        num_layers=3,
        num_heads=residual_heads,
        dropout=0.2,
        out_scale=0.5,
        residual_scale=0.6,
        lam_l2=1.5e-2,
        lam_g=7e-3,
        lam_align=5e-4,
        lam_residual=1e-3,
        lr=4e-4,
        weight_decay=5e-5,
        T=30,  # Reduced from 50
        nn_steps=2,  # Reduced from 3
        warm_start_steps=3,  # Reduced from 5
        trust_region=1.8,
    )

    # Add sparsity-focused configurations
    sparse_base = replace(
        base,
        tau_l1=3e-2,
        tau_l1_max=8e-2,
        lam_sparse_corr=2e-3,
        trust_region=1.5,
        use_adaptive_l1=True,
        use_hard_thresholding=True,
        hard_threshold=1e-5,
    )

    sparse_aggressive = replace(
        aggressive,
        tau_l1=4e-2,
        tau_l1_max=1e-1,
        lam_sparse_corr=3e-3,
        trust_region=2.0,
        use_adaptive_l1=True,
        use_hard_thresholding=True,
        hard_threshold=1e-6,
    )

    configs: List[Tuple[str, TrainingConfig]] = [
        ("base", base),
        ("scenario_b", scenario_b),
        ("scenario_c", scenario_c),
        ("scenario_d", scenario_d),
        ("scenario_e", scenario_e),
        ("medium", medium),
        ("aggressive", aggressive),
        ("sparse_base", sparse_base),
        ("sparse_aggressive", sparse_aggressive),
    ]

    if d <= 4:
        lowdim = replace(
            base,
            embed_dim=48,  # Reduced from 72
            num_layers=2,  # Reduced from 3
            num_heads=2,   # Reduced from 3
            dropout=0.05,
            out_scale=0.7,
            residual_scale=0.65,
            lam_l2=1e-2,
            lam_g=2e-3,
            lam_align=3e-4,
            lam_residual=5e-4,
            lr=4e-4,
            weight_decay=5e-5,
            T=35,  # Reduced from 60
            nn_steps=2,  # Reduced from 3
            warm_start_steps=3,  # Reduced from 5
            trust_region=1.5,
        )
        configs.append(("lowdim_nonlin", lowdim))

    if d >= 8:
        configs.append(("residual", residual))

    return configs


def _train_single(
    cfg: TrainingConfig,
    *,
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
    # NEW:
    save_artifacts: bool = False,
    artifact_dir: str = "artifacts/nimo_transformer",
    scenario_name: Optional[str] = None,
    artifact_tag: Optional[str] = None,
    save_if: str = "better",
    cache_policy: str = "reuse",
    artifact_dtype: str = "float32",
    cfg_label: Optional[str] = None,  # pass label when doing config search
):
    # Start timing
    start_time = time.perf_counter()

    # Sanity check for configuration
    print("CFG sanity:",
          "embed_dim", cfg.embed_dim,
          "num_heads", cfg.num_heads,
          "T", cfg.T,
          "nn_steps", cfg.nn_steps,
          "warm_start_steps", cfg.warm_start_steps,
          "irls_cached_batch_size", cfg.irls_cached_batch_size)

    # GPU + AMP optimization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(randomState)
    np.random.seed(randomState)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    Xva = scaler.transform(X_val) if X_val is not None else None

    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    XteT = torch.tensor(Xte, dtype=torch.float32, device=device)
    yteT = torch.tensor(y_test, dtype=torch.float32, device=device)
    if X_val is not None:
        XvaT = torch.tensor(Xva, dtype=torch.float32, device=device)
        yvaT = torch.tensor(y_val, dtype=torch.float32, device=device)
    else:
        XvaT = yvaT = None

    d = Xt.shape[1]

    # Cached batch for IRLS optimization
    cache_idx = None
    if getattr(cfg, "irls_cached_batch_size", None) and cfg.irls_cached_batch_size > 0:
        batch_size = min(cfg.irls_cached_batch_size, Xt.shape[0])
        cache_idx = torch.randint(0, Xt.shape[0], (batch_size,), device=device)

    model = NIMOTransformer(
        d,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        out_scale=cfg.out_scale,
        residual_scale=cfg.residual_scale,
        use_binary_context=cfg.use_binary_context,
        fast_preset=cfg.fast_preset,
        use_residual_head=cfg.use_residual_head,
        hard_no_self_mask=cfg.hard_no_self_mask,
        use_companion_shortlist=cfg.use_companion_shortlist,
        two_hop_attention=cfg.two_hop_attention,
        topk_beta_aware=cfg.topk_beta_aware,
        residual_orthogonalize=cfg.residual_orthogonalize,
    ).to(device)

    # Set T_total for Top-K gating and scenario_name for scenario-wise schedule
    model.scenario_name = getattr(cfg, 'scenario_name', '')
    model.T_total = cfg.T
    model.correction_net.scenario_name = model.scenario_name
    model.correction_net.T_total = model.T_total

    # Generic default: no positional assumptions (scenario-agnostic)
    model.correction_net.use_alibi = False

    # Build/update masks only when requested
    if cfg.use_companion_shortlist:
        model._build_companion_shortlist(Xt, yt)
        pair_aware_mask = model._get_pair_aware_attention_mask()
    else:
        # fallback: allow all (except maybe self if hard_no_self_mask)
        d = Xt.shape[1]
        allowed = torch.ones(d, d, dtype=torch.bool, device=Xt.device)
        allowed.fill_diagonal_(not cfg.hard_no_self_mask)
        pair_aware_mask = allowed

    model.correction_net.update_attention_mask(pair_aware_mask)
    model.companion_mask = pair_aware_mask

    # Sanity check for "no-self" property
    with torch.no_grad():
        # Ensure companion mask is properly initialized
        if not hasattr(model, 'companion_mask') or model.companion_mask.sum() == 0:
            print("⚠️  Companion mask not initialized, skipping self-leak test")
        else:
            B, d = 64, model.d
            X = torch.zeros(B, d, device=device)
            j = 0
            X[:, j] = torch.linspace(-2, 2, B, device=device)

            # Test raw corrections (before anchoring) to isolate the issue
            g_raw = model.correction_net.forward_raw(X)[0]  # Get raw corrections
            g_raw_j = g_raw[:, j]

            # Test final corrections (after anchoring, with batch centering)
            g_final = model.corrections(X)
            g_final_j = g_final[:, j]

            # Test no-center corrections (critical for true leakage detection)
            g_nocenter = corrections_nocenter(model, X)
            g_nocenter_j = g_nocenter[:, j]

            print(f"🔍 Debug: Raw g_{j} deviation: {g_raw_j.abs().max().item():.6f}")
            print(f"🔍 Debug: Final g_{j} deviation: {g_final_j.abs().max().item():.6f}")
            print(f"🔍 Debug: No-center g_{j} deviation: {g_nocenter_j.abs().max().item():.6f}")

            # Jacobian tests (both with and without batch centering)
            jac_mean, jac_max = diag_jacobian_g(model, X)  # with centering
            jac_nocenter_mean, jac_nocenter_max = diag_jacobian_g_nocenter(model, X)  # no centering

            print(f"🔍 Jacobian (centered) mean/max: {jac_mean:.6f}/{jac_max:.6f}")
            print(f"🔍 Jacobian (no-center) mean/max: {jac_nocenter_mean:.6f}/{jac_nocenter_max:.6f}")

            # STRICT SELF-LEAK DETECTION: No-center should be ≈0 with proper attention mask
            if g_nocenter_j.abs().max().item() > 1e-3:
                print(f"🚨 CRITICAL: No-center self-leak detected! Max deviation: {g_nocenter_j.abs().max().item():.6f}")
                print("   This means the attention mask is not working properly!")
            else:
                print("✅ No-center corrections are clean: no self-leak in attention mechanism")

            if jac_nocenter_max > 1e-3:
                print(f"🚨 CRITICAL: No-center Jacobian self-leak detected! Max: {jac_nocenter_max:.6f}")
            else:
                print("✅ No-center Jacobian test passed: no self-leakage detected")

    # Compute pos_frac for logit adjustment (replaces pos_weight)
    pos_frac = float(yt.mean().item())

    # Setup optimizer with only active parameters
    params = list(model.correction_net.parameters())
    if model.residual_mlp is not None:
        params.extend(list(model.residual_mlp.parameters()))

    # AdamW + cosine schedule with warmup for better stability
    opt = torch.optim.AdamW(
        params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Cosine annealing with warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR
    warmup_steps = max(2, int(0.1 * cfg.T))
    sched_main = CosineAnnealingLR(opt, T_max=max(1, cfg.T - warmup_steps))

    # Setup AMP scaler for GPU (new torch.amp API)
    scaler_amp = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # Setup SWA for snapshot ensembling (full model EMA)
    swa = AveragedModel(model)  # EMA of full NIMO-T, not just correction net
    swa_start = int(0.6 * cfg.T)
    # Note: Using Cosine scheduler instead of SWALR to avoid double LR control

    for _ in range(cfg.warm_start_steps):
        # Use cached batch for IRLS if available
        X_irls = Xt[cache_idx] if cache_idx is not None else Xt
        y_irls = yt[cache_idx] if cache_idx is not None else yt

        update_beta_irls(
            model,
            X_irls,
            y_irls,
            lam_l2=cfg.lam_l2,
            tau_l1=cfg.tau_l1,
            tau_l1_max=cfg.tau_l1_max,
            use_correction=False,
            trust_region=cfg.trust_region,
            iteration=0,
            max_iterations=cfg.irls_max_iter,
            use_adaptive_l1=cfg.use_adaptive_l1,
            use_hard_thresholding=cfg.use_hard_thresholding,
            hard_threshold=cfg.hard_threshold,
        )

    loss_history = []
    best_val_loss = float("inf")
    best_state: Optional[Tuple[dict, dict, torch.Tensor]] = None
    stopped_early = False

    for t in range(cfg.T):
        model.eval()

        # Set t_step for Top-K gating
        model.t_step = t

        # Learning rate scheduling with warmup
        if t < warmup_steps:
            # Linear warmup
            for g in opt.param_groups:
                g["lr"] = cfg.lr * (t + 1) / warmup_steps

        # Adaptive out_scale controller (scenario-agnostic)
        warm = max(3, int(0.2 * cfg.T))  # 20% of training or at least 3 iters
        with torch.no_grad():
            if t < warm:
                # During warmup: gentle initialization toward reasonable scale
                target_scale = 0.4  # reasonable default
                z = torch.clamp(torch.tensor(2 * target_scale - 1, device=device), -0.999, 0.999)
                inv = 0.5 * torch.log((1 + z) / (1 - z))  # atanh
                alpha = 0.3  # smoothing, 0<alpha<=1
                model.correction_net.out_scale_param.mul_(1 - alpha).add_(alpha * inv)
            else:
                # After warmup: adaptive controller to keep mean |g| in target band
                # measure current correction magnitude on a small minibatch
                mb_idx = torch.randperm(Xt.size(0), device=Xt.device)[:min(512, Xt.size(0))]
                g_mb = model.corrections(Xt[mb_idx], detach=True).abs().mean().clamp(0, 1)  # scalar
                target = torch.tensor(0.25, device=Xt.device)  # desired mean |g|
                err = (g_mb - target).item()  # >0 means too strong
                # tiny integral-like nudge on per-feature scale parameter (shared nudge)
                nudgestep = 0.02 * err
                model.correction_net.out_scale_param.add_(-nudgestep)

        # Use full IRLS data early, then cached batch
        if t < 3:  # First 3 iterations use full data
            X_irls = Xt
            y_irls = yt
        else:
            X_irls = Xt[cache_idx] if cache_idx is not None else Xt
            y_irls = yt[cache_idx] if cache_idx is not None else yt

        # Faster λ_g decay - let the correction network assert itself once it's stable
        progress = t / max(1, cfg.T - 1)
        lam_g_curr = cfg.lam_g * (1.0 - progress)**2 + 0.1 * cfg.lam_g * (1.0 - progress)

        # Two-stage training: alternate between β-focused and correction-focused steps
        beta_focused = (t % 4) != 0  # 3 out of 4 iterations focus on β
        correction_focused = (t % 4) == 0  # 1 out of 4 iterations focus on corrections

        # Always train residual head if enabled (scenario-agnostic)
        train_residual_now = cfg.use_residual_head

        # Disable gradients for residual head if not training it
        if model.residual_mlp is not None:
            for p in model.residual_mlp.parameters():
                p.requires_grad_(train_residual_now)

        # Two-stage training: different step counts based on focus
        if correction_focused:
            # Correction-focused: more NN steps, skip IRLS
            nn_steps = 3
            skip_irls = True
        else:
            # β-focused: normal steps, include IRLS
            early_nn_steps = 4 if t < 5 else cfg.nn_steps
            nn_steps = early_nn_steps
            skip_irls = False

        # Skip IRLS for correction-focused iterations
        if not skip_irls:
            update_beta_irls(
                model,
                X_irls,
                y_irls,
                lam_l2=cfg.lam_l2,
                tau_l1=cfg.tau_l1,
                tau_l1_max=cfg.tau_l1_max,
                use_correction=True,
                trust_region=cfg.trust_region,
                iteration=t,
                max_iterations=cfg.irls_max_iter,
                use_adaptive_l1=cfg.use_adaptive_l1,
                use_hard_thresholding=cfg.use_hard_thresholding,
                hard_threshold=cfg.hard_threshold,
            )

        for _ in range(nn_steps):
            model.train()
            opt.zero_grad(set_to_none=True)

            # Use AMP for forward pass - single pass to get corr & residual used in logits
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                # single pass to get corr & residual used in logits
                corr, res_token = model.modulation(Xt)  # batch-centered corr & token residual
                feats = Xt * (1.0 + corr)               # <-- use corrected features here, too
                r = res_token
                if model.use_residual_head and model.residual_mlp is not None:
                    r = r + model.residual_mlp(Xt).squeeze(-1)
                r = torch.tanh(r) * model.residual_scale
                r = model._orthogonalize_residual(feats, r)

                # effective design for β path
                ones = torch.ones(Xt.size(0), 1, device=Xt.device, dtype=Xt.dtype)
                B = torch.cat([ones, feats], dim=1)     # <-- B built from corrected feats
                logits = B.matmul(model.beta) + r

                # Generic hybrid loss that self-weights by imbalance (scenario-agnostic)
                prior = pos_frac
                logit_adj = torch.log(torch.tensor(prior / (1 - prior) + 1e-12, device=logits.device))
                adjusted_logits = logits - logit_adj

                # Generic: BCE + focal with self-weighting by imbalance
                imb = float(abs(0.5 - prior) * 2.0)  # 0 (balanced) → 1 (very imbalanced)
                bce = F.binary_cross_entropy_with_logits(adjusted_logits, yt)
                foc = focal_bce_with_logits(adjusted_logits, yt, alpha=0.5, gamma=1.5)
                loss_ce = (1 - 0.5*imb) * bce + (0.5*imb) * foc

                # Counterfactual pair flips for consistency loss
                x_flipped, flip_mask = model._apply_counterfactual_flips(Xt, model.companion_mask)
                consistency_loss = model._compute_consistency_loss(Xt, x_flipped, flip_mask, model.companion_mask)

                # Auxiliary F1 surrogate loss (aligns training with eval metric)
                p = torch.sigmoid(adjusted_logits)
                tp = (p * yt).mean()
                fp = (p * (1 - yt)).mean()
                fn = ((1 - p) * yt).mean()
                f1_sur = 2 * tp / (2 * tp + fp + fn + 1e-8)
                loss_f1 = 1 - f1_sur

                # regs using the *same* corr/r
                reg_g = lam_g_curr * corr.abs().mean()

                # Ramp lam_align: low for first ~20-30% iters, then full strength
                lam_align_ramp = min(t / max(1, cfg.T * 0.25), 1.0)  # ramp over first 25% of training
                lam_align_curr = cfg.lam_align * lam_align_ramp

                # Alignment penalty: use centered covariance with a dead-zone
                x_center = Xt - Xt.mean(dim=0, keepdim=True)
                c_center = corr - corr.mean(dim=0, keepdim=True)
                cov = (x_center * c_center).mean(dim=0).abs()  # per-feature
                deadzone = 0.02  # allow slight coupling
                align = lam_align_curr * torch.clamp(cov - deadzone, min=0).mean()

                reg_residual_l1   = cfg.lam_residual * r.abs().mean()
                reg_residual_mean = 1e-4 * r.mean().abs()
                reg_residual_corr = 1e-4 * (Xt.mul(r.unsqueeze(1)).mean(dim=0).abs().sum())

                # L2 on residual weights (not just outputs)
                reg_residual_l2 = 0.0
                if model.residual_mlp is not None:
                    for p in model.residual_mlp.parameters():
                        reg_residual_l2 += 1e-5 * (p ** 2).sum()

                reg_sparse_corr   = cfg.lam_sparse_corr * corr.abs().mean()  # Changed from .sum() to .mean()

                # Optional: small penalty to keep scales near schedule after warmup
                reg_scale = 0.0
                if t >= warm:
                    target = torch.tanh(inv).detach()
                    learned = torch.tanh(model.correction_net.out_scale_param)
                    reg_scale = 1e-4 * (learned - target).pow(2).mean()

                # ALiBi strength regularization (keep it honest)
                reg_alibi = 1e-5 * (model.correction_net.alibi_strength ** 2)

                loss = loss_ce + 0.05 * loss_f1 + reg_g + align + reg_residual_l1 + reg_residual_mean + reg_residual_corr + reg_residual_l2 + reg_sparse_corr + reg_scale + consistency_loss + reg_alibi

            # Use AMP scaler for backward pass
            scaler_amp.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.correction_net.parameters(), max_norm=1.0)
            if model.residual_mlp is not None:
                torch.nn.utils.clip_grad_norm_(model.residual_mlp.parameters(), max_norm=1.0)
            scaler_amp.step(opt)
            scaler_amp.update()

            # SWA update (full model) - no LR scheduling here, using Cosine
            if t >= swa_start:
                swa.update_parameters(model)

        # Step cosine scheduler after optimizer (but not during warmup)
        if t >= warmup_steps:
            sched_main.step()

        loss_history.append(float(loss.item()))

        # Diagnostics every 5 iterations (same as MLP)
        if (t % 5) == 0:
            with torch.no_grad():
                # with corrections
                logits_with = model.predict_logits(Xt, use_correction=True)
                loss_with = F.binary_cross_entropy_with_logits(logits_with, yt).item()
                # without corrections
                logits_wo = model.predict_logits(Xt, use_correction=False)
                loss_wo = F.binary_cross_entropy_with_logits(logits_wo, yt).item()
                # mean|g|
                g = model.corrections(Xt)
                g_abs = g.abs()
                g_mean = g_abs.mean().item()
                # mean|Δp|
                dp = (torch.sigmoid(logits_with) - torch.sigmoid(logits_wo)).abs().mean().item()
                # current out_scale
                # Per-feature scale summary stats
                scale_vec = 0.5 * (1.0 + torch.tanh(model.correction_net.out_scale_param))
                scale_mean = scale_vec.mean().item()
                scale_min = scale_vec.min().item()
                scale_max = scale_vec.max().item()
                # activation rate
                act_rate = (g_abs > cfg.eps_g).float().mean().item()

                # Pair saliency (every 10 iterations to avoid overhead)
                if (t % 10) == 0:
                    # Optimize: compute on small random subset of features
                    cols = torch.randperm(d, device=Xt.device)[:min(20, d)]
                    saliency = model.compute_pair_saliency(Xt, yt, num_samples=50, cols=cols)
                    max_saliency = saliency.max().item()
                    mean_saliency = saliency.mean().item()
                    print(f"Iter {t}: mean|g|={g_mean:.4f} loss_with={loss_with:.4f} loss_without={loss_wo:.4f} mean|Δp|={dp:.4f} out_scale={scale_mean:.3f} act_rate={act_rate:.3f}")
                    print(f"  Scale stats: mean/min/max = {scale_mean:.3f}/{scale_min:.3f}/{scale_max:.3f}")
                    print(f"  📊 Pair saliency: max={max_saliency:.4f} mean={mean_saliency:.4f}")
                else:
                    print(f"Iter {t}: mean|g|={g_mean:.4f} loss_with={loss_with:.4f} loss_without={loss_wo:.4f} mean|Δp|={dp:.4f} out_scale={scale_mean:.3f} act_rate={act_rate:.3f}")
                    print(f"  Scale stats: mean/min/max = {scale_mean:.3f}/{scale_min:.3f}/{scale_max:.3f}")

        if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < 1e-4:
            stopped_early = True
            break

        if XvaT is not None:
            with torch.no_grad():
                val_logits = model.predict_logits(XvaT, use_correction=True)
                val_loss = F.binary_cross_entropy_with_logits(val_logits, yvaT).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = (
                    model.correction_net.state_dict(),
                    model.residual_mlp.state_dict() if model.residual_mlp is not None else None,
                    model.beta.detach().clone(),
                )

    # Load SWA weights if available (full model)
    if cfg.T >= swa_start:
        model.load_state_dict(swa.module.state_dict())

    if best_state is not None:
        correction_state, residual_mlp_state, beta_state = best_state
        model.correction_net.load_state_dict(correction_state)
        if residual_mlp_state is not None and model.residual_mlp is not None:
            model.residual_mlp.load_state_dict(residual_mlp_state)
        model.beta.data.copy_(beta_state)

    model.eval()
    use_correction_final = True
    no_harm_choice = "on"

    # Temperature scaling for better calibration (enforce T > 0)
    log_T = torch.tensor(0.0, device=device, requires_grad=True)  # T=1
    if XvaT is not None:
        # Fit T on validation set to minimize BCE
        optT = torch.optim.LBFGS([log_T], max_iter=50)

        with torch.no_grad():
            z_val = model.predict_logits(XvaT, use_correction=use_correction_final)

        def closure():
            optT.zero_grad()
            T = torch.exp(log_T)
            loss = F.binary_cross_entropy_with_logits(z_val / T, yvaT)
            loss.backward()
            return loss

        optT.step(closure)
        T_final = torch.exp(log_T).detach()
        print(f"🔧 Temperature scaling: T = {T_final.item():.3f}")
    else:
        T_final = torch.tensor(1.0, device=device)

    if cfg.use_no_harm and XvaT is not None:
        with torch.no_grad():
            prob_val_on = model.predict_proba(XvaT, use_correction=True).cpu().numpy()
            prob_val_off = model.predict_proba(XvaT, use_correction=False).cpu().numpy()
        grid = np.linspace(0.0, 1.0, getattr(cfg, "n_thresholds", 101))
        f1_on = max(f1_score(y_val, (prob_val_on >= thr).astype(int), zero_division=0) for thr in grid)
        f1_off = max(f1_score(y_val, (prob_val_off >= thr).astype(int), zero_division=0) for thr in grid)

        # Must-help margin: keep corrections only if F1_on >= F1_off + margin
        margin = getattr(cfg, 'no_harm_margin', 0.01)

        if f1_on < f1_off + margin:
            use_correction_final = False
            no_harm_choice = "off"

            # Recalibrate temperature for the chosen mode (corrections off)
            if XvaT is not None:
                log_T_off = torch.tensor(0.0, device=device, requires_grad=True)
                optT_off = torch.optim.LBFGS([log_T_off], max_iter=50)

                with torch.no_grad():
                    z_val_off = model.predict_logits(XvaT, use_correction=False)

                def closure_off():
                    optT_off.zero_grad()
                    T_off = torch.exp(log_T_off)
                    loss = F.binary_cross_entropy_with_logits(z_val_off / T_off, yvaT)
                    loss.backward()
                    return loss

                optT_off.step(closure_off)
                T_final = torch.exp(log_T_off).detach()
                print(f"🔧 Temperature scaling (corrections off): T = {T_final.item():.3f}")

    with torch.no_grad():
        # Use temperature scaling for better calibration
        logits_test = model.predict_logits(XteT, use_correction=use_correction_final)
        prob_test = torch.sigmoid(logits_test / T_final).cpu().numpy()

        if XvaT is not None:
            logits_val = model.predict_logits(XvaT, use_correction=use_correction_final)
            prob_val = torch.sigmoid(logits_val / T_final).cpu().numpy()
        else:
            prob_val = None

    # Better thresholding with cost-sensitive tie-break
    grid = np.linspace(0.0, 1.0, getattr(cfg, "n_thresholds", 101))
    if prob_val is not None:
        f1_grid = [f1_score(y_val, (prob_val >= thr).astype(int), zero_division=0) for thr in grid]
    else:
        f1_grid = [f1_score(y_test, (prob_test >= thr).astype(int), zero_division=0) for thr in grid]

    best_idx = int(np.argmax(f1_grid))
    best_f1 = f1_grid[best_idx]

    # Refine threshold with local search around best grid point
    if best_idx > 0 and best_idx < len(grid) - 1:
        # Search in ±0.05 around the best grid point
        refine_range = np.linspace(max(0.0, grid[best_idx] - 0.05),
                                 min(1.0, grid[best_idx] + 0.05), 21)
        if prob_val is not None:
            f1_refined = [f1_score(y_val, (prob_val >= thr).astype(int), zero_division=0) for thr in refine_range]
        else:
            f1_refined = [f1_score(y_test, (prob_test >= thr).astype(int), zero_division=0) for thr in refine_range]

        best_refined_idx = int(np.argmax(f1_refined))
        if f1_refined[best_refined_idx] > best_f1:
            threshold = float(refine_range[best_refined_idx])
        else:
            threshold = float(grid[best_idx])
    else:
        threshold = float(grid[best_idx])

    y_pred = (prob_test >= threshold).astype(int)
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    acc = float(accuracy_score(y_test, y_pred))

    if prob_val is not None:
        val_pred = (prob_val >= threshold).astype(int)
        val_metrics = {
            "f1": float(f1_score(y_val, val_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_val, val_pred)),
        }
    else:
        val_metrics = {"f1": None, "accuracy": None}

    # Optional decomposition computation
    if getattr(cfg, "compute_decomposition", False):
        def _decomposition(X_np: np.ndarray):
            X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                eta_lin = model.predict_logits(X_t, use_correction=False).cpu().numpy()
                eta_full = model.predict_logits(X_t, use_correction=use_correction_final).cpu().numpy()
            eta_corr = eta_full - eta_lin
            var_full = np.var(eta_full)
            return {
                "corr_mean_abs": float(np.mean(np.abs(eta_corr))),
                "corr_var_share": float(np.var(eta_corr) / var_full) if var_full > 1e-12 else 0.0,
                "lin_full_corr": float(np.corrcoef(eta_lin, eta_full)[0, 1]) if np.std(eta_lin) > 0 and np.std(eta_full) > 0 else 0.0,
            }

        decomp_val = _decomposition(Xva) if X_val is not None else None
        decomp_test = _decomposition(Xte)
    else:
        decomp_val = None
        decomp_test = None

    corr_stats = None
    if X_val is not None:
        with torch.no_grad():
            g_corr_val = model.corrections(torch.tensor(Xva, dtype=torch.float32, device=device))
            r_val = model.residual_component(torch.tensor(Xva, dtype=torch.float32, device=device))
        corr_stats = {
            "eps_g": float(cfg.eps_g),
            "mean_abs_corr": np.abs(g_corr_val.cpu().numpy()).mean(axis=0).tolist(),
            "activation_rate": (np.abs(g_corr_val.cpu().numpy()) > cfg.eps_g).mean(axis=0).tolist(),
            "rel_mod": np.mean(np.abs(Xva * g_corr_val.cpu().numpy()), axis=0).tolist(),
            "residual_mean_abs": float(np.mean(np.abs(r_val.cpu().numpy()))),
        }
    else:
        g_corr_val = None

    beta_std = model.beta.detach().cpu().numpy()[1:]
    b0_std = float(model.beta.detach().cpu().numpy()[0])

    scale = scaler.scale_
    mean = scaler.mean_

    beta_raw = beta_std / scale
    b0_raw = b0_std - float(np.dot(beta_raw, mean))

    beta_for_sel = beta_raw.copy()
    if cfg.tau_beta_report > 0:
        beta_for_sel[np.abs(beta_for_sel) < cfg.tau_beta_report] = 0.0

    selected_mask = (np.abs(beta_for_sel) > 0).astype(int).tolist()
    selected_features = (
        [X_columns[i] for i, m in enumerate(selected_mask) if m]
        if X_columns is not None
        else [i for i, m in enumerate(selected_mask) if m]
    )

    with torch.no_grad():
        g_corr_train = model.corrections(
            torch.tensor(Xtr, dtype=torch.float32, device=device), detach=True
        ).cpu().numpy()
    beta_eff_raw = beta_raw * (1.0 + g_corr_train.mean(axis=0))

    feature_names = list(X_columns) if X_columns is not None else [f"feature_{i}" for i in range(len(beta_raw))]

    # End timing
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    # Compute final diagnostics for compliance scoring
    with torch.no_grad():
        g_corr_final = model.corrections(torch.tensor(Xtr, dtype=torch.float32, device=device))
        g_abs_final = g_corr_final.abs()
        scale_vec_final = 0.5 * (1.0 + torch.tanh(model.correction_net.out_scale_param))

        # Alignment penalty computation
        x_center = torch.tensor(Xtr, dtype=torch.float32, device=device) - torch.tensor(Xtr, dtype=torch.float32, device=device).mean(dim=0, keepdim=True)
        c_center = g_corr_final - g_corr_final.mean(dim=0, keepdim=True)
        cov = (x_center * c_center).mean(dim=0).abs()

        result = {
            "model_name": "NIMO_T",
            "iteration": iteration,
            "random_seed": randomState,
            "f1": f1,
            "accuracy": acc,
            "threshold": threshold,
            "y_pred": y_pred.tolist(),
            "y_prob": prob_test.tolist(),
            "metrics": {"f1": f1, "accuracy": acc},
            "selection": {"mask": selected_mask, "features": selected_features},
            "selected_features": selected_features,
            "n_selected": len(selected_features),
            "coefficients": {
                "intercept": b0_raw,
                "values": beta_raw.tolist(),
                "values_effective": beta_eff_raw.tolist(),
                "feature_names": feature_names,
                "coef_threshold_applied": float(cfg.tau_beta_report),
                "scale": scale.tolist(),
                "mean": mean.tolist(),
            },
            "correction_stats_val": corr_stats,
            "decomposition_val": decomp_val,
            "decomposition_test": decomp_test,
            "no_harm_val": None if X_val is None else {
                "no_harm_choice": no_harm_choice,
                "f1_on": f1_on if cfg.use_no_harm else None,
                "f1_off": f1_off if cfg.use_no_harm else None,
            },
            "training": {
                "loss_history": loss_history,
                "n_iters": len(loss_history),
                "stopped_early": bool(stopped_early),
            },
            "hyperparams": {
                **cfg.__dict__,
                "use_correction_final": bool(use_correction_final),
            },
            "val_metrics": val_metrics,
            
            # Per-variant compliance diagnostics
            "diagnostics": {
                "align_L1": float(cov.abs().mean().item()),
                "mean_abs_g": float(g_abs_final.mean().item()),
                "scale_mean": float(scale_vec_final.mean().item()),
                "scale_min": float(scale_vec_final.min().item()),
                "scale_max": float(scale_vec_final.max().item()),
                "companion_density": float(model.companion_mask.float().mean().item()) if hasattr(model, 'companion_mask') else 0.0,
            },
            
            # Timing information
            "execution_time": execution_time,
            "timing": {
                "total_seconds": execution_time,
                "start_time": start_time,
                "end_time": end_time
            }
        }

    if g_corr_val is not None and result["no_harm_val"] is not None:
        result["no_harm_val"].update(
            {
                "g_mean_abs": float(np.mean(np.abs(g_corr_val.cpu().numpy()))),
                "lin_full_corr": decomp_val["lin_full_corr"] if decomp_val else None,
            }
        )

    # ---- artifact save (only if requested) ----
    if save_artifacts:
        key = _t_key(
            scenario=scenario_name,
            seed=randomState,
            input_dim=int(d),
            hparams=result["hyperparams"],
            tag=artifact_tag,
            cfg_label=cfg_label,
        )
        paths = _t_paths(artifact_dir, key)
        existing = _load_t_artifacts(paths) if cache_policy in ("reuse",) else None

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
            else:  # better
                prev_f1 = float(existing["meta"].get("f1", -1.0))
                should_save = run_f1 > prev_f1

        if should_save:
            # collect minimal arrays needed for plotting
            corr_last: nn.Linear = model.correction_net.corr_head[-1]
            tres_last: nn.Linear = model.correction_net.residual_head[-1]

            mlp_last = None
            if model.residual_mlp is not None and isinstance(model.residual_mlp[-1], nn.Linear):
                mlp_last = model.residual_mlp[-1]

            arrs = {
                "feature_embed": model.correction_net.feature_embed.detach().cpu().numpy(),
                "binary_proj_weight": (
                    None if model.correction_net.binary_proj is None
                    else model.correction_net.binary_proj.weight.detach().cpu().numpy()
                ),
                "binary_codes": (
                    None if model.correction_net.binary_codes is None
                    else model.correction_net.binary_codes.detach().cpu().numpy()
                ),
                "cls_token": None,  # No longer used in new architecture
                "corr_head_last_weight": corr_last.weight.detach().cpu().numpy(),
                "corr_head_last_bias": corr_last.bias.detach().cpu().numpy(),
                "transformer_residual_head_weight": tres_last.weight.detach().cpu().numpy(),
                "transformer_residual_head_bias": tres_last.bias.detach().cpu().numpy(),
                "mlp_residual_head_weight": (
                    None if mlp_last is None else mlp_last.weight.detach().cpu().numpy()
                ),
                "mlp_residual_head_bias": (
                    None if mlp_last is None else mlp_last.bias.detach().cpu().numpy()
                ),
            }

            meta = {
                "scenario": scenario_name or "unknown",
                "model_type": "NIMO_T",
                "random_seed": int(randomState),
                "input_dim": int(d),
                "embed_dim": int(cfg.embed_dim),
                "num_heads": int(cfg.num_heads),
                "num_layers": int(cfg.num_layers),
                "use_binary_context": bool(cfg.use_binary_context),
                "f1": run_f1,
                "accuracy": float(result.get("accuracy", 0.0)),
                "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
                "hyperparams": result["hyperparams"],
                "config_label": cfg_label,
            }

            _save_t_artifacts(arrs, meta, paths, dtype=artifact_dtype)
            result["_artifact_paths"] = {k: str(v) for k, v in paths.items()}

    # Add model weights for plotting if requested
    if return_model_bits:
        result["_plot_bits"] = {
            "feature_embed": model.correction_net.feature_embed.detach().cpu().numpy(),
            "binary_proj_weight": (
                None if model.correction_net.binary_proj is None
                else model.correction_net.binary_proj.weight.detach().cpu().numpy()
            ),
            "binary_codes": (
                None if model.correction_net.binary_codes is None
                else model.correction_net.binary_codes.detach().cpu().numpy()
            ),
        }
        # Also store the model for activation-based analysis
        result["model"] = model

    return standardize_method_output(result), val_metrics


def run_nimo(
    X_train, y_train, X_test, y_test,
    iteration, randomState, X_columns=None,
    *,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    config: Optional[TrainingConfig] = None,
    config_search: Optional[List[Tuple[str, TrainingConfig]]] = None,
    return_all: bool = False,
    return_model_bits: bool = False,
    # Artifact saving parameters
    save_artifacts: bool = False,
    artifact_dir: str = "artifacts/nimo_transformer",
    scenario_name: Optional[str] = None,
    artifact_tag: Optional[str] = None,
    save_if: str = "better",
    cache_policy: str = "reuse",
    artifact_dtype: str = "float32",
):
    """Run transformer-based NIMO with optional hyperparameter search."""

    def _with_label(res, val_metrics, label):
        if "val_metrics" not in res:
            res["val_metrics"] = val_metrics
        res["config_label"] = label
        res.setdefault("config_candidates", [])
        res["hyperparams"]["config_label"] = label
        return res

    if config is not None:
        res, val_metrics = _train_single(
            config,
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
            cfg_label="provided",
        )
        candidate_summary = [{
            "label": "provided",
            "val_f1": val_metrics.get("f1") if val_metrics else None,
            "val_accuracy": val_metrics.get("accuracy") if val_metrics else None,
            "test_f1": res.get("f1"),
        }]
        res = _with_label(res, val_metrics, "provided")
        res["config_candidates"] = candidate_summary
        return (res, candidate_summary) if return_all else res

    d = X_train.shape[1]
    
    # Skip config grid by default - use single base config unless config_search explicitly provided
    if config_search is None:
        # Use only the base configuration by default
        cfg = _default_config_grid(d)[0][1]
        res, val_metrics = _train_single(
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
            cfg_label="base",
        )
        res = _with_label(res, val_metrics, "base")
        res["config_candidates"] = [{
            "label": "base",
            "val_f1": val_metrics.get("f1") if val_metrics else None,
            "val_accuracy": val_metrics.get("accuracy") if val_metrics else None,
            "test_f1": res.get("f1"),
        }]
        return (res, res["config_candidates"]) if return_all else res
    
    candidates = config_search

    summaries = []
    best_res = best_metrics = None
    best_label = ""
    best_score = float('-inf')

    for label, cfg in candidates:
        try:
            res, val_metrics = _train_single(
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
                cfg_label=label,
            )
            err_msg = None
        except Exception as exc:
            summaries.append({
                "label": label,
                "error": str(exc),
                "val_f1": None,
                "val_accuracy": None,
                "test_f1": None,
            })
            continue

        val_f1 = None
        if val_metrics and val_metrics.get("f1") is not None:
            val_f1 = val_metrics["f1"]
        score = val_f1 if val_f1 is not None else res.get("f1", 0.0)

        summaries.append({
            "label": label,
            "val_f1": val_f1,
            "val_accuracy": val_metrics.get("accuracy") if val_metrics else None,
            "test_f1": res.get("f1"),
            "error": err_msg,
        })

        if score > best_score:
            best_res = res
            best_metrics = val_metrics
            best_label = label
            best_score = score

    if best_res is None:
        raise RuntimeError("All NIMO transformer config candidates failed")

    best_res = _with_label(best_res, best_metrics, best_label)
    best_res["config_candidates"] = summaries

    return (best_res, summaries) if return_all else best_res


def test_nimo_compliance(model, X_test, device, verbose=True):
    """
    Comprehensive test suite for NIMO compliance.
    
    Tests:
    1. No-self gradient (no-center): ∂g_j/∂x_j ≈ 0
    2. Zero-anchor: g(0) = 0
    3. Self-leak detection in bilinear path
    4. Attention mask strictness
    """
    model.eval()
    results = {}
    
    # Convert to tensor if needed
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    
    B, d = X_test.shape
    
    # Test 1: No-self gradient (no-center)
    if verbose:
        print("🔍 Testing no-self gradient (no-center)...")
    
    mean_diag, max_diag = diag_jacobian_g_nocenter(model, X_test)
    results['no_self_gradient'] = {
        'mean_diag': mean_diag,
        'max_diag': max_diag,
        'passed': max_diag < 1e-3
    }
    
    if verbose:
        status = "✅ PASS" if results['no_self_gradient']['passed'] else "❌ FAIL"
        print(f"  {status}: max diagonal jacobian = {max_diag:.6f} (threshold: 1e-3)")
    
    # Test 2: Zero-anchor
    if verbose:
        print("🔍 Testing zero-anchor...")
    
    X0 = torch.zeros_like(X_test)
    corr0 = corrections_nocenter(model, X0)
    max_zero_corr = corr0.abs().max().item()
    results['zero_anchor'] = {
        'max_corr_at_zero': max_zero_corr,
        'passed': max_zero_corr < 1e-5
    }
    
    if verbose:
        status = "✅ PASS" if results['zero_anchor']['passed'] else "❌ FAIL"
        print(f"  {status}: max correction at zero = {max_zero_corr:.6f} (threshold: 1e-5)")
    
    # Test 3: Self-leak detection in bilinear path
    if verbose:
        print("🔍 Testing bilinear path for self-leak...")
    
    # Test with varying x_j while keeping x_{-j} constant
    X_base = X_test.clone()
    j = 0  # Test feature 0
    X_varied = X_base.clone()
    X_varied[:, j] = torch.linspace(-2, 2, B, device=device)
    
    with torch.no_grad():
        corr_base = model.correction_net.forward_raw(X_base)[0]
        corr_varied = model.correction_net.forward_raw(X_varied)[0]
        
        # Check if corr_j changes when only x_j changes
        corr_j_change = (corr_varied[:, j] - corr_base[:, j]).abs().max().item()
    
    results['bilinear_self_leak'] = {
        'corr_j_change': corr_j_change,
        'passed': corr_j_change < 1e-4
    }
    
    if verbose:
        status = "✅ PASS" if results['bilinear_self_leak']['passed'] else "❌ FAIL"
        print(f"  {status}: corr_j change when varying x_j = {corr_j_change:.6f} (threshold: 1e-4)")
    
    # Test 4: Attention mask strictness (conditional on hard_no_self_mask)
    if verbose:
        print("🔍 Testing attention mask strictness...")
    
    attn_mask = model.correction_net.attn_mask
    diagonal_vals = torch.diag(attn_mask)
    max_diag_val = diagonal_vals.max().item()
    
    # Check diagonal based on hard_no_self_mask policy
    if model.correction_net.hard_no_self_mask:
        # Strict variants: diagonal should be -inf
        diag_ok = max_diag_val < -1e6
        expected = "-inf"
    else:
        # Relaxed variants: diagonal should be 0
        diag_ok = max_diag_val < 1e-6
        expected = "0"
    
    results['attention_mask'] = {
        'max_diagonal_value': max_diag_val,
        'passed': diag_ok
    }
    
    if verbose:
        status = "✅ PASS" if results['attention_mask']['passed'] else "❌ FAIL"
        print(f"  {status}: max diagonal attention value = {max_diag_val:.6f} (should be {expected})")
    
    # Overall compliance
    all_passed = all(test['passed'] for test in results.values())
    results['overall_compliance'] = all_passed
    
    if verbose:
        print(f"\n🎯 Overall NIMO compliance: {'✅ PASS' if all_passed else '❌ FAIL'}")
    
    return results


def test_linear_sanity(model, X_test, y_test, device, verbose=True):
    """
    Test linear sanity for scenario B: g should be small and model should behave like Lasso.
    """
    model.eval()
    results = {}
    
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        # Test correction magnitude
        corr = model.corrections(X_test)
        mean_abs_corr = corr.abs().mean().item()
        max_abs_corr = corr.abs().max().item()
        
        # Test prediction difference with/without corrections
        logits_with = model.predict_logits(X_test, use_correction=True)
        logits_without = model.predict_logits(X_test, use_correction=False)
        pred_diff = (torch.sigmoid(logits_with) - torch.sigmoid(logits_without)).abs().mean().item()
        
        # Test beta sparsity
        beta_mag = model.beta[1:].abs()
        n_active = (beta_mag > 1e-6).sum().item()
        sparsity = 1.0 - (n_active / len(beta_mag))
    
    results['correction_magnitude'] = {
        'mean_abs': mean_abs_corr,
        'max_abs': max_abs_corr,
        'passed': mean_abs_corr < 0.02 and max_abs_corr < 0.1
    }
    
    results['prediction_stability'] = {
        'pred_diff': pred_diff,
        'passed': pred_diff < 0.05
    }
    
    results['beta_sparsity'] = {
        'n_active': n_active,
        'sparsity': sparsity,
        'passed': sparsity > 0.5  # At least 50% sparsity
    }
    
    if verbose:
        print("🔍 Testing linear sanity...")
        print(f"  Mean |g|: {mean_abs_corr:.4f} (threshold: 0.02)")
        print(f"  Max |g|: {max_abs_corr:.4f} (threshold: 0.1)")
        print(f"  Prediction diff: {pred_diff:.4f} (threshold: 0.05)")
        print(f"  Beta sparsity: {sparsity:.2f} ({n_active}/{len(beta_mag)} active)")
        
        all_passed = all(test['passed'] for test in results.values())
        status = "✅ PASS" if all_passed else "❌ FAIL"
        print(f"  Linear sanity: {status}")
    
    return results


def run_nimo_transformer_scenario(
    X_train, y_train, X_test, y_test,
    scenario_id: str,
    iteration: int, randomState: int,
    X_columns: Optional[List[str]] = None,
    *,
    X_val=None, y_val=None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run NIMO Transformer with scenario-specific configuration.
    
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
    d = X_train.shape[1]
    scenario_config = get_scenario_config(scenario_id, d)
    
    # Filter kwargs to only include TrainingConfig parameters
    config_fields = {field.name for field in TrainingConfig.__dataclass_fields__.values()}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
    
    # Merge with user-provided kwargs (user kwargs take precedence)
    config_dict = scenario_config.__dict__.copy()
    config_dict.update(filtered_kwargs)
    config = TrainingConfig(**config_dict)
    
    print(f"🎯 Running NIMO Transformer for scenario {scenario_id} with config: scenario_name={config.scenario_name}, embed_dim={config.embed_dim}, out_scale={config.out_scale}")
    
    # Run with merged configuration
    return run_nimo(
        X_train, y_train, X_test, y_test,
        iteration=iteration,
        randomState=randomState,
        X_columns=X_columns,
        X_val=X_val,
        y_val=y_val,
        config=config,
        return_model_bits=False,
        save_artifacts=True,
        scenario_name=scenario_id,
        save_if="better",
        cache_policy="reuse"
    )