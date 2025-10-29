#!/usr/bin/env python3
"""
Clean Synthetic Data Generator - NIMO-inspired
Generates simple, focused synthetic datasets for LASSO vs NIMO comparison.
Uses fixed big test sets and stratified sampling.
"""

import numpy as np
import json
import os
import shutil
from scipy.special import expit

# ===== Helper functions =====
def _softmax_np(x, tau=1.0):
    x = np.asarray(x) / max(tau, 1e-8)
    x = x - x.max()
    ex = np.exp(x)
    return ex / ex.sum()

def _edges_from_nl(nl):
    """
    Extract an edge list (and optional k-way tuples) from `nl` to store GT interactions.
    Only includes kinds we plot/evaluate (extend as you add kinds).
    """
    edges = []
    for term in nl:
        kind = term[0]

        # pairwise main+nonlinear (your existing kinds)
        if kind in {"main_int_linear", "main_int_tanh", "main_int_sin", "main_int_cos"}:
            if kind == "main_int_linear":
                _, i, j, base, w = term
                scale = None
            else:
                _, i, j, base, w, scale = term
            edges.append({
                "arity": 2, "type": kind,
                "i": int(i), "j": int(j),
                "weight": float(w), "base": float(base),
                "scale": (None if kind == "main_int_linear" else float(scale))
            })

        # your 3-way example
        elif kind == "main_int_arctan_prod":
            # (kind, i, j, k, factor, base, w, scale)
            _, i, j, k, factor, base, w, scale = term
            edges.append({
                "arity": 3, "type": kind,
                "i": int(i), "j": int(j), "k": int(k),
                "factor": float(factor), "base": float(base),
                "weight": float(w), "scale": float(scale)
            })

        # NEW kinds added below: record as interaction "patterns"
        elif kind in {"set_match", "sparse_kway", "cluster_max", "index_coded", "switchboard_pair", "group_bipartite", "pointer_from_bits", "two_hop_exist", "conditional_int", "sign_switch", "heteroskedastic_int", "piecewise_int", "higher_order_int", "custom_tanh_prod_plus_sin"}:
            edges.append({
                "arity": "varies", "type": kind,
                "spec": [float(x) if isinstance(x, (int, float)) else x for x in term[1:]]
            })
        # else: ignore purely univariate kinds in the interaction graph

    return edges

# ===== Distribution-based scenarios for method comparison =====
SCENARIOS = {
    # "A": {  # Linear baseline
    #     "p": 7, "sigma": 0.1, "b0": 1.0,
    #     "beta": {0: 2.0, 1: -3.0, 2: 1.5, 3: -2.0},
    #     "nl": [],
    #     "dist": ("normal", 0, 1),
    #     "desc": "Scenario A: Purely linear baseline (N(0,1))"
    # },
    # "B": {  # Main effect + strong interaction
    #     "p": 6, "sigma": 0.1, "b0": 0.5,
    #     # added x2 baseline as well
    #     "beta": {0: 1.5, 1: 2.5, 2: 0.5, 3: -0.5},
    #     "nl": [
    #         ("main_int_linear", 1, 2, 0.0, 15.0)
    #     ],
    #     "dist": ("normal", 0, 1),
    #     "desc": "Scenario B: Main effect + strong x2*x3 interaction"
    # },

    "C": {  # Main effect + strong interaction
        "p": 9, "sigma": 0.1, "b0": 0.5,
        # added x2 baseline as well
        "beta": {0: 0.25, 1: 1.5, 2: 1.5, 3: -1.5, 4: -2.0, 5: 0.25},
        "nl": [
            ("main_int_linear", 1, 2, 0.0, 15.0),

            ("main_int_linear", 3, 4, 0.0, 15.0),
        ],
        "dist": ("normal", 0, 1),
        "desc": "Scenario B: Main effect + strong x2*x3 interaction"
    },



    "D": {
        "p": 3, "sigma": 0.1, "b0": 1.0,   # constant +1
        # true linear coefficients: x1=+1, x2=-2
        "beta": {0: 2.0, 1: -2.0},

        "nl": [
            # + 2 * x1 * tanh(1 * x2)             (no linear base here)
            ("main_int_tanh", 0, 1, 0.0,  4.0, 1.0),

            # + 6 * x2 * sin(2 * x1)              (no linear base here)
            ("main_int_sin",  1, 0, 0.0,  -6.0, 2.0),

            # - 2 * x2 * tanh(2 * x1)             (already base=0.0)
            ("main_int_tanh", 1, 0, 0.0, -2.0, 2.0)
        ],

        "dist": ("normal", 0, 1),#("uniform", -3, 3),
        "desc": "Scenario C: linear x1=+1, x2=-2 with nonlinear interactions (tanh & sin)."
     },

     "E": {  # n=10000, p=3 - Toy classification with tanh and sin
         "p": 3, "sigma": 0.1, "b0": 0.0,
         "beta": {0: 3.0, 1: -3.0},
         "nl": [
             # +3 * x1 * [1 + tanh(10*x2)]
             ("main_int_tanh", 0, 1, 0.0, 3.0, 10.0),  # 3 * x1 * (1 + tanh(10*x2))

             # -3 * x2 * [1 + sin(-2*x1)]
             ("main_int_sin", 1, 0, 0.0, -3.0, -2.0)   # -3 * x2 * (1 + sin(-2*x1))
         ],
         "dist": ("normal", 0, 1),
         "desc": "Setting 5 (toy): n=10000, p=3 - Classification with tanh and sin interactions"
     },

     "F": {
          "p": 50, "sigma": 0.1, "b0": 0.0,

          # put the bracket “+1” linear parts here:
          #  -2*x1 + 2*x2 + 3*x4 - 1*x5
          "beta": {0: -2.0, 1: 2.0, 3: 3.0, 4: -1.0},

          # nonlinear *only* (no base inside nl terms)
          "nl": [
            # -2 * x1 * tanh(x2*x4)
            ("main_int_tanh_prod",      0, 1, 3, 0.0, -2.0, 1.0),

            # + 2 * x2 * (2/pi) * arctan(x4 - x5)  ==> weight = 4/pi
            ("main_int_arctan_diff",    1, 3, 4, 0.0, (4.0/np.pi), 1.0),

            # + 3 * x4 * tanh(x2 + sin(x5))
            ("main_int_tanh_x_plus_sin",3, 1, 4, 0.0, 3.0, 1.0),

            # - 1 * x5 * (2σ(x1*x4) - 1)  ==  - x5 * tanh((x1*x4)/2)
            ("main_int_tanh_prod",      4, 0, 3, 0.0, -1.0, 0.5),
          ],

          "dist": ("normal", 0, 1),
          "desc": "High-dim sparse interactions with linear parts in beta, pure interactions in nl"
    }


 }


def gen_data(n, spec):
    """Generate synthetic data according to specification."""
    rng = np.random.default_rng(42)
    p, sigma, b0 = spec["p"], spec["sigma"], spec.get("b0", 0.0)

    # Beta vector
    beta = np.zeros(p)
    for j, v in spec["beta"].items():
        beta[j] = float(v)

    # ---- Feature distribution ----
    dist = spec.get("dist", ("normal", 0, 1))
    if dist[0] == "normal":
        mu, std = dist[1], dist[2]
        X = rng.normal(loc=mu, scale=std, size=(n, p))
    elif dist[0] == "uniform":
        low, high = dist[1], dist[2]
        X = rng.uniform(low, high, size=(n, p))
    elif dist[0] == "t":
        df = dist[1]
        X = rng.standard_t(df, size=(n, p))
    else:
        raise ValueError(f"Unknown distribution {dist[0]}")

    # ---- Linear predictor ----
    eta = b0 + X @ beta

    # ---- Nonlinear terms ----
    for term in spec["nl"]:
        kind = term[0]
        if kind == "int":
            _, i, j, w = term
            eta += float(w) * (X[:, i] * X[:, j])
        elif kind == "int_tanh":
            _, i, j, w, scale = term
            eta += float(w) * (X[:, i] * np.tanh(scale * X[:, j]))
        elif kind == "int_sin":
            _, i, j, w, scale = term
            eta += float(w) * (X[:, i] * np.sin(scale * X[:, j]))

        # ---- deine neuen "main+nonlinear" Varianten ----
        elif kind == "main_int_tanh":
            # (kind, i, j, base, w, scale)
            _, i, j, base, w, scale = term
            eta += X[:, i] * (base + w * np.tanh(scale * X[:, j]))

        elif kind == "main_int_linear":
            # (kind, i, j, base, w)
            _, i, j, base, w = term
            eta += X[:, i] * (base + w * X[:, j])

        elif kind == "main_int_sin":
            # (kind, i, j, base, w, scale)
            _, i, j, base, w, scale = term
            eta += X[:, i] * (base + w * np.sin(scale * X[:, j]))

        # ---- Rest unverändert ----
        elif kind == "sin":
            _, j, w = term
            eta += float(w) * np.sin(X[:, j])
        elif kind == "sin_scaled":
            _, j, w, scale = term
            eta += float(w) * np.sin(scale * X[:, j])
        elif kind == "tanh":
            _, j, w = term
            eta += float(w) * np.tanh(X[:, j])
        elif kind == "int_cos":
            _, i, j, w, scale = term
            eta += float(w) * (X[:, i] * np.cos(scale * X[:, j]))
        elif kind == "sqc":
            _, j, w = term
            xj = X[:, j]
            eta += float(w) * (xj**2 - 1.0)
        elif kind == "main_int_cos":
            # (kind, i, j, base, w, scale)
            _, i, j, base, w, scale = term
            eta += X[:, i] * (base + w * np.cos(scale * X[:, j]))
        elif kind == "main_int_arctan_prod":

            # (kind, i, j, k, factor, base, w, scale)

            _, i, j, k, factor, base, w, scale = term

            eta += factor * X[:, i] * (base + w * np.arctan(scale * (X[:, j] * X[:, k])))

        elif kind == "main_int_tanh_prod":
            # (kind, i, j, k, base, w, scale)
            _, i, j, k, base, w, scale = term
            eta += X[:, i] * (base + w * np.tanh(scale * (X[:, j] * X[:, k])))

        elif kind == "main_int_arctan_diff":
            # (kind, i, j, k, base, w, scale)
            _, i, j, k, base, w, scale = term
            eta += X[:, i] * (base + w * np.arctan(scale * (X[:, j] - X[:, k])))

        elif kind == "main_int_tanh_x_plus_sin":
            # (kind, i, j, k, base, w, scale)  # scale applies to sin
            _, i, j, k, base, w, scale = term
            eta += X[:, i] * (base + w * np.tanh(X[:, j] + np.sin(scale * X[:, k])))

        # NEW scenario-specific terms
        elif kind == "conditional_int":
            # (kind, i, j, gate_idx, w) - 1{gate>0} * w * xi * xj
            _, i, j, gate_idx, w = term
            eta += w * (X[:, gate_idx] > 0).astype(float) * X[:, i] * X[:, j]

        elif kind == "sign_switch":
            # (kind, i, j, w, scale) - w * xi * tanh(scale * xj)
            _, i, j, w, scale = term
            eta += w * X[:, i] * np.tanh(scale * X[:, j])

        elif kind == "heteroskedastic_int":
            # (kind, i, j, scale_idx, w, scale_factor) - (1 + scale_factor*|x_scale_idx|) * w * xi * xj
            _, i, j, scale_idx, w, scale_factor = term
            eta += (1 + scale_factor * np.abs(X[:, scale_idx])) * w * X[:, i] * X[:, j]

        elif kind == "piecewise_int":
            # (kind, i, j, gate_idx, w, scale) - w * xi * xj * (σ(scale*x_gate) - σ(-scale*x_gate))
            _, i, j, gate_idx, w, scale = term
            gate_val = X[:, gate_idx]
            eta += w * X[:, i] * X[:, j] * (expit(scale * gate_val) - expit(-scale * gate_val))

        elif kind == "higher_order_int":
            # (kind, i, j, k, w, scale) - w * xi * tanh(scale * xj * xk)
            _, i, j, k, w, scale = term
            eta += w * X[:, i] * np.tanh(scale * X[:, j] * X[:, k])

        elif kind == "custom_tanh_prod_plus_sin":
            # (kind, i, j, k, l, base, w, scale) - w * xi * tanh(xj*xk + sin(scale*xl))
            _, i, j, k, l, base, w, scale = term
            eta += X[:, i] * (base + w * np.tanh(X[:, j] * X[:, k] + np.sin(scale * X[:, l])))

        else:
            raise ValueError(f"Unknown nonlinear term type: {kind}")

    # Capture noiseless eta and interaction edges
    eta_noiseless = eta.copy()
    nl_edges = _edges_from_nl(spec["nl"])

    # ---- Noise + Labels ----
    eta += rng.normal(0.0, sigma, size=n)
    
    # All scenarios are classification tasks (apply sigmoid + Bernoulli)
    y = rng.binomial(1, expit(eta))

    # Effective linearization (least-squares of noiseless eta on X)
    beta_eff = np.linalg.lstsq(X, eta_noiseless, rcond=None)[0]

    true_support = np.where(beta != 0)[0].tolist()
    return X, y, beta, true_support, eta_noiseless, beta_eff, nl_edges

def make_fixed_test_indices(y, test_frac=0.5, seed=123):
    """Create stratified fixed big test set; returns (idx_test, idx_pool)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    
    # Get indices for each class
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    # Stratified split
    k0 = int(round(test_frac * len(idx0)))
    k1 = int(round(test_frac * len(idx1)))
    
    idx_test = np.concatenate([idx0[:k0], idx1[:k1]])
    idx_pool = np.concatenate([idx0[k0:], idx1[k1:]])
    
    # Shuffle final indices
    rng.shuffle(idx_test)
    rng.shuffle(idx_pool)
    
    return idx_test, idx_pool

def stratified_train_val_from_pool(y, idx_pool, n_train, n_val, target_pos=0.5, seed=42):
    """
    Robust stratified train/val splitter with undersampling guards.
    
    Args:
        y: Full label array
        idx_pool: Pool indices to sample from
        n_train: Desired training size
        n_val: Desired validation size
        target_pos: Target positive class proportion (None for natural distribution)
        seed: Random seed
        
    Returns:
        train_idx, val_idx: Stratified indices with guards against empty splits
    """
    rng = np.random.default_rng(seed)
    y_pool = y[idx_pool]
    pos_pool = idx_pool[y_pool == 1]
    neg_pool = idx_pool[y_pool == 0]

    # Guard 1: if any class empty, fall back to no rebalance (just stratify as is)
    if len(pos_pool) == 0 or len(neg_pool) == 0:
        # simple split preserving whatever is there
        idx_all = idx_pool.copy()
        rng.shuffle(idx_all)
        n_tot = min(len(idx_all), n_train + n_val)
        take = idx_all[:n_tot]
        tr = take[:min(n_train, len(take))]
        va = take[min(n_train, len(take)):min(n_train + n_val, len(take))]
        return np.array(tr), np.array(va)
    
    # If no rebalancing requested, use natural distribution
    if target_pos is None:
        # Simple stratified split preserving natural proportions
        n_total = min(len(idx_pool), n_train + n_val)
        n_train_actual = min(n_train, n_total)
        n_val_actual = min(n_val, n_total - n_train_actual)
        
        # Stratified split maintaining natural proportions
        pos_frac = len(pos_pool) / len(idx_pool)
        n_tr_pos = int(round(n_train_actual * pos_frac))
        n_tr_neg = n_train_actual - n_tr_pos
        n_va_pos = int(round(n_val_actual * pos_frac))
        n_va_neg = n_val_actual - n_va_pos
        
        # Cap by availability
        n_tr_pos = min(n_tr_pos, len(pos_pool))
        n_tr_neg = min(n_tr_neg, len(neg_pool))
        n_va_pos = min(n_va_pos, len(pos_pool) - n_tr_pos)
        n_va_neg = min(n_va_neg, len(neg_pool) - n_tr_neg)
        
        # Sample
        pos_shuf = rng.permutation(pos_pool)
        neg_shuf = rng.permutation(neg_pool)
        tr_pos = pos_shuf[:n_tr_pos]
        tr_neg = neg_shuf[:n_tr_neg]
        va_pos = pos_shuf[n_tr_pos:n_tr_pos + n_va_pos]
        va_neg = neg_shuf[n_tr_neg:n_tr_neg + n_va_neg]
        
        train_idx = np.concatenate([tr_pos, tr_neg])
        val_idx = np.concatenate([va_pos, va_neg])
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        
        return np.array(train_idx), np.array(val_idx)

    # Desired counts under undersampling
    n_tr_pos = int(round(n_train * target_pos))
    n_tr_neg = n_train - n_tr_pos
    n_va_pos = int(round(n_val * target_pos))
    n_va_neg = n_val - n_va_pos

    # Guard 2: cap by availability
    n_tr_pos = min(n_tr_pos, len(pos_pool))
    n_tr_neg = min(n_tr_neg, len(neg_pool))
    # reserve for val
    rem_pos = len(pos_pool) - n_tr_pos
    rem_neg = len(neg_pool) - n_tr_neg
    n_va_pos = min(n_va_pos, rem_pos)
    n_va_neg = min(n_va_neg, rem_neg)

    # Guard 3: if still short, relax target_pos to "as available"
    if n_tr_pos + n_tr_neg < n_train or n_va_pos + n_va_neg < n_val:
        # fill remaining from the larger class
        need_tr = n_train - (n_tr_pos + n_tr_neg)
        need_va = n_val - (n_va_pos + n_va_neg)
        if need_tr > 0:
            # take from whichever has more left
            add_from_pos = min(need_tr, rem_pos - n_va_pos) if rem_pos > rem_neg else 0
            add_from_neg = need_tr - add_from_pos
            n_tr_pos += max(0, add_from_pos)
            n_tr_neg += max(0, add_from_neg)
        if need_va > 0:
            # now use whatever remains
            rem_pos2 = len(pos_pool) - n_tr_pos
            rem_neg2 = len(neg_pool) - n_tr_neg
            add_from_pos = min(need_va, rem_pos2)
            add_from_neg = min(need_va - add_from_pos, rem_neg2)
            n_va_pos += add_from_pos
            n_va_neg += add_from_neg

    # Sample
    pos_shuf = rng.permutation(pos_pool)
    neg_shuf = rng.permutation(neg_pool)
    tr_pos = pos_shuf[:n_tr_pos]
    tr_neg = neg_shuf[:n_tr_neg]
    va_pos = pos_shuf[n_tr_pos:n_tr_pos + n_va_pos]
    va_neg = neg_shuf[n_tr_neg:n_tr_neg + n_va_neg]

    train_idx = np.concatenate([tr_pos, tr_neg])
    val_idx   = np.concatenate([va_pos, va_neg])

    # Final guard: never return empty splits
    if len(train_idx) == 0:
        # fall back to random
        all_shuf = rng.permutation(idx_pool)
        train_idx = all_shuf[:min(n_train, len(all_shuf))]
        val_idx   = all_shuf[min(n_train, len(all_shuf)) : min(n_train + n_val, len(all_shuf))]

    return np.array(train_idx), np.array(val_idx)

def main():
    """Generate and save all synthetic datasets."""
    
    # Clean output directory
    output_dir = "../data/synthetic"
    if os.path.exists(output_dir):
        print(f"🗑️  Cleaning existing data in {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("CLEAN SYNTHETIC DATA GENERATOR - NIMO-INSPIRED")
    print("="*60)
    print(f"Scenarios: {list(SCENARIOS.keys())}")
    print(f"Output directory: {output_dir}")
    print()

    # Fixed parameters
    seed = 42

    for scenario_name, spec in SCENARIOS.items():
        print(f"Generating Scenario {scenario_name}: {spec['desc']}")
        
        # Use consistent sample size for all scenarios
        n_full = 10000   # Large pool for all scenarios
        
        # Generate data
        X, y, beta, true_support, eta_noiseless, beta_eff, nl_edges = gen_data(n_full, spec)
        
        # Create fixed big test set (stratified) + pool for train/val sampling
        idx_test, idx_pool = make_fixed_test_indices(y, test_frac=0.5, seed=seed+7)
        
        # Save arrays and indices
        np.save(f"{output_dir}/scenario_{scenario_name}_X_full.npy", X)
        np.save(f"{output_dir}/scenario_{scenario_name}_y_full.npy", y.astype(int))
        np.save(f"{output_dir}/scenario_{scenario_name}_idx_test_big.npy", idx_test)
        np.save(f"{output_dir}/scenario_{scenario_name}_idx_pool.npy", idx_pool)
        
        # Minimal metadata
        metadata = {
            "scenario": scenario_name,
            "desc": spec["desc"],
            "p": spec["p"],
            "sigma": spec["sigma"],
            "b0": spec.get("b0", 0.0),
            "beta_nonzero": {int(k): float(v) for k, v in spec["beta"].items()},
            "nl": spec["nl"],
            "true_support": true_support,
            "n_full": n_full,
            "sizes": {"test_big": len(idx_test), "pool": len(idx_pool)},
            "task": "classification",
            "class_dist_full": np.bincount(y).tolist(),
            "nl_edges": nl_edges,
            "beta_linearization": beta_eff.tolist()
        }
        
        with open(f"{output_dir}/scenario_{scenario_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Saved data and metadata for scenario {scenario_name}")
        print(f"    Full: {X.shape[0]} samples, Features: {X.shape[1]}")
        print(f"    Test set: {len(idx_test)} samples, Pool: {len(idx_pool)} samples")
        print(f"    True support: {true_support}")
        print(f"    Beta: {[f'{beta[i]:.1f}' for i in true_support]}")
        print(f"    Class distribution: {np.bincount(y)}")
        print()

    print("="*60)
    print("CLEAN DATA GENERATION COMPLETED")
    print("="*60)
    print(f"All datasets saved to: {output_dir}")
    print("Ready for experiments!")
    print()
    print("Usage in experiment loop:")
    print("  X = np.load('.../scenario_A_X_full.npy')")
    print("  y = np.load('.../scenario_A_y_full.npy')")
    print("  idx_test = np.load('.../scenario_A_idx_test_big.npy')")
    print("  idx_pool = np.load('.../scenario_A_idx_pool.npy')")
    print("  # Sample train/val from idx_pool for each run")

if __name__ == "__main__":
    main()