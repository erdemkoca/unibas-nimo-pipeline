#!/usr/bin/env python3
"""
Combined Dataset Runtime Boxplots
Creates a single figure with two side-by-side plots:
- Left: All synthetic datasets combined across all iterations
- Right: All real datasets combined across all iterations
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# --- Matplotlib styling (similar to your other plotting script) ---
matplotlib.use("Agg")
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 1.2,
    'grid.linewidth': 1.0,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.5,
    'xtick.major.size': 6,
    'xtick.minor.size': 4,
    'ytick.major.size': 6,
    'ytick.minor.size': 4,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.grid': False,
    'axes.axisbelow': True,
})

# Method palette (feel free to adjust)
BOX_COLORS = {
    "Lasso":        "#396AB1",
    "LassoNet":     "#00A0A0",
    "NN":           "#2CA02C",
    "NIMO_T":       "#DA7C30",
    "NIMO_MLP":     "#B07AA1",
    "RF":           "#9C755F",
}
METHOD_ORDER = ["Lasso", "LassoNet", "NN", "NIMO_T", "NIMO_MLP", "RF"]

def parse_maybe_json(s):
    """Parse JSON-like strings safely (for 'timing' column)."""
    if pd.isna(s) or s == "":
        return None
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except Exception:
        # try literal_eval as a fallback (handles single quotes)
        import ast
        try:
            return ast.literal_eval(s)
        except Exception:
            return None

def coerce_numeric(col):
    """Coerce a pandas series to numeric, ignoring non-numeric."""
    return pd.to_numeric(col, errors="coerce")

def standard_boxplot(ax, df, x="model_name", y="value", order=None, colors=None,
                     title="", ylabel="", whis=1.5, width=0.85, logy=False):
    """
    Standard boxplot (Tukey) with visible outliers and consistent coloring per method.
    df: long-form with columns [x, y]
    """
    cats = order or sorted(df[x].unique().tolist())
    data = [df.loc[df[x] == c, y].dropna().to_numpy() for c in cats]

    # No data case
    if all(len(d) == 0 for d in data):
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    bp = ax.boxplot(
        data,
        notch=False,
        widths=width,
        whis=whis,
        showmeans=False,
        showfliers=True,
        patch_artist=True,
        flierprops=dict(marker="o", markersize=4, markerfacecolor="black",
                        markeredgewidth=0, alpha=0.7),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    # Color each box
    if colors:
        for box, cat in zip(bp["boxes"], cats):
            c = colors.get(cat, "#cccccc")
            box.set_facecolor(c)
            box.set_edgecolor("black")
            box.set_linewidth(1.0)

    # X ticks/labels
    ax.set_xticks(np.arange(1, len(cats) + 1))
    ax.set_xticklabels(cats, rotation=0, ha="center")

    # Title/labels
    ax.set_title(title, fontsize=14, pad=8)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=12)

    if logy:
        ax.set_yscale("log")

    # Grid + frame + tickmarks
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=1.0, color="#999999", alpha=0.7)
    ax.xaxis.grid(False)
    ax.tick_params(axis="both", which="both", direction="out", length=6, width=1.2)
    for s in ax.spines.values():
        s.set_color("black"); s.set_linewidth(1.2)

    ax.margins(x=0.02)

def extract_time_columns(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """
    Build a clean long-form time DataFrame with columns:
    [dataset_description/scenario_description, model_name, iteration, time_kind, value]
    time_kind ∈ {train, predict, total} — include only those we can derive.
    """
    df = df.copy()

    # Basic numeric columns if present
    df["training_time_num"] = coerce_numeric(df.get("training_time"))
    df["execution_time_num"] = coerce_numeric(df.get("execution_time"))

    # Parse timing dict (if present)
    timing_parsed = df.get("timing")
    if timing_parsed is not None:
        df["_timing_dict"] = timing_parsed.apply(parse_maybe_json)
        df["timing_fit"] = df["_timing_dict"].apply(lambda d: (d or {}).get("fit") if isinstance(d, dict) else np.nan)
        df["timing_predict"] = df["_timing_dict"].apply(lambda d: (d or {}).get("predict") if isinstance(d, dict) else np.nan)
        df["timing_total"] = df["_timing_dict"].apply(lambda d: (d or {}).get("total") if isinstance(d, dict) else np.nan)
        df["timing_fit"] = coerce_numeric(df["timing_fit"])
        df["timing_predict"] = coerce_numeric(df["timing_predict"])
        df["timing_total"] = coerce_numeric(df["timing_total"])
    else:
        df["timing_fit"] = np.nan
        df["timing_predict"] = np.nan
        df["timing_total"] = np.nan

    # Build long-form rows for whichever measures are available
    rows = []
    if dataset_type == "synthetic":
        cols_needed = ["scenario_description", "model_name", "iteration"]
    else:  # real
        cols_needed = ["dataset_description", "model_name", "iteration"]
    
    for _, r in df.iterrows():
        base = {k: r.get(k) for k in cols_needed}

        # train time
        train_val = r["timing_fit"]
        if pd.isna(train_val) and not pd.isna(r["training_time_num"]):
            train_val = r["training_time_num"]
        if not pd.isna(train_val):
            rows.append({**base, "time_kind": "train", "value": float(train_val)})

        # predict time
        pred_val = r["timing_predict"]
        if not pd.isna(pred_val):
            rows.append({**base, "time_kind": "predict", "value": float(pred_val)})

        # total time
        total_val = r["timing_total"]
        if pd.isna(total_val) and (not pd.isna(r["execution_time_num"]) or not pd.isna(train_val)):
            # try to approximate total if we have pieces
            if not pd.isna(r["execution_time_num"]) and not pd.isna(train_val):
                total_val = float(r["execution_time_num"]) + float(train_val)
            elif not pd.isna(r["execution_time_num"]):
                total_val = float(r["execution_time_num"])
            elif not pd.isna(train_val):
                total_val = float(train_val)
        if not pd.isna(total_val):
            rows.append({**base, "time_kind": "total", "value": float(total_val)})

    if not rows:
        if dataset_type == "synthetic":
            return pd.DataFrame(columns=["scenario_description", "model_name", "iteration", "time_kind", "value"])
        else:
            return pd.DataFrame(columns=["dataset_description", "model_name", "iteration", "time_kind", "value"])

    long_df = pd.DataFrame(rows)
    return long_df

def draw_combined_figure(synthetic_df: pd.DataFrame, real_df: pd.DataFrame, save_path: Path):
    """
    Make a combined figure with two side-by-side plots:
    - Left: All synthetic datasets combined across all iterations
    - Right: All real datasets combined across all iterations
    """
    # Filter for total time only
    synthetic_total = synthetic_df[synthetic_df["time_kind"] == "total"].copy()
    real_total = real_df[real_df["time_kind"] == "total"].copy()
    
    if synthetic_total.empty and real_total.empty:
        print("(!) No total timing data found in either dataset")
        return

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get all methods present in either dataset
    all_methods = set()
    if not synthetic_total.empty:
        all_methods.update(synthetic_total["model_name"].unique())
    if not real_total.empty:
        all_methods.update(real_total["model_name"].unique())
    
    present_methods = [m for m in METHOD_ORDER if m in all_methods]

    # Left plot: Synthetic datasets
    if not synthetic_total.empty:
        standard_boxplot(
            ax1, synthetic_total, x="model_name", y="value",
            order=present_methods,
            colors={k: BOX_COLORS.get(k, "#cccccc") for k in present_methods},
            title="Synthetic Datasets (All Scenarios)",
            ylabel="Total Time (Seconds)",
            whis=1.5,
            width=0.85,
            logy=True,
        )
    else:
        ax1.text(0.5, 0.5, "No synthetic data", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_axis_off()

    # Right plot: Real datasets
    if not real_total.empty:
        standard_boxplot(
            ax2, real_total, x="model_name", y="value",
            order=present_methods,
            colors={k: BOX_COLORS.get(k, "#cccccc") for k in present_methods},
            title="Real Datasets (All Datasets)",
            ylabel="Total Time (Seconds)",
            whis=1.5,
            width=0.85,
            logy=True,
        )
    else:
        ax2.text(0.5, 0.5, "No real data", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_axis_off()

    # Adjust layout
    fig.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.15, wspace=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"✓ Saved: {save_path}")

def main():
    ap = argparse.ArgumentParser(description="Combined runtime boxplots for synthetic and real datasets")
    ap.add_argument("--synthetic-results", type=str, default="../../../results/synthetic/experiment_results.csv")
    ap.add_argument("--real-results", type=str, default="../../../results/real/experiment_results.csv")
    ap.add_argument("--save-dir", type=str, default="combined_time_plots")
    args = ap.parse_args()

    synthetic_path = Path(args.synthetic_results)
    real_path = Path(args.real_results)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load and process synthetic data
    synthetic_df = None
    if synthetic_path.exists():
        print(f"Loading synthetic data from: {synthetic_path}")
        synthetic_raw = pd.read_csv(synthetic_path)
        synthetic_df = extract_time_columns(synthetic_raw, "synthetic")
        print(f"Synthetic datasets: {sorted(synthetic_raw['scenario_description'].dropna().unique().tolist())}")
        print("Synthetic time kinds found:", sorted(synthetic_df["time_kind"].unique()))
    else:
        print(f"Warning: Synthetic results not found at {synthetic_path}")

    # Load and process real data
    real_df = None
    if real_path.exists():
        print(f"Loading real data from: {real_path}")
        real_raw = pd.read_csv(real_path)
        real_df = extract_time_columns(real_raw, "real")
        print(f"Real datasets: {sorted(real_raw['dataset_description'].dropna().unique().tolist())}")
        print("Real time kinds found:", sorted(real_df["time_kind"].unique()))
    else:
        print(f"Warning: Real results not found at {real_path}")

    if synthetic_df is None and real_df is None:
        print("Error: No data found in either synthetic or real results files")
        return

    # Create combined plot
    out = save_dir / "runtime.png"
    draw_combined_figure(synthetic_df, real_df, out)

if __name__ == "__main__":
    main()

