#!/usr/bin/env python3
"""
F1 All Scenarios Compact Plot - Single plot showing F1 scores over iterations for all scenarios in a compact layout.
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FixedLocator

# Force consistent settings for perfect plots
matplotlib.use('Agg')
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 18,
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

# Typography: smaller titles, serif look, CM-style math
plt.rcParams.update({
    "axes.titlesize": 14,          # was 18
    "font.family": "serif",
    "font.serif": ["CMU Serif", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
})

def standard_boxplot(ax, df, x="Method", y="F1", order=None, colors=None,
                     title="F1 over iterations", whis=1.5, width=0.85):
    """
    Standard boxplot with visible outliers.
    Uses Matplotlib's boxplot for full control.
    """
    cats = order or sorted(df[x].unique().tolist())
    data = [df.loc[df[x] == c, y].to_numpy() for c in cats]

    bp = ax.boxplot(
        data,
        notch=False,              # ← turn off notches
        widths=width,
        whis=1.5,                 # ← standard Tukey definition
        showmeans=False,
        showfliers=True,
        patch_artist=True,
        flierprops=dict(marker="o", markersize=3, markerfacecolor="black",
                        markeredgewidth=0, alpha=0.7),
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )

    # Color each box
    if colors:
        for box, cat in zip(bp["boxes"], cats):
            c = colors.get(cat, "#cccccc")
            box.set_facecolor(c)
            box.set_edgecolor("black")
            box.set_linewidth(0.8)

    # X ticks/labels
    ax.set_xticks(np.arange(1, len(cats) + 1))
    ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=10)

    # Title/labels
    ax.set_title(title, fontsize=12, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("Test F1", fontsize=10)

    # Grid + frame + tickmarks
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color="#999999", alpha=0.7)
    ax.xaxis.grid(False)
    ax.tick_params(axis="both", which="both", direction="out", length=4, width=1.0)
    for s in ax.spines.values():
        s.set_color("black"); s.set_linewidth(1.0)

    # Slight margin to reduce whitespace
    ax.margins(x=0.02)

def count_outliers_by_group(df, x="Method", y="F1", order=None, whis=1.5):
    """Count outliers per group for diagnostic purposes."""
    cats = order or sorted(df[x].unique().tolist())
    out = {}
    for c in cats:
        v = df.loc[df[x]==c, y].to_numpy()
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        if isinstance(whis, (int, float)):
            low, high = q1 - whis*iqr, q3 + whis*iqr
        else:
            low, high = np.percentile(v, whis[0]), np.percentile(v, whis[1])
        out[c] = int(((v < low) | (v > high)).sum())
    return out

def create_f1_all_scenarios_compact_plot(df_synthetic, save_path=None):
    """Create a compact single plot showing F1 scores for all scenarios."""
    
    print("Creating F1 all scenarios compact plot...")
    
    if df_synthetic.empty:
        print("No data found")
        return None
    
    # Dynamically detect available methods and scenarios
    available_methods = sorted(df_synthetic['model_name'].unique().tolist())
    available_scenarios = sorted(df_synthetic['dataset_id'].unique().tolist())
    print(f"Available methods: {available_methods}")
    print(f"Available scenarios: {available_scenarios}")
    
    # Define consistent, distinct palettes
    BOX_COLORS = {}
    
    # Add colors for available methods dynamically
    method_colors = {
        "Lasso": "#396AB1",          # blue
        "LassoNet": "#00A0A0",       # teal
        "NN": "#B07AA1",             # purple
        "NIMO": "#DA7C30",           # orange
        "NIMO_T": "#DA7C30",         # orange (same as NIMO)
        "NIMO_MLP": "#FF6B35",       # red-orange
        "RF": "#9C755F",             # brown
    }
    
    for method in available_methods:
        if method in method_colors:
            BOX_COLORS[method] = method_colors[method]
        else:
            # Default color for unknown methods
            BOX_COLORS[method] = "#666666"
    
    # Create subplots for each scenario
    n_scenarios = len(available_scenarios)
    
    # Calculate grid dimensions - always use 2x3 for compact layout
    nrows, ncols = 2, 3
    figsize = (15, 10)  # Compact size
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    # Global figure title
    fig.suptitle("F1 Scores Across All Scenarios", fontsize=16, y=0.95)
    
    # Create a plot for each scenario
    for i, scenario in enumerate(available_scenarios):
        ax = axes[i]
        
        # Filter data for this scenario
        dd = df_synthetic[df_synthetic['dataset_id'] == scenario].copy()
        
        if dd.empty:
            ax.text(0.5, 0.5, f"No data for {scenario}", 
                   ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        
        # Extract F1 scores for boxplot
        f1_tbl = dd.pivot_table(index="iteration", columns="model_name", values="f1")
        
        # Convert to long format for plotting
        f1_long = f1_tbl.reset_index().melt(id_vars="iteration", var_name="Method", value_name="F1")
        
        # Filter to only methods that exist in this scenario
        present_methods = [m for m in available_methods if m in f1_long['Method'].unique()]
        df_plot = f1_long[f1_long['Method'].isin(present_methods)].copy()
        
        if df_plot.empty:
            ax.text(0.5, 0.5, f"No F1 data for {scenario}", 
                   ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        
        # Create standard boxplot with visible outliers
        standard_boxplot(
            ax, df_plot, x="Method", y="F1", order=present_methods,
            colors={k: BOX_COLORS[k] for k in present_methods},
            title=f"Scenario {scenario}",
            whis=1.5,          # standard Tukey definition
            width=0.8          # make boxes a bit narrower for compact layout
        )
        
        # Optional: Check outlier counts for diagnostic purposes
        outlier_counts = count_outliers_by_group(df_plot, order=present_methods, whis=1.5)
        print(f"  - Scenario {scenario} outliers per method: {outlier_counts}")
    
    # Hide unused subplots
    for i in range(n_scenarios, len(axes)):
        axes[i].axis("off")
    
    # Add a single legend for all subplots
    # Get the legend from the first subplot that has data
    legend_added = False
    for i, scenario in enumerate(available_scenarios):
        ax = axes[i]
        if not ax.get_visible() or len(ax.get_children()) == 0:
            continue
        
        # Get the legend handles and labels
        handles, labels = ax.get_legend_handles_labels()
        if handles and not legend_added:
            # Add legend to the figure
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                      ncol=len(handles), frameon=True, fancybox=True, shadow=True)
            legend_added = True
            break
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.15)  # Make room for suptitle and legend
    
    # Always save the plot when run
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"✓ F1 all scenarios compact plot saved to {save_path}")
    else:
        default_path = "withResiduals/f1_all_scenarios_compactWithResiduals.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"✓ F1 all scenarios compact plot saved to {default_path}")
    
    plt.close()
    return fig

def main():
    """Main function to generate F1 all scenarios compact plot."""
    print("=== Generating F1 All Scenarios Compact Plot ===")
    print("This will create a compact single plot showing F1 scores for all scenarios!")
    
    # Load data
    results_path = '../../../results/synthetic/experiment_results.csv'
    df = pd.read_csv(results_path)
    
    # Filter dynamically from whatever is in the results file
    synthetic_datasets = sorted(df['dataset_id'].unique().tolist())
    df_synthetic = df[df['dataset_id'].isin(synthetic_datasets)].copy()
    
    print(f"Loaded {len(df_synthetic)} synthetic experiments")
    print(f"Methods: {df_synthetic['model_name'].unique()}")
    print(f"Scenarios: {synthetic_datasets}")
    
    # Generate the plot
    try:
        save_path = ""
        fig = create_f1_all_scenarios_compact_plot(df_synthetic, save_path=save_path)
        print(f"\n🎉 Successfully generated F1 all scenarios compact plot!")
        print(f"File saved: {save_path}")
    except Exception as e:
        print(f"✗ Error creating F1 all scenarios compact plot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
