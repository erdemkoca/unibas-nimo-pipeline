#!/usr/bin/env python3
"""
Final perfect plots script - combines working simple approach with full functionality.
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

def parse_json_safe(x):
    """Safely parse JSON-like strings."""
    if pd.isna(x) or x == "":
        return None
    try:
        import ast
        return ast.literal_eval(x)
    except Exception:
        return None

def destring_coeff(row):
    """Extract coefficient information from a row."""
    co = parse_json_safe(row.get("coefficients", "{}"))
    if not isinstance(co, dict):
        return None, None
    
    vals = np.array(co.get("values", []), dtype=float)
    mean = np.array(co.get("mean", []), dtype=float) if "mean" in co else None
    scale = np.array(co.get("scale", []), dtype=float) if "scale" in co else None
    space = co.get("space", "raw")
    
    return {
        "values": vals, 
        "mean": mean, 
        "scale": scale, 
        "space": space,
        "intercept": co.get("intercept", 0.0)
    }, co.get("feature_names", None)

def to_raw_beta(coeff_info):
    """Convert standardized coefficients to raw scale."""
    vals = coeff_info["values"]
    if coeff_info["space"] == "standardized":
        scale = coeff_info["scale"]
        if scale is None:
            raise ValueError("Need scale to de-standardize Lasso coefficients")
        return vals / scale
    return vals

def mean_ci(x, axis=0, alpha=0.05):
    """Calculate mean and confidence interval."""
    mean = np.mean(x, axis=axis)
    n = x.shape[axis]
    se = np.std(x, axis=axis) / np.sqrt(n)
    ci = se * 1.96  # 95% CI
    return mean, ci

def standard_boxplot(ax, df, x="Method", y="F1", order=None, colors=None,
                     title="Scenario — F1 over iterations", whis=1.5, width=0.85):
    """
    Standard boxplot with visible outliers.
    Uses Matplotlib's boxplot for full control.
    """
    cats = order or sorted(df[x].unique().tolist())
    
    # Filter out methods with no valid data
    valid_cats = []
    data = []
    for c in cats:
        method_data = df.loc[df[x] == c, y].dropna()
        if len(method_data) > 0:  # Only include methods with valid data
            valid_cats.append(c)
            data.append(method_data.to_numpy())
        else:
            print(f"Warning: {c} has no valid F1 data, skipping from boxplot")
    
    if not data:
        ax.text(0.5, 0.5, "No valid data for boxplot", 
                ha="center", va="center", transform=ax.transAxes)
        return
    
    bp = ax.boxplot(
        data,
        notch=False,              # ← turn off notches
        widths=width,
        whis=1.5,                 # ← standard Tukey definition
        showmeans=False,
        showfliers=False,         # ← hide outliers
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    # Color each box
    if colors:
        for box, cat in zip(bp["boxes"], valid_cats):
            c = colors.get(cat, "#cccccc")
            box.set_facecolor(c)
            box.set_edgecolor("black")
            box.set_linewidth(1.0)

    # X ticks/labels
    ax.set_xticks(np.arange(1, len(valid_cats) + 1))
    ax.set_xticklabels(valid_cats, rotation=90, ha="right")

    # Title/labels
    ax.set_title(title, fontsize=14, pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("Test F1", fontsize=12)

    # Grid + frame + tickmarks
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=1.0, color="#999999", alpha=0.7)
    ax.xaxis.grid(False)
    ax.tick_params(axis="both", which="both", direction="out", length=6, width=1.2)
    for s in ax.spines.values():
        s.set_color("black"); s.set_linewidth(1.2)

    # Slight margin to reduce whitespace
    ax.margins(x=0.02)

def count_outliers_by_group(df, x="Method", y="F1", order=None, whis=1.5):
    """Count outliers per group for diagnostic purposes."""
    cats = order or sorted(df[x].unique().tolist())
    out = {}
    for c in cats:
        v = df.loc[df[x]==c, y].to_numpy()
        # Handle empty arrays or arrays with all NaN values
        if len(v) == 0 or np.all(np.isnan(v)):
            out[c] = 0
            continue
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        if isinstance(whis, (int, float)):
            low, high = q1 - whis*iqr, q3 + whis*iqr
        else:
            low, high = np.percentile(v, whis[0]), np.percentile(v, whis[1])
        out[c] = int(((v < low) | (v > high)).sum())
    return out

def create_final_plot(scenario_id, df_synthetic, save_path=None):
    """Create the final perfect per-scenario figure."""
    
    print(f"Creating final plot for Scenario {scenario_id}...")
    
    # Filter data for this scenario
    dd = df_synthetic[df_synthetic['dataset_id'] == scenario_id].copy()
    
    if dd.empty:
        print(f"No data found for scenario {scenario_id}")
        return None
    
    # Dynamically detect available methods in the data
    available_methods = sorted(dd['model_name'].unique().tolist())
    print(f"Available methods for scenario {scenario_id}: {available_methods}")
    
    # Get ground truth
    base_row = dd.iloc[0]
    beta_true = parse_json_safe(base_row.get('beta_true', '[]'))
    if beta_true is None:
        beta_true = []
    beta_true = np.array(beta_true, dtype=float)
    d = len(beta_true)
    
    # True support / zeros
    nz_idx = np.where(np.abs(beta_true) > 0)[0]
    z_idx = np.where(np.abs(beta_true) == 0)[0]
    
    # Get feature names and refactor to beta notation
    feature_names = None
    for _, row in dd.iterrows():
        _, names = destring_coeff(row)
        if names is not None:
            feature_names = names
            break
    
    if feature_names is None:
        feature_names = [f"x{j+1}" for j in range(d)]
    
    # Refactor feature names to beta notation (β₁, β₂, β₃, etc.)
    def refactor_feature_names(names):
        """Convert feature_0, feature_1, etc. to β₁, β₂, etc."""
        refactored = []
        for name in names:
            if name.startswith("feature_"):
                try:
                    num = int(name.split("_")[1])
                    refactored.append(f"β{num+1}")
                except (ValueError, IndexError):
                    refactored.append(name)
            else:
                refactored.append(name)
        return refactored
    
    feature_names = refactor_feature_names(feature_names)
    
    # Check if F1 column exists
    if 'f1' not in dd.columns:
        print(f"Warning: No 'f1' column found in data for scenario {scenario_id}")
        print("Available columns:", list(dd.columns))
        print("This suggests the synthetic experiments failed to complete successfully.")
        print("Skipping F1 boxplot creation...")
        f1_tbl = None
    else:
        # Extract F1 scores for boxplot
        f1_tbl = dd.pivot_table(index="iteration", columns="model_name", values="f1")
        
        # Debug: Print F1 data availability
        print(f"F1 data availability for scenario {scenario_id}:")
        for method in available_methods:
            if method in f1_tbl.columns:
                valid_count = f1_tbl[method].notna().sum()
                total_count = len(f1_tbl[method])
                print(f"  {method}: {valid_count}/{total_count} valid F1 scores")
            else:
                print(f"  {method}: No F1 data in pivot table")
        print()
    
    # Collect model coefficients
    method_betas = {}
    print(f"Coefficient data availability for scenario {scenario_id}:")
    for method in available_methods:
        try:
            betas = []
            method_rows = dd[dd['model_name'] == method]
            for _, row in method_rows.iterrows():
                info, _ = destring_coeff(row)
                if info is not None and info["values"] is not None:
                    try:
                        beta_raw = to_raw_beta(info)
                        betas.append(beta_raw)
                    except Exception as e:
                        print(f"Warning: Could not process coefficients for {method}: {e}")
                        continue
            if betas:
                method_betas[method] = np.array(betas)
                print(f"  {method}: {len(betas)} valid coefficient sets")
            else:
                print(f"  {method}: No valid coefficient data")
        except Exception as e:
            print(f"Warning: Could not collect coefficients for {method}: {e}")
            print(f"  {method}: Error in coefficient collection")
    print()
    
    # Define consistent, distinct palettes
    # Boxplot colors (method names as in your CSV)
    BOX_COLORS = {}
    
    # Add colors for available methods dynamically
    method_colors = {
        "Lasso": "#396AB1",          # blue
        "LassoNet": "#00A0A0",       # teal
        "NN": "#B8860B",             # matte dark goldenrod
        "NIMO": "#DA7C30",           # orange
        "NIMO_T": "#DA7C30",         # orange (same as NIMO)
        "NIMO_MLP": "#B07AA1",       # purple
        "RF": "#654321",             # dark brown
    }
    
    for method in available_methods:
        if method in method_colors:
            BOX_COLORS[method] = method_colors[method]
        else:
            # Default color for unknown methods
            BOX_COLORS[method] = "#666666"
    
    # Coefficient panel colors (GT gets green, NN gets gray)
    COEF_COLORS = {
        "GT":    "#2CA02C",          # green (was NN's color)
    }
    
    # Add colors for available methods dynamically
    method_colors = {
        "Lasso": "#396AB1",          # blue
        "NIMO":  "#DA7C30",          # orange
        "NIMO_T": "#DA7C30",         # orange (same as NIMO)
        "NIMO_MLP": "#B07AA1",       # purple
        "RF": "#654321",             # dark brown
        "NN": "#B8860B",             # matte dark goldenrod
        "LassoNet": "#00A0A0",       # teal
    }
    
    for method in available_methods:
        if method in method_colors:
            COEF_COLORS[method] = method_colors[method]
        else:
            # Default color for unknown methods
            COEF_COLORS[method] = "#666666"
    
    # Shared styling constants
    TITLE_SIZE = 14
    TICK_SIZE = 12
    
    # Create the plot - special layout for scenarios D and E
    if scenario_id in ["D", "E"]:
        # 3x1 layout for D and E
        fig = plt.figure(figsize=(18, 6))
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(nrows=1, ncols=3, figure=fig, 
                     width_ratios=[1.2, 1.2, 0.6])
        
        # Global figure title
        fig.suptitle(f"Scenario {scenario_id}", fontsize=16, y=0.98)
        
        # Left: F1 boxplot
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Middle: Feature coefficient variance plot
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Right: Non-zero coefficient plot
        ax3 = fig.add_subplot(gs[0, 2])
    else:
        # Standard 2x3 layout for other scenarios
        fig = plt.figure(figsize=(15, 7))
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(nrows=2, ncols=3, figure=fig, 
                     width_ratios=[1.1, 1.0, 1.4], height_ratios=[1.25, 0.95])
        
        # Global figure title
        fig.suptitle(f"Scenario {scenario_id}", fontsize=16, y=0.98)
        
        # Left: F1 boxplot spans both rows
        ax1 = fig.add_subplot(gs[:, 0])
        
        # Right–top: nonzeros
        ax2 = fig.add_subplot(gs[0, 1:])
        
        # Right–bottom: zeros heatmap
        ax3 = fig.add_subplot(gs[1, 1:])
    
    # --- (A) F1 boxplot with custom styling ---
    if f1_tbl is not None:
        f1_long = f1_tbl.reset_index().melt(id_vars="iteration", var_name="Method", value_name="F1")
        # Use dynamically detected methods instead of hardcoded list
        present = available_methods
        df_plot = f1_long[f1_long['Method'].isin(present)].copy()
        
        # Create standard boxplot with visible outliers
        standard_boxplot(
            ax1, df_plot, x="Method", y="F1", order=present,
            colors={k: BOX_COLORS[k] for k in present},
            title="F1 (across iterations)",
            whis=1.5,          # standard Tukey definition
            width=0.9          # make boxes a bit wider
        )
        
        # Optional: Check outlier counts for diagnostic purposes
        # Filter df_plot to only include methods with valid data
        df_plot_valid = df_plot.dropna(subset=['F1'])
        outlier_counts = count_outliers_by_group(df_plot_valid, order=present, whis=1.5)
        print(f"  - Outliers per method: {outlier_counts}")
    else:
        # Show message when F1 data is not available
        ax1.text(0.5, 0.5, "F1 data not available\n(Synthetic experiments failed)", 
                ha="center", va="center", transform=ax1.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        ax1.set_title("F1 (across iterations) - Data Not Available", fontsize=14)
        ax1.set_xlabel("")
        ax1.set_ylabel("Test F1", fontsize=12)
        ax1.axis("off")
    
    # --- (B) Non-zero coefficients bar+CI vs ground truth ---
    if len(nz_idx) > 0 and method_betas:
        # Get ground truth intercept
        intercept_gt = base_row.get('b0_true', 0.0)
        if isinstance(intercept_gt, str):
            try:
                intercept_gt = float(intercept_gt)
            except:
                intercept_gt = 0.0
        
        # Create coefficient names with β₀ FIRST, then β₁, β₂, ... using mathtext
        labels_math = [r"$\beta_0$"] + [rf"$\beta_{j+1}$" for j in nz_idx]
        
        # Prepare data for plotting
        coef_data = []
        
        # Add intercept β₀ FIRST
        coef_data.append({
            'coef_name': r"$\beta_0$",
            'method': 'GT',
            'value': intercept_gt,
            'ci_low': intercept_gt,
            'ci_high': intercept_gt
        })
        
        # Add intercept data for each available method
        for method in available_methods:
            if method in method_betas:
                # Get intercept for this method
                intercept_value = 0.0
                for _, row in dd[dd['model_name'] == method].iterrows():
                    info, _ = destring_coeff(row)
                    if info is not None:
                        intercept_value = float(info.get('intercept', 0.0))
                        break
                
                coef_data.append({
                    'coef_name': r"$\beta_0$",
                    'method': method,
                    'value': intercept_value,
                    'ci_low': intercept_value,
                    'ci_high': intercept_value
                })
        
        # Add non-zero coefficients
        for i, j in enumerate(nz_idx):
            coef_data.append({
                'coef_name': rf"$\beta_{j+1}$",
                'method': 'GT',
                'value': beta_true[j],
                'ci_low': beta_true[j],
                'ci_high': beta_true[j]
            })
            
            # Add data for each available method
            for method in available_methods:
                if method in method_betas:
                    method_betas_array = method_betas[method]
                    mean_nz, ci_nz = mean_ci(method_betas_array[:, nz_idx], axis=0)
                    coef_data.append({
                        'coef_name': rf"$\beta_{j+1}$",
                        'method': method,
                        'value': mean_nz[i],
                        'ci_low': mean_nz[i] - ci_nz[i],
                        'ci_high': mean_nz[i] + ci_nz[i]
                    })
        
        # Create DataFrame and plot
        coef_df = pd.DataFrame(coef_data)
        coef_df['coef_name'] = pd.Categorical(coef_df['coef_name'], labels_math, ordered=True)
        coef_df.sort_values('coef_name', inplace=True)
        
        # Plot the coefficients
        # Reorder methods: NIMO_T, NIMO_MLP, GT, Lasso
        methods = []
        if 'NIMO_T' in available_methods:
            methods.append('NIMO_T')
        if 'NIMO_MLP' in available_methods:
            methods.append('NIMO_MLP')
        methods.append('GT')  # GT always comes after NIMO methods
        if 'Lasso' in available_methods:
            methods.append('Lasso')
        
        w = 0.2  # Bar width - make bars wider to ensure they touch
        x = np.arange(len(labels_math))
        
        # Calculate positions to group bars tightly by coefficient (feature)
        n_methods = len(methods)
        # Total width of all bars for one coefficient group
        group_width = w * n_methods
        
        # Method name mapping for legend
        method_display_names = {
            "GT": "Ground Truth",
            "NIMO_T": "NIMO_T",
            "NIMO_MLP": "NIMO_MLP", 
            "Lasso": "Lasso",
            "NN": "NN",
            "RF": "RF",
            "LassoNet": "LassoNet"
        }
        
        # Draw bars + CI whiskers
        for i, m in enumerate(methods):
            sub = coef_df[coef_df["method"] == m]
            if not sub.empty:
                # Position bars to be tightly grouped by coefficient with NO gaps between bars
                # Each coefficient gets its own group of bars that touch each other
                x_positions = x - group_width/2 + i*w
                display_name = method_display_names.get(m, m)  # Use display name for legend
                ax2.bar(x_positions, sub["value"], width=w,
                        color=COEF_COLORS[m], edgecolor="black", linewidth=0.5, label=display_name)
                ax2.errorbar(
                    x_positions, sub["value"],
                    yerr=[sub["value"]-sub["ci_low"], sub["ci_high"]-sub["value"]],
                    fmt="none", ecolor="black", elinewidth=0.7, capsize=2
                )
        
        # Red lines for best iteration values are now disabled
        
        # Style the coefficient plot
        ax2.set_title("Non-zero coefficients (across iterations)", fontsize=TITLE_SIZE)
        ax2.set_ylabel("Coefficient", fontsize=TICK_SIZE)
        ax2.tick_params(axis="both", which="both", direction="out", length=6, width=1.2, labelsize=TICK_SIZE)
        
        # Add vertical lines to separate coefficient groups for better visual clarity
        for i in range(1, len(x)):
            ax2.axvline(x[i] - group_width/2 - 0.1, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        
        # Dynamic y-grid based on data range
        from matplotlib.ticker import MultipleLocator
        y_range = ax2.get_ylim()[1] - ax2.get_ylim()[0]
        if y_range <= 2:
            interval = 0.25
        elif y_range <= 4:
            interval = 0.5
        elif y_range <= 8:
            interval = 1.0
        else:
            interval = 2.0
        ax2.yaxis.set_major_locator(MultipleLocator(interval))
        ax2.yaxis.grid(True, linestyle="--", linewidth=1.0, color="#999999", alpha=0.7)
        ax2.xaxis.grid(False)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels_math, rotation=25, ha='right')
        ax2.set_facecolor((1,1,1,0))
        
        # black frame
        for s in ax2.spines.values():
            s.set_color('black')
            s.set_linewidth(1.2)
        
        # Red line explanation removed
        
        # Scenario-specific legend positioning
        if scenario_id == "B":
            # Scenario B: legend in top right
            ax2.legend(
                frameon=True, fancybox=True, framealpha=0.9,
                borderpad=0.4, handlelength=1.5, labelspacing=0.5,
                prop={"size": 12},
                loc="upper right", bbox_to_anchor=(0.98, 0.98), bbox_transform=ax2.transAxes
            )
        elif scenario_id == "C":
            # Scenario C: legend in bottom right
            ax2.legend(
                frameon=True, fancybox=True, framealpha=0.9,
                borderpad=0.4, handlelength=1.5, labelspacing=0.5,
                prop={"size": 12},
                loc="lower right", bbox_to_anchor=(0.98, 0.02), bbox_transform=ax2.transAxes
            )
        else:
            # Default: legend in bottom left
            ax2.legend(
                frameon=True, fancybox=True, framealpha=0.9,
                borderpad=0.4, handlelength=1.5, labelspacing=0.5,
                prop={"size": 12},
                loc="lower left", bbox_to_anchor=(0.01, 0.02), bbox_transform=ax2.transAxes
            )
    else:
        ax2.text(0.5, 0.5, "No coefficient data available", 
                ha="center", va="center", transform=ax2.transAxes)
        ax2.axis("off")
    
    # --- (C) Zero coefficients heatmap (best-F1 run) ---
    if len(z_idx) > 0 and 'f1' in dd.columns:
        heatmap_data = []
        for method in available_methods:
            # pick the best F1 iteration for this method
            method_rows = dd[dd['model_name'] == method]
            if method_rows.empty:
                continue
            
            # Check if there are any valid (non-NaN) F1 values
            valid_f1_mask = method_rows['f1'].notna()
            if not valid_f1_mask.any():
                continue
                
            best_row = method_rows.loc[method_rows['f1'].idxmax()]

            info, _ = destring_coeff(best_row)
            if info is not None and info["values"] is not None:
                try:
                    beta_raw = to_raw_beta(info)
                    for j in z_idx:
                        heatmap_data.append({
                            'method': method,
                            'feature': feature_names[j],
                            'value': beta_raw[j]
                        })
                except Exception as e:
                    print(f"Warning: Could not process coefficients for {method} in best iteration: {e}")
                    continue

        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data)
            heatmap_pivot = heatmap_df.pivot(index='method', columns='feature', values='value')

            # Scenario F specific: Filter out features where all methods have abs(value) <= 0.05
            if scenario_id == "F":
                # Find features where all methods have absolute values <= 0.05
                features_to_keep = []
                for feature in heatmap_pivot.columns:
                    feature_values = heatmap_pivot[feature].fillna(0)
                    if not feature_values.abs().le(0.1).all():
                        features_to_keep.append(feature)
                
                if features_to_keep:
                    heatmap_pivot = heatmap_pivot[features_to_keep]
                    print(f"Scenario F: Filtered from {len(heatmap_pivot.columns) + len(z_idx) - len(features_to_keep)} to {len(features_to_keep)} features with |value| > 0.05")
                else:
                    print("Scenario F: All zero coefficient features have |value| <= 0.05 for all methods, showing empty heatmap")

            sns.heatmap(
                heatmap_pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                ax=ax3, cbar_kws={'shrink': 0.8}
            )
            
            # Scenario F specific title change
            if scenario_id == "F":
                ax3.set_title('Zero Coefficients (>0 features, best iteration)', fontsize=TITLE_SIZE)
            else:
                ax3.set_title('Zero Coefficients (best iteration)', fontsize=TITLE_SIZE)
                ax3.set_xlabel('Feature')
                ax3.set_ylabel('')
        else:
            ax3.text(0.5, 0.5, "No zero coefficient data available", 
                     ha="center", va="center", transform=ax3.transAxes)
            ax3.axis("off")
    else:
        ax3.text(0.5, 0.5, "No zero coefficient data available", 
                ha="center", va="center", transform=ax3.transAxes)
        ax3.axis("off")
    
    # Adjust layout - different for D/E vs other scenarios
    if scenario_id in ["D", "E"]:
        fig.subplots_adjust(left=0.08, right=0.99, top=0.85, bottom=0.15, wspace=0.25)
    else:
        fig.subplots_adjust(left=0.08, right=0.99, top=0.85, bottom=0.12, wspace=0.35, hspace=0.42)
    
    # Always save the plot when run
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"✓ Final plot saved to {save_path}")
    else:
        default_path = f"scenario_{scenario_id}_final.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"✓ Final plot saved to {default_path}")
    
    plt.close()
    return fig

def main():
    """Main function to generate all final perfect plots."""
    print("=== Generating Final Perfect Plots ===")
    print("This will create the same perfect plots I see in windows!")
    
    # Load data - try synthetic first, fallback to real if synthetic failed
    synthetic_path = '/Users/erdemkoca/Desktop/Uni/Master/Master Thesis/results/synthetic/experiment_results.csv'
    real_path = '/Users/erdemkoca/Desktop/Uni/Master/Master Thesis/results/real/experiment_results.csv'
    
    try:
        df = pd.read_csv(synthetic_path)
        # Check if synthetic data has performance metrics
        if 'f1' not in df.columns:
            print("Synthetic experiments failed (no F1 column found). Switching to real data...")
            df = pd.read_csv(real_path)
            print("Using real experiment data instead.")
        else:
            print("Using synthetic experiment data.")
    except FileNotFoundError:
        print("Synthetic results not found. Using real experiment data...")
        df = pd.read_csv(real_path)
    
    # Filter dynamically from whatever is in the results file
    datasets = sorted(df['dataset_id'].unique().tolist())
    df_filtered = df[df['dataset_id'].isin(datasets)].copy()
    
    print(f"Loaded {len(df_filtered)} experiments")
    print(f"Methods: {df_filtered['model_name'].unique()}")
    print(f"Datasets: {datasets}")
    
    # Generate all plots dynamically
    scenarios = datasets
    saved_plots = []
    
    for scenario in scenarios:
        try:
            save_path = f"scenario_{scenario}_final.png"
            fig = create_final_plot(scenario, df_filtered, save_path=save_path)
            saved_plots.append(save_path)
        except Exception as e:
            print(f"✗ Error creating plot for scenario {scenario}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Successfully generated {len(saved_plots)} final plots!")
    print("Files saved:")
    for plot_path in saved_plots:
        print(f"  - {plot_path}")
    print("\nThese should look exactly like the perfect plots I see in windows!")

if __name__ == "__main__":
    main()
