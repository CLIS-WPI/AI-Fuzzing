# -*- coding: utf-8 -*-
# Analysis Script for O-RAN Fuzzing Results
# Version 9.0: Enhanced for IEEE Publication Quality
# - Refined plot styles, fonts, and colors.
# - Improved layout and legend placement for clarity.
# - Optimized for vector graphics (PDF) and high-resolution output (PNG).

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
import seaborn as sns
import numpy as np
import ast
import os
from scipy.stats import ttest_ind

# --- Constants for easy configuration ---
# NEW: Define output formats as a constant
OUTPUT_FORMATS = ['pdf', 'png']
# NEW: Define a colorblind-friendly palette
COLOR_PALETTE_CATEGORICAL = 'colorblind'
COLOR_PALETTE_SEQUENTIAL = 'viridis' # Viridis is also a good, perceptible choice

def get_ieee_font():
    """Checks for Times New Roman font, falling back to a generic serif font."""
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    return 'Times New Roman' if 'Times New Roman' in available_fonts else 'serif'

def setup_plot_style():
    """Sets a professional, publication-ready style for all plots (IEEE-centric)."""
    # CHANGED: More specific styling for IEEE papers
    sns.set_context("paper", font_scale=1.6) # Slightly larger font scale for readability
    sns.set_style("whitegrid")
    
    plt.rcParams.update({
        'font.family': get_ieee_font(),
        'font.weight': 'normal', # Normal weight is often clearer than bold
        'axes.labelweight': 'bold', # Make axis labels bold for emphasis
        'axes.titleweight': 'bold', # Make titles bold
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.title_fontsize': 13,
        'figure.dpi': 300, # NEW: Set DPI for rasterized images
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'grid.linestyle': '--', # Use dashed lines for the grid
        'grid.linewidth': 0.5, # Finer grid lines
        'lines.linewidth': 2.0, # NEW: Thicker lines for CDFs/line plots
        'lines.markersize': 8, # NEW: Larger markers for scatter plots
    })

def parse_list_column(series):
    """Safely parse stringified list columns."""
    parsed_list = []
    for item in series:
        if isinstance(item, str):
            try:
                parsed_list.append(ast.literal_eval(item))
            except (ValueError, SyntaxError):
                parsed_list.append([])
        elif pd.isna(item):
            parsed_list.append([])
        else:
            parsed_list.append(item)
    return parsed_list

def load_and_preprocess_data(csv_filepath):
    """Loads and prepares the simulation data for analysis."""
    if not os.path.exists(csv_filepath):
        print(f"ERROR: CSV file not found at '{csv_filepath}'")
        return None
    
    print(f"Loading data from {csv_filepath}...")
    df = pd.read_csv(csv_filepath)
    
    for col in ['vulnerabilities', 'ue_locations_str', 'assigned_sinr_list_str']:
        if col in df.columns:
            df[col] = parse_list_column(df[col])
            
    df['vulnerability_count'] = df['vulnerabilities'].apply(len)
    
    if 'algorithm' in df.columns and 'fuzzer_type' in df.columns:
        df['plot_hue'] = df['algorithm'] + ' (' + df['fuzzer_type'] + ')'
        
    return df

def plot_vulnerability_breakdown(df, output_dir):
    """Generate clean multi-facet vulnerability bar plot without legend and with optimal layout."""
    print("Generating optimized vulnerability breakdown plot...")

    required_cols = {'vulnerabilities', 'algorithm', 'fuzzer_type', 'scenario'}
    if not required_cols.issubset(df.columns):
        print("Skipping plot: required columns missing.")
        return

    df = df.explode('vulnerabilities').dropna(subset=['vulnerabilities'])
    if df.empty:
        print("No vulnerability data available.")
        return

    df['vuln_category'] = df['vulnerabilities'].apply(lambda x: str(x).split(':')[0])

    # Global style
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.15)

    # Create FacetGrid plot
    g = sns.catplot(
        data=df,
        kind='count',
        x='algorithm',
        hue='vuln_category',
        row='fuzzer_type',
        col='scenario',
        palette=COLOR_PALETTE_CATEGORICAL,
        height=4,
        aspect=1.4,
        legend=False
    )

    # Axis formatting
    for ax in g.axes.flatten():
        ax.set_xlabel("Algorithm", labelpad=8)
        ax.set_ylabel("Vulnerabilities", labelpad=8)
        ax.tick_params(axis='x', rotation=30)

    # Remove verbose subplot titles
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    # Main title
    g.fig.suptitle("Vulnerability Breakdown by Category", fontsize=18, y=1.03, fontweight='bold')

    # Clean layout
    g.tight_layout()
    g.fig.subplots_adjust(top=0.88, hspace=0.35, wspace=0.25)

    # Save output
    for ext in OUTPUT_FORMATS:
        g.savefig(os.path.join(output_dir, f"vulnerability_breakdown_chart.{ext}"))

    print(f"Chart saved to: {output_dir}")
    plt.close('all')

def plot_vulnerability_legend(df, output_dir):
    """Generates a standalone legend for vulnerability categories."""
    print("Generating standalone legend for vulnerability categories...")

    # استخراج دسته‌های موجود
    vuln_df = df.explode('vulnerabilities').dropna(subset=['vulnerabilities'])
    if vuln_df.empty:
        print("No vulnerabilities to create legend.")
        return

    vuln_df['vuln_category'] = vuln_df['vulnerabilities'].apply(lambda x: str(x).split(':')[0])
    unique_categories = sorted(vuln_df['vuln_category'].unique())

    # ایجاد dummy plot برای ساخت legend
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=col, markersize=10)
        for col in sns.color_palette(COLOR_PALETTE_CATEGORICAL, len(unique_categories))
    ]

    ax.axis('off')  # پنهان‌سازی محورها
    legend = ax.legend(
        handles,
        unique_categories,
        title="Vulnerability Type",
        loc='center',
        ncol=len(unique_categories),
        frameon=False,
        fontsize=12,
        title_fontsize=13
    )

    for ext in OUTPUT_FORMATS:
        legend_path = os.path.join(output_dir, f"vulnerability_legend.{ext}")
        fig.savefig(legend_path, bbox_inches='tight', dpi=300)
    
    print(f"Standalone legend saved to: {output_dir}")
    plt.close(fig)




def plot_performance_tradeoff(df, output_dir):
    """Creates a scatter plot for performance trade-off across all scenarios."""
    print("Generating performance trade-off plot...")
    
    if not all(k in df.columns for k in ['scenario', 'fuzzer_type', 'fairness_index', 'handover_count_iter', 'sinr_5th_percentile']):
        print("Skipping trade-off plot due to missing columns.")
        return
        
    tradeoff_df = df[df['fuzzer_type'] == 'AI']
    if tradeoff_df.empty:
        print("No data for AI Fuzzer scenario to plot trade-off.")
        return
        
    plt.figure(figsize=(10, 7))
    # CHANGED: Using more distinct markers and a sequential palette for size
    markers_list = ['o', 'X', 's', '^', 'D'] # Add more if you have more algorithms
    
    scatter = sns.scatterplot(
        data=tradeoff_df,
        x='fairness_index',
        y='handover_count_iter',
        hue='algorithm',
        size='sinr_5th_percentile',
        sizes=(50, 400),
        style='algorithm',
        palette=COLOR_PALETTE_CATEGORICAL,
        markers=markers_list[:len(tradeoff_df['algorithm'].unique())],
        alpha=0.8,
    )
    
    plt.title('Performance Trade-off Under AI Fuzzer', weight='bold')
    plt.xlabel("Jain's Fairness Index (Higher is Better)")
    plt.ylabel("Average Handover Rate (Lower is Better)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Algorithm & SINR (5th %)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    for ext in OUTPUT_FORMATS:
        plot_path = os.path.join(output_dir, f"performance_tradeoff_scatter.{ext}")
        plt.savefig(plot_path)
    print(f"Saved performance trade-off plot to {output_dir}")
    plt.close()

def plot_combined_cdfs(df, output_dir):
    """Generates a single figure with combined CDFs for key metrics."""
    print("Generating combined CDF plots...")
    metrics_to_plot = {
        'fairness_index': "Fairness Index",
        'handover_count_iter': "Handover Rate",
        'sinr_5th_percentile': r'SINR 5th Percentile [dB]'
    }

    if 'plot_hue' not in df.columns:
        print("Skipping combined CDF plots, 'plot_hue' column missing.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle('Cumulative Distribution of Performance Metrics', fontsize=18, weight='bold')

    for i, (metric, title) in enumerate(metrics_to_plot.items()):
        ax = axes[i]
        sns.ecdfplot(data=df, x=metric, hue='plot_hue', ax=ax, palette=COLOR_PALETTE_SEQUENTIAL)
        ax.set_title(f'({chr(97+i)}) {title}', fontsize=14)
        ax.set_xlabel(title)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    axes[0].set_ylabel('Cumulative Probability')

    # 👇 FIX: Place legend inside figure, centered at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.30)  # enough room for internal legend
    fig.legend(handles, labels,
               title='Algorithm (Fuzzer)',
               loc='lower center',
               bbox_to_anchor=(0.5, 0.02),
               ncol=4,
               frameon=False)

    for ext in OUTPUT_FORMATS:
        plot_path = os.path.join(output_dir, f"combined_cdfs_chart.{ext}")
        plt.savefig(plot_path)
    print(f"Saved combined CDF chart to {output_dir}")
    plt.close(fig)


def main():
    """Main function to run the complete analysis."""
    setup_plot_style()
    
    # Use a variable for the input file path
    csv_filepath = 'fuzzing_results_v25_2_single_gpu_fix.csv'
    
    output_directory = f"analysis_output_{os.path.splitext(os.path.basename(csv_filepath))[0]}"
    os.makedirs(output_directory, exist_ok=True)
    
    df = load_and_preprocess_data(csv_filepath)
    if df is None:
        return

    # --- Generate Plots ---
    plot_vulnerability_breakdown(df, output_dir=output_directory)
    plot_performance_tradeoff(df, output_dir=output_directory)
    plot_combined_cdfs(df, output_dir=output_directory)
    
    # --- Perform and Print Statistical Analysis ---
    print("\n--- Statistical Analysis ---")
    if 'fuzzer_type' in df.columns and 'vulnerability_count' in df.columns:
        ai_vulns = df[df['fuzzer_type'] == 'AI']['vulnerability_count']
        random_vulns = df[df['fuzzer_type'] == 'Random']['vulnerability_count']
        if not ai_vulns.empty and not random_vulns.empty:
            t_stat, p_val_fuzzer = ttest_ind(ai_vulns, random_vulns, equal_var=False, nan_policy='omit')
            print(f"T-test for vulnerability count (AI vs. Random): t-statistic = {t_stat:.4f}, p-value = {p_val_fuzzer:.4g}")

    # --- Generate Summary CSV ---
    summary_path = os.path.join(output_directory, "performance_summary.csv")
    summary_cols = ['scenario', 'fuzzer_type', 'algorithm', 'fairness_index', 
                    'handover_count_iter', 'sinr_5th_percentile', 'vulnerability_count']
    if all(k in df.columns for k in summary_cols):
        summary_df = df.groupby(['scenario', 'fuzzer_type', 'algorithm']).agg(
            avg_fairness=('fairness_index', 'mean'),
            std_fairness=('fairness_index', 'std'), # NEW: Adding standard deviation
            avg_ho_rate=('handover_count_iter', 'mean'),
            std_ho_rate=('handover_count_iter', 'std'), # NEW: Adding standard deviation
            avg_sinr_5th=('sinr_5th_percentile', 'mean'),
            total_vulns=('vulnerability_count', 'sum')
        ).reset_index()
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved detailed summary (with std dev) to {summary_path}")

    print("\nAnalysis complete.")
    print(f"All outputs saved in: {output_directory}")
    plot_vulnerability_legend(df, output_dir=output_directory)

if __name__ == "__main__":
    main()