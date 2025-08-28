# -*- coding: utf-8 -*-
# Analysis Script for O-RAN Fuzzing Results
# Version 10.0: Updated for new data schema with QoE metrics and new plots.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
import seaborn as sns
import numpy as np
import ast
import os
from scipy.stats import ttest_ind

# --- Constants for easy configuration ---
OUTPUT_FORMATS = ['pdf', 'png']
COLOR_PALETTE_CATEGORICAL = 'tab10'
COLOR_PALETTE_SEQUENTIAL = 'viridis'

def get_ieee_font():
    """Checks for Times New Roman font, falling back to a generic serif font."""
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    return 'Times New Roman' if 'Times New Roman' in available_fonts else 'serif'

def setup_plot_style():
    """Sets a professional, publication-ready style for all plots (IEEE-centric)."""
    sns.set_context("paper", font_scale=1.6)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': get_ieee_font(),
        'font.weight': 'normal',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.title_fontsize': 13,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
    })

def load_and_preprocess_data(csv_filepath):
    """Loads and prepares the simulation data for analysis."""
    if not os.path.exists(csv_filepath):
        print(f"ERROR: CSV file not found at '{csv_filepath}'")
        return None
    
    print(f"Loading data from {csv_filepath}...")
    df = pd.read_csv(csv_filepath)
    
    # Parse list-like columns safely
    for col in ['vulnerabilities']:
        if col in df.columns:
            if isinstance(df[col].iloc[0], str):
                try:
                    df[col] = pd.eval(df[col])
                except Exception as e:
                    print(f"Error parsing column {col}: {e}")
                    df[col] = df[col].apply(lambda x: [] if pd.isna(x) else x)
            df['vulnerability_count'] = df['vulnerabilities'].apply(len)

    # Check for NaN or invalid values
    for col in ['jain_fairness_index', 'handover_count_iter', 'throughput_5th_percentile_mbps', 'avg_transmission_time_ms']:
        if col in df.columns:
            invalid_count = df[col].isna().sum() + (df[col] == float('inf')).sum() + (df[col] == -float('inf')).sum()
            if invalid_count > 0:
                print(f"Warning: Found {invalid_count} invalid values in {col}. Replacing with 0.")
                df[col] = df[col].replace([float('inf'), -float('inf')], 0).fillna(0)

    if 'algorithm' in df.columns and 'fuzzer_type' in df.columns:
        df['plot_hue'] = df['algorithm'] + ' (' + df['fuzzer_type'] + ')'

    print(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
    return df

def plot_vulnerability_breakdown(df, output_dir):
    print("Generating vulnerability breakdown plot...")
    vuln_counts = df.explode('vulnerabilities')['vulnerabilities'].value_counts()
    if vuln_counts.empty:
        print("No vulnerabilities found for plotting. Skipping...")
        return
    plt.figure(figsize=(10, 6))
    sns.barplot(x=vuln_counts.values, y=vuln_counts.index, palette=COLOR_PALETTE_CATEGORICAL)
    plt.title('Distribution of Vulnerability Types', weight='bold')
    plt.xlabel('Count')
    plt.ylabel('Vulnerability Type')
    for ext in OUTPUT_FORMATS:
        plt.savefig(os.path.join(output_dir, f'vuln_breakdown.{ext}'))
    plt.close()

# --- MODIFIED: More impactful performance trade-off plot ---
def plot_performance_tradeoff_qoe(df, output_dir):
    """Creates a scatter plot focusing on the QoE trade-off (Fairness vs. Stability vs. Worst-User Throughput)."""
    print("Generating QoE performance trade-off plot...")
    
    # CHANGED: Using new column names
    required_cols = ['scenario', 'fuzzer_type', 'jain_fairness_index', 'handover_count_iter', 'throughput_5th_percentile_mbps']
    if not all(k in df.columns for k in required_cols):
        print("Skipping QoE trade-off plot due to missing columns.")
        return
    tradeoff_df = df[df['fuzzer_type'] == 'AI'].copy()
    if tradeoff_df.empty:
        print("No data for AI Fuzzer to plot QoE trade-off.")
        return
    tradeoff_df = tradeoff_df[tradeoff_df['throughput_5th_percentile_mbps'] > 0]  # Filter invalid data
    if tradeoff_df.empty:
        print("No valid data after filtering for QoE trade-off plot.")
        return
    plt.figure(figsize=(10, 7))
    markers_list = ['o', 'X', 's', '^', 'D']
    sns.scatterplot(
        data=tradeoff_df,
        x='jain_fairness_index',
        y='handover_count_iter',
        hue='algorithm',
        size='throughput_5th_percentile_mbps',
        sizes=(50, 400),
        style='algorithm',
        palette=COLOR_PALETTE_CATEGORICAL,
        markers=markers_list[:len(tradeoff_df['algorithm'].unique())],
        alpha=0.8,
    )
    plt.title('QoE Trade-off: Fairness vs. Handovers vs. Worst-User Throughput', weight='bold')
    plt.xlabel("Jain's Fairness Index (Higher is Better)")
    plt.ylabel("Handover Rate (Lower is Better)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Algorithm & Throughput (Mbps)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    for ext in OUTPUT_FORMATS:
        plt.savefig(os.path.join(output_dir, f'qoe_tradeoff_scatter.{ext}'))
    plt.close()

# --- MODIFIED: More comprehensive CDF plot ---
def plot_qoe_cdfs(df, output_dir):
    """Generates a 2x2 figure with CDFs for key QoE metrics."""
    print("Generating combined QoE CDF plots...")
    
    # CHANGED: Using new, more impactful metrics
    metrics_to_plot = {
        'jain_fairness_index': "Jain's Fairness Index",
        'handover_count_iter': "Handover Rate",
        'throughput_5th_percentile_mbps': r'5th Percentile Throughput (Mbps)',
        'avg_transmission_time_ms': 'Avg. Transmission Time (ms)'
    }

    if 'plot_hue' not in df.columns or not all(k in df.columns for k in metrics_to_plot.keys()):
        print("Skipping combined QoE CDF plots, required columns missing.")
        return

    df_filtered = df.copy()
    for metric in metrics_to_plot.keys():
        df_filtered = df_filtered[df_filtered[metric].notna() & (df_filtered[metric] != float('inf')) & (df_filtered[metric] != -float('inf'))]
    if df_filtered.empty:
        print("No valid data for CDF plots after filtering.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cumulative Distribution of Key Performance and QoE Metrics', fontsize=18, weight='bold')
    axes_flat = axes.flatten()
    for i, (metric, title) in enumerate(metrics_to_plot.items()):
        ax = axes_flat[i]
        sns.ecdfplot(data=df_filtered, x=metric, hue='plot_hue', ax=ax, palette=COLOR_PALETTE_CATEGORICAL)
        ax.set_title(f'({chr(97+i)}) CDF of {title}', fontsize=14)
        ax.set_xlabel(title)
        ax.set_ylabel('Cumulative Probability')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        if metric == 'throughput_5th_percentile_mbps':
            ax.axvline(x=1.0, color='r', linestyle='--', label='QoE Threshold (1 Mbps)')
            ax.legend()
        if ax.get_legend() is not None and metric != 'throughput_5th_percentile_mbps':
            ax.get_legend().remove()
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])
    fig.legend(handles, labels, title='Algorithm (Fuzzer)', loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4, frameon=False)
    for ext in OUTPUT_FORMATS:
        plt.savefig(os.path.join(output_dir, f'qoe_cdfs_chart.{ext}'))
    plt.close(fig)

# --- NEW PLOT: Comparing different fairness metrics ---
def plot_fairness_comparison(df, output_dir):
    """Creates a bar plot to compare different fairness metrics across algorithms."""
    print("Generating fairness metrics comparison plot...")
    
    fairness_metrics = ['jain_fairness_index', 'alpha_fairness_a1', 'alpha_fairness_a2']
    if not all(k in df.columns for k in fairness_metrics):
        print("Skipping fairness comparison plot due to missing columns.")
        return

    df_filtered = df[fairness_metrics + ['algorithm', 'scenario']].dropna()
    if df_filtered.empty:
        print("No valid data for fairness comparison plot.")
        return
    df_melted = df_filtered.melt(id_vars=['algorithm', 'scenario'], value_vars=fairness_metrics, 
                                 var_name='Fairness Metric', value_name='Score')
    plt.figure(figsize=(14, 8))
    g = sns.catplot(
        data=df_melted,
        x='Fairness Metric',
        y='Score',
        hue='algorithm',
        col='scenario',
        kind='bar',
        palette=COLOR_PALETTE_CATEGORICAL,
        height=6,
        aspect=1.2
    )
    g.fig.suptitle('Comparison of Fairness Metrics Across Algorithms and Scenarios', y=1.03, weight='bold')
    g.set_xticklabels(rotation=30, ha='right')
    g.set_axis_labels("Fairness Metric Type", "Average Score")
    for ext in OUTPUT_FORMATS:
        plt.savefig(os.path.join(output_dir, f'fairness_comparison_barchart.{ext}'))
    print(f"Saved fairness comparison barchart to {output_dir}")
    plt.close('all')

# --- NEW PLOT: Comparing Algorithm Performance Across Scenarios ---
def plot_scenario_comparison(df, output_dir):
    """Creates a bar plot to compare a key metric for each algorithm across all scenarios."""
    print("Generating scenario performance comparison plot...")
    
    key_metric = 'throughput_5th_percentile_mbps'
    if key_metric not in df.columns:
        print(f"Skipping scenario comparison plot, missing key metric: {key_metric}")
        return

    # We are interested in the performance under the most stressful fuzzer
    df_ai_fuzzer = df[df['fuzzer_type'] == 'AI'].copy()
    df_ai_fuzzer = df_ai_fuzzer[df_ai_fuzzer[key_metric].notna() & (df_ai_fuzzer[key_metric] > 0)]
    if df_ai_fuzzer.empty:
        print("No valid data for scenario comparison plot.")
        return
    plt.figure(figsize=(14, 8))
    g = sns.catplot(
        data=df_ai_fuzzer,
        x='scenario',
        y=key_metric,
        hue='algorithm',
        kind='bar',
        palette=COLOR_PALETTE_CATEGORICAL,
        height=7,
        aspect=1.8,
        legend=False
    )
    g.fig.suptitle('Algorithm Robustness Across Scenarios (AI Fuzzer)', y=1.03, weight='bold')
    g.set_axis_labels("Scenario", "5th Percentile Throughput (Mbps)")
    g.set_xticklabels(rotation=30, ha='right')
    plt.legend(title='Algorithm', loc='upper right')
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='QoE Threshold (1 Mbps)')
    for ext in OUTPUT_FORMATS:
        plt.savefig(os.path.join(output_dir, f'scenario_comparison_barchart.{ext}'))
    plt.close('all')
    
def main():
    """Main function to run the complete analysis."""
    setup_plot_style()
    
    # Use only the correct CSV file name
    csv_filepath = 'fuzzing_results_v25_4_runtime_opt.csv'
    output_directory = f"analysis_output_{os.path.splitext(os.path.basename(csv_filepath))[0]}"
    os.makedirs(output_directory, exist_ok=True)
    
    df = load_and_preprocess_data(csv_filepath)
    if df is None:
        return

    # Generate Plots
    # plot_network_topology(output_dir=output_directory)  # Uncomment if you have this function
    plot_vulnerability_breakdown(df.copy(), output_dir=output_directory)
    plot_performance_tradeoff_qoe(df.copy(), output_dir=output_directory)
    plot_qoe_cdfs(df.copy(), output_dir=output_directory)
    plot_fairness_comparison(df.copy(), output_dir=output_directory)
    plot_scenario_comparison(df.copy(), output_dir=output_directory)

    # --- Statistical Analysis ---
    print("\n--- Statistical Analysis ---")
    # (Existing statistical analysis code can be kept here)

    # --- Generate Summary CSV (MODIFIED) ---
    summary_path = os.path.join(output_directory, "qoe_performance_summary.csv")
    summary_cols = [
        'scenario', 'fuzzer_type', 'algorithm', 
        'jain_fairness_index', 'handover_count_iter', 
        'throughput_5th_percentile_mbps', 'vulnerability_count', 'vulnerabilities'
    ]
    if all(k in df.columns for k in summary_cols):
        summary_df = df.groupby(['scenario', 'fuzzer_type', 'algorithm']).agg(
            avg_jain_fairness=('jain_fairness_index', 'mean'),
            std_jain_fairness=('jain_fairness_index', 'std'),
            avg_ho_rate=('handover_count_iter', 'mean'),
            std_ho_rate=('handover_count_iter', 'std'),
            avg_throughput_5th=('throughput_5th_percentile_mbps', 'mean'),
            std_throughput_5th=('throughput_5th_percentile_mbps', 'std'),
            total_vulns=('vulnerability_count', 'sum'),
            vulnerabilities=('vulnerabilities', list)
        ).reset_index()
        # Add vulnerability diversity metric
        vuln_diversity = lambda x: len(set([item for sublist in x for item in sublist]))
        summary_df['vuln_diversity'] = df.groupby(['scenario', 'fuzzer_type', 'algorithm'])['vulnerabilities'].agg(vuln_diversity).values
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved detailed QoE summary to {summary_path}")

    print("\nAnalysis complete.")
    print(f"All outputs saved in: {output_directory}")

if __name__ == "__main__":
    main()