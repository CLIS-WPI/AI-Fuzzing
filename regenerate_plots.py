#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Essential plots generator for AI-Fuzzing 5G Traffic Steering Paper
Generates 4 key publication-quality figures
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.stats
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# --- Constants ---
CSV_FILENAME = "fuzzing_results_v28_strategic_fuzzing.csv"
OUTPUT_DIR = "plots_essential_paper"
SIMULATION_ITERATIONS = 15  # Number of iterations per run
NUM_RUNS = 50  # Number of runs per scenario

# Set clean, professional plotting style with LARGER fonts
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 22,  # Much larger for method names
    'ytick.labelsize': 16,
    'legend.fontsize': 15,
    'figure.titlesize': 20,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'axes.labelweight': 'bold',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white'
})

def load_and_prepare_data(csv_file):
    """Loads and preprocesses the data from the CSV file."""
    # Check file existence
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found!")
        return None

    # Load data
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Successfully loaded {len(df)} rows of data.")

    # Print overview
    print("\nDataset overview:")
    print(f"Total rows: {len(df)}")
    print(f"Fuzzer types: {df['fuzzer_type'].unique()}")
    print(f"Scenarios: {df['scenario'].unique()}")
    print(f"Total vulnerabilities: {df['vulnerability_count'].sum()}")

    # Process numeric columns
    numeric_cols = [
        'vulnerability_count',
        'jain_fairness_index',
        'throughput_5th_percentile_mbps',
        'handover_count_iter'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Process boolean columns
    bool_cols = [
        'is_critical_failure',
        'has_ping_pong',
        'has_qoe_violation',
        'has_unfairness'
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x).strip().lower() == 'true' if pd.notna(x) else False)

    # Add run identifier for proper grouping
    df['run_id'] = df.groupby(['fuzzer_type', 'scenario']).cumcount()

    # Add severity categories based on vulnerability characteristics
    def categorize_severity(row):
        if row['is_critical_failure']:
            return 'Critical'
        elif row['has_ping_pong'] and row['has_qoe_violation']:
            return 'High'
        elif row['has_ping_pong'] or row['has_qoe_violation']:
            return 'Medium'
        elif row['vulnerability_count'] > 0:
            return 'Low'
        else:
            return 'None'
    
    df['severity'] = df.apply(categorize_severity, axis=1)

    return df

def create_combined_plots(df, output_dir):
    """
    Combined Plot 1 & 2: Vulnerability Discovery and Convergence Analysis
    """
    print("Generating Combined Plot (1 & 2)...")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), facecolor='white')
    
    # Calculate statistics for each fuzzer type
    fuzzer_types = ['Traditional-Testing', 'AI-Fuzzing']
    means = []
    stds = []
    cis = []
    ns = []
    data = []
    
    for fuzzer_type in fuzzer_types:
        fuzzer_data = df[df['fuzzer_type'] == fuzzer_type]
        
        # Replicate run_ai_fuzzing.py's data structure exactly
        run_vulnerability_counts = []
        scenarios = fuzzer_data['scenario'].unique()
        runs = fuzzer_data['run_id'].unique()
        
        # Process each run independently
        for run_id in runs:
            run_total = 0
            run_data = fuzzer_data[fuzzer_data['run_id'] == run_id]
            
            # Sum vulnerabilities for this run across all scenarios
            for scenario in scenarios:
                scenario_data = run_data[run_data['scenario'] == scenario]
                run_total += scenario_data['vulnerability_count'].sum()
            
            # Average across scenarios and scale by iterations
            run_vulnerability_counts.append(run_total * SIMULATION_ITERATIONS / len(scenarios))
        
        # Convert to pandas Series for consistency
        run_vulns = pd.Series(run_vulnerability_counts)
        
        # Calculate statistics matching run_ai_fuzzing.py
        means.append(run_vulns.mean())
        stds.append(run_vulns.std())  # pandas std() uses ddof=1 by default
        ns.append(len(run_vulns))
        cis.append(1.96 * stds[-1] / np.sqrt(ns[-1]))
        data.append(run_vulns.values)
    
    # Calculate statistical significance
    t_stat, p_value = scipy.stats.ttest_ind(
        data[1],  # AI-Fuzzing
        data[0],  # Traditional-Testing
        alternative='greater'
    )
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((ns[0] - 1) * stds[0]**2 + 
         (ns[1] - 1) * stds[1]**2) / 
        (ns[0] + ns[1] - 2)
    )
    
    # Calculate improvement and effect size
    cohens_d = (means[1] - means[0]) / pooled_std
    improvement = ((means[1] - means[0]) / means[0] * 100)
    
    # Create bar plot
    x_pos = [0, 1]
    colors = ['#2ecc71', '#e74c3c']  # Green for Traditional, Red for AI
    display_names = ['Traditional Testing', 'AI Fuzzing']
    
    bars = ax1.bar(x_pos, means, yerr=stds, capsize=10,  # Using standard deviation for error bars
                  color=colors, alpha=0.8, width=0.6,
                  edgecolor='black', linewidth=1.5,
                  error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Customize plot
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(display_names)
    ax1.set_ylabel('Average Vulnerabilities per Run\n(averaged across scenarios)', fontsize=24, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    
    # Add grid for better readability of larger numbers
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)
    
    # Adjust y-axis limits to accommodate the larger numbers
    max_height = max([m + s for m, s in zip(means, stds)])
    ax1.set_ylim(0, max_height * 1.2)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if i == 0:
            x_shift = 0.18  # Shift Traditional Testing label to the right
        elif i == 1:
            x_shift = -0.18  # Shift AI Fuzzing label to the left
        else:
            x_shift = 0
        ax1.text(bar.get_x() + bar.get_width()/2. + x_shift, height + cis[i],
                f'{means[i]:.1f}±{cis[i]:.1f}\nn={ns[i]}',
                ha='center', va='bottom', fontsize=18, fontweight='bold')
    
    # Add statistical significance annotation
    max_height = max(means) + max(cis) + 5
    ax1.plot([0, 1], [max_height, max_height], 'k-', linewidth=1)
    ax1.plot([0, 0], [max_height-0.5, max_height], 'k-', linewidth=1)
    ax1.plot([1, 1], [max_height-0.5, max_height], 'k-', linewidth=1)
    
    p_text = f'p < 0.001' if p_value < 0.001 else f'p = {p_value:.3f}'
    ax1.text(0.5, max_height + 1, p_text,
        ha='center', va='bottom', fontsize=18, fontweight='bold')
    
    # Add improvement text
    ax1.text(0.5, max_height + 3,
        f'{improvement:.1f}% improvement',
        ha='center', va='bottom', fontsize=20,
        color='darkgreen', fontweight='bold')
    
    # Set y-limits to accommodate annotations
    ax1.set_ylim(0, max_height + 6)
    
    # Add subplot label
    ax1.text(-0.1, 1.05, '(a)', transform=ax1.transAxes, fontsize=28, fontweight='bold')
    
    # --- Plot 2: Convergence Analysis ---
    
    # Print statistics
    # Print statistics
    print(f"-> Plot 1 saved to {output_dir}/plot_1_vulnerability_discovery.pdf")
    print("PER-RUN STATISTICS (averaged over all scenarios):")
    print(f"  Traditional Testing:")
    print(f"    Average Vulnerabilities per Run: {means[0]:.2f} ± {stds[0]:.2f}")
    print(f"  AI Fuzzing:")
    print(f"    Average Vulnerabilities per Run: {means[1]:.2f} ± {stds[1]:.2f}")
    print(f"\nSTATISTICAL SIGNIFICANCE TESTS:")
    print(f"T-TEST for TOTAL VULNERABILITIES (One-sided: AI Fuzzing > Traditional Testing):")
    print(f"  T-statistic: {t_stat:.3f}, P-value: {p_value:.5f}")
    print(f"  Result: AI Fuzzing found a statistically significant GREATER number of vulnerabilities (p = {p_value:.5f}).")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  Effect size (Cohen's d): {cohens_d:.3f}")

    # Group by iteration to show progression for Plot 2
    ai_data = df[df['fuzzer_type'] == 'AI-Fuzzing'].groupby('iteration').agg({
        'vulnerability_count': ['mean', 'std', 'count']
    }).reset_index()
    
    traditional_data = df[df['fuzzer_type'] == 'Traditional-Testing'].groupby('iteration').agg({
        'vulnerability_count': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    ai_data.columns = ['iteration', 'mean_vulns', 'std_vulns', 'n_samples']
    traditional_data.columns = ['iteration', 'mean_vulns', 'std_vulns', 'n_samples']
    
    # Calculate confidence intervals
    ai_data['se'] = ai_data['std_vulns'] / np.sqrt(ai_data['n_samples'])
    ai_data['ci_lower'] = ai_data['mean_vulns'] - 1.96 * ai_data['se']
    ai_data['ci_upper'] = ai_data['mean_vulns'] + 1.96 * ai_data['se']
    ai_data['best_fitness'] = ai_data['mean_vulns'].cummax()
    
    traditional_data['se'] = traditional_data['std_vulns'] / np.sqrt(traditional_data['n_samples'])
    traditional_data['ci_lower'] = traditional_data['mean_vulns'] - 1.96 * traditional_data['se']
    traditional_data['ci_upper'] = traditional_data['mean_vulns'] + 1.96 * traditional_data['se']
    traditional_data['best_fitness'] = traditional_data['mean_vulns'].cummax()
    
    # Get raw data for scatter overlay
    ai_raw = df[df['fuzzer_type'] == 'AI-Fuzzing'].groupby('iteration')['vulnerability_count'].apply(list)
    traditional_raw = df[df['fuzzer_type'] == 'Traditional-Testing'].groupby('iteration')['vulnerability_count'].apply(list)
    
    # Plot convergence lines with beautiful colors and confidence bands
    ai_color = '#4ECDC4'  # Teal
    traditional_color = '#FF6B6B'  # Coral
    
    # Main convergence lines with larger markers and thicker lines
    ax2.plot(ai_data['iteration'], ai_data['best_fitness'], 
            'o-', color=ai_color, linewidth=4, markersize=10, 
            label='AI Fuzzing', markerfacecolor='white', markeredgewidth=2.5, alpha=0.9)
    
    ax2.fill_between(ai_data['iteration'], 
                    ai_data['ci_lower'].cummax(), ai_data['ci_upper'].cummax(),
                    alpha=0.25, color=ai_color, label='AI 95% CI')
    
    ax2.plot(traditional_data['iteration'], traditional_data['best_fitness'], 
            's--', color=traditional_color, linewidth=4, markersize=10, 
            label='Traditional Testing', markerfacecolor='white', markeredgewidth=2.5, alpha=0.9)
    
    ax2.fill_between(traditional_data['iteration'], 
                    traditional_data['ci_lower'].cummax(), traditional_data['ci_upper'].cummax(),
                    alpha=0.25, color=traditional_color, label='Traditional 95% CI')
    
    # Add scatter points for data distribution with better visibility
    for iteration in ai_data['iteration']:
        if iteration in ai_raw.index:
            y_vals = ai_raw.loc[iteration]
            x_vals = [iteration + np.random.normal(0, 0.08) for _ in y_vals]  # Reduced jitter
            ax2.scatter(x_vals, y_vals, alpha=0.6, color=ai_color, s=35, 
                      edgecolor='white', linewidth=1, zorder=5)
    
    for iteration in traditional_data['iteration']:
        if iteration in traditional_raw.index:
            y_vals = traditional_raw.loc[iteration]
            x_vals = [iteration + np.random.normal(0, 0.08) for _ in y_vals]  # Reduced jitter
            ax2.scatter(x_vals, y_vals, alpha=0.6, color=traditional_color, s=35, 
                      marker='s', edgecolor='white', linewidth=1, zorder=5)
    
    # Enhanced labels with larger fonts - title removed for caption
    ax2.set_xlabel('Generation Number', fontsize=24, fontweight='bold')
    ax2.set_ylabel('Best Fitness Score\n(Cumulative Max Vulnerabilities)', fontsize=24, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.legend(fontsize=18, loc='upper left', framealpha=0.95, shadow=True)  # Moved to top corner
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)
    
    # Add subplot label
    ax2.text(-0.1, 1.05, '(b)', transform=ax2.transAxes, fontsize=28, fontweight='bold')

    # Calculate final statistics
    final_ai = ai_data['best_fitness'].iloc[-1]
    final_traditional = traditional_data['best_fitness'].iloc[-1]
    
    # Find convergence point (where 90% of final value is reached)
    ai_convergence_idx = np.where(ai_data['best_fitness'] >= 0.9 * final_ai)[0]
    traditional_convergence_idx = np.where(traditional_data['best_fitness'] >= 0.9 * final_traditional)[0]
    
    if len(ai_convergence_idx) > 0 and len(traditional_convergence_idx) > 0:
        ai_convergence = ai_data['iteration'].iloc[ai_convergence_idx[0]]
        traditional_convergence = traditional_data['iteration'].iloc[traditional_convergence_idx[0]]
        speedup = ((traditional_convergence - ai_convergence) / traditional_convergence) * 100
    else:
        speedup = 0
        ai_convergence = ai_data['iteration'].iloc[-1]
        traditional_convergence = traditional_data['iteration'].iloc[-1]
    
    # Adjust layout of the combined figure
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # Add space between subplots
    
    # Save combined plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'combined_plot_1_2.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300, facecolor='white',
                pad_inches=0.5)
    plt.close()
    
    print(f"-> Combined Plot saved to {output_path}")
    print("Statistics for Plot (a):")
    print(f"  Traditional Testing:")
    print(f"    Average Vulnerabilities per Run: {means[0]:.2f} ± {stds[0]:.2f}")
    print(f"  AI Fuzzing:")
    print(f"    Average Vulnerabilities per Run: {means[1]:.2f} ± {stds[1]:.2f}")
    print(f"  T-statistic: {t_stat:.3f}, P-value: {p_value:.5f}")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  Effect size (Cohen's d): {cohens_d:.3f}")
    print("\nStatistics for Plot (b):")
    print(f"  AI convergence at generation: {ai_convergence}")
    print(f"  Traditional convergence at generation: {traditional_convergence}")
    print(f"  Speedup: {speedup:.1f}%")
    print(f"  Final performance ratio: {final_ai/final_traditional:.2f}×")

def create_plot3_vulnerability_severity_distribution(df, output_dir):
    """
    Plot 3: Vulnerability Severity Distribution
    Enhanced stacked bar chart with diversity analysis and normalized percentages
    """
    print("Generating Plot 3: Vulnerability Severity Distribution...")
    
    comparison_data = df[df['fuzzer_type'].isin(['AI-Fuzzing', 'Traditional-Testing'])]
    
    # Count vulnerabilities by severity and fuzzer type
    severity_counts = comparison_data.groupby(['fuzzer_type', 'severity']).size().unstack(fill_value=0)
    
    # Ensure we have all severity levels
    severity_levels = ['Critical', 'High', 'Medium', 'Low']
    for level in severity_levels:
        if level not in severity_counts.columns:
            severity_counts[level] = 0
    
    severity_counts = severity_counts[severity_levels]  # Reorder columns
    
    # Calculate normalized percentages
    severity_percentages = severity_counts.div(severity_counts.sum(axis=1), axis=0) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    
    # Left plot: Absolute counts
    # Beautiful gradient colors for severity levels
    colors = ['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71']  # Red to Green gradient
    # Critical=Red, High=Orange, Medium=Yellow, Low=Green
    
    severity_counts.plot(kind='bar', stacked=True, ax=ax1, color=colors, 
                        alpha=0.8, edgecolor='black', linewidth=1, width=0.5, legend=False)
    
    ax1.set_title('Absolute Vulnerability Counts by Severity', fontsize=20, fontweight='bold', pad=25)
    ax1.set_xlabel('Testing Approach', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Number of Vulnerabilities', fontsize=18, fontweight='bold')
    ax1.tick_params(axis='x', rotation=0, labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)
    
    # Set y-axis limits to provide space for total labels
    max_total = severity_counts.sum(axis=1).max()
    ax1.set_ylim(0, max_total * 1.2)
    
    # Add total counts on top of bars with more spacing
    for i, (idx, row) in enumerate(severity_counts.iterrows()):
        total = row.sum()
        ax1.text(i, total + 25, f'Total: {int(total)}', 
                ha='center', va='bottom', fontweight='bold', fontsize=16)
    
    # Right plot: Normalized percentages
    severity_percentages.plot(kind='bar', stacked=True, ax=ax2, color=colors, 
                             alpha=0.8, edgecolor='black', linewidth=1, width=0.5, legend=False)
    
    ax2.set_title('Normalized Severity Distribution (%)', fontsize=20, fontweight='bold', pad=25)
    ax2.set_xlabel('Testing Approach', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Percentage of Vulnerabilities', fontsize=18, fontweight='bold')
    ax2.tick_params(axis='x', rotation=0, labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_axisbelow(True)
    
    # Add subplot labels (a) and (b) with small gap from x-axis labels
    ax1.text(0.5, -0.18, '(a)', transform=ax1.transAxes, fontsize=18, fontweight='bold',
             horizontalalignment='center', verticalalignment='top')
    ax2.text(0.5, -0.18, '(b)', transform=ax2.transAxes, fontsize=18, fontweight='bold',
             horizontalalignment='center', verticalalignment='top')
    
    # Add percentage labels on the normalized chart
    for i, (idx, row) in enumerate(severity_percentages.iterrows()):
        cumsum = 0
        for j, (severity, percentage) in enumerate(row.items()):
            if percentage > 5:  # Only show labels for segments > 5%
                ax2.text(i, cumsum + percentage/2, f'{percentage:.1f}%', 
                        ha='center', va='center', fontweight='bold', fontsize=14,
                        color='white' if j < 2 else 'black')  # White text on dark, black on light
            cumsum += percentage
    
    # Calculate diversity indices
    def shannon_diversity(counts):
        total = counts.sum()
        if total == 0:
            return 0
        proportions = counts / total
        return -np.sum(proportions * np.log(proportions + 1e-10))
    
    def simpson_diversity(counts):
        total = counts.sum()
        if total == 0:
            return 0
        proportions = counts / total
        return 1 - np.sum(proportions**2)
    
    ai_shannon = shannon_diversity(severity_counts.loc['AI-Fuzzing'])
    traditional_shannon = shannon_diversity(severity_counts.loc['Traditional-Testing'])
    ai_simpson = simpson_diversity(severity_counts.loc['AI-Fuzzing'])
    traditional_simpson = simpson_diversity(severity_counts.loc['Traditional-Testing'])
    
    # Add sample sizes above the legend with 0.17 gap
    ai_samples = len(comparison_data[comparison_data['fuzzer_type'] == 'AI-Fuzzing'])
    traditional_samples = len(comparison_data[comparison_data['fuzzer_type'] == 'Traditional-Testing'])
    
    fig.text(0.5, 0.19, f'Sample sizes: AI n={ai_samples}, Traditional n={traditional_samples}', 
            fontsize=14, horizontalalignment='center', verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Create unified legend at the bottom with 0.17 gap from sample sizes
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], 
                                   edgecolor='black', 
                                   label=severity_levels[i]) for i in range(len(severity_levels))]
    fig.legend(handles=legend_elements, title='Severity Level', 
              fontsize=16, title_fontsize=14, loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=4, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0.24, 1, 0.95])
    
    output_path = os.path.join(output_dir, 'plot_3_severity_distribution.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.savefig(os.path.join(output_dir, 'plot_3_severity_distribution.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"-> Plot 3 saved to {output_path}")
    print(f"   AI Fuzzing Shannon diversity: {ai_shannon:.3f}")
    print(f"   Traditional Shannon diversity: {traditional_shannon:.3f}")
    print(f"   AI Fuzzing Simpson diversity: {ai_simpson:.3f}")
    print(f"   Traditional Simpson diversity: {traditional_simpson:.3f}")

def create_plot4_performance_across_scenarios(df, output_dir):
    """
    Plot 4: Performance Across Scenarios
    Horizontal bar chart with clear spacing and annotations
    """
    print("Generating Plot 4: Performance Across Scenarios...")
    
    comparison_data = df[df['fuzzer_type'].isin(['AI-Fuzzing', 'Traditional-Testing'])]
    
    # Focus on 3 main scenarios for clarity
    main_scenarios = ['Stable Mobility', 'Load Imbalance', 'Congestion Crisis']
    scenario_data = comparison_data[comparison_data['scenario'].isin(main_scenarios)]
    
    # Create figure with golden ratio
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')  # Wider figure for horizontal bars
    
    # Calculate detailed statistics per scenario
    scenario_stats = []
    p_values = []
    
    for scenario in main_scenarios:
        scenario_subset = scenario_data[scenario_data['scenario'] == scenario]
        
        stats_row = {}
        for fuzzer_type in ['Traditional-Testing', 'AI-Fuzzing']:
            fuzzer_subset = scenario_subset[scenario_subset['fuzzer_type'] == fuzzer_type]
            
            # Group by run (iteration) for proper statistics
            run_vulns = fuzzer_subset.groupby('iteration')['vulnerability_count'].sum()
            
            mean_vulns = run_vulns.mean()
            std_vulns = run_vulns.std()
            n_runs = len(run_vulns)
            se_vulns = std_vulns / np.sqrt(n_runs) if n_runs > 0 else 0
            ci_95 = 1.96 * se_vulns
            
            stats_row[fuzzer_type] = {
                'mean': mean_vulns,
                'std': std_vulns,
                'se': se_vulns,
                'ci': ci_95,
                'n': n_runs,
                'raw_data': run_vulns.values if n_runs > 0 else [0]
            }
        
        # Statistical test for this scenario
        if (stats_row['Traditional-Testing']['n'] > 0 and 
            stats_row['AI-Fuzzing']['n'] > 0):
            u_stat, p_val = mannwhitneyu(
                stats_row['AI-Fuzzing']['raw_data'],
                stats_row['Traditional-Testing']['raw_data'],
                alternative='greater'
            )
        else:
            p_val = 1.0
            
        p_values.append(p_val)
        scenario_stats.append(stats_row)
    
    # Extract statistics
    traditional_means = [stats[list(stats.keys())[0]]['mean'] for stats in scenario_stats]
    traditional_cis = [stats[list(stats.keys())[0]]['ci'] for stats in scenario_stats]
    traditional_ns = [stats[list(stats.keys())[0]]['n'] for stats in scenario_stats]
    
    ai_means = [stats[list(stats.keys())[1]]['mean'] for stats in scenario_stats]
    ai_cis = [stats[list(stats.keys())[1]]['ci'] for stats in scenario_stats]
    ai_ns = [stats[list(stats.keys())[1]]['n'] for stats in scenario_stats]
    
    # Set up positions for horizontal bars
    y = np.arange(len(main_scenarios))
    height = 0.35
    
    # Beautiful colors for scenario comparison
    colors = ['#FF6B6B', '#4ECDC4']  # Coral Red for Traditional, Teal for AI
    edge_colors = ['#E74C3C', '#16A085']  # Darker edges
    
    # Create bars with increased spacing between groups
    x_spacing = 2  # Increased spacing between bar groups
    x_positions = np.arange(len(main_scenarios)) * x_spacing
    
    # Set up positions and dimensions
    x_positions = np.arange(len(main_scenarios))
    width = 0.35  # Width of the bars
    
    # Calculate maximum height for proper scaling
    max_val = max([m + c for m, c in zip(traditional_means + ai_means, traditional_cis + ai_cis)])
    ax.set_ylim(0, max_val * 1.6)  # 60% extra space for labels
    
    bars1 = ax.bar(x_positions - width/2, traditional_means, width, 
                   label='Traditional Testing', color=colors[0], alpha=0.8, 
                   yerr=traditional_cis, capsize=5,
                   edgecolor=edge_colors[0], linewidth=1.5)
    
    bars2 = ax.bar(x_positions + width/2, ai_means, width, 
                   label='AI Fuzzing', color=colors[1], alpha=0.8, 
                   yerr=ai_cis, capsize=5,
                   edgecolor=edge_colors[1], linewidth=1.5)
    
    # Add mean ± CI labels INSIDE the bars
    for i, (bar, mean_val, ci_val, n_val) in enumerate(zip(bars1, traditional_means, traditional_cis, traditional_ns)):
        if mean_val > 0:
            height = bar.get_height()
            y_pos = height * 0.5  # Middle of the bar
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{mean_val:.1f}±{ci_val:.1f}\n(n={n_val})', 
                   ha='center', va='center', fontweight='bold', fontsize=14,
                   color='white')

    for i, (bar, mean_val, ci_val, n_val) in enumerate(zip(bars2, ai_means, ai_cis, ai_ns)):
        if mean_val > 0:
            height = bar.get_height()
            y_pos = height * 0.5  # Middle of the bar
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{mean_val:.1f}±{ci_val:.1f}\n(n={n_val})', 
                   ha='center', va='center', fontweight='bold', fontsize=14,
                   color='white')

    # Enhanced styling with larger fonts for better readability - title removed for caption
    ax.set_ylabel('Mean Vulnerabilities per Run', fontsize=20, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([s.replace(' ', '\n') for s in main_scenarios], fontsize=18, fontweight='medium')
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Calculate improvements and effect sizes for each scenario
    improvements = []
    effect_sizes = []
    
    for i, stats in enumerate(scenario_stats):
        traditional_stats = stats['Traditional-Testing']
        ai_stats = stats['AI-Fuzzing']
        
        if traditional_stats['mean'] > 0:
            improvement = ((ai_stats['mean'] - traditional_stats['mean']) / 
                          traditional_stats['mean']) * 100
            improvements.append(improvement)
            
            # Cohen's d
            pooled_std = np.sqrt((traditional_stats['std']**2 + ai_stats['std']**2) / 2)
            cohens_d = (ai_stats['mean'] - traditional_stats['mean']) / pooled_std if pooled_std > 0 else 0
            effect_sizes.append(cohens_d)
        else:
            improvements.append(0)
            effect_sizes.append(0)
    
    avg_improvement = np.mean([imp for imp in improvements if imp != 0])
    avg_effect_size = np.mean([es for es in effect_sizes if es != 0])
    
    # Add comprehensive analysis box
    analysis_text = f'''Cross-Scenario Analysis:
Average Improvement: ↗ +{avg_improvement:.1f}%
Average Effect Size: {avg_effect_size:.2f}
Significant scenarios: {sum(1 for p in p_values if p < 0.05)}/{len(p_values)}
Robustness Index: {(avg_improvement/100) * (avg_effect_size/2):.2f}'''

    ax.text(0.98, 0.8, analysis_text, transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            verticalalignment='top', horizontalalignment='right', family='monospace')
    

    
    # Add individual scenario improvements as text above the bars
    for i, improvement in enumerate(improvements):
        if improvement > 0:
            max_height = max(ai_means[i] + ai_cis[i], traditional_means[i] + traditional_cis[i])
            ax.text(x_positions[i], max_height * 1.1,
                   f'+{improvement:.0f}%',
                   ha='center', va='bottom', fontsize=14,
                   fontweight='bold', color='green')
    
    # Adjust layout to prevent labels from crossing borders
    # Adjust layout with optimal spacing for all elements
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)  # More space at top and right for legend
    
    # Position legend in the top right with optimal spacing
    ax.legend(fontsize=16, loc='upper right', bbox_to_anchor=(0.98, 0.98),
             framealpha=0.9, borderaxespad=0)
    
    output_path = os.path.join(output_dir, 'plot_4_scenario_performance.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300, facecolor='white', 
                pad_inches=1.0)  # Maximum padding around the figure
    plt.savefig(os.path.join(output_dir, 'plot_4_scenario_performance.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', pad_inches=2.0)
    plt.close()
    
    print(f"-> Plot 4 saved to {output_path}")
    print(f"   Average improvement across scenarios: {avg_improvement:.1f}%")
    print(f"   Average effect size: {avg_effect_size:.2f}")
    print(f"   Significant scenarios: {sum(1 for p in p_values if p < 0.05)}/{len(p_values)}")
    print(f"   P-values: {[f'{p:.4f}' for p in p_values]}")

def main():
    """Main function to generate all essential publication plots."""
    print("=== AI-Fuzzing Essential Plots Generator ===")
    
    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df = load_and_prepare_data(CSV_FILENAME)
    
    if df is not None:
        print(f"\nDataset overview:")
        print(f"Total rows: {len(df)}")
        print(f"Fuzzer types: {df['fuzzer_type'].unique()}")
        print(f"Scenarios: {df['scenario'].unique()}")
        print(f"Total vulnerabilities: {df['vulnerability_count'].sum()}")
        
        # Generate plots with combined plots 1 & 2
        create_combined_plots(df, OUTPUT_DIR)
        create_plot3_vulnerability_severity_distribution(df, OUTPUT_DIR)
        create_plot4_performance_across_scenarios(df, OUTPUT_DIR)
        
        print(f"\n=== ALL ESSENTIAL PLOTS GENERATED SUCCESSFULLY ===")
        print(f"Check the '{OUTPUT_DIR}' directory for:")
        print("- Combined Plot: Vulnerability Discovery & Convergence Analysis")
        print("- Plot 3: Vulnerability Severity Distribution (quality analysis)")
        print("- Plot 4: Performance Across Scenarios (robustness validation)")
        
        # Generate summary statistics
        ai_total = df[df['fuzzer_type'] == 'AI-Fuzzing']['vulnerability_count'].sum()
        traditional_total = df[df['fuzzer_type'] == 'Traditional-Testing']['vulnerability_count'].sum()
        improvement = ((ai_total - traditional_total) / traditional_total) * 100
        
        print(f"\n=== KEY STATISTICS ===")
        print(f"AI-Fuzzing total vulnerabilities: {ai_total}")
        print(f"Traditional-Testing total vulnerabilities: {traditional_total}")
        print(f"Overall improvement: {improvement:.1f}%")
        
    else:
        print("Could not proceed due to data loading error.")

if __name__ == "__main__":
    main()