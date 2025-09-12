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
from scipy import stats
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# --- Constants ---
CSV_FILENAME = "fuzzing_results_v28_strategic_fuzzing.csv"
OUTPUT_DIR = "plots_essential_paper"

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
    print(f"Loading data from {csv_file}...")
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found!")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"Successfully loaded {len(df)} rows of data.")
    
    # Clean and prepare data
    df['fuzzer_type'] = df['fuzzer_type']
    
    # Ensure boolean columns are correctly typed
    for col in ['is_critical_failure', 'has_ping_pong', 'has_qoe_violation', 'has_unfairness']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x).strip().lower() == 'true' if pd.notna(x) else False)
    
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

def create_plot1_vulnerability_discovery_comparison(df, output_dir):
    """
    Plot 1: Clean and Professional Vulnerability Discovery Comparison
    Simple, readable bar chart without overlapping elements
    """
    print("Generating Plot 1: Clean Vulnerability Discovery Comparison...")
    
    # Filter data for the two main approaches
    comparison_data = df[df['fuzzer_type'].isin(['AI-Fuzzing', 'Traditional-Testing'])]
    
    # Calculate detailed statistics
    stats_data = []
    raw_data = {}
    
    for fuzzer_type in ['Traditional-Testing', 'AI-Fuzzing']:
        fuzzer_data = comparison_data[comparison_data['fuzzer_type'] == fuzzer_type]
        
        # Group by run (scenario + iteration) to get per-run vulnerability counts
        run_vulns = fuzzer_data.groupby(['scenario', 'iteration'])['vulnerability_count'].sum()
        raw_data[fuzzer_type] = run_vulns.values
        
        total_vulns = fuzzer_data['vulnerability_count'].sum()
        mean_vulns = run_vulns.mean()
        std_vulns = run_vulns.std()
        n_runs = len(run_vulns)
        se_vulns = std_vulns / np.sqrt(n_runs)  # Standard error
        ci_95 = 1.96 * se_vulns  # 95% confidence interval
        
        stats_data.append({
            'fuzzer_type': fuzzer_type,
            'total_vulnerabilities': total_vulns,
            'mean_vulnerabilities': mean_vulns,
            'std_vulnerabilities': std_vulns,
            'ci_95': ci_95,
            'n_runs': n_runs,
            'se': se_vulns
        })
    
    # Statistical tests
    traditional_runs = raw_data['Traditional-Testing']
    ai_runs = raw_data['AI-Fuzzing']
    
    # Mann-Whitney U test
    u_stat, p_value = mannwhitneyu(ai_runs, traditional_runs, alternative='greater')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(traditional_runs) - 1) * np.std(traditional_runs, ddof=1)**2 + 
                         (len(ai_runs) - 1) * np.std(ai_runs, ddof=1)**2) / 
                        (len(traditional_runs) + len(ai_runs) - 2))
    cohens_d = (np.mean(ai_runs) - np.mean(traditional_runs)) / pooled_std
    
    # Create clean, simple plot with EXTREME MAXIMUM spacing
    fig, ax = plt.subplots(1, 1, figsize=(16, 14), facecolor='white')
    
    fuzzer_types = [data['fuzzer_type'] for data in stats_data]
    # Change display names for cleaner appearance
    display_names = []
    for fuzzer_type in fuzzer_types:
        if fuzzer_type == 'Traditional-Testing':
            display_names.append('Traditional Testing')
        else:
            display_names.append(fuzzer_type)
    
    means = [data['mean_vulnerabilities'] for data in stats_data]
    cis = [data['ci_95'] for data in stats_data]
    n_runs = [data['n_runs'] for data in stats_data]
    
    # Simple, professional colors
    colors = ['#4472C4', '#E15759']  # Blue for Traditional, Red for AI
    
    # Create simple bars with clean error bars using display names
    bars = ax.bar(display_names, means, color=colors, alpha=0.8, 
                  yerr=cis, capsize=8, width=0.5,
                  edgecolor='black', linewidth=1.5,
                  error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Clean axis labels with LARGER font sizes
    ax.set_ylabel('Vulnerabilities per Run', fontsize=20, fontweight='bold')
    ax.tick_params(axis='x', labelsize=24, pad=15)  # Much larger x-axis labels
    ax.tick_params(axis='y', labelsize=18)
    
    # Calculate dimensions for EXTREME spacing
    max_bar_height = max(means)
    max_error_height = max([m + c for m, c in zip(means, cis)])
    
    # Add ONLY value labels on bars - positioned EXTREMELY high above error bars
    for i, (bar, mean_val) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ci_val = cis[i]
        # Position label EXTREMELY high above error bar
        ax.text(bar.get_x() + bar.get_width()/2., height + ci_val + 3.0,
                f'{mean_val:.1f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=22)
    
    # Calculate improvement percentage
    improvement = ((means[1] - means[0]) / means[0]) * 100
    
    # Position significance elements with ENORMOUS gaps
    y_start_annotations = max_error_height + 8.0  # ENORMOUS gap from error bars
    
    # Significance bracket positioned extremely high
    y_bracket = y_start_annotations
    ax.plot([0, 1], [y_bracket, y_bracket], 'k-', linewidth=2)
    ax.plot([0, 0], [y_bracket-0.8, y_bracket], 'k-', linewidth=2)
    ax.plot([1, 1], [y_bracket-0.8, y_bracket], 'k-', linewidth=2)
    
    # P-value text positioned with gap above bracket
    p_text = f'p < 0.001' if p_value < 0.001 else f'p = {p_value:.3f}'
    ax.text(0.5, y_bracket + 3.5, f'{p_text}', 
            ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # Improvement text positioned with MUCH LARGER gap above p-value
    ax.text(0.5, y_bracket + 8.5, f'{improvement:.1f}% improvement', 
            ha='center', va='bottom', fontsize=24, color='darkgreen', fontweight='bold')
    
    # Set y-limits with extra space for larger gap between texts
    ax.set_ylim(0, max_error_height + 22.0)  # Even more space at top
    
    # Simple grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    # Keep improvement text at top, minimal bottom margin
    plt.tight_layout(pad=2.0)  
    plt.subplots_adjust(bottom=0.15, top=0.6, left=0.2, right=0.9)  # Minimal bottom space
    
    # Save with clean settings
    output_path = os.path.join(output_dir, 'plot_1_vulnerability_discovery.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300, 
                facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(output_dir, 'plot_1_vulnerability_discovery.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"-> Plot 1 saved to {output_path}")
    print(f"   Traditional Testing: {means[0]:.1f} ± {cis[0]:.1f} vulnerabilities/run")
    print(f"   AI Fuzzing: {means[1]:.1f} ± {cis[1]:.1f} vulnerabilities/run")
    print(f"   Improvement: {improvement:.1f}%")
    print(f"   Effect size (Cohen's d): {cohens_d:.2f}")
    print(f"   P-value: {p_value:.6f}")

def create_plot2_convergence_analysis(df, output_dir):
    """
    Plot 2: Convergence Analysis
    Enhanced line plot with confidence bands and scatter points
    """
    print("Generating Plot 2: Convergence Analysis...")
    
    # Group by iteration to show progression
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
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8), facecolor='white')  # Optimized size for convergence plot
    
    # Plot convergence lines with beautiful colors and confidence bands
    ai_color = '#4ECDC4'  # Teal
    traditional_color = '#FF6B6B'  # Coral
    
    # Main convergence lines with larger markers and thicker lines
    ax.plot(ai_data['iteration'], ai_data['best_fitness'], 
            'o-', color=ai_color, linewidth=4, markersize=10, 
            label='AI Fuzzing', markerfacecolor='white', markeredgewidth=2.5, alpha=0.9)
    
    ax.fill_between(ai_data['iteration'], 
                    ai_data['ci_lower'].cummax(), ai_data['ci_upper'].cummax(),
                    alpha=0.25, color=ai_color, label='AI 95% CI')
    
    ax.plot(traditional_data['iteration'], traditional_data['best_fitness'], 
            's--', color=traditional_color, linewidth=4, markersize=10, 
            label='Traditional Testing', markerfacecolor='white', markeredgewidth=2.5, alpha=0.9)
    
    ax.fill_between(traditional_data['iteration'], 
                    traditional_data['ci_lower'].cummax(), traditional_data['ci_upper'].cummax(),
                    alpha=0.25, color=traditional_color, label='Traditional 95% CI')
    
    # Add scatter points for data distribution with better visibility
    for iteration in ai_data['iteration']:
        if iteration in ai_raw.index:
            y_vals = ai_raw.loc[iteration]
            x_vals = [iteration + np.random.normal(0, 0.08) for _ in y_vals]  # Reduced jitter
            ax.scatter(x_vals, y_vals, alpha=0.6, color=ai_color, s=35, 
                      edgecolor='white', linewidth=1, zorder=5)
    
    for iteration in traditional_data['iteration']:
        if iteration in traditional_raw.index:
            y_vals = traditional_raw.loc[iteration]
            x_vals = [iteration + np.random.normal(0, 0.08) for _ in y_vals]  # Reduced jitter
            ax.scatter(x_vals, y_vals, alpha=0.6, color=traditional_color, s=35, 
                      marker='s', edgecolor='white', linewidth=1, zorder=5)
    
    # Enhanced labels with larger fonts - title removed for caption
    ax.set_xlabel('Generation Number', fontsize=24, fontweight='bold')
    ax.set_ylabel('Best Fitness Score\n(Cumulative Max Vulnerabilities)', fontsize=24, fontweight='bold')
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(fontsize=18, loc='upper left', framealpha=0.95, shadow=True)  # Moved to top corner
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Calculate convergence metrics
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
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)  # Optimized margins
    
    output_path = os.path.join(output_dir, 'plot_2_convergence_analysis.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300, facecolor='white', 
                pad_inches=0.3)  # Clean padding
    plt.savefig(os.path.join(output_dir, 'plot_2_convergence_analysis.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close()
    
    print(f"-> Plot 2 saved to {output_path}")
    print(f"   AI convergence at generation: {ai_convergence}")
    print(f"   Traditional convergence at generation: {traditional_convergence}")
    print(f"   Speedup: {speedup:.1f}%")
    print(f"   Final performance ratio: {final_ai/final_traditional:.2f}×")

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
    Enhanced grouped bar chart with error bars and statistical significance
    """
    print("Generating Plot 4: Performance Across Scenarios...")
    
    comparison_data = df[df['fuzzer_type'].isin(['AI-Fuzzing', 'Traditional-Testing'])]
    
    # Focus on 3 main scenarios for clarity
    main_scenarios = ['Stable Mobility', 'Load Imbalance', 'Congestion Crisis']
    scenario_data = comparison_data[comparison_data['scenario'].isin(main_scenarios)]
    
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
    
    # Create enhanced plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 12), facecolor='white')  # Further increased height for maximum space
    
    x = np.arange(len(main_scenarios))
    width = 0.35
    
    # Grayscale-friendly colors with patterns
    colors = ['#666666', '#333333']  # Gray tones
    hatches = ['///', '']  # Patterns for distinction
    
    traditional_means = [stats[list(stats.keys())[0]]['mean'] for stats in scenario_stats]
    traditional_cis = [stats[list(stats.keys())[0]]['ci'] for stats in scenario_stats]
    traditional_ns = [stats[list(stats.keys())[0]]['n'] for stats in scenario_stats]
    
    ai_means = [stats[list(stats.keys())[1]]['mean'] for stats in scenario_stats]
    ai_cis = [stats[list(stats.keys())[1]]['ci'] for stats in scenario_stats]
    ai_ns = [stats[list(stats.keys())[1]]['n'] for stats in scenario_stats]
    
    # Beautiful colors for scenario comparison
    colors = ['#FF6B6B', '#4ECDC4']  # Coral Red for Traditional, Teal for AI
    edge_colors = ['#E74C3C', '#16A085']  # Darker edges
    
    bars1 = ax.bar(x - width/2, traditional_means, width, 
                   label='Traditional Testing', color=colors[0], alpha=0.8, 
                   yerr=traditional_cis, capsize=5,
                   edgecolor=edge_colors[0], linewidth=1.5)
    
    bars2 = ax.bar(x + width/2, ai_means, width, 
                   label='AI Fuzzing', color=colors[1], alpha=0.8, 
                   yerr=ai_cis, capsize=5,
                   edgecolor=edge_colors[1], linewidth=1.5)
    
    # Add mean ± CI labels on bars with larger fonts
    for i, (bar, mean_val, ci_val, n_val) in enumerate(zip(bars1, traditional_means, traditional_cis, traditional_ns)):
        if mean_val > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + ci_val + 0.1,
                    f'{mean_val:.1f}±{ci_val:.1f}\n(n={n_val})', 
                    ha='center', va='bottom', fontweight='bold', fontsize=16)

    for i, (bar, mean_val, ci_val, n_val) in enumerate(zip(bars2, ai_means, ai_cis, ai_ns)):
        if mean_val > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + ci_val + 0.1,
                    f'{mean_val:.1f}±{ci_val:.1f}\n(n={n_val})', 
                    ha='center', va='bottom', fontweight='bold', fontsize=16)

    # Enhanced styling with larger fonts for better readability - title removed for caption
    ax.set_ylabel('Mean Vulnerabilities per Run', fontsize=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(' ', '\n') for s in main_scenarios], fontsize=18, fontweight='medium')
    ax.tick_params(axis='y', labelsize=18)
    ax.legend(fontsize=16, loc='upper left', framealpha=0.9)
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
    
    ax.text(0.98, 0.98, analysis_text, transform=ax.transAxes, 
            fontsize=14, fontweight='normal',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            verticalalignment='top', horizontalalignment='right', family='monospace')
    
    # Add individual scenario improvements as arrows
    for i, improvement in enumerate(improvements):
        if improvement > 0:
            ax.annotate(f'+{improvement:.0f}%', xy=(i, max(ai_means[i] + ai_cis[i], traditional_means[i] + traditional_cis[i]) * 0.8),
                       xytext=(i, max(ai_means[i] + ai_cis[i], traditional_means[i] + traditional_cis[i]) * 0.9),
                       ha='center', fontsize=14, fontweight='bold', color='green',
                       arrowprops=dict(arrowstyle='->', color='green', lw=1))
    
    # Adjust layout to prevent labels from crossing borders
    plt.tight_layout()
    plt.subplots_adjust(top=0.7)  # Maximum space at top for bar labels (30% reserved)
    
    output_path = os.path.join(output_dir, 'plot_4_scenario_performance.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300, facecolor='white', 
                pad_inches=1.0)  # Maximum padding around the figure
    plt.savefig(os.path.join(output_dir, 'plot_4_scenario_performance.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', pad_inches=1.0)
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
        
        # Generate the 4 essential plots
        create_plot1_vulnerability_discovery_comparison(df, OUTPUT_DIR)
        create_plot2_convergence_analysis(df, OUTPUT_DIR)
        create_plot3_vulnerability_severity_distribution(df, OUTPUT_DIR)
        create_plot4_performance_across_scenarios(df, OUTPUT_DIR)
        
        print(f"\n=== ALL ESSENTIAL PLOTS GENERATED SUCCESSFULLY ===")
        print(f"Check the '{OUTPUT_DIR}' directory for:")
        print("- Plot 1: Vulnerability Discovery Comparison (main effectiveness)")
        print("- Plot 2: Convergence Analysis (scientific validation)")
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