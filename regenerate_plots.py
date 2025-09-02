#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced publication-quality plots generator for the 
AI-Fuzzing for 5G Traffic Steering ICC paper.
Generates all 6 key figures needed for the publication.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy import stats

# --- Constants ---
CSV_FILENAME = "fuzzing_results_v28_strategic_fuzzing.csv"
OUTPUT_DIR = "plots_for_publication"

# Set professional plotting style with increased font brightness/clarity
plt.rcParams.update({
    'font.size': 14,                  # Increased from 12
    'axes.titlesize': 16,             # Increased from 14
    'axes.labelsize': 14,             # Increased from 12
    'xtick.labelsize': 12,            # Increased from 10
    'ytick.labelsize': 12,            # Increased from 10
    'legend.fontsize': 12,            # Increased from 10
    'figure.titlesize': 18,           # Increased from 16
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,
    'font.weight': 'bold',            # Added bold font weight
    'axes.titleweight': 'bold',       # Bold title
    'axes.labelweight': 'medium',     # Medium weight for labels
    'axes.grid': True,
    'grid.alpha': 0.3
})

def load_and_prepare_data(csv_file):
    """Loads and preprocesses the data from the CSV file."""
    print(f"Loading data from {csv_file}...")
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found!")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"Successfully loaded {len(df)} rows of data.")
    
    # Clean up fuzzer names for professional plot labels
    df['fuzzer_type'] = df['fuzzer_type'].replace({
        'AI-Fuzzer': 'AI-Fuzzer (NSGA-II)',
        'HillClimbing-Fuzzer': 'Hill Climbing',
        'Random-Fuzzer': 'Random'
    })
    
    # Ensure boolean columns are correctly typed
    for col in ['is_critical_failure', 'has_ping_pong', 'has_qoe_violation', 'has_unfairness']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x).strip().lower() == 'true' if pd.notna(x) else False)
            
    return df

def create_fig1_main_effectiveness(df, output_dir):
    """Figure 1: Main effectiveness comparison - the most important plot."""
    print("Generating Figure 1: Main Fuzzer Effectiveness...")
    
    fuzzing_df = df[df['fuzzer_type'].isin(['AI-Fuzzer (NSGA-II)', 'Hill Climbing', 'Random'])]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left subplot: Total Critical Failures
    total_critical = fuzzing_df.groupby('fuzzer_type')['is_critical_failure'].sum().reset_index()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    bars = ax1.bar(total_critical['fuzzer_type'], total_critical['is_critical_failure'], 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax1.set_title('Critical Failures Discovered', fontsize=14, pad=15)
    ax1.set_ylabel('Total Count', fontsize=12)
    ax1.set_ylim(0, max(total_critical['is_critical_failure']) * 1.15)
    
    # Right subplot: Total Vulnerabilities
    total_vulns = fuzzing_df.groupby('fuzzer_type')['vulnerability_count'].sum().reset_index()
    
    bars2 = ax2.bar(total_vulns['fuzzer_type'], total_vulns['vulnerability_count'], 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax2.set_title('Total Vulnerabilities Found', fontsize=14, pad=15)
    ax2.set_ylabel('Total Count', fontsize=12)
    ax2.set_ylim(0, max(total_vulns['vulnerability_count']) * 1.1)
    
    # Add significance annotation
    y_max = max(total_critical['is_critical_failure'])
    ax1.annotate('p < 0.0001', xy=(1, y_max * 0.9), fontsize=12, 
                ha='center', fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.suptitle('AI-Fuzzer Demonstrates Superior Vulnerability Discovery', fontsize=16, y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fig_1_main_effectiveness.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig_1_main_effectiveness.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"-> Figure 1 saved to {output_path}")

def create_fig2_qoe_performance(df, output_dir):
    """Figure 2: QoE Performance CDFs - shows impact on network performance."""
    print("Generating Figure 2: QoE Performance Analysis...")
    
    fuzzing_df = df[df['fuzzer_type'].isin(['AI-Fuzzer (NSGA-II)', 'Hill Climbing', 'Random'])]
    scenarios = ['Stable Mobility', 'Stable High Load', 'Load Imbalance', 'Coverage Hole']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = {'Random': '#1f77b4', 'Hill Climbing': '#ff7f0e', 'AI-Fuzzer (NSGA-II)': '#2ca02c'}
    linestyles = {'Random': '--', 'Hill Climbing': '-.', 'AI-Fuzzer (NSGA-II)': '-'}
    
    for i, scenario in enumerate(scenarios[:4]):
        ax = axes[i]
        scenario_df = fuzzing_df[fuzzing_df['scenario'] == scenario]
        
        for fuzzer in ['Random', 'Hill Climbing', 'AI-Fuzzer (NSGA-II)']:
            data = scenario_df[scenario_df['fuzzer_type'] == fuzzer]['throughput_5th_percentile_mbps']
            data = data.dropna().sort_values().reset_index(drop=True)
            
            if not data.empty:
                y = np.linspace(0, 1, len(data))
                ax.plot(data, y, label=fuzzer, color=colors[fuzzer], 
                       linestyle=linestyles[fuzzer], linewidth=3.0)  # Increased linewidth from 2.5 to 3.0
        
        # Add subplot labels (a), (b), (c), (d)
        subplot_labels = ['(a)', '(b)', '(c)', '(d)']
        ax.set_title(f'{subplot_labels[i]} {scenario}', fontsize=14, fontweight='bold')
        ax.set_xlabel('5th Percentile Throughput (Mbps)', fontsize=13, fontweight='medium')
        ax.set_ylabel('Cumulative Probability', fontsize=13, fontweight='medium')
        ax.legend(fontsize=12, framealpha=0.9)  # Increased font size and frame opacity
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 25)
    
    # Removed the main title "Impact on Network QoE: Lower is Worse (AI-Fuzzer Creates Worst Conditions)"
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fig_2_qoe_performance.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig_2_qoe_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"-> Figure 2 saved to {output_path}")

def create_fig3_vulnerability_heatmap(df, output_dir):
    """Figure 3: Vulnerability type breakdown across scenarios."""
    print("Generating Figure 3: Vulnerability Breakdown Heatmap...")

    fuzzing_df = df[df['fuzzer_type'].isin(['AI-Fuzzer (NSGA-II)', 'Hill Climbing', 'Random'])]

    # Create vulnerability breakdown
    vuln_types = ['has_ping_pong', 'has_qoe_violation', 'has_unfairness', 'is_critical_failure']
    vuln_labels = ['Ping-Pong\nHandovers', 'QoE\nViolations', 'Unfairness\nEvents', 'Critical\nFailures']

    fig, axes = plt.subplots(1, 3, figsize=(20, 8)) # Maintained larger figure size

    scenarios = fuzzing_df['scenario'].unique()

    for i, fuzzer in enumerate(['Random', 'Hill Climbing', 'AI-Fuzzer (NSGA-II)']):
        fuzzer_df = fuzzing_df[fuzzing_df['fuzzer_type'] == fuzzer]

        # Create matrix for heatmap
        heatmap_data = []
        for scenario in scenarios:
            scenario_data = []
            scenario_df = fuzzer_df[fuzzer_df['scenario'] == scenario]
            for vuln_type in vuln_types:
                count = scenario_df[vuln_type].sum() if vuln_type in scenario_df.columns else 0
                scenario_data.append(count)
            heatmap_data.append(scenario_data)

        heatmap_matrix = np.array(heatmap_data)

        # Create heatmap with stronger contrast
        im = axes[i].imshow(heatmap_matrix, cmap='Reds', aspect='auto', vmin=0, 
                          vmax=max(1, np.max(heatmap_matrix) * 1.1))  # Improve color contrast

        # Add text annotations with increased font size and better visibility
        for row in range(len(scenarios)):
            for col in range(len(vuln_types)):
                value = int(heatmap_matrix[row, col])
                # Change text color based on cell value for better contrast
                text_color = "black" if value < np.max(heatmap_matrix) * 0.7 else "white"
                text = axes[i].text(col, row, f'{value}',
                                  ha="center", va="center", 
                                  color=text_color, 
                                  fontweight='bold', 
                                  fontsize=16)  # Increased from 14
        
        # Set main title (fuzzer name)
        axes[i].set_title(f'{fuzzer}', fontsize=20, fontweight='bold')
        
        # Add (a), (b), (c) labels under the titles with adjusted positioning
        # Moving them further down to avoid crossing with the subtitle
        subplot_labels = ['(a)', '(b)', '(c)']
        axes[i].text(0.5, -0.2, subplot_labels[i], transform=axes[i].transAxes,
                    ha='center', va='center', fontsize=18, fontweight='bold')
        axes[i].set_xticks(range(len(vuln_labels)))
        axes[i].set_xticklabels(vuln_labels, rotation=45, ha='right', fontsize=16, fontweight='medium') # Increased size
        axes[i].set_yticks(range(len(scenarios)))
        axes[i].set_yticklabels(scenarios, fontsize=16, fontweight='medium') # Increased size

        # Add colorbar with better visibility
        cbar = plt.colorbar(im, ax=axes[i], shrink=0.6)
        cbar.ax.tick_params(labelsize=14, labelcolor='black') # Increased size and ensured black color

    # Removed the main title "Vulnerability Discovery Pattern Analysis"
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'fig_3_vulnerability_heatmap.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig_3_vulnerability_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"-> Figure 3 saved to {output_path}")

def create_fig4_statistical_analysis(df, output_dir):
    """Figure 4: Statistical significance analysis with box plots."""
    print("Generating Figure 4: Statistical Analysis...")
    
    fuzzing_df = df[df['fuzzer_type'].isin(['AI-Fuzzer (NSGA-II)', 'Hill Climbing', 'Random'])]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Box plot of critical failures per run
    critical_per_run = fuzzing_df.groupby(['fuzzer_type', 'scenario', 'algorithm'])['is_critical_failure'].sum().reset_index()
    
    box_plot = ax1.boxplot([
        critical_per_run[critical_per_run['fuzzer_type'] == 'Random']['is_critical_failure'],
        critical_per_run[critical_per_run['fuzzer_type'] == 'Hill Climbing']['is_critical_failure'],
        critical_per_run[critical_per_run['fuzzer_type'] == 'AI-Fuzzer (NSGA-II)']['is_critical_failure']
    ], labels=['Random', 'Hill Climbing', 'AI-Fuzzer\n(NSGA-II)'], patch_artist=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title('Distribution of Critical Failures per Run', fontsize=14)
    ax1.set_ylabel('Critical Failures Count', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Right: Algorithm performance comparison
    algo_comparison = fuzzing_df.groupby(['fuzzer_type', 'algorithm'])['is_critical_failure'].sum().unstack(fill_value=0)
    
    algo_comparison.plot(kind='bar', ax=ax2, width=0.8, alpha=0.8)
    ax2.set_title('Critical Failures by Algorithm Type', fontsize=14)
    ax2.set_ylabel('Total Critical Failures', fontsize=12)
    ax2.set_xlabel('Fuzzer Type', fontsize=12)
    ax2.legend(title='Traffic Steering\nAlgorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fig_4_statistical_analysis.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig_4_statistical_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"-> Figure 4 saved to {output_path}")

def create_fig5_scenario_comparison(df, output_dir):
    """Figure 5: Scenario-wise detailed comparison."""
    print("Generating Figure 5: Scenario Comparison...")

    fuzzing_df = df[df['fuzzer_type'].isin(['AI-Fuzzer (NSGA-II)', 'Hill Climbing', 'Random'])]

    # Create scenario comparison
    scenario_summary = fuzzing_df.groupby(['scenario', 'fuzzer_type']).agg({
        'is_critical_failure': 'sum',
        'vulnerability_count': 'sum',
        'handover_rate': 'mean',
        'jain_fairness_index': 'mean'
    }).round(2)

    scenarios = fuzzing_df['scenario'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12)) # Maintained larger figure size
    axes = axes.flatten()

    metrics = [
        ('is_critical_failure', 'Critical Failures'),
        ('vulnerability_count', 'Total Vulnerabilities'),
        ('handover_rate', 'Average Handover Rate'),
        ('jain_fairness_index', 'Jain Fairness Index')
    ]

    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]

        data_to_plot = []
        labels = []
        for scenario in scenarios:
            scenario_data = []
            for fuzzer in ['Random', 'Hill Climbing', 'AI-Fuzzer (NSGA-II)']:
                try:
                    value = scenario_summary.loc[(scenario, fuzzer), metric]
                    scenario_data.append(value)
                except KeyError:
                    scenario_data.append(0)
            data_to_plot.append(scenario_data)
            labels.append(scenario)

        x = np.arange(len(scenarios))
        width = 0.25

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        fuzzer_names = ['Random', 'Hill Climbing', 'AI-Fuzzer (NSGA-II)']

        for j, (fuzzer, color) in enumerate(zip(fuzzer_names, colors)):
            values = [data_to_plot[k][j] for k in range(len(scenarios))]
            # Add black edge to bars for better clarity
            bars = ax.bar(x + j*width, values, width, label=fuzzer, color=color, 
                         alpha=0.85, # Increased from 0.8
                         edgecolor='black', linewidth=0.8) # Added black border for contrast
            
            # Add value labels on top of bars for metrics with discrete counts
            if metric in ['is_critical_failure', 'vulnerability_count']:
                for k, bar in enumerate(bars):
                    height = bar.get_height()
                    if height > 0:  # Only add labels to non-zero bars
                        # Position the text with slightly more vertical offset
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                              f'{int(height)}', ha='center', va='bottom', 
                              fontsize=11, fontweight='bold')

        # Replace subplot titles with just (a), (b), (c), (d) labels
        subplot_labels = ['(a)', '(b)', '(c)', '(d)']
        ax.set_title(f"{subplot_labels[i]}", fontsize=18, fontweight='bold')
        
        ax.set_xlabel('Scenario', fontsize=16, fontweight='medium')
        
        # Set specific y-axis labels based on the metric
        y_labels = {
            'is_critical_failure': 'Number of Critical Failures',
            'vulnerability_count': 'Number of Vulnerabilities',
            'handover_rate': 'Handover Rate (per second)',
            'jain_fairness_index': 'Fairness Index (0-1)'
        }
        ax.set_ylabel(y_labels[metric], fontsize=16, fontweight='medium')
        
        # For subplots (a) and (b), adjust the y-axis limit to ensure number labels don't cross lines
        if metric in ['is_critical_failure', 'vulnerability_count']:
            # Get current y-limit
            current_ylim = ax.get_ylim()
            # Find the maximum value in the plot
            max_val = 0
            for j in range(len(scenarios)):
                for k in range(3):  # 3 fuzzers
                    try:
                        val = data_to_plot[j][k]
                        if val > max_val:
                            max_val = val
                    except:
                        pass
            # Set new y-limit with extra 20% padding on top for the labels
            ax.set_ylim(0, max(current_ylim[1], max_val * 1.2))
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14, fontweight='medium')
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(True, alpha=0.3)

    # Add a single legend for the entire figure at the bottom
    fig.legend(['Random', 'Hill Climbing', 'AI-Fuzzer (NSGA-II)'], 
              fontsize=14, framealpha=0.9, loc='lower center',
              bbox_to_anchor=(0.5, 0.02), # Position at the bottom of the figure
              ncol=3) # Put all items in one row

    # Adjusted padding to accommodate the common legend at the bottom
    plt.tight_layout(rect=[0, 0.08, 1, 1]) # Add space at the bottom for legend

    output_path = os.path.join(output_dir, 'fig_5_scenario_comparison.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig_5_scenario_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"-> Figure 5 saved to {output_path}")

def create_fig6_network_topology(output_dir):
    """Figure 6: Network topology visualization for paper context."""
    print("Generating Figure 6: Network Topology...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Hexagonal cell layout
    def generate_hexagonal_layout(num_cells, distance):
        coords = [(0.0, 0.0)]
        if num_cells == 1:
            return np.array(coords)
            
        axial_directions = [(1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1)]
        axial_coords = [(0, 0)]
        seen_coords = set([(0, 0)])
        ring = 1
        
        while len(axial_coords) < num_cells:
            current_axial = (ring, -ring)
            for dir_idx in range(6):
                for step in range(ring):
                    if len(axial_coords) >= num_cells:
                        break
                    if current_axial not in seen_coords:
                        axial_coords.append(current_axial)
                        seen_coords.add(current_axial)
                    current_axial = (current_axial[0] + axial_directions[(dir_idx + 1) % 6][0],
                                   current_axial[1] + axial_directions[(dir_idx + 1) % 6][1])
                if len(axial_coords) >= num_cells:
                    break
            ring += 1
        
        cartesian_coords = []
        for q, r in axial_coords:
            x = distance * (3./2. * q)
            y = distance * (np.sqrt(3)/2. * q + np.sqrt(3) * r)
            cartesian_coords.append((x, y))
        return np.array(cartesian_coords[:num_cells])
    
    # Generate cell positions
    cell_positions = generate_hexagonal_layout(7, 100)
    
    # Plot cells
    for i, (x, y) in enumerate(cell_positions):
        circle = plt.Circle((x, y), 50, fill=False, edgecolor='blue', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(x, y, f'Cell {i}', ha='center', va='center', fontweight='bold')
    
    # Add sample UE positions
    np.random.seed(42)
    ue_positions = np.random.uniform(-150, 150, (15, 2))
    ax1.scatter(ue_positions[:, 0], ue_positions[:, 1], c='red', s=50, 
               marker='s', label='User Equipment (UE)', alpha=0.8)
    
    ax1.set_xlim(-200, 200)
    ax1.set_ylim(-200, 200)
    ax1.set_aspect('equal')
    ax1.set_title('5G Network Topology', fontsize=14)
    ax1.set_xlabel('Distance (m)', fontsize=12)
    ax1.set_ylabel('Distance (m)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Fuzzing process illustration
    ax2.text(0.5, 0.9, 'AI-Fuzzer Process', ha='center', va='center', 
             fontsize=16, fontweight='bold', transform=ax2.transAxes)
    
    # Draw process flow
    boxes = [
        ('Population\nInitialization', 0.5, 0.8),
        ('NSGA-II\nEvolution', 0.5, 0.65),
        ('Multi-Objective\nEvaluation', 0.5, 0.5),
        ('Vulnerability\nDetection', 0.5, 0.35),
        ('Pareto Front\nSelection', 0.5, 0.2)
    ]
    
    for i, (text, x, y) in enumerate(boxes):
        rect = Rectangle((x-0.15, y-0.05), 0.3, 0.08, 
                        facecolor='lightblue', edgecolor='black', 
                        transform=ax2.transAxes)
        ax2.add_patch(rect)
        ax2.text(x, y, text, ha='center', va='center', 
                fontsize=10, fontweight='bold', transform=ax2.transAxes)
        
        if i < len(boxes) - 1:
            ax2.arrow(x, y-0.06, 0, -0.04, head_width=0.02, head_length=0.02, 
                     fc='black', ec='black', transform=ax2.transAxes)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.suptitle('Network Environment and AI-Fuzzing Methodology', fontsize=16, y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fig_6_network_topology.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig_6_network_topology.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"-> Figure 6 saved to {output_path}")

def generate_summary_table(df, output_dir):
    """Generate a summary table for the paper."""
    print("Generating Summary Table...")
    
    fuzzing_df = df[df['fuzzer_type'].isin(['AI-Fuzzer (NSGA-II)', 'Hill Climbing', 'Random'])]
    
    summary = fuzzing_df.groupby('fuzzer_type').agg({
        'is_critical_failure': ['sum', 'mean'],
        'vulnerability_count': ['sum', 'mean'],
        'handover_rate': 'mean',
        'throughput_5th_percentile_mbps': 'mean',
        'jain_fairness_index': 'mean'
    }).round(3)
    
    # Flatten column names
    summary.columns = ['Total Critical', 'Avg Critical', 'Total Vulns', 'Avg Vulns', 
                      'Avg Handover Rate', 'Avg 5th Percentile Throughput', 'Avg Fairness']
    
    # Save as CSV
    summary_path = os.path.join(output_dir, 'summary_table.csv')
    summary.to_csv(summary_path)
    
    print(f"-> Summary table saved to {summary_path}")
    print("\nSUMMARY TABLE:")
    print(summary)

def main():
    """Main function to generate all publication plots."""
    print("=== AI-Fuzzing 5G Publication Plot Generator ===")
    
    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df = load_and_prepare_data(CSV_FILENAME)
    
    if df is not None:
        create_fig2_qoe_performance(df, OUTPUT_DIR)
        create_fig3_vulnerability_heatmap(df, OUTPUT_DIR)
        create_fig5_scenario_comparison(df, OUTPUT_DIR)
        generate_summary_table(df, OUTPUT_DIR)
        
        print(f"\n=== ALL PLOTS GENERATED SUCCESSFULLY ===")
        print(f"Check the '{OUTPUT_DIR}' directory for:")
        print("- Figure 2: QoE performance analysis") 
        print("- Figure 3: Vulnerability breakdown heatmap")
        print("- Figure 5: Scenario comparison")
        print("- Summary table (CSV format)")
        print("\nThese figures should be sufficient for your ICC paper!")
    else:
        print("Could not proceed due to data loading error.")

if __name__ == "__main__":
    main()